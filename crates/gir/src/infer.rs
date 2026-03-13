//! Shape inference for IR operations.
//!
//! Given a node's [`OpKind`], its input shapes/dtypes, and its attributes, the
//! [`infer_shapes`] function computes the output [`ValueInfo`]s.  The results
//! are expressed in terms of [`DimExpr`] so that symbolic dimensions propagate
//! through the graph correctly.
//!
//! # Design note
//!
//! Each operation has its own inference function.  When extending the IR with
//! new operations, add a corresponding arm to `infer_shapes` and implement the
//! helper function below.

use crate::attr::{Attr, PaddingMode};
use crate::dim::DimExpr;
use crate::dtype::DType;
use crate::node::Node;
use crate::shape::Shape;
use crate::value::ValueInfo;

/// Errors that can occur during shape inference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferError {
    /// An input that the operation requires is missing.
    MissingInput {
        op: String,
        expected: usize,
        got: usize,
    },
    /// A required attribute is absent.
    MissingAttr { op: String, attr: String },
    /// An attribute has the wrong type or value.
    BadAttr {
        op: String,
        attr: String,
        msg: String,
    },
    /// Shape mismatch between inputs.
    ShapeMismatch { op: String, msg: String },
    /// Rank mismatch.
    RankMismatch {
        op: String,
        expected: usize,
        got: usize,
    },
    /// Operation not yet supported for inference.
    Unsupported { op: String },
}

impl std::fmt::Display for InferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingInput { op, expected, got } => {
                write!(f, "{op}: expected {expected} inputs, got {got}")
            }
            Self::MissingAttr { op, attr } => {
                write!(f, "{op}: missing required attribute '{attr}'")
            }
            Self::BadAttr { op, attr, msg } => {
                write!(f, "{op}: bad attribute '{attr}': {msg}")
            }
            Self::ShapeMismatch { op, msg } => {
                write!(f, "{op}: shape mismatch: {msg}")
            }
            Self::RankMismatch { op, expected, got } => {
                write!(f, "{op}: expected rank {expected}, got {got}")
            }
            Self::Unsupported { op } => {
                write!(f, "{op}: shape inference not implemented")
            }
        }
    }
}

impl std::error::Error for InferError {}

/// Infer the output [`ValueInfo`]s for a node given its input value infos.
///
/// # Errors
///
/// Returns [`InferError`] if the inputs are incompatible with the operation or
/// required attributes are missing.
pub fn infer_shapes(node: &Node, inputs: &[&ValueInfo]) -> Result<Vec<ValueInfo>, InferError> {
    use crate::op::OpKind;

    match &node.op {
        // ── Element-wise unary ───────────────────────────────────────
        OpKind::Relu | OpKind::Sigmoid | OpKind::Tanh | OpKind::Clip => {
            check_input_count(&node.op, inputs, 1)?;
            Ok(vec![inputs[0].clone()])
        }

        // ── Element-wise binary (with broadcast) ────────────────────
        OpKind::Add | OpKind::Sub | OpKind::Mul => {
            check_input_count(&node.op, inputs, 2)?;
            let out_shape = broadcast_shapes(&node.op, &inputs[0].shape, &inputs[1].shape)?;
            Ok(vec![ValueInfo::new(inputs[0].dtype, out_shape)])
        }

        // ── Softmax ─────────────────────────────────────────────────
        OpKind::Softmax => {
            check_input_count(&node.op, inputs, 1)?;
            Ok(vec![inputs[0].clone()])
        }

        // ── Conv2d ──────────────────────────────────────────────────
        OpKind::Conv2d => infer_conv2d(node, inputs),

        // ── DepthwiseConv2d ─────────────────────────────────────────
        OpKind::DepthwiseConv2d => infer_depthwise_conv2d(node, inputs),

        // ── FullyConnected ──────────────────────────────────────────
        OpKind::FullyConnected => infer_fully_connected(node, inputs),

        // ── MatMul ──────────────────────────────────────────────────
        OpKind::MatMul => infer_matmul(node, inputs),

        // ── Pooling ─────────────────────────────────────────────────
        OpKind::MaxPool2d | OpKind::AvgPool2d => infer_pool2d(node, inputs),
        OpKind::GlobalAvgPool => infer_global_avg_pool(node, inputs),

        // ── Normalisation ───────────────────────────────────────────
        OpKind::BatchNorm | OpKind::LayerNorm => {
            // Output shape = input shape; may need scale/bias inputs but
            // shape inference is straightforward.
            check_min_inputs(&node.op, inputs, 1)?;
            Ok(vec![inputs[0].clone()])
        }

        // ── Reshape ─────────────────────────────────────────────────
        OpKind::Reshape => infer_reshape(node, inputs),

        // ── Flatten ─────────────────────────────────────────────────
        OpKind::Flatten => infer_flatten(node, inputs),

        // ── Transpose ───────────────────────────────────────────────
        OpKind::Transpose => infer_transpose(node, inputs),

        // ── Concat ──────────────────────────────────────────────────
        OpKind::Concat => infer_concat(node, inputs),

        // ── Shape ───────────────────────────────────────────────────
        OpKind::Shape => infer_shape(node, inputs),

        // ── Quantize / Dequantize ───────────────────────────────────
        OpKind::Quantize => {
            check_min_inputs(&node.op, inputs, 1)?;
            // Output dtype comes from a "dtype" attr; default to I8.
            let out_dtype = DType::I8;
            Ok(vec![ValueInfo::new(out_dtype, inputs[0].shape.clone())])
        }
        OpKind::Dequantize => {
            check_min_inputs(&node.op, inputs, 1)?;
            let out_dtype = DType::F32;
            Ok(vec![ValueInfo::new(out_dtype, inputs[0].shape.clone())])
        }

        // ── Constant ────────────────────────────────────────────────
        OpKind::Constant => {
            // The constant's shape and dtype must be provided via the node's
            // output ValueInfo at construction time; inference is a no-op.
            Err(InferError::Unsupported {
                op: format!("{}", node.op),
            })
        }

        // ── Catch-all for ops not yet wired up ──────────────────────
        _ => Err(InferError::Unsupported {
            op: format!("{}", node.op),
        }),
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

fn op_name(op: &crate::op::OpKind) -> String {
    format!("{op}")
}

fn check_input_count(
    op: &crate::op::OpKind,
    inputs: &[&ValueInfo],
    expected: usize,
) -> Result<(), InferError> {
    if inputs.len() < expected {
        return Err(InferError::MissingInput {
            op: op_name(op),
            expected,
            got: inputs.len(),
        });
    }
    Ok(())
}

fn check_min_inputs(
    op: &crate::op::OpKind,
    inputs: &[&ValueInfo],
    min: usize,
) -> Result<(), InferError> {
    if inputs.len() < min {
        return Err(InferError::MissingInput {
            op: op_name(op),
            expected: min,
            got: inputs.len(),
        });
    }
    Ok(())
}

/// Compute the broadcasted shape of two inputs following NumPy-style rules.
fn broadcast_shapes(op: &crate::op::OpKind, a: &Shape, b: &Shape) -> Result<Shape, InferError> {
    let rank = a.rank().max(b.rank());
    let mut dims = Vec::with_capacity(rank);

    let ad = a.dims();
    let bd = b.dims();

    for i in 0..rank {
        let da = if i < rank - a.rank() {
            &DimExpr::fixed(1)
        } else {
            &ad[i - (rank - a.rank())]
        };
        let db = if i < rank - b.rank() {
            &DimExpr::fixed(1)
        } else {
            &bd[i - (rank - b.rank())]
        };

        // Both fixed: check compatibility.
        if let (Some(va), Some(vb)) = (da.try_fixed(), db.try_fixed()) {
            if va == vb {
                dims.push(DimExpr::fixed(va));
            } else if va == 1 {
                dims.push(DimExpr::fixed(vb));
            } else if vb == 1 {
                dims.push(DimExpr::fixed(va));
            } else {
                return Err(InferError::ShapeMismatch {
                    op: op_name(op),
                    msg: format!("incompatible broadcast dims: {va} vs {vb}"),
                });
            }
        } else {
            // If one side is known to be 1, take the other.
            if da.try_fixed() == Some(1) {
                dims.push(db.clone());
            } else if db.try_fixed() == Some(1) {
                dims.push(da.clone());
            } else {
                // Both symbolic — assume they match (could add a constraint system later).
                dims.push(da.clone());
            }
        }
    }
    Ok(Shape::new(dims))
}

/// Retrieve an integer-list attribute or all-ones default of given length.
fn ints_attr_or_default(node: &Node, key: &str, len: usize, default_val: i64) -> Vec<i64> {
    node.get_ints_attr(key)
        .map(<[i64]>::to_vec)
        .unwrap_or_else(|| vec![default_val; len])
}

/// Safely convert an `i64` to `u64`, clamping negative values to 0.
fn to_u64(v: i64) -> u64 {
    u64::try_from(v).unwrap_or(0)
}

/// Safely convert an `i64` to `usize`, clamping negative values to 0.
fn to_usize(v: i64) -> usize {
    usize::try_from(v).unwrap_or(0)
}

/// Compute spatial output dimension for conv / pool.
///
/// `out = floor((input + pad_begin + pad_end - dilation*(kernel-1) - 1) / stride) + 1`
///
/// When `padding == Same`, the formula simplifies to `ceil(input / stride)`.
fn spatial_out_dim(
    input_dim: &DimExpr,
    kernel: u64,
    stride: u64,
    dilation: u64,
    pad_begin: u64,
    pad_end: u64,
    padding: PaddingMode,
) -> DimExpr {
    match padding {
        PaddingMode::Same => input_dim.clone().ceil_div(DimExpr::fixed(stride)),
        PaddingMode::Valid | PaddingMode::Explicit => {
            // effective_kernel = dilation * (kernel - 1) + 1
            let ek = dilation * (kernel - 1) + 1;
            let total_pad = pad_begin + pad_end;
            // (input + total_pad - ek) / stride + 1
            let numerator = if total_pad >= ek {
                input_dim.clone() + DimExpr::fixed(total_pad - ek)
            } else {
                input_dim.clone() - DimExpr::fixed(ek - total_pad)
            };
            numerator / DimExpr::fixed(stride) + DimExpr::fixed(1)
        }
    }
}

/// Get `PaddingMode` from node attributes (default: Valid).
fn get_padding_mode(node: &Node) -> PaddingMode {
    match node.get_attr("padding") {
        Some(Attr::Padding(p)) => *p,
        _ => PaddingMode::Valid,
    }
}

// ── Per-op inference functions ───────────────────────────────────────

/// `Conv2d`: inputs=`[X, W, optional bias]`, layout NCHW, kernel OIHW.
fn infer_conv2d(node: &Node, inputs: &[&ValueInfo]) -> Result<Vec<ValueInfo>, InferError> {
    check_min_inputs(&node.op, inputs, 2)?;
    let x = &inputs[0];
    let w = &inputs[1];

    // Only 2D supported for now
    if x.shape.rank() != 4 {
        return Err(InferError::RankMismatch {
            op: op_name(&node.op),
            expected: 4,
            got: x.shape.rank(),
        });
    }

    let padding = get_padding_mode(node);
    let strides = ints_attr_or_default(node, "strides", 2, 1);
    let dilations = ints_attr_or_default(node, "dilations", 2, 1);
    let pads = ints_attr_or_default(node, "pads", 4, 0); // [top, left, bottom, right]

    // Kernel spatial dims
    let kh = w.shape.dims()[2].try_fixed().unwrap_or(1);
    let kw = w.shape.dims()[3].try_fixed().unwrap_or(1);

    let batch = x.shape.dims()[0].clone();
    let out_channels = w.shape.dims()[0].clone();

    let out_h = spatial_out_dim(
        &x.shape.dims()[2],
        kh,
        to_u64(strides[0]),
        to_u64(dilations[0]),
        to_u64(pads[0]),
        to_u64(pads[2]),
        padding,
    );
    let out_w = spatial_out_dim(
        &x.shape.dims()[3],
        kw,
        to_u64(strides[1]),
        to_u64(dilations[1]),
        to_u64(pads[1]),
        to_u64(pads[3]),
        padding,
    );

    let out_shape = Shape::new(vec![batch, out_channels, out_h, out_w]);
    Ok(vec![ValueInfo::new(x.dtype, out_shape)])
}

/// Depthwise Conv2d: like Conv2d but `out_channels = in_channels * channel_multiplier`.
fn infer_depthwise_conv2d(
    node: &Node,
    inputs: &[&ValueInfo],
) -> Result<Vec<ValueInfo>, InferError> {
    // Reuse Conv2d inference — depthwise is just groups == in_channels.
    infer_conv2d(node, inputs)
}

/// Fully connected: inputs `[X(N, in), W(out, in), bias?]`.
fn infer_fully_connected(node: &Node, inputs: &[&ValueInfo]) -> Result<Vec<ValueInfo>, InferError> {
    check_min_inputs(&node.op, inputs, 2)?;
    let x = &inputs[0];
    let w = &inputs[1];

    if x.shape.rank() < 1 || w.shape.rank() != 2 {
        return Err(InferError::RankMismatch {
            op: op_name(&node.op),
            expected: 2,
            got: w.shape.rank(),
        });
    }

    // Output: (batch_dims..., out_features)
    let mut out_dims: Vec<DimExpr> = x.shape.dims()[..x.shape.rank() - 1].to_vec();
    out_dims.push(w.shape.dims()[0].clone());

    Ok(vec![ValueInfo::new(x.dtype, Shape::new(out_dims))])
}

/// Matrix multiply with broadcasting.
fn infer_matmul(node: &Node, inputs: &[&ValueInfo]) -> Result<Vec<ValueInfo>, InferError> {
    check_input_count(&node.op, inputs, 2)?;
    let a = &inputs[0];
    let b = &inputs[1];

    if a.shape.rank() < 2 || b.shape.rank() < 2 {
        return Err(InferError::RankMismatch {
            op: op_name(&node.op),
            expected: 2,
            got: a.shape.rank().min(b.shape.rank()),
        });
    }

    // Batch dims broadcast, last two are matmul'd: (M, K) x (K, N) -> (M, N)
    let a_dims = a.shape.dims();
    let b_dims = b.shape.dims();

    let m = a_dims[a.shape.rank() - 2].clone();
    let n = b_dims[b.shape.rank() - 1].clone();

    // Broadcast batch dims
    let a_batch = &a_dims[..a.shape.rank() - 2];
    let b_batch = &b_dims[..b.shape.rank() - 2];
    let batch_shape = broadcast_shapes(
        &node.op,
        &Shape::new(a_batch.to_vec()),
        &Shape::new(b_batch.to_vec()),
    )?;

    let mut out_dims = batch_shape.dims().to_vec();
    out_dims.push(m);
    out_dims.push(n);

    Ok(vec![ValueInfo::new(a.dtype, Shape::new(out_dims))])
}

/// Pool2d (max / avg): inputs=[X], layout NCHW.
fn infer_pool2d(node: &Node, inputs: &[&ValueInfo]) -> Result<Vec<ValueInfo>, InferError> {
    check_input_count(&node.op, inputs, 1)?;
    let x = &inputs[0];

    if x.shape.rank() != 4 {
        return Err(InferError::RankMismatch {
            op: op_name(&node.op),
            expected: 4,
            got: x.shape.rank(),
        });
    }

    let kernel = node
        .get_ints_attr("kernel_size")
        .ok_or_else(|| InferError::MissingAttr {
            op: op_name(&node.op),
            attr: "kernel_size".to_owned(),
        })?;
    let strides = ints_attr_or_default(node, "strides", 2, 1);
    let pads = ints_attr_or_default(node, "pads", 4, 0);
    let padding = get_padding_mode(node);

    let out_h = spatial_out_dim(
        &x.shape.dims()[2],
        to_u64(kernel[0]),
        to_u64(strides[0]),
        1,
        to_u64(pads[0]),
        to_u64(pads[2]),
        padding,
    );
    let out_w = spatial_out_dim(
        &x.shape.dims()[3],
        to_u64(kernel[1]),
        to_u64(strides[1]),
        1,
        to_u64(pads[1]),
        to_u64(pads[3]),
        padding,
    );

    let out_shape = Shape::new(vec![
        x.shape.dims()[0].clone(),
        x.shape.dims()[1].clone(),
        out_h,
        out_w,
    ]);
    Ok(vec![ValueInfo::new(x.dtype, out_shape)])
}

/// Global average pool: (N, C, H, W) -> (N, C, 1, 1).
fn infer_global_avg_pool(node: &Node, inputs: &[&ValueInfo]) -> Result<Vec<ValueInfo>, InferError> {
    check_input_count(&node.op, inputs, 1)?;
    let x = &inputs[0];

    if x.shape.rank() != 4 {
        return Err(InferError::RankMismatch {
            op: op_name(&node.op),
            expected: 4,
            got: x.shape.rank(),
        });
    }

    let out = Shape::new(vec![
        x.shape.dims()[0].clone(),
        x.shape.dims()[1].clone(),
        DimExpr::fixed(1),
        DimExpr::fixed(1),
    ]);
    Ok(vec![ValueInfo::new(x.dtype, out)])
}

/// Reshape: target shape comes from "shape" attribute (list of ints).
///
/// A single `-1` dimension is allowed and is resolved based on the input's
/// total element count.
fn infer_reshape(node: &Node, inputs: &[&ValueInfo]) -> Result<Vec<ValueInfo>, InferError> {
    check_input_count(&node.op, inputs, 1)?;
    let target_dims = node
        .get_ints_attr("shape")
        .ok_or_else(|| InferError::MissingAttr {
            op: op_name(&node.op),
            attr: "shape".to_owned(),
        })?;

    let x = &inputs[0];

    let mut out_dims = Vec::with_capacity(target_dims.len());
    let mut neg_one_idx: Option<usize> = None;
    // Track which input dimension indices are *not* copied verbatim (d==0).
    // The inferred -1 dim = product(non-copied input dims) / product(explicit target dims).
    let mut copied_indices: Vec<usize> = Vec::new();
    let mut explicit_product: u64 = 1;

    for (i, &d) in target_dims.iter().enumerate() {
        if d == -1 {
            if neg_one_idx.is_some() {
                return Err(InferError::BadAttr {
                    op: op_name(&node.op),
                    attr: "shape".to_owned(),
                    msg: "at most one -1 dimension allowed".to_owned(),
                });
            }
            neg_one_idx = Some(i);
            out_dims.push(DimExpr::fixed(0)); // placeholder
        } else if d == 0 {
            // 0 means "copy from input"
            if i < x.shape.rank() {
                let dim = x.shape.dims()[i].clone();
                copied_indices.push(i);
                out_dims.push(dim);
            } else {
                return Err(InferError::BadAttr {
                    op: op_name(&node.op),
                    attr: "shape".to_owned(),
                    msg: format!("dim 0 at index {i} but input rank is {}", x.shape.rank()),
                });
            }
        } else {
            let v = to_u64(d);
            explicit_product *= v;
            out_dims.push(DimExpr::fixed(v));
        }
    }

    if let Some(idx) = neg_one_idx {
        // Compute the product of input dimensions that are NOT copied,
        // then divide by the product of explicit target dimensions.
        // This avoids symbolic / symbolic division (the copied symbolic dims
        // cancel out).
        let remaining_product = x
            .shape
            .dims()
            .iter()
            .enumerate()
            .filter(|(i, _)| !copied_indices.contains(i))
            .map(|(_, d)| d.clone())
            .reduce(|a, b| a * b)
            .unwrap_or(DimExpr::fixed(1));

        if explicit_product > 1 {
            out_dims[idx] = remaining_product / DimExpr::fixed(explicit_product);
        } else {
            out_dims[idx] = remaining_product;
        }
    }

    Ok(vec![ValueInfo::new(x.dtype, Shape::new(out_dims))])
}

/// Flatten: collapse dims `[axis..]` into one.
fn infer_flatten(node: &Node, inputs: &[&ValueInfo]) -> Result<Vec<ValueInfo>, InferError> {
    check_input_count(&node.op, inputs, 1)?;
    let x = &inputs[0];
    let axis = to_usize(node.get_int_attr("axis", 1));

    if axis > x.shape.rank() {
        return Err(InferError::BadAttr {
            op: op_name(&node.op),
            attr: "axis".to_owned(),
            msg: format!("axis {axis} > rank {}", x.shape.rank()),
        });
    }

    let leading: Vec<DimExpr> = x.shape.dims()[..axis].to_vec();
    let trailing_product = if axis < x.shape.rank() {
        x.shape.dims()[axis..]
            .iter()
            .cloned()
            .reduce(|a, b| a * b)
            .unwrap_or(DimExpr::fixed(1))
    } else {
        DimExpr::fixed(1)
    };

    let mut out_dims = leading;
    out_dims.push(trailing_product);

    Ok(vec![ValueInfo::new(x.dtype, Shape::new(out_dims))])
}

/// Transpose: permute dimensions according to "perm" attribute.
fn infer_transpose(node: &Node, inputs: &[&ValueInfo]) -> Result<Vec<ValueInfo>, InferError> {
    check_input_count(&node.op, inputs, 1)?;
    let x = &inputs[0];

    let perm = node
        .get_ints_attr("perm")
        .ok_or_else(|| InferError::MissingAttr {
            op: op_name(&node.op),
            attr: "perm".to_owned(),
        })?;

    if perm.len() != x.shape.rank() {
        return Err(InferError::BadAttr {
            op: op_name(&node.op),
            attr: "perm".to_owned(),
            msg: format!("perm length {} != rank {}", perm.len(), x.shape.rank()),
        });
    }

    let out_dims: Vec<DimExpr> = perm
        .iter()
        .map(|&p| x.shape.dims()[to_usize(p)].clone())
        .collect();

    Ok(vec![ValueInfo::new(x.dtype, Shape::new(out_dims))])
}

/// Concat: join tensors along a given axis.
fn infer_concat(node: &Node, inputs: &[&ValueInfo]) -> Result<Vec<ValueInfo>, InferError> {
    if inputs.is_empty() {
        return Err(InferError::MissingInput {
            op: op_name(&node.op),
            expected: 1,
            got: 0,
        });
    }
    let axis = to_usize(node.get_int_attr("axis", 0));
    let rank = inputs[0].shape.rank();

    if axis >= rank {
        return Err(InferError::BadAttr {
            op: op_name(&node.op),
            attr: "axis".to_owned(),
            msg: format!("axis {axis} >= rank {rank}"),
        });
    }

    // Sum the axis dimension across all inputs.
    let mut concat_dim = inputs[0].shape.dims()[axis].clone();
    for inp in &inputs[1..] {
        concat_dim = concat_dim + inp.shape.dims()[axis].clone();
    }

    let mut out_dims = inputs[0].shape.dims().to_vec();
    out_dims[axis] = concat_dim;

    Ok(vec![ValueInfo::new(inputs[0].dtype, Shape::new(out_dims))])
}

/// ONNX Shape operator.
///
/// Output is a 1-D I32 tensor whose length equals the (optionally sliced)
/// number of input dimensions.  Supports optional `start` and `end` attributes
/// with Python-style negative indexing.
fn infer_shape(node: &Node, inputs: &[&ValueInfo]) -> Result<Vec<ValueInfo>, InferError> {
    check_input_count(&node.op, inputs, 1)?;
    let rank = inputs[0].shape.rank() as i64;

    // Resolve start/end with Python-style negative indexing.
    let start_raw = node.get_int_attr("start", 0);
    let end_raw = node.get_int_attr("end", rank);

    let start = if start_raw < 0 {
        (rank + start_raw).max(0)
    } else {
        start_raw.min(rank)
    };
    let end = if end_raw < 0 {
        (rank + end_raw).max(0)
    } else {
        end_raw.min(rank)
    };

    let output_len = (end - start).max(0) as u64;
    Ok(vec![ValueInfo::new(
        DType::I32,
        Shape::from_fixed(&[output_len]),
    )])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Node, NodeId};
    use crate::op::OpKind;
    use crate::value::ValueId;

    fn vi(dtype: DType, shape: Shape) -> ValueInfo {
        ValueInfo::new(dtype, shape)
    }

    #[test]
    fn relu_preserves_shape() {
        let node = Node::new(
            NodeId::new(0),
            OpKind::Relu,
            vec![ValueId::new(0)],
            vec![ValueId::new(1)],
        );
        let input = vi(DType::F32, Shape::from_fixed(&[1, 64, 32, 32]));
        let out = infer_shapes(&node, &[&input]).unwrap();
        assert_eq!(out[0].shape, input.shape);
    }

    #[test]
    fn add_broadcast() {
        let node = Node::new(
            NodeId::new(0),
            OpKind::Add,
            vec![ValueId::new(0), ValueId::new(1)],
            vec![ValueId::new(2)],
        );
        let a = vi(DType::F32, Shape::from_fixed(&[4, 3, 32, 32]));
        let b = vi(DType::F32, Shape::from_fixed(&[1, 3, 1, 1])); // bias-like
        let out = infer_shapes(&node, &[&a, &b]).unwrap();
        assert_eq!(out[0].shape, Shape::from_fixed(&[4, 3, 32, 32]));
    }

    #[test]
    fn conv2d_valid_padding() {
        let mut node = Node::new(
            NodeId::new(0),
            OpKind::Conv2d,
            vec![ValueId::new(0), ValueId::new(1)],
            vec![ValueId::new(2)],
        );
        node.set_attr("strides", Attr::Ints(vec![1, 1]));
        // X: (1, 1, 5, 5), W: (1, 1, 3, 3) => valid out = (1, 1, 3, 3)
        let x = vi(DType::F32, Shape::from_fixed(&[1, 1, 5, 5]));
        let w = vi(DType::F32, Shape::from_fixed(&[1, 1, 3, 3]));
        let out = infer_shapes(&node, &[&x, &w]).unwrap();
        assert_eq!(out[0].shape, Shape::from_fixed(&[1, 1, 3, 3]));
    }

    #[test]
    fn conv2d_same_padding() {
        let mut node = Node::new(
            NodeId::new(0),
            OpKind::Conv2d,
            vec![ValueId::new(0), ValueId::new(1)],
            vec![ValueId::new(2)],
        );
        node.set_attr("strides", Attr::Ints(vec![1, 1]));
        node.set_attr("padding", Attr::Padding(PaddingMode::Same));
        let x = vi(DType::F32, Shape::from_fixed(&[1, 1, 5, 5]));
        let w = vi(DType::F32, Shape::from_fixed(&[1, 1, 3, 3]));
        let out = infer_shapes(&node, &[&x, &w]).unwrap();
        assert_eq!(out[0].shape, Shape::from_fixed(&[1, 1, 5, 5]));
    }

    #[test]
    fn conv2d_symbolic_batch() {
        let mut node = Node::new(
            NodeId::new(0),
            OpKind::Conv2d,
            vec![ValueId::new(0), ValueId::new(1)],
            vec![ValueId::new(2)],
        );
        node.set_attr("strides", Attr::Ints(vec![1, 1]));
        node.set_attr("padding", Attr::Padding(PaddingMode::Same));

        let x = vi(
            DType::F32,
            Shape::new(vec![
                DimExpr::sym("N"),
                DimExpr::fixed(3),
                DimExpr::fixed(224),
                DimExpr::fixed(224),
            ]),
        );
        let w = vi(DType::F32, Shape::from_fixed(&[64, 3, 7, 7]));
        let out = infer_shapes(&node, &[&x, &w]).unwrap();

        // Batch should propagate as N
        assert_eq!(out[0].shape.dims()[0], DimExpr::sym("N"));
        assert_eq!(out[0].shape.dims()[1], DimExpr::fixed(64));

        // Evaluate with concrete batch
        let env = [("N", 8)].into_iter().collect();
        let concrete = out[0].shape.evaluate(&env).unwrap();
        assert_eq!(concrete, vec![8, 64, 224, 224]);
    }

    #[test]
    fn fully_connected_shape() {
        let node = Node::new(
            NodeId::new(0),
            OpKind::FullyConnected,
            vec![ValueId::new(0), ValueId::new(1)],
            vec![ValueId::new(2)],
        );
        let x = vi(
            DType::F32,
            Shape::new(vec![DimExpr::sym("N"), DimExpr::fixed(512)]),
        );
        let w = vi(DType::F32, Shape::from_fixed(&[10, 512]));
        let out = infer_shapes(&node, &[&x, &w]).unwrap();
        assert_eq!(out[0].shape.dims()[0], DimExpr::sym("N"));
        assert_eq!(out[0].shape.dims()[1], DimExpr::fixed(10));
    }

    #[test]
    fn flatten_shape() {
        let mut node = Node::new(
            NodeId::new(0),
            OpKind::Flatten,
            vec![ValueId::new(0)],
            vec![ValueId::new(1)],
        );
        node.set_attr("axis", Attr::Int(1));
        let x = vi(
            DType::F32,
            Shape::new(vec![
                DimExpr::sym("N"),
                DimExpr::fixed(64),
                DimExpr::fixed(7),
                DimExpr::fixed(7),
            ]),
        );
        let out = infer_shapes(&node, &[&x]).unwrap();
        // (N, 64*7*7) = (N, 3136)
        assert_eq!(out[0].shape.dims()[0], DimExpr::sym("N"));
        assert_eq!(out[0].shape.dims()[1].try_fixed(), Some(3136));
    }

    #[test]
    fn global_avg_pool_shape() {
        let node = Node::new(
            NodeId::new(0),
            OpKind::GlobalAvgPool,
            vec![ValueId::new(0)],
            vec![ValueId::new(1)],
        );
        let x = vi(
            DType::F32,
            Shape::new(vec![
                DimExpr::sym("N"),
                DimExpr::fixed(512),
                DimExpr::fixed(7),
                DimExpr::fixed(7),
            ]),
        );
        let out = infer_shapes(&node, &[&x]).unwrap();
        assert_eq!(out[0].shape.dims()[0], DimExpr::sym("N"));
        assert_eq!(out[0].shape.dims()[1], DimExpr::fixed(512));
        assert_eq!(out[0].shape.dims()[2], DimExpr::fixed(1));
        assert_eq!(out[0].shape.dims()[3], DimExpr::fixed(1));
    }

    #[test]
    fn concat_along_axis() {
        let mut node = Node::new(
            NodeId::new(0),
            OpKind::Concat,
            vec![ValueId::new(0), ValueId::new(1)],
            vec![ValueId::new(2)],
        );
        node.set_attr("axis", Attr::Int(1));
        let a = vi(DType::F32, Shape::from_fixed(&[1, 32, 8, 8]));
        let b = vi(DType::F32, Shape::from_fixed(&[1, 64, 8, 8]));
        let out = infer_shapes(&node, &[&a, &b]).unwrap();
        assert_eq!(out[0].shape, Shape::from_fixed(&[1, 96, 8, 8]));
    }
}
