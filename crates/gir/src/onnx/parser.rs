//! Core ONNX → weaver-IR translation logic.

use std::collections::HashMap;

use prost::Message;

use crate::attr::{Attr, PaddingMode};
use crate::builder::GraphBuilder;
use crate::data::TensorData;
use crate::dim::DimExpr;
use crate::dtype::DType;
use crate::graph::Graph;
use crate::op::OpKind;
use crate::shape::Shape;
use crate::value::{ValueId, ValueInfo};
use crate::verify::verify;

use super::proto;

// ─── Error type ─────────────────────────────────────────────────────

/// Errors produced during ONNX parsing.
#[derive(Debug, Clone)]
pub enum OnnxError {
    /// The protobuf bytes could not be decoded.
    DecodeFailed(String),
    /// The model has no graph.
    MissingGraph,
    /// An ONNX element type code has no mapping to our [`DType`].
    UnsupportedDtype { onnx_code: i32, context: String },
    /// An ONNX `op_type` has no mapping to our [`OpKind`].
    UnsupportedOp { op_type: String, node_name: String },
    /// A node references a value that was never defined.
    UndefinedValue { name: String, node_name: String },
    /// An input or `value_info` lacks the required type/shape information.
    MissingTypeInfo { name: String },
    /// A required attribute is missing from an onnx operator.
    MissingAttribute { op: OpKind, attribute_name: String },
    /// Shape inference failed for a node.
    InferenceFailed {
        node_name: String,
        op_type: String,
        reason: String,
    },
    /// A dynamic dimension was found where we don't allow it (non-input).
    UnsupportedDynamic { name: String, detail: String },
    /// Graph verification failed after construction.
    VerificationFailed(Vec<String>),
    /// The reshape target shape could not be resolved from initializers.
    UnresolvedReshapeShape { node_name: String },
}

impl std::fmt::Display for OnnxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DecodeFailed(e) => write!(f, "failed to decode ONNX protobuf: {e}"),
            Self::MissingGraph => write!(f, "ONNX model has no graph"),
            Self::UnsupportedDtype { onnx_code, context } => {
                write!(f, "unsupported ONNX dtype {onnx_code} in {context}")
            }
            Self::UnsupportedOp { op_type, node_name } => {
                write!(f, "unsupported ONNX op '{op_type}' at node '{node_name}'")
            }
            Self::UndefinedValue { name, node_name } => {
                write!(f, "undefined value '{name}' at node '{node_name}'")
            }
            Self::MissingTypeInfo { name } => {
                write!(f, "missing type/shape info for '{name}'")
            }
            Self::MissingAttribute { op, attribute_name } => {
                write!(f, "missing attribute '{attribute_name}' for operator {op}")
            }
            Self::InferenceFailed {
                node_name,
                op_type,
                reason,
            } => write!(f, "shape inference for {op_type} ('{node_name}'): {reason}"),
            Self::UnsupportedDynamic { name, detail } => {
                write!(f, "unsupported dynamic shape for '{name}': {detail}")
            }
            Self::VerificationFailed(errs) => {
                write!(f, "verification failed: {}", errs.join("; "))
            }
            Self::UnresolvedReshapeShape { node_name } => {
                write!(f, "cannot resolve reshape target shape at '{node_name}'")
            }
        }
    }
}

impl std::error::Error for OnnxError {}

// ─── Public entry point ─────────────────────────────────────────────

/// Parse raw ONNX protobuf bytes into a weaver IR [`Graph`].
///
/// # Errors
///
/// Returns [`OnnxError`] if the model:
/// - cannot be decoded,
/// - contains unsupported operations or data types,
/// - has dynamic shapes that cannot be expressed in the IR, or
/// - fails verification after construction.
pub fn parse_onnx(bytes: &[u8]) -> Result<Graph, OnnxError> {
    let model =
        proto::ModelProto::decode(bytes).map_err(|e| OnnxError::DecodeFailed(e.to_string()))?;

    let onnx_graph = model.graph.ok_or(OnnxError::MissingGraph)?;
    let graph_name = if onnx_graph.name.is_empty() {
        "onnx_model".to_owned()
    } else {
        onnx_graph.name.clone()
    };

    let mut ctx = ParseCtx::new(&graph_name, &onnx_graph)?;
    ctx.parse_graph(&onnx_graph)?;
    let graph = ctx.finish()?;

    verify(&graph).map_err(|errs| {
        OnnxError::VerificationFailed(errs.into_iter().map(|e| e.message).collect())
    })?;

    Ok(graph)
}

// ─── Internal context ───────────────────────────────────────────────

/// Parsing context carrying builder state and name→id mappings.
struct ParseCtx {
    builder: GraphBuilder,
    /// ONNX tensor name → weaver `ValueId`.
    name_to_id: HashMap<String, ValueId>,
    /// Pre-extracted int64 data from small constant initializers (for Reshape).
    initializer_i64: HashMap<String, Vec<i64>>,
    /// ONNX `value_info` shapes: name → `ValueInfo`.
    value_info_map: HashMap<String, ValueInfo>,
    /// Names of graph outputs (to mark them at the end).
    output_names: Vec<String>,
}

impl ParseCtx {
    fn new(name: &str, onnx_graph: &proto::GraphProto) -> Result<Self, OnnxError> {
        let mut initializer_i64 = HashMap::new();

        // Pre-index initializers.
        for init in &onnx_graph.initializer {
            // Extract int64 data for small constants (e.g. reshape target shapes).
            if let Some(data) = extract_i64_data(init) {
                initializer_i64.insert(init.name.clone(), data);
            }
        }

        // Pre-index value_info.
        let mut value_info_map = HashMap::new();
        for vi in &onnx_graph.value_info {
            if let Some(info) = try_convert_value_info(vi)? {
                value_info_map.insert(vi.name.clone(), info);
            }
        }
        // Also index output value_infos.
        for vi in &onnx_graph.output {
            if let Some(info) = try_convert_value_info(vi)? {
                value_info_map.insert(vi.name.clone(), info);
            }
        }

        let output_names = onnx_graph.output.iter().map(|o| o.name.clone()).collect();

        Ok(Self {
            builder: GraphBuilder::new(name),
            name_to_id: HashMap::new(),
            initializer_i64,
            value_info_map,
            output_names,
        })
    }

    fn parse_graph(&mut self, onnx_graph: &proto::GraphProto) -> Result<(), OnnxError> {
        // ── 1. Register graph inputs ────────────────────────────────
        let init_names: std::collections::HashSet<&str> = onnx_graph
            .initializer
            .iter()
            .map(|i| i.name.as_str())
            .collect();

        let mut initializer_inputs = std::collections::HashMap::new();
        for inp in &onnx_graph.input {
            if init_names.contains(inp.name.as_str()) {
                // Cover the onnx IR <= 3 case where initializer tensor content is
                // stored in the initializer section but shape and value info is stored
                // in the inputs. We only process true graph inputs here and handle the
                // the intializer separately
                initializer_inputs.insert(inp.name.clone(), inp);
                continue;
            }
            let info = convert_value_info(inp, /*allow_sym=*/ true)?;
            let id = self
                .builder
                .add_input(info.dtype, info.shape, Some(&inp.name));
            self.name_to_id.insert(inp.name.clone(), id);
        }

        // ── 2. Register initializers ─────
        for init in &onnx_graph.initializer {
            let info = if let Some(v) = initializer_inputs.get(init.name.as_str()) {
                // Initializer value info is specified as a graph input. This is legacy onnx IR <= 3
                // behaviour. If that is the case we use the input value info here but convert the
                // initializer to constant tensor regardless
                convert_value_info(v, false)?
            } else {
                let dtype = map_tensor_dtype(init.data_type, &init.name)?;
                let shape = Shape::from_fixed(
                    &init
                        .dims
                        .iter()
                        .map(|&d| u64::try_from(d).unwrap_or(0))
                        .collect::<Vec<_>>(),
                );
                ValueInfo::new(dtype, shape).with_name(&init.name)
            };
            let dtype = info.dtype;
            let ids = self.builder.add_node_explicit(
                OpKind::Constant,
                &[],
                vec![info],
                vec![],
                Some(&init.name),
            );
            // Attach the actual weight data.
            if let Some(tensor_data) = extract_tensor_data(init, dtype) {
                self.builder.set_constant_data(ids[0], tensor_data);
            }
            self.name_to_id.insert(init.name.clone(), ids[0]);
        }

        // ── 3. Process nodes in order ───────────────────────────────
        for node in &onnx_graph.node {
            self.convert_node(node)?;
        }

        Ok(())
    }

    fn finish(self) -> Result<Graph, OnnxError> {
        let mut output_ids = Vec::new();
        for name in &self.output_names {
            let id =
                self.name_to_id
                    .get(name)
                    .copied()
                    .ok_or_else(|| OnnxError::UndefinedValue {
                        name: name.clone(),
                        node_name: "<graph output>".to_owned(),
                    })?;
            output_ids.push(id);
        }
        Ok(self.builder.build(output_ids))
    }

    // ── Node conversion ─────────────────────────────────────────────

    fn convert_node(&mut self, node: &proto::NodeProto) -> Result<(), OnnxError> {
        let node_name = if node.name.is_empty() {
            format!("{}_{}", node.op_type, self.name_to_id.len())
        } else {
            node.name.clone()
        };

        // Resolve the IR op kind.
        let op = map_op_type(&node.op_type, &node_name)?;

        // Resolve input ValueIds.
        let input_ids = self.resolve_inputs(node, &node_name)?;

        // Extract attributes.
        let attrs = self.convert_attrs(node, &op)?;

        // Number of outputs (most ops have 1; BatchNorm has 5 during training
        // but we only care about the first one for inference).
        let num_outputs = node.output.len().max(1);

        // Try shape inference via the builder.
        if num_outputs == 1 {
            // Special case: Constant op — use explicit shape from value_info or
            // the node's "value" attribute.
            if op == OpKind::Constant {
                let out_info = self.resolve_constant_output(node, &node_name)?;
                let ids = self.builder.add_node_explicit(
                    op,
                    &input_ids,
                    vec![out_info],
                    attrs,
                    Some(&node_name),
                );
                if let Some(out_name) = node.output.first() {
                    if !out_name.is_empty() {
                        self.name_to_id.insert(out_name.clone(), ids[0]);
                    }
                }
                return Ok(());
            }

            match self
                .builder
                .add_node_simple(op.clone(), &input_ids, attrs, Some(&node_name))
            {
                Ok(out_id) => {
                    if let Some(out_name) = node.output.first() {
                        if !out_name.is_empty() {
                            self.name_to_id.insert(out_name.clone(), out_id);
                        }
                    }
                }
                Err(infer_err) => {
                    // Fallback: use value_info if available.
                    if let Some(out_name) = node.output.first() {
                        if let Some(info) = self.value_info_map.get(out_name) {
                            let ids = self.builder.add_node_explicit(
                                op,
                                &input_ids,
                                vec![info.clone()],
                                vec![],
                                Some(&node_name),
                            );
                            self.name_to_id.insert(out_name.clone(), ids[0]);
                            return Ok(());
                        }
                    }
                    return Err(OnnxError::InferenceFailed {
                        node_name,
                        op_type: node.op_type.clone(),
                        reason: infer_err.to_string(),
                    });
                }
            }
        } else {
            // Multi-output node (e.g. BatchNorm inference has 1 useful output
            // but may list 5). We use value_info or inference for the first.
            let mut out_infos = Vec::with_capacity(num_outputs);
            for (i, out_name) in node.output.iter().enumerate() {
                if out_name.is_empty() {
                    // Optional output, skip with a dummy.
                    out_infos.push(ValueInfo::new(DType::F32, Shape::scalar()));
                    continue;
                }
                if let Some(vi) = self.value_info_map.get(out_name) {
                    out_infos.push(vi.clone());
                } else if i == 0 {
                    // For the primary output, try inference.
                    // Build a temporary node to infer.
                    let input_vis: Vec<&ValueInfo> = input_ids
                        .iter()
                        .filter_map(|id| self.builder.value_info(*id))
                        .collect();
                    let tmp_node_id = crate::node::NodeId::new(0);
                    let mut tmp_node =
                        crate::node::Node::new(tmp_node_id, op.clone(), vec![], vec![]);
                    // Copy attrs.
                    for (k, v) in &attrs {
                        tmp_node.set_attr(*k, v.clone());
                    }
                    match crate::infer::infer_shapes(&tmp_node, &input_vis) {
                        Ok(mut vis) => out_infos.push(vis.remove(0)),
                        Err(e) => {
                            return Err(OnnxError::InferenceFailed {
                                node_name,
                                op_type: node.op_type.clone(),
                                reason: e.to_string(),
                            });
                        }
                    }
                } else {
                    // Secondary outputs — best-effort with a scalar placeholder.
                    out_infos.push(ValueInfo::new(DType::F32, Shape::scalar()));
                }
            }

            let ids =
                self.builder
                    .add_node_explicit(op, &input_ids, out_infos, attrs, Some(&node_name));
            for (out_name, &id) in node.output.iter().zip(&ids) {
                if !out_name.is_empty() {
                    self.name_to_id.insert(out_name.clone(), id);
                }
            }
        }

        Ok(())
    }

    /// Resolve ONNX input names to [`ValueId`]s, skipping empty optional inputs.
    fn resolve_inputs(
        &self,
        node: &proto::NodeProto,
        node_name: &str,
    ) -> Result<Vec<ValueId>, OnnxError> {
        let mut ids = Vec::new();
        for name in &node.input {
            if name.is_empty() {
                continue; // optional input slot
            }
            let id =
                self.name_to_id
                    .get(name)
                    .copied()
                    .ok_or_else(|| OnnxError::UndefinedValue {
                        name: name.clone(),
                        node_name: node_name.to_owned(),
                    })?;
            ids.push(id);
        }
        Ok(ids)
    }

    /// Convert ONNX node attributes to our IR [`Attr`] list.
    #[allow(clippy::too_many_lines)]
    fn convert_attrs<'a>(
        &self,
        node: &proto::NodeProto,
        op: &OpKind,
    ) -> Result<Vec<(&'a str, Attr)>, OnnxError> {
        let onnx_attrs = &node.attribute;
        let mut attrs: Vec<(&str, Attr)> = Vec::new();

        // Helper to find an ONNX attribute by name.
        let find = |name: &str| -> Option<&proto::AttributeProto> {
            onnx_attrs.iter().find(|a| a.name == name)
        };

        match op {
            OpKind::Conv2d | OpKind::DepthwiseConv2d => {
                // ref: https://onnx.ai/onnx/operators/onnx__Conv.html

                // We do not check the kernel size attribute here because we will infer it from the
                // weight input itself which will be statically known.

                // strides: default is [1, 1] (1 along each spatial axis).
                let strides = find("strides")
                    .map(|a| a.ints.clone())
                    .unwrap_or_else(|| vec![1, 1]);
                attrs.push(("strides", Attr::Ints(strides)));

                // dilations: default is [1, 1] (1 along each spatial axis).
                let dilations = find("dilations")
                    .map(|a| a.ints.clone())
                    .unwrap_or_else(|| vec![1, 1]);
                attrs.push(("dilations", Attr::Ints(dilations)));

                // group: default is 1.
                let group = find("group").map(|a| a.i).unwrap_or(1);
                attrs.push(("group", Attr::Int(group)));

                // Padding: ONNX uses "auto_pad" or explicit "pads".
                // auto_pad default is "NOTSET" (i.e. use explicit pads).
                let auto_pad = find("auto_pad")
                    .and_then(|a| std::str::from_utf8(&a.s).ok().map(str::to_owned));
                match auto_pad.as_deref() {
                    Some("SAME_UPPER" | "SAME_LOWER") => {
                        attrs.push(("padding", Attr::Padding(PaddingMode::Same)));
                    }
                    Some("VALID") => {
                        attrs.push(("padding", Attr::Padding(PaddingMode::Valid)));
                    }
                    None | Some("NOTSET") => {
                        // Check explicit pads; default is all zeros (Valid).
                        if let Some(a) = find("pads") {
                            if a.ints.iter().any(|&p| p != 0) {
                                attrs.push(("pads", Attr::Ints(a.ints.clone())));
                                attrs.push(("padding", Attr::Padding(PaddingMode::Explicit)));
                            } else {
                                attrs.push(("padding", Attr::Padding(PaddingMode::Valid)));
                            }
                        } else {
                            attrs.push(("padding", Attr::Padding(PaddingMode::Valid)));
                        }
                    }
                    _ => {}
                }
            }

            OpKind::MaxPool2d | OpKind::AvgPool2d => {
                // ref: https://onnx.ai/onnx/operators/onnx__MaxPool.html
                // ref: https://onnx.ai/onnx/operators/onnx__AveragePool.html

                // required parameter
                if let Some(a) = find("kernel_shape") {
                    attrs.push(("kernel_size", Attr::Ints(a.ints.clone())));
                } else {
                    return Err(OnnxError::MissingAttribute {
                        op: op.clone(),
                        attribute_name: "kernel_shape".to_string(),
                    });
                }

                // strides: default is [1, 1] (1 along each spatial axis).
                let strides = find("strides")
                    .map(|a| a.ints.clone())
                    .unwrap_or_else(|| vec![1, 1]);
                attrs.push(("strides", Attr::Ints(strides)));

                // dilations: default is [1, 1] (1 along each spatial axis).
                let dilations = find("dilations")
                    .map(|a| a.ints.clone())
                    .unwrap_or_else(|| vec![1, 1]);
                attrs.push(("dilations", Attr::Ints(dilations)));

                // Padding: same logic as Conv2d.
                let auto_pad = find("auto_pad")
                    .and_then(|a| std::str::from_utf8(&a.s).ok().map(str::to_owned));
                match auto_pad.as_deref() {
                    Some("SAME_UPPER" | "SAME_LOWER") => {
                        attrs.push(("padding", Attr::Padding(PaddingMode::Same)));
                    }
                    Some("VALID") => {
                        attrs.push(("padding", Attr::Padding(PaddingMode::Valid)));
                    }
                    None | Some("NOTSET") => {
                        if let Some(a) = find("pads") {
                            if a.ints.iter().any(|&p| p != 0) {
                                attrs.push(("pads", Attr::Ints(a.ints.clone())));
                                attrs.push(("padding", Attr::Padding(PaddingMode::Explicit)));
                            } else {
                                attrs.push(("padding", Attr::Padding(PaddingMode::Valid)));
                            }
                        } else {
                            attrs.push(("padding", Attr::Padding(PaddingMode::Valid)));
                        }
                    }
                    _ => {}
                }
            }

            OpKind::Reshape => {
                // In ONNX, the target shape is the second input (a constant).
                // We need to extract it and pass as an attribute.
                if let Some(shape_input_name) = node.input.get(1) {
                    if let Some(shape_data) = self.initializer_i64.get(shape_input_name) {
                        attrs.push(("shape", Attr::Ints(shape_data.clone())));
                    } else {
                        return Err(OnnxError::UnresolvedReshapeShape {
                            node_name: node.name.clone(),
                        });
                    }
                }
            }

            OpKind::Flatten => {
                if let Some(a) = find("axis") {
                    attrs.push(("axis", Attr::Int(a.i)));
                }
            }

            OpKind::Transpose => {
                if let Some(a) = find("perm") {
                    attrs.push(("perm", Attr::Ints(a.ints.clone())));
                }
            }

            OpKind::Concat => {
                if let Some(a) = find("axis") {
                    attrs.push(("axis", Attr::Int(a.i)));
                }
            }

            OpKind::Softmax => {
                if let Some(a) = find("axis") {
                    attrs.push(("axis", Attr::Int(a.i)));
                }
            }

            OpKind::Clip => {
                // ONNX Clip: min/max might be attributes (opset < 11) or inputs.
                if let Some(a) = find("min") {
                    attrs.push(("min", Attr::Float(f64::from(a.f))));
                }
                if let Some(a) = find("max") {
                    attrs.push(("max", Attr::Float(f64::from(a.f))));
                }
            }

            OpKind::Shape => {
                // Optional start/end attributes to slice dimensions.
                if let Some(a) = find("start") {
                    attrs.push(("start", Attr::Int(a.i)));
                }
                if let Some(a) = find("end") {
                    attrs.push(("end", Attr::Int(a.i)));
                }
            }

            // Elementwise & unary ops: no special attributes needed.
            _ => {}
        }

        Ok(attrs)
    }

    /// Resolve the output `ValueInfo` for a `Constant` node.
    fn resolve_constant_output(
        &self,
        node: &proto::NodeProto,
        node_name: &str,
    ) -> Result<ValueInfo, OnnxError> {
        // Try value_info first.
        if let Some(out_name) = node.output.first() {
            if let Some(vi) = self.value_info_map.get(out_name) {
                return Ok(vi.clone());
            }
        }

        // Try the "value" attribute (TensorProto).
        let value_attr = node.attribute.iter().find(|a| a.name == "value");
        if let Some(attr) = value_attr {
            if let Some(ref t) = attr.t {
                let dtype = map_tensor_dtype(t.data_type, node_name)?;
                let shape = Shape::from_fixed(
                    &t.dims
                        .iter()
                        .map(|&d| u64::try_from(d).unwrap_or(0))
                        .collect::<Vec<_>>(),
                );
                return Ok(ValueInfo::new(dtype, shape).with_name(node_name));
            }
        }

        Err(OnnxError::MissingTypeInfo {
            name: node_name.to_owned(),
        })
    }
}

// ─── Op mapping ─────────────────────────────────────────────────────

fn map_op_type(op_type: &str, node_name: &str) -> Result<OpKind, OnnxError> {
    match op_type {
        "Conv" => Ok(OpKind::Conv2d),
        "ConvInteger" => Ok(OpKind::Conv2d),
        "Gemm" => Ok(OpKind::FullyConnected),
        "MatMul" | "MatMulInteger" => Ok(OpKind::MatMul),
        "Add" => Ok(OpKind::Add),
        "Sub" => Ok(OpKind::Sub),
        "Mul" => Ok(OpKind::Mul),
        "Relu" => Ok(OpKind::Relu),
        "Sigmoid" => Ok(OpKind::Sigmoid),
        "Tanh" => Ok(OpKind::Tanh),
        "Clip" => Ok(OpKind::Clip),
        "Softmax" => Ok(OpKind::Softmax),
        "MaxPool" => Ok(OpKind::MaxPool2d),
        "AveragePool" => Ok(OpKind::AvgPool2d),
        "GlobalAveragePool" => Ok(OpKind::GlobalAvgPool),
        "BatchNormalization" => Ok(OpKind::BatchNorm),
        "LayerNormalization" => Ok(OpKind::LayerNorm),
        "Reshape" => Ok(OpKind::Reshape),
        "Transpose" => Ok(OpKind::Transpose),
        "Flatten" => Ok(OpKind::Flatten),
        "Concat" => Ok(OpKind::Concat),
        "QuantizeLinear" => Ok(OpKind::Quantize),
        "DequantizeLinear" => Ok(OpKind::Dequantize),
        "GRU" => Ok(OpKind::Gru),
        "LSTM" => Ok(OpKind::Lstm),
        "Resize" | "Upsample" => Ok(OpKind::Resize),
        "Pad" => Ok(OpKind::Pad),
        "Shape" => Ok(OpKind::Shape),
        "Constant" | "ConstantOfShape" => Ok(OpKind::Constant),
        _ => Err(OnnxError::UnsupportedOp {
            op_type: op_type.to_owned(),
            node_name: node_name.to_owned(),
        }),
    }
}

// ─── Dtype mapping ──────────────────────────────────────────────────

fn map_tensor_dtype(onnx_code: i32, context: &str) -> Result<DType, OnnxError> {
    // Values from TensorProto::DataType enum.
    match onnx_code {
        1 => Ok(DType::F32),  // FLOAT
        2 => Ok(DType::U8),   // UINT8
        3 => Ok(DType::I8),   // INT8
        4 => Ok(DType::U16),  // UINT16
        5 => Ok(DType::I16),  // INT16
        6 => Ok(DType::I32),  // INT32
        7 => Ok(DType::I32),  // INT64 → clamped to I32
        9 => Ok(DType::Bool), // BOOL
        10 => Ok(DType::F16), // FLOAT16
        11 => Ok(DType::F32), // DOUBLE → clamped to F32
        12 => Ok(DType::U32), // UINT32
        13 => Ok(DType::U32), // UINT64 → clamped to U32
        _ => Err(OnnxError::UnsupportedDtype {
            onnx_code,
            context: context.to_owned(),
        }),
    }
}

fn map_elem_type(onnx_code: i32, context: &str) -> Result<DType, OnnxError> {
    map_tensor_dtype(onnx_code, context)
}

// ─── Shape / type extraction ────────────────────────────────────────

/// Convert an ONNX `ValueInfoProto` to our `ValueInfo`.
///
/// When `allow_sym` is true, unknown dims become symbolic `DimExpr`s.
/// Otherwise, they produce an error.
fn convert_value_info(vi: &proto::ValueInfoProto, allow_sym: bool) -> Result<ValueInfo, OnnxError> {
    let tp = vi
        .r#type
        .as_ref()
        .ok_or_else(|| OnnxError::MissingTypeInfo {
            name: vi.name.clone(),
        })?;

    let Some(proto::type_proto::Value::TensorType(tensor_type)) = &tp.value else {
        return Err(OnnxError::MissingTypeInfo {
            name: vi.name.clone(),
        });
    };

    let dtype = map_elem_type(tensor_type.elem_type, &vi.name)?;
    let shape = convert_shape(tensor_type.shape.as_ref(), &vi.name, allow_sym)?;

    Ok(ValueInfo::new(dtype, shape).with_name(&vi.name))
}

/// Like `convert_value_info` but returns `None` for non-tensor types instead
/// of erroring.
fn try_convert_value_info(vi: &proto::ValueInfoProto) -> Result<Option<ValueInfo>, OnnxError> {
    let Some(ref tp) = vi.r#type else {
        return Ok(None);
    };
    let Some(proto::type_proto::Value::TensorType(ref t)) = tp.value else {
        return Ok(None);
    };
    let dtype = map_elem_type(t.elem_type, &vi.name)?;
    let shape = convert_shape(t.shape.as_ref(), &vi.name, true)?;
    Ok(Some(ValueInfo::new(dtype, shape).with_name(&vi.name)))
}

fn convert_shape(
    shape: Option<&proto::TensorShapeProto>,
    name: &str,
    allow_sym: bool,
) -> Result<Shape, OnnxError> {
    let Some(sp) = shape else {
        return Err(OnnxError::MissingTypeInfo {
            name: name.to_owned(),
        });
    };

    let mut dims = Vec::with_capacity(sp.dim.len());
    for d in &sp.dim {
        match &d.value {
            Some(proto::tensor_shape_proto::dimension::Value::DimValue(v)) => {
                dims.push(DimExpr::fixed(u64::try_from(*v).unwrap_or(0)));
            }
            Some(proto::tensor_shape_proto::dimension::Value::DimParam(s)) => {
                if !allow_sym {
                    return Err(OnnxError::UnsupportedDynamic {
                        name: name.to_owned(),
                        detail: format!("symbolic dim '{s}'"),
                    });
                }
                let sym_name = if s.is_empty() { "N" } else { s.as_str() };
                dims.push(DimExpr::sym(sym_name));
            }
            None => {
                // No dimension info at all.
                if !allow_sym {
                    return Err(OnnxError::UnsupportedDynamic {
                        name: name.to_owned(),
                        detail: "dimension has no value".to_owned(),
                    });
                }
                // Treat as symbolic with a generated name.
                dims.push(DimExpr::sym("N"));
            }
        }
    }
    Ok(Shape::new(dims))
}

// ─── Initializer data extraction ────────────────────────────────────

/// Extract the raw tensor data from an ONNX `TensorProto` initializer and wrap
/// it as a [`TensorData`].
///
/// Supports `raw_data` (zero-copy when dtype matches) as well as the typed
/// repeated fields (`float_data`, `int32_data`, `int64_data`, etc.).
///
/// Returns `None` only if the tensor has no data at all (e.g. an empty shape).
fn extract_tensor_data(tensor: &proto::TensorProto, dtype: DType) -> Option<TensorData> {
    // ── 1. Prefer raw_data (already in little-endian byte order). ─────
    if !tensor.raw_data.is_empty() {
        return Some(TensorData::from_raw(dtype, tensor.raw_data.clone()));
    }

    // ── 2. Typed repeated fields. ─────────────────────────────────────
    match tensor.data_type {
        // FLOAT (1)
        1 if !tensor.float_data.is_empty() => Some(TensorData::from_f32s(&tensor.float_data)),
        // UINT8 (2)
        2 if !tensor.int32_data.is_empty() => {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let bytes: Vec<u8> = tensor.int32_data.iter().map(|&v| v as u8).collect();
            Some(TensorData::from_u8s(&bytes))
        }
        // INT8 (3)
        3 if !tensor.int32_data.is_empty() => {
            #[allow(clippy::cast_possible_truncation)]
            let bytes: Vec<i8> = tensor.int32_data.iter().map(|&v| v as i8).collect();
            Some(TensorData::from_i8s(&bytes))
        }
        // INT32 (6)
        6 if !tensor.int32_data.is_empty() => Some(TensorData::from_i32s(&tensor.int32_data)),
        // INT64 (7) → stored as I32
        7 if !tensor.int64_data.is_empty() => {
            Some(TensorData::from_i64s_as_i32(&tensor.int64_data))
        }
        // DOUBLE (11) → stored as F32
        11 if !tensor.double_data.is_empty() => {
            #[allow(clippy::cast_possible_truncation)]
            let f32s: Vec<f32> = tensor.double_data.iter().map(|&v| v as f32).collect();
            Some(TensorData::from_f32s(&f32s))
        }
        _ => None,
    }
}

/// Extract int64 values from a `TensorProto` (for Reshape shapes, etc.).
fn extract_i64_data(tensor: &proto::TensorProto) -> Option<Vec<i64>> {
    // Only extract from 1-D integer tensors.
    if tensor.dims.len() > 1 {
        return None;
    }

    // Prefer int64_data.
    if !tensor.int64_data.is_empty() {
        return Some(tensor.int64_data.clone());
    }

    // Try raw_data (little-endian int64 bytes).
    if !tensor.raw_data.is_empty() && tensor.data_type == 7 {
        // INT64
        let data: Vec<i64> = tensor
            .raw_data
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap_or_default()))
            .collect();
        return Some(data);
    }

    // Try int32_data for INT32 type.
    if !tensor.int32_data.is_empty() {
        return Some(tensor.int32_data.iter().map(|&v| i64::from(v)).collect());
    }

    None
}

#[cfg(test)]
#[allow(dead_code)]
mod tests {
    use super::*;
    use crate::onnx::proto;
    use prost::Message;

    /// Helper to build a simple `ValueInfoProto` with a tensor type.
    fn make_value_info(
        name: &str,
        elem_type: i32,
        dims: Vec<proto::tensor_shape_proto::Dimension>,
    ) -> proto::ValueInfoProto {
        proto::ValueInfoProto {
            name: name.to_owned(),
            r#type: Some(proto::TypeProto {
                value: Some(proto::type_proto::Value::TensorType(
                    proto::type_proto::Tensor {
                        elem_type,
                        shape: Some(proto::TensorShapeProto { dim: dims }),
                    },
                )),
                denotation: String::new(),
            }),
            doc_string: String::new(),
            metadata_props: vec![],
        }
    }

    fn fixed_dim(v: i64) -> proto::tensor_shape_proto::Dimension {
        proto::tensor_shape_proto::Dimension {
            value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(v)),
            denotation: String::new(),
        }
    }

    fn sym_dim(name: &str) -> proto::tensor_shape_proto::Dimension {
        proto::tensor_shape_proto::Dimension {
            value: Some(proto::tensor_shape_proto::dimension::Value::DimParam(
                name.to_owned(),
            )),
            denotation: String::new(),
        }
    }

    fn make_node(
        op_type: &str,
        inputs: &[&str],
        outputs: &[&str],
        name: &str,
        attrs: Vec<proto::AttributeProto>,
    ) -> proto::NodeProto {
        proto::NodeProto {
            input: inputs.iter().map(|s| (*s).to_owned()).collect(),
            output: outputs.iter().map(|s| (*s).to_owned()).collect(),
            name: name.to_owned(),
            op_type: op_type.to_owned(),
            domain: String::new(),
            overload: String::new(),
            attribute: attrs,
            doc_string: String::new(),
            metadata_props: vec![],
        }
    }

    fn int_attr(name: &str, val: i64) -> proto::AttributeProto {
        proto::AttributeProto {
            name: name.to_owned(),
            i: val,
            r#type: proto::attribute_proto::AttributeType::Int as i32,
            ..Default::default()
        }
    }

    fn ints_attr(name: &str, vals: Vec<i64>) -> proto::AttributeProto {
        proto::AttributeProto {
            name: name.to_owned(),
            ints: vals,
            r#type: proto::attribute_proto::AttributeType::Ints as i32,
            ..Default::default()
        }
    }

    fn string_attr(name: &str, val: &str) -> proto::AttributeProto {
        proto::AttributeProto {
            name: name.to_owned(),
            s: val.as_bytes().to_vec(),
            r#type: proto::attribute_proto::AttributeType::String as i32,
            ..Default::default()
        }
    }

    fn make_initializer(name: &str, dims: &[i64], data_type: i32) -> proto::TensorProto {
        proto::TensorProto {
            name: name.to_owned(),
            dims: dims.to_vec(),
            data_type,
            raw_data: vec![0; usize::try_from(dims.iter().product::<i64>()).unwrap_or(0) * 4],
            ..Default::default()
        }
    }

    fn make_int64_initializer(name: &str, values: &[i64]) -> proto::TensorProto {
        proto::TensorProto {
            name: name.to_owned(),
            dims: vec![i64::try_from(values.len()).unwrap_or(0)],
            data_type: 7, // INT64
            int64_data: values.to_vec(),
            ..Default::default()
        }
    }

    fn make_model(graph: proto::GraphProto) -> proto::ModelProto {
        proto::ModelProto {
            ir_version: 8,
            graph: Some(graph),
            opset_import: vec![proto::OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn parse_simple_relu() {
        let graph = proto::GraphProto {
            name: "simple_relu".to_owned(),
            input: vec![make_value_info("X", 1, vec![sym_dim("N"), fixed_dim(64)])],
            output: vec![make_value_info("Y", 1, vec![sym_dim("N"), fixed_dim(64)])],
            node: vec![make_node("Relu", &["X"], &["Y"], "relu_0", vec![])],
            ..Default::default()
        };

        let model = make_model(graph);
        let mut buf = Vec::new();
        model.encode(&mut buf).unwrap();

        let ir = parse_onnx(&buf).unwrap();
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.free_symbols(), vec!["N"]);
    }

    #[test]
    fn parse_conv_relu() {
        let graph = proto::GraphProto {
            name: "conv_relu".to_owned(),
            input: vec![make_value_info(
                "X",
                1,
                vec![sym_dim("N"), fixed_dim(3), fixed_dim(32), fixed_dim(32)],
            )],
            initializer: vec![make_initializer("W", &[16, 3, 3, 3], 1)],
            output: vec![make_value_info(
                "Y",
                1,
                vec![sym_dim("N"), fixed_dim(16), fixed_dim(32), fixed_dim(32)],
            )],
            node: vec![
                make_node(
                    "Conv",
                    &["X", "W"],
                    &["conv_out"],
                    "conv_0",
                    vec![
                        ints_attr("strides", vec![1, 1]),
                        ints_attr("pads", vec![1, 1, 1, 1]),
                    ],
                ),
                make_node("Relu", &["conv_out"], &["Y"], "relu_0", vec![]),
            ],
            value_info: vec![make_value_info(
                "conv_out",
                1,
                vec![sym_dim("N"), fixed_dim(16), fixed_dim(32), fixed_dim(32)],
            )],
            ..Default::default()
        };

        let model = make_model(graph);
        let mut buf = Vec::new();
        model.encode(&mut buf).unwrap();

        let ir = parse_onnx(&buf).unwrap();
        assert_eq!(ir.num_nodes(), 3); // Constant(W) + Conv + Relu or 2 if W is input

        let env = [("N", 2)].into_iter().collect();
        let shapes = ir.evaluate_shapes(&env);
        // Find the output value
        let out_shape = &shapes[&ir.outputs[0]];
        assert_eq!(out_shape, &[2, 16, 32, 32]);
    }

    #[test]
    fn unsupported_op_rejected() {
        let graph = proto::GraphProto {
            name: "bad_op".to_owned(),
            input: vec![make_value_info("X", 1, vec![fixed_dim(2), fixed_dim(3)])],
            output: vec![make_value_info("Y", 1, vec![fixed_dim(2), fixed_dim(3)])],
            node: vec![make_node("MyCustomOp", &["X"], &["Y"], "custom_0", vec![])],
            ..Default::default()
        };

        let model = make_model(graph);
        let mut buf = Vec::new();
        model.encode(&mut buf).unwrap();

        let err = parse_onnx(&buf).unwrap_err();
        assert!(matches!(err, OnnxError::UnsupportedOp { .. }));
    }
}
