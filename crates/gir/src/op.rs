//! IR operation kinds.
//!
//! Each variant of [`OpKind`] describes a single, side-effect-free tensor
//! operation.  The set is deliberately minimal — covering the operators most
//! commonly found in embedded ML inference workloads — but the enum is
//! non-exhaustive so that future extensions (e.g. control-flow, custom
//! accelerator ops) can be added without breaking existing code.
//!
//! Shape inference for each operation is implemented in
//! [`crate::infer`].

use std::fmt;

use serde::Serialize;

/// The kind of computation performed by an IR node.
///
/// # Extensibility
///
/// This enum is marked `#[non_exhaustive]`; new variants can be introduced in
/// minor releases.  When matching on `OpKind`, always include a wildcard arm.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub enum OpKind {
    // ── Linear algebra / core ────────────────────────────────────────
    /// 2-D convolution (input, weight, optional bias).
    Conv2d,
    /// Depthwise-separable 2-D convolution.
    DepthwiseConv2d,
    /// Fully-connected / dense / matrix-multiply + bias.
    FullyConnected,
    /// General matrix multiply (no bias).
    MatMul,

    // ── Element-wise arithmetic ──────────────────────────────────────
    /// Element-wise addition (with broadcast).
    Add,
    /// Element-wise subtraction (with broadcast).
    Sub,
    /// Element-wise multiplication (with broadcast).
    Mul,

    // ── Activation functions ─────────────────────────────────────────
    /// Rectified linear unit: `max(0, x)`.
    Relu,
    /// Sigmoid: `1 / (1 + exp(-x))`.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// Clamp to `[min, max]`.
    Clip,
    /// Softmax along a given axis.
    Softmax,

    // ── Pooling ──────────────────────────────────────────────────────
    /// Max pooling over a spatial window.
    MaxPool2d,
    /// Average pooling over a spatial window.
    AvgPool2d,
    /// Global average pooling (reduce spatial dims to 1×1).
    GlobalAvgPool,

    // ── Normalisation ────────────────────────────────────────────────
    /// Batch normalisation (inference mode only).
    BatchNorm,
    /// Layer normalisation.
    LayerNorm,

    // ── Shape manipulation ───────────────────────────────────────────
    /// Reshape tensor to a new shape (same number of elements).
    Reshape,
    /// Transpose / permute dimensions.
    Transpose,
    /// Flatten dimensions `[start_dim .. end_dim]` into one.
    Flatten,
    /// Concatenate tensors along a given axis.
    Concat,
    /// Extract input tensor's shape as a 1-D integer tensor.
    ///
    /// ONNX `Shape` operator — optional `start` / `end` attributes select a
    /// slice of dimensions (Python-style negative indexing is supported).
    Shape,

    // ── Quantisation ─────────────────────────────────────────────────
    /// Quantise float tensor to integer (scale + zero-point).
    Quantize,
    /// Dequantise integer tensor back to float.
    Dequantize,

    // ── Recurrent ────────────────────────────────────────────────────
    /// Gated Recurrent Unit.
    Gru,
    /// Long Short-Term Memory.
    Lstm,

    // ── Misc ─────────────────────────────────────────────────────────
    /// Constant tensor embedded in the IR (weights, biases, etc.).
    Constant,
    /// Resize / upsample (nearest or bilinear).
    Resize,
    /// Pad tensor with a given value.
    Pad,
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Use the Debug name which matches the variant identifier.
        fmt::Debug::fmt(self, f)
    }
}
