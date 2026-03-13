//! # weaver-ir — Embedded ML Model Intermediate Representation
//!
//! A simple, scalable intermediate representation for machine-learning models
//! targeting embedded / edge devices.
//!
//! ## Design principles
//!
//! | Principle | Detail |
//! |---|---|
//! | **Static typing** | Every tensor carries a [`DType`](dtype::DType) and [`Shape`](shape::Shape) that are statically known or expressible as symbolic expressions. |
//! | **Symbolic dimensions** | A [`DimExpr`](dim::DimExpr) wraps an [`AffineExpr`](weaver_affine::expr::AffineExpr) — a canonical affine expression over named symbols. Arithmetic normalises automatically and interoperates directly with the constraint solver in `weaver-affine`. |
//! | **DAG-only (for now)** | The graph is a pure DAG — no conditionals, loops, or sub-graphs. The [`OpKind`](op::OpKind) enum is `#[non_exhaustive]` so control-flow constructs can be introduced later without breaking existing code. |
//! | **Embedded focus** | The default [`DType`](dtype::DType) palette emphasises quantised integers and small floats common on MCUs and NPUs. |
//!
//! ## Quick start
//!
//! ```
//! use gir::prelude::*;
//!
//! let mut b = GraphBuilder::new("tiny_net");
//!
//! // Symbolic-batch input tensor
//! let x = b.add_input(
//!     DType::F32,
//!     Shape::new(vec![DimExpr::sym("N"), DimExpr::fixed(784)]),
//!     Some("input"),
//! );
//!
//! // Weight matrix (constant)
//! let w = b.add_input(DType::F32, Shape::from_fixed(&[10, 784]), Some("fc.weight"));
//!
//! // Fully-connected layer
//! let fc = b.add_node_simple(OpKind::FullyConnected, &[x, w], vec![], None).unwrap();
//!
//! // Activation
//! let out = b.add_node_simple(OpKind::Relu, &[fc], vec![], None).unwrap();
//!
//! let graph = b.build(vec![out]);
//!
//! // Shapes propagate symbolically
//! assert_eq!(graph.free_symbols(), vec!["N"]);
//!
//! // Bind N=4 and compute concrete shapes
//! let env = [("N", 4)].into_iter().collect();
//! let shapes = graph.evaluate_shapes(&env);
//! assert_eq!(shapes[&out], vec![4, 10]);
//! ```
//!
//! ## Modules
//!
//! | Module | Purpose |
//! |---|---|
//! | [`dim`] | Symbolic dimension expressions |
//! | [`dtype`] | Scalar element types |
//! | [`shape`] | Tensor shapes |
//! | [`value`] | SSA-style value identifiers and type info |
//! | [`attr`] | Compile-time operation attributes |
//! | [`op`] | Operation kinds |
//! | [`node`] | Graph nodes |
//! | [`graph`] | The IR graph |
//! | [`builder`] | Graph construction API |
//! | [`infer`] | Shape inference |
//! | [`data`] | Dense tensor data (weights) |
//! | [`verify`] | Graph verification |
//! | [`onnx`] | ONNX parser (feature-gated) |

pub mod attr;
pub mod builder;
pub mod data;
pub mod dim;
pub mod dtype;
pub mod graph;
pub mod infer;
pub mod node;
pub mod op;
pub mod shape;
pub mod value;
pub mod verify;

#[cfg(feature = "onnx")]
pub mod onnx;

/// Convenience re-exports for common usage.
pub mod prelude {
    pub use crate::attr::{Attr, PaddingMode};
    pub use crate::builder::GraphBuilder;
    pub use crate::data::TensorData;
    pub use crate::dim::DimExpr;
    pub use crate::dtype::DType;
    pub use crate::graph::Graph;
    pub use crate::infer::{InferError, infer_shapes};
    pub use crate::node::{Node, NodeId};
    pub use crate::op::OpKind;
    pub use crate::shape::Shape;
    pub use crate::value::{ValueId, ValueInfo};
    pub use crate::verify::verify;

    // Re-export weaver-affine types for constraint-based reasoning.
    pub use gir_affine::constraint::Constraint;
    pub use gir_affine::expr::AffineExpr;
    pub use gir_affine::solver::{ConstraintSystem, SolveError, SolveResult};
}
