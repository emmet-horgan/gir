//! Ergonomic builder for constructing IR graphs.
//!
//! The [`GraphBuilder`] handles value-id allocation, automatic shape inference
//! for node outputs, and graph assembly.  It is the recommended way to
//! programmatically construct a [`Graph`].
//!
//! # Example
//!
//! ```
//! use gir::builder::GraphBuilder;
//! use gir::dim::DimExpr;
//! use gir::dtype::DType;
//! use gir::shape::Shape;
//! use gir::op::OpKind;
//! use gir::attr::{Attr, PaddingMode};
//!
//! let mut b = GraphBuilder::new("tiny_cnn");
//!
//! // Model input: (N, 1, 28, 28) — symbolic batch
//! let x = b.add_input(
//!     DType::F32,
//!     Shape::new(vec![
//!         DimExpr::sym("N"),
//!         DimExpr::fixed(1),
//!         DimExpr::fixed(28),
//!         DimExpr::fixed(28),
//!     ]),
//!     Some("input"),
//! );
//!
//! // Conv2d weight (constant): (8, 1, 3, 3)
//! let w = b.add_input(DType::F32, Shape::from_fixed(&[8, 1, 3, 3]), Some("conv.weight"));
//!
//! // Conv2d with same-padding
//! let conv = b.add_node_simple(
//!     OpKind::Conv2d,
//!     &[x, w],
//!     vec![
//!         ("strides", Attr::Ints(vec![1, 1])),
//!         ("padding", Attr::Padding(PaddingMode::Same)),
//!     ],
//!     None,
//! ).unwrap();
//!
//! // ReLU
//! let relu = b.add_node_simple(OpKind::Relu, &[conv], vec![], None).unwrap();
//!
//! let graph = b.build(vec![relu]);
//! assert_eq!(graph.free_symbols(), vec!["N"]);
//!
//! // Resolve shapes at runtime
//! let env = [("N", 16)].into_iter().collect();
//! let shapes = graph.evaluate_shapes(&env);
//! let relu_shape = shapes.get(&relu).unwrap();
//! assert_eq!(relu_shape, &[16, 8, 28, 28]);
//! ```

use crate::attr::Attr;
use crate::data::TensorData;
use crate::dtype::DType;
use crate::graph::Graph;
use crate::infer::{InferError, infer_shapes};
use crate::node::{Node, NodeId};
use crate::op::OpKind;
use crate::shape::Shape;
use crate::value::{ValueId, ValueInfo};

/// Builder for constructing a [`Graph`] incrementally.
pub struct GraphBuilder {
    name: String,
    next_value: u32,
    next_node: u32,
    inputs: Vec<ValueId>,
    nodes: Vec<Node>,
    values: std::collections::HashMap<ValueId, ValueInfo>,
    constants: std::collections::HashMap<ValueId, TensorData>,
}

impl GraphBuilder {
    /// Create a new builder with the given graph name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            next_value: 0,
            next_node: 0,
            inputs: Vec::new(),
            nodes: Vec::new(),
            values: std::collections::HashMap::new(),
            constants: std::collections::HashMap::new(),
        }
    }

    /// Allocate a fresh [`ValueId`].
    fn alloc_value(&mut self) -> ValueId {
        let id = ValueId::new(self.next_value);
        self.next_value += 1;
        id
    }

    /// Allocate a fresh [`NodeId`].
    fn alloc_node(&mut self) -> NodeId {
        let id = NodeId::new(self.next_node);
        self.next_node += 1;
        id
    }

    /// Register a graph-level input.
    ///
    /// Returns the [`ValueId`] that downstream nodes should reference.
    pub fn add_input(&mut self, dtype: DType, shape: Shape, name: Option<&str>) -> ValueId {
        let id = self.alloc_value();
        let mut info = ValueInfo::new(dtype, shape);
        if let Some(n) = name {
            info = info.with_name(n);
        }
        self.values.insert(id, info);
        self.inputs.push(id);
        id
    }

    /// Look up the [`ValueInfo`] for a value produced so far.
    ///
    /// # Errors
    ///
    /// Returns `None` if the id hasn't been registered.
    #[must_use]
    pub fn value_info(&self, id: ValueId) -> Option<&ValueInfo> {
        self.values.get(&id)
    }

    /// Attach dense tensor data to a value (e.g. weights, biases).
    ///
    /// This associates the raw weight bytes with the given `ValueId` so
    /// they can be retrieved from [`Graph::constant_data`] after building.
    pub fn set_constant_data(&mut self, id: ValueId, data: TensorData) {
        self.constants.insert(id, data);
    }

    /// Add a node that produces exactly one output, with automatic shape
    /// inference.
    ///
    /// Returns the output [`ValueId`] on success.
    ///
    /// # Errors
    ///
    /// Returns [`InferError`] if shape inference fails for the given
    /// combination of inputs, operation, and attributes.
    pub fn add_node_simple(
        &mut self,
        op: OpKind,
        input_ids: &[ValueId],
        attrs: Vec<(&str, Attr)>,
        name: Option<&str>,
    ) -> Result<ValueId, InferError> {
        let out_id = self.alloc_value();
        let node_id = self.alloc_node();

        let mut node = Node::new(node_id, op, input_ids.to_vec(), vec![out_id]);
        for (k, v) in attrs {
            node.set_attr(k, v);
        }
        if let Some(n) = name {
            node.name = Some(n.to_owned());
        }

        // Gather input value infos for shape inference.
        let input_infos: Vec<&ValueInfo> = input_ids
            .iter()
            .map(|id| {
                self.values.get(id).expect(
                    "input ValueId not found — was it produced by an earlier node or add_input?",
                )
            })
            .collect();

        let mut out_infos = infer_shapes(&node, &input_infos)?;
        let out_info = out_infos.remove(0);
        self.values.insert(out_id, out_info);
        self.nodes.push(node);

        Ok(out_id)
    }

    /// Add a node with explicit output type information (bypasses inference).
    ///
    /// Useful for `Constant` nodes or custom ops where inference is not
    /// implemented.
    pub fn add_node_explicit(
        &mut self,
        op: OpKind,
        input_ids: &[ValueId],
        outputs: Vec<ValueInfo>,
        attrs: Vec<(&str, Attr)>,
        name: Option<&str>,
    ) -> Vec<ValueId> {
        let node_id = self.alloc_node();
        let out_ids: Vec<ValueId> = (0..outputs.len()).map(|_| self.alloc_value()).collect();

        let mut node = Node::new(node_id, op, input_ids.to_vec(), out_ids.clone());
        for (k, v) in attrs {
            node.set_attr(k, v);
        }
        if let Some(n) = name {
            node.name = Some(n.to_owned());
        }

        for (id, info) in out_ids.iter().zip(outputs) {
            self.values.insert(*id, info);
        }
        self.nodes.push(node);

        out_ids
    }

    /// Finalise and return the constructed [`Graph`].
    ///
    /// `output_ids` specifies which values are the graph-level outputs.
    #[must_use]
    pub fn build(self, output_ids: Vec<ValueId>) -> Graph {
        Graph {
            name: self.name,
            inputs: self.inputs,
            outputs: output_ids,
            nodes: self.nodes,
            values: self.values,
            constants: self.constants,
        }
    }
}
