//! The IR graph — a DAG of [`Node`]s connected through typed [`ValueId`]s.
//!
//! A [`Graph`] owns all nodes and value metadata for a single model (or
//! sub-graph).  It is the primary data structure that compiler passes read and
//! transform.
//!
//! # Layout
//!
//! * **inputs** — the graph-level input `ValueId`s (model inputs).
//! * **outputs** — the graph-level output `ValueId`s (model outputs).
//! * **values** — metadata ([`ValueInfo`]) for every value in the graph.
//! * **nodes** — the ordered list of [`Node`]s (topological order).
//!
//! Because the graph is a DAG (no conditionals or loops — yet), a simple
//! topological ordering of the node list is sufficient for scheduling.

use std::collections::HashMap;
use std::fmt;

use serde::Serialize;

use crate::data::TensorData;
use crate::node::Node;
use crate::op::OpKind;
use crate::value::{ValueId, ValueInfo};

/// A directed acyclic graph of tensor operations.
#[derive(Debug, Clone, Serialize)]
pub struct Graph {
    /// Human-readable model / sub-graph name.
    pub name: String,
    /// Graph-level inputs (in order).
    pub inputs: Vec<ValueId>,
    /// Graph-level outputs (in order).
    pub outputs: Vec<ValueId>,
    /// All nodes, in topological order.
    pub(crate) nodes: Vec<Node>,
    /// Type information for every value.
    pub(crate) values: HashMap<ValueId, ValueInfo>,
    /// Dense data for constant values (weights, biases, etc.).
    pub(crate) constants: HashMap<ValueId, TensorData>,
}

impl Graph {
    /// Create an empty graph with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            nodes: Vec::new(),
            values: HashMap::new(),
            constants: HashMap::new(),
        }
    }

    // ── Accessors ────────────────────────────────────────────────────

    /// Iterate over all nodes in topological order.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Look up the [`ValueInfo`] for a given value.
    #[must_use]
    pub fn value_info(&self, id: ValueId) -> Option<&ValueInfo> {
        self.values.get(&id)
    }

    /// Number of nodes in the graph.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of distinct values (tensors).
    #[must_use]
    pub fn num_values(&self) -> usize {
        self.values.len()
    }

    /// Look up the constant [`TensorData`] for a value, if it has one.
    #[must_use]
    pub fn constant_data(&self, id: ValueId) -> Option<&TensorData> {
        self.constants.get(&id)
    }

    /// Number of constants (values with associated tensor data).
    #[must_use]
    pub fn num_constants(&self) -> usize {
        self.constants.len()
    }

    /// Find the node that produces a given value, if any.
    #[must_use]
    pub fn producer_of(&self, value: ValueId) -> Option<&Node> {
        self.nodes.iter().find(|n| n.outputs.contains(&value))
    }

    /// Find all nodes that consume a given value.
    #[must_use]
    pub fn consumers_of(&self, value: ValueId) -> Vec<&Node> {
        self.nodes
            .iter()
            .filter(|n| n.inputs.contains(&value))
            .collect()
    }

    /// Evaluate all value shapes given a concrete binding for symbolic dims.
    ///
    /// Returns a map from `ValueId` to a vector of concrete dimension sizes.
    #[must_use]
    pub fn evaluate_shapes(&self, env: &HashMap<&str, u64>) -> HashMap<ValueId, Vec<u64>> {
        self.values
            .iter()
            .filter_map(|(id, info)| info.shape.evaluate(env).map(|dims| (*id, dims)))
            .collect()
    }

    /// Collect all free symbolic dimension names across the graph.
    #[must_use]
    pub fn free_symbols(&self) -> Vec<String> {
        let mut syms: Vec<String> = self
            .values
            .values()
            .flat_map(|v| v.shape.free_symbols())
            .collect();
        syms.sort();
        syms.dedup();
        syms
    }

    /// Compute the total number of parameters in the model from the constant tensor
    /// storage.
    #[must_use]
    pub fn parameters(&self) -> u64 {
        let mut params = 0u64;
        let consts = self
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, OpKind::Constant));
        for n in consts {
            let out_id = n.outputs[0];
            let info = self
                .value_info(out_id)
                .expect("output value not stored in the graph");
            let els = info.shape.num_elements();

            // We are analyzing static constant data, the dimensions *must* be
            // fixed.
            params += els
                .try_fixed()
                .expect("constant tensors must have static dimensions");
        }

        params
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "graph {}:", self.name)?;

        // Inputs
        for &id in &self.inputs {
            if let Some(info) = self.values.get(&id) {
                writeln!(f, "  input {id}: {info}")?;
            }
        }

        // Nodes
        for node in &self.nodes {
            write!(f, "  {node}")?;
            // Annotate output types
            for out in &node.outputs {
                if let Some(info) = self.values.get(out) {
                    write!(f, "  // {out}: {info}")?;
                }
            }
            writeln!(f)?;
        }

        // Outputs
        write!(f, "  return ")?;
        let outs: Vec<String> = self.outputs.iter().map(ToString::to_string).collect();
        writeln!(f, "{}", outs.join(", "))?;

        Ok(())
    }
}
