//! IR graph nodes.
//!
//! A [`Node`] binds an [`OpKind`] to its input and output [`ValueId`]s and any
//! compile-time [`Attr`]ibutes.  Nodes are stored inside a [`Graph`] and form
//! a directed acyclic graph (DAG) via their value references.

use std::collections::BTreeMap;
use std::fmt;

use serde::Serialize;

use crate::attr::Attr;
use crate::op::OpKind;
use crate::value::ValueId;

/// A unique identifier for a node within a [`Graph`](crate::graph::Graph).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct NodeId(u32);

impl NodeId {
    #[must_use]
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node_{}", self.0)
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({})", self.0)
    }
}

/// A single operation in the IR graph.
#[derive(Debug, Clone, Serialize)]
pub struct Node {
    /// Unique identifier within the graph.
    pub id: NodeId,
    /// The kind of operation this node performs.
    pub op: OpKind,
    /// Ordered list of input value references.
    pub inputs: Vec<ValueId>,
    /// Ordered list of output value references.
    pub outputs: Vec<ValueId>,
    /// Named attributes (compile-time parameters).
    pub attrs: BTreeMap<String, Attr>,
    /// Optional human-readable name for debugging.
    pub name: Option<String>,
}

impl Node {
    /// Create a new node with no attributes and no name.
    #[must_use]
    pub fn new(id: NodeId, op: OpKind, inputs: Vec<ValueId>, outputs: Vec<ValueId>) -> Self {
        Self {
            id,
            op,
            inputs,
            outputs,
            attrs: BTreeMap::new(),
            name: None,
        }
    }

    /// Set a named attribute.
    pub fn set_attr(&mut self, key: impl Into<String>, value: Attr) {
        self.attrs.insert(key.into(), value);
    }

    /// Get a named attribute.
    #[must_use]
    pub fn get_attr(&self, key: &str) -> Option<&Attr> {
        self.attrs.get(key)
    }

    /// Convenience: get an integer attribute or return a default.
    #[must_use]
    pub fn get_int_attr(&self, key: &str, default: i64) -> i64 {
        match self.attrs.get(key) {
            Some(Attr::Int(v)) => *v,
            _ => default,
        }
    }

    /// Convenience: get an integer-list attribute or return a default.
    #[must_use]
    pub fn get_ints_attr(&self, key: &str) -> Option<&[i64]> {
        match self.attrs.get(key) {
            Some(Attr::Ints(v)) => Some(v.as_slice()),
            _ => None,
        }
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Outputs
        let outs: Vec<String> = self.outputs.iter().map(ToString::to_string).collect();
        write!(f, "{}", outs.join(", "))?;
        write!(f, " = {}", self.op)?;

        // Inputs
        let ins: Vec<String> = self.inputs.iter().map(ToString::to_string).collect();
        write!(f, "({})", ins.join(", "))?;

        // Attributes
        if !self.attrs.is_empty() {
            write!(f, " {{")?;
            for (i, (k, v)) in self.attrs.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{k}={v}")?;
            }
            write!(f, "}}")?;
        }

        Ok(())
    }
}
