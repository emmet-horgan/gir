//! Graph verification — check structural and type consistency.
//!
//! [`verify`] walks the graph and reports any issues it finds. This is intended
//! to be run after graph construction (or after a transformation pass) as a
//! safety check.

use crate::graph::Graph;
use crate::value::ValueId;

/// A single verification failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyError {
    pub message: String,
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "verify: {}", self.message)
    }
}

impl std::error::Error for VerifyError {}

/// Verify structural and type consistency of a [`Graph`].
///
/// Returns `Ok(())` when the graph passes all checks, or a list of errors.
///
/// # Errors
///
/// Returns a `Vec<VerifyError>` describing every inconsistency found.
pub fn verify(graph: &Graph) -> Result<(), Vec<VerifyError>> {
    let mut errors = Vec::new();

    // 1. Every graph input must have a ValueInfo.
    for &id in &graph.inputs {
        if graph.value_info(id).is_none() {
            errors.push(VerifyError {
                message: format!("graph input {id} has no ValueInfo"),
            });
        }
    }

    // 2. Every graph output must have a ValueInfo.
    for &id in &graph.outputs {
        if graph.value_info(id).is_none() {
            errors.push(VerifyError {
                message: format!("graph output {id} has no ValueInfo"),
            });
        }
    }

    // 3. Every node input must be defined (either a graph input or an output of
    //    a preceding node).
    let mut defined: std::collections::HashSet<ValueId> = graph.inputs.iter().copied().collect();

    for node in graph.nodes() {
        for &inp in &node.inputs {
            if !defined.contains(&inp) {
                errors.push(VerifyError {
                    message: format!("node {} ({}) uses undefined value {inp}", node.id, node.op),
                });
            }
        }
        // Mark outputs as defined.
        for &out in &node.outputs {
            if !defined.insert(out) {
                errors.push(VerifyError {
                    message: format!(
                        "node {} ({}) re-defines already-defined value {out}",
                        node.id, node.op
                    ),
                });
            }
        }
    }

    // 4. Every graph output must be produced by some node (or be a pass-through
    //    input).
    for &id in &graph.outputs {
        if !defined.contains(&id) {
            errors.push(VerifyError {
                message: format!("graph output {id} is not produced by any node"),
            });
        }
    }

    // 5. Every node output must have a ValueInfo.
    for node in graph.nodes() {
        for &out in &node.outputs {
            if graph.value_info(out).is_none() {
                errors.push(VerifyError {
                    message: format!(
                        "node {} ({}) output {out} has no ValueInfo",
                        node.id, node.op
                    ),
                });
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use crate::dim::DimExpr;
    use crate::dtype::DType;
    use crate::op::OpKind;
    use crate::shape::Shape;

    #[test]
    fn valid_graph_passes_verification() {
        let mut b = GraphBuilder::new("test");
        let x = b.add_input(
            DType::F32,
            Shape::new(vec![DimExpr::sym("N"), DimExpr::fixed(64)]),
            Some("x"),
        );
        let relu = b.add_node_simple(OpKind::Relu, &[x], vec![], None).unwrap();
        let graph = b.build(vec![relu]);
        assert!(verify(&graph).is_ok());
    }
}
