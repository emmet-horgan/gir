//! Tensor shapes composed of [`DimExpr`](crate::dim::DimExpr) dimensions.
//!
//! A [`Shape`] is a ranked, ordered list of dimension expressions.  Every tensor
//! in the IR carries a shape so that downstream passes can reason about memory
//! layout, tiling, and buffer allocation — even when one or more dimensions are
//! symbolic.

use std::collections::HashMap;
use std::fmt;

use serde::Serialize;

use crate::dim::DimExpr;

/// An ordered list of dimension expressions describing a tensor's extents.
#[derive(Clone, PartialEq, Eq, Hash, Serialize)]
pub struct Shape {
    dims: Vec<DimExpr>,
}

impl Shape {
    // ── Constructors ─────────────────────────────────────────────────

    /// Create a shape from an explicit list of dimension expressions.
    #[must_use]
    pub fn new(dims: Vec<DimExpr>) -> Self {
        Self { dims }
    }

    /// Convenience: build a fully-static shape from plain integers.
    #[must_use]
    pub fn from_fixed(dims: &[u64]) -> Self {
        Self {
            dims: dims.iter().map(|&d| DimExpr::fixed(d)).collect(),
        }
    }

    /// A scalar (rank-0) shape.
    #[must_use]
    pub fn scalar() -> Self {
        Self { dims: Vec::new() }
    }

    // ── Queries ──────────────────────────────────────────────────────

    /// The rank (number of dimensions) of this shape.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Access the dimension expressions as a slice.
    #[must_use]
    pub fn dims(&self) -> &[DimExpr] {
        &self.dims
    }

    /// Check whether all dimensions are fixed (no symbolic variables).
    #[must_use]
    pub fn is_static(&self) -> bool {
        self.dims.iter().all(DimExpr::is_static)
    }

    /// Try to evaluate every dimension and return concrete extents.
    ///
    /// Returns `None` if any dimension cannot be evaluated.
    #[must_use]
    pub fn evaluate(&self, env: &HashMap<&str, u64>) -> Option<Vec<u64>> {
        self.dims.iter().map(|d| d.evaluate(env)).collect()
    }

    /// Compute the total number of elements (product of all dimensions).
    ///
    /// The result is itself a `DimExpr`, which may be symbolic.
    #[must_use]
    pub fn num_elements(&self) -> DimExpr {
        if self.dims.is_empty() {
            return DimExpr::fixed(1); // scalar
        }
        self.dims
            .iter()
            .cloned()
            .reduce(|acc, d| acc * d)
            .unwrap_or(DimExpr::fixed(1))
    }

    /// Collect all free symbolic names across every dimension.
    #[must_use]
    pub fn free_symbols(&self) -> Vec<String> {
        let mut syms: Vec<String> = self.dims.iter().flat_map(DimExpr::free_symbols).collect();
        syms.sort();
        syms.dedup();
        syms
    }

    /// Substitute a symbolic name in every dimension.
    #[must_use]
    pub fn substitute(&self, name: &str, replacement: &DimExpr) -> Self {
        Self {
            dims: self
                .dims
                .iter()
                .map(|d| d.substitute(name, replacement))
                .collect(),
        }
    }
}

// ── Display / Debug ──────────────────────────────────────────────────

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape({self})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_shape() {
        let s = Shape::from_fixed(&[1, 3, 224, 224]);
        assert_eq!(s.rank(), 4);
        assert!(s.is_static());
        assert_eq!(s.num_elements().try_fixed(), Some(3 * 224 * 224));
    }

    #[test]
    fn symbolic_shape() {
        let s = Shape::new(vec![
            DimExpr::sym("N"),
            DimExpr::fixed(3),
            DimExpr::fixed(224),
            DimExpr::fixed(224),
        ]);
        assert!(!s.is_static());
        assert_eq!(s.free_symbols(), vec!["N".to_owned()]);

        let env: HashMap<&str, u64> = [("N", 4)].into_iter().collect();
        assert_eq!(s.evaluate(&env), Some(vec![4, 3, 224, 224]));
    }

    #[test]
    fn display_shape() {
        let s = Shape::new(vec![DimExpr::sym("N"), DimExpr::fixed(64)]);
        assert_eq!(format!("{s}"), "[N, 64]");
    }

    #[test]
    fn scalar_shape() {
        let s = Shape::scalar();
        assert_eq!(s.rank(), 0);
        assert!(s.is_static());
        assert_eq!(s.num_elements().try_fixed(), Some(1));
    }
}
