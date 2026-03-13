//! # weaver-affine — Affine Integer Algebra & Constraint System
//!
//! A self-contained affine expression and constraint system for symbolic
//! tensor dimension reasoning.  No external SMT libraries required.
//!
//! ## Overview
//!
//! | Module | Purpose |
//! |---|---|
//! | [`expr`] | Canonical affine expressions over named symbols |
//! | [`constraint`] | Linear equality/inequality and divisibility constraints |
//! | [`solver`] | Partial compile-time constraint solver |
//!
//! ## Quick example
//!
//! ```
//! use gir_affine::prelude::*;
//!
//! let mut sys = ConstraintSystem::new();
//!
//! // N - 8 == 0  →  N = 8
//! let expr = AffineExpr::symbol("N") - AffineExpr::constant(8);
//! sys.add(Constraint::EqZero(expr));
//!
//! let result = sys.solve();
//! assert!(result.is_ok());
//! assert_eq!(sys.solved_value("N"), Some(8));
//! ```

pub mod constraint;
pub mod expr;
pub mod solver;

/// Convenience re-exports.
pub mod prelude {
    pub use crate::constraint::Constraint;
    pub use crate::expr::AffineExpr;
    pub use crate::solver::{ConstraintSystem, SolveError, SolveResult};
}
