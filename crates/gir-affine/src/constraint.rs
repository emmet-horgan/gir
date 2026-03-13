//! Constraint types for affine expressions.
//!
//! All constraints are stored over [`AffineExpr`] values and are normalised
//! on construction.

use std::fmt;

use serde::Serialize;

use crate::expr::AffineExpr;

/// A single constraint over affine expressions.
///
/// All variants store their expression in canonical normalised form.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub enum Constraint {
    /// `expr == 0`
    EqZero(AffineExpr),

    /// `expr >= 0`
    GeZero(AffineExpr),

    /// `expr > 0`
    GtZero(AffineExpr),

    /// `expr` is divisible by `k`:  `expr % k == 0`.
    ///
    /// The divisor `k` is always stored as a positive value.
    Divisible(AffineExpr, i64),
}

impl Constraint {
    /// Create an equality constraint: `lhs == rhs` ⇔ `(lhs - rhs) == 0`.
    ///
    /// The stored expression is normalised for equality.
    #[must_use]
    pub fn eq(lhs: &AffineExpr, rhs: &AffineExpr) -> Self {
        Self::EqZero(lhs.sub(rhs).normalize_eq())
    }

    /// Create a `>=` constraint: `lhs >= rhs` ⇔ `(lhs - rhs) >= 0`.
    ///
    /// The stored expression is normalised for inequalities (GCD only, no
    /// sign flip).
    #[must_use]
    pub fn ge(lhs: &AffineExpr, rhs: &AffineExpr) -> Self {
        Self::GeZero(lhs.sub(rhs).normalize_ineq())
    }

    /// Create a `>` constraint: `lhs > rhs` ⇔ `(lhs - rhs) > 0`.
    #[must_use]
    pub fn gt(lhs: &AffineExpr, rhs: &AffineExpr) -> Self {
        Self::GtZero(lhs.sub(rhs).normalize_ineq())
    }

    /// Create a divisibility constraint: `expr % k == 0`.
    ///
    /// # Panics
    ///
    /// Panics if `k` is zero.
    #[must_use]
    pub fn divisible(expr: AffineExpr, k: i64) -> Self {
        assert!(k != 0, "divisor must be nonzero");
        // Do NOT use normalize_eq here — dividing by GCD would destroy
        // the divisibility relationship (e.g. 12 % 4 == 0 → 1 % 4 != 0).
        Self::Divisible(expr, k.abs())
    }

    /// Reference to the inner expression.
    #[must_use]
    pub fn expr(&self) -> &AffineExpr {
        match self {
            Self::EqZero(e) | Self::GeZero(e) | Self::GtZero(e) | Self::Divisible(e, _) => e,
        }
    }

    /// Substitute concrete values into the constraint's expression.
    #[must_use]
    pub fn substitute(&self, env: &std::collections::HashMap<String, i64>) -> Self {
        match self {
            Self::EqZero(e) => Self::EqZero(e.substitute(env).normalize_eq()),
            Self::GeZero(e) => Self::GeZero(e.substitute(env).normalize_ineq()),
            Self::GtZero(e) => Self::GtZero(e.substitute(env).normalize_ineq()),
            Self::Divisible(e, k) => Self::Divisible(e.substitute(env), *k),
        }
    }

    /// Check whether this constraint is trivially satisfied (i.e. the
    /// expression is fully constant and the condition holds).
    #[must_use]
    pub fn is_trivially_satisfied(&self) -> Option<bool> {
        match self {
            Self::EqZero(e) => e.evaluate_if_constant().map(|c| c == 0),
            Self::GeZero(e) => e.evaluate_if_constant().map(|c| c >= 0),
            Self::GtZero(e) => e.evaluate_if_constant().map(|c| c > 0),
            Self::Divisible(e, k) => e.evaluate_if_constant().map(|c| c % k == 0),
        }
    }

    /// `true` if the constraint is trivially violated.
    #[must_use]
    pub fn is_contradiction(&self) -> bool {
        self.is_trivially_satisfied() == Some(false)
    }
}

impl fmt::Display for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EqZero(e) => write!(f, "{e} == 0"),
            Self::GeZero(e) => write!(f, "{e} >= 0"),
            Self::GtZero(e) => write!(f, "{e} > 0"),
            Self::Divisible(e, k) => write!(f, "{e} divisible by {k}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eq_normalises() {
        // N == 8  →  (N - 8) == 0
        let c = Constraint::eq(&AffineExpr::symbol("N"), &AffineExpr::constant(8));
        let Constraint::EqZero(e) = &c else {
            panic!("expected EqZero");
        };
        assert_eq!(e.coeff("N"), 1);
        assert_eq!(e.constant_term(), -8);
    }

    #[test]
    fn eq_normalises_with_gcd() {
        // 2N == 8  →  (2N - 8) norm → (N - 4) == 0
        let c = Constraint::eq(
            &AffineExpr::symbol("N").mul_by_constant(2),
            &AffineExpr::constant(8),
        );
        let Constraint::EqZero(e) = &c else {
            panic!("expected EqZero");
        };
        assert_eq!(e.coeff("N"), 1);
        assert_eq!(e.constant_term(), -4);
    }

    #[test]
    fn ge_constraint() {
        // N >= 1  →  (N - 1) >= 0
        let c = Constraint::ge(&AffineExpr::symbol("N"), &AffineExpr::constant(1));
        let Constraint::GeZero(e) = &c else {
            panic!("expected GeZero");
        };
        assert_eq!(e.coeff("N"), 1);
        assert_eq!(e.constant_term(), -1);
    }

    #[test]
    fn trivially_satisfied() {
        let c = Constraint::EqZero(AffineExpr::constant(0));
        assert_eq!(c.is_trivially_satisfied(), Some(true));

        let c2 = Constraint::EqZero(AffineExpr::constant(5));
        assert_eq!(c2.is_trivially_satisfied(), Some(false));
        assert!(c2.is_contradiction());
    }

    #[test]
    fn symbolic_not_trivial() {
        let c = Constraint::EqZero(AffineExpr::symbol("N"));
        assert_eq!(c.is_trivially_satisfied(), None);
    }

    #[test]
    fn divisibility() {
        let c = Constraint::divisible(AffineExpr::constant(12), 4);
        assert_eq!(c.is_trivially_satisfied(), Some(true));

        let c2 = Constraint::divisible(AffineExpr::constant(13), 4);
        assert_eq!(c2.is_trivially_satisfied(), Some(false));
    }

    #[test]
    fn display() {
        let c = Constraint::eq(&AffineExpr::symbol("N"), &AffineExpr::constant(8));
        assert_eq!(c.to_string(), "N - 8 == 0");

        let c2 = Constraint::ge(&AffineExpr::symbol("H"), &AffineExpr::constant(1));
        assert_eq!(c2.to_string(), "H - 1 >= 0");
    }

    #[test]
    fn substitute_solves() {
        let c = Constraint::eq(&AffineExpr::symbol("N"), &AffineExpr::constant(8));
        let env = [("N".to_owned(), 8)].into_iter().collect();
        let c2 = c.substitute(&env);
        assert_eq!(c2.is_trivially_satisfied(), Some(true));
    }

    #[test]
    #[should_panic(expected = "divisor must be nonzero")]
    fn divisible_zero_panics() {
        let _ = Constraint::divisible(AffineExpr::symbol("N"), 0);
    }
}
