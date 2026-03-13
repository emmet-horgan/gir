//! Canonical affine expressions over named integer symbols.
//!
//! An [`AffineExpr`] represents a linear combination of symbolic variables
//! plus a constant:
//!
//! ```text
//! a₁·x₁ + a₂·x₂ + … + c
//! ```
//!
//! All operations maintain canonical form automatically:
//! - Zero coefficients are removed.
//! - Symbols are stored in a [`BTreeMap`] for deterministic ordering.
//! - Equivalent expressions are structurally identical.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::ops;

use serde::Serialize;

/// A canonical affine expression: `Σ (coeff_i * symbol_i) + constant`.
///
/// Coefficients are signed 64-bit integers.  Symbols are identified by
/// [`String`].  The internal [`BTreeMap`] guarantees deterministic iteration
/// order.
///
/// # Invariants
///
/// - No entry with a zero coefficient exists in `coeffs`.
/// - After every operation the expression is fully reduced.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct AffineExpr {
    /// Symbol name → coefficient.  Zero entries are never stored.
    coeffs: BTreeMap<String, i64>,
    /// The constant term.
    constant: i64,
}

// ── Constructors ─────────────────────────────────────────────────────

impl AffineExpr {
    /// The zero expression (`0`).
    #[must_use]
    pub fn zero() -> Self {
        Self {
            coeffs: BTreeMap::new(),
            constant: 0,
        }
    }

    /// A pure symbolic variable with coefficient 1: `1 · name`.
    #[must_use]
    pub fn symbol(name: &str) -> Self {
        let mut coeffs = BTreeMap::new();
        coeffs.insert(name.to_owned(), 1);
        Self {
            coeffs,
            constant: 0,
        }
    }

    /// A constant expression with no symbols.
    #[must_use]
    pub fn constant(c: i64) -> Self {
        Self {
            coeffs: BTreeMap::new(),
            constant: c,
        }
    }

    /// Build from raw parts, canonicalising by stripping zero coefficients.
    #[must_use]
    pub fn from_parts(coeffs: BTreeMap<String, i64>, constant: i64) -> Self {
        let mut expr = Self { coeffs, constant };
        expr.canonicalize();
        expr
    }
}

// ── Accessors ────────────────────────────────────────────────────────

impl AffineExpr {
    /// The constant term.
    #[must_use]
    pub fn constant_term(&self) -> i64 {
        self.constant
    }

    /// Coefficient of a given symbol (0 if absent).
    #[must_use]
    pub fn coeff(&self, sym: &str) -> i64 {
        self.coeffs.get(sym).copied().unwrap_or(0)
    }

    /// An iterator over `(symbol, coefficient)` pairs in sorted order.
    pub fn terms(&self) -> impl Iterator<Item = (&str, i64)> {
        self.coeffs.iter().map(|(s, &c)| (s.as_str(), c))
    }

    /// The set of free symbols in sorted order.
    #[must_use]
    pub fn symbols(&self) -> Vec<&str> {
        self.coeffs.keys().map(String::as_str).collect()
    }

    /// Number of distinct symbols with nonzero coefficients.
    #[must_use]
    pub fn num_symbols(&self) -> usize {
        self.coeffs.len()
    }

    /// `true` if the expression has no symbolic terms.
    #[must_use]
    pub fn is_constant(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Returns the constant value if no symbols are present.
    #[must_use]
    pub fn evaluate_if_constant(&self) -> Option<i64> {
        if self.is_constant() {
            Some(self.constant)
        } else {
            None
        }
    }
}

// ── Arithmetic operations ────────────────────────────────────────────

impl AffineExpr {
    /// `self + other`, preserving canonical form.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        let mut coeffs = self.coeffs.clone();
        for (sym, &c) in &other.coeffs {
            let entry = coeffs.entry(sym.clone()).or_insert(0);
            *entry += c;
        }
        let mut result = Self {
            coeffs,
            constant: self.constant + other.constant,
        };
        result.canonicalize();
        result
    }

    /// `self - other`, preserving canonical form.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        let mut coeffs = self.coeffs.clone();
        for (sym, &c) in &other.coeffs {
            let entry = coeffs.entry(sym.clone()).or_insert(0);
            *entry -= c;
        }
        let mut result = Self {
            coeffs,
            constant: self.constant - other.constant,
        };
        result.canonicalize();
        result
    }

    /// `self * k` for a scalar constant, preserving canonical form.
    #[must_use]
    pub fn mul_by_constant(&self, k: i64) -> Self {
        if k == 0 {
            return Self::zero();
        }
        let coeffs = self
            .coeffs
            .iter()
            .map(|(s, &c)| (s.clone(), c * k))
            .collect();
        Self {
            coeffs,
            constant: self.constant * k,
        }
        // No need to canonicalize: if k != 0, nonzero coeffs stay nonzero.
    }

    /// Negate the entire expression: `-self`.
    #[must_use]
    pub fn negate(&self) -> Self {
        self.mul_by_constant(-1)
    }

    /// Substitute concrete values for some or all symbols.
    ///
    /// Symbols not present in `env` are left symbolic.
    #[must_use]
    pub fn substitute(&self, env: &HashMap<String, i64>) -> Self {
        let mut constant = self.constant;
        let mut coeffs = BTreeMap::new();

        for (sym, &c) in &self.coeffs {
            if let Some(&val) = env.get(sym) {
                constant += c * val;
            } else {
                coeffs.insert(sym.clone(), c);
            }
        }

        Self { coeffs, constant }
    }

    /// Compute the GCD of all coefficients and the constant.
    ///
    /// Returns 1 if the expression is zero.
    #[must_use]
    pub fn content_gcd(&self) -> u64 {
        let mut g: u64 = self.constant.unsigned_abs();
        for &c in self.coeffs.values() {
            g = gcd(g, c.unsigned_abs());
        }
        if g == 0 { 1 } else { g }
    }

    /// Normalize an equality expression:
    ///
    /// 1. Divide by the GCD of all terms.
    /// 2. Ensure the first nonzero symbol coefficient is positive.
    #[must_use]
    pub fn normalize_eq(&self) -> Self {
        if self.is_constant() && self.constant == 0 {
            return Self::zero();
        }

        let g = self.content_gcd();
        let mut expr = if g > 1 {
            self.div_by_positive(g)
        } else {
            self.clone()
        };

        // Sign convention: first nonzero symbol coeff must be positive.
        let needs_flip = expr.coeffs.values().next().is_some_and(|&c| c < 0);

        if needs_flip {
            expr = expr.negate();
        } else if expr.coeffs.is_empty() && expr.constant < 0 {
            // Pure constant: make nonnegative.
            expr.constant = -expr.constant;
        }

        expr
    }

    /// Normalize an inequality expression (only divide by positive GCD,
    /// never flip sign — flipping an inequality reverses its direction).
    #[must_use]
    pub fn normalize_ineq(&self) -> Self {
        let g = self.content_gcd();
        if g > 1 {
            self.div_by_positive(g)
        } else {
            self.clone()
        }
    }
}

// ── Internal helpers ─────────────────────────────────────────────────

impl AffineExpr {
    /// Remove zero-coefficient entries from the map.
    fn canonicalize(&mut self) {
        self.coeffs.retain(|_, c| *c != 0);
    }

    /// Divide all coefficients and constant by a positive divisor.
    ///
    /// This performs integer (truncating) division.
    fn div_by_positive(&self, d: u64) -> Self {
        #[allow(clippy::cast_possible_wrap)]
        let d_i64 = d as i64; // safe: GCD of i64 values fits in i64
        let coeffs = self
            .coeffs
            .iter()
            .map(|(s, &c)| (s.clone(), c / d_i64))
            .collect();
        Self {
            coeffs,
            constant: self.constant / d_i64,
        }
    }
}

// ── Operator overloads ───────────────────────────────────────────────

impl ops::Add for AffineExpr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::add(&self, &rhs)
    }
}

impl ops::Add<&Self> for AffineExpr {
    type Output = Self;
    fn add(self, rhs: &Self) -> Self {
        Self::add(&self, rhs)
    }
}

impl ops::Sub for AffineExpr {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::sub(&self, &rhs)
    }
}

impl ops::Sub<&Self> for AffineExpr {
    type Output = Self;
    fn sub(self, rhs: &Self) -> Self {
        Self::sub(&self, rhs)
    }
}

impl ops::Neg for AffineExpr {
    type Output = Self;
    fn neg(self) -> Self {
        self.negate()
    }
}

impl ops::Mul<i64> for AffineExpr {
    type Output = Self;
    fn mul(self, rhs: i64) -> Self {
        self.mul_by_constant(rhs)
    }
}

impl ops::Mul<AffineExpr> for i64 {
    type Output = AffineExpr;
    fn mul(self, rhs: AffineExpr) -> AffineExpr {
        rhs.mul_by_constant(self)
    }
}

// ── Display ──────────────────────────────────────────────────────────

impl fmt::Display for AffineExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_constant() && self.constant == 0 {
            return write!(f, "0");
        }

        let mut first = true;
        for (sym, &c) in &self.coeffs {
            if first {
                match c {
                    1 => write!(f, "{sym}")?,
                    -1 => write!(f, "-{sym}")?,
                    _ => write!(f, "{c}·{sym}")?,
                }
                first = false;
            } else if c == 1 {
                write!(f, " + {sym}")?;
            } else if c == -1 {
                write!(f, " - {sym}")?;
            } else if c > 0 {
                write!(f, " + {c}·{sym}")?;
            } else {
                write!(f, " - {}·{sym}", -c)?;
            }
        }

        if self.constant > 0 {
            if first {
                write!(f, "{}", self.constant)?;
            } else {
                write!(f, " + {}", self.constant)?;
            }
        } else if self.constant < 0 {
            if first {
                write!(f, "{}", self.constant)?;
            } else {
                write!(f, " - {}", -self.constant)?;
            }
        }

        Ok(())
    }
}

// ── Free functions ───────────────────────────────────────────────────

/// Greatest common divisor (Euclidean algorithm).
#[must_use]
pub(crate) fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_constant() {
        let z = AffineExpr::zero();
        assert!(z.is_constant());
        assert_eq!(z.evaluate_if_constant(), Some(0));
        assert_eq!(z.to_string(), "0");
    }

    #[test]
    fn symbol_basics() {
        let n = AffineExpr::symbol("N");
        assert!(!n.is_constant());
        assert_eq!(n.coeff("N"), 1);
        assert_eq!(n.constant_term(), 0);
        assert_eq!(n.symbols(), vec!["N"]);
        assert_eq!(n.to_string(), "N");
    }

    #[test]
    fn constant_basics() {
        let c = AffineExpr::constant(42);
        assert!(c.is_constant());
        assert_eq!(c.evaluate_if_constant(), Some(42));
        assert_eq!(c.to_string(), "42");
    }

    #[test]
    fn add_combines_terms() {
        let a = AffineExpr::symbol("B")
            .mul_by_constant(2)
            .add(&AffineExpr::constant(3));
        let b = AffineExpr::symbol("C").mul_by_constant(3);
        let sum = a.add(&b);
        assert_eq!(sum.coeff("B"), 2);
        assert_eq!(sum.coeff("C"), 3);
        assert_eq!(sum.constant_term(), 3);
    }

    #[test]
    fn canonical_ordering_equivalence() {
        // 2B + 3C must equal 3C + 2B
        let lhs =
            AffineExpr::symbol("B").mul_by_constant(2) + AffineExpr::symbol("C").mul_by_constant(3);
        let rhs =
            AffineExpr::symbol("C").mul_by_constant(3) + AffineExpr::symbol("B").mul_by_constant(2);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn zero_coefficients_removed() {
        // B + B + 3 - 3 = 2B, and B - B = 0
        let expr = AffineExpr::symbol("B") + AffineExpr::symbol("B") + AffineExpr::constant(3)
            - AffineExpr::constant(3);
        assert_eq!(expr.coeff("B"), 2);
        assert_eq!(expr.constant_term(), 0);
        assert_eq!(expr.num_symbols(), 1);

        let zero = AffineExpr::symbol("B") - AffineExpr::symbol("B");
        assert!(zero.is_constant());
        assert_eq!(zero.evaluate_if_constant(), Some(0));
    }

    #[test]
    fn sub_works() {
        let a = AffineExpr::symbol("N").add(&AffineExpr::constant(10));
        let b = AffineExpr::constant(3);
        let diff = a.sub(&b);
        assert_eq!(diff.coeff("N"), 1);
        assert_eq!(diff.constant_term(), 7);
    }

    #[test]
    fn mul_by_zero_yields_zero() {
        let expr = AffineExpr::symbol("X").add(&AffineExpr::constant(5));
        let z = expr.mul_by_constant(0);
        assert!(z.is_constant());
        assert_eq!(z.evaluate_if_constant(), Some(0));
    }

    #[test]
    fn negate() {
        let expr = AffineExpr::symbol("A").mul_by_constant(3) + AffineExpr::constant(-2);
        let neg = expr.negate();
        assert_eq!(neg.coeff("A"), -3);
        assert_eq!(neg.constant_term(), 2);
    }

    #[test]
    fn substitute_partial() {
        let expr = AffineExpr::symbol("N").mul_by_constant(2)
            + AffineExpr::symbol("M").mul_by_constant(3)
            + AffineExpr::constant(5);
        let env: HashMap<String, i64> = [("N".to_owned(), 4)].into_iter().collect();
        let result = expr.substitute(&env);
        assert_eq!(result.coeff("N"), 0);
        assert_eq!(result.coeff("M"), 3);
        assert_eq!(result.constant_term(), 13); // 2*4 + 5
    }

    #[test]
    fn substitute_full() {
        let expr = AffineExpr::symbol("X") + AffineExpr::constant(1);
        let env: HashMap<String, i64> = [("X".to_owned(), 9)].into_iter().collect();
        let result = expr.substitute(&env);
        assert_eq!(result.evaluate_if_constant(), Some(10));
    }

    #[test]
    fn content_gcd_computation() {
        // 6N + 4 → gcd = 2
        let expr = AffineExpr::symbol("N").mul_by_constant(6) + AffineExpr::constant(4);
        assert_eq!(expr.content_gcd(), 2);

        // N + 1 → gcd = 1
        let expr2 = AffineExpr::symbol("N") + AffineExpr::constant(1);
        assert_eq!(expr2.content_gcd(), 1);
    }

    #[test]
    fn normalize_eq_divides_by_gcd() {
        // 2B - 4 → B - 2
        let expr = AffineExpr::symbol("B").mul_by_constant(2) + AffineExpr::constant(-4);
        let norm = expr.normalize_eq();
        assert_eq!(norm.coeff("B"), 1);
        assert_eq!(norm.constant_term(), -2);
    }

    #[test]
    fn normalize_eq_sign_convention() {
        // -2B + 4 → B - 2  (flip so first coeff is positive)
        let expr = AffineExpr::symbol("B").mul_by_constant(-2) + AffineExpr::constant(4);
        let norm = expr.normalize_eq();
        assert_eq!(norm.coeff("B"), 1);
        assert_eq!(norm.constant_term(), -2);
    }

    #[test]
    fn normalize_eq_equivalence() {
        let a = AffineExpr::symbol("B").mul_by_constant(2) + AffineExpr::constant(-4);
        let b = AffineExpr::symbol("B").mul_by_constant(-2) + AffineExpr::constant(4);
        assert_eq!(a.normalize_eq(), b.normalize_eq());
    }

    #[test]
    fn normalize_ineq_positive_gcd_only() {
        // 6N + 4 >= 0  →  3N + 2 >= 0  (divide by gcd=2, no sign flip)
        let expr = AffineExpr::symbol("N").mul_by_constant(6) + AffineExpr::constant(4);
        let norm = expr.normalize_ineq();
        assert_eq!(norm.coeff("N"), 3);
        assert_eq!(norm.constant_term(), 2);
    }

    #[test]
    fn operator_overloads() {
        let sym_x = AffineExpr::symbol("X");
        let five = AffineExpr::constant(5);
        let sum = sym_x.clone() + five.clone();
        assert_eq!(sum.coeff("X"), 1);
        assert_eq!(sum.constant_term(), 5);

        let diff = sym_x.clone() - five;
        assert_eq!(diff.constant_term(), -5);

        let scaled = sym_x * 3;
        assert_eq!(scaled.coeff("X"), 3);

        let lhs_scalar = 2 * AffineExpr::symbol("Y");
        assert_eq!(lhs_scalar.coeff("Y"), 2);

        let negated = -AffineExpr::symbol("Z");
        assert_eq!(negated.coeff("Z"), -1);
    }

    #[test]
    fn display_formatting() {
        let expr = AffineExpr::symbol("A").mul_by_constant(2)
            + AffineExpr::symbol("B").mul_by_constant(-1)
            + AffineExpr::constant(3);
        assert_eq!(expr.to_string(), "2·A - B + 3");

        let neg_const = AffineExpr::symbol("X") + AffineExpr::constant(-7);
        assert_eq!(neg_const.to_string(), "X - 7");

        let neg_only = AffineExpr::constant(-5);
        assert_eq!(neg_only.to_string(), "-5");
    }

    #[test]
    fn from_parts_strips_zeros() {
        let mut coeffs = BTreeMap::new();
        coeffs.insert("A".to_owned(), 3);
        coeffs.insert("B".to_owned(), 0);
        coeffs.insert("C".to_owned(), -1);
        let expr = AffineExpr::from_parts(coeffs, 5);
        assert_eq!(expr.num_symbols(), 2);
        assert_eq!(expr.coeff("B"), 0);
    }

    #[test]
    fn gcd_edge_cases() {
        assert_eq!(gcd(0, 0), 0);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(7, 0), 7);
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(17, 13), 1);
    }
}
