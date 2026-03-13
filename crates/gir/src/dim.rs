//! Symbolic dimension expressions for static and semi-dynamic shape inference.
//!
//! Dimensions in the IR are represented as [`DimExpr`] — an affine expression
//! that can be a concrete integer, a named symbolic variable (e.g. batch size `N`),
//! or a linear combination of those.  This allows the IR to propagate shapes
//! through the graph even when some input dimensions are only known at runtime.
//!
//! Internally, `DimExpr` wraps an [`AffineExpr`](weaver_affine::expr::AffineExpr)
//! numerator and a constant divisor, giving you canonical normalisation,
//! deterministic equality, and direct interop with the constraint solver in
//! `weaver-affine`.
//!
//! # Examples
//!
//! ```
//! use gir::dim::DimExpr;
//!
//! let n = DimExpr::sym("N");
//! let fixed = DimExpr::fixed(32);
//! let expr = n + fixed; // N + 32
//! assert_eq!(expr.evaluate(&[("N", 4)].into_iter().collect()), Some(36));
//! ```

use std::collections::HashMap;
use std::fmt;
use std::ops;

use serde::Serialize;
use gir_affine::expr::AffineExpr;

// ── Core type ────────────────────────────────────────────────────────

/// An affine expression over dimension variables, optionally divided by a
/// constant.
///
/// Semantically the value is `floor(numerator / divisor)` where `numerator`
/// is an [`AffineExpr`] (`a₁·x₁ + a₂·x₂ + … + c`) and `divisor` is a
/// positive integer (default 1, meaning no division).
///
/// This representation handles the standard spatial output formulas:
///
/// * **Same padding**: `ceil(input / stride) = floor((input + stride - 1) / stride)`
/// * **Valid padding**: `floor((input + pad - ek) / stride) + 1`
///
/// Arithmetic automatically normalises the expression so structurally
/// identical expressions compare equal.  When both the numerator and
/// divisor are concrete the result is eagerly folded to a constant.
#[derive(Clone, PartialEq, Eq, Hash, Serialize)]
pub struct DimExpr {
    /// The affine numerator.
    num: AffineExpr,
    /// Positive constant divisor (≥ 1).
    div: u64,
}

impl DimExpr {
    // ── Constructors ─────────────────────────────────────────────────

    /// Create a fixed (concrete) dimension.
    #[must_use]
    pub fn fixed(value: u64) -> Self {
        #[allow(clippy::cast_possible_wrap)]
        Self {
            num: AffineExpr::constant(value as i64),
            div: 1,
        }
    }

    /// Create a named symbolic dimension.
    #[must_use]
    pub fn sym(name: &str) -> Self {
        Self {
            num: AffineExpr::symbol(name),
            div: 1,
        }
    }

    /// Wrap a raw [`AffineExpr`] (divisor = 1).
    #[must_use]
    pub fn from_affine(expr: AffineExpr) -> Self {
        Self { num: expr, div: 1 }
    }

    /// Build from numerator and divisor parts.
    #[must_use]
    fn from_parts(num: AffineExpr, div: u64) -> Self {
        assert!(div > 0, "DimExpr: divisor must be positive");
        let mut expr = Self { num, div };
        expr.try_reduce();
        expr
    }

    /// Ceiling division by a **constant** divisor.
    ///
    /// `ceil(a / b) = floor((a + b - 1) / b)`
    ///
    /// The divisor (`rhs`) must be a concrete positive integer.
    #[must_use]
    pub fn ceil_div(self, rhs: Self) -> Self {
        let b = rhs.try_fixed().expect(
            "DimExpr::ceil_div requires a constant divisor (stride/dilation is always concrete)",
        );
        assert!(b > 0, "DimExpr::ceil_div: division by zero");
        if b == 1 {
            return self;
        }
        // ceil(a / b) = floor((a + b - 1) / b)
        let shifted = self + DimExpr::fixed(b - 1);
        shifted.floor_div_const(b)
    }

    /// Floor division by a known positive constant.
    #[must_use]
    pub fn floor_div_const(self, d: u64) -> Self {
        assert!(d > 0, "DimExpr::floor_div_const: division by zero");
        if d == 1 {
            return self;
        }
        Self::from_parts(self.num, self.div * d)
    }

    // ── Internal normalisation ───────────────────────────────────────

    /// If the numerator is a constant, eagerly fold the division.
    fn try_reduce(&mut self) {
        if self.div == 1 {
            return;
        }
        if let Some(n) = self.num.evaluate_if_constant() {
            #[allow(clippy::cast_possible_wrap)]
            let d = self.div as i64;
            // Floor division that rounds towards negative infinity for
            // consistency.  In practice dimensions are non-negative.
            let result = n.div_euclid(d);
            self.num = AffineExpr::constant(result);
            self.div = 1;
        }
    }

    // ── Queries ──────────────────────────────────────────────────────

    /// Returns `true` when the expression contains no symbolic variables.
    #[must_use]
    pub fn is_static(&self) -> bool {
        self.num.is_constant()
    }

    /// Attempt to evaluate the expression to a concrete `u64`, given a mapping
    /// of symbolic names to values.
    ///
    /// Returns `None` if any referenced symbol is missing from `env`.
    #[must_use]
    pub fn evaluate(&self, env: &HashMap<&str, u64>) -> Option<u64> {
        let i64_env: HashMap<String, i64> = env
            .iter()
            .map(|(&k, &v)| {
                #[allow(clippy::cast_possible_wrap)]
                (k.to_owned(), v as i64)
            })
            .collect();
        let result = self.num.substitute(&i64_env);
        let numer = result.evaluate_if_constant()?;
        #[allow(clippy::cast_possible_wrap)]
        let d = self.div as i64;
        let val = numer.div_euclid(d);
        // Clamp negative to 0 (saturating semantics).
        #[allow(clippy::cast_sign_loss)]
        Some(val.max(0) as u64)
    }

    /// Attempt to reduce the expression to a fixed value without any environment.
    #[must_use]
    pub fn try_fixed(&self) -> Option<u64> {
        let numer = self.num.evaluate_if_constant()?;
        #[allow(clippy::cast_possible_wrap)]
        let d = self.div as i64;
        let val = numer.div_euclid(d);
        #[allow(clippy::cast_sign_loss)]
        Some(val.max(0) as u64)
    }

    /// Collect the set of symbolic names referenced in this expression.
    #[must_use]
    pub fn free_symbols(&self) -> Vec<String> {
        self.num.symbols().iter().map(|s| (*s).to_owned()).collect()
    }

    /// Substitute every occurrence of `name` with `replacement`.
    ///
    /// The replacement must have `divisor == 1` (i.e. a plain affine expression
    /// or constant).  This is always the case for concrete dimension bindings.
    #[must_use]
    pub fn substitute(&self, name: &str, replacement: &Self) -> Self {
        assert_eq!(
            replacement.div, 1,
            "DimExpr::substitute: replacement must have divisor 1"
        );
        let coeff = self.num.coeff(name);
        if coeff == 0 {
            return self.clone();
        }
        let without = self
            .num
            .sub(&AffineExpr::symbol(name).mul_by_constant(coeff));
        let added = without.add(&replacement.num.mul_by_constant(coeff));
        Self::from_parts(added, self.div)
    }

    /// Access the numerator as an [`AffineExpr`] for direct use with the
    /// constraint solver.
    #[must_use]
    pub fn as_affine(&self) -> &AffineExpr {
        &self.num
    }

    /// Consume and return the numerator [`AffineExpr`].
    ///
    /// Panics if `divisor != 1`.
    #[must_use]
    pub fn into_affine(self) -> AffineExpr {
        assert_eq!(
            self.div, 1,
            "DimExpr::into_affine: cannot convert divided expression"
        );
        self.num
    }

    /// The constant divisor (1 when no division is pending).
    #[must_use]
    pub fn divisor(&self) -> u64 {
        self.div
    }
}

// ── Operator overloads ───────────────────────────────────────────────

/// Helper: bring two `DimExpr`s to a common divisor before performing
/// additive arithmetic.  Returns `(lhs_num', rhs_num', common_div)` such
/// that `lhs_num' / common_div == lhs` and `rhs_num' / common_div == rhs`.
fn common_divisor(lhs: &DimExpr, rhs: &DimExpr) -> (AffineExpr, AffineExpr, u64) {
    if lhs.div == rhs.div {
        (lhs.num.clone(), rhs.num.clone(), lhs.div)
    } else {
        // Cross-multiply to a common denominator.
        #[allow(clippy::cast_possible_wrap)]
        let ld = lhs.div as i64;
        #[allow(clippy::cast_possible_wrap)]
        let rd = rhs.div as i64;
        let ln = lhs.num.mul_by_constant(rd);
        let rn = rhs.num.mul_by_constant(ld);
        (ln, rn, lhs.div * rhs.div)
    }
}

impl ops::Add for DimExpr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let (ln, rn, d) = common_divisor(&self, &rhs);
        Self::from_parts(ln.add(&rn), d)
    }
}

impl ops::Add for &DimExpr {
    type Output = DimExpr;
    fn add(self, rhs: Self) -> DimExpr {
        let (ln, rn, d) = common_divisor(self, rhs);
        DimExpr::from_parts(ln.add(&rn), d)
    }
}

impl ops::Sub for DimExpr {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let (ln, rn, d) = common_divisor(&self, &rhs);
        Self::from_parts(ln.sub(&rn), d)
    }
}

impl ops::Sub for &DimExpr {
    type Output = DimExpr;
    fn sub(self, rhs: Self) -> DimExpr {
        let (ln, rn, d) = common_divisor(self, rhs);
        DimExpr::from_parts(ln.sub(&rn), d)
    }
}

impl ops::Mul for DimExpr {
    type Output = Self;
    /// Multiply two `DimExpr`s.
    ///
    /// At least one operand **must** be a constant.  If both are symbolic
    /// this panics — true symbolic × symbolic is non-affine.
    fn mul(self, rhs: Self) -> Self {
        // Try constant on right.
        if let Some(k) = rhs.try_fixed() {
            #[allow(clippy::cast_possible_wrap)]
            return Self::from_parts(self.num.mul_by_constant(k as i64), self.div);
        }
        // Try constant on left.
        if let Some(k) = self.try_fixed() {
            #[allow(clippy::cast_possible_wrap)]
            return Self::from_parts(rhs.num.mul_by_constant(k as i64), rhs.div);
        }
        panic!(
            "DimExpr::mul requires at least one constant operand, got: ({}) * ({})",
            self, rhs
        );
    }
}

impl ops::Mul for &DimExpr {
    type Output = DimExpr;
    fn mul(self, rhs: Self) -> DimExpr {
        self.clone() * rhs.clone()
    }
}

impl ops::Div for DimExpr {
    type Output = Self;
    /// Integer (floor) division.  The divisor must be a constant.
    fn div(self, rhs: Self) -> Self {
        let d = rhs.try_fixed().expect(
            "DimExpr::div requires a constant divisor (stride/dilation is always concrete)",
        );
        self.floor_div_const(d)
    }
}

impl ops::Div for &DimExpr {
    type Output = DimExpr;
    fn div(self, rhs: Self) -> DimExpr {
        self.clone() / rhs.clone()
    }
}

// ── Display / Debug ──────────────────────────────────────────────────

impl fmt::Display for DimExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.div == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "({}) / {}", self.num, self.div)
        }
    }
}

impl fmt::Debug for DimExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DimExpr({self})")
    }
}

// ── From conversions ─────────────────────────────────────────────────

impl From<AffineExpr> for DimExpr {
    fn from(expr: AffineExpr) -> Self {
        Self { num: expr, div: 1 }
    }
}

impl From<DimExpr> for AffineExpr {
    fn from(dim: DimExpr) -> Self {
        assert_eq!(
            dim.div, 1,
            "cannot convert DimExpr with pending division to AffineExpr"
        );
        dim.num
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_evaluates() {
        let d = DimExpr::fixed(42);
        assert_eq!(d.try_fixed(), Some(42));
        assert!(d.is_static());
    }

    #[test]
    fn symbolic_needs_env() {
        let n = DimExpr::sym("N");
        assert_eq!(n.try_fixed(), None);
        assert!(!n.is_static());

        let env: HashMap<&str, u64> = [("N", 8)].into_iter().collect();
        assert_eq!(n.evaluate(&env), Some(8));
    }

    #[test]
    fn arithmetic_propagation() {
        let n = DimExpr::sym("N");
        let expr = &n * &DimExpr::fixed(32);
        let env: HashMap<&str, u64> = [("N", 4)].into_iter().collect();
        assert_eq!(expr.evaluate(&env), Some(128));
    }

    #[test]
    fn ceil_div_works() {
        let expr = DimExpr::fixed(7).ceil_div(DimExpr::fixed(2));
        assert_eq!(expr.try_fixed(), Some(4));
    }

    #[test]
    fn ceil_div_exact() {
        let expr = DimExpr::fixed(8).ceil_div(DimExpr::fixed(4));
        assert_eq!(expr.try_fixed(), Some(2));
    }

    #[test]
    fn ceil_div_by_one() {
        let expr = DimExpr::fixed(7).ceil_div(DimExpr::fixed(1));
        assert_eq!(expr.try_fixed(), Some(7));
    }

    #[test]
    fn substitute_works() {
        let n = DimExpr::sym("N");
        let expr = &n + &DimExpr::fixed(1);
        let replaced = expr.substitute("N", &DimExpr::fixed(10));
        assert_eq!(replaced.try_fixed(), Some(11));
    }

    #[test]
    fn free_symbols_collected() {
        let expr = &DimExpr::sym("M") + &DimExpr::fixed(1);
        let combined = &DimExpr::sym("N") + &expr;
        let mut syms = combined.free_symbols();
        syms.sort();
        assert_eq!(syms, vec!["M".to_owned(), "N".to_owned()]);
    }

    #[test]
    fn display_formatting() {
        let n = DimExpr::sym("N");
        let c = DimExpr::fixed(32);
        let expr = n + c;
        assert_eq!(format!("{expr}"), "N + 32");
    }

    #[test]
    fn addition_is_canonical() {
        let expr1 = DimExpr::sym("N") + DimExpr::fixed(3) + DimExpr::fixed(5);
        let expr2 = DimExpr::sym("N") + DimExpr::fixed(8);
        assert_eq!(expr1, expr2);
    }

    #[test]
    fn subtraction_basic() {
        let expr = DimExpr::fixed(10) - DimExpr::fixed(3);
        assert_eq!(expr.try_fixed(), Some(7));
    }

    #[test]
    fn floor_div_const_basic() {
        let expr = DimExpr::fixed(10).floor_div_const(3);
        assert_eq!(expr.try_fixed(), Some(3));
    }

    #[test]
    fn mul_constant_left() {
        let expr = DimExpr::fixed(3) * DimExpr::sym("N");
        let env: HashMap<&str, u64> = [("N", 5)].into_iter().collect();
        assert_eq!(expr.evaluate(&env), Some(15));
    }

    #[test]
    fn mul_constant_right() {
        let expr = DimExpr::sym("N") * DimExpr::fixed(3);
        let env: HashMap<&str, u64> = [("N", 5)].into_iter().collect();
        assert_eq!(expr.evaluate(&env), Some(15));
    }

    #[test]
    fn div_by_constant() {
        let expr = DimExpr::fixed(20) / DimExpr::fixed(3);
        assert_eq!(expr.try_fixed(), Some(6));
    }

    #[test]
    fn affine_roundtrip() {
        let dim = DimExpr::sym("N") + DimExpr::fixed(5);
        let affine = dim.clone().into_affine();
        let back = DimExpr::from_affine(affine);
        assert_eq!(dim, back);
    }

    #[test]
    fn evaluate_saturates_negative() {
        let expr = DimExpr::fixed(3) - DimExpr::fixed(10);
        let env: HashMap<&str, u64> = HashMap::new();
        assert_eq!(expr.evaluate(&env), Some(0));
    }

    #[test]
    fn spatial_output_formula_valid_padding() {
        // out = floor((input + pad - ek) / stride) + 1
        // input=224, kernel=7, stride=2, dilation=1, pad=0
        // ek = 1*(7-1)+1 = 7, numerator = 224 - 7 = 217
        // out = 217/2 + 1 = 108 + 1 = 109
        let input = DimExpr::fixed(224);
        let ek: u64 = 1 * (7 - 1) + 1;
        let numerator = input - DimExpr::fixed(ek);
        let out = numerator / DimExpr::fixed(2) + DimExpr::fixed(1);
        assert_eq!(out.try_fixed(), Some(109));
    }

    #[test]
    fn spatial_output_formula_same_padding() {
        // out = ceil(input / stride) = ceil(224/2) = 112
        let input = DimExpr::fixed(224);
        let out = input.ceil_div(DimExpr::fixed(2));
        assert_eq!(out.try_fixed(), Some(112));
    }

    #[test]
    fn symbolic_spatial_same_padding() {
        // out = ceil(N / 2) = floor((N + 1) / 2)
        let n = DimExpr::sym("N");
        let out = n.ceil_div(DimExpr::fixed(2));
        // N=224 => 225/2 = 112
        let env: HashMap<&str, u64> = [("N", 224)].into_iter().collect();
        assert_eq!(out.evaluate(&env), Some(112));
        // N=225 => 226/2 = 113
        let env2: HashMap<&str, u64> = [("N", 225)].into_iter().collect();
        assert_eq!(out.evaluate(&env2), Some(113));
    }

    #[test]
    fn symbolic_valid_padding_formula() {
        // out = floor((N + 0 - 3) / 1) + 1 = N - 2
        let n = DimExpr::sym("N");
        let ek: u64 = 3;
        let numerator = n - DimExpr::fixed(ek);
        let out = numerator / DimExpr::fixed(1) + DimExpr::fixed(1);
        let env: HashMap<&str, u64> = [("N", 10)].into_iter().collect();
        assert_eq!(out.evaluate(&env), Some(8)); // 10 - 3 + 1 = 8
    }

    #[test]
    fn add_divided_expressions() {
        // floor(N / 2) + 1 should evaluate correctly
        let expr = DimExpr::sym("N").floor_div_const(2) + DimExpr::fixed(1);
        let env: HashMap<&str, u64> = [("N", 10)].into_iter().collect();
        assert_eq!(expr.evaluate(&env), Some(6)); // 10/2 + 1 = 6
    }

    #[test]
    fn mul_reduces_divisor() {
        // (N / 2) * 2 should simplify: numerator is 2*N, div is 2
        // evaluate with N=5 => 2*5 / 2 = 5
        let expr = DimExpr::sym("N").floor_div_const(2) * DimExpr::fixed(2);
        let env: HashMap<&str, u64> = [("N", 5)].into_iter().collect();
        assert_eq!(expr.evaluate(&env), Some(5));
    }
}
