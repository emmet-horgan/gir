//! Partial compile-time constraint solver.
//!
//! The solver applies a simple fixpoint loop:
//!
//! 1. Substitute all currently solved values into every constraint.
//! 2. Detect constant contradictions.
//! 3. Solve single-variable equalities (`a·x + c = 0` where `-c` is
//!    divisible by `a`).
//! 4. Remove trivially satisfied constraints.
//! 5. Repeat until no new solutions are found.
//!
//! Multi-variable constraints are left unresolved and classified as
//! requiring runtime checks.

use std::collections::HashMap;
use std::fmt;

use serde::Serialize;

use crate::constraint::Constraint;

/// Errors that can occur during constraint solving.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolveError {
    /// A constraint was found to be unsatisfiable at compile time.
    Contradiction {
        /// A human-readable description of the violated constraint.
        constraint: String,
    },
}

impl fmt::Display for SolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contradiction { constraint } => {
                write!(f, "constraint contradiction: {constraint}")
            }
        }
    }
}

impl std::error::Error for SolveError {}

/// The result of a solver run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SolveResult {
    /// Symbols that were solved to concrete values.
    pub solved: HashMap<String, i64>,

    /// Constraints that could not be resolved at compile time and must
    /// be checked at runtime.
    pub unresolved: Vec<Constraint>,
}

/// A collection of constraints with a partial solver.
///
/// Add constraints via [`add`](Self::add), then call [`solve`](Self::solve)
/// to run the fixpoint solver.
#[derive(Debug, Clone, Serialize)]
pub struct ConstraintSystem {
    constraints: Vec<Constraint>,
    solved_values: HashMap<String, i64>,
}

impl ConstraintSystem {
    /// Create an empty constraint system.
    #[must_use]
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            solved_values: HashMap::new(),
        }
    }

    /// Add a constraint to the system.
    pub fn add(&mut self, c: Constraint) {
        self.constraints.push(c);
    }

    /// Add multiple constraints.
    pub fn add_all(&mut self, cs: impl IntoIterator<Item = Constraint>) {
        self.constraints.extend(cs);
    }

    /// Retrieve the solved value for a symbol, if known.
    #[must_use]
    pub fn solved_value(&self, sym: &str) -> Option<i64> {
        self.solved_values.get(sym).copied()
    }

    /// All currently solved symbol → value mappings.
    #[must_use]
    pub fn solved_values(&self) -> &HashMap<String, i64> {
        &self.solved_values
    }

    /// The current list of constraints (some may already be satisfied).
    #[must_use]
    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    /// Run the fixpoint solver.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::Contradiction`] if a constraint is found to be
    /// unsatisfiable with the solved values.
    pub fn solve(&mut self) -> Result<SolveResult, SolveError> {
        loop {
            let progress = self.solve_step()?;
            if !progress {
                break;
            }
        }

        // Collect unresolved constraints.
        let unresolved: Vec<Constraint> = self
            .constraints
            .iter()
            .filter(|c| c.is_trivially_satisfied() != Some(true))
            .cloned()
            .collect();

        Ok(SolveResult {
            solved: self.solved_values.clone(),
            unresolved,
        })
    }

    /// A single pass of the solver.  Returns `true` if progress was made.
    fn solve_step(&mut self) -> Result<bool, SolveError> {
        let mut progress = false;

        // 1. Substitute solved values into all constraints.
        if !self.solved_values.is_empty() {
            for c in &mut self.constraints {
                let subst = c.substitute(&self.solved_values);
                if subst != *c {
                    *c = subst;
                    progress = true;
                }
            }
        }

        // 2. Check for contradictions.
        for c in &self.constraints {
            if c.is_contradiction() {
                return Err(SolveError::Contradiction {
                    constraint: c.to_string(),
                });
            }
        }

        // 3. Solve single-variable equalities.
        // TODO: This implementation allows multipe symbols to be solved in this
        // step with each one overwriting the last because we do not check for a
        // contradiction with `new_solutions`
        let mut new_solutions: Vec<(String, i64)> = Vec::new();

        for c in &self.constraints {
            if let Constraint::EqZero(expr) = c {
                if expr.num_symbols() == 1 {
                    // a·x + c = 0  →  x = -c / a
                    let (sym, coeff) = expr.terms().next().expect("one symbol");
                    let constant = expr.constant_term();

                    if coeff != 0 && (-constant) % coeff == 0 {
                        let val = -constant / coeff;
                        let sym_owned = sym.to_owned();

                        // Only record if not already solved (or consistent).
                        if let Some(&existing) = self.solved_values.get(&sym_owned) {
                            if existing != val {
                                return Err(SolveError::Contradiction {
                                    constraint: format!(
                                        "{sym_owned} solved to both {existing} and {val}"
                                    ),
                                });
                            }
                        } else {
                            new_solutions.push((sym_owned, val));
                        }
                    }
                }
            }
        }

        for (sym, val) in new_solutions {
            if self.solved_values.insert(sym, val).is_none() {
                progress = true;
            }
        }

        // 4. Also try to solve single-variable divisibility constraints
        //    that have become constant after substitution.
        //    (Already handled by is_trivially_satisfied / is_contradiction.)

        // 5. Remove trivially satisfied constraints.
        let before = self.constraints.len();
        self.constraints
            .retain(|c| c.is_trivially_satisfied() != Some(true));
        if self.constraints.len() < before {
            progress = true;
        }

        Ok(progress)
    }
}

impl Default for ConstraintSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ConstraintSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ConstraintSystem {{")?;
        if !self.solved_values.is_empty() {
            writeln!(f, "  solved:")?;
            // Sort for deterministic output.
            let mut pairs: Vec<_> = self.solved_values.iter().collect();
            pairs.sort_by_key(|(k, _)| (*k).clone());
            for (sym, val) in pairs {
                writeln!(f, "    {sym} = {val}")?;
            }
        }
        if !self.constraints.is_empty() {
            writeln!(f, "  constraints:")?;
            for c in &self.constraints {
                writeln!(f, "    {c}")?;
            }
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::AffineExpr;

    #[test]
    fn solve_single_variable() {
        let mut sys = ConstraintSystem::new();
        // N - 8 == 0  →  N = 8
        sys.add(Constraint::eq(
            &AffineExpr::symbol("N"),
            &AffineExpr::constant(8),
        ));
        let result = sys.solve().unwrap();
        assert_eq!(result.solved.get("N"), Some(&8));
        assert!(result.unresolved.is_empty());
    }

    #[test]
    fn solve_chain() {
        let mut sys = ConstraintSystem::new();
        // N == 8
        sys.add(Constraint::eq(
            &AffineExpr::symbol("N"),
            &AffineExpr::constant(8),
        ));
        // M == 2*N  →  M - 2N == 0
        sys.add(Constraint::eq(
            &AffineExpr::symbol("M"),
            &AffineExpr::symbol("N").mul_by_constant(2),
        ));

        let result = sys.solve().unwrap();
        assert_eq!(result.solved.get("N"), Some(&8));
        assert_eq!(result.solved.get("M"), Some(&16));
        assert!(result.unresolved.is_empty());
    }

    #[test]
    fn contradiction_detected() {
        let mut sys = ConstraintSystem::new();
        // N == 8  and  N == 5
        sys.add(Constraint::eq(
            &AffineExpr::symbol("N"),
            &AffineExpr::constant(8),
        ));
        sys.add(Constraint::eq(
            &AffineExpr::symbol("N"),
            &AffineExpr::constant(5),
        ));
        let result = sys.solve();
        assert!(result.is_err());
    }

    #[test]
    fn constant_contradiction() {
        let mut sys = ConstraintSystem::new();
        // 5 == 0 is a contradiction
        sys.add(Constraint::EqZero(AffineExpr::constant(5)));
        let result = sys.solve();
        assert!(result.is_err());
    }

    #[test]
    fn inequality_contradiction() {
        let mut sys = ConstraintSystem::new();
        // -3 > 0 is a contradiction
        sys.add(Constraint::GtZero(AffineExpr::constant(-3)));
        let result = sys.solve();
        assert!(result.is_err());
    }

    #[test]
    fn multi_variable_left_unresolved() {
        let mut sys = ConstraintSystem::new();
        // N + M - 10 == 0  →  can't solve without more info
        sys.add(Constraint::eq(
            &(AffineExpr::symbol("N") + AffineExpr::symbol("M")),
            &AffineExpr::constant(10),
        ));
        let result = sys.solve().unwrap();
        assert!(result.solved.is_empty());
        assert_eq!(result.unresolved.len(), 1);
    }

    #[test]
    fn divisibility_satisfied() {
        let mut sys = ConstraintSystem::new();
        // N = 16, then check N % 8 == 0
        sys.add(Constraint::eq(
            &AffineExpr::symbol("N"),
            &AffineExpr::constant(16),
        ));
        sys.add(Constraint::divisible(AffineExpr::symbol("N"), 8));
        let result = sys.solve().unwrap();
        assert_eq!(result.solved.get("N"), Some(&16));
        assert!(result.unresolved.is_empty());
    }

    #[test]
    fn divisibility_violated() {
        let mut sys = ConstraintSystem::new();
        // N = 15, then N % 8 == 0 → contradiction
        sys.add(Constraint::eq(
            &AffineExpr::symbol("N"),
            &AffineExpr::constant(15),
        ));
        sys.add(Constraint::divisible(AffineExpr::symbol("N"), 8));
        let result = sys.solve();
        assert!(result.is_err());
    }

    #[test]
    fn trivially_true_removed() {
        let mut sys = ConstraintSystem::new();
        // 0 == 0 → trivially true, should be removed
        sys.add(Constraint::EqZero(AffineExpr::zero()));
        // 5 >= 0 → trivially true
        sys.add(Constraint::GeZero(AffineExpr::constant(5)));
        let result = sys.solve().unwrap();
        assert!(result.solved.is_empty());
        assert!(result.unresolved.is_empty());
    }

    #[test]
    fn undivisible_single_var_left_unresolved() {
        let mut sys = ConstraintSystem::new();
        // 3*N - 7 == 0  →  N = 7/3 (not integer) → can't solve
        sys.add(Constraint::EqZero(
            AffineExpr::symbol("N").mul_by_constant(3) + AffineExpr::constant(-7),
        ));
        let result = sys.solve().unwrap();
        assert!(result.solved.is_empty());
        assert_eq!(result.unresolved.len(), 1);
    }

    #[test]
    fn solve_with_coefficient() {
        let mut sys = ConstraintSystem::new();
        // 2*N - 10 == 0  → after normalization: N - 5 == 0  →  N = 5
        sys.add(Constraint::eq(
            &AffineExpr::symbol("N").mul_by_constant(2),
            &AffineExpr::constant(10),
        ));
        let result = sys.solve().unwrap();
        assert_eq!(result.solved.get("N"), Some(&5));
    }

    #[test]
    fn ge_with_solved_value() {
        let mut sys = ConstraintSystem::new();
        // N == 4
        sys.add(Constraint::eq(
            &AffineExpr::symbol("N"),
            &AffineExpr::constant(4),
        ));
        // N >= 1  →  after sub: 4 - 1 >= 0 → 3 >= 0 → trivially true
        sys.add(Constraint::ge(
            &AffineExpr::symbol("N"),
            &AffineExpr::constant(1),
        ));
        let result = sys.solve().unwrap();
        assert_eq!(result.solved.get("N"), Some(&4));
        assert!(result.unresolved.is_empty());
    }

    #[test]
    fn display_formatting() {
        let mut sys = ConstraintSystem::new();
        sys.add(Constraint::eq(
            &AffineExpr::symbol("N"),
            &AffineExpr::constant(8),
        ));
        let _ = sys.solve().unwrap();
        let s = sys.to_string();
        assert!(s.contains("N = 8"));
    }

    #[test]
    fn default_is_empty() {
        let sys = ConstraintSystem::default();
        assert!(sys.constraints().is_empty());
        assert!(sys.solved_values().is_empty());
    }

    #[test]
    fn three_variable_chain() {
        let mut sys = ConstraintSystem::new();
        // A = 2
        sys.add(Constraint::eq(
            &AffineExpr::symbol("A"),
            &AffineExpr::constant(2),
        ));
        // B = 3*A  →  B - 3A = 0
        sys.add(Constraint::eq(
            &AffineExpr::symbol("B"),
            &AffineExpr::symbol("A").mul_by_constant(3),
        ));
        // C = A + B  →  C - A - B = 0
        sys.add(Constraint::eq(
            &AffineExpr::symbol("C"),
            &(AffineExpr::symbol("A") + AffineExpr::symbol("B")),
        ));

        let result = sys.solve().unwrap();
        assert_eq!(result.solved.get("A"), Some(&2));
        assert_eq!(result.solved.get("B"), Some(&6));
        assert_eq!(result.solved.get("C"), Some(&8));
        assert!(result.unresolved.is_empty());
    }

    #[test]
    fn negative_solution() {
        let mut sys = ConstraintSystem::new();
        // N + 5 == 0  →  N = -5
        sys.add(Constraint::EqZero(
            AffineExpr::symbol("N") + AffineExpr::constant(5),
        ));
        let result = sys.solve().unwrap();
        assert_eq!(result.solved.get("N"), Some(&-5));
    }

    #[test]
    fn add_all_convenience() {
        let mut sys = ConstraintSystem::new();
        sys.add_all([
            Constraint::eq(&AffineExpr::symbol("X"), &AffineExpr::constant(1)),
            Constraint::eq(&AffineExpr::symbol("Y"), &AffineExpr::constant(2)),
        ]);
        let result = sys.solve().unwrap();
        assert_eq!(result.solved.get("X"), Some(&1));
        assert_eq!(result.solved.get("Y"), Some(&2));
    }
}
