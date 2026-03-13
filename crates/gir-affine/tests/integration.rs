//! Integration tests — realistic tensor dimension scenarios.

use gir_affine::prelude::*;

// ── Broadcast constraints ────────────────────────────────────────────

/// When broadcasting `[N, C, H, W]` with `[1, C, 1, 1]`, the C
/// dimensions must be equal.
#[test]
fn broadcast_channel_equality() {
    let mut sys = ConstraintSystem::new();

    // C_input == C_bias  →  constraint emitted by broadcast_shapes
    sys.add(Constraint::eq(
        &AffineExpr::symbol("C_in"),
        &AffineExpr::symbol("C_bias"),
    ));
    // C_in is known to be 64 from the weight shape
    sys.add(Constraint::eq(
        &AffineExpr::symbol("C_in"),
        &AffineExpr::constant(64),
    ));

    let result = sys.solve().unwrap();
    assert_eq!(result.solved.get("C_in"), Some(&64));
    assert_eq!(result.solved.get("C_bias"), Some(&64));
    assert!(result.unresolved.is_empty());
}

/// Broadcasting incompatible shapes should be detected.
#[test]
fn broadcast_contradiction() {
    let mut sys = ConstraintSystem::new();

    // Two channels that must be equal but are given conflicting values.
    sys.add(Constraint::eq(
        &AffineExpr::symbol("C"),
        &AffineExpr::constant(64),
    ));
    sys.add(Constraint::eq(
        &AffineExpr::symbol("C"),
        &AffineExpr::constant(128),
    ));

    assert!(sys.solve().is_err());
}

// ── MatMul inner-dimension constraint ────────────────────────────────

/// `MatMul(A[M, K], B[K, N])` requires the K dims to match.
#[test]
fn matmul_inner_dim() {
    let mut sys = ConstraintSystem::new();

    // K_a == K_b  (emitted by infer_matmul)
    sys.add(Constraint::eq(
        &AffineExpr::symbol("K_a"),
        &AffineExpr::symbol("K_b"),
    ));
    // K_a is known (e.g. from weight shape)
    sys.add(Constraint::eq(
        &AffineExpr::symbol("K_a"),
        &AffineExpr::constant(512),
    ));

    let result = sys.solve().unwrap();
    assert_eq!(result.solved.get("K_a"), Some(&512));
    assert_eq!(result.solved.get("K_b"), Some(&512));
}

// ── Reshape constraints ──────────────────────────────────────────────

/// Reshape from `[N, 64, 7, 7]` to `[N, -1]` means the flat dim = 3136.
/// The total element count must be preserved.
#[test]
fn reshape_element_count() {
    let mut sys = ConstraintSystem::new();

    // N * 64 * 7 * 7 == N * flat_dim
    // Simplifies to: 3136 - flat_dim == 0
    sys.add(Constraint::eq(
        &AffineExpr::constant(3136),
        &AffineExpr::symbol("flat"),
    ));

    let result = sys.solve().unwrap();
    assert_eq!(result.solved.get("flat"), Some(&3136));
}

// ── Conv2d spatial dimension constraints ─────────────────────────────

/// The spatial output of a conv must be positive.
#[test]
fn conv_output_positive() {
    let mut sys = ConstraintSystem::new();

    // H_out = (H_in - 3) / 1 + 1 = H_in - 2
    // Constraint: H_out > 0  →  H_in - 2 > 0
    sys.add(Constraint::gt(
        &(AffineExpr::symbol("H_in") - AffineExpr::constant(2)),
        &AffineExpr::zero(),
    ));
    // H_in = 32
    sys.add(Constraint::eq(
        &AffineExpr::symbol("H_in"),
        &AffineExpr::constant(32),
    ));

    let result = sys.solve().unwrap();
    assert_eq!(result.solved.get("H_in"), Some(&32));
    // After substitution: 32 - 2 > 0 → 30 > 0 → trivially true
    assert!(result.unresolved.is_empty());
}

/// Conv output must be positive — should fail for too-small input.
#[test]
fn conv_output_too_small() {
    let mut sys = ConstraintSystem::new();

    // H_out = H_in - 6  (e.g. 7×7 kernel, valid padding)
    // H_out > 0  →  H_in - 6 > 0
    sys.add(Constraint::gt(
        &(AffineExpr::symbol("H_in") - AffineExpr::constant(6)),
        &AffineExpr::zero(),
    ));
    // But H_in = 3 (too small!)
    sys.add(Constraint::eq(
        &AffineExpr::symbol("H_in"),
        &AffineExpr::constant(3),
    ));

    // 3 - 6 = -3 > 0 → contradiction
    assert!(sys.solve().is_err());
}

// ── Divisibility: quantisation alignment ─────────────────────────────

/// Channels must be divisible by the SIMD width for quantised kernels.
#[test]
fn quantisation_alignment() {
    let mut sys = ConstraintSystem::new();

    // C % 8 == 0
    sys.add(Constraint::divisible(AffineExpr::symbol("C"), 8));
    // C = 64
    sys.add(Constraint::eq(
        &AffineExpr::symbol("C"),
        &AffineExpr::constant(64),
    ));

    let result = sys.solve().unwrap();
    assert_eq!(result.solved.get("C"), Some(&64));
    assert!(result.unresolved.is_empty());
}

#[test]
fn quantisation_alignment_fails() {
    let mut sys = ConstraintSystem::new();

    // C % 8 == 0, C = 65
    sys.add(Constraint::divisible(AffineExpr::symbol("C"), 8));
    sys.add(Constraint::eq(
        &AffineExpr::symbol("C"),
        &AffineExpr::constant(65),
    ));

    assert!(sys.solve().is_err());
}

// ── Multi-layer chain ────────────────────────────────────────────────

/// Simulate a small pipeline: Conv → FC → Softmax.
/// Batch dimension N flows through; spatial dims are computed.
#[test]
fn multi_layer_pipeline() {
    let mut sys = ConstraintSystem::new();

    // Input: [N, 1, 28, 28]
    // Conv: kernel 5×5, valid → output H = 28 - 4 = 24, W = 24
    // Pool 2×2 → H = 12, W = 12
    // Flatten → N × (C_out * 12 * 12)
    // FC → need flat_dim == weight_in

    // C_out = 16
    sys.add(Constraint::eq(
        &AffineExpr::symbol("C_out"),
        &AffineExpr::constant(16),
    ));

    // flat_dim = C_out * 12 * 12 = C_out * 144
    // We'd store this as: flat_dim - 144 * C_out == 0
    //   → after solving C_out = 16: flat_dim - 2304 == 0  → flat_dim = 2304

    // For a pure affine system we express this linearly:
    // flat_dim == 2304  (pre-computed from the known spatial dims)
    sys.add(Constraint::eq(
        &AffineExpr::symbol("flat_dim"),
        &AffineExpr::constant(2304),
    ));

    // FC weight shape: [10, flat_dim]
    // Constraint: weight_in == flat_dim
    sys.add(Constraint::eq(
        &AffineExpr::symbol("weight_in"),
        &AffineExpr::symbol("flat_dim"),
    ));

    let result = sys.solve().unwrap();
    assert_eq!(result.solved.get("C_out"), Some(&16));
    assert_eq!(result.solved.get("flat_dim"), Some(&2304));
    assert_eq!(result.solved.get("weight_in"), Some(&2304));
    assert!(result.unresolved.is_empty());
}

// ── Symbolic batch left unresolved ───────────────────────────────────

/// When the batch dimension is purely symbolic and not constrained to a
/// concrete value, it should remain unresolved.
#[test]
fn symbolic_batch_unresolved() {
    let mut sys = ConstraintSystem::new();

    // N_input == N_output  (from shape propagation)
    sys.add(Constraint::eq(
        &AffineExpr::symbol("N_in"),
        &AffineExpr::symbol("N_out"),
    ));
    // N >= 1  (positive batch size)
    sys.add(Constraint::ge(
        &AffineExpr::symbol("N_in"),
        &AffineExpr::constant(1),
    ));

    let result = sys.solve().unwrap();
    // No concrete solutions — both still symbolic.
    assert!(result.solved.is_empty());
    // But we should have unresolved constraints.
    assert!(!result.unresolved.is_empty());
}

// ── Substitute API ───────────────────────────────────────────────────

#[test]
fn partial_substitution() {
    let expr = AffineExpr::symbol("N").mul_by_constant(2)
        + AffineExpr::symbol("M")
        + AffineExpr::constant(3);

    // Only bind N.
    let env = [("N".to_owned(), 5)].into_iter().collect();
    let result = expr.substitute(&env);

    assert_eq!(result.coeff("M"), 1);
    assert_eq!(result.constant_term(), 13); // 2*5 + 3
    assert!(!result.is_constant());

    // Now bind M too.
    let env2 = [("M".to_owned(), 7)].into_iter().collect();
    let final_result = result.substitute(&env2);
    assert_eq!(final_result.evaluate_if_constant(), Some(20)); // 13 + 7
}

// ── Edge cases ───────────────────────────────────────────────────────

#[test]
fn empty_system() {
    let mut sys = ConstraintSystem::new();
    let result = sys.solve().unwrap();
    assert!(result.solved.is_empty());
    assert!(result.unresolved.is_empty());
}

#[test]
fn duplicate_constraint_is_harmless() {
    let mut sys = ConstraintSystem::new();
    // Same constraint twice.
    sys.add(Constraint::eq(
        &AffineExpr::symbol("N"),
        &AffineExpr::constant(4),
    ));
    sys.add(Constraint::eq(
        &AffineExpr::symbol("N"),
        &AffineExpr::constant(4),
    ));
    let result = sys.solve().unwrap();
    assert_eq!(result.solved.get("N"), Some(&4));
    assert!(result.unresolved.is_empty());
}

#[test]
fn large_coefficient_chain() {
    let mut sys = ConstraintSystem::new();
    // 6*A == 42  → after norm (gcd=6): A - 7 == 0  →  A = 7
    sys.add(Constraint::eq(
        &AffineExpr::symbol("A").mul_by_constant(6),
        &AffineExpr::constant(42),
    ));
    // B == A + 3  → B - A - 3 == 0  →  after sub A=7: B - 10 == 0 → B = 10
    sys.add(Constraint::eq(
        &AffineExpr::symbol("B"),
        &(AffineExpr::symbol("A") + AffineExpr::constant(3)),
    ));

    let result = sys.solve().unwrap();
    assert_eq!(result.solved.get("A"), Some(&7));
    assert_eq!(result.solved.get("B"), Some(&10));
}
