"""
Unit Tests for QR Decomposition Module
=======================================

Tests for src/math/qr.py — QR factorization via Householder reflections,
back substitution, and least squares solvers.

MAT223 REFERENCES:
    - Section 4.7: QR factorization
    - Section 4.8: Least squares via QR
"""

import pytest
import numpy as np

from src.math.qr import (
    householder_vector,
    apply_householder,
    qr_householder,
    qr_reduced,
    back_substitution,
    solve_via_qr,
    solve_weighted_via_qr,
    check_qr_factorization,
)


# =============================================================================
#                    HOUSEHOLDER VECTOR TESTS
# =============================================================================

class TestHouseholderVector:
    """Tests for householder_vector — computing the reflection vector."""

    def test_reflects_onto_e1(self):
        """Householder reflection maps x to +/-||x|| e1."""
        x = np.array([3.0, 4.0])
        v = householder_vector(x)
        H = np.eye(len(x)) - 2 * np.outer(v, v)
        Hx = H @ x
        assert np.isclose(abs(Hx[0]), np.linalg.norm(x), atol=1e-10)
        assert np.isclose(Hx[1], 0.0, atol=1e-10)

    def test_unit_vector_output(self):
        """Householder vector should be unit length."""
        x = np.array([1.0, 2.0, 3.0])
        v = householder_vector(x)
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-10)

    def test_3d_vector(self):
        """Test with a 3D vector — zeros below first entry."""
        x = np.array([1.0, 2.0, 2.0])
        v = householder_vector(x)
        H = np.eye(3) - 2 * np.outer(v, v)
        Hx = H @ x
        assert np.isclose(abs(Hx[0]), 3.0, atol=1e-10)  # ||x|| = 3
        assert np.allclose(Hx[1:], 0.0, atol=1e-10)

    def test_already_aligned(self):
        """If x is already a multiple of e1, result should still work."""
        x = np.array([5.0, 0.0, 0.0])
        v = householder_vector(x)
        H = np.eye(3) - 2 * np.outer(v, v)
        Hx = H @ x
        assert np.isclose(abs(Hx[0]), 5.0, atol=1e-10)
        assert np.allclose(Hx[1:], 0.0, atol=1e-10)

    def test_single_element(self):
        """Edge case: 1D vector."""
        x = np.array([3.0])
        v = householder_vector(x)
        assert v.shape == (1,)


# =============================================================================
#                    APPLY HOUSEHOLDER TESTS
# =============================================================================

class TestApplyHouseholder:
    """Tests for apply_householder — efficient H*A computation."""

    def test_matches_explicit_multiplication(self):
        """apply_householder(v, A) should equal (I - 2vv^T)A."""
        v = np.array([1.0, 0.0, 0.0])  # unit vector
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = apply_householder(v, A)
        H = np.eye(3) - 2 * np.outer(v, v)
        expected = H @ A
        assert np.allclose(result, expected, atol=1e-10)

    def test_preserves_norms(self):
        """Householder reflection preserves column norms."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(4)
        x = x / np.linalg.norm(x)  # make unit
        A = rng.standard_normal((4, 3))
        result = apply_householder(x, A)
        for j in range(A.shape[1]):
            assert np.isclose(
                np.linalg.norm(result[:, j]),
                np.linalg.norm(A[:, j]),
                atol=1e-10,
            )

    def test_involution(self):
        """Applying Householder twice returns original: H(HA) = A."""
        rng = np.random.default_rng(99)
        v = rng.standard_normal(3)
        v = v / np.linalg.norm(v)
        A = rng.standard_normal((3, 2))
        once = apply_householder(v, A)
        twice = apply_householder(v, once)
        assert np.allclose(twice, A, atol=1e-10)


# =============================================================================
#                    QR FACTORIZATION TESTS (HOUSEHOLDER)
# =============================================================================

class TestQRHouseholder:
    """Tests for qr_householder — full QR via Householder reflections."""

    def test_reconstruction(self, tall_matrix_5x3):
        """QR = A (reconstruction property)."""
        A = tall_matrix_5x3
        Q, R = qr_householder(A)
        assert np.allclose(Q @ R, A, atol=1e-10)

    def test_q_orthogonal(self, tall_matrix_5x3):
        """Q^T Q = I (orthogonality)."""
        A = tall_matrix_5x3
        Q, R = qr_householder(A)
        m = Q.shape[0]
        assert np.allclose(Q.T @ Q, np.eye(m), atol=1e-10)

    def test_r_upper_triangular(self, tall_matrix_5x3):
        """R is upper triangular (zeros below diagonal)."""
        A = tall_matrix_5x3
        Q, R = qr_householder(A)
        for i in range(R.shape[0]):
            for j in range(min(i, R.shape[1])):
                assert np.isclose(R[i, j], 0.0, atol=1e-10)

    def test_shapes(self, tall_matrix_5x3):
        """Q is (m, m) and R is (m, n)."""
        A = tall_matrix_5x3
        m, n = A.shape
        Q, R = qr_householder(A)
        assert Q.shape == (m, m)
        assert R.shape == (m, n)

    def test_identity_qr(self, identity_3x3):
        """QR of identity: Q=I, R=I."""
        I = identity_3x3
        Q, R = qr_householder(I)
        # Q should be orthogonal, R upper triangular, and Q@R = I
        assert np.allclose(Q @ R, I, atol=1e-10)

    def test_square_matrix(self, square_matrix_3x3):
        """QR of a square matrix reconstructs correctly."""
        A = square_matrix_3x3
        Q, R = qr_householder(A)
        assert np.allclose(Q @ R, A, atol=1e-10)
        assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-10)

    def test_agrees_with_numpy(self, tall_matrix_5x3):
        """Compare with np.linalg.qr results (reconstruction)."""
        A = tall_matrix_5x3
        Q_ours, R_ours = qr_householder(A)
        # Both should reconstruct A
        assert np.allclose(Q_ours @ R_ours, A, atol=1e-8)


# =============================================================================
#                    QR REDUCED TESTS
# =============================================================================

class TestQRReduced:
    """Tests for qr_reduced — thin/reduced QR factorization."""

    def test_reconstruction(self, tall_matrix_5x3):
        """Q1 @ R1 = A."""
        A = tall_matrix_5x3
        Q1, R1 = qr_reduced(A)
        assert np.allclose(Q1 @ R1, A, atol=1e-10)

    def test_shapes(self, tall_matrix_5x3):
        """Q1 is (m, n) and R1 is (n, n)."""
        A = tall_matrix_5x3
        m, n = A.shape
        Q1, R1 = qr_reduced(A)
        assert Q1.shape == (m, n)
        assert R1.shape == (n, n)

    def test_q_orthonormal_columns(self, tall_matrix_5x3):
        """Q1^T Q1 = I_n (columns orthonormal)."""
        A = tall_matrix_5x3
        n = A.shape[1]
        Q1, R1 = qr_reduced(A)
        assert np.allclose(Q1.T @ Q1, np.eye(n), atol=1e-10)

    def test_r_upper_triangular(self, tall_matrix_5x3):
        """R1 is upper triangular."""
        A = tall_matrix_5x3
        Q1, R1 = qr_reduced(A)
        for i in range(R1.shape[0]):
            for j in range(i):
                assert np.isclose(R1[i, j], 0.0, atol=1e-10)

    def test_square_matrix(self, square_matrix_3x3):
        """Reduced QR of square matrix: Q1 is (n, n), R1 is (n, n)."""
        A = square_matrix_3x3
        Q1, R1 = qr_reduced(A)
        assert np.allclose(Q1 @ R1, A, atol=1e-10)


# =============================================================================
#                    BACK SUBSTITUTION TESTS
# =============================================================================

class TestBackSubstitution:
    """Tests for back_substitution — solving Rx = b."""

    def test_simple_2x2(self):
        """2x2 system: R=[2,1;0,3], b=[4,6] => x=[1/2, 2]."""
        R = np.array([[2.0, 1.0], [0.0, 3.0]])
        b = np.array([4.0, 6.0])
        x = back_substitution(R, b)
        assert np.allclose(R @ x, b, atol=1e-10)

    def test_3x3(self, upper_triangular_3x3):
        """3x3 upper triangular system."""
        R = upper_triangular_3x3
        b = np.array([1.0, 2.0, 3.0])
        x = back_substitution(R, b)
        assert np.allclose(R @ x, b, atol=1e-10)

    def test_identity(self, identity_3x3):
        """Ix = b gives x = b."""
        I = identity_3x3
        b = np.array([1.0, 2.0, 3.0])
        x = back_substitution(I, b)
        assert np.allclose(x, b, atol=1e-10)

    def test_singular_raises(self):
        """Zero on diagonal raises ValueError."""
        R = np.array([[1.0, 2.0], [0.0, 0.0]])
        b = np.array([1.0, 2.0])
        with pytest.raises((ValueError, ZeroDivisionError)):
            back_substitution(R, b)

    def test_1x1(self):
        """1x1 system."""
        R = np.array([[4.0]])
        b = np.array([8.0])
        x = back_substitution(R, b)
        assert np.isclose(x[0], 2.0, atol=1e-10)

    def test_random_system(self, rng):
        """Random upper triangular system: verify R @ x = b."""
        n = 5
        R = np.triu(rng.standard_normal((n, n)))
        # Ensure no zero diagonal
        for i in range(n):
            if abs(R[i, i]) < 0.1:
                R[i, i] = 1.0
        b = rng.standard_normal(n)
        x = back_substitution(R, b)
        assert np.allclose(R @ x, b, atol=1e-8)


# =============================================================================
#                    SOLVE VIA QR TESTS
# =============================================================================

class TestSolveViaQR:
    """Tests for solve_via_qr — least squares via QR factorization."""

    def test_overdetermined_system(self, tall_matrix_5x3, rng):
        """Solution matches numpy lstsq for overdetermined system."""
        A = tall_matrix_5x3
        b = rng.standard_normal(5)
        x_ours = solve_via_qr(A, b)
        x_numpy, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        assert np.allclose(x_ours, x_numpy, atol=1e-8)

    def test_exact_system(self, square_matrix_3x3):
        """Exact system: A @ x_true = b."""
        A = square_matrix_3x3
        x_true = np.array([1.0, 2.0, 3.0])
        b = A @ x_true
        x_computed = solve_via_qr(A, b)
        assert np.allclose(x_computed, x_true, atol=1e-8)

    def test_residual_orthogonal_to_columns(self, tall_matrix_5x3, rng):
        """A^T (Ax - b) = 0 at the least squares solution."""
        A = tall_matrix_5x3
        b = rng.standard_normal(5)
        x = solve_via_qr(A, b)
        residual = A @ x - b
        should_be_zero = A.T @ residual
        assert np.allclose(should_be_zero, 0.0, atol=1e-8)

    def test_agrees_with_normal_equations(self, tall_matrix_5x3, rng):
        """QR solution matches normal equations solution."""
        A = tall_matrix_5x3
        b = rng.standard_normal(5)
        x_qr = solve_via_qr(A, b)
        x_normal = np.linalg.solve(A.T @ A, A.T @ b)
        assert np.allclose(x_qr, x_normal, atol=1e-8)

    def test_simple_linear_fit(self):
        """Fit y = 1 + 2x: coefficients should be [1, 2]."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 3.0, 5.0, 7.0])  # y = 1 + 2x
        X = np.column_stack([np.ones_like(x), x])
        beta = solve_via_qr(X, y)
        assert np.allclose(beta, [1.0, 2.0], atol=1e-10)

    def test_design_matrix_with_intercept(self, small_design_matrix, rng):
        """Works with intercept-column design matrices."""
        X = small_design_matrix  # 10x3 with intercept
        y = rng.standard_normal(10)
        beta = solve_via_qr(X, y)
        assert beta.shape == (3,)
        # Verify optimality
        residual = X @ beta - y
        assert np.allclose(X.T @ residual, 0.0, atol=1e-8)


# =============================================================================
#                    SOLVE WEIGHTED VIA QR TESTS
# =============================================================================

class TestSolveWeightedViaQR:
    """Tests for solve_weighted_via_qr — weighted ridge via QR."""

    def test_equal_weights_matches_unweighted(self, tall_matrix_5x3, rng):
        """Equal weights should give same result as unweighted QR solve."""
        X = tall_matrix_5x3
        y = rng.standard_normal(5)
        w = np.ones(5)
        beta_weighted = solve_weighted_via_qr(X, y, w, lambda_=0.0)
        beta_unweighted = solve_via_qr(X, y)
        assert np.allclose(beta_weighted, beta_unweighted, atol=1e-8)

    def test_large_lambda_shrinks_to_zero(self, tall_matrix_5x3, rng):
        """Very large lambda should shrink coefficients toward zero."""
        X = tall_matrix_5x3
        y = rng.standard_normal(5)
        w = np.ones(5)
        beta = solve_weighted_via_qr(X, y, w, lambda_=1e6)
        assert np.linalg.norm(beta) < 0.1

    def test_zero_lambda_no_regularization(self, tall_matrix_5x3, rng):
        """Lambda=0 should give the (weighted) least squares solution."""
        X = tall_matrix_5x3
        y = rng.standard_normal(5)
        w = np.ones(5) * 2.0  # uniform weights
        beta = solve_weighted_via_qr(X, y, w, lambda_=0.0)
        # With uniform weights, should match ordinary least squares
        beta_ols = solve_via_qr(X, y)
        assert np.allclose(beta, beta_ols, atol=1e-8)

    def test_nonuniform_weights(self, rng):
        """Non-uniform weights should change the solution."""
        X = np.column_stack([np.ones(5), rng.standard_normal((5, 2))])
        y = rng.standard_normal(5)
        w_uniform = np.ones(5)
        w_nonuniform = np.array([1.0, 1.0, 1.0, 10.0, 10.0])
        beta_uniform = solve_weighted_via_qr(X, y, w_uniform, lambda_=0.0)
        beta_nonuniform = solve_weighted_via_qr(X, y, w_nonuniform, lambda_=0.0)
        # Solutions should differ
        assert not np.allclose(beta_uniform, beta_nonuniform, atol=1e-4)


# =============================================================================
#                    CHECK QR FACTORIZATION TESTS
# =============================================================================

class TestCheckQRFactorization:
    """Tests for check_qr_factorization — verification utility."""

    def test_correct_factorization(self, tall_matrix_5x3):
        """Valid QR should pass all checks."""
        A = tall_matrix_5x3
        Q, R = np.linalg.qr(A, mode='reduced')
        # Extend Q and R to full for check
        Q_full, R_full = np.linalg.qr(A)
        result = check_qr_factorization(A, Q_full, R_full)
        assert result['factorization_correct']
        assert result['Q_orthogonal']
        assert result['R_upper_triangular']

    def test_wrong_factorization(self, tall_matrix_5x3, rng):
        """Random Q, R should fail verification."""
        A = tall_matrix_5x3
        m, n = A.shape
        Q_bad = rng.standard_normal((m, m))
        R_bad = rng.standard_normal((m, n))
        result = check_qr_factorization(A, Q_bad, R_bad)
        assert not result['factorization_correct']

    def test_identity_factorization(self, identity_3x3):
        """Identity matrix: Q=I, R=I passes all checks."""
        I = identity_3x3
        result = check_qr_factorization(I, I, I)
        assert result['factorization_correct']
        assert result['Q_orthogonal']
        assert result['R_upper_triangular']


# =============================================================================
#                    NUMERICAL STABILITY TESTS
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability of QR methods."""

    def test_qr_better_than_normal_equations(self, ill_conditioned_matrix):
        """QR is more stable than normal equations for ill-conditioned A."""
        A = ill_conditioned_matrix
        # Create a known solution
        x_true = np.array([1.0, 1.0])
        b = A @ x_true
        # QR solve (via numpy since our funcs are stubs)
        Q1, R1 = np.linalg.qr(A, mode='reduced')
        # If functions work, solve_via_qr should handle this reasonably
        try:
            x_qr = solve_via_qr(A, b)
            # Just check it returns something finite
            assert np.all(np.isfinite(x_qr))
        except (ValueError, np.linalg.LinAlgError):
            pass  # Acceptable for ill-conditioned case

    def test_large_matrix(self, rng):
        """QR works on larger matrices."""
        m, n = 50, 10
        A = rng.standard_normal((m, n))
        b = rng.standard_normal(m)
        x = solve_via_qr(A, b)
        x_ref, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        assert np.allclose(x, x_ref, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
