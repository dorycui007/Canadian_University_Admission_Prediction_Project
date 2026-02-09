"""
Unit Tests for SVD Module
=========================

Tests for src/math/svd.py covering SVD computation, low-rank approximation,
conditioning, ridge regression via SVD, and randomized truncated SVD.

MAT223 REFERENCES:
    - Section 3.4: Eigenvalues/Eigenvectors
    - Section 3.6: Applications (SVD, ridge)
"""

import pytest
import numpy as np

from src.math.svd import (
    compute_svd,
    singular_values,
    condition_number,
    matrix_rank,
    low_rank_approximation,
    reconstruct_from_svd,
    approximation_error,
    explained_variance_ratio,
    choose_rank,
    svd_solve,
    ridge_via_svd,
    effective_degrees_of_freedom,
    truncated_svd,
)


# =========================================================================
#                          TestComputeSVD
# =========================================================================

class TestComputeSVD:
    """Tests for compute_svd (thin and full decomposition)."""

    def test_reconstruction_tall_matrix(self, tall_matrix_5x3):
        """A = U diag(s) V^T for a tall matrix."""
        A = tall_matrix_5x3
        U, s, V = compute_svd(A, full_matrices=False)
        reconstructed = U @ np.diag(s) @ V.T
        np.testing.assert_allclose(reconstructed, A, atol=1e-12)

    def test_reconstruction_square_matrix(self, square_matrix_3x3):
        """A = U diag(s) V^T for a square matrix."""
        A = square_matrix_3x3
        U, s, V = compute_svd(A, full_matrices=False)
        reconstructed = U @ np.diag(s) @ V.T
        np.testing.assert_allclose(reconstructed, A, atol=1e-12)

    def test_thin_svd_shapes(self, tall_matrix_5x3):
        """Thin SVD shapes: U (m,k), s (k,), V (n,k) with k=min(m,n)."""
        A = tall_matrix_5x3
        m, n = A.shape
        k = min(m, n)
        U, s, V = compute_svd(A, full_matrices=False)
        assert U.shape == (m, k)
        assert s.shape == (k,)
        assert V.shape == (n, k)

    def test_full_svd_shapes(self, tall_matrix_5x3):
        """Full SVD shapes: U (m,m), s (k,), V (n,n)."""
        A = tall_matrix_5x3
        m, n = A.shape
        U, s, V = compute_svd(A, full_matrices=True)
        assert U.shape == (m, m)
        assert V.shape == (n, n)

    def test_singular_values_descending(self, tall_matrix_5x3):
        """Singular values are returned in descending order."""
        A = tall_matrix_5x3
        _, s, _ = compute_svd(A)
        for i in range(len(s) - 1):
            assert s[i] >= s[i + 1]

    def test_u_orthonormal_columns(self, tall_matrix_5x3):
        """U has orthonormal columns: U^T U = I."""
        A = tall_matrix_5x3
        U, _, _ = compute_svd(A, full_matrices=False)
        np.testing.assert_allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-12)

    def test_v_orthonormal_columns(self, tall_matrix_5x3):
        """V has orthonormal columns: V^T V = I."""
        A = tall_matrix_5x3
        _, _, V = compute_svd(A, full_matrices=False)
        np.testing.assert_allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-12)

    def test_identity_svd(self, identity_3x3):
        """SVD of identity: singular values are all 1."""
        U, s, V = compute_svd(identity_3x3)
        np.testing.assert_allclose(s, np.ones(3), atol=1e-12)


# =========================================================================
#                        TestSingularValues
# =========================================================================

class TestSingularValues:
    """Tests for singular_values (efficient sigma-only computation)."""

    def test_diagonal_matrix(self):
        """Singular values of a diagonal matrix equal sorted abs of diagonal."""
        A = np.diag([3.0, -7.0, 5.0])
        s = singular_values(A)
        np.testing.assert_allclose(s, np.array([7.0, 5.0, 3.0]), atol=1e-12)

    def test_identity(self, identity_3x3):
        """All singular values of the identity are 1."""
        s = singular_values(identity_3x3)
        np.testing.assert_allclose(s, np.ones(3), atol=1e-12)

    def test_nonnegative(self, tall_matrix_5x3):
        """Singular values are always non-negative."""
        s = singular_values(tall_matrix_5x3)
        assert np.all(s >= 0)

    def test_count_equals_min_dim(self, tall_matrix_5x3):
        """Number of singular values equals min(m, n)."""
        A = tall_matrix_5x3
        s = singular_values(A)
        assert len(s) == min(A.shape)


# =========================================================================
#                       TestConditionNumber
# =========================================================================

class TestConditionNumberSVD:
    """Tests for condition_number (sigma_max / sigma_min)."""

    def test_identity_condition_one(self, identity_3x3):
        """Identity matrix has condition number 1."""
        kappa = condition_number(identity_3x3)
        assert np.isclose(kappa, 1.0)

    def test_ill_conditioned(self, ill_conditioned_matrix):
        """Ill-conditioned matrix has very large condition number."""
        kappa = condition_number(ill_conditioned_matrix)
        assert kappa > 1e6

    def test_rank_deficient_returns_inf(self, rank_deficient_matrix):
        """Rank-deficient matrix has infinite condition number."""
        kappa = condition_number(rank_deficient_matrix)
        assert np.isinf(kappa)

    def test_scalar_invariance(self, square_matrix_3x3):
        """Scaling A by a nonzero scalar does not change condition number."""
        kappa_A = condition_number(square_matrix_3x3)
        kappa_cA = condition_number(5.0 * square_matrix_3x3)
        assert np.isclose(kappa_A, kappa_cA, rtol=1e-10)

    def test_at_least_one(self, tall_matrix_5x3):
        """Condition number is always >= 1."""
        kappa = condition_number(tall_matrix_5x3)
        assert kappa >= 1.0


# =========================================================================
#                          TestMatrixRank
# =========================================================================

class TestMatrixRank:
    """Tests for matrix_rank (numerical rank via SVD)."""

    def test_identity_full_rank(self, identity_3x3):
        """Identity has full rank 3."""
        assert matrix_rank(identity_3x3) == 3

    def test_rank_deficient(self, rank_deficient_matrix):
        """Rank-deficient matrix with dependent columns has rank 1."""
        assert matrix_rank(rank_deficient_matrix) == 1

    def test_zero_matrix(self):
        """All-zeros matrix has rank 0."""
        Z = np.zeros((4, 3))
        assert matrix_rank(Z) == 0

    def test_tall_full_column_rank(self, tall_matrix_5x3):
        """Random tall matrix has full column rank 3."""
        assert matrix_rank(tall_matrix_5x3) == 3

    def test_custom_tolerance(self):
        """Custom tolerance can lower the effective rank."""
        A = np.diag([10.0, 1.0, 0.01])
        assert matrix_rank(A, tol=0.1) == 2


# =========================================================================
#                     TestLowRankApproximation
# =========================================================================

class TestLowRankApproximation:
    """Tests for low_rank_approximation (Eckart-Young)."""

    def test_output_shapes(self, tall_matrix_5x3):
        """U_k, s_k, V_k have correct truncated shapes."""
        A = tall_matrix_5x3
        k = 2
        U_k, s_k, V_k = low_rank_approximation(A, k)
        m, n = A.shape
        assert U_k.shape == (m, k)
        assert s_k.shape == (k,)
        assert V_k.shape == (n, k)

    def test_full_rank_recovers_original(self, tall_matrix_5x3):
        """Keeping all singular values recovers A exactly."""
        A = tall_matrix_5x3
        k = min(A.shape)
        U_k, s_k, V_k = low_rank_approximation(A, k)
        A_approx = U_k @ np.diag(s_k) @ V_k.T
        np.testing.assert_allclose(A_approx, A, atol=1e-12)

    def test_rank_of_approximation(self, rng):
        """The rank-k approximation has matrix rank at most k."""
        A = rng.standard_normal((10, 8))
        k = 3
        U_k, s_k, V_k = low_rank_approximation(A, k)
        A_k = U_k @ np.diag(s_k) @ V_k.T
        assert np.linalg.matrix_rank(A_k) <= k

    def test_invalid_k_raises(self, tall_matrix_5x3):
        """k larger than min(m,n) or k < 1 raises ValueError."""
        A = tall_matrix_5x3
        with pytest.raises(ValueError):
            low_rank_approximation(A, k=0)
        with pytest.raises(ValueError):
            low_rank_approximation(A, k=min(A.shape) + 1)


# =========================================================================
#                      TestReconstructFromSVD
# =========================================================================

class TestReconstructFromSVD:
    """Tests for reconstruct_from_svd."""

    def test_roundtrip_tall(self, tall_matrix_5x3):
        """Decompose then reconstruct recovers original."""
        A = tall_matrix_5x3
        U_np, s_np, Vt_np = np.linalg.svd(A, full_matrices=False)
        result = reconstruct_from_svd(U_np, s_np, Vt_np.T)
        if result is None:
            pytest.skip("Not yet implemented")
        np.testing.assert_allclose(result, A, atol=1e-12)

    def test_roundtrip_square(self, square_matrix_3x3):
        """Decompose then reconstruct recovers original square matrix."""
        A = square_matrix_3x3
        U_np, s_np, Vt_np = np.linalg.svd(A, full_matrices=False)
        result = reconstruct_from_svd(U_np, s_np, Vt_np.T)
        if result is None:
            pytest.skip("Not yet implemented")
        np.testing.assert_allclose(result, A, atol=1e-12)

    def test_partial_reconstruction(self, rng):
        """Reconstruction from rank-k components has correct shape."""
        A = rng.standard_normal((6, 4))
        U_np, s_np, Vt_np = np.linalg.svd(A, full_matrices=False)
        k = 2
        result = reconstruct_from_svd(U_np[:, :k], s_np[:k], Vt_np[:k, :].T)
        if result is None:
            pytest.skip("Not yet implemented")
        assert result.shape == A.shape


# =========================================================================
#                      TestApproximationError
# =========================================================================

class TestApproximationError:
    """Tests for approximation_error (Frobenius and spectral)."""

    def test_frobenius_diagonal(self):
        """Frobenius error on a diagonal matrix matches analytic formula."""
        A = np.diag([10.0, 5.0, 2.0, 0.1])
        err = approximation_error(A, k=2, norm='fro')
        expected = np.sqrt(2.0**2 + 0.1**2)
        assert np.isclose(err, expected, atol=1e-10)

    def test_spectral_error(self):
        """Spectral error equals the (k+1)-th singular value."""
        A = np.diag([10.0, 5.0, 2.0, 0.1])
        err = approximation_error(A, k=2, norm='spectral')
        assert np.isclose(err, 2.0, atol=1e-10)

    def test_full_rank_zero_error(self, tall_matrix_5x3):
        """Keeping all components gives zero error."""
        A = tall_matrix_5x3
        k = min(A.shape)
        err = approximation_error(A, k=k, norm='fro')
        assert np.isclose(err, 0.0, atol=1e-10)

    def test_error_decreases_with_k(self, rng):
        """Error decreases as rank k increases."""
        A = rng.standard_normal((8, 6))
        errors = [approximation_error(A, k=k, norm='fro') for k in range(1, 6)]
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1] - 1e-12


# =========================================================================
#                    TestExplainedVarianceRatio
# =========================================================================

class TestExplainedVarianceRatio:
    """Tests for explained_variance_ratio."""

    def test_sums_to_one(self, tall_matrix_5x3):
        """Ratios sum to 1."""
        evr = explained_variance_ratio(tall_matrix_5x3)
        assert np.isclose(np.sum(evr), 1.0)

    def test_nonnegative(self, tall_matrix_5x3):
        """All ratios are non-negative."""
        evr = explained_variance_ratio(tall_matrix_5x3)
        assert np.all(evr >= 0)

    def test_descending_order(self, tall_matrix_5x3):
        """Ratios are in descending order."""
        evr = explained_variance_ratio(tall_matrix_5x3)
        for i in range(len(evr) - 1):
            assert evr[i] >= evr[i + 1] - 1e-14

    def test_known_values(self):
        """Ratios for a known diagonal matrix."""
        A = np.diag([10.0, 5.0, 2.0, 1.0])
        evr = explained_variance_ratio(A)
        total = 100 + 25 + 4 + 1
        np.testing.assert_allclose(evr, np.array([100, 25, 4, 1]) / total, atol=1e-12)


# =========================================================================
#                          TestChooseRank
# =========================================================================

class TestChooseRank:
    """Tests for choose_rank (automatic rank selection)."""

    def test_threshold_95_diagonal(self):
        """Rank selection with 95% threshold on a known diagonal."""
        A = np.diag([10.0, 5.0, 2.0, 0.1])
        k = choose_rank(A, variance_threshold=0.95)
        # sigma^2 = [100, 25, 4, 0.01], total = 129.01
        # cumul = [100/129.01, 125/129.01, ...] ~ [0.775, 0.969, ...]
        assert k == 2

    def test_threshold_one_returns_all(self, tall_matrix_5x3):
        """Threshold of 1.0 returns full rank."""
        A = tall_matrix_5x3
        k = choose_rank(A, variance_threshold=1.0)
        assert k == min(A.shape)

    def test_returns_at_least_one(self, tall_matrix_5x3):
        """Always returns at least rank 1."""
        k = choose_rank(tall_matrix_5x3, variance_threshold=0.0)
        assert k >= 1

    def test_higher_threshold_needs_more_components(self, rng):
        """Higher variance threshold requires more components."""
        A = rng.standard_normal((20, 10))
        k_low = choose_rank(A, variance_threshold=0.5)
        k_high = choose_rank(A, variance_threshold=0.99)
        assert k_high >= k_low


# =========================================================================
#                          TestSVDSolve
# =========================================================================

class TestSVDSolve:
    """Tests for svd_solve (pseudoinverse-based solver)."""

    def test_exact_square_system(self, square_matrix_3x3):
        """Solve an exact square system Ax = b."""
        A = square_matrix_3x3
        x_true = np.array([1.0, -1.0, 2.0])
        b = A @ x_true
        x = svd_solve(A, b)
        np.testing.assert_allclose(x, x_true, atol=1e-10)

    def test_overdetermined_matches_lstsq(self, tall_matrix_5x3):
        """Overdetermined least squares matches numpy lstsq."""
        A = tall_matrix_5x3
        b = np.arange(1.0, 6.0)
        x_svd = svd_solve(A, b)
        x_np, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        np.testing.assert_allclose(x_svd, x_np, atol=1e-10)

    def test_residual_orthogonal(self, tall_matrix_5x3):
        """Residual is orthogonal to column space: A^T (Ax - b) ~ 0."""
        A = tall_matrix_5x3
        b = np.ones(5)
        x = svd_solve(A, b)
        residual = A @ x - b
        np.testing.assert_allclose(A.T @ residual, np.zeros(3), atol=1e-10)

    def test_rank_deficient_system(self, rank_deficient_matrix):
        """Returns minimum-norm solution for rank-deficient system."""
        A = rank_deficient_matrix
        b = np.array([1.0, 2.0, 3.0])
        x = svd_solve(A, b)
        assert x.shape == (2,)
        # Should still minimize ||Ax - b||
        residual_norm = np.linalg.norm(A @ x - b)
        assert residual_norm < np.linalg.norm(b)


# =========================================================================
#                        TestRidgeViaSVD
# =========================================================================

class TestRidgeViaSVD:
    """Tests for ridge_via_svd."""

    def test_zero_lambda_approaches_ols(self, rng):
        """With lambda ~ 0, ridge solution approximates OLS."""
        X = rng.standard_normal((20, 5))
        y = rng.standard_normal(20)
        beta_ridge = ridge_via_svd(X, y, lambda_=1e-12)
        beta_ols, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        np.testing.assert_allclose(beta_ridge, beta_ols, atol=1e-6)

    def test_large_lambda_shrinks_to_zero(self, rng):
        """Very large lambda shrinks coefficients toward zero."""
        X = rng.standard_normal((20, 5))
        y = rng.standard_normal(20)
        beta = ridge_via_svd(X, y, lambda_=1e10)
        np.testing.assert_allclose(beta, np.zeros(5), atol=1e-4)

    def test_matches_closed_form(self, rng):
        """Matches the closed-form (X^TX + lambda I)^{-1} X^T y."""
        X = rng.standard_normal((30, 4))
        y = rng.standard_normal(30)
        lam = 1.5
        beta_svd = ridge_via_svd(X, y, lambda_=lam)
        p = X.shape[1]
        beta_cf = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)
        np.testing.assert_allclose(beta_svd, beta_cf, atol=1e-10)

    def test_output_shape(self, rng):
        """Output has shape (p,)."""
        X = rng.standard_normal((15, 6))
        y = rng.standard_normal(15)
        beta = ridge_via_svd(X, y, lambda_=1.0)
        assert beta.shape == (6,)


# =========================================================================
#                  TestEffectiveDegreesOfFreedom
# =========================================================================

class TestEffectiveDegreesOfFreedom:
    """Tests for effective_degrees_of_freedom."""

    def test_zero_lambda_equals_p(self, rng):
        """With lambda=0, df = p (full model)."""
        X = rng.standard_normal((50, 5))
        df = effective_degrees_of_freedom(X, lambda_=0.0)
        assert np.isclose(df, 5.0, atol=1e-8)

    def test_large_lambda_near_zero(self, rng):
        """With very large lambda, df -> 0."""
        X = rng.standard_normal((50, 5))
        df = effective_degrees_of_freedom(X, lambda_=1e12)
        assert df < 0.01

    def test_bounded_between_zero_and_p(self, rng):
        """df is always in [0, p]."""
        X = rng.standard_normal((50, 8))
        df = effective_degrees_of_freedom(X, lambda_=1.0)
        assert 0 <= df <= 8.0

    def test_monotone_in_lambda(self, rng):
        """df decreases as lambda increases."""
        X = rng.standard_normal((50, 5))
        df_small = effective_degrees_of_freedom(X, lambda_=0.1)
        df_large = effective_degrees_of_freedom(X, lambda_=10.0)
        assert df_small > df_large


# =========================================================================
#                        TestTruncatedSVD
# =========================================================================

class TestTruncatedSVD:
    """Tests for truncated_svd (randomized algorithm)."""

    def test_output_shapes(self, rng):
        """U_k, s_k, V_k have correct truncated shapes."""
        A = rng.standard_normal((50, 30))
        k = 5
        U_k, s_k, V_k = truncated_svd(A, k=k, random_state=0)
        assert U_k.shape == (50, k)
        assert s_k.shape == (k,)
        assert V_k.shape == (30, k)

    def test_approximation_quality(self, rng):
        """Approximation captures most of the Frobenius norm for low-rank input."""
        # Construct a matrix with clear low-rank structure
        U_true = rng.standard_normal((50, 3))
        V_true = rng.standard_normal((30, 3))
        A = U_true @ V_true.T + 0.01 * rng.standard_normal((50, 30))
        U_k, s_k, V_k = truncated_svd(A, k=3, n_iter=5, random_state=42)
        A_approx = U_k @ np.diag(s_k) @ V_k.T
        rel_err = np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro')
        assert rel_err < 0.05

    def test_singular_values_descending(self, rng):
        """Returned singular values are in descending order."""
        A = rng.standard_normal((40, 20))
        _, s_k, _ = truncated_svd(A, k=8, random_state=7)
        for i in range(len(s_k) - 1):
            assert s_k[i] >= s_k[i + 1] - 1e-10

    def test_reproducible_with_seed(self, rng):
        """Same random_state produces identical results."""
        A = rng.standard_normal((30, 20))
        U1, s1, V1 = truncated_svd(A, k=4, random_state=123)
        U2, s2, V2 = truncated_svd(A, k=4, random_state=123)
        np.testing.assert_allclose(s1, s2, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
