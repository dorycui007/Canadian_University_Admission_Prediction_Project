"""
Unit Tests for Projections Module
==================================

Tests for src/math/projections.py validating vector/subspace projection
operations and least squares solutions.

MAT223 References: Sections 4.2-4.3 (Orthogonal projections, least squares)
"""

import pytest
import numpy as np

from src.math.projections import (
    project_onto_vector,
    project_onto_subspace,
    compute_residual,
    verify_orthogonality,
    solve_normal_equations,
    solve_weighted_normal_equations,
    compute_hat_matrix,
    compute_leverage,
    projection_matrix_onto_complement,
    sum_of_squared_residuals,
    r_squared,
)


# =============================================================================
#                    VECTOR PROJECTION TESTS
# =============================================================================

class TestProjectOntoVector:
    """Tests for project_onto_vector(y, a) = ((y.a)/(a.a)) * a."""

    def test_project_onto_x_axis(self):
        """proj_{[1,0]}([3,4]) = [3, 0]."""
        result = project_onto_vector(np.array([3.0, 4.0]), np.array([1.0, 0.0]))
        np.testing.assert_array_almost_equal(result, [3.0, 0.0])

    def test_project_onto_diagonal(self):
        """proj_{[1,1]}([1,0]) = [0.5, 0.5]."""
        result = project_onto_vector(np.array([1.0, 0.0]), np.array([1.0, 1.0]))
        np.testing.assert_array_almost_equal(result, [0.5, 0.5])

    def test_residual_orthogonal(self):
        """v - proj is orthogonal to a."""
        v = np.array([3.0, 4.0, 5.0])
        a = np.array([1.0, 2.0, 2.0])
        proj = project_onto_vector(v, a)
        residual = v - proj
        assert np.isclose(np.dot(residual, a), 0.0, atol=1e-10)

    def test_projection_parallel_to_a(self):
        """Projection is a scalar multiple of a."""
        v = np.array([3.0, 4.0])
        a = np.array([1.0, 2.0])
        proj = project_onto_vector(v, a)
        ratio = proj / a
        assert np.isclose(ratio[0], ratio[1])

    def test_project_onto_self(self):
        """proj_v(v) = v."""
        v = np.array([3.0, 4.0])
        result = project_onto_vector(v, v)
        np.testing.assert_array_almost_equal(result, v)

    def test_project_orthogonal_gives_zero(self, standard_basis_2d):
        """proj_{e1}(e2) = 0."""
        e1, e2 = standard_basis_2d
        result = project_onto_vector(e2, e1)
        np.testing.assert_array_almost_equal(result, np.zeros(2))

    def test_project_already_parallel(self):
        """v = 3u implies proj_u(v) = v."""
        u = np.array([1.0, 2.0, 3.0])
        v = 3.0 * u
        result = project_onto_vector(v, u)
        np.testing.assert_array_almost_equal(result, v)

    def test_projection_magnitude_bounded(self):
        """||proj_u(v)|| <= ||v||."""
        v = np.array([3.0, 4.0, 5.0])
        u = np.array([1.0, 1.0, 1.0])
        proj = project_onto_vector(v, u)
        assert np.linalg.norm(proj) <= np.linalg.norm(v) + 1e-10

    def test_zero_vector_a_raises(self):
        """Projecting onto zero vector should raise error."""
        with pytest.raises((ValueError, ZeroDivisionError, Exception)):
            project_onto_vector(np.array([1.0, 2.0]), np.array([0.0, 0.0]))


# =============================================================================
#                    SUBSPACE PROJECTION TESTS
# =============================================================================

class TestProjectOntoSubspace:
    """Tests for project_onto_subspace(y, X) = X(X^T X)^{-1} X^T y."""

    def test_project_onto_plane(self, plane_basis):
        """Projection lies in column space of X."""
        X = plane_basis
        y = np.array([1.0, 2.0, 3.0])
        proj = project_onto_subspace(y, X)
        # proj = X @ coeffs for some coeffs
        coeffs, _, _, _ = np.linalg.lstsq(X, proj, rcond=None)
        np.testing.assert_array_almost_equal(X @ coeffs, proj)

    def test_residual_orthogonal_to_columns(self, plane_basis):
        """X^T (y - proj) = 0 (normal equations)."""
        X = plane_basis
        y = np.array([1.0, 2.0, 3.0])
        proj = project_onto_subspace(y, X)
        residual = y - proj
        xt_residual = X.T @ residual
        np.testing.assert_array_almost_equal(xt_residual, np.zeros(X.shape[1]))

    def test_projection_in_column_space(self, plane_basis):
        """Projection is expressible as X @ beta."""
        X = plane_basis
        y = np.array([1.0, 2.0, 3.0])
        proj = project_onto_subspace(y, X)
        beta, _, _, _ = np.linalg.lstsq(X, proj, rcond=None)
        np.testing.assert_array_almost_equal(X @ beta, proj)

    def test_vector_already_in_subspace(self, plane_basis):
        """If y in col(X), proj(y) = y."""
        X = plane_basis
        c = np.array([1.5, 2.5])
        y = X @ c
        proj = project_onto_subspace(y, X)
        np.testing.assert_array_almost_equal(proj, y)

    def test_project_onto_full_space(self, orthonormal_basis_3d):
        """When X spans all of R^n, proj(y) = y."""
        X = orthonormal_basis_3d
        y = np.array([1.0, 2.0, 3.0])
        proj = project_onto_subspace(y, X)
        np.testing.assert_array_almost_equal(proj, y)

    def test_projection_minimizes_distance(self, plane_basis, rng):
        """||y - proj|| < ||y - Xw|| for any other w."""
        X = plane_basis
        y = np.array([1.0, 2.0, 3.0])
        proj = project_onto_subspace(y, X)
        min_dist = np.linalg.norm(y - proj)
        for _ in range(10):
            w = rng.standard_normal(X.shape[1])
            other_point = X @ w
            other_dist = np.linalg.norm(y - other_point)
            assert min_dist <= other_dist + 1e-10


# =============================================================================
#                    COMPUTE RESIDUAL TESTS
# =============================================================================

class TestComputeResidual:
    """Tests for compute_residual(y, X, beta) = y - X @ beta."""

    def test_basic_residual(self):
        """Residual = y - X @ beta."""
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = np.array([1.0, 2.0, 4.0])
        beta = np.array([1.0, 2.0])
        r = compute_residual(y, X, beta)
        np.testing.assert_array_almost_equal(r, y - X @ beta)

    def test_perfect_fit_zero_residual(self):
        """Residual is zero when y = X @ beta exactly."""
        X = np.eye(3)
        beta = np.array([1.0, 2.0, 3.0])
        y = X @ beta
        r = compute_residual(y, X, beta)
        np.testing.assert_array_almost_equal(r, np.zeros(3))

    def test_residual_shape(self, tall_matrix_5x3, rng):
        """Residual has same shape as y."""
        X = tall_matrix_5x3
        y = rng.standard_normal(5)
        beta = rng.standard_normal(3)
        r = compute_residual(y, X, beta)
        assert r.shape == (5,)

    def test_residual_orthogonal_to_columns(self, tall_matrix_5x3):
        """For LS solution, X^T r = 0."""
        X = tall_matrix_5x3
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        beta_ls, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        r = compute_residual(y, X, beta_ls)
        np.testing.assert_array_almost_equal(X.T @ r, np.zeros(3))


# =============================================================================
#                    VERIFY ORTHOGONALITY TESTS
# =============================================================================

class TestVerifyOrthogonality:
    """Tests for verify_orthogonality(X, r, tol=1e-8)."""

    def test_true_for_ls_solution(self, tall_matrix_5x3):
        """LS residual is orthogonal to columns."""
        X = tall_matrix_5x3
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        beta_ls, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta_ls
        assert verify_orthogonality(X, r) is True

    def test_false_for_bad_beta(self, tall_matrix_5x3):
        """Arbitrary beta gives non-orthogonal residual."""
        X = tall_matrix_5x3
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bad_beta = np.array([100.0, 0.0, 0.0])
        r = y - X @ bad_beta
        assert verify_orthogonality(X, r, tol=1e-4) is False

    def test_tolerance_sensitivity(self, tall_matrix_5x3):
        """Tight tolerance may reject, loose tolerance accepts."""
        X = tall_matrix_5x3
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        beta_ls, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta_ls
        # Should pass with reasonable tolerance
        assert verify_orthogonality(X, r, tol=1e-6) is True


# =============================================================================
#                    SOLVE NORMAL EQUATIONS TESTS
# =============================================================================

class TestSolveNormalEquations:
    """Tests for solve_normal_equations(X, y) = (X^T X)^{-1} X^T y."""

    def test_basic_solution(self, tall_matrix_5x3):
        """Compare to numpy lstsq."""
        X = tall_matrix_5x3
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        beta = solve_normal_equations(X, y)
        beta_np, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        np.testing.assert_array_almost_equal(beta, beta_np)

    def test_exact_system(self, square_matrix_3x3):
        """Exact solution when system is square and invertible."""
        A = square_matrix_3x3
        x_true = np.array([1.0, 2.0, 3.0])
        b = A @ x_true
        x_computed = solve_normal_equations(A, b)
        np.testing.assert_array_almost_equal(x_computed, x_true)

    def test_solution_shape(self, tall_matrix_5x3):
        """Solution has shape (n_features,)."""
        X = tall_matrix_5x3
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        beta = solve_normal_equations(X, y)
        assert beta.shape == (3,)

    def test_residual_orthogonality(self, tall_matrix_5x3):
        """Normal equations guarantee X^T(y - X beta) = 0."""
        X = tall_matrix_5x3
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        beta = solve_normal_equations(X, y)
        r = y - X @ beta
        np.testing.assert_array_almost_equal(X.T @ r, np.zeros(3))


# =============================================================================
#              SOLVE WEIGHTED NORMAL EQUATIONS TESTS
# =============================================================================

class TestSolveWeightedNormalEquations:
    """Tests for solve_weighted_normal_equations(X, z, W, lambda_=0)."""

    def test_equal_weights_matches_unweighted(self, tall_matrix_5x3):
        """Equal weights = unweighted normal equations."""
        X = tall_matrix_5x3
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        W = np.ones(5)
        beta_w = solve_weighted_normal_equations(X, y, W, lambda_=0.0)
        beta_u = solve_normal_equations(X, y)
        np.testing.assert_array_almost_equal(beta_w, beta_u, decimal=5)

    def test_with_ridge(self, tall_matrix_5x3):
        """Ridge penalty shrinks coefficients."""
        X = tall_matrix_5x3
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        W = np.ones(5)
        beta_unreg = solve_weighted_normal_equations(X, y, W, lambda_=0.0)
        beta_reg = solve_weighted_normal_equations(X, y, W, lambda_=10.0)
        assert np.linalg.norm(beta_reg) < np.linalg.norm(beta_unreg)

    def test_solution_shape(self, tall_matrix_5x3):
        """Solution has shape (n_features,)."""
        X = tall_matrix_5x3
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        W = np.ones(5)
        beta = solve_weighted_normal_equations(X, y, W, lambda_=0.1)
        assert beta.shape == (3,)


# =============================================================================
#                    HAT MATRIX TESTS
# =============================================================================

class TestComputeHatMatrix:
    """Tests for compute_hat_matrix(X) = X(X^T X)^{-1}X^T."""

    def test_idempotent(self, plane_basis):
        """H^2 = H."""
        X = plane_basis
        H = compute_hat_matrix(X)
        np.testing.assert_array_almost_equal(H @ H, H)

    def test_symmetric(self, plane_basis):
        """H^T = H."""
        X = plane_basis
        H = compute_hat_matrix(X)
        np.testing.assert_array_almost_equal(H.T, H)

    def test_rank_equals_column_rank(self, plane_basis):
        """rank(H) = rank(X)."""
        X = plane_basis
        H = compute_hat_matrix(X)
        assert np.linalg.matrix_rank(H) == X.shape[1]

    def test_trace_equals_rank(self, plane_basis):
        """trace(H) = rank(X)."""
        X = plane_basis
        H = compute_hat_matrix(X)
        assert np.isclose(np.trace(H), X.shape[1])

    def test_eigenvalues_zero_or_one(self, plane_basis):
        """Eigenvalues of H are 0 or 1."""
        X = plane_basis
        H = compute_hat_matrix(X)
        eigenvalues = np.linalg.eigvalsh(H)
        for ev in eigenvalues:
            assert np.isclose(ev, 0.0, atol=1e-8) or np.isclose(ev, 1.0, atol=1e-8)

    def test_projects_y_to_column_space(self, plane_basis):
        """H @ y = proj_col(X)(y)."""
        X = plane_basis
        y = np.array([1.0, 2.0, 3.0])
        H = compute_hat_matrix(X)
        proj_h = H @ y
        proj_direct = project_onto_subspace(y, X)
        np.testing.assert_array_almost_equal(proj_h, proj_direct)


# =============================================================================
#                    COMPUTE LEVERAGE TESTS
# =============================================================================

class TestComputeLeverage:
    """Tests for compute_leverage(X) = diag(H)."""

    def test_bounds(self, tall_matrix_5x3):
        """0 <= h_ii <= 1."""
        X = tall_matrix_5x3
        h = compute_leverage(X)
        assert np.all(h >= -1e-10)
        assert np.all(h <= 1.0 + 1e-10)

    def test_sum_equals_rank(self, tall_matrix_5x3):
        """sum(h_ii) = rank(X)."""
        X = tall_matrix_5x3
        h = compute_leverage(X)
        assert np.isclose(np.sum(h), min(X.shape))

    def test_identity_leverage(self, identity_3x3):
        """For X = I, all leverages = 1."""
        h = compute_leverage(identity_3x3)
        np.testing.assert_array_almost_equal(h, np.ones(3))


# =============================================================================
#              PROJECTION ONTO COMPLEMENT TESTS
# =============================================================================

class TestProjectionMatrixOntoComplement:
    """Tests for projection_matrix_onto_complement(X) = I - H."""

    def test_idempotent(self, plane_basis):
        """(I - H)^2 = (I - H)."""
        X = plane_basis
        M = projection_matrix_onto_complement(X)
        np.testing.assert_array_almost_equal(M @ M, M)

    def test_symmetric(self, plane_basis):
        """(I - H)^T = (I - H)."""
        X = plane_basis
        M = projection_matrix_onto_complement(X)
        np.testing.assert_array_almost_equal(M.T, M)

    def test_annihilates_columns(self, plane_basis):
        """(I - H) X = 0."""
        X = plane_basis
        M = projection_matrix_onto_complement(X)
        np.testing.assert_array_almost_equal(M @ X, np.zeros_like(X))


# =============================================================================
#              SUM OF SQUARED RESIDUALS TESTS
# =============================================================================

class TestSumOfSquaredResiduals:
    """Tests for sum_of_squared_residuals(y, X, beta) = ||y - Xb||^2."""

    def test_basic_computation(self):
        """SSR = sum((y - Xb)^2)."""
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([1.0, 2.0])
        beta = np.array([0.5, 1.5])
        ssr = sum_of_squared_residuals(y, X, beta)
        expected = (1.0 - 0.5)**2 + (2.0 - 1.5)**2  # 0.25 + 0.25 = 0.5
        assert np.isclose(ssr, expected)

    def test_perfect_fit_is_zero(self):
        """SSR = 0 when y = X @ beta."""
        X = np.eye(3)
        beta = np.array([1.0, 2.0, 3.0])
        y = X @ beta
        assert np.isclose(sum_of_squared_residuals(y, X, beta), 0.0)

    def test_non_negative(self, tall_matrix_5x3, rng):
        """SSR >= 0 always."""
        X = tall_matrix_5x3
        y = rng.standard_normal(5)
        beta = rng.standard_normal(3)
        assert sum_of_squared_residuals(y, X, beta) >= -1e-10


# =============================================================================
#                    R-SQUARED TESTS
# =============================================================================

class TestRSquared:
    """Tests for r_squared(y, X, beta) = 1 - SSR/SST."""

    def test_perfect_fit(self):
        """R^2 = 1 for perfect fit."""
        X = np.eye(3)
        beta = np.array([1.0, 2.0, 3.0])
        y = X @ beta
        assert np.isclose(r_squared(y, X, beta), 1.0)

    def test_range(self, tall_matrix_5x3):
        """R^2 <= 1 for LS solution."""
        X = tall_matrix_5x3
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        beta_ls, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        r2 = r_squared(y, X, beta_ls)
        assert r2 <= 1.0 + 1e-10

    def test_mean_only_near_zero(self):
        """R^2 near 0 when model is just the mean."""
        n = 50
        rng = np.random.default_rng(42)
        X = np.ones((n, 1))
        y = rng.standard_normal(n)
        beta = np.array([np.mean(y)])
        r2 = r_squared(y, X, beta)
        assert np.isclose(r2, 0.0, atol=1e-10)

    def test_better_model_higher_r2(self, rng):
        """Adding informative features increases R^2."""
        n = 50
        x = rng.standard_normal(n)
        y = 2.0 * x + rng.standard_normal(n) * 0.1

        # Intercept-only model
        X1 = np.ones((n, 1))
        beta1 = np.array([np.mean(y)])
        r2_1 = r_squared(y, X1, beta1)

        # With informative feature
        X2 = np.column_stack([np.ones(n), x])
        beta2, _, _, _ = np.linalg.lstsq(X2, y, rcond=None)
        r2_2 = r_squared(y, X2, beta2)

        assert r2_2 > r2_1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
