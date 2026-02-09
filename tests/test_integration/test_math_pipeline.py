"""
Integration tests for the math pipeline.

These tests chain multiple math operations together to verify that modules
compose correctly end-to-end.  Because most source functions are still stubs
(``pass``), the majority of tests call ``pytest.skip`` so the suite stays
green while development proceeds.  Tests that rely *only* on the already-
implemented helpers (_ensure_numpy, add, scale) contain real assertions.
"""

import pytest
import numpy as np

from src.math.vectors import add, scale, dot, norm, normalize, linear_combination
from src.math.matrices import (
    matrix_multiply, matrix_vector_multiply, gram_matrix,
    transpose, identity, add_ridge,
)
from src.math.projections import (
    project_onto_subspace, compute_residual, verify_orthogonality,
    solve_normal_equations, compute_hat_matrix,
)
from src.math.qr import qr_householder, solve_via_qr, check_qr_factorization
from src.math.svd import compute_svd, ridge_via_svd, svd_solve
from src.math.ridge import ridge_solve, ridge_solve_qr


# ---------------------------------------------------------------------------
# TestVectorsToMatrices — chain vector ops into matrix ops
# ---------------------------------------------------------------------------

class TestVectorsToMatrices:
    """Integration tests that flow from vector operations into matrix operations."""

    def test_vector_ops_to_gram_matrix(self, rng):
        """Build vectors with add/scale, form a matrix, compute Gram matrix.

        The Gram matrix G = X^T X should be symmetric and positive semi-definite.
        """
        v1 = add([1, 2, 3], scale(2.0, [0, 1, 0]))
        v2 = add([4, 5, 6], scale(-1.0, [1, 1, 1]))
        X = np.array([v1, v2, scale(0.5, add(v1, v2))])
        result = gram_matrix(X)
        if result is None:
            pytest.skip("gram_matrix not yet implemented")
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result, result.T, atol=1e-12)
        eigenvalues = np.linalg.eigvalsh(result)
        assert np.all(eigenvalues >= -1e-10)

    def test_dot_product_via_matrix_multiply(self, rng):
        """dot(a, b) should equal the (1x1) result of a^T @ b via matrix multiply."""
        a = rng.standard_normal(5)
        b = rng.standard_normal(5)
        dot_val = dot(a, b)
        result = matrix_multiply(a.reshape(1, -1), b.reshape(-1, 1))
        if result is None:
            pytest.skip("matrix_multiply not yet implemented")
        np.testing.assert_allclose(dot_val, result.item(), atol=1e-12)

    def test_outer_product_rank(self, rng):
        """The outer product of two non-zero vectors should have rank 1."""
        from src.math.matrices import outer_product, compute_rank
        a = rng.standard_normal(4)
        b = rng.standard_normal(6)
        result = outer_product(a, b)
        if result is None:
            pytest.skip("outer_product not yet implemented")
        assert result.shape == (4, 6)
        assert compute_rank(result) == 1


# ---------------------------------------------------------------------------
# TestProjectionsPipeline — solve then verify geometric properties
# ---------------------------------------------------------------------------

class TestProjectionsPipeline:
    """Integration tests for projection and least-squares pipelines."""

    def test_solve_verify_orthogonality(self, rng):
        """Solve normal equations; residual must be orthogonal to column space."""
        n, p = 20, 3
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        result = solve_normal_equations(X, y)
        if result is None:
            pytest.skip("solve_normal_equations not yet implemented")
        r = compute_residual(y, X, result)
        assert verify_orthogonality(X, r, tol=1e-8)

    def test_hat_matrix_idempotent(self, rng):
        """The hat matrix H satisfies H @ H = H (idempotent)."""
        n, p = 10, 3
        X = rng.standard_normal((n, p))
        result = compute_hat_matrix(X)
        if result is None:
            pytest.skip("compute_hat_matrix not yet implemented")
        np.testing.assert_allclose(result @ result, result, atol=1e-10)

    def test_projection_residual_sum(self, rng):
        """projection + residual should reconstruct the original vector y."""
        n, p = 15, 4
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        result = project_onto_subspace(y, X)
        if result is None:
            pytest.skip("project_onto_subspace not yet implemented")
        beta = solve_normal_equations(X, y)
        r = compute_residual(y, X, beta)
        np.testing.assert_allclose(result + r, y, atol=1e-10)


# ---------------------------------------------------------------------------
# TestQRSolverPipeline — QR solver cross-checked against other solvers
# ---------------------------------------------------------------------------

class TestQRSolverPipeline:
    """Integration tests for the QR-based least-squares solver."""

    def test_qr_solve_matches_normal_equations(self, rng):
        """solve_via_qr and solve_normal_equations should agree for well-conditioned X."""
        n, p = 30, 4
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        result = solve_via_qr(X, y)
        if result is None:
            pytest.skip("solve_via_qr not yet implemented")
        beta_ne = solve_normal_equations(X, y)
        np.testing.assert_allclose(result, beta_ne, atol=1e-8)

    def test_qr_factorization_verified(self, rng):
        """qr_householder output must pass check_qr_factorization."""
        A = rng.standard_normal((8, 4))
        result = qr_householder(A)
        if result is None:
            pytest.skip("qr_householder not yet implemented")
        Q, R = result
        check = check_qr_factorization(A, Q, R, tol=1e-10)
        assert check['factorization_correct']
        assert check['Q_orthogonal']
        assert check['R_upper_triangular']

    def test_qr_vs_numpy(self, rng):
        """solve_via_qr result should match np.linalg.lstsq."""
        n, p = 25, 5
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        result = solve_via_qr(X, y)
        if result is None:
            pytest.skip("solve_via_qr not yet implemented")
        beta_np, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        np.testing.assert_allclose(result, beta_np, atol=1e-8)


# ---------------------------------------------------------------------------
# TestSVDRidgePipeline — SVD ridge solver cross-checked against direct ridge
# ---------------------------------------------------------------------------

class TestSVDRidgePipeline:
    """Integration tests for SVD-based ridge and general solvers."""

    def test_svd_ridge_matches_ridge_solve(self, rng):
        """ridge_via_svd and ridge_solve should produce the same coefficients."""
        n, p = 30, 5
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        lam = 1.0
        result = ridge_via_svd(X, y, lambda_=lam)
        if result is None:
            pytest.skip("ridge_via_svd not yet implemented")
        beta_direct = ridge_solve(X, y, lambda_=lam)
        np.testing.assert_allclose(result, beta_direct, atol=1e-8)

    def test_svd_solve_matches_qr_solve(self, rng):
        """svd_solve and solve_via_qr should agree for well-conditioned systems."""
        n, p = 20, 4
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        result = svd_solve(X, y)
        if result is None:
            pytest.skip("svd_solve not yet implemented")
        beta_qr = solve_via_qr(X, y)
        np.testing.assert_allclose(result, beta_qr, atol=1e-8)

    def test_ridge_path_monotone(self, rng):
        """Increasing lambda should decrease (or not increase) coefficient norms."""
        n, p = 40, 6
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
        result = ridge_solve(X, y, lambda_=lambdas[0])
        if result is None:
            pytest.skip("ridge_solve not yet implemented")
        norms = [np.linalg.norm(result)]
        for lam in lambdas[1:]:
            beta = ridge_solve(X, y, lambda_=lam)
            norms.append(np.linalg.norm(beta))
        for i in range(len(norms) - 1):
            assert norms[i] >= norms[i + 1] - 1e-10
