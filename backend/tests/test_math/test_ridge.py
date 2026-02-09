"""
Unit Tests for Ridge Regression Module
=======================================

Tests for src/math/ridge.py covering the ridge solver, QR-based solver,
weighted ridge, cross-validation, regularization path, GCV,
feature standardization, and effective degrees of freedom.

MAT223 REFERENCES:
    - Section 3.6: Applications of Eigenvalues
    - Section 4.8: Approximating Solutions
"""

import pytest
import numpy as np

from src.math.ridge import (
    RidgeConvergenceError,
    IllConditionedError,
    ridge_solve,
    ridge_solve_qr,
    weighted_ridge_solve,
    ridge_loocv,
    ridge_cv,
    ridge_path,
    ridge_gcv,
    standardize_features,
    ridge_effective_df,
)


# =========================================================================
#                          TestRidgeSolve
# =========================================================================

class TestRidgeSolve:
    """Tests for ridge_solve via Cholesky / normal equations."""

    def test_known_solution(self):
        """Verify against manually computed solution for a small system."""
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([3.0, 4.0])
        lam = 1.0
        # (X^TX + I) = 2I, X^Ty = [3,4], beta = [1.5, 2.0]
        beta = ridge_solve(X, y, lam)
        np.testing.assert_allclose(beta, np.array([1.5, 2.0]), atol=1e-12)

    def test_output_shape(self, rng):
        """Output shape is (p,)."""
        X = rng.standard_normal((20, 5))
        y = rng.standard_normal(20)
        beta = ridge_solve(X, y, lambda_=1.0)
        assert beta.shape == (5,)

    def test_large_lambda_shrinks(self, rng):
        """Very large lambda drives coefficients toward zero."""
        X = rng.standard_normal((30, 4))
        y = rng.standard_normal(30)
        beta = ridge_solve(X, y, lambda_=1e10)
        np.testing.assert_allclose(beta, np.zeros(4), atol=1e-3)

    def test_zero_lambda_approaches_ols(self, rng):
        """Lambda near zero approximates OLS."""
        X = rng.standard_normal((30, 4))
        y = rng.standard_normal(30)
        beta_ridge = ridge_solve(X, y, lambda_=1e-14)
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        np.testing.assert_allclose(beta_ridge, beta_ols, atol=1e-6)

    def test_symmetric_pd_input(self, symmetric_pd_matrix, rng):
        """Works with a symmetric PD design matrix."""
        X = symmetric_pd_matrix
        y = rng.standard_normal(X.shape[0])
        beta = ridge_solve(X, y, lambda_=0.5)
        assert beta.shape == (X.shape[1],)

    def test_ill_conditioned_stabilized(self, ill_conditioned_matrix):
        """Ridge regularization stabilizes an ill-conditioned system."""
        X = ill_conditioned_matrix
        y = np.array([1.0, 2.0])
        beta = ridge_solve(X, y, lambda_=1.0)
        assert np.all(np.isfinite(beta))


# =========================================================================
#                        TestRidgeSolveQR
# =========================================================================

class TestRidgeSolveQR:
    """Tests for ridge_solve_qr (augmented QR approach)."""

    def test_agrees_with_ridge_solve(self, rng):
        """QR and Cholesky solvers give the same answer."""
        X = rng.standard_normal((25, 5))
        y = rng.standard_normal(25)
        lam = 2.0
        beta_chol = ridge_solve(X, y, lam)
        beta_qr = ridge_solve_qr(X, y, lam)
        np.testing.assert_allclose(beta_qr, beta_chol, atol=1e-8)

    def test_output_shape(self, rng):
        """Output shape is (p,)."""
        X = rng.standard_normal((20, 6))
        y = rng.standard_normal(20)
        beta = ridge_solve_qr(X, y, lambda_=1.0)
        assert beta.shape == (6,)

    def test_large_lambda_shrinks(self, rng):
        """Very large lambda drives coefficients toward zero."""
        X = rng.standard_normal((20, 4))
        y = rng.standard_normal(20)
        beta = ridge_solve_qr(X, y, lambda_=1e10)
        np.testing.assert_allclose(beta, np.zeros(4), atol=1e-3)

    def test_known_solution(self):
        """Same known solution as TestRidgeSolve."""
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([3.0, 4.0])
        beta = ridge_solve_qr(X, y, lambda_=1.0)
        np.testing.assert_allclose(beta, np.array([1.5, 2.0]), atol=1e-10)


# =========================================================================
#                    TestWeightedRidgeSolve
# =========================================================================

class TestWeightedRidgeSolve:
    """Tests for weighted_ridge_solve (IRLS inner step)."""

    def test_uniform_weights_match_ridge(self, rng):
        """Uniform weights W=1 reduce to ordinary ridge."""
        X = rng.standard_normal((20, 4))
        y = rng.standard_normal(20)
        lam = 1.5
        W = np.ones(20)
        beta_w = weighted_ridge_solve(X, y, W, lam)
        beta_r = ridge_solve(X, y, lam)
        np.testing.assert_allclose(beta_w, beta_r, atol=1e-10)

    def test_output_shape(self, rng):
        """Output shape is (p,)."""
        X = rng.standard_normal((15, 3))
        z = rng.standard_normal(15)
        W = rng.uniform(0.1, 1.0, 15)
        beta = weighted_ridge_solve(X, z, W, lambda_=1.0)
        assert beta.shape == (3,)

    def test_higher_weight_gives_closer_fit(self, rng):
        """Up-weighting one observation pulls fitted value closer."""
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        z = np.array([10.0, 0.0, 0.0])
        W_low = np.array([1.0, 1.0, 1.0])
        W_high = np.array([100.0, 1.0, 1.0])
        beta_low = weighted_ridge_solve(X, z, W_low, lambda_=0.1)
        beta_high = weighted_ridge_solve(X, z, W_high, lambda_=0.1)
        fit_low = X[0] @ beta_low
        fit_high = X[0] @ beta_high
        assert abs(fit_high - 10.0) < abs(fit_low - 10.0)

    def test_scalar_weight(self, rng):
        """Scalar weight is broadcast to all observations."""
        X = rng.standard_normal((10, 3))
        z = rng.standard_normal(10)
        beta_scalar = weighted_ridge_solve(X, z, 2.0, lambda_=1.0)
        beta_array = weighted_ridge_solve(X, z, 2.0 * np.ones(10), lambda_=1.0)
        np.testing.assert_allclose(beta_scalar, beta_array, atol=1e-12)


# =========================================================================
#                         TestRidgeLOOCV
# =========================================================================

class TestRidgeLOOCV:
    """Tests for ridge_loocv (efficient leave-one-out CV)."""

    def test_positive_score(self, rng):
        """LOO-CV error is non-negative."""
        X = rng.standard_normal((30, 4))
        y = rng.standard_normal(30)
        score = ridge_loocv(X, y, lambda_=1.0)
        assert score >= 0

    def test_increases_with_very_large_lambda(self, rng):
        """Prediction error increases when lambda is excessively large."""
        X = rng.standard_normal((40, 5))
        beta_true = np.array([1.0, 2.0, -1.0, 0.5, 0.0])
        y = X @ beta_true + 0.1 * rng.standard_normal(40)
        score_good = ridge_loocv(X, y, lambda_=0.1)
        score_bad = ridge_loocv(X, y, lambda_=1e6)
        assert score_bad > score_good

    def test_finite_output(self, rng):
        """Output is finite for well-posed problem."""
        X = rng.standard_normal((25, 3))
        y = rng.standard_normal(25)
        score = ridge_loocv(X, y, lambda_=1.0)
        assert np.isfinite(score)

    def test_returns_scalar(self, rng):
        """Returns a single float."""
        X = rng.standard_normal((20, 3))
        y = rng.standard_normal(20)
        score = ridge_loocv(X, y, lambda_=0.5)
        assert isinstance(score, (float, np.floating))


# =========================================================================
#                          TestRidgeCV
# =========================================================================

class TestRidgeCV:
    """Tests for ridge_cv (k-fold cross-validation)."""

    def test_returns_best_lambda_and_scores(self, rng):
        """Returns (best_lambda, cv_scores) with correct types."""
        X = rng.standard_normal((50, 5))
        y = rng.standard_normal(50)
        lambdas = [0.01, 0.1, 1.0, 10.0]
        best_lam, scores = ridge_cv(X, y, lambdas, cv_folds=3)
        assert best_lam in lambdas
        assert len(scores) == len(lambdas)

    def test_scores_nonnegative(self, rng):
        """All CV scores are non-negative."""
        X = rng.standard_normal((50, 4))
        y = rng.standard_normal(50)
        lambdas = [0.1, 1.0, 10.0]
        _, scores = ridge_cv(X, y, lambdas, cv_folds=5)
        assert np.all(scores >= 0)

    def test_best_lambda_reasonable(self, rng):
        """Best lambda is not extreme for a well-posed problem."""
        X = rng.standard_normal((80, 5))
        beta_true = np.array([2.0, -1.0, 0.5, 0.0, 1.0])
        y = X @ beta_true + 0.5 * rng.standard_normal(80)
        lambdas = np.logspace(-3, 3, 15).tolist()
        best_lam, _ = ridge_cv(X, y, lambdas, cv_folds=5)
        assert best_lam < 100  # should not choose extremely large lambda

    def test_scores_length(self, rng):
        """Length of scores matches length of lambdas."""
        X = rng.standard_normal((40, 3))
        y = rng.standard_normal(40)
        lambdas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        _, scores = ridge_cv(X, y, lambdas)
        assert len(scores) == 6


# =========================================================================
#                         TestRidgePath
# =========================================================================

class TestRidgePath:
    """Tests for ridge_path (coefficient path across lambdas)."""

    def test_output_shape(self, rng):
        """Path has shape (n_lambdas, p)."""
        X = rng.standard_normal((30, 4))
        y = rng.standard_normal(30)
        lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
        path = ridge_path(X, y, lambdas)
        assert path.shape == (5, 4)

    def test_shrinkage_increases_with_lambda(self, rng):
        """Coefficient norms decrease as lambda increases."""
        X = rng.standard_normal((40, 5))
        y = rng.standard_normal(40)
        lambdas = [0.01, 1.0, 100.0]
        path = ridge_path(X, y, lambdas)
        norms = [np.linalg.norm(path[i]) for i in range(len(lambdas))]
        for i in range(len(norms) - 1):
            assert norms[i] >= norms[i + 1] - 1e-8

    def test_first_row_matches_small_lambda(self, rng):
        """First row of path equals ridge_solve with same lambda."""
        X = rng.standard_normal((30, 4))
        y = rng.standard_normal(30)
        lam = 0.5
        path = ridge_path(X, y, [lam])
        beta = ridge_solve(X, y, lam)
        np.testing.assert_allclose(path[0], beta, atol=1e-10)

    def test_large_lambda_near_zero(self, rng):
        """Last row (large lambda) has near-zero coefficients."""
        X = rng.standard_normal((30, 4))
        y = rng.standard_normal(30)
        lambdas = [0.01, 1e8]
        path = ridge_path(X, y, lambdas)
        np.testing.assert_allclose(path[-1], np.zeros(4), atol=1e-2)


# =========================================================================
#                          TestRidgeGCV
# =========================================================================

class TestRidgeGCV:
    """Tests for ridge_gcv (generalized cross-validation)."""

    def test_nonnegative(self, rng):
        """GCV score is non-negative."""
        X = rng.standard_normal((30, 4))
        y = rng.standard_normal(30)
        score = ridge_gcv(X, y, lambda_=1.0)
        assert score >= 0

    def test_finite(self, rng):
        """GCV score is finite."""
        X = rng.standard_normal((30, 4))
        y = rng.standard_normal(30)
        score = ridge_gcv(X, y, lambda_=1.0)
        assert np.isfinite(score)

    def test_large_lambda_increases_error(self, rng):
        """Excessively large lambda yields higher GCV."""
        X = rng.standard_normal((50, 5))
        beta_true = np.array([1.0, 2.0, -1.0, 0.5, 0.0])
        y = X @ beta_true + 0.1 * rng.standard_normal(50)
        gcv_good = ridge_gcv(X, y, lambda_=0.1)
        gcv_bad = ridge_gcv(X, y, lambda_=1e6)
        assert gcv_bad > gcv_good

    def test_returns_scalar(self, rng):
        """GCV returns a single float."""
        X = rng.standard_normal((20, 3))
        y = rng.standard_normal(20)
        score = ridge_gcv(X, y, lambda_=0.5)
        assert isinstance(score, (float, np.floating))


# =========================================================================
#                    TestStandardizeFeatures
# =========================================================================

class TestStandardizeFeatures:
    """Tests for standardize_features."""

    def test_zero_mean(self, rng):
        """Centered features have mean approximately zero."""
        X = rng.standard_normal((50, 4)) * 10 + 5
        X_std, means, stds = standardize_features(X)
        np.testing.assert_allclose(X_std.mean(axis=0), np.zeros(4), atol=1e-12)

    def test_unit_std(self, rng):
        """Scaled features have standard deviation approximately one."""
        X = rng.standard_normal((50, 4)) * 10 + 5
        X_std, means, stds = standardize_features(X)
        np.testing.assert_allclose(X_std.std(axis=0), np.ones(4), atol=1e-10)

    def test_no_center(self, rng):
        """With center=False, means are zero (no shift applied)."""
        X = rng.standard_normal((30, 3)) + 100
        X_std, means, stds = standardize_features(X, center=False, scale=True)
        np.testing.assert_allclose(means, np.zeros(3), atol=1e-12)

    def test_no_scale(self, rng):
        """With scale=False, stds are one (no scaling applied)."""
        X = rng.standard_normal((30, 3)) * 50
        X_std, means, stds = standardize_features(X, center=True, scale=False)
        np.testing.assert_allclose(stds, np.ones(3), atol=1e-12)

    def test_constant_column_safe(self):
        """A constant column does not cause division by zero."""
        X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        X_std, means, stds = standardize_features(X)
        assert np.all(np.isfinite(X_std))

    def test_returns_correct_shapes(self, rng):
        """Returned means and stds have shape (p,)."""
        X = rng.standard_normal((20, 6))
        X_std, means, stds = standardize_features(X)
        assert X_std.shape == (20, 6)
        assert means.shape == (6,)
        assert stds.shape == (6,)


# =========================================================================
#                     TestRidgeEffectiveDf
# =========================================================================

class TestRidgeEffectiveDf:
    """Tests for ridge_effective_df."""

    def test_zero_lambda_equals_p(self, rng):
        """With lambda=0, df equals p."""
        X = rng.standard_normal((40, 6))
        df = ridge_effective_df(X, lambda_=0.0)
        assert np.isclose(df, 6.0, atol=1e-8)

    def test_large_lambda_near_zero(self, rng):
        """With very large lambda, df approaches zero."""
        X = rng.standard_normal((40, 6))
        df = ridge_effective_df(X, lambda_=1e12)
        assert df < 0.01

    def test_bounded(self, rng):
        """df is always between 0 and p."""
        X = rng.standard_normal((40, 7))
        df = ridge_effective_df(X, lambda_=2.0)
        assert 0 <= df <= 7.0

    def test_monotone_decreasing(self, rng):
        """df decreases as lambda increases."""
        X = rng.standard_normal((40, 5))
        df1 = ridge_effective_df(X, lambda_=0.01)
        df2 = ridge_effective_df(X, lambda_=10.0)
        assert df1 > df2

    def test_identity_design(self):
        """For X = I_p, df = p * 1/(1+lambda)."""
        p = 5
        X = np.eye(p)
        lam = 3.0
        df = ridge_effective_df(X, lambda_=lam)
        expected = p * 1.0 / (1.0 + lam)
        assert np.isclose(df, expected, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
