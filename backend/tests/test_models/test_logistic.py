"""
Unit Tests for Logistic Regression Module
==========================================

Tests for src/models/logistic.py — sigmoid, log_loss, compute_gradient,
compute_hessian, irls_step, and LogisticModel.
"""

import pytest
import numpy as np

from src.models.logistic import (
    IRLSConvergenceError,
    sigmoid,
    log_loss,
    compute_gradient,
    compute_hessian,
    irls_step,
    LogisticModel,
)


# =============================================================================
#                        SIGMOID FUNCTION TESTS
# =============================================================================

class TestSigmoid:
    """Tests for the numerically stable sigmoid function."""

    def test_at_zero(self):
        """sigmoid(0) = 0.5."""
        assert np.isclose(sigmoid(np.array(0.0)), 0.5)

    def test_symmetry(self):
        """sigmoid(-x) = 1 - sigmoid(x)."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        assert np.allclose(sigmoid(-x), 1 - sigmoid(x), atol=1e-12)

    def test_range(self):
        """sigmoid output is always in (0, 1)."""
        x = np.linspace(-100, 100, 1000)
        y = sigmoid(x)
        assert np.all(y > 0)
        assert np.all(y < 1)

    def test_monotonic(self):
        """sigmoid is monotonically increasing."""
        x = np.linspace(-10, 10, 100)
        y = sigmoid(x)
        assert np.all(np.diff(y) > 0)

    def test_limits(self):
        """sigmoid(large) -> 1, sigmoid(-large) -> 0."""
        assert np.isclose(sigmoid(np.array(100.0)), 1.0, atol=1e-10)
        assert np.isclose(sigmoid(np.array(-100.0)), 0.0, atol=1e-10)

    def test_numerical_stability(self):
        """No overflow or NaN for very large inputs."""
        large_pos = sigmoid(np.array(1000.0))
        large_neg = sigmoid(np.array(-1000.0))
        assert np.isfinite(large_pos)
        assert np.isfinite(large_neg)
        assert np.isclose(large_pos, 1.0)
        assert np.isclose(large_neg, 0.0)

    def test_vectorized(self):
        """Works on multi-dimensional arrays."""
        x = np.array([[-1.0, 0.0], [1.0, 2.0]])
        y = sigmoid(x)
        assert y.shape == x.shape
        assert np.isclose(y[0, 1], 0.5)


# =============================================================================
#                        LOG LOSS TESTS
# =============================================================================

class TestLogLoss:
    """Tests for log_loss (negative log-likelihood / cross-entropy)."""

    def test_perfect_prediction_low_loss(self):
        """Near-perfect predictions give near-zero loss."""
        y = np.array([1.0, 0.0])
        p = np.array([1.0 - 1e-10, 1e-10])
        loss = log_loss(y, p)
        assert loss < 1e-8

    def test_wrong_prediction_high_loss(self):
        """Confident wrong predictions give high loss."""
        y = np.array([1.0, 0.0])
        p = np.array([0.01, 0.99])
        loss = log_loss(y, p)
        assert loss > 3.0

    def test_non_negative(self):
        """Loss is always non-negative."""
        rng = np.random.default_rng(42)
        y = rng.choice([0.0, 1.0], size=50)
        p = rng.uniform(0.01, 0.99, size=50)
        loss = log_loss(y, p)
        assert loss >= 0

    def test_uncertain_prediction(self):
        """p=0.5 gives -log(0.5) = log(2) per sample."""
        y = np.array([1.0])
        p = np.array([0.5])
        loss = log_loss(y, p)
        assert np.isclose(loss, np.log(2), atol=1e-10)

    def test_baseline_loss(self):
        """Random guessing (p=0.5 for all) gives log(2) ≈ 0.693."""
        y = np.array([1.0, 0.0, 1.0, 0.0])
        p = np.array([0.5, 0.5, 0.5, 0.5])
        loss = log_loss(y, p)
        assert np.isclose(loss, np.log(2), atol=1e-10)


# =============================================================================
#                        GRADIENT TESTS
# =============================================================================

class TestComputeGradient:
    """Tests for compute_gradient of regularized log loss."""

    def test_shape(self, binary_classification_data):
        """Gradient has shape (p,) matching number of features."""
        X, y = binary_classification_data
        n_features = X.shape[1]
        beta = np.zeros(n_features)
        p = sigmoid(X @ beta)
        grad = compute_gradient(X, y, p, beta, lambda_=0.0)
        assert grad.shape == (n_features,)

    def test_numerical_gradient_check(self, binary_classification_data):
        """Analytical gradient matches numerical gradient."""
        X, y = binary_classification_data
        rng = np.random.default_rng(123)
        beta = rng.standard_normal(X.shape[1]) * 0.1
        p = sigmoid(X @ beta)
        analytical_grad = compute_gradient(X, y, p, beta, lambda_=0.1)

        # Numerical gradient via central differences
        eps = 1e-5
        numerical_grad = np.zeros_like(beta)
        for i in range(len(beta)):
            beta_plus = beta.copy()
            beta_plus[i] += eps
            beta_minus = beta.copy()
            beta_minus[i] -= eps
            p_plus = sigmoid(X @ beta_plus)
            p_minus = sigmoid(X @ beta_minus)
            loss_plus = log_loss(y, p_plus) + 0.5 * 0.1 * np.sum(beta_plus ** 2) / len(y)
            loss_minus = log_loss(y, p_minus) + 0.5 * 0.1 * np.sum(beta_minus ** 2) / len(y)
            numerical_grad[i] = (loss_plus - loss_minus) / (2 * eps)

        # Scale: compute_gradient returns un-averaged form X^T(p-y) + lambda*beta
        # Compare directions at minimum
        assert np.allclose(
            analytical_grad / np.linalg.norm(analytical_grad),
            numerical_grad / np.linalg.norm(numerical_grad),
            atol=0.1,
        )

    def test_zero_regularization(self, binary_classification_data):
        """With lambda=0, gradient is X^T(p-y)."""
        X, y = binary_classification_data
        beta = np.zeros(X.shape[1])
        p = sigmoid(X @ beta)
        grad = compute_gradient(X, y, p, beta, lambda_=0.0)
        expected = X.T @ (p - y)
        assert np.allclose(grad, expected, atol=1e-10)


# =============================================================================
#                        HESSIAN TESTS
# =============================================================================

class TestComputeHessian:
    """Tests for compute_hessian of regularized log loss."""

    def test_shape(self, binary_classification_data):
        """Hessian has shape (p, p)."""
        X, y = binary_classification_data
        n_features = X.shape[1]
        p = np.full(X.shape[0], 0.5)
        H = compute_hessian(X, p, lambda_=0.0)
        assert H.shape == (n_features, n_features)

    def test_symmetric(self, binary_classification_data):
        """Hessian is symmetric."""
        X, y = binary_classification_data
        p = sigmoid(X @ np.zeros(X.shape[1]))
        H = compute_hessian(X, p, lambda_=0.1)
        assert np.allclose(H, H.T, atol=1e-10)

    def test_positive_semi_definite(self, binary_classification_data):
        """Hessian is positive semi-definite (convex loss)."""
        X, y = binary_classification_data
        rng = np.random.default_rng(42)
        beta = rng.standard_normal(X.shape[1]) * 0.1
        p = sigmoid(X @ beta)
        H = compute_hessian(X, p, lambda_=0.1)
        eigenvalues = np.linalg.eigvalsh(H)
        assert np.all(eigenvalues >= -1e-10)


# =============================================================================
#                        IRLS STEP TESTS
# =============================================================================

class TestIRLSStep:
    """Tests for a single IRLS iteration."""

    def test_reduces_loss(self, binary_classification_data):
        """One IRLS step should reduce (or not increase) the loss."""
        X, y = binary_classification_data
        beta_init = np.zeros(X.shape[1])
        p_init = sigmoid(X @ beta_init)
        loss_init = log_loss(y, p_init)

        beta_new = irls_step(X, y, beta_init, lambda_=0.1)
        p_new = sigmoid(X @ beta_new)
        loss_new = log_loss(y, p_new)

        assert loss_new <= loss_init + 1e-6

    def test_output_shape(self, binary_classification_data):
        """IRLS step returns coefficients of correct shape."""
        X, y = binary_classification_data
        beta = np.zeros(X.shape[1])
        beta_new = irls_step(X, y, beta, lambda_=0.1)
        assert beta_new.shape == beta.shape

    def test_finite_output(self, binary_classification_data):
        """IRLS step returns finite values."""
        X, y = binary_classification_data
        beta = np.zeros(X.shape[1])
        beta_new = irls_step(X, y, beta, lambda_=0.1)
        assert np.all(np.isfinite(beta_new))


# =============================================================================
#                        LOGISTICMODEL TESTS
# =============================================================================

class TestLogisticModel:
    """Tests for the LogisticModel class."""

    def test_init(self):
        """Model initializes with correct defaults."""
        model = LogisticModel()
        assert model.is_fitted is False
        assert model.coefficients is None

    def test_custom_init(self):
        """Custom hyperparameters are stored."""
        model = LogisticModel(lambda_=0.5, max_iter=50, tol=1e-4)
        assert model.lambda_ == 0.5
        assert model.max_iter == 50
        assert model.tol == 1e-4

    def test_fit_returns_self(self, binary_classification_data):
        """fit() returns self for method chaining."""
        X, y = binary_classification_data
        model = LogisticModel(lambda_=0.1, max_iter=20)
        result = model.fit(X, y)
        assert result is model

    def test_fit_sets_fitted(self, binary_classification_data):
        """is_fitted is True after fit()."""
        X, y = binary_classification_data
        model = LogisticModel(lambda_=0.1, max_iter=20)
        model.fit(X, y)
        assert model.is_fitted is True

    def test_coefficients_stored(self, binary_classification_data):
        """Coefficients are stored after fit."""
        X, y = binary_classification_data
        model = LogisticModel(lambda_=0.1, max_iter=20)
        model.fit(X, y)
        assert model.coefficients is not None
        # With fit_intercept=True, expect n_features + 1 coefficients
        expected_dim = X.shape[1] + 1 if model.fit_intercept else X.shape[1]
        assert len(model.coefficients) == expected_dim

    def test_predict_proba_in_range(self, binary_classification_data):
        """Predicted probabilities are in [0, 1]."""
        X, y = binary_classification_data
        model = LogisticModel(lambda_=0.1, max_iter=50)
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_predict_binary(self, binary_classification_data):
        """predict() returns 0 or 1."""
        X, y = binary_classification_data
        model = LogisticModel(lambda_=0.1, max_iter=50)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_matches_threshold(self, binary_classification_data):
        """predict() matches thresholded predict_proba()."""
        X, y = binary_classification_data
        model = LogisticModel(lambda_=0.1, max_iter=50)
        model.fit(X, y)
        probs = model.predict_proba(X)
        preds = model.predict(X)
        expected = (probs >= 0.5).astype(int)
        assert np.array_equal(preds, expected)

    def test_predict_proba_before_fit_raises(self):
        """predict_proba() before fit raises RuntimeError."""
        model = LogisticModel()
        X = np.random.randn(10, 3)
        with pytest.raises(RuntimeError):
            model.predict_proba(X)

    def test_name(self):
        """Model has a name string."""
        model = LogisticModel()
        assert isinstance(model.name, str)
        assert len(model.name) > 0


# =============================================================================
#                        CONVERGENCE TESTS
# =============================================================================

class TestConvergence:
    """Tests for IRLS convergence behavior."""

    def test_loss_decreases(self, binary_classification_data):
        """Loss should decrease across iterations."""
        X, y = binary_classification_data
        model = LogisticModel(lambda_=0.1, max_iter=20)
        model.fit(X, y)
        losses = model.training_history.get('loss', [])
        if len(losses) > 1:
            for i in range(1, len(losses)):
                assert losses[i] <= losses[i - 1] + 1e-6

    def test_converges_before_max_iter(self, binary_classification_data):
        """Model should converge in fewer than max_iter iterations."""
        X, y = binary_classification_data
        model = LogisticModel(lambda_=0.1, max_iter=100)
        model.fit(X, y)
        assert model.n_iter_ < 100


# =============================================================================
#                        REGULARIZATION TESTS
# =============================================================================

class TestRegularization:
    """Tests for ridge regularization effects."""

    def test_larger_lambda_smaller_coefficients(self, binary_classification_data):
        """Larger lambda produces smaller coefficient norms."""
        X, y = binary_classification_data
        model_small = LogisticModel(lambda_=0.01, max_iter=50)
        model_small.fit(X, y)

        model_large = LogisticModel(lambda_=10.0, max_iter=50)
        model_large.fit(X, y)

        norm_small = np.linalg.norm(model_small.coefficients)
        norm_large = np.linalg.norm(model_large.coefficients)
        assert norm_large < norm_small

    def test_very_large_lambda_near_zero(self, binary_classification_data):
        """Very large lambda shrinks coefficients toward zero."""
        X, y = binary_classification_data
        model = LogisticModel(lambda_=1e4, max_iter=50)
        model.fit(X, y)
        assert np.linalg.norm(model.coefficients) < 0.5


# =============================================================================
#                        EXCEPTION TESTS
# =============================================================================

class TestIRLSConvergenceError:
    """Tests for the custom exception class."""

    def test_instantiation(self):
        """IRLSConvergenceError can be instantiated."""
        err = IRLSConvergenceError("Did not converge")
        assert str(err) == "Did not converge"

    def test_inherits_from_exception(self):
        """IRLSConvergenceError inherits from Exception."""
        assert issubclass(IRLSConvergenceError, Exception)


# =============================================================================
#                    STUB COVERAGE TESTS
# =============================================================================

class TestLogisticProperties:
    """Tests for LogisticModel properties and utility methods."""

    def test_is_fitted_false_before_fit(self):
        """is_fitted is False before fitting."""
        model = LogisticModel()
        assert model.is_fitted is False

    def test_n_features_none_before_fit(self):
        """n_features is None before fitting."""
        model = LogisticModel()
        assert model.n_features is None

    def test_n_features_after_fit(self, binary_classification_data):
        """n_features reflects coefficient count after fitting."""
        X, y = binary_classification_data
        model = LogisticModel(lambda_=0.1, max_iter=20)
        model.fit(X, y)
        # With fit_intercept=True, n_features = X.shape[1] + 1
        assert model.n_features == X.shape[1] + 1

    def test_n_params_zero_before_fit(self):
        """n_params is 0 before fitting."""
        model = LogisticModel()
        assert model.n_params == 0

    def test_n_params_after_fit(self, binary_classification_data):
        """n_params matches coefficient count after fitting."""
        X, y = binary_classification_data
        model = LogisticModel(lambda_=0.1, max_iter=20)
        model.fit(X, y)
        assert model.n_params == X.shape[1] + 1

    def test_add_intercept(self):
        """_add_intercept prepends a column of ones."""
        model = LogisticModel()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = model._add_intercept(X)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result[:, 0], np.ones(2))
        np.testing.assert_array_equal(result[:, 1:], X)

    def test_get_feature_importance_unfitted_returns_none(self):
        """get_feature_importance returns None when model is not fitted."""
        model = LogisticModel()
        assert model.get_feature_importance() is None

    def test_get_feature_importance_fitted(self, binary_classification_data):
        """get_feature_importance returns a dict sorted by |coefficient|."""
        X, y = binary_classification_data
        model = LogisticModel(lambda_=0.1, max_iter=50)
        model.fit(X, y)
        importance = model.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) == X.shape[1] + 1  # includes intercept
        # Values should be descending (sorted by importance)
        vals = list(importance.values())
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1]

    def test_get_odds_ratios_unfitted_returns_none(self):
        """get_odds_ratios returns None when model is not fitted."""
        model = LogisticModel()
        assert model.get_odds_ratios() is None

    def test_get_odds_ratios_fitted(self, binary_classification_data):
        """get_odds_ratios returns exp(coefficients)."""
        X, y = binary_classification_data
        model = LogisticModel(lambda_=0.1, max_iter=50)
        model.fit(X, y)
        odds = model.get_odds_ratios()
        assert isinstance(odds, dict)
        assert len(odds) == X.shape[1] + 1
        # All odds ratios should be positive
        assert all(v > 0 for v in odds.values())

    def test_get_params_returns_dict(self):
        """get_params returns a dict with expected keys."""
        model = LogisticModel(lambda_=0.5, max_iter=50)
        params = model.get_params()
        assert isinstance(params, dict)
        assert 'lambda_' in params
        assert params['lambda_'] == 0.5
        assert params['max_iter'] == 50

    def test_set_params_restores_state(self):
        """set_params restores model coefficients and settings."""
        model = LogisticModel()
        model.set_params({
            'coefficients': [0.1, 0.2, 0.3],
            'lambda_': 0.5,
        })
        assert model._is_fitted is True
        assert model.lambda_ == 0.5
        np.testing.assert_array_equal(model.coefficients, np.array([0.1, 0.2, 0.3]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
