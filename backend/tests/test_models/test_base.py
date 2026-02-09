"""
Unit Tests for Abstract Base Model Module
==========================================

Tests for src/models/base.py — ModelConfig, TrainingHistory,
EvaluationMetrics dataclasses, and BaseModel ABC.
"""

import pytest
import numpy as np
from dataclasses import fields

from src.models.base import (
    ModelConfig,
    TrainingHistory,
    EvaluationMetrics,
    BaseModel,
)


# =============================================================================
#                    MODELCONFIG TESTS
# =============================================================================

class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        """Default config has expected values."""
        config = ModelConfig()
        assert config.lambda_ == 0.1
        assert config.max_iter == 100
        assert config.tol == 1e-6
        assert config.random_state is None
        assert config.extra_params == {}

    def test_custom_values(self):
        """Custom values are stored correctly."""
        config = ModelConfig(lambda_=0.5, max_iter=200, tol=1e-8, random_state=42)
        assert config.lambda_ == 0.5
        assert config.max_iter == 200
        assert config.tol == 1e-8
        assert config.random_state == 42

    def test_extra_params(self):
        """Extra params dict works correctly."""
        config = ModelConfig(extra_params={'solver': 'qr', 'verbose': True})
        assert config.extra_params['solver'] == 'qr'
        assert config.extra_params['verbose'] is True

    def test_has_expected_fields(self):
        """ModelConfig has exactly the expected fields."""
        field_names = {f.name for f in fields(ModelConfig)}
        expected = {'lambda_', 'max_iter', 'tol', 'random_state', 'extra_params'}
        assert field_names == expected

    def test_extra_params_default_independent(self):
        """Each instance gets its own extra_params dict."""
        c1 = ModelConfig()
        c2 = ModelConfig()
        c1.extra_params['key'] = 'val'
        assert 'key' not in c2.extra_params


# =============================================================================
#                    TRAININGHISTORY TESTS
# =============================================================================

class TestTrainingHistory:
    """Tests for TrainingHistory dataclass."""

    def test_default_values(self):
        """Default history has empty lists and False converged."""
        history = TrainingHistory()
        assert history.losses == []
        assert history.gradient_norms == []
        assert history.timestamps == []
        assert history.converged is False
        assert history.final_iteration == 0

    def test_append_loss(self):
        """Can append to losses list."""
        history = TrainingHistory()
        history.losses.append(0.693)
        history.losses.append(0.500)
        assert len(history.losses) == 2

    def test_converged_flag(self):
        """Converged flag can be set."""
        history = TrainingHistory(converged=True, final_iteration=25)
        assert history.converged is True
        assert history.final_iteration == 25

    def test_lists_default_independent(self):
        """Each instance gets its own lists."""
        h1 = TrainingHistory()
        h2 = TrainingHistory()
        h1.losses.append(1.0)
        assert len(h2.losses) == 0


# =============================================================================
#                    EVALUATIONMETRICS TESTS
# =============================================================================

class TestEvaluationMetrics:
    """Tests for EvaluationMetrics dataclass."""

    def test_default_values(self):
        """Default metrics are all 0.0."""
        metrics = EvaluationMetrics()
        assert metrics.roc_auc == 0.0
        assert metrics.pr_auc == 0.0
        assert metrics.brier_score == 0.0
        assert metrics.ece == 0.0
        assert metrics.mce == 0.0
        assert metrics.log_loss == 0.0
        assert metrics.accuracy == 0.0

    def test_custom_values(self):
        """Custom metric values stored correctly."""
        metrics = EvaluationMetrics(roc_auc=0.85, brier_score=0.15)
        assert metrics.roc_auc == 0.85
        assert metrics.brier_score == 0.15

    def test_has_expected_fields(self):
        """EvaluationMetrics has all expected metric fields."""
        field_names = {f.name for f in fields(EvaluationMetrics)}
        expected = {'roc_auc', 'pr_auc', 'brier_score', 'ece', 'mce',
                    'log_loss', 'accuracy'}
        assert field_names == expected

    def test_to_dict(self):
        """to_dict() returns a dictionary with all metrics."""
        metrics = EvaluationMetrics(roc_auc=0.8, accuracy=0.75)
        d = metrics.to_dict()
        assert isinstance(d, dict)
        assert d['roc_auc'] == 0.8
        assert d['accuracy'] == 0.75

    def test_str_representation(self):
        """__str__() returns a non-empty string."""
        metrics = EvaluationMetrics(roc_auc=0.8)
        s = str(metrics)
        assert isinstance(s, str)
        assert len(s) > 0


# =============================================================================
#                    BASEMODEL ABC TESTS
# =============================================================================

class TestBaseModelABC:
    """Tests for BaseModel abstract base class enforcement."""

    def test_cannot_instantiate_directly(self):
        """BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel()

    def test_subclass_must_implement_name(self):
        """Subclass without 'name' property raises TypeError."""
        class IncompleteModel(BaseModel):
            def fit(self, X, y, sample_weight=None): return self
            def predict_proba(self, X): return np.zeros(len(X))
            def get_params(self): return {}
            def set_params(self, params): pass

            @property
            def n_params(self): return 0

        with pytest.raises(TypeError):
            IncompleteModel()

    def test_subclass_must_implement_fit(self):
        """Subclass without 'fit' method raises TypeError."""
        class IncompleteModel(BaseModel):
            @property
            def name(self): return "test"
            def predict_proba(self, X): return np.zeros(len(X))
            def get_params(self): return {}
            def set_params(self, params): pass

            @property
            def n_params(self): return 0

        with pytest.raises(TypeError):
            IncompleteModel()

    def test_subclass_must_implement_predict_proba(self):
        """Subclass without 'predict_proba' raises TypeError."""
        class IncompleteModel(BaseModel):
            @property
            def name(self): return "test"
            def fit(self, X, y, sample_weight=None): return self
            def get_params(self): return {}
            def set_params(self, params): pass

            @property
            def n_params(self): return 0

        with pytest.raises(TypeError):
            IncompleteModel()

    def test_complete_subclass_instantiates(self):
        """Fully implemented subclass can be instantiated."""
        class DummyModel(BaseModel):
            @property
            def name(self): return "Dummy"

            @property
            def n_params(self): return 0

            def fit(self, X, y, sample_weight=None):
                self._is_fitted = True
                return self

            def predict_proba(self, X):
                return np.full(X.shape[0], 0.5)

            def get_params(self): return {}
            def set_params(self, params): pass

        model = DummyModel()
        assert model.name == "Dummy"


# =============================================================================
#                    BASEMODEL CONCRETE METHODS TESTS
# =============================================================================

class TestBaseModelConcreteMethods:
    """Tests for BaseModel's concrete (non-abstract) methods using a DummyModel."""

    @pytest.fixture
    def dummy_model(self):
        """Create a minimal concrete subclass for testing."""
        class DummyModel(BaseModel):
            @property
            def name(self): return "Dummy"

            @property
            def n_params(self): return 3

            def fit(self, X, y, sample_weight=None):
                self._is_fitted = True
                self._n_features = X.shape[1]
                self._coefs = np.zeros(X.shape[1])
                return self

            def predict_proba(self, X):
                return np.full(X.shape[0], 0.5)

            def get_params(self): return {'coefs': self._coefs if hasattr(self, '_coefs') else None}
            def set_params(self, params): self._coefs = params.get('coefs')

        return DummyModel()

    def test_is_fitted_before_fit(self, dummy_model):
        """is_fitted is False before fit() is called."""
        assert dummy_model.is_fitted is False

    def test_is_fitted_after_fit(self, dummy_model, rng):
        """is_fitted is True after fit() is called."""
        X = rng.standard_normal((10, 3))
        y = rng.choice([0, 1], size=10).astype(float)
        dummy_model.fit(X, y)
        assert dummy_model.is_fitted is True

    def test_predict_thresholds(self, dummy_model, rng):
        """predict() thresholds probabilities at 0.5."""
        X = rng.standard_normal((10, 3))
        y = rng.choice([0, 1], size=10).astype(float)
        dummy_model.fit(X, y)
        preds = dummy_model.predict(X)
        # All probabilities are 0.5, so predictions depend on >= convention
        assert all(p in (0, 1) for p in preds)

    def test_n_features_before_fit(self, dummy_model):
        """n_features is None before fitting."""
        assert dummy_model.n_features is None

    def test_summary_returns_string(self, dummy_model, rng):
        """summary() returns a string."""
        X = rng.standard_normal((10, 3))
        y = rng.choice([0, 1], size=10).astype(float)
        dummy_model.fit(X, y)
        s = dummy_model.summary()
        assert isinstance(s, str)


# =============================================================================
#                    STUB COVERAGE TESTS
# =============================================================================

class TestBaseModelConcreteMethodsDetailed:
    """Detailed tests for BaseModel's concrete methods using DummyModel."""

    @pytest.fixture
    def dummy_model(self):
        """Create a minimal concrete subclass for testing."""
        class DummyModel(BaseModel):
            @property
            def name(self): return "Dummy"

            @property
            def n_params(self): return 3

            def fit(self, X, y, sample_weight=None):
                self._is_fitted = True
                self._n_features = X.shape[1]
                return self

            def predict_proba(self, X):
                return np.full(X.shape[0], 0.5)

            def get_params(self): return {'dummy': True}
            def set_params(self, params): pass

        return DummyModel()

    def test_evaluate_returns_metrics(self, dummy_model, rng):
        """evaluate() returns an EvaluationMetrics object with valid values."""
        X = rng.standard_normal((10, 3))
        y = rng.choice([0, 1], size=10).astype(float)
        dummy_model.fit(X, y)
        result = dummy_model.evaluate(X, y)
        assert isinstance(result, EvaluationMetrics)
        assert result.brier_score >= 0
        assert result.accuracy >= 0
        assert result.log_loss >= 0

    def test_validate_input_valid_X(self, dummy_model, rng):
        """validate_input accepts valid 2D array X."""
        X = rng.standard_normal((10, 3))
        dummy_model.validate_input(X)  # should not raise

    def test_validate_input_invalid_X_1d(self, dummy_model):
        """validate_input rejects 1D array."""
        with pytest.raises(ValueError):
            dummy_model.validate_input(np.array([1, 2, 3]))

    def test_validate_input_invalid_type(self, dummy_model):
        """validate_input rejects non-ndarray."""
        with pytest.raises(TypeError):
            dummy_model.validate_input([[1, 2], [3, 4]])

    def test_validate_input_with_y_valid(self, dummy_model, rng):
        """validate_input accepts matching X and y."""
        X = rng.standard_normal((10, 3))
        y = rng.choice([0, 1], size=10).astype(float)
        dummy_model.validate_input(X, y)  # should not raise

    def test_validate_input_shape_mismatch(self, dummy_model, rng):
        """validate_input rejects mismatched X and y lengths."""
        X = rng.standard_normal((10, 3))
        y = rng.choice([0, 1], size=5).astype(float)
        with pytest.raises(ValueError):
            dummy_model.validate_input(X, y)

    def test_validate_input_nan_x(self, dummy_model):
        """validate_input rejects X with NaN."""
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        with pytest.raises(ValueError):
            dummy_model.validate_input(X)

    def test_save_creates_file(self, dummy_model, rng, tmp_path):
        """save() creates a JSON file with model data."""
        X = rng.standard_normal((10, 3))
        y = rng.choice([0, 1], size=10).astype(float)
        dummy_model.fit(X, y)
        path = str(tmp_path / "model.json")
        dummy_model.save(path)
        import json
        with open(path) as f:
            data = json.load(f)
        assert data['model_class'] == 'DummyModel'
        assert 'params' in data
        assert 'saved_at' in data

    def test_save_unfitted_raises(self, dummy_model):
        """save() raises RuntimeError if model is not fitted."""
        with pytest.raises(RuntimeError):
            dummy_model.save("/tmp/should_not_exist.json")

    def test_load_base_returns_none(self):
        """BaseModel.load returns None (abstract — subclasses should override)."""
        result = BaseModel.load("/tmp/nonexistent.json")
        assert result is None

    def test_predict_returns_binary(self, dummy_model, rng):
        """predict() returns binary predictions."""
        X = rng.standard_normal((10, 3))
        y = rng.choice([0, 1], size=10).astype(float)
        dummy_model.fit(X, y)
        preds = dummy_model.predict(X)
        assert preds.shape == (10,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_summary_unfitted_contains_not_fitted(self, dummy_model):
        """summary() on unfitted model includes 'Not fitted'."""
        s = dummy_model.summary()
        assert isinstance(s, str)
        assert "Not fitted" in s

    def test_summary_fitted_contains_model_name(self, dummy_model, rng):
        """summary() on fitted model includes model name."""
        X = rng.standard_normal((10, 3))
        y = rng.choice([0, 1], size=10).astype(float)
        dummy_model.fit(X, y)
        s = dummy_model.summary()
        assert "Dummy" in s
        assert "Fitted" in s


class TestEvaluationMetricsDetailed:
    """Detailed tests for EvaluationMetrics methods."""

    def test_to_dict_returns_all_keys(self):
        """to_dict() returns a dict with all metric keys."""
        metrics = EvaluationMetrics(roc_auc=0.8, accuracy=0.75)
        d = metrics.to_dict()
        assert isinstance(d, dict)
        assert d['roc_auc'] == 0.8
        assert d['accuracy'] == 0.75
        assert 'brier_score' in d
        assert 'ece' in d

    def test_str_contains_metric_names(self):
        """__str__() includes metric names and values."""
        metrics = EvaluationMetrics(roc_auc=0.8, brier_score=0.15)
        s = str(metrics)
        assert "ROC-AUC" in s
        assert "0.8000" in s
        assert "Brier" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
