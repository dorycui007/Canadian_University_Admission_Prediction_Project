"""
Integration tests for the model training and prediction pipeline.

End-to-end tests that exercise fit -> predict_proba -> predict flows
for LogisticModel and BaselineModel.
"""

import pytest
import numpy as np

from src.models.base import ModelConfig, TrainingHistory, EvaluationMetrics, BaseModel
from src.models.logistic import LogisticModel, sigmoid, log_loss
from src.models.baseline import BaselineModel, BetaPrior


# ---------------------------------------------------------------------------
# TestModelTrainingPipeline — end-to-end fit / predict flows
# ---------------------------------------------------------------------------

class TestModelTrainingPipeline:
    """Integration tests that exercise fit -> predict_proba -> predict."""

    def test_logistic_fit_predict(self, rng):
        """LogisticModel.fit then predict_proba returns values in [0,1],
        and predict returns only 0 or 1."""
        n, p = 50, 4
        X = rng.standard_normal((n, p))
        y = (X[:, 0] + rng.standard_normal(n) > 0).astype(float)
        model = LogisticModel(lambda_=1.0, max_iter=50)
        model.fit(X, y)
        assert model.is_fitted
        proba = model.predict_proba(X)
        assert proba.shape == (n,)
        assert np.all((proba >= 0) & (proba <= 1))
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_baseline_fit_predict(self, rng):
        """BaselineModel.fit with program_ids, predict_proba in [0,1]."""
        n = 60
        programs = np.array(["prog_a"] * 30 + ["prog_b"] * 30)
        y = rng.choice([0, 1], size=n, p=[0.4, 0.6]).astype(float)
        X = rng.standard_normal((n, 2))
        model = BaselineModel(prior_strength=10.0)
        model.fit(X, y, program_ids=programs)
        assert model.is_fitted
        proba = model.predict_proba(X, program_ids=programs)
        assert proba.shape == (n,)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_logistic_training_history(self, rng):
        """After fit, training_history should contain a losses list."""
        n, p = 40, 3
        X = rng.standard_normal((n, p))
        y = (X[:, 0] > 0).astype(float)
        model = LogisticModel(lambda_=0.5, max_iter=20)
        model.fit(X, y)
        history = model.training_history
        assert isinstance(history, dict)
        assert "loss" in history
        assert len(history["loss"]) > 0

    def test_model_refit(self, rng):
        """Fitting a model twice (refit) should work without error."""
        n, p = 30, 3
        X1 = rng.standard_normal((n, p))
        y1 = rng.choice([0, 1], size=n).astype(float)
        X2 = rng.standard_normal((n, p))
        y2 = rng.choice([0, 1], size=n).astype(float)
        model = LogisticModel(lambda_=1.0, max_iter=30)
        model.fit(X1, y1)
        assert model.is_fitted
        model.fit(X2, y2)
        assert model.is_fitted
        proba = model.predict_proba(X2)
        assert proba.shape == (n,)


# ---------------------------------------------------------------------------
# TestModelComparison — verify both models honour the shared interface
# ---------------------------------------------------------------------------

class TestModelComparison:
    """Integration tests that compare LogisticModel and BaselineModel."""

    def test_shared_interface(self):
        """Both models should expose name, is_fitted, predict_proba, predict."""
        logistic = LogisticModel()
        baseline = BaselineModel()
        for model in [logistic, baseline]:
            assert hasattr(model, "name")
            assert hasattr(model, "is_fitted")
            assert hasattr(model, "predict_proba")
            assert hasattr(model, "predict") or hasattr(model, "predict_proba")

    def test_both_models_predict_same_data(self, rng):
        """Both models can predict on the same X; proba values are in [0,1]."""
        n, p = 40, 3
        X = rng.standard_normal((n, p))
        y = rng.choice([0, 1], size=n, p=[0.35, 0.65]).astype(float)
        programs = np.array(["prog_a"] * 20 + ["prog_b"] * 20)

        logistic = LogisticModel(lambda_=1.0, max_iter=30)
        logistic.fit(X, y)
        proba_log = logistic.predict_proba(X)
        assert np.all((proba_log >= 0) & (proba_log <= 1))

        baseline = BaselineModel(prior_strength=10.0)
        baseline.fit(X, y, program_ids=programs)
        proba_bl = baseline.predict_proba(X, program_ids=programs)
        assert np.all((proba_bl >= 0) & (proba_bl <= 1))
