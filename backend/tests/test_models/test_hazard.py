"""
Unit Tests for Hazard Model Module
====================================

Tests for src/models/hazard.py -- TimingPrediction dataclass,
expand_to_person_period, create_time_dummies, and HazardModel.

All source functions are implemented. Tests verify return types, shapes,
value ranges, and expected mathematical properties.
"""

import pytest
import numpy as np
from dataclasses import fields

from src.models.hazard import (
    TimingPrediction,
    expand_to_person_period,
    create_time_dummies,
    HazardModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)


def _fit_hazard_model(max_time=10, n_samples=50, n_features=3, lambda_=1.0):
    """Helper to create and fit a hazard model on synthetic data."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features))
    # Event times between 1 and max_time
    event_times = rng.integers(1, max_time + 1, size=n_samples)
    model = HazardModel(max_time=max_time, lambda_=lambda_, max_iter=50, tol=1e-4)
    model.fit(X, event_times)
    return model, X, event_times


# =========================================================================
# TestTimingPrediction
# =========================================================================
class TestTimingPrediction:
    """Tests for the TimingPrediction dataclass."""

    def _make_prediction(self):
        return TimingPrediction(
            survival_curve=np.array([1.0, 0.9, 0.8]),
            hazard_curve=np.array([0.1, 0.1, 0.1]),
            event_probs=np.array([0.1, 0.1, 0.1]),
            expected_time=5.0,
            median_time=4.0,
            ci_80_lower=3.0,
            ci_80_upper=7.0,
        )

    def test_fields_exist(self):
        names = {f.name for f in fields(TimingPrediction)}
        expected = {
            "survival_curve", "hazard_curve", "event_probs",
            "expected_time", "median_time", "ci_80_lower", "ci_80_upper",
        }
        assert expected.issubset(names)

    def test_fields_stored(self):
        pred = self._make_prediction()
        assert pred.expected_time == 5.0
        assert pred.median_time == 4.0
        assert pred.ci_80_lower == 3.0
        assert pred.ci_80_upper == 7.0
        np.testing.assert_array_equal(pred.survival_curve, [1.0, 0.9, 0.8])
        np.testing.assert_array_equal(pred.hazard_curve, [0.1, 0.1, 0.1])
        np.testing.assert_array_equal(pred.event_probs, [0.1, 0.1, 0.1])

    def test_summary_returns_string(self):
        pred = self._make_prediction()
        result = pred.summary()
        assert isinstance(result, str)
        assert "5.0" in result
        assert "4.0" in result
        assert "3.0" in result
        assert "7.0" in result

    def test_summary_contains_key_info(self):
        pred = self._make_prediction()
        result = pred.summary()
        assert "Expected" in result or "expected" in result.lower()
        assert "Median" in result or "median" in result.lower()


# =========================================================================
# TestExpandToPersonPeriod
# =========================================================================
class TestExpandToPersonPeriod:
    """Tests for the expand_to_person_period free function."""

    def test_basic_call(self):
        X = np.array([[92.0, 1.0], [88.0, 0.0]])
        event_times = np.array([3, 2])
        X_exp, y_exp, time_indices = expand_to_person_period(X, event_times, max_time=5)
        # 3 rows for first app + 2 rows for second = 5 total
        assert X_exp.shape[0] == 5
        assert y_exp.shape[0] == 5
        assert time_indices.shape[0] == 5

    def test_single_application(self):
        X = np.array([[90.0]])
        event_times = np.array([1])
        X_exp, y_exp, time_indices = expand_to_person_period(X, event_times, max_time=5)
        assert X_exp.shape[0] == 1
        assert y_exp[0] == 1.0

    def test_features_repeated_correctly(self):
        """Each person-period row should have the same features as the original."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        event_times = np.array([3, 2])
        X_exp, y_exp, time_indices = expand_to_person_period(X, event_times, max_time=5)
        # First 3 rows should have features [1.0, 2.0]
        for i in range(3):
            np.testing.assert_array_equal(X_exp[i], [1.0, 2.0])
        # Next 2 rows should have features [3.0, 4.0]
        for i in range(3, 5):
            np.testing.assert_array_equal(X_exp[i], [3.0, 4.0])

    def test_event_indicator_correct(self):
        """y should be 1 only at the event time, 0 otherwise."""
        X = np.array([[1.0], [2.0]])
        event_times = np.array([3, 2])
        X_exp, y_exp, time_indices = expand_to_person_period(X, event_times, max_time=5)
        # First app: event at time 3 -> y=[0, 0, 1]
        assert y_exp[0] == 0.0
        assert y_exp[1] == 0.0
        assert y_exp[2] == 1.0
        # Second app: event at time 2 -> y=[0, 1]
        assert y_exp[3] == 0.0
        assert y_exp[4] == 1.0

    def test_time_indices_correct(self):
        """Time indices should be 1-indexed and match the time period."""
        X = np.array([[1.0]])
        event_times = np.array([4])
        X_exp, y_exp, time_indices = expand_to_person_period(X, event_times, max_time=5)
        np.testing.assert_array_equal(time_indices, [1, 2, 3, 4])

    def test_total_rows(self):
        """Total number of expanded rows should equal sum of event_times."""
        X = np.array([[1.0], [2.0], [3.0]])
        event_times = np.array([2, 5, 3])
        X_exp, y_exp, time_indices = expand_to_person_period(X, event_times, max_time=10)
        assert X_exp.shape[0] == 10  # 2 + 5 + 3


# =========================================================================
# TestCreateTimeDummies
# =========================================================================
class TestCreateTimeDummies:
    """Tests for the create_time_dummies free function."""

    def test_basic_call(self):
        time_indices = np.array([1, 2, 3])
        result = create_time_dummies(time_indices, max_time=5)
        assert result.shape == (3, 5)
        # Row 0 should have a 1 in column 0 (time=1, 0-indexed)
        assert result[0, 0] == 1.0
        # Row 1 should have a 1 in column 1 (time=2)
        assert result[1, 1] == 1.0
        # Row 2 should have a 1 in column 2 (time=3)
        assert result[2, 2] == 1.0

    def test_all_same_time(self):
        time_indices = np.array([2, 2, 2])
        result = create_time_dummies(time_indices, max_time=4)
        assert result.shape == (3, 4)
        np.testing.assert_array_equal(result[:, 1], [1.0, 1.0, 1.0])
        # Other columns should be 0
        np.testing.assert_array_equal(result[:, 0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result[:, 2], [0.0, 0.0, 0.0])

    def test_each_row_sums_to_one(self):
        """Each row should have exactly one 1 (one-hot encoding)."""
        time_indices = np.array([1, 3, 5, 2])
        result = create_time_dummies(time_indices, max_time=5)
        row_sums = result.sum(axis=1)
        np.testing.assert_array_equal(row_sums, np.ones(4))

    def test_max_time_index(self):
        """Time index equal to max_time should set last column."""
        time_indices = np.array([5])
        result = create_time_dummies(time_indices, max_time=5)
        assert result[0, 4] == 1.0  # 0-indexed: column 4

    def test_output_dtype_float(self):
        time_indices = np.array([1, 2])
        result = create_time_dummies(time_indices, max_time=3)
        assert result.dtype == np.float64


# =========================================================================
# TestHazardModel
# =========================================================================
class TestHazardModel:
    """Tests for the HazardModel class."""

    # -- properties --------------------------------------------------------

    def test_name_property(self):
        model = HazardModel(max_time=10, lambda_=0.1)
        result = model.name
        assert isinstance(result, str)
        assert "Hazard" in result

    def test_is_fitted_property(self):
        model = HazardModel(max_time=10)
        assert model.is_fitted is False

    def test_init_stores_params(self):
        model = HazardModel(max_time=15, lambda_=0.5, max_iter=200, tol=1e-8)
        assert model.max_time == 15
        assert model.lambda_ == 0.5
        assert model.max_iter == 200
        assert model.tol == 1e-8
        assert model.baseline_hazards is None
        assert model.covariate_effects is None

    # -- fit ---------------------------------------------------------------

    def test_fit_returns_self(self):
        model, X, event_times = _fit_hazard_model()
        assert model is not None
        assert model.is_fitted is True

    def test_fit_sets_baseline_hazards(self):
        model, _, _ = _fit_hazard_model(max_time=10)
        assert model.baseline_hazards is not None
        assert model.baseline_hazards.shape == (10,)

    def test_fit_sets_covariate_effects(self):
        model, _, _ = _fit_hazard_model(max_time=10, n_features=3)
        assert model.covariate_effects is not None
        assert model.covariate_effects.shape == (3,)

    def test_fit_with_sample_weight(self):
        """Fit should work with sample_weight argument (even if not used internally)."""
        rng = np.random.default_rng(42)
        model = HazardModel(max_time=10, lambda_=1.0, max_iter=50)
        X = rng.standard_normal((20, 3))
        event_times = rng.integers(1, 10, size=20)
        weights = np.ones(20)
        result = model.fit(X, event_times, sample_weight=weights)
        assert result is model
        assert model.is_fitted is True

    # -- predict_hazard ----------------------------------------------------

    def test_predict_hazard_unfitted_returns_none(self):
        model = HazardModel(max_time=10)
        X = RNG.standard_normal((5, 3))
        result = model.predict_hazard(X)
        assert result is None

    def test_predict_hazard_shape(self):
        model, X, _ = _fit_hazard_model(max_time=10, n_features=3)
        result = model.predict_hazard(X[:5])
        assert result.shape == (5, 10)

    def test_predict_hazard_in_zero_one(self):
        """Hazard values should be in (0, 1) since they are sigmoid outputs."""
        model, X, _ = _fit_hazard_model(max_time=10, n_features=3)
        result = model.predict_hazard(X[:5])
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_predict_hazard_with_specific_times(self):
        model, X, _ = _fit_hazard_model(max_time=10, n_features=3)
        times = np.array([1, 5, 10])
        result = model.predict_hazard(X[:5], times=times)
        assert result.shape == (5, 3)

    # -- predict_survival --------------------------------------------------

    def test_predict_survival_unfitted_returns_none(self):
        model = HazardModel(max_time=10)
        X = RNG.standard_normal((5, 3))
        result = model.predict_survival(X)
        assert result is None

    def test_predict_survival_shape(self):
        model, X, _ = _fit_hazard_model(max_time=10, n_features=3)
        result = model.predict_survival(X[:5])
        assert result.shape == (5, 10)

    def test_predict_survival_monotonically_decreasing(self):
        """Survival curves should be monotonically non-increasing."""
        model, X, _ = _fit_hazard_model(max_time=10, n_features=3)
        survival = model.predict_survival(X[:5])
        for i in range(5):
            diffs = np.diff(survival[i])
            assert np.all(diffs <= 1e-10), f"Survival curve {i} is not monotonically decreasing"

    def test_predict_survival_in_zero_one(self):
        """Survival values should be in [0, 1]."""
        model, X, _ = _fit_hazard_model(max_time=10, n_features=3)
        survival = model.predict_survival(X[:5])
        assert np.all(survival >= 0)
        assert np.all(survival <= 1)

    # -- predict_event_prob ------------------------------------------------

    def test_predict_event_prob_unfitted_returns_none(self):
        model = HazardModel(max_time=10)
        X = RNG.standard_normal((5, 3))
        result = model.predict_event_prob(X)
        assert result is None

    def test_predict_event_prob_shape(self):
        model, X, _ = _fit_hazard_model(max_time=10, n_features=3)
        result = model.predict_event_prob(X[:5])
        assert result.shape == (5, 10)

    def test_predict_event_prob_nonnegative(self):
        """Event probabilities should be non-negative."""
        model, X, _ = _fit_hazard_model(max_time=10, n_features=3)
        result = model.predict_event_prob(X[:5])
        assert np.all(result >= 0)

    def test_predict_event_prob_sums_close_to_one(self):
        """Event probabilities across time should sum to approximately 1."""
        model, X, _ = _fit_hazard_model(max_time=20, n_features=3)
        event_probs = model.predict_event_prob(X[:5])
        row_sums = event_probs.sum(axis=1)
        # Should be close to 1, but might be slightly less due to truncation at max_time
        assert np.all(row_sums <= 1.0 + 1e-7)
        assert np.all(row_sums > 0.5), "Event probs sum too small; most mass should be within max_time"

    # -- predict_timing ----------------------------------------------------

    def test_predict_timing_unfitted_returns_none(self):
        model = HazardModel(max_time=10)
        X = RNG.standard_normal((3, 3))
        result = model.predict_timing(X)
        assert result is None

    def test_predict_timing_returns_list_of_predictions(self):
        model, X, _ = _fit_hazard_model(max_time=10, n_features=3)
        result = model.predict_timing(X[:3])
        assert isinstance(result, list)
        assert len(result) == 3
        assert isinstance(result[0], TimingPrediction)

    def test_predict_timing_fields_valid(self):
        model, X, _ = _fit_hazard_model(max_time=10, n_features=3)
        result = model.predict_timing(X[:3])
        for pred in result:
            assert pred.survival_curve.shape == (10,)
            assert pred.hazard_curve.shape == (10,)
            assert pred.event_probs.shape == (10,)
            assert pred.expected_time > 0
            assert pred.median_time >= 1
            assert pred.ci_80_lower >= 1
            assert pred.ci_80_upper >= pred.ci_80_lower

    # -- predict_expected_time ---------------------------------------------

    def test_predict_expected_time_unfitted_returns_none(self):
        model = HazardModel(max_time=10)
        X = RNG.standard_normal((5, 3))
        result = model.predict_expected_time(X)
        assert result is None

    def test_predict_expected_time_shape(self):
        model, X, _ = _fit_hazard_model(max_time=10, n_features=3)
        result = model.predict_expected_time(X[:5])
        assert result.shape == (5,)

    def test_predict_expected_time_positive(self):
        model, X, _ = _fit_hazard_model(max_time=10, n_features=3)
        result = model.predict_expected_time(X[:5])
        assert np.all(result > 0)

    def test_predict_expected_time_within_range(self):
        """Expected time should be between 1 and max_time."""
        model, X, _ = _fit_hazard_model(max_time=10, n_features=3)
        result = model.predict_expected_time(X[:5])
        assert np.all(result >= 1)
        assert np.all(result <= 10)

    # -- predict_median_time -----------------------------------------------

    def test_predict_median_time_unfitted_returns_none(self):
        model = HazardModel(max_time=10)
        X = RNG.standard_normal((5, 3))
        result = model.predict_median_time(X)
        assert result is None

    def test_predict_median_time_shape(self):
        model, X, _ = _fit_hazard_model(max_time=10, n_features=3)
        result = model.predict_median_time(X[:5])
        assert result.shape == (5,)

    def test_predict_median_time_positive(self):
        model, X, _ = _fit_hazard_model(max_time=10, n_features=3)
        result = model.predict_median_time(X[:5])
        assert np.all(result >= 1)

    # -- get_baseline_hazard_curve -----------------------------------------

    def test_get_baseline_hazard_curve_unfitted_returns_none(self):
        model = HazardModel(max_time=10)
        result = model.get_baseline_hazard_curve()
        assert result is None

    def test_get_baseline_hazard_curve_shape(self):
        model, _, _ = _fit_hazard_model(max_time=10)
        result = model.get_baseline_hazard_curve()
        assert isinstance(result, np.ndarray)
        assert result.shape == (10,)

    def test_get_baseline_hazard_curve_in_zero_one(self):
        """Baseline hazards converted via sigmoid should be in (0, 1)."""
        model, _, _ = _fit_hazard_model(max_time=10)
        result = model.get_baseline_hazard_curve()
        assert np.all(result > 0)
        assert np.all(result < 1)

    # -- get_params / set_params -------------------------------------------

    def test_get_params_returns_dict(self):
        model = HazardModel(max_time=10)
        result = model.get_params()
        assert isinstance(result, dict)
        assert result["max_time"] == 10
        assert result["_is_fitted"] is False
        assert result["baseline_hazards"] is None
        assert result["covariate_effects"] is None

    def test_get_params_after_fit(self):
        model, _, _ = _fit_hazard_model(max_time=10, n_features=3)
        result = model.get_params()
        assert result["_is_fitted"] is True
        assert result["baseline_hazards"] is not None
        assert len(result["baseline_hazards"]) == 10
        assert result["covariate_effects"] is not None
        assert len(result["covariate_effects"]) == 3

    def test_set_params_updates_model(self):
        model = HazardModel(max_time=10)
        model.set_params({
            "max_time": 20,
            "lambda_": 0.5,
            "max_iter": 200,
        })
        assert model.max_time == 20
        assert model.lambda_ == 0.5
        assert model.max_iter == 200

    def test_get_set_params_roundtrip(self):
        """Verify get_params -> set_params roundtrip preserves state."""
        model1, _, _ = _fit_hazard_model(max_time=10, n_features=3)
        params = model1.get_params()

        model2 = HazardModel(max_time=5)
        model2.set_params(params)

        assert model2.max_time == 10
        assert model2._is_fitted is True
        np.testing.assert_array_equal(
            model2.baseline_hazards, model1.baseline_hazards
        )
        np.testing.assert_array_equal(
            model2.covariate_effects, model1.covariate_effects
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
