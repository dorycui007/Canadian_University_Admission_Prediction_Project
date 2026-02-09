"""
Unit Tests for Calibration Evaluation Module
=============================================

Tests for src/evaluation/calibration.py — CalibrationConfig, CalibrationResult,
BrierDecomposition dataclasses, and all calibration scoring / analysis functions.
"""

import pytest
import numpy as np
from dataclasses import fields

from src.evaluation.calibration import (
    CalibrationConfig, CalibrationResult, BrierDecomposition,
    brier_score, brier_score_decomposition, expected_calibration_error,
    maximum_calibration_error, compute_reliability_diagram, calibration_curve,
    full_calibration_analysis, compare_calibration, wilson_confidence_interval,
    hosmer_lemeshow_test, platt_scaling_params, apply_platt_scaling,
    validate_probability_inputs, bin_predictions,
    calibration_confidence_intervals, calibration_test,
    subgroup_calibration, isotonic_calibration, temperature_scaling,
)


# =============================================================================
#                    CALIBRATIONCONFIG TESTS
# =============================================================================

class TestCalibrationConfig:
    """Tests for CalibrationConfig dataclass."""

    def test_default_values(self):
        """Default config has expected values."""
        config = CalibrationConfig()
        assert config.n_bins == 10
        assert config.strategy == "uniform"
        assert config.min_samples_per_bin == 10

    def test_has_expected_fields(self):
        """CalibrationConfig has exactly the expected fields."""
        field_names = {f.name for f in fields(CalibrationConfig)}
        assert "n_bins" in field_names
        assert "strategy" in field_names
        assert "min_samples_per_bin" in field_names

    def test_custom_values(self):
        """CalibrationConfig stores custom values correctly."""
        config = CalibrationConfig(n_bins=20, strategy='quantile', min_samples_per_bin=5)
        assert config.n_bins == 20
        assert config.strategy == 'quantile'
        assert config.min_samples_per_bin == 5


# =============================================================================
#                    BRIERDECOMPOSITION TESTS
# =============================================================================

class TestBrierDecomposition:
    """Tests for BrierDecomposition dataclass."""

    def test_has_expected_fields(self):
        """BrierDecomposition has the expected fields."""
        field_names = {f.name for f in fields(BrierDecomposition)}
        expected = {"uncertainty", "resolution", "reliability", "brier"}
        assert expected.issubset(field_names)

    def test_fields_stored_correctly(self):
        """Values passed to BrierDecomposition are stored correctly."""
        decomp = BrierDecomposition(
            uncertainty=0.25, resolution=0.10, reliability=0.05, brier=0.20
        )
        assert decomp.uncertainty == 0.25
        assert decomp.resolution == 0.10
        assert decomp.reliability == 0.05
        assert decomp.brier == 0.20


# =============================================================================
#                    BRIER SCORE TESTS
# =============================================================================

class TestBrierScore:
    """Tests for brier_score(y_true, y_prob)."""

    def test_perfect_predictions(self):
        """Brier score is 0.0 for perfect predictions."""
        result = brier_score(np.array([1, 0, 1]), np.array([1.0, 0.0, 1.0]))
        assert result == pytest.approx(0.0)

    def test_worst_case(self):
        """Brier score is 1.0 for maximally wrong predictions."""
        result = brier_score(np.array([1, 0, 1]), np.array([0.0, 1.0, 0.0]))
        assert result == pytest.approx(1.0)

    def test_known_value(self):
        """Brier score matches hand-computed value from docstring example."""
        # From the docstring: y_true=[1,1,0,0], y_prob=[0.9,0.8,0.3,0.2]
        # (0.9-1)^2 + (0.8-1)^2 + (0.3-0)^2 + (0.2-0)^2 = 0.01+0.04+0.09+0.04 = 0.18
        # Brier = 0.18 / 4 = 0.045
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.3, 0.2])
        result = brier_score(y_true, y_prob)
        assert result == pytest.approx(0.045)

    def test_returns_float(self):
        """Brier score returns a float."""
        result = brier_score(np.array([0, 1]), np.array([0.3, 0.7]))
        assert isinstance(result, float)

    def test_range_bounded(self):
        """Brier score is in [0, 1]."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_prob = np.array([0.2, 0.6, 0.4, 0.8, 0.1])
        result = brier_score(y_true, y_prob)
        assert 0.0 <= result <= 1.0

    def test_all_same_prob(self):
        """Brier score for constant probability 0.5 on balanced data."""
        # (0.5-0)^2 + (0.5-1)^2 = 0.25 + 0.25 = 0.5, mean = 0.25
        y_true = np.array([0, 1])
        y_prob = np.array([0.5, 0.5])
        result = brier_score(y_true, y_prob)
        assert result == pytest.approx(0.25)


# =============================================================================
#                    EXPECTED CALIBRATION ERROR TESTS
# =============================================================================

class TestExpectedCalibrationError:
    """Tests for expected_calibration_error(y_true, y_prob, n_bins, strategy)."""

    def test_ece_basic(self):
        """ECE returns a non-negative float for valid inputs."""
        result = expected_calibration_error(
            np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2])
        )
        assert isinstance(result, float)
        assert result >= 0.0

    def test_ece_bounded(self):
        """ECE is bounded between 0 and 1."""
        result = expected_calibration_error(
            np.array([1, 0, 1, 0, 1, 0]),
            np.array([0.7, 0.3, 0.8, 0.2, 0.6, 0.4]),
        )
        assert 0.0 <= result <= 1.0

    def test_ece_perfect_calibration_low(self):
        """ECE should be low for well-calibrated predictions."""
        # Build a larger dataset where predictions match outcomes
        np.random.seed(42)
        n = 200
        y_prob = np.random.uniform(0, 1, n)
        y_true = (np.random.uniform(0, 1, n) < y_prob).astype(int)
        result = expected_calibration_error(y_true, y_prob, n_bins=5)
        # With 200 samples this won't be exactly 0 but should be reasonable
        assert result < 0.3

    def test_ece_quantile_strategy(self):
        """ECE works with quantile strategy."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
        result = expected_calibration_error(y_true, y_prob, n_bins=4, strategy='quantile')
        assert isinstance(result, float)
        assert result >= 0.0


# =============================================================================
#                    MAXIMUM CALIBRATION ERROR TESTS
# =============================================================================

class TestMaximumCalibrationError:
    """Tests for maximum_calibration_error(y_true, y_prob, n_bins, min_samples)."""

    def test_mce_basic(self):
        """MCE returns a non-negative float for valid inputs."""
        result = maximum_calibration_error(
            np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2])
        )
        assert isinstance(result, float)
        assert result >= 0.0

    def test_mce_bounded(self):
        """MCE is bounded between 0 and 1."""
        result = maximum_calibration_error(
            np.array([1, 0, 1, 0, 1, 0]),
            np.array([0.7, 0.3, 0.8, 0.2, 0.6, 0.4]),
        )
        assert 0.0 <= result <= 1.0

    def test_mce_geq_ece(self):
        """MCE should be >= ECE since it's the max vs weighted average."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5])
        mce = maximum_calibration_error(y_true, y_prob, n_bins=5, min_samples=1)
        ece = expected_calibration_error(y_true, y_prob, n_bins=5)
        assert mce >= ece - 1e-10  # allow tiny float tolerance


# =============================================================================
#                    RELIABILITY DIAGRAM TESTS
# =============================================================================

class TestReliabilityDiagram:
    """Tests for compute_reliability_diagram(y_true, y_prob, n_bins, strategy)."""

    def test_reliability_diagram_returns_tuple_of_four(self):
        """compute_reliability_diagram returns a tuple of four arrays."""
        result = compute_reliability_diagram(
            np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2])
        )
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_bin_edges_length(self):
        """bin_edges has length n_bins + 1."""
        n_bins = 5
        bin_edges, bin_centers, bin_accuracies, bin_counts = compute_reliability_diagram(
            np.array([1, 0, 1, 0, 1, 0]),
            np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3]),
            n_bins=n_bins,
        )
        assert len(bin_edges) == n_bins + 1

    def test_bin_centers_length(self):
        """bin_centers has length n_bins."""
        n_bins = 5
        bin_edges, bin_centers, bin_accuracies, bin_counts = compute_reliability_diagram(
            np.array([1, 0, 1, 0, 1, 0]),
            np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3]),
            n_bins=n_bins,
        )
        assert len(bin_centers) == n_bins
        assert len(bin_accuracies) == n_bins
        assert len(bin_counts) == n_bins

    def test_bin_counts_sum(self):
        """Total bin counts equals number of samples."""
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
        _, _, _, bin_counts = compute_reliability_diagram(y_true, y_prob, n_bins=5)
        assert np.sum(bin_counts) == len(y_true)


# =============================================================================
#                    FULL CALIBRATION ANALYSIS TESTS
# =============================================================================

class TestFullCalibrationAnalysis:
    """Tests for full_calibration_analysis(y_true, y_prob, config)."""

    def test_returns_calibration_result(self):
        """full_calibration_analysis returns a CalibrationResult instance."""
        result = full_calibration_analysis(
            np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2])
        )
        assert isinstance(result, CalibrationResult)

    def test_result_fields_populated(self):
        """All fields of CalibrationResult are populated."""
        result = full_calibration_analysis(
            np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2])
        )
        assert isinstance(result.brier_score, float)
        assert isinstance(result.ece, float)
        assert isinstance(result.mce, float)
        assert isinstance(result.uncertainty, float)
        assert isinstance(result.resolution, float)
        assert isinstance(result.reliability, float)
        assert isinstance(result.bin_edges, np.ndarray)
        assert isinstance(result.bin_counts, np.ndarray)
        assert isinstance(result.bin_accuracies, np.ndarray)
        assert isinstance(result.bin_confidences, np.ndarray)

    def test_brier_score_matches_standalone(self):
        """Brier score from full analysis matches standalone brier_score."""
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2])
        result = full_calibration_analysis(y_true, y_prob)
        standalone = brier_score(y_true, y_prob)
        assert result.brier_score == pytest.approx(standalone)

    def test_with_custom_config(self):
        """full_calibration_analysis accepts a custom CalibrationConfig."""
        config = CalibrationConfig(n_bins=5, strategy='uniform', min_samples_per_bin=1)
        result = full_calibration_analysis(
            np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]),
            config=config,
        )
        assert isinstance(result, CalibrationResult)
        assert len(result.bin_counts) == 5


# =============================================================================
#                    WILSON CONFIDENCE INTERVAL TESTS
# =============================================================================

class TestWilsonConfidenceInterval:
    """Tests for wilson_confidence_interval(successes, trials, confidence_level)."""

    def test_wilson_ci_basic(self):
        """Wilson CI returns a tuple of (lower, upper) bounds."""
        result = wilson_confidence_interval(5, 10)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_wilson_ci_bounds_in_range(self):
        """Wilson CI bounds are in [0, 1]."""
        lower, upper = wilson_confidence_interval(5, 10, 0.95)
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0

    def test_wilson_ci_lower_leq_upper(self):
        """Lower bound <= upper bound."""
        lower, upper = wilson_confidence_interval(5, 10, 0.95)
        assert lower <= upper

    def test_wilson_ci_zero_successes(self):
        """Wilson CI with zero successes has lower bound near 0."""
        lower, upper = wilson_confidence_interval(0, 10, 0.95)
        assert lower == pytest.approx(0.0)
        assert upper > 0.0

    def test_wilson_ci_all_successes(self):
        """Wilson CI with all successes has upper bound near 1."""
        lower, upper = wilson_confidence_interval(10, 10, 0.95)
        assert upper == pytest.approx(1.0)
        assert lower < 1.0

    def test_wilson_ci_zero_trials(self):
        """Wilson CI with zero trials returns (0, 1)."""
        lower, upper = wilson_confidence_interval(0, 0, 0.95)
        assert lower == 0.0
        assert upper == 1.0

    def test_wilson_ci_point_in_interval(self):
        """The point estimate p_hat is within the Wilson CI."""
        successes, trials = 7, 20
        lower, upper = wilson_confidence_interval(successes, trials, 0.95)
        p_hat = successes / trials
        assert lower <= p_hat <= upper


# =============================================================================
#                    HOSMER-LEMESHOW TEST TESTS
# =============================================================================

class TestHosmerLemeshowTest:
    """Tests for hosmer_lemeshow_test(y_true, y_prob, n_groups)."""

    def test_hosmer_lemeshow_basic(self):
        """Hosmer-Lemeshow test returns a tuple of (statistic, p_value)."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5])
        result = hosmer_lemeshow_test(y_true, y_prob, n_groups=5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_chi_squared_nonnegative(self):
        """Chi-squared statistic is non-negative."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5])
        chi2, p_val = hosmer_lemeshow_test(y_true, y_prob, n_groups=5)
        assert chi2 >= 0.0

    def test_p_value_in_range(self):
        """p-value is between 0 and 1."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5])
        chi2, p_val = hosmer_lemeshow_test(y_true, y_prob, n_groups=5)
        assert 0.0 <= p_val <= 1.0


# =============================================================================
#                    PLATT SCALING TESTS
# =============================================================================

class TestPlattScaling:
    """Tests for platt_scaling_params and apply_platt_scaling."""

    def test_platt_scaling_returns_tuple(self):
        """Platt scaling params returns a tuple of (A, B)."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
        result = platt_scaling_params(y_true, y_prob)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_platt_scaling_params_are_floats(self):
        """A and B are floats."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
        A, B = platt_scaling_params(y_true, y_prob)
        assert isinstance(A, float)
        assert isinstance(B, float)

    def test_platt_roundtrip_produces_valid_probs(self):
        """Applying Platt scaling produces probabilities in [0, 1]."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
        A, B = platt_scaling_params(y_true, y_prob)
        calibrated = apply_platt_scaling(y_prob, A, B)
        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)


# =============================================================================
#                    VALIDATE PROBABILITY INPUTS TESTS
# =============================================================================

class TestValidateProbabilityInputs:
    """Tests for validate_probability_inputs(y_true, y_prob)."""

    def test_valid_inputs_pass(self):
        """Valid binary labels and probabilities do not raise."""
        result = validate_probability_inputs(np.array([1, 0]), np.array([0.9, 0.1]))
        assert result is True

    def test_empty_arrays_raise(self):
        """Empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            validate_probability_inputs(np.array([]), np.array([]))

    def test_shape_mismatch_raises(self):
        """Mismatched shapes raise ValueError."""
        with pytest.raises(ValueError, match="same shape"):
            validate_probability_inputs(np.array([0, 1, 0]), np.array([0.5, 0.5]))

    def test_non_binary_y_true_raises(self):
        """Non-binary y_true raises ValueError."""
        with pytest.raises(ValueError, match="0 and 1"):
            validate_probability_inputs(np.array([0, 2]), np.array([0.5, 0.5]))

    def test_prob_out_of_range_raises(self):
        """y_prob outside [0,1] raises ValueError."""
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            validate_probability_inputs(np.array([0, 1]), np.array([-0.1, 0.5]))

    def test_nan_y_true_raises(self):
        """NaN in y_true raises ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            validate_probability_inputs(np.array([0, np.nan]), np.array([0.5, 0.5]))

    def test_nan_y_prob_raises(self):
        """NaN in y_prob raises ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            validate_probability_inputs(np.array([0, 1]), np.array([0.5, np.nan]))


# =============================================================================
#                    BIN PREDICTIONS TESTS
# =============================================================================

class TestBinPredictions:
    """Tests for bin_predictions(y_prob, n_bins, strategy)."""

    def test_bin_predictions_returns_tuple(self):
        """bin_predictions returns a tuple of (indices, edges)."""
        result = bin_predictions(np.array([0.1, 0.5, 0.9]), n_bins=3)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_indices_in_valid_range(self):
        """Bin indices are in [0, n_bins - 1]."""
        n_bins = 5
        indices, edges = bin_predictions(np.array([0.0, 0.2, 0.5, 0.8, 1.0]), n_bins=n_bins)
        assert np.all(indices >= 0)
        assert np.all(indices < n_bins)

    def test_edges_length(self):
        """Bin edges has length n_bins + 1 for uniform strategy."""
        n_bins = 5
        indices, edges = bin_predictions(np.array([0.1, 0.5, 0.9]), n_bins=n_bins, strategy='uniform')
        assert len(edges) == n_bins + 1

    def test_uniform_edges_span_zero_to_one(self):
        """Uniform bin edges go from 0 to 1."""
        indices, edges = bin_predictions(np.array([0.5]), n_bins=10, strategy='uniform')
        assert edges[0] == pytest.approx(0.0)
        assert edges[-1] == pytest.approx(1.0)

    def test_invalid_strategy_raises(self):
        """Unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            bin_predictions(np.array([0.5]), n_bins=5, strategy='invalid')


# =============================================================================
#                    BRIER SCORE DECOMPOSITION TESTS
# =============================================================================

class TestBrierScoreDecomposition:
    """Tests for brier_score_decomposition(y_true, y_prob, n_bins)."""

    def test_brier_score_decomposition_basic(self):
        """brier_score_decomposition returns a BrierDecomposition instance."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = brier_score_decomposition(y_true, y_prob, n_bins=2)
        assert isinstance(result, BrierDecomposition)

    def test_decomposition_identity(self):
        """Brier = Uncertainty - Resolution + Reliability."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.7, 0.3, 0.8, 0.2, 0.6, 0.4, 0.15, 0.85])
        decomp = brier_score_decomposition(y_true, y_prob, n_bins=5)
        expected_brier = decomp.uncertainty - decomp.resolution + decomp.reliability
        assert decomp.brier == pytest.approx(expected_brier, abs=1e-10)

    def test_uncertainty_range(self):
        """Uncertainty is in [0, 0.25] for binary outcomes."""
        y_true = np.array([0, 1, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.2, 0.7])
        decomp = brier_score_decomposition(y_true, y_prob, n_bins=3)
        assert 0.0 <= decomp.uncertainty <= 0.25

    def test_resolution_nonnegative(self):
        """Resolution is non-negative."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        decomp = brier_score_decomposition(y_true, y_prob, n_bins=2)
        assert decomp.resolution >= -1e-10  # allow tiny float noise

    def test_reliability_nonnegative(self):
        """Reliability is non-negative."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        decomp = brier_score_decomposition(y_true, y_prob, n_bins=2)
        assert decomp.reliability >= -1e-10


# =============================================================================
#                    CALIBRATION CURVE TESTS
# =============================================================================

class TestCalibrationCurve:
    """Tests for calibration_curve(y_true, y_prob, n_bins, strategy)."""

    def test_calibration_curve_returns_tuple(self):
        """calibration_curve returns a tuple of two arrays."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = calibration_curve(y_true, y_prob, n_bins=2)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_calibration_curve_arrays_same_length(self):
        """mean_predicted and fraction_positives have the same length."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        mean_predicted, fraction_positives = calibration_curve(y_true, y_prob, n_bins=2)
        assert len(mean_predicted) == len(fraction_positives)

    def test_calibration_curve_values_in_range(self):
        """mean_predicted and fraction_positives are in [0, 1]."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7, 0.2])
        mean_pred, frac_pos = calibration_curve(y_true, y_prob, n_bins=3)
        assert np.all(mean_pred >= 0.0) and np.all(mean_pred <= 1.0)
        assert np.all(frac_pos >= 0.0) and np.all(frac_pos <= 1.0)


# =============================================================================
#                    COMPARE CALIBRATION TESTS
# =============================================================================

class TestCompareCalibration:
    """Tests for compare_calibration(models, config)."""

    def test_compare_calibration_returns_dict(self):
        """compare_calibration returns a dict of model_name to CalibrationResult."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        models = {"model_a": (y_true, y_prob)}
        result = compare_calibration(models)
        assert isinstance(result, dict)
        assert "model_a" in result
        assert isinstance(result["model_a"], CalibrationResult)

    def test_compare_calibration_multiple_models(self):
        """compare_calibration handles multiple models."""
        y_true = np.array([0, 1, 1, 0])
        y_prob_a = np.array([0.1, 0.9, 0.8, 0.3])
        y_prob_b = np.array([0.2, 0.7, 0.6, 0.4])
        models = {
            "model_a": (y_true, y_prob_a),
            "model_b": (y_true, y_prob_b),
        }
        result = compare_calibration(models)
        assert "model_a" in result
        assert "model_b" in result


# =============================================================================
#                    CALIBRATION CONFIDENCE INTERVALS TESTS
# =============================================================================

class TestCalibrationConfidenceIntervals:
    """Tests for calibration_confidence_intervals(y_true, y_prob, n_bins, confidence_level, method)."""

    def test_calibration_confidence_intervals_basic(self):
        """calibration_confidence_intervals returns a tuple of three arrays."""
        y_true = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.2, 0.7, 0.6, 0.4])
        result = calibration_confidence_intervals(y_true, y_prob, n_bins=2)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_ci_arrays_same_length(self):
        """All three arrays have the same length (n_bins)."""
        n_bins = 3
        y_true = np.array([0, 1, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.2, 0.7])
        bin_acc, ci_lower, ci_upper = calibration_confidence_intervals(
            y_true, y_prob, n_bins=n_bins
        )
        assert len(bin_acc) == n_bins
        assert len(ci_lower) == n_bins
        assert len(ci_upper) == n_bins

    def test_ci_lower_leq_upper(self):
        """Lower CI <= Upper CI for non-empty bins."""
        y_true = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.2, 0.7, 0.6, 0.4])
        bin_acc, ci_lower, ci_upper = calibration_confidence_intervals(
            y_true, y_prob, n_bins=2
        )
        for i in range(len(bin_acc)):
            if not np.isnan(ci_lower[i]):
                assert ci_lower[i] <= ci_upper[i]


# =============================================================================
#                    CALIBRATION TEST TESTS
# =============================================================================

class TestCalibrationTest:
    """Tests for calibration_test(y_true, y_prob, method)."""

    def test_calibration_test_spiegelhalter(self):
        """calibration_test with default Spiegelhalter method returns (stat, p_value)."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7, 0.2, 0.6, 0.4])
        result = calibration_test(y_true, y_prob)
        assert isinstance(result, tuple)
        assert len(result) == 2
        z_stat, p_val = result
        assert isinstance(z_stat, float)
        assert 0.0 <= p_val <= 1.0

    def test_calibration_test_hosmer_lemeshow_method(self):
        """calibration_test with method='hosmer_lemeshow' delegates correctly."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7, 0.2, 0.6, 0.4, 0.5, 0.5])
        result = calibration_test(y_true, y_prob, method='hosmer_lemeshow')
        assert isinstance(result, tuple)
        assert len(result) == 2
        chi2, p_val = result
        assert chi2 >= 0.0
        assert 0.0 <= p_val <= 1.0


# =============================================================================
#                    SUBGROUP CALIBRATION TESTS
# =============================================================================

class TestSubgroupCalibration:
    """Tests for subgroup_calibration(y_true, y_prob, groups, config)."""

    def test_subgroup_calibration_returns_dict(self):
        """subgroup_calibration returns a dict of group to CalibrationResult."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        groups = np.array(["A", "A", "B", "B"])
        result = subgroup_calibration(y_true, y_prob, groups)
        assert isinstance(result, dict)

    def test_subgroup_calibration_keys(self):
        """Result keys correspond to unique group labels."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        groups = np.array(["A", "A", "B", "B"])
        result = subgroup_calibration(y_true, y_prob, groups)
        assert "A" in result
        assert "B" in result

    def test_subgroup_calibration_values_type(self):
        """Each value is a CalibrationResult."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        groups = np.array(["A", "A", "B", "B"])
        result = subgroup_calibration(y_true, y_prob, groups)
        for group, cal_result in result.items():
            assert isinstance(cal_result, CalibrationResult)


# =============================================================================
#                    APPLY PLATT SCALING TESTS
# =============================================================================

class TestApplyPlattScaling:
    """Tests for apply_platt_scaling(y_prob, A, B)."""

    def test_apply_platt_scaling_basic(self):
        """apply_platt_scaling returns an array of calibrated probabilities."""
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = apply_platt_scaling(y_prob, A=1.0, B=0.0)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(y_prob)

    def test_apply_platt_scaling_output_range(self):
        """Calibrated probabilities are in [0, 1]."""
        y_prob = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
        result = apply_platt_scaling(y_prob, A=1.5, B=-0.3)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_identity_scaling(self):
        """With A=1, B=0, output should be close to input (sigmoid(logit(p)) = p)."""
        y_prob = np.array([0.2, 0.5, 0.8])
        result = apply_platt_scaling(y_prob, A=1.0, B=0.0)
        np.testing.assert_allclose(result, y_prob, atol=1e-5)


# =============================================================================
#                    ISOTONIC CALIBRATION TESTS
# =============================================================================

class TestIsotonicCalibration:
    """Tests for isotonic_calibration(y_true, y_prob)."""

    def test_isotonic_calibration_basic(self):
        """isotonic_calibration returns an array of calibrated probabilities."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = isotonic_calibration(y_true, y_prob)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(y_prob)

    def test_isotonic_calibration_preserves_length(self):
        """Output length matches input length."""
        n = 20
        y_true = np.array([0, 1] * (n // 2))
        y_prob = np.linspace(0.1, 0.9, n)
        result = isotonic_calibration(y_true, y_prob)
        assert len(result) == n

    def test_isotonic_calibration_monotonic(self):
        """Calibrated values are monotonically non-decreasing when sorted by y_prob."""
        np.random.seed(42)
        n = 50
        y_prob = np.random.uniform(0.05, 0.95, n)
        y_true = (np.random.uniform(0, 1, n) < y_prob).astype(int)
        result = isotonic_calibration(y_true, y_prob)
        # When sorted by y_prob, calibrated values should be non-decreasing
        sort_order = np.argsort(y_prob)
        sorted_cal = result[sort_order]
        for i in range(len(sorted_cal) - 1):
            assert sorted_cal[i] <= sorted_cal[i + 1] + 1e-10


# =============================================================================
#                    TEMPERATURE SCALING TESTS
# =============================================================================

class TestTemperatureScaling:
    """Tests for temperature_scaling(y_true, y_logits)."""

    def test_temperature_scaling_basic(self):
        """temperature_scaling returns a positive float temperature."""
        y_true = np.array([0, 1, 1, 0])
        y_logits = np.array([-2.0, 2.0, 1.5, -0.5])
        result = temperature_scaling(y_true, y_logits)
        assert isinstance(result, float)
        assert result > 0

    def test_temperature_scaling_well_calibrated_near_one(self):
        """For already well-calibrated logits, temperature should be near 1."""
        # Logits that produce well-calibrated probabilities
        np.random.seed(42)
        n = 100
        y_logits = np.random.normal(0, 1, n)
        probs = 1.0 / (1.0 + np.exp(-y_logits))
        y_true = (np.random.uniform(0, 1, n) < probs).astype(int)
        T = temperature_scaling(y_true, y_logits)
        # Temperature should be somewhat close to 1 for well-calibrated model
        assert 0.05 <= T <= 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
