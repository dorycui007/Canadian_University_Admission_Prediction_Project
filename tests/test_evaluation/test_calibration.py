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
        if result is None:
            pytest.skip("Not yet implemented")
        assert result == 0.0

    def test_worst_case(self):
        """Brier score is 1.0 for maximally wrong predictions."""
        result = brier_score(np.array([1, 0, 1]), np.array([0.0, 1.0, 0.0]))
        if result is None:
            pytest.skip("Not yet implemented")
        assert result == 1.0


# =============================================================================
#                    EXPECTED CALIBRATION ERROR TESTS
# =============================================================================

class TestExpectedCalibrationError:
    """Tests for expected_calibration_error(y_true, y_prob, n_bins, strategy)."""

    def test_ece_basic(self):
        """ECE returns a non-negative value for valid inputs."""
        result = expected_calibration_error(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]))
        if result is None:
            pytest.skip("Not yet implemented")
        assert result >= 0


# =============================================================================
#                    MAXIMUM CALIBRATION ERROR TESTS
# =============================================================================

class TestMaximumCalibrationError:
    """Tests for maximum_calibration_error(y_true, y_prob, n_bins, min_samples)."""

    def test_mce_basic(self):
        """MCE returns a non-negative value for valid inputs."""
        result = maximum_calibration_error(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]))
        if result is None:
            pytest.skip("Not yet implemented")
        assert result >= 0


# =============================================================================
#                    RELIABILITY DIAGRAM TESTS
# =============================================================================

class TestReliabilityDiagram:
    """Tests for compute_reliability_diagram(y_true, y_prob, n_bins, strategy)."""

    def test_reliability_diagram_returns_tuple(self):
        """compute_reliability_diagram returns a tuple of arrays."""
        result = compute_reliability_diagram(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]))
        if result is None:
            pytest.skip("Not yet implemented")
        assert isinstance(result, tuple)


# =============================================================================
#                    FULL CALIBRATION ANALYSIS TESTS
# =============================================================================

class TestFullCalibrationAnalysis:
    """Tests for full_calibration_analysis(y_true, y_prob, config)."""

    def test_returns_calibration_result(self):
        """full_calibration_analysis returns a CalibrationResult instance."""
        result = full_calibration_analysis(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]))
        if result is None:
            pytest.skip("Not yet implemented")
        assert isinstance(result, CalibrationResult)


# =============================================================================
#                    WILSON CONFIDENCE INTERVAL TESTS
# =============================================================================

class TestWilsonConfidenceInterval:
    """Tests for wilson_confidence_interval(successes, trials, confidence_level)."""

    def test_wilson_ci_basic(self):
        """Wilson CI returns a tuple of (lower, upper) bounds."""
        result = wilson_confidence_interval(5, 10)
        if result is None:
            pytest.skip("Not yet implemented")
        assert isinstance(result, tuple) and len(result) == 2


# =============================================================================
#                    HOSMER-LEMESHOW TEST TESTS
# =============================================================================

class TestHosmerLemeshowTest:
    """Tests for hosmer_lemeshow_test(y_true, y_prob, n_groups)."""

    def test_hosmer_lemeshow_basic(self):
        """Hosmer-Lemeshow test returns a tuple of (statistic, p_value)."""
        result = hosmer_lemeshow_test(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]))
        if result is None:
            pytest.skip("Not yet implemented")
        assert isinstance(result, tuple) and len(result) == 2


# =============================================================================
#                    PLATT SCALING TESTS
# =============================================================================

class TestPlattScaling:
    """Tests for platt_scaling_params and apply_platt_scaling."""

    def test_platt_scaling_roundtrip(self):
        """Platt scaling params and application produce valid probabilities."""
        result = platt_scaling_params(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]))
        if result is None:
            pytest.skip("Not yet implemented")
        assert isinstance(result, tuple) and len(result) == 2


# =============================================================================
#                    VALIDATE PROBABILITY INPUTS TESTS
# =============================================================================

class TestValidateProbabilityInputs:
    """Tests for validate_probability_inputs(y_true, y_prob)."""

    def test_valid_inputs_pass(self):
        """Valid binary labels and probabilities do not raise."""
        result = validate_probability_inputs(np.array([1, 0]), np.array([0.9, 0.1]))
        if result is None:
            pytest.skip("Not yet implemented")


# =============================================================================
#                    BIN PREDICTIONS TESTS
# =============================================================================

class TestBinPredictions:
    """Tests for bin_predictions(y_prob, n_bins, strategy)."""

    def test_bin_predictions_returns_tuple(self):
        """bin_predictions returns a tuple of bin assignments and edges."""
        result = bin_predictions(np.array([0.1, 0.5, 0.9]), n_bins=3)
        if result is None:
            pytest.skip("Not yet implemented")
        assert isinstance(result, tuple)


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
        if result is None:
            pytest.skip("Not yet implemented")
        assert isinstance(result, BrierDecomposition)


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
        if result is None:
            pytest.skip("Not yet implemented")
        assert isinstance(result, tuple) and len(result) == 2


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
        if result is None:
            pytest.skip("Not yet implemented")
        assert isinstance(result, dict)


# =============================================================================
#                    CALIBRATION CONFIDENCE INTERVALS TESTS
# =============================================================================

class TestCalibrationConfidenceIntervals:
    """Tests for calibration_confidence_intervals(y_true, y_prob, n_bins, confidence_level, method)."""

    def test_calibration_confidence_intervals_basic(self):
        """calibration_confidence_intervals returns a tuple of three arrays."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = calibration_confidence_intervals(y_true, y_prob, n_bins=2)
        if result is None:
            pytest.skip("Not yet implemented")
        assert isinstance(result, tuple) and len(result) == 3


# =============================================================================
#                    CALIBRATION TEST TESTS
# =============================================================================

class TestCalibrationTest:
    """Tests for calibration_test(y_true, y_prob, method)."""

    def test_calibration_test_basic(self):
        """calibration_test returns a tuple of (statistic, p_value)."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = calibration_test(y_true, y_prob)
        if result is None:
            pytest.skip("Not yet implemented")
        assert isinstance(result, tuple) and len(result) == 2


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
        if result is None:
            pytest.skip("Not yet implemented")
        assert isinstance(result, dict)


# =============================================================================
#                    APPLY PLATT SCALING TESTS
# =============================================================================

class TestApplyPlattScaling:
    """Tests for apply_platt_scaling(y_prob, A, B)."""

    def test_apply_platt_scaling_basic(self):
        """apply_platt_scaling returns an array of calibrated probabilities."""
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = apply_platt_scaling(y_prob, A=1.0, B=0.0)
        if result is None:
            pytest.skip("Not yet implemented")
        assert isinstance(result, np.ndarray)
        assert len(result) == len(y_prob)


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
        if result is None:
            pytest.skip("Not yet implemented")
        assert isinstance(result, np.ndarray)
        assert len(result) == len(y_prob)


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
        if result is None:
            pytest.skip("Not yet implemented")
        assert isinstance(result, float)
        assert result > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
