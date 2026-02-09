"""
Unit Tests for Discrimination Evaluation Module
================================================

Tests for src/evaluation/discrimination.py — DiscriminationConfig, ROCResult,
PRResult, LiftResult, DiscriminationResult dataclasses, and all discrimination
scoring / analysis functions.
"""

import pytest
import numpy as np
from dataclasses import fields

from src.evaluation.discrimination import (
    DiscriminationConfig, ROCResult, PRResult, LiftResult, DiscriminationResult,
    compute_roc_curve, roc_auc_score, roc_auc_confidence_interval,
    optimal_threshold_roc, compute_pr_curve, pr_auc_score,
    average_precision_score, f1_at_threshold, optimal_threshold_f1,
    compute_lift_curve, compute_gains_curve, lift_at_percentile,
    confusion_matrix_at_threshold, classification_metrics_at_threshold,
    full_discrimination_analysis, compare_discrimination, compare_roc_curves,
    delong_test, subgroup_discrimination, validate_discrimination_inputs,
    rank_order_statistics,
)


# =============================================================================
#                    DISCRIMINATIONCONFIG TESTS
# =============================================================================

class TestDiscriminationConfig:
    """Tests for DiscriminationConfig dataclass."""

    def test_default_values(self):
        """Default config has expected values."""
        config = DiscriminationConfig()
        assert config.n_thresholds == 1000
        assert config.positive_label == 1
        assert config.lift_percentiles == [10, 20, 30, 50]

    def test_has_expected_fields(self):
        """DiscriminationConfig has exactly the expected fields."""
        field_names = {f.name for f in fields(DiscriminationConfig)}
        assert "n_thresholds" in field_names
        assert "positive_label" in field_names
        assert "lift_percentiles" in field_names


# =============================================================================
#                    ROCRESULT TESTS
# =============================================================================

class TestROCResult:
    """Tests for ROCResult dataclass."""

    def test_has_expected_fields(self):
        """ROCResult has the expected fields."""
        field_names = {f.name for f in fields(ROCResult)}
        expected = {"fpr", "tpr", "thresholds", "auc", "auc_ci", "optimal_threshold"}
        assert expected.issubset(field_names)


# =============================================================================
#                    PRRESULT TESTS
# =============================================================================

class TestPRResult:
    """Tests for PRResult dataclass."""

    def test_has_expected_fields(self):
        """PRResult has the expected fields."""
        field_names = {f.name for f in fields(PRResult)}
        expected = {"precision", "recall", "thresholds", "auc", "ap",
                    "f1_scores", "optimal_threshold"}
        assert expected.issubset(field_names)


# =============================================================================
#                    COMPUTE ROC CURVE TESTS
# =============================================================================

class TestComputeROCCurve:
    """Tests for compute_roc_curve(y_true, y_prob)."""

    def test_roc_curve_returns_tuple_of_three(self):
        """compute_roc_curve returns a tuple of (fpr, tpr, thresholds)."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        result = compute_roc_curve(y_true, y_prob)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_fpr_tpr_same_length(self):
        """FPR and TPR arrays have the same length."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        fpr, tpr, thresholds = compute_roc_curve(y_true, y_prob)
        assert len(fpr) == len(tpr)
        assert len(fpr) == len(thresholds)

    def test_starts_at_origin(self):
        """ROC curve starts at (0, 0)."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        fpr, tpr, _ = compute_roc_curve(y_true, y_prob)
        assert fpr[0] == pytest.approx(0.0)
        assert tpr[0] == pytest.approx(0.0)

    def test_ends_at_one_one(self):
        """ROC curve ends at (1, 1)."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        fpr, tpr, _ = compute_roc_curve(y_true, y_prob)
        assert fpr[-1] == pytest.approx(1.0)
        assert tpr[-1] == pytest.approx(1.0)

    def test_fpr_tpr_in_range(self):
        """FPR and TPR values are in [0, 1]."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        fpr, tpr, _ = compute_roc_curve(y_true, y_prob)
        assert np.all(fpr >= 0.0) and np.all(fpr <= 1.0)
        assert np.all(tpr >= 0.0) and np.all(tpr <= 1.0)

    def test_fpr_monotonically_nondecreasing(self):
        """FPR is monotonically non-decreasing."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        fpr, _, _ = compute_roc_curve(y_true, y_prob)
        for i in range(len(fpr) - 1):
            assert fpr[i] <= fpr[i + 1] + 1e-10


# =============================================================================
#                    ROC AUC SCORE TESTS
# =============================================================================

class TestROCAUCScore:
    """Tests for roc_auc_score(y_true, y_prob)."""

    def test_perfect_predictions_auc(self):
        """AUC is 1.0 for perfectly separable predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        result = roc_auc_score(y_true, y_prob)
        assert result == pytest.approx(1.0)

    def test_auc_in_range(self):
        """AUC is between 0 and 1."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        result = roc_auc_score(y_true, y_prob)
        assert 0.0 <= result <= 1.0

    def test_auc_returns_float(self):
        """AUC returns a float."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        result = roc_auc_score(y_true, y_prob)
        assert isinstance(result, float)

    def test_inverted_predictions_low_auc(self):
        """AUC is near 0 for inverted predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.2, 0.1])
        result = roc_auc_score(y_true, y_prob)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_random_predictions_auc_near_half(self):
        """AUC is approximately 0.5 for truly random predictions."""
        np.random.seed(42)
        n = 500
        y_true = np.random.randint(0, 2, n)
        y_prob = np.random.uniform(0, 1, n)
        result = roc_auc_score(y_true, y_prob)
        # With 500 samples, random predictions should yield AUC ~ 0.5
        assert 0.35 <= result <= 0.65


# =============================================================================
#                    OPTIMAL THRESHOLD ROC TESTS
# =============================================================================

class TestOptimalThresholdROC:
    """Tests for optimal_threshold_roc(y_true, y_prob, method)."""

    def test_optimal_threshold_in_range(self):
        """Optimal threshold is a reasonable value."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        result = optimal_threshold_roc(y_true, y_prob)
        assert isinstance(result, float)

    def test_youden_method(self):
        """Youden's J method returns a threshold."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        result = optimal_threshold_roc(y_true, y_prob, method='youden')
        assert isinstance(result, float)

    def test_closest_method(self):
        """Closest to (0,1) method returns a threshold."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        result = optimal_threshold_roc(y_true, y_prob, method='closest')
        assert isinstance(result, float)

    def test_unknown_method_raises(self):
        """Unknown method raises ValueError."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        with pytest.raises(ValueError, match="Unknown method"):
            optimal_threshold_roc(y_true, y_prob, method='invalid')


# =============================================================================
#                    COMPUTE PR CURVE TESTS
# =============================================================================

class TestComputePRCurve:
    """Tests for compute_pr_curve(y_true, y_prob)."""

    def test_pr_curve_returns_tuple_of_three(self):
        """compute_pr_curve returns a tuple of (precision, recall, thresholds)."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        result = compute_pr_curve(y_true, y_prob)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_precision_recall_same_length(self):
        """Precision, recall, and thresholds have the same length."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        precision, recall, thresholds = compute_pr_curve(y_true, y_prob)
        assert len(precision) == len(recall)
        assert len(precision) == len(thresholds)

    def test_precision_recall_in_range(self):
        """Precision and recall values are in [0, 1]."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        precision, recall, _ = compute_pr_curve(y_true, y_prob)
        assert np.all(precision >= 0.0) and np.all(precision <= 1.0)
        assert np.all(recall >= 0.0) and np.all(recall <= 1.0)

    def test_pr_starts_with_recall_zero(self):
        """PR curve starts with recall=0 (prepended point)."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        precision, recall, _ = compute_pr_curve(y_true, y_prob)
        assert recall[0] == pytest.approx(0.0)
        assert precision[0] == pytest.approx(1.0)


# =============================================================================
#                    PR AUC SCORE TESTS
# =============================================================================

class TestPRAUCScore:
    """Tests for pr_auc_score(y_true, y_prob)."""

    def test_pr_auc_in_range(self):
        """PR AUC is between 0 and 1."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        result = pr_auc_score(y_true, y_prob)
        assert 0.0 <= result <= 1.0

    def test_pr_auc_returns_float(self):
        """PR AUC returns a float."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        result = pr_auc_score(y_true, y_prob)
        assert isinstance(result, float)

    def test_perfect_pr_auc(self):
        """PR AUC is high for perfectly separated predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        result = pr_auc_score(y_true, y_prob)
        assert result > 0.9


# =============================================================================
#                    F1 AT THRESHOLD TESTS
# =============================================================================

class TestF1AtThreshold:
    """Tests for f1_at_threshold(y_true, y_prob, threshold)."""

    def test_f1_in_range(self):
        """F1 score is between 0 and 1."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        result = f1_at_threshold(y_true, y_prob, threshold=0.5)
        assert 0.0 <= result <= 1.0

    def test_f1_returns_float(self):
        """F1 score returns a float."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        result = f1_at_threshold(y_true, y_prob, threshold=0.5)
        assert isinstance(result, float)

    def test_f1_perfect_predictions(self):
        """F1 is 1.0 for perfect predictions at proper threshold."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        result = f1_at_threshold(y_true, y_prob, threshold=0.5)
        assert result == pytest.approx(1.0)

    def test_f1_no_true_positives(self):
        """F1 is 0 when threshold is so high no positives are predicted."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.6, 0.7, 0.3])
        result = f1_at_threshold(y_true, y_prob, threshold=0.99)
        assert result == pytest.approx(0.0)

    def test_f1_known_value(self):
        """F1 matches a hand-computed value."""
        # threshold=0.5: predict [0, 1, 1, 0, 1]
        # y_true = [0, 1, 1, 0, 1]
        # TP=3, FP=0, FN=0 -> P=1, R=1, F1=1
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        result = f1_at_threshold(y_true, y_prob, threshold=0.5)
        assert result == pytest.approx(1.0)


# =============================================================================
#                    LIFT CURVE TESTS
# =============================================================================

class TestLiftCurve:
    """Tests for compute_lift_curve(y_true, y_prob, n_bins)."""

    def test_lift_curve_returns_tuple_of_two(self):
        """compute_lift_curve returns a tuple of (percentiles, lift_values)."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        result = compute_lift_curve(y_true, y_prob)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_lift_arrays_same_length(self):
        """percentiles and lift_values have the same length."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        percentiles, lift_values = compute_lift_curve(y_true, y_prob, n_bins=5)
        assert len(percentiles) == len(lift_values)

    def test_lift_values_nonnegative(self):
        """Lift values are non-negative."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        _, lift_values = compute_lift_curve(y_true, y_prob, n_bins=5)
        assert np.all(lift_values >= 0.0)

    def test_top_decile_lift_for_good_model(self):
        """Lift at top decile is > 1 for a model with some discrimination."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.15, 0.25, 0.9, 0.8, 0.85, 0.7, 0.3, 0.35])
        percentiles, lift_values = compute_lift_curve(y_true, y_prob, n_bins=5)
        # The first bin (top predictions) should have lift > 1 for a discriminative model
        assert lift_values[0] >= 1.0


# =============================================================================
#                    CONFUSION MATRIX TESTS
# =============================================================================

class TestConfusionMatrix:
    """Tests for confusion_matrix_at_threshold(y_true, y_prob, threshold)."""

    def test_confusion_matrix_shape(self):
        """Confusion matrix is 2x2."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        result = confusion_matrix_at_threshold(y_true, y_prob, threshold=0.5)
        assert result.shape == (2, 2)

    def test_confusion_matrix_elements_sum(self):
        """All elements of the confusion matrix sum to n."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        cm = confusion_matrix_at_threshold(y_true, y_prob, threshold=0.5)
        assert np.sum(cm) == len(y_true)

    def test_confusion_matrix_perfect(self):
        """Perfect predictions yield correct confusion matrix."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        cm = confusion_matrix_at_threshold(y_true, y_prob, threshold=0.5)
        # TN=2, FP=0, FN=0, TP=2
        assert cm[0, 0] == 2  # TN
        assert cm[0, 1] == 0  # FP
        assert cm[1, 0] == 0  # FN
        assert cm[1, 1] == 2  # TP

    def test_confusion_matrix_nonnegative(self):
        """All confusion matrix entries are non-negative."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        cm = confusion_matrix_at_threshold(y_true, y_prob, threshold=0.5)
        assert np.all(cm >= 0)


# =============================================================================
#                    FULL DISCRIMINATION ANALYSIS TESTS
# =============================================================================

class TestFullDiscriminationAnalysis:
    """Tests for full_discrimination_analysis(y_true, y_prob, config)."""

    def test_returns_discrimination_result(self):
        """full_discrimination_analysis returns a DiscriminationResult instance."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        result = full_discrimination_analysis(y_true, y_prob)
        assert isinstance(result, DiscriminationResult)

    def test_result_has_roc(self):
        """Result contains ROCResult."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        result = full_discrimination_analysis(y_true, y_prob)
        assert isinstance(result.roc, ROCResult)
        assert isinstance(result.roc.auc, float)

    def test_result_has_pr(self):
        """Result contains PRResult."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        result = full_discrimination_analysis(y_true, y_prob)
        assert isinstance(result.pr, PRResult)
        assert isinstance(result.pr.auc, float)

    def test_result_has_lift(self):
        """Result contains LiftResult."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        result = full_discrimination_analysis(y_true, y_prob)
        assert isinstance(result.lift, LiftResult)

    def test_result_has_summary(self):
        """Result contains summary dict with expected keys."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7])
        result = full_discrimination_analysis(y_true, y_prob)
        assert isinstance(result.summary, dict)
        assert 'roc_auc' in result.summary
        assert 'pr_auc' in result.summary
        assert 'best_f1' in result.summary


# =============================================================================
#                    ROC AUC CONFIDENCE INTERVAL TESTS
# =============================================================================

class TestROCAUCConfidenceInterval:
    """Tests for roc_auc_confidence_interval(y_true, y_prob, confidence_level, n_bootstrap)."""

    def test_returns_tuple_of_three(self):
        """roc_auc_confidence_interval returns (auc, ci_lower, ci_upper)."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = roc_auc_confidence_interval(y_true, y_prob, n_bootstrap=50)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_ci_bounds_order(self):
        """CI lower <= AUC <= CI upper."""
        y_true = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.2, 0.7, 0.6, 0.4])
        auc, ci_lower, ci_upper = roc_auc_confidence_interval(
            y_true, y_prob, confidence_level=0.95, n_bootstrap=100
        )
        assert ci_lower <= auc + 1e-10
        assert auc <= ci_upper + 1e-10

    def test_ci_in_range(self):
        """CI bounds are in [0, 1]."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        auc, ci_lower, ci_upper = roc_auc_confidence_interval(
            y_true, y_prob, n_bootstrap=50
        )
        assert 0.0 <= ci_lower <= 1.0
        assert 0.0 <= ci_upper <= 1.0


# =============================================================================
#                    AVERAGE PRECISION SCORE TESTS
# =============================================================================

class TestAveragePrecisionScore:
    """Tests for average_precision_score(y_true, y_prob)."""

    def test_ap_in_range(self):
        """Average precision score is between 0 and 1."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = average_precision_score(y_true, y_prob)
        assert 0.0 <= result <= 1.0

    def test_ap_returns_float(self):
        """Average precision score returns a float."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = average_precision_score(y_true, y_prob)
        assert isinstance(result, float)

    def test_perfect_ap(self):
        """AP is 1.0 for perfectly separated predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        result = average_precision_score(y_true, y_prob)
        assert result == pytest.approx(1.0)


# =============================================================================
#                    OPTIMAL THRESHOLD F1 TESTS
# =============================================================================

class TestOptimalThresholdF1:
    """Tests for optimal_threshold_f1(y_true, y_prob)."""

    def test_returns_tuple_of_two(self):
        """optimal_threshold_f1 returns (optimal_threshold, max_f1_score)."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = optimal_threshold_f1(y_true, y_prob)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_threshold_in_range(self):
        """Optimal threshold is between 0 and 1."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        threshold, f1 = optimal_threshold_f1(y_true, y_prob)
        assert 0.0 <= threshold <= 1.0
        assert 0.0 <= f1 <= 1.0

    def test_max_f1_geq_any_threshold(self):
        """Max F1 is at least as good as F1 at 0.5."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        threshold, max_f1 = optimal_threshold_f1(y_true, y_prob)
        f1_at_half = f1_at_threshold(y_true, y_prob, threshold=0.5)
        assert max_f1 >= f1_at_half - 1e-10


# =============================================================================
#                    COMPUTE GAINS CURVE TESTS
# =============================================================================

class TestComputeGainsCurve:
    """Tests for compute_gains_curve(y_true, y_prob, n_points)."""

    def test_returns_tuple_of_arrays(self):
        """compute_gains_curve returns a tuple of two arrays."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = compute_gains_curve(y_true, y_prob)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_gains_values_in_range(self):
        """Gains values are between 0 and 1."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        pop_pct, gains = compute_gains_curve(y_true, y_prob, n_points=10)
        assert np.all(gains >= 0.0) and np.all(gains <= 1.0)

    def test_gains_ends_at_one(self):
        """At 100% of population, all positives should be captured."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7, 0.2])
        pop_pct, gains = compute_gains_curve(y_true, y_prob, n_points=10)
        assert gains[-1] == pytest.approx(1.0)

    def test_pop_pct_in_range(self):
        """Population percentages are in (0, 1]."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        pop_pct, _ = compute_gains_curve(y_true, y_prob, n_points=10)
        assert np.all(pop_pct > 0.0) and np.all(pop_pct <= 1.0)

    def test_gains_nondecreasing(self):
        """Gains are monotonically non-decreasing."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7, 0.2])
        _, gains = compute_gains_curve(y_true, y_prob, n_points=20)
        for i in range(len(gains) - 1):
            assert gains[i] <= gains[i + 1] + 1e-10


# =============================================================================
#                    LIFT AT PERCENTILE TESTS
# =============================================================================

class TestLiftAtPercentile:
    """Tests for lift_at_percentile(y_true, y_prob, percentile)."""

    def test_returns_float(self):
        """lift_at_percentile returns a float."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = lift_at_percentile(y_true, y_prob, percentile=10)
        assert isinstance(result, (int, float))

    def test_lift_nonnegative(self):
        """Lift value is non-negative."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = lift_at_percentile(y_true, y_prob, percentile=50)
        assert result >= 0.0

    def test_lift_at_100_equals_one(self):
        """Lift at 100th percentile equals 1.0 (entire population)."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7, 0.2])
        result = lift_at_percentile(y_true, y_prob, percentile=100)
        assert result == pytest.approx(1.0)

    def test_good_model_top_lift(self):
        """For a discriminative model, top percentile lift > 1."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.15, 0.25, 0.3, 0.9, 0.8, 0.85, 0.7, 0.75])
        result = lift_at_percentile(y_true, y_prob, percentile=50)
        assert result >= 1.0


# =============================================================================
#                    CLASSIFICATION METRICS AT THRESHOLD TESTS
# =============================================================================

class TestClassificationMetricsAtThreshold:
    """Tests for classification_metrics_at_threshold(y_true, y_prob, threshold)."""

    def test_returns_dict(self):
        """classification_metrics_at_threshold returns a dictionary."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = classification_metrics_at_threshold(y_true, y_prob, threshold=0.5)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        """Result dict contains expected metric keys."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = classification_metrics_at_threshold(y_true, y_prob, threshold=0.5)
        expected_keys = {'accuracy', 'precision', 'recall', 'specificity',
                         'f1', 'mcc', 'balanced_accuracy'}
        assert expected_keys.issubset(result.keys())

    def test_values_in_range(self):
        """All metric values are in valid ranges."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = classification_metrics_at_threshold(y_true, y_prob, threshold=0.5)
        assert 0.0 <= result['accuracy'] <= 1.0
        assert 0.0 <= result['precision'] <= 1.0
        assert 0.0 <= result['recall'] <= 1.0
        assert 0.0 <= result['specificity'] <= 1.0
        assert 0.0 <= result['f1'] <= 1.0
        assert 0.0 <= result['balanced_accuracy'] <= 1.0
        assert -1.0 <= result['mcc'] <= 1.0

    def test_perfect_predictions_metrics(self):
        """Perfect predictions yield accuracy=1, precision=1, recall=1, f1=1."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        result = classification_metrics_at_threshold(y_true, y_prob, threshold=0.5)
        assert result['accuracy'] == pytest.approx(1.0)
        assert result['precision'] == pytest.approx(1.0)
        assert result['recall'] == pytest.approx(1.0)
        assert result['f1'] == pytest.approx(1.0)
        assert result['mcc'] == pytest.approx(1.0)


# =============================================================================
#                    COMPARE DISCRIMINATION TESTS
# =============================================================================

class TestCompareDiscrimination:
    """Tests for compare_discrimination(models, config)."""

    def test_returns_dict(self):
        """compare_discrimination returns a dict keyed by model name."""
        y_true = np.array([0, 1, 1, 0])
        y_prob_a = np.array([0.1, 0.9, 0.8, 0.3])
        y_prob_b = np.array([0.2, 0.7, 0.6, 0.4])
        models = {
            'model_a': (y_true, y_prob_a),
            'model_b': (y_true, y_prob_b),
        }
        result = compare_discrimination(models)
        assert isinstance(result, dict)
        assert 'model_a' in result
        assert 'model_b' in result

    def test_result_contains_auc_keys(self):
        """Each model's result contains roc_auc and pr_auc."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        models = {'test_model': (y_true, y_prob)}
        result = compare_discrimination(models)
        assert 'roc_auc' in result['test_model']
        assert 'pr_auc' in result['test_model']


# =============================================================================
#                    COMPARE ROC CURVES TESTS
# =============================================================================

class TestCompareROCCurves:
    """Tests for compare_roc_curves(models)."""

    def test_returns_dict_of_roc_results(self):
        """compare_roc_curves returns a dict of model_name -> ROCResult."""
        y_true = np.array([0, 1, 1, 0])
        y_prob_a = np.array([0.1, 0.9, 0.8, 0.3])
        y_prob_b = np.array([0.2, 0.7, 0.6, 0.4])
        models = {
            'model_a': (y_true, y_prob_a),
            'model_b': (y_true, y_prob_b),
        }
        result = compare_roc_curves(models)
        assert isinstance(result, dict)
        assert 'model_a' in result
        assert 'model_b' in result
        assert isinstance(result['model_a'], ROCResult)
        assert isinstance(result['model_b'], ROCResult)


# =============================================================================
#                    DELONG TEST TESTS
# =============================================================================

class TestDeLongTest:
    """Tests for delong_test(y_true, y_prob1, y_prob2)."""

    def test_returns_tuple_of_two(self):
        """delong_test returns (z_statistic, p_value)."""
        y_true = np.array([0, 1, 1, 0])
        y_prob1 = np.array([0.1, 0.9, 0.8, 0.3])
        y_prob2 = np.array([0.2, 0.7, 0.6, 0.4])
        result = delong_test(y_true, y_prob1, y_prob2)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_p_value_in_range(self):
        """p-value is between 0 and 1."""
        y_true = np.array([0, 1, 1, 0])
        y_prob1 = np.array([0.1, 0.9, 0.8, 0.3])
        y_prob2 = np.array([0.2, 0.7, 0.6, 0.4])
        z_stat, p_value = delong_test(y_true, y_prob1, y_prob2)
        assert 0.0 <= p_value <= 1.0

    def test_identical_models(self):
        """Identical models should have p_value close to 1 (no significant difference)."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        z_stat, p_value = delong_test(y_true, y_prob, y_prob)
        # z_stat should be 0 and p_value should be 1
        assert z_stat == pytest.approx(0.0, abs=1e-10)
        assert p_value == pytest.approx(1.0, abs=0.01)


# =============================================================================
#                    SUBGROUP DISCRIMINATION TESTS
# =============================================================================

class TestSubgroupDiscrimination:
    """Tests for subgroup_discrimination(y_true, y_prob, groups, config)."""

    def test_returns_dict_keyed_by_group(self):
        """subgroup_discrimination returns a dict keyed by group label."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        groups = np.array(['A', 'A', 'B', 'B'])
        result = subgroup_discrimination(y_true, y_prob, groups)
        assert isinstance(result, dict)
        assert 'A' in result
        assert 'B' in result

    def test_subgroup_metrics_contain_sample_counts(self):
        """Each subgroup result includes sample count information."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        groups = np.array(['A', 'A', 'B', 'B'])
        result = subgroup_discrimination(y_true, y_prob, groups)
        for group, metrics in result.items():
            assert 'n_samples' in metrics
            assert 'n_positive' in metrics
            assert 'n_negative' in metrics


# =============================================================================
#                    VALIDATE DISCRIMINATION INPUTS TESTS
# =============================================================================

class TestValidateDiscriminationInputs:
    """Tests for validate_discrimination_inputs(y_true, y_prob)."""

    def test_valid_inputs_no_error(self):
        """Valid inputs return True."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = validate_discrimination_inputs(y_true, y_prob)
        assert result is True

    def test_nan_raises(self):
        """NaN values raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            validate_discrimination_inputs(
                np.array([0, np.nan, 1]), np.array([0.1, 0.5, 0.9])
            )

    def test_length_mismatch_raises(self):
        """Different lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            validate_discrimination_inputs(
                np.array([0, 1, 1]), np.array([0.1, 0.9])
            )

    def test_non_binary_raises(self):
        """Non-binary y_true raises ValueError."""
        with pytest.raises(ValueError, match="binary"):
            validate_discrimination_inputs(
                np.array([0, 2, 1]), np.array([0.1, 0.5, 0.9])
            )

    def test_prob_out_of_range_raises(self):
        """y_prob values outside [0,1] raise ValueError."""
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            validate_discrimination_inputs(
                np.array([0, 1]), np.array([-0.1, 0.5])
            )

    def test_no_positive_raises(self):
        """All-negative y_true raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            validate_discrimination_inputs(
                np.array([0, 0, 0]), np.array([0.1, 0.5, 0.9])
            )

    def test_no_negative_raises(self):
        """All-positive y_true raises ValueError."""
        with pytest.raises(ValueError, match="negative"):
            validate_discrimination_inputs(
                np.array([1, 1, 1]), np.array([0.1, 0.5, 0.9])
            )


# =============================================================================
#                    RANK ORDER STATISTICS TESTS
# =============================================================================

class TestRankOrderStatistics:
    """Tests for rank_order_statistics(y_true, y_prob)."""

    def test_returns_dict(self):
        """rank_order_statistics returns a dictionary."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = rank_order_statistics(y_true, y_prob)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        """Result contains concordance, discordance, ties, somers_d."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = rank_order_statistics(y_true, y_prob)
        expected_keys = {'concordance', 'discordance', 'ties', 'somers_d'}
        assert expected_keys.issubset(result.keys())

    def test_concordance_discordance_ties_sum_to_one(self):
        """Concordance + discordance + ties = 1.0."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = rank_order_statistics(y_true, y_prob)
        total = result['concordance'] + result['discordance'] + result['ties']
        assert total == pytest.approx(1.0)

    def test_perfect_predictions_all_concordant(self):
        """Perfect predictions yield concordance=1.0."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        result = rank_order_statistics(y_true, y_prob)
        assert result['concordance'] == pytest.approx(1.0)
        assert result['discordance'] == pytest.approx(0.0)

    def test_somers_d_range(self):
        """Somers' D = 2*AUC - 1, so in [-1, 1]."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = rank_order_statistics(y_true, y_prob)
        assert -1.0 <= result['somers_d'] <= 1.0

    def test_somers_d_matches_auc(self):
        """Somers' D = 2*AUC - 1."""
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        result = rank_order_statistics(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        expected_d = 2 * auc - 1
        assert result['somers_d'] == pytest.approx(expected_d, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
