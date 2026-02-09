"""
Unit Tests for Evaluation Visualization Module
================================================

Tests for src/visualization/eval_viz.py -- EvalPlotConfig dataclass
and all evaluation plotting functions.

All functions are implemented and return (fig, ax) tuples.
Tests verify return types, figure/axes structure, and basic properties.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.figure
from dataclasses import fields

from src.visualization.eval_viz import (
    EvalPlotConfig,
    plot_reliability_diagram,
    plot_calibration_comparison,
    plot_brier_decomposition,
    plot_expected_calibration_error,
    plot_roc_curve,
    plot_roc_comparison,
    plot_pr_curve,
    plot_lift_chart,
    plot_cumulative_gains,
    plot_confusion_matrix,
    plot_threshold_metrics,
    plot_precision_recall_tradeoff,
    plot_subgroup_performance,
    plot_fairness_comparison,
    plot_performance_over_time,
    plot_prediction_distribution,
    compute_calibration_curve,
    compute_brier_decomposition,
)


# ------------------------------------------------------------------ helpers
_Y_TRUE = np.array([0, 1, 1, 0, 1, 0])
_Y_PROB = np.array([0.2, 0.8, 0.7, 0.3, 0.9, 0.1])
_Y_PRED = np.array([0, 1, 1, 0, 1, 0])


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to avoid memory leaks."""
    yield
    plt.close('all')


# ==========================================================================
#  TestEvalPlotConfig
# ==========================================================================
class TestEvalPlotConfig:
    """Tests for the EvalPlotConfig dataclass."""

    def test_default_values(self):
        """Default field values match the specification."""
        cfg = EvalPlotConfig()
        assert cfg.figsize == (10, 6)
        assert cfg.dpi == 100
        assert cfg.n_bins == 10
        assert cfg.ci_alpha == pytest.approx(0.1)
        assert cfg.show_histogram is True
        assert cfg.show_ci is True

    def test_custom_values(self):
        """Custom values override defaults correctly."""
        cfg = EvalPlotConfig(
            figsize=(14, 10),
            dpi=200,
            n_bins=20,
            ci_alpha=0.05,
            show_histogram=False,
            show_ci=False,
        )
        assert cfg.figsize == (14, 10)
        assert cfg.dpi == 200
        assert cfg.n_bins == 20
        assert cfg.ci_alpha == pytest.approx(0.05)
        assert cfg.show_histogram is False
        assert cfg.show_ci is False

    def test_field_names(self):
        """All expected field names are present on the dataclass."""
        expected = {
            "figsize",
            "dpi",
            "n_bins",
            "ci_alpha",
            "diagonal_color",
            "good_color",
            "bad_color",
            "neutral_color",
            "show_histogram",
            "show_ci",
            "model_colors",
        }
        actual = {f.name for f in fields(EvalPlotConfig)}
        assert expected == actual


# ==========================================================================
#  TestCalibrationViz
# ==========================================================================
class TestCalibrationViz:
    """Tests for calibration-related plotting functions."""

    def test_plot_reliability_diagram(self):
        fig, ax = plot_reliability_diagram(_Y_TRUE, _Y_PROB)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_reliability_diagram_no_histogram(self):
        fig, ax = plot_reliability_diagram(_Y_TRUE, _Y_PROB, show_histogram=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_calibration_comparison(self):
        y_true = np.array([0, 1, 1, 0])
        models = {
            "Model A": (y_true, np.array([0.2, 0.8, 0.7, 0.3])),
            "Model B": (y_true, np.array([0.3, 0.7, 0.6, 0.4])),
        }
        fig, ax = plot_calibration_comparison(models)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_brier_decomposition(self):
        fig, ax = plot_brier_decomposition(
            uncertainty=0.25,
            resolution=0.15,
            reliability=0.02,
            brier_score=0.12,
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_expected_calibration_error(self):
        ece_values = {
            "Logistic": 0.085,
            "Ridge": 0.052,
            "Baseline": 0.031,
        }
        fig, ax = plot_expected_calibration_error(ece_values)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None


# ==========================================================================
#  TestDiscriminationViz
# ==========================================================================
class TestDiscriminationViz:
    """Tests for discrimination-related plotting functions."""

    def test_plot_roc_curve(self):
        fig, ax = plot_roc_curve(_Y_TRUE, _Y_PROB)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_roc_comparison(self):
        y_true = np.array([0, 1, 1, 0])
        models = {
            "Model A": (y_true, np.array([0.2, 0.8, 0.7, 0.3])),
            "Model B": (y_true, np.array([0.3, 0.7, 0.6, 0.4])),
        }
        fig, ax = plot_roc_comparison(models)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_pr_curve(self):
        fig, ax = plot_pr_curve(_Y_TRUE, _Y_PROB)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_lift_chart(self):
        fig, ax = plot_lift_chart(_Y_TRUE, _Y_PROB)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_cumulative_gains(self):
        fig, ax = plot_cumulative_gains(_Y_TRUE, _Y_PROB)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_confusion_matrix(self):
        fig, ax = plot_confusion_matrix(_Y_TRUE, _Y_PRED)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None


# ==========================================================================
#  TestThresholdViz
# ==========================================================================
class TestThresholdViz:
    """Tests for threshold analysis plotting functions."""

    def test_plot_threshold_metrics(self):
        fig, ax = plot_threshold_metrics(_Y_TRUE, _Y_PROB)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_precision_recall_tradeoff(self):
        fig, ax = plot_precision_recall_tradeoff(_Y_TRUE, _Y_PROB)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None


# ==========================================================================
#  TestSubgroupViz
# ==========================================================================
class TestSubgroupViz:
    """Tests for subgroup and fairness plotting functions."""

    def test_plot_subgroup_performance(self):
        metrics = {
            "Ontario": {"auc": 0.85, "n": 1000},
            "Quebec": {"auc": 0.82, "n": 500},
            "BC": {"auc": 0.78, "n": 300},
        }
        fig, ax = plot_subgroup_performance(metrics, metric_name="auc")
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_fairness_comparison(self):
        group_metrics = {
            "Ontario": {"demographic_parity": 0.9, "equal_opportunity": 0.85},
            "Quebec": {"demographic_parity": 0.8, "equal_opportunity": 0.75},
        }
        fig, ax = plot_fairness_comparison(group_metrics)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_performance_over_time(self):
        metrics_by_period = {
            "auc": [0.85, 0.84, 0.83, 0.80],
            "precision": [0.70, 0.69, 0.68, 0.65],
        }
        period_labels = ["Q1", "Q2", "Q3", "Q4"]
        fig, ax = plot_performance_over_time(metrics_by_period, period_labels)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None


# ==========================================================================
#  TestTemporalViz
# ==========================================================================
class TestTemporalViz:
    """Tests for prediction distribution plotting."""

    def test_plot_prediction_distribution_no_outcome(self):
        fig, ax = plot_prediction_distribution(_Y_PROB)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_prediction_distribution_with_outcome(self):
        fig, ax = plot_prediction_distribution(_Y_PROB, by_outcome=_Y_TRUE)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None


# ==========================================================================
#  TestUtilityFunctions
# ==========================================================================
class TestUtilityFunctions:
    """Tests for utility / computation helper functions."""

    def test_compute_calibration_curve(self):
        mean_pred, actual_frac, counts = compute_calibration_curve(_Y_TRUE, _Y_PROB, n_bins=5)
        assert isinstance(mean_pred, np.ndarray)
        assert isinstance(actual_frac, np.ndarray)
        assert isinstance(counts, np.ndarray)
        assert len(mean_pred) == len(actual_frac) == len(counts)
        # All mean predictions should be in [0, 1]
        assert np.all(mean_pred >= 0) and np.all(mean_pred <= 1)
        # All actual fractions should be in [0, 1]
        assert np.all(actual_frac >= 0) and np.all(actual_frac <= 1)
        # All counts should be positive
        assert np.all(counts > 0)
        # Sum of counts should equal total samples
        assert np.sum(counts) == len(_Y_TRUE)

    def test_compute_brier_decomposition(self):
        result = compute_brier_decomposition(_Y_TRUE, _Y_PROB, n_bins=5)
        assert isinstance(result, dict)
        assert "brier" in result
        assert "uncertainty" in result
        assert "resolution" in result
        assert "reliability" in result
        # Brier score should be non-negative
        assert result["brier"] >= 0
        # Uncertainty = y_bar * (1 - y_bar), with y_bar = 0.5 => 0.25
        assert result["uncertainty"] == pytest.approx(0.25)
        # Brier = uncertainty - resolution + reliability
        expected_brier = result["uncertainty"] - result["resolution"] + result["reliability"]
        assert result["brier"] == pytest.approx(expected_brier, abs=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
