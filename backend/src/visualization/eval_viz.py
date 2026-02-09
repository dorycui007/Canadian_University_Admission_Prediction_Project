"""
Evaluation Visualizations for Grade Prediction System
======================================================

This module provides visualizations for model evaluation metrics, calibration,
and performance analysis. It enables data-driven decision making about model
quality and fairness.

STA257 REFERENCES (Probability Theory):
    - Section 2.3: Conditional probability (reliability diagrams)
    - Section 3.1: Expected value (Brier score)
    - Section 4.2: Variance decomposition (Brier decomposition)

CSC148 REFERENCES:
    - Section 1.3-1.5: Function design recipe (all functions)
    - Section 3.1-3.4: OOP and dataclasses (EvalConfig)

==============================================================================
                    SYSTEM ARCHITECTURE - WHERE THIS MODULE FITS
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                      EVALUATION PIPELINE OVERVIEW                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌───────────────┐     ┌───────────────┐     ┌───────────────┐             │
│   │  src/models   │────▶│ src/evaluation│────▶│  THIS MODULE  │             │
│   │               │     │               │     │   eval_viz    │             │
│   │ predictions   │     │ • metrics.py  │     │               │             │
│   │    y_prob     │     │ • calibration │     │ Matplotlib    │             │
│   │               │     │ • validators  │     │ Plotly        │             │
│   └───────────────┘     └───────────────┘     └───────────────┘             │
│                                                      │                       │
│                                                      ▼                       │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    VISUALIZATION CATEGORIES                          │   │
│   ├─────────────────────────────────────────────────────────────────────┤   │
│   │                                                                      │   │
│   │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │   │
│   │  │  CALIBRATION  │  │DISCRIMINATION │  │   FAIRNESS    │            │   │
│   │  │               │  │               │  │               │            │   │
│   │  │ • Reliability │  │ • ROC curve   │  │ • Subgroup    │            │   │
│   │  │   diagram     │  │ • PR curve    │  │   performance │            │   │
│   │  │ • Brier       │  │ • Lift chart  │  │ • Parity      │            │   │
│   │  │   decomp      │  │ • Confusion   │  │   comparison  │            │   │
│   │  └───────────────┘  └───────────────┘  └───────────────┘            │   │
│   │                                                                      │   │
│   │  ┌───────────────┐  ┌───────────────┐                               │   │
│   │  │  THRESHOLDS   │  │   TEMPORAL    │                               │   │
│   │  │               │  │               │                               │   │
│   │  │ • Metrics vs  │  │ • Performance │                               │   │
│   │  │   threshold   │  │   over time   │                               │   │
│   │  │ • Optimal     │  │ • Drift       │                               │   │
│   │  │   selection   │  │   detection   │                               │   │
│   │  └───────────────┘  └───────────────┘                               │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

==============================================================================
                    UNDERSTANDING CALIBRATION VS DISCRIMINATION
==============================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                          │
    │   A model can be good at DISCRIMINATION but bad at CALIBRATION:          │
    │   ──────────────────────────────────────────────────────────────         │
    │                                                                          │
    │   DISCRIMINATION (Separability):                                         │
    │   "Can the model rank applicants correctly?"                             │
    │                                                                          │
    │   • Measured by: AUC-ROC, precision, recall                              │
    │   • High AUC means: admitted students ranked higher than rejected        │
    │   • Doesn't care about actual probability values                         │
    │                                                                          │
    │   CALIBRATION (Probability correctness):                                 │
    │   "When model says 70%, does 70% actually get admitted?"                 │
    │                                                                          │
    │   • Measured by: Brier score, reliability diagram                        │
    │   • Good calibration means: predicted probabilities are accurate         │
    │   • Critical for: uncertainty quantification, decision thresholds        │
    │                                                                          │
    │   ┌────────────────────────────────────────────────────────────┐        │
    │   │                                                            │        │
    │   │   Example of GOOD discrimination + BAD calibration:        │        │
    │   │                                                            │        │
    │   │   Model predicts:  [0.95, 0.90, 0.85, 0.80, 0.10, 0.05]   │        │
    │   │   Actual outcomes: [1,    1,    1,    0,    0,    0   ]   │        │
    │   │                                                            │        │
    │   │   • AUC = 0.89 (good ranking!)                             │        │
    │   │   • But 0.85 is way too confident (should be ~0.67)        │        │
    │   │                                                            │        │
    │   │   This model is OVERCONFIDENT - needs calibration!         │        │
    │   │                                                            │        │
    │   └────────────────────────────────────────────────────────────┘        │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

==============================================================================
                    USAGE EXAMPLES
==============================================================================

    Example 1: Check if model is well-calibrated
    ─────────────────────────────────────────────
    >>> from src.visualization.eval_viz import plot_reliability_diagram
    >>> from src.models.logistic import LogisticRegression
    >>>
    >>> model = LogisticRegression()
    >>> model.fit(X_train, y_train)
    >>> y_prob = model.predict_proba(X_test)
    >>>
    >>> fig, ax = plot_reliability_diagram(y_test, y_prob)
    >>> plt.show()
    >>>
    >>> # If points are ABOVE diagonal → model is underconfident
    >>> # If points are BELOW diagonal → model is overconfident


    Example 2: Compare models using ROC curves
    ───────────────────────────────────────────
    >>> from src.visualization.eval_viz import plot_roc_comparison
    >>>
    >>> models = {
    ...     "Logistic (no regularization)": (y_test, probs_no_reg),
    ...     "Logistic (ridge λ=0.1)": (y_test, probs_ridge),
    ...     "Baseline (prior)": (y_test, probs_baseline),
    ... }
    >>>
    >>> fig, ax = plot_roc_comparison(models)
    >>> plt.show()


    Example 3: Fairness analysis across provinces
    ──────────────────────────────────────────────
    >>> from src.visualization.eval_viz import plot_subgroup_performance
    >>>
    >>> # Compute AUC for each province
    >>> subgroup_metrics = {
    ...     "Ontario": {"auc": 0.85, "n": 1000},
    ...     "Quebec": {"auc": 0.82, "n": 500},
    ...     "BC": {"auc": 0.78, "n": 300},
    ... }
    >>>
    >>> fig, ax = plot_subgroup_performance(subgroup_metrics, metric_name="auc")
    >>> plt.show()

==============================================================================
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =============================================================================
#                              CONFIGURATION
# =============================================================================

@dataclass
class EvalPlotConfig:
    """
    Configuration for evaluation plot styling.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                       EVAL PLOT CONFIGURATION                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   ┌─────────────────────┐                                                │
    │   │  EvalPlotConfig     │                                                │
    │   ├─────────────────────┤                                                │
    │   │ figsize: (10, 6)    │◄─── Default figure size                        │
    │   │ dpi: 100            │◄─── Resolution for exports                     │
    │   │ n_bins: 10          │◄─── Calibration bin count                      │
    │   │ ci_alpha: 0.1       │◄─── Confidence interval shading                │
    │   │ diagonal_color: gray│◄─── Reference line color                       │
    │   │ good_color: green   │◄─── Good performance indicator                 │
    │   │ bad_color: red      │◄─── Bad performance indicator                  │
    │   │ threshold_colors    │◄─── For threshold selection plots              │
    │   └─────────────────────┘                                                │
    │                                                                          │
    │   CALIBRATION-SPECIFIC SETTINGS:                                         │
    │   ───────────────────────────────                                         │
    │   • n_bins: Number of probability bins for reliability diagram           │
    │   • show_histogram: Whether to show prediction distribution              │
    │   • show_ci: Whether to show confidence intervals                        │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    figsize: Tuple[int, int] = (10, 6)
    dpi: int = 100
    n_bins: int = 10
    ci_alpha: float = 0.1
    diagonal_color: str = 'gray'
    good_color: str = '#2ca02c'  # Green
    bad_color: str = '#d62728'   # Red
    neutral_color: str = '#1f77b4'  # Blue
    show_histogram: bool = True
    show_ci: bool = True

    # Color palette for multi-model comparison
    model_colors: List[str] = field(default_factory=lambda: [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
    ])


# =============================================================================
#                      CALIBRATION VISUALIZATIONS
# =============================================================================

def plot_reliability_diagram(y_true: np.ndarray,
                             y_prob: np.ndarray,
                             n_bins: int = 10,
                             ax: Optional[Any] = None,
                             show_histogram: bool = True,
                             show_ci: bool = True,
                             config: Optional[EvalPlotConfig] = None) -> Tuple[Any, Any]:
    """
    Plot reliability diagram (calibration curve).

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                       RELIABILITY DIAGRAM                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   THE MOST IMPORTANT CALIBRATION VISUALIZATION                           │
    │   ─────────────────────────────────────────────                          │
    │                                                                          │
    │   What it shows:                                                         │
    │   • X-axis: Mean predicted probability in each bin                       │
    │   • Y-axis: Actual fraction of positives in each bin                     │
    │   • Perfect calibration = points on the diagonal                         │
    │                                                                          │
    │   Actual Positive Rate                                                   │
    │   1.0 ┤                              ╱ ← Perfect calibration             │
    │       │                           ╱   ○ ← Our model                      │
    │   0.8 ┤                        ╱   ○                                     │
    │       │                     ╱   ○                                        │
    │   0.6 ┤                  ╱   ○                                           │
    │       │               ╱   ○                                              │
    │   0.4 ┤            ╱   ○                                                 │
    │       │         ╱   ○                                                    │
    │   0.2 ┤      ╱   ○                                                       │
    │       │   ╱   ○                                                          │
    │   0.0 ○──────────────────────────────────────→                           │
    │       0.0   0.2   0.4   0.6   0.8   1.0                                  │
    │                 Mean Predicted Probability                               │
    │                                                                          │
    │   INTERPRETATION:                                                        │
    │   ────────────────                                                       │
    │                                                                          │
    │   Points ABOVE diagonal:                                                 │
    │   ┌─────────────────────────────────────────────────────────┐           │
    │   │ Model is UNDERCONFIDENT                                  │           │
    │   │ • Model says 40% but actual is 60%                       │           │
    │   │ • Probabilities are too low                              │           │
    │   │ • Fix: Decrease temperature / recalibrate                │           │
    │   └─────────────────────────────────────────────────────────┘           │
    │                                                                          │
    │   Points BELOW diagonal:                                                 │
    │   ┌─────────────────────────────────────────────────────────┐           │
    │   │ Model is OVERCONFIDENT                                   │           │
    │   │ • Model says 80% but actual is 60%                       │           │
    │   │ • Probabilities are too extreme                          │           │
    │   │ • Fix: Increase regularization / Platt scaling           │           │
    │   └─────────────────────────────────────────────────────────┘           │
    │                                                                          │
    │   HISTOGRAM BELOW (optional):                                            │
    │   ───────────────────────────                                            │
    │   Shows distribution of predictions. Many predictions at 0.5             │
    │   means model is uncertain; clustered at 0/1 means confident.            │
    │                                                                          │
    │       ████                              ████                             │
    │       ████████                      ████████                             │
    │       ████████████              ████████████                             │
    │       ────────────────────────────────────────                           │
    │       0.0      0.5             1.0                                       │
    │                                                                          │
    │   CONFIDENCE INTERVALS:                                                  │
    │   ─────────────────────                                                  │
    │   • Shaded regions show uncertainty due to finite sample size            │
    │   • Wider CI = fewer samples in that probability bin                     │
    │   • Formula: p̂ ± 1.96 * sqrt(p̂(1-p̂)/n)                                │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels (0 or 1), shape (n_samples,)
        y_prob: Predicted probabilities, shape (n_samples,)
        n_bins: Number of bins for grouping predictions (default: 10)
        ax: Matplotlib axes object. If None, creates new figure.
        show_histogram: If True, add histogram of predictions below main plot
        show_ci: If True, show confidence intervals
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)

    Raises:
        ValueError: If y_true contains values other than 0 and 1
        ValueError: If y_prob contains values outside [0, 1]

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_prob = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
        >>> fig, ax = plot_reliability_diagram(y_true, y_prob, n_bins=5)

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Bin predictions: Use np.digitize or np.histogram
    2. For each bin, compute:
       - mean_pred = mean of predictions in bin
       - actual_rate = mean of true labels in bin
       - count = number of samples in bin
    3. Compute CI: ci = 1.96 * np.sqrt(actual_rate * (1-actual_rate) / count)
    4. Plot points with error bars
    5. Add diagonal reference line
    6. If show_histogram: create additional subplot below
    7. Use gridspec for layout with histogram

    MATHEMATICAL NOTES (STA257):
    ────────────────────────────
    For each bin b with n_b samples:
    - Expected value: E[Y | P(Y=1) ∈ bin b] ≈ mean predicted probability
    - If well-calibrated: actual_rate ≈ mean_pred (they should match!)
    """
    config = config or EvalPlotConfig()

    mean_pred, actual_frac, counts = compute_calibration_curve(y_true, y_prob, n_bins)

    if show_histogram and ax is None:
        fig = plt.figure(figsize=config.figsize, dpi=config.dpi)
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
        ax_cal = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
    else:
        if ax is None:
            fig, ax_cal = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        else:
            ax_cal = ax
            fig = ax_cal.figure
        ax_hist = None

    # Diagonal reference line
    ax_cal.plot([0, 1], [0, 1], linestyle='--', color=config.diagonal_color,
                label='Perfect calibration')

    # Calibration curve
    ax_cal.plot(mean_pred, actual_frac, 's-', color=config.neutral_color,
                label='Model')

    # Confidence intervals
    if show_ci and len(counts) > 0:
        ci = 1.96 * np.sqrt(actual_frac * (1.0 - actual_frac) / counts)
        ax_cal.fill_between(mean_pred, actual_frac - ci, actual_frac + ci,
                            alpha=config.ci_alpha, color=config.neutral_color)

    ax_cal.set_xlabel('Mean Predicted Probability')
    ax_cal.set_ylabel('Actual Fraction of Positives')
    ax_cal.set_title('Reliability Diagram')
    ax_cal.set_xlim([0, 1])
    ax_cal.set_ylim([0, 1])
    ax_cal.legend(loc='lower right')

    # Histogram
    if ax_hist is not None:
        ax_hist.hist(y_prob, bins=n_bins, range=(0, 1), color=config.neutral_color,
                     alpha=0.7, edgecolor='black')
        ax_hist.set_xlabel('Predicted Probability')
        ax_hist.set_ylabel('Count')
        ax_hist.set_xlim([0, 1])

    ax = ax_cal
    return (fig, ax)


def plot_calibration_comparison(models: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                n_bins: int = 10,
                                config: Optional[EvalPlotConfig] = None) -> Tuple[Any, Any]:
    """
    Compare calibration across multiple models.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   CALIBRATION COMPARISON                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Overlay reliability diagrams to compare model calibration.             │
    │                                                                          │
    │   Actual Rate                                                            │
    │   1.0 ┤                              ╱                                   │
    │       │                           ╱   ○ Logistic                         │
    │   0.8 ┤                        ╱   ○  ● Ridge                            │
    │       │                     ╱   ○  ●   □ Baseline                        │
    │   0.6 ┤                  ╱   ○  ●   □                                    │
    │       │               ╱   ○  ●  □                                        │
    │   0.4 ┤            ╱   ○  ● □                                            │
    │       │         ╱   ○ ● □                                                │
    │   0.2 ┤      ╱  ○ ●□                                                     │
    │       │   ╱ ○●□                                                          │
    │   0.0 ├──────────────────────────────────→                               │
    │       0.0   0.2   0.4   0.6   0.8   1.0                                  │
    │                                                                          │
    │   OBSERVATIONS:                                                          │
    │   • Ridge (●) is closest to diagonal → best calibrated                  │
    │   • Baseline (□) is overconfident at high probabilities                 │
    │   • Logistic (○) is underconfident overall                              │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        models: Dict mapping model name to (y_true, y_prob) tuples
        n_bins: Number of bins for calibration curves
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> models = {
        ...     "Logistic": (y_test, probs_logistic),
        ...     "Ridge λ=0.1": (y_test, probs_ridge),
        ...     "Baseline": (y_test, probs_baseline),
        ... }
        >>> fig, ax = plot_calibration_comparison(models)

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Iterate over models, compute calibration curve for each
    2. Use different colors/markers for each model
    3. Add legend with model names
    4. Consider adding Brier score in legend: "Model (Brier=0.15)"
    """
    config = config or EvalPlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Diagonal reference
    ax.plot([0, 1], [0, 1], linestyle='--', color=config.diagonal_color,
            label='Perfect calibration')

    markers = ['s', 'o', '^', 'D', 'v', 'P']
    for i, (name, (y_true, y_prob)) in enumerate(models.items()):
        mean_pred, actual_frac, counts = compute_calibration_curve(y_true, y_prob, n_bins)
        brier = float(np.mean((y_true - y_prob) ** 2))
        color = config.model_colors[i % len(config.model_colors)]
        marker = markers[i % len(markers)]
        ax.plot(mean_pred, actual_frac, marker=marker, linestyle='-',
                color=color, label=f'{name} (Brier={brier:.3f})')

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Actual Fraction of Positives')
    ax.set_title('Calibration Comparison')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right')

    return (fig, ax)


def plot_brier_decomposition(uncertainty: float,
                             resolution: float,
                             reliability: float,
                             brier_score: float,
                             model_name: Optional[str] = None,
                             config: Optional[EvalPlotConfig] = None) -> Tuple[Any, Any]:
    """
    Visualize Brier score decomposition.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     BRIER DECOMPOSITION                                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   BRIER SCORE = UNCERTAINTY - RESOLUTION + RELIABILITY                  │
    │                                                                          │
    │   ┌───────────────────────────────────────────────────────────┐         │
    │   │                                                           │         │
    │   │   Uncertainty (UNC) = ȳ(1-ȳ)                              │         │
    │   │   ══════════════════════════════════════════════════════  │         │
    │   │   Property of the DATA, not the model.                    │         │
    │   │   Maximum at ȳ = 0.5 (50% base rate).                     │         │
    │   │                                                           │         │
    │   │   Resolution (RES) = (1/N) Σ_k n_k (ō_k - ȳ)²            │         │
    │   │   ══════════════════════════════════════════════════════  │         │
    │   │   Model's ability to SEPARATE classes.                    │         │
    │   │   Higher is BETTER. Similar to discrimination.            │         │
    │   │                                                           │         │
    │   │   Reliability (REL) = (1/N) Σ_k n_k (f̄_k - ō_k)²        │         │
    │   │   ══════════════════════════════════════════════════════  │         │
    │   │   Model's CALIBRATION error.                              │         │
    │   │   Lower is BETTER. Perfect = 0.                           │         │
    │   │                                                           │         │
    │   └───────────────────────────────────────────────────────────┘         │
    │                                                                          │
    │   STACKED BAR VISUALIZATION:                                             │
    │   ────────────────────────────                                           │
    │                                                                          │
    │   ┌────────────────────────────────────────────────────────┐            │
    │   │                 Uncertainty (UNC) = 0.25                │ ← data    │
    │   ├────────────────────────────────────────────────────────┤            │
    │   │    Resolution (RES) = 0.15     │ + │ Brier = 0.12      │            │
    │   │    (higher = better)           │   │                   │            │
    │   ├────────────────────────────────┤   │                   │            │
    │   │ Reliability = 0.02             │   │                   │            │
    │   │ (lower = better)               │   │                   │            │
    │   └────────────────────────────────┴───┴───────────────────┘            │
    │                                                                          │
    │   INTERPRETATION FOR OUR PROJECT:                                        │
    │   ─────────────────────────────────                                      │
    │   • High RES: Model knows which applicants are more likely to succeed   │
    │   • Low REL: Model's probabilities are trustworthy                      │
    │   • Brier: Overall score (lower is better), max = uncertainty           │
    │                                                                          │
    │   QUALITY CHECK:                                                         │
    │   ───────────────                                                        │
    │   • Brier should always equal: UNC - RES + REL                          │
    │   • If Brier > UNC: model is worse than predicting base rate!           │
    │   • Good model: high RES, low REL                                       │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        uncertainty: Dataset uncertainty component (property of data)
        resolution: Model resolution component (higher is better)
        reliability: Model reliability component (lower is better)
        brier_score: Total Brier score (should equal UNC - RES + REL)
        model_name: Optional model name for title
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)

    Preconditions:
        - All values should be in [0, 0.25] (max Brier is 0.25)
        - brier_score ≈ uncertainty - resolution + reliability

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Create stacked bar chart showing components
    2. Use colors: neutral for UNC, good for RES, bad for REL
    3. Add text annotations with values
    4. Add verification: check if Brier ≈ UNC - RES + REL
    5. Consider horizontal layout with arrows showing decomposition
    """
    config = config or EvalPlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    components = ['Uncertainty', 'Resolution', 'Reliability', 'Brier Score']
    values = [uncertainty, resolution, reliability, brier_score]
    colors = [config.neutral_color, config.good_color, config.bad_color, '#7f7f7f']

    bars = ax.bar(components, values, color=colors, edgecolor='black', width=0.6)

    # Annotate values on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    title = 'Brier Score Decomposition'
    if model_name:
        title += f' - {model_name}'
    ax.set_title(title)
    ax.set_ylabel('Value')
    ax.set_ylim(0, max(values) * 1.2 + 0.01)

    return (fig, ax)


def plot_expected_calibration_error(ece_values: Dict[str, float],
                                    config: Optional[EvalPlotConfig] = None) -> Tuple[Any, Any]:
    """
    Bar chart comparing Expected Calibration Error across models.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                  EXPECTED CALIBRATION ERROR (ECE)                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   ECE = Σ_k (n_k/N) |actual_rate_k - predicted_prob_k|                  │
    │                                                                          │
    │   Weighted average of calibration error across bins.                     │
    │   Lower is better. Perfect calibration = 0.                              │
    │                                                                          │
    │   ECE (%)                                                                │
    │   │                                                                      │
    │   │ ████ 8.5%  Logistic (no reg)                                        │
    │   │ ████                                                                 │
    │   │ ████  ████ 5.2%  Logistic + Ridge                                   │
    │   │ ████  ████                                                           │
    │   │ ████  ████  ████ 3.1%  Ridge + Platt                                │
    │   │ ████  ████  ████                                                     │
    │   │ ████  ████  ████  ████ 2.0%  Ridge + Isotonic                       │
    │   └─────────────────────────────────────→                                │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        ece_values: Dict mapping model name to ECE value
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)
    """
    config = config or EvalPlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    names = list(ece_values.keys())
    values = [ece_values[n] for n in names]
    colors = [config.model_colors[i % len(config.model_colors)] for i in range(len(names))]

    bars = ax.bar(names, values, color=colors, edgecolor='black', width=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Model')
    ax.set_ylabel('Expected Calibration Error (ECE)')
    ax.set_title('ECE Comparison')
    ax.set_ylim(0, max(values) * 1.3 + 0.001)

    return (fig, ax)


# =============================================================================
#                    DISCRIMINATION VISUALIZATIONS
# =============================================================================

def plot_roc_curve(y_true: np.ndarray,
                   y_prob: np.ndarray,
                   ax: Optional[Any] = None,
                   show_auc: bool = True,
                   show_optimal_threshold: bool = False,
                   config: Optional[EvalPlotConfig] = None) -> Tuple[Any, Any]:
    """
    Plot ROC curve with AUC annotation.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                          ROC CURVE                                       │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   RECEIVER OPERATING CHARACTERISTIC (ROC):                               │
    │   ─────────────────────────────────────────                              │
    │   Shows tradeoff between True Positive Rate and False Positive Rate.     │
    │                                                                          │
    │   True Positive Rate (Sensitivity, Recall)                               │
    │   1.0 ┤                    ┌───────────────○                             │
    │       │               ┌────┘                                             │
    │   0.8 ┤          ┌────┘                                                  │
    │       │      ┌───┘                                                       │
    │   0.6 ┤   ┌──┘           ○ Optimal threshold                             │
    │       │  ┌┘                 (Youden's J)                                 │
    │   0.4 ┤ ┌┘                                                               │
    │       │┌┘                                                                │
    │   0.2 ┤│         ╱ Random classifier                                    │
    │       ││      ╱    (AUC = 0.5)                                           │
    │   0.0 ○────────────────────────────────→                                 │
    │       0.0   0.2   0.4   0.6   0.8   1.0                                  │
    │               False Positive Rate                                        │
    │               (1 - Specificity)                                          │
    │                                                                          │
    │   AUC = 0.85                                                             │
    │                                                                          │
    │   DEFINITIONS:                                                           │
    │   ─────────────                                                          │
    │                                                                          │
    │   TPR = TP / (TP + FN) = P(predict + | actual +)                        │
    │       "Of all admitted students, what fraction did we predict?"          │
    │                                                                          │
    │   FPR = FP / (FP + TN) = P(predict + | actual -)                        │
    │       "Of all rejected students, what fraction did we falsely predict?"  │
    │                                                                          │
    │   INTERPRETATION:                                                        │
    │   ────────────────                                                       │
    │   • Upper-left = good (high TPR, low FPR)                               │
    │   • Diagonal = random (AUC = 0.5)                                        │
    │   • Below diagonal = worse than random (flip predictions!)               │
    │                                                                          │
    │   AUC INTERPRETATION:                                                    │
    │   ───────────────────                                                    │
    │   P(score(positive) > score(negative)) = AUC                             │
    │   "Probability that randomly chosen admitted student is ranked           │
    │    higher than randomly chosen rejected student"                         │
    │                                                                          │
    │   AUC ~ 0.50: Random (useless)                                          │
    │   AUC ~ 0.70: Acceptable                                                │
    │   AUC ~ 0.80: Good                                                      │
    │   AUC ~ 0.90: Excellent                                                 │
    │   AUC ~ 0.99: Almost perfect (check for data leakage!)                  │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        ax: Matplotlib axes
        show_auc: If True, display AUC value in legend
        show_optimal_threshold: If True, mark Youden's J optimal point
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Compute ROC: fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_prob)
    2. Compute AUC: auc = sklearn.metrics.roc_auc_score(y_true, y_prob)
    3. Plot curve: ax.plot(fpr, tpr)
    4. Add diagonal: ax.plot([0,1], [0,1], linestyle='--')
    5. If show_optimal_threshold:
       - Youden's J = TPR - FPR
       - optimal_idx = np.argmax(tpr - fpr)
       - Mark point at (fpr[optimal_idx], tpr[optimal_idx])
    6. Add legend with AUC value
    """
    config = config or EvalPlotConfig()

    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    else:
        fig = ax.figure

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Compute ROC manually: sort by scores descending
    desc_idx = np.argsort(-y_prob)
    y_sorted = y_true[desc_idx]
    scores_sorted = y_prob[desc_idx]

    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)

    # Walk through sorted scores and compute cumulative TP and FP
    tp = 0.0
    fp = 0.0
    tpr_list = [0.0]
    fpr_list = [0.0]
    thresholds_list = []
    prev_score = None

    for i in range(len(y_sorted)):
        if prev_score is not None and scores_sorted[i] != prev_score:
            tpr_val = tp / total_pos if total_pos > 0 else 0.0
            fpr_val = fp / total_neg if total_neg > 0 else 0.0
            tpr_list.append(tpr_val)
            fpr_list.append(fpr_val)
            thresholds_list.append(prev_score)
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        prev_score = scores_sorted[i]

    # Final point
    tpr_list.append(tp / total_pos if total_pos > 0 else 0.0)
    fpr_list.append(fp / total_neg if total_neg > 0 else 0.0)
    if prev_score is not None:
        thresholds_list.append(prev_score)

    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)

    # Compute AUC using trapezoidal rule
    auc_val = float(np.abs(np.trapz(tpr_arr, fpr_arr)))

    # Plot ROC curve
    label = 'ROC curve'
    if show_auc:
        label += f' (AUC = {auc_val:.3f})'
    ax.plot(fpr_arr, tpr_arr, color=config.neutral_color, lw=2, label=label)

    # Diagonal reference
    ax.plot([0, 1], [0, 1], linestyle='--', color=config.diagonal_color,
            label='Random (AUC = 0.5)')

    # Optimal threshold (Youden's J)
    if show_optimal_threshold and len(thresholds_list) > 0:
        # Use the points that have corresponding thresholds (indices 1..len(thresholds_list))
        j_scores = tpr_arr[1:len(thresholds_list) + 1] - fpr_arr[1:len(thresholds_list) + 1]
        opt_idx = int(np.argmax(j_scores))
        opt_fpr = fpr_arr[opt_idx + 1]
        opt_tpr = tpr_arr[opt_idx + 1]
        ax.plot(opt_fpr, opt_tpr, 'ro', markersize=10,
                label=f'Optimal (t={thresholds_list[opt_idx]:.2f})')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.legend(loc='lower right')

    return (fig, ax)


def plot_roc_comparison(models: Dict[str, Tuple[np.ndarray, np.ndarray]],
                        config: Optional[EvalPlotConfig] = None) -> Tuple[Any, Any]:
    """
    Compare ROC curves across models.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      ROC COMPARISON                                      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   TPR                                                                    │
    │   1.0 ┤                ┌──────────────────                               │
    │       │           ┌────┘                                                 │
    │   0.8 ┤      ┌────┘       ───── Logistic+Ridge (AUC=0.87)               │
    │       │   ┌──┘            ───── Logistic (AUC=0.82)                     │
    │   0.6 ┤  ┌┘               - - - Baseline (AUC=0.65)                     │
    │       │ ┌┘                                                               │
    │   0.4 ┤┌┘                                                                │
    │       ││                                                                 │
    │   0.2 ┤│                                                                 │
    │       │                                                                  │
    │   0.0 ├────────────────────────────────────→                             │
    │       0.0   0.2   0.4   0.6   0.8   1.0                                  │
    │                        FPR                                               │
    │                                                                          │
    │   COMPARISON INSIGHTS:                                                   │
    │   • Ridge regularization improved AUC by 0.05                           │
    │   • Both models significantly beat baseline                              │
    │   • At FPR=0.1, Ridge has TPR=0.6 vs Logistic TPR=0.5                   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        models: Dict mapping model name to (y_true, y_prob) tuples
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)
    """
    config = config or EvalPlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    for i, (name, (y_true, y_prob)) in enumerate(models.items()):
        color = config.model_colors[i % len(config.model_colors)]
        # Compute ROC for this model
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        desc_idx = np.argsort(-y_prob)
        y_sorted = y_true[desc_idx]
        total_pos = np.sum(y_true == 1)
        total_neg = np.sum(y_true == 0)

        tp = 0.0
        fp = 0.0
        tpr_list = [0.0]
        fpr_list = [0.0]

        scores_sorted = y_prob[desc_idx]
        prev_score = None
        for j in range(len(y_sorted)):
            if prev_score is not None and scores_sorted[j] != prev_score:
                tpr_list.append(tp / total_pos if total_pos > 0 else 0.0)
                fpr_list.append(fp / total_neg if total_neg > 0 else 0.0)
            if y_sorted[j] == 1:
                tp += 1
            else:
                fp += 1
            prev_score = scores_sorted[j]
        tpr_list.append(tp / total_pos if total_pos > 0 else 0.0)
        fpr_list.append(fp / total_neg if total_neg > 0 else 0.0)

        tpr_arr = np.array(tpr_list)
        fpr_arr = np.array(fpr_list)
        auc_val = float(np.abs(np.trapz(tpr_arr, fpr_arr)))

        ax.plot(fpr_arr, tpr_arr, color=color, lw=2,
                label=f'{name} (AUC = {auc_val:.3f})')

    ax.plot([0, 1], [0, 1], linestyle='--', color=config.diagonal_color,
            label='Random (AUC = 0.5)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Comparison')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.legend(loc='lower right')

    return (fig, ax)


def plot_pr_curve(y_true: np.ndarray,
                  y_prob: np.ndarray,
                  ax: Optional[Any] = None,
                  show_ap: bool = True,
                  config: Optional[EvalPlotConfig] = None) -> Tuple[Any, Any]:
    """
    Plot Precision-Recall curve.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    PRECISION-RECALL CURVE                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   WHY USE PR INSTEAD OF ROC?                                             │
    │   ────────────────────────────                                           │
    │   • Better for imbalanced datasets (many negatives)                      │
    │   • Focuses on positive class performance                                │
    │   • More sensitive to false positives                                    │
    │                                                                          │
    │   Precision                                                              │
    │   1.0 ┤○                                                                 │
    │       │ ╲                                                                │
    │   0.8 ┤  ○──○                                                            │
    │       │      ╲                                                           │
    │   0.6 ┤       ○──○                                                       │
    │       │           ╲                                                      │
    │   0.4 ┤            ○──○                                                  │
    │       │                ╲                                                 │
    │   0.2 ┤ ─ ─ ─ ─ ─ ─ ─ ─○──○   Random baseline                           │
    │       │                                                                  │
    │   0.0 ┼─────────────────────────────────→                                │
    │       0.0   0.2   0.4   0.6   0.8   1.0                                  │
    │                        Recall                                            │
    │                                                                          │
    │   Average Precision (AP) = 0.72                                          │
    │                                                                          │
    │   DEFINITIONS:                                                           │
    │   ─────────────                                                          │
    │   Precision = TP / (TP + FP)                                            │
    │       "Of those we predicted positive, how many were correct?"           │
    │                                                                          │
    │   Recall = TP / (TP + FN)                                               │
    │       "Of all positives, how many did we catch?"                         │
    │                                                                          │
    │   TRADEOFF:                                                              │
    │   ─────────                                                              │
    │   • High threshold: High precision, low recall                          │
    │   • Low threshold: High recall, low precision                           │
    │                                                                          │
    │   PROJECT CONTEXT:                                                       │
    │   ─────────────────                                                      │
    │   If admission rate is low (e.g., 10%), PR curve is more informative    │
    │   than ROC because it focuses on the rare positive class.               │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        ax: Matplotlib axes
        show_ap: If True, display Average Precision in legend
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Compute PR: precision, recall, _ = sklearn.metrics.precision_recall_curve(...)
    2. Compute AP: ap = sklearn.metrics.average_precision_score(...)
    3. Plot curve: ax.plot(recall, precision)
    4. Add baseline: horizontal line at y = positive_rate
    5. Add legend with AP value
    """
    config = config or EvalPlotConfig()

    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    else:
        fig = ax.figure

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Compute PR curve manually from sorted scores
    desc_idx = np.argsort(-y_prob)
    y_sorted = y_true[desc_idx]
    scores_sorted = y_prob[desc_idx]
    total_pos = np.sum(y_true == 1)

    tp = 0.0
    fp = 0.0
    precision_list = [1.0]
    recall_list = [0.0]

    prev_score = None
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        if prev_score is None or scores_sorted[i] != prev_score or i == len(y_sorted) - 1:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / total_pos if total_pos > 0 else 0.0
            precision_list.append(prec)
            recall_list.append(rec)
        prev_score = scores_sorted[i]

    precision_arr = np.array(precision_list)
    recall_arr = np.array(recall_list)

    # Compute average precision (area under PR curve)
    # Sort by recall for proper integration
    sort_idx = np.argsort(recall_arr)
    recall_sorted = recall_arr[sort_idx]
    precision_sorted = precision_arr[sort_idx]
    ap = float(np.trapz(precision_sorted, recall_sorted))

    label = 'PR curve'
    if show_ap:
        label += f' (AP = {ap:.3f})'
    ax.plot(recall_arr, precision_arr, color=config.neutral_color, lw=2, label=label)

    # Baseline: horizontal line at positive rate
    pos_rate = total_pos / len(y_true) if len(y_true) > 0 else 0.5
    ax.axhline(y=pos_rate, linestyle='--', color=config.diagonal_color,
               label=f'Baseline ({pos_rate:.2f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.legend(loc='upper right')

    return (fig, ax)


def plot_lift_chart(y_true: np.ndarray,
                    y_prob: np.ndarray,
                    n_bins: int = 10,
                    config: Optional[EvalPlotConfig] = None) -> Tuple[Any, Any]:
    """
    Plot lift chart.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                          LIFT CHART                                      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Lift shows how much better the model is than random selection.         │
    │                                                                          │
    │   Lift = (positive rate in top k%) / (overall positive rate)            │
    │                                                                          │
    │   Lift                                                                   │
    │   5.0 ┤●                                                                 │
    │       │ ╲                                                                │
    │   4.0 ┤  ●                                                               │
    │       │   ╲                                                              │
    │   3.0 ┤    ●                                                             │
    │       │     ╲                                                            │
    │   2.0 ┤      ●─●                                                         │
    │       │          ╲                                                       │
    │   1.0 ┤──────────────●───●───●───●───●  Random baseline                  │
    │       │                                                                  │
    │   0.0 ┼────────────────────────────────────→                             │
    │       0%   10%   20%   30%   40%   50%  100%                             │
    │                   % of Population Contacted                              │
    │                                                                          │
    │   INTERPRETATION:                                                        │
    │   ────────────────                                                       │
    │   • Lift = 5 at 10% means: Top 10% by model score has 5x the            │
    │     positive rate compared to random 10% selection                       │
    │   • Lift converges to 1.0 as you include entire population              │
    │   • Area under lift curve is another metric                              │
    │                                                                          │
    │   PROJECT APPLICATION:                                                   │
    │   ─────────────────────                                                  │
    │   "If we only interview top 20% of applicants by model score,            │
    │    we capture 3x more future-admits than random selection"               │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins (deciles if n_bins=10)
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Sort by predicted probability (descending)
    2. Split into n_bins groups
    3. For each group, compute: (positive_rate_in_group) / (overall_positive_rate)
    4. Plot cumulative lift
    5. Add horizontal line at lift=1 (random baseline)
    """
    config = config or EvalPlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Sort by predicted probability descending
    desc_idx = np.argsort(-y_prob)
    y_sorted = y_true[desc_idx]

    overall_pos_rate = np.mean(y_true)
    n = len(y_true)
    bin_size = n // n_bins

    percentages = []
    lifts = []
    for b in range(n_bins):
        start = b * bin_size
        end = start + bin_size if b < n_bins - 1 else n
        bin_labels = y_sorted[start:end]
        bin_pos_rate = np.mean(bin_labels) if len(bin_labels) > 0 else 0.0
        lift = bin_pos_rate / overall_pos_rate if overall_pos_rate > 0 else 1.0
        pct = (b + 1) / n_bins * 100
        percentages.append(pct)
        lifts.append(lift)

    ax.plot(percentages, lifts, 's-', color=config.neutral_color, lw=2, label='Model')
    ax.axhline(y=1.0, linestyle='--', color=config.diagonal_color, label='Random baseline')

    ax.set_xlabel('% of Population (decile)')
    ax.set_ylabel('Lift')
    ax.set_title('Lift Chart')
    ax.legend(loc='upper right')

    return (fig, ax)


def plot_cumulative_gains(y_true: np.ndarray,
                          y_prob: np.ndarray,
                          config: Optional[EvalPlotConfig] = None) -> Tuple[Any, Any]:
    """
    Plot cumulative gains chart.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     CUMULATIVE GAINS CHART                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Shows % of positives captured vs % of population selected.             │
    │                                                                          │
    │   % of Positives Captured                                                │
    │   100% ┤                          ──────○                                │
    │        │                    ─────┘                                       │
    │    80% ┤              ────┘        ← Model curve                         │
    │        │         ────┘                                                   │
    │    60% ┤     ───┘                                                        │
    │        │   ──┘           ╱ Random (diagonal)                            │
    │    40% ┤ ──┘          ╱                                                 │
    │        │──┘        ╱                                                    │
    │    20% ┤        ╱                                                       │
    │        │     ╱                                                          │
    │     0% ┼──────────────────────────────────→                              │
    │        0%    20%    40%    60%    80%   100%                             │
    │               % of Population Selected                                   │
    │                                                                          │
    │   INTERPRETATION:                                                        │
    │   ────────────────                                                       │
    │   At 20% of population: Model captures 60% of positives                 │
    │   Random would only capture 20% of positives                            │
    │                                                                          │
    │   AREA BETWEEN CURVES = measure of model quality                        │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)
    """
    config = config or EvalPlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Sort by predicted probability descending
    desc_idx = np.argsort(-y_prob)
    y_sorted = y_true[desc_idx]

    total_pos = np.sum(y_true == 1)
    n = len(y_true)

    # Cumulative gains
    cum_pos = np.cumsum(y_sorted)
    pct_population = np.arange(1, n + 1) / n * 100
    pct_positives = cum_pos / total_pos * 100 if total_pos > 0 else cum_pos

    # Add origin
    pct_population = np.concatenate([[0], pct_population])
    pct_positives = np.concatenate([[0], pct_positives])

    ax.plot(pct_population, pct_positives, color=config.neutral_color, lw=2,
            label='Model')
    ax.plot([0, 100], [0, 100], linestyle='--', color=config.diagonal_color,
            label='Random')

    ax.set_xlabel('% of Population Selected')
    ax.set_ylabel('% of Positives Captured')
    ax.set_title('Cumulative Gains Chart')
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 105])
    ax.legend(loc='lower right')

    return (fig, ax)


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          labels: Optional[List[str]] = None,
                          normalize: bool = False,
                          config: Optional[EvalPlotConfig] = None) -> Tuple[Any, Any]:
    """
    Plot confusion matrix heatmap.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      CONFUSION MATRIX                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │                        Predicted                                         │
    │                    Rejected   Admitted                                   │
    │                  ┌───────────┬───────────┐                               │
    │   Actual  Rej.   │   850     │    50     │                               │
    │                  │   (TN)    │   (FP)    │                               │
    │                  │   ✓       │   Type I  │                               │
    │                  ├───────────┼───────────┤                               │
    │           Adm.   │    30     │   120     │                               │
    │                  │   (FN)    │   (TP)    │                               │
    │                  │  Type II  │    ✓      │                               │
    │                  └───────────┴───────────┘                               │
    │                                                                          │
    │   METRICS FROM CONFUSION MATRIX:                                         │
    │   ─────────────────────────────                                          │
    │                                                                          │
    │   Accuracy    = (TN + TP) / Total = (850 + 120) / 1050 = 0.924          │
    │   Precision   = TP / (TP + FP) = 120 / 170 = 0.706                      │
    │   Recall      = TP / (TP + FN) = 120 / 150 = 0.800                      │
    │   Specificity = TN / (TN + FP) = 850 / 900 = 0.944                      │
    │   F1 Score    = 2 * (Prec * Rec) / (Prec + Rec) = 0.750                 │
    │                                                                          │
    │   ERROR TYPES:                                                           │
    │   ─────────────                                                          │
    │   • FP (Type I):  Predicted admit but actually rejected                  │
    │                   → Wasted interview resources                           │
    │   • FN (Type II): Predicted reject but actually admitted                 │
    │                   → Missed promising candidates                          │
    │                                                                          │
    │   NORMALIZED VERSION (by row):                                           │
    │   ─────────────────────────────                                          │
    │                    Rejected   Admitted                                   │
    │                  ┌───────────┬───────────┐                               │
    │   Actual  Rej.   │   94.4%   │    5.6%   │  = 100%                       │
    │                  ├───────────┼───────────┤                               │
    │           Adm.   │   20.0%   │   80.0%   │  = 100%                       │
    │                  └───────────┴───────────┘                               │
    │                                                                          │
    │   Shows: 80% of admits correctly predicted (sensitivity = 0.80)         │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        y_true: True labels
        y_pred: Predicted labels (binary, not probabilities)
        labels: Optional class names (e.g., ["Rejected", "Admitted"])
        normalize: If True, normalize by row (shows rates instead of counts)
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Compute matrix: cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    2. If normalize: cm = cm / cm.sum(axis=1, keepdims=True)
    3. Use seaborn.heatmap() for visualization
    4. Add annotations with counts/percentages
    5. Color: TN/TP in green, FP/FN in red
    """
    config = config or EvalPlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Build 2x2 confusion matrix manually
    cm = np.zeros((2, 2), dtype=float)
    cm[0, 0] = np.sum((y_true == 0) & (y_pred == 0))  # TN
    cm[0, 1] = np.sum((y_true == 0) & (y_pred == 1))  # FP
    cm[1, 0] = np.sum((y_true == 1) & (y_pred == 0))  # FN
    cm[1, 1] = np.sum((y_true == 1) & (y_pred == 1))  # TP

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid div by zero
        cm = cm / row_sums

    # Plot as heatmap using imshow
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)

    # Labels
    if labels is None:
        labels = ['0', '1']

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    # Annotate cells
    fmt = '.2f' if normalize else '.0f'
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=14)

    return (fig, ax)


# =============================================================================
#                      THRESHOLD ANALYSIS
# =============================================================================

def plot_threshold_metrics(y_true: np.ndarray,
                           y_prob: np.ndarray,
                           metrics: Optional[List[str]] = None,
                           config: Optional[EvalPlotConfig] = None) -> Tuple[Any, Any]:
    """
    Plot metrics vs classification threshold.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   METRICS VS THRESHOLD                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Shows how different metrics change as we vary the decision threshold.  │
    │                                                                          │
    │   Metric Value                                                           │
    │   1.0 ┤───── Precision                                                   │
    │       │  ╲                                                               │
    │   0.8 ┤   ╲─────                                                         │
    │       │        ╲                                                         │
    │   0.6 ┤         ╲────── F1 Score                                         │
    │       │   ────────────────────                                           │
    │   0.4 ┤              ╲─────                                              │
    │       │                    ╲                                             │
    │   0.2 ┤                     ────── Recall                                │
    │       │          │                                                       │
    │   0.0 ┼──────────┼───────────────────────────────→                       │
    │       0.0   0.2  │0.4   0.6   0.8   1.0                                  │
    │              Threshold                                                   │
    │                  │                                                       │
    │                  └── Optimal threshold = 0.35                            │
    │                      (maximizes F1)                                      │
    │                                                                          │
    │   THRESHOLD SELECTION STRATEGIES:                                        │
    │   ─────────────────────────────────                                      │
    │                                                                          │
    │   1. Maximize F1 (balanced)                                              │
    │      threshold = argmax F1                                               │
    │                                                                          │
    │   2. Fixed Precision (e.g., 0.9)                                         │
    │      threshold = min t where Precision(t) >= 0.9                         │
    │                                                                          │
    │   3. Fixed Recall (e.g., 0.95)                                           │
    │      threshold = max t where Recall(t) >= 0.95                           │
    │                                                                          │
    │   4. Cost-sensitive                                                      │
    │      threshold = argmin (cost_FP * FP + cost_FN * FN)                   │
    │                                                                          │
    │   PROJECT APPLICATION:                                                   │
    │   ─────────────────────                                                  │
    │   If missing a good applicant (FN) costs more than interviewing          │
    │   a weak one (FP), lower the threshold to increase recall.               │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        metrics: List of metrics to plot. Default: ["precision", "recall", "f1"]
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Create array of thresholds: np.linspace(0, 1, 100)
    2. For each threshold, compute y_pred = (y_prob >= threshold)
    3. Compute each metric using sklearn
    4. Plot all metrics on same axes
    5. Find and mark optimal threshold for each metric
    6. Add legend
    """
    config = config or EvalPlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    if metrics is None:
        metrics = ['precision', 'recall', 'f1']

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    thresholds = np.linspace(0.01, 0.99, 99)
    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)

    results = {m: [] for m in metrics}
    for t in thresholds:
        y_pred = (y_prob >= t).astype(float)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        metric_map = {'precision': prec, 'recall': rec, 'f1': f1}
        for m in metrics:
            results[m].append(metric_map.get(m, 0.0))

    colors_map = {'precision': config.neutral_color, 'recall': config.good_color,
                  'f1': config.bad_color}
    for m in metrics:
        color = colors_map.get(m, config.neutral_color)
        ax.plot(thresholds, results[m], lw=2, label=m.capitalize(), color=color)

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_title('Metrics vs Classification Threshold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.legend(loc='best')

    return (fig, ax)


def plot_precision_recall_tradeoff(y_true: np.ndarray,
                                   y_prob: np.ndarray,
                                   target_precision: Optional[float] = None,
                                   target_recall: Optional[float] = None,
                                   config: Optional[EvalPlotConfig] = None) -> Tuple[Any, Any]:
    """
    Plot precision vs recall with threshold annotations.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                 PRECISION-RECALL TRADEOFF                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Precision                                                              │
    │   1.0 ┤●                                                                 │
    │       │ ╲                                                                │
    │   0.9 ┤──●─── Target precision                                          │
    │       │   ╲ (0.45)                                                      │
    │   0.8 ┤    ●                                                             │
    │       │     ╲                                                            │
    │   0.7 ┤      ●                                                           │
    │       │       ╲                                                          │
    │   0.6 ┤        ●──●                                                      │
    │       │            ╲                                                     │
    │   0.5 ┤             ●──●                                                 │
    │       │                 ╲                                                │
    │   0.4 ┤                  ●────●                                          │
    │       │                                                                  │
    │   0.3 ┼────────────────────────────────────→                             │
    │       0.0   0.2   0.4   0.6   0.8   1.0                                  │
    │                        Recall                                            │
    │                        ↑                                                 │
    │                     At target precision 0.9, recall = 0.45               │
    │                                                                          │
    │   ANNOTATIONS ON CURVE:                                                  │
    │   ─────────────────────                                                  │
    │   Each point labeled with threshold value.                               │
    │   This helps select threshold for desired operating point.               │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        target_precision: If provided, mark point achieving this precision
        target_recall: If provided, mark point achieving this recall
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)
    """
    config = config or EvalPlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    thresholds = np.linspace(0.01, 0.99, 99)

    precision_list = []
    recall_list = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(float)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision_list.append(prec)
        recall_list.append(rec)

    precision_arr = np.array(precision_list)
    recall_arr = np.array(recall_list)

    ax.plot(recall_arr, precision_arr, color=config.neutral_color, lw=2,
            label='Precision-Recall')

    # Mark target precision
    if target_precision is not None:
        # Find threshold where precision >= target_precision
        valid = precision_arr >= target_precision
        if np.any(valid):
            # Among valid, pick the one with highest recall
            idx = np.where(valid)[0]
            best = idx[np.argmax(recall_arr[idx])]
            ax.axhline(y=target_precision, linestyle=':', color=config.bad_color, alpha=0.7)
            ax.plot(recall_arr[best], precision_arr[best], 'ro', markersize=10,
                    label=f'Target Prec={target_precision:.2f} (Rec={recall_arr[best]:.2f})')

    # Mark target recall
    if target_recall is not None:
        valid = recall_arr >= target_recall
        if np.any(valid):
            idx = np.where(valid)[0]
            best = idx[np.argmax(precision_arr[idx])]
            ax.axvline(x=target_recall, linestyle=':', color=config.good_color, alpha=0.7)
            ax.plot(recall_arr[best], precision_arr[best], 'g^', markersize=10,
                    label=f'Target Rec={target_recall:.2f} (Prec={precision_arr[best]:.2f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Tradeoff')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.legend(loc='upper right')

    return (fig, ax)


# =============================================================================
#                      SUBGROUP ANALYSIS
# =============================================================================

def plot_subgroup_performance(metrics: Dict[str, Dict[str, float]],
                              metric_name: str = 'auc',
                              show_sample_size: bool = True,
                              config: Optional[EvalPlotConfig] = None) -> Tuple[Any, Any]:
    """
    Compare performance across subgroups.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   SUBGROUP PERFORMANCE                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Essential for detecting BIAS in the model.                             │
    │                                                                          │
    │   AUC                                                                    │
    │   1.0 ┤ ████                                                             │
    │       │ ████  ████                                                       │
    │   0.8 ┤ ████  ████  ████  ████                                           │
    │       │ ████  ████  ████  ████  ████                                     │
    │   0.6 ┤ ████  ████  ████  ████  ████                                     │
    │       │ ████  ████  ████  ████  ████                                     │
    │   0.4 ┤ ████  ████  ████  ████  ████                                     │
    │       │ ████  ████  ████  ████  ████                                     │
    │   0.2 ┤ ████  ████  ████  ████  ████                                     │
    │       │ ████  ████  ████  ████  ████                                     │
    │   0.0 ┼─────────────────────────────────────→                            │
    │        Ontario Quebec   BC   Alberta Other                               │
    │        (1000)  (500)  (300)  (200)  (100)  ← sample sizes                │
    │                                                                          │
    │   ANALYSIS:                                                              │
    │   ─────────                                                              │
    │   • Model performs best in Ontario (largest sample)                      │
    │   • BC has lower AUC (0.78) → investigate why                           │
    │   • Other provinces have small samples → wider confidence intervals      │
    │                                                                          │
    │   FAIRNESS CONCERNS:                                                     │
    │   ──────────────────                                                     │
    │   Large performance gaps between groups may indicate:                    │
    │   • Different feature distributions                                      │
    │   • Label noise in some groups                                          │
    │   • Model bias that needs correction                                    │
    │                                                                          │
    │   WITH ERROR BARS:                                                       │
    │   ─────────────────                                                      │
    │   Add confidence intervals using bootstrap or analytical formula.        │
    │   Helps distinguish real differences from sampling variation.            │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        metrics: Dict mapping group name to dict of metrics
                 e.g., {"Ontario": {"auc": 0.85, "n": 1000}}
        metric_name: Which metric to plot (e.g., 'auc', 'brier', 'ece')
        show_sample_size: If True, show sample sizes below bars
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Extract metric values for each group
    2. Create bar chart with group names on x-axis
    3. Add error bars if 'ci' or 'std' in metrics
    4. Annotate with sample sizes
    5. Add horizontal line at overall metric value
    6. Color bars by performance relative to overall
    """
    config = config or EvalPlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    groups = list(metrics.keys())
    values = [metrics[g].get(metric_name, 0.0) for g in groups]
    colors = [config.model_colors[i % len(config.model_colors)] for i in range(len(groups))]

    bars = ax.bar(groups, values, color=colors, edgecolor='black', width=0.6)

    # Annotate values
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Show sample sizes below x-axis labels
    if show_sample_size:
        sample_labels = []
        for g in groups:
            n = metrics[g].get('n', None)
            if n is not None:
                sample_labels.append(f'{g}\n(n={int(n)})')
            else:
                sample_labels.append(g)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(sample_labels)

    ax.set_ylabel(metric_name.upper())
    ax.set_title(f'{metric_name.upper()} by Subgroup')

    # Overall average line
    overall = np.mean(values) if len(values) > 0 else 0.0
    ax.axhline(y=overall, linestyle='--', color=config.diagonal_color,
               label=f'Overall mean ({overall:.3f})')
    ax.legend()

    return (fig, ax)


def plot_fairness_comparison(group_metrics: Dict[str, Dict[str, float]],
                             fairness_criteria: Optional[List[str]] = None,
                             config: Optional[EvalPlotConfig] = None) -> Tuple[Any, Any]:
    """
    Compare fairness metrics across groups.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    FAIRNESS COMPARISON                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Shows multiple fairness criteria across demographic groups.            │
    │                                                                          │
    │   FAIRNESS CRITERIA:                                                     │
    │   ──────────────────                                                     │
    │                                                                          │
    │   1. Demographic Parity                                                  │
    │      P(Ŷ=1 | Group=A) = P(Ŷ=1 | Group=B)                               │
    │      "Equal positive prediction rates across groups"                     │
    │                                                                          │
    │   2. Equal Opportunity                                                   │
    │      P(Ŷ=1 | Y=1, Group=A) = P(Ŷ=1 | Y=1, Group=B)                     │
    │      "Equal true positive rates (recall) across groups"                  │
    │                                                                          │
    │   3. Equalized Odds                                                      │
    │      TPR equal AND FPR equal across groups                              │
    │      "Equal TPR and FPR across groups"                                   │
    │                                                                          │
    │   4. Calibration by Group                                                │
    │      P(Y=1 | Ŷ=p, Group=A) = P(Y=1 | Ŷ=p, Group=B) = p                 │
    │      "Probabilities mean the same thing for all groups"                  │
    │                                                                          │
    │   RADAR/SPIDER CHART:                                                    │
    │   ────────────────────                                                   │
    │                                                                          │
    │              Demographic Parity                                          │
    │                    ╱╲                                                    │
    │                   ╱  ╲                                                   │
    │          Equal   ╱    ╲  Calibration                                    │
    │         Opportunity     │                                                │
    │                 ╲    ╱                                                   │
    │                  ╲  ╱                                                    │
    │                   ╲╱                                                     │
    │              Equalized Odds                                              │
    │                                                                          │
    │   Each line = one group. Closer to center = worse on that criterion.     │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        group_metrics: Dict mapping group to dict of fairness metrics
        fairness_criteria: List of criteria to include
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)
    """
    config = config or EvalPlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    groups = list(group_metrics.keys())

    # Determine fairness criteria from data if not provided
    if fairness_criteria is None:
        all_criteria = set()
        for g in groups:
            all_criteria.update(group_metrics[g].keys())
        fairness_criteria = sorted(all_criteria)

    n_groups = len(groups)
    n_criteria = len(fairness_criteria)
    x = np.arange(n_criteria)
    width = 0.8 / n_groups if n_groups > 0 else 0.4

    for i, group in enumerate(groups):
        vals = [group_metrics[group].get(c, 0.0) for c in fairness_criteria]
        color = config.model_colors[i % len(config.model_colors)]
        offset = (i - n_groups / 2.0 + 0.5) * width
        ax.bar(x + offset, vals, width, label=group, color=color, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in fairness_criteria],
                       rotation=30, ha='right')
    ax.set_ylabel('Value')
    ax.set_title('Fairness Comparison Across Groups')
    ax.legend()

    fig.tight_layout()
    return (fig, ax)


# =============================================================================
#                      TEMPORAL ANALYSIS
# =============================================================================

def plot_performance_over_time(metrics_by_period: Dict[str, List[float]],
                               period_labels: List[str],
                               config: Optional[EvalPlotConfig] = None) -> Tuple[Any, Any]:
    """
    Plot model performance trends over time.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   PERFORMANCE OVER TIME                                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Monitors for CONCEPT DRIFT - when model performance degrades.          │
    │                                                                          │
    │   Metric                                                                 │
    │   1.0 ┤                                                                  │
    │       │                                                                  │
    │   0.9 ┤───── AUC                                                         │
    │       │ ●──●──●──●──●──●                                                 │
    │   0.8 ┤                 ╲──●                                             │
    │       │                     ╲──●──●  ← performance dropping!             │
    │   0.7 ┤                                                                  │
    │       │ ●──●──●──●──●──●──●──●──●                                        │
    │   0.6 ┤───── Precision                                                   │
    │       │                                                                  │
    │   0.5 ┼──────────────────────────────────→                               │
    │        Q1   Q2   Q3   Q4   Q1   Q2   Q3   Q4                             │
    │       2023                 2024                                          │
    │                                                                          │
    │   DRIFT DETECTION:                                                       │
    │   ─────────────────                                                      │
    │   • Sudden drop: Data quality issue or distribution shift               │
    │   • Gradual decline: Concept drift, model becoming stale                │
    │   • Seasonal patterns: Expected, model for it                           │
    │                                                                          │
    │   ACTION TRIGGERS:                                                       │
    │   ─────────────────                                                      │
    │   • AUC drops > 0.05: Investigate immediately                           │
    │   • AUC drops > 0.10: Consider retraining                               │
    │   • Sustained decline over 3+ periods: Retrain with recent data         │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        metrics_by_period: Dict mapping metric name to list of values per period
        period_labels: Labels for each time period
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Plot each metric as a line over time
    2. Add confidence bands if available
    3. Mark significant changes with annotations
    4. Add horizontal lines for baseline/target
    5. Consider secondary y-axis for sample size
    """
    config = config or EvalPlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    markers = ['o', 's', '^', 'D', 'v', 'P']
    for i, (metric_name, values) in enumerate(metrics_by_period.items()):
        color = config.model_colors[i % len(config.model_colors)]
        marker = markers[i % len(markers)]
        ax.plot(period_labels[:len(values)], values, marker=marker, linestyle='-',
                color=color, lw=2, label=metric_name)

    ax.set_xlabel('Period')
    ax.set_ylabel('Metric Value')
    ax.set_title('Performance Over Time')
    ax.legend(loc='best')

    return (fig, ax)


def plot_prediction_distribution(y_prob: np.ndarray,
                                 by_outcome: Optional[np.ndarray] = None,
                                 config: Optional[EvalPlotConfig] = None) -> Tuple[Any, Any]:
    """
    Histogram of prediction probabilities, optionally split by outcome.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   PREDICTION DISTRIBUTION                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Shows how confident the model is in its predictions.                   │
    │                                                                          │
    │   SPLIT BY OUTCOME:                                                      │
    │   ──────────────────                                                     │
    │                                                                          │
    │   Count                                                                  │
    │       │    ▄▄                            ▄▄                              │
    │       │   ████                          ████                             │
    │       │   ████                          ████                             │
    │       │  ██████  Rejected              ██████  Admitted                  │
    │       │  ██████  (actual=0)            ██████  (actual=1)                │
    │       │ ████████                      ████████                           │
    │       │ ████████                      ████████                           │
    │       └────────────────────────────────────────→                         │
    │         0.0       0.5                  1.0                               │
    │               Predicted Probability                                      │
    │                                                                          │
    │   INTERPRETATION:                                                        │
    │   ────────────────                                                       │
    │                                                                          │
    │   ✓ GOOD: Distributions well-separated                                  │
    │     (rejected clustered low, admitted clustered high)                    │
    │                                                                          │
    │   ✗ BAD: Overlapping distributions                                      │
    │     (model can't distinguish between groups)                             │
    │                                                                          │
    │   ⚠ OVERCONFIDENT: All predictions near 0 or 1                          │
    │     (may need calibration)                                               │
    │                                                                          │
    │   ⚠ UNDERCONFIDENT: All predictions near 0.5                            │
    │     (model is uncertain, may need more features)                         │
    │                                                                          │
    │   QUANTILES:                                                             │
    │   ──────────                                                             │
    │   Show 10th, 50th, 90th percentile of predictions.                       │
    │   Helps understand prediction concentration.                             │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        y_prob: Predicted probabilities
        by_outcome: Optional true labels to split distribution
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. If by_outcome is None: single histogram
    2. If by_outcome provided: overlapping histograms with alpha
    3. Use plt.hist() with alpha=0.5 for overlap
    4. Add vertical lines at quantiles
    5. Add legend if split by outcome
    6. Consider density=True for normalized comparison
    """
    config = config or EvalPlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    y_prob = np.asarray(y_prob, dtype=float)

    if by_outcome is not None:
        by_outcome = np.asarray(by_outcome, dtype=float)
        mask_neg = by_outcome == 0
        mask_pos = by_outcome == 1

        ax.hist(y_prob[mask_neg], bins=config.n_bins, range=(0, 1), alpha=0.5,
                color=config.bad_color, edgecolor='black', label='Negative (0)',
                density=True)
        ax.hist(y_prob[mask_pos], bins=config.n_bins, range=(0, 1), alpha=0.5,
                color=config.good_color, edgecolor='black', label='Positive (1)',
                density=True)
        ax.legend()
    else:
        ax.hist(y_prob, bins=config.n_bins, range=(0, 1), alpha=0.7,
                color=config.neutral_color, edgecolor='black', density=True)

    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Distribution')
    ax.set_xlim([0, 1])

    return (fig, ax)


# =============================================================================
#                      UTILITY FUNCTIONS
# =============================================================================

def compute_calibration_curve(y_true: np.ndarray,
                              y_prob: np.ndarray,
                              n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve data.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins

    Returns:
        Tuple of (mean_predicted, actual_fraction, bin_counts)

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Create bin edges: np.linspace(0, 1, n_bins + 1)
    2. Assign each prediction to a bin: np.digitize(y_prob, bins)
    3. For each bin:
       - mean_predicted = mean of predictions in bin
       - actual_fraction = mean of true labels in bin
       - count = number of samples in bin
    4. Return arrays for non-empty bins only
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize returns 1..n_bins+1; clip to [1, n_bins]
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])  # 0..n_bins-1

    mean_predicted = []
    actual_fraction = []
    bin_counts = []

    for b in range(n_bins):
        mask = bin_indices == b
        count = int(np.sum(mask))
        if count > 0:
            mean_predicted.append(float(np.mean(y_prob[mask])))
            actual_fraction.append(float(np.mean(y_true[mask])))
            bin_counts.append(count)

    return (np.array(mean_predicted),
            np.array(actual_fraction),
            np.array(bin_counts))


def compute_brier_decomposition(y_true: np.ndarray,
                                y_prob: np.ndarray,
                                n_bins: int = 10) -> Dict[str, float]:
    """
    Compute Brier score decomposition.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins

    Returns:
        Dict with keys: 'brier', 'uncertainty', 'resolution', 'reliability'

    MATHEMATICAL FORMULAS:
    ──────────────────────
    Let:
    - ȳ = overall mean of y_true (base rate)
    - n_k = samples in bin k
    - ō_k = actual positive rate in bin k
    - f̄_k = mean predicted probability in bin k
    - N = total samples

    Uncertainty = ȳ(1 - ȳ)
    Resolution = (1/N) Σ_k n_k (ō_k - ȳ)²
    Reliability = (1/N) Σ_k n_k (f̄_k - ō_k)²
    Brier = Uncertainty - Resolution + Reliability
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    N = len(y_true)

    y_bar = np.mean(y_true)
    uncertainty = y_bar * (1.0 - y_bar)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    resolution = 0.0
    reliability = 0.0

    for b in range(n_bins):
        mask = bin_indices == b
        n_k = int(np.sum(mask))
        if n_k > 0:
            o_k = np.mean(y_true[mask])
            f_k = np.mean(y_prob[mask])
            resolution += n_k * (o_k - y_bar) ** 2
            reliability += n_k * (f_k - o_k) ** 2

    resolution /= N
    reliability /= N
    brier = uncertainty - resolution + reliability

    return {
        'brier': brier,
        'uncertainty': uncertainty,
        'resolution': resolution,
        'reliability': reliability,
    }


# =============================================================================
#                              TODO LIST
# =============================================================================
"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TODO LIST                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  HIGH PRIORITY (Core metrics visualization):                                 │
│  ──────────────────────────────────────────                                 │
│  [ ] plot_reliability_diagram() - Critical for calibration assessment       │
│      - [ ] Bin predictions and compute actual rates                         │
│      - [ ] Add confidence intervals                                         │
│      - [ ] Add histogram of predictions below                               │
│      - [ ] Add diagonal reference line                                      │
│                                                                              │
│  [ ] plot_roc_curve() - Standard discrimination metric                      │
│      - [ ] Compute and plot ROC curve                                        │
│      - [ ] Show AUC in legend                                                │
│      - [ ] Mark optimal threshold (Youden's J)                               │
│      - [ ] Add random baseline diagonal                                      │
│                                                                              │
│  [ ] plot_confusion_matrix() - Essential for threshold analysis             │
│      - [ ] Heatmap with counts/percentages                                   │
│      - [ ] Row normalization option                                          │
│      - [ ] Metric annotations                                                │
│                                                                              │
│  MEDIUM PRIORITY (Advanced analysis):                                        │
│  ─────────────────────────────────────                                       │
│  [ ] plot_brier_decomposition() - Understand Brier score components        │
│      - [ ] Stacked bar visualization                                         │
│      - [ ] Component annotations                                             │
│      - [ ] Verify Brier = UNC - RES + REL                                   │
│                                                                              │
│  [ ] plot_threshold_metrics() - Threshold selection                         │
│      - [ ] Multi-line plot for different metrics                            │
│      - [ ] Mark optimal thresholds                                          │
│      - [ ] Support cost-sensitive threshold selection                       │
│                                                                              │
│  [ ] plot_pr_curve() - Important for imbalanced data                        │
│      - [ ] Compute and plot PR curve                                         │
│      - [ ] Show Average Precision                                            │
│      - [ ] Add baseline (positive rate)                                      │
│                                                                              │
│  [ ] plot_subgroup_performance() - Fairness analysis                        │
│      - [ ] Bar chart by group                                                │
│      - [ ] Error bars for uncertainty                                        │
│      - [ ] Sample size annotations                                           │
│                                                                              │
│  LOWER PRIORITY (Advanced features):                                         │
│  ────────────────────────────────────                                        │
│  [ ] plot_lift_chart() - Business value visualization                       │
│  [ ] plot_cumulative_gains() - Marketing-style analysis                     │
│  [ ] plot_calibration_comparison() - Multi-model comparison                 │
│  [ ] plot_roc_comparison() - Multi-model ROC overlay                        │
│  [ ] plot_fairness_comparison() - Multi-criteria fairness                   │
│  [ ] plot_performance_over_time() - Drift monitoring                        │
│  [ ] plot_prediction_distribution() - Confidence analysis                   │
│                                                                              │
│  UTILITY FUNCTIONS:                                                          │
│  ──────────────────                                                          │
│  [ ] compute_calibration_curve() - Helper for reliability diagram           │
│  [ ] compute_brier_decomposition() - Helper for Brier visualization         │
│                                                                              │
│  TESTING:                                                                    │
│  ────────                                                                    │
│  [ ] Create synthetic well-calibrated data for testing                      │
│  [ ] Create synthetic overconfident/underconfident data                     │
│  [ ] Compare outputs with sklearn.calibration                               │
│  [ ] Test edge cases (all 0s, all 1s, single class)                        │
│                                                                              │
│  DOCUMENTATION:                                                              │
│  ───────────────                                                             │
│  [ ] Add example notebook with real data                                    │
│  [ ] Screenshot gallery comparing good vs bad calibration                   │
│  [ ] Link to STA257 notes for probability concepts                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    print("Evaluation Visualizations Module")
    print("=" * 50)
    print("This module provides visualizations for:")
    print("  - Calibration assessment (reliability diagrams)")
    print("  - Discrimination metrics (ROC, PR curves)")
    print("  - Fairness analysis (subgroup performance)")
    print("  - Threshold selection (metrics vs threshold)")
    print()
    print("Import and use individual functions, e.g.:")
    print("  from src.visualization.eval_viz import plot_reliability_diagram")
    print()
    print("See docstrings for detailed usage instructions.")
