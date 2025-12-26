"""
Calibration Metrics for Grade Prediction System.

==============================================================================
SYSTEM ARCHITECTURE - WHERE THIS MODULE FITS
==============================================================================

This module evaluates CALIBRATION - how well predicted probabilities match
actual outcome frequencies. A well-calibrated model has predictions that can
be interpreted as true probabilities.

┌─────────────────────────────────────────────────────────────────────────────┐
│                        EVALUATION PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         MODEL PREDICTIONS                           │   │
│   │                                                                     │   │
│   │    Model.predict_proba(X) → [0.72, 0.35, 0.88, 0.51, ...]          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│         ┌──────────────────────────┼──────────────────────────┐            │
│         │                          │                          │            │
│         ▼                          ▼                          ▼            │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │   CALIBRATION    │    │  DISCRIMINATION  │    │   VALIDATION     │      │
│  │   (This Module)  │    │ (discrimination) │    │  (validation.py) │      │
│  │                  │    │                  │    │                  │      │
│  │ "Are probabilities│   │ "Can model rank  │    │ "How to split    │      │
│  │  accurate?"       │   │  instances?"     │    │  data properly?" │      │
│  │                  │    │                  │    │                  │      │
│  │ • Brier Score    │    │ • ROC-AUC        │    │ • Temporal split │      │
│  │ • ECE            │    │ • PR-AUC         │    │ • K-fold CV      │      │
│  │ • Reliability    │    │ • Lift curves    │    │ • Stratification │      │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
WHAT IS CALIBRATION?
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  CALIBRATION = PROBABILITY ACCURACY                                        │
│                                                                             │
│  A model is well-calibrated if:                                             │
│  When it predicts P(admit) = 0.7 for many students,                         │
│  approximately 70% of those students are actually admitted.                 │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PERFECTLY CALIBRATED MODEL                                        │    │
│  │                                                                     │    │
│  │  Predicted Prob │ Actual Admit Rate │ Match?                       │    │
│  │  ───────────────┼───────────────────┼─────────                      │    │
│  │      0.20       │       0.20        │   ✓                          │    │
│  │      0.40       │       0.40        │   ✓                          │    │
│  │      0.60       │       0.60        │   ✓                          │    │
│  │      0.80       │       0.80        │   ✓                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  OVERCONFIDENT MODEL (Poor Calibration)                            │    │
│  │                                                                     │    │
│  │  Predicted Prob │ Actual Admit Rate │ Issue                        │    │
│  │  ───────────────┼───────────────────┼─────────                      │    │
│  │      0.20       │       0.35        │  Underestimates              │    │
│  │      0.80       │       0.65        │  Overestimates               │    │
│  │                                                                     │    │
│  │  Predictions are too extreme compared to reality!                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
RELIABILITY DIAGRAM (CALIBRATION CURVE)
==============================================================================

The reliability diagram is the primary calibration visualization:

┌─────────────────────────────────────────────────────────────────────────────┐
│  RELIABILITY DIAGRAM                                                        │
│                                                                             │
│  Actual                                                                     │
│  Outcome                                                                    │
│  Rate                                                                       │
│    │                                                                        │
│  1.0┤                                               ╱ Perfect              │
│    │                                             ╱   calibration           │
│  0.8┤                                    ○     ╱                           │
│    │                              ○         ╱                              │
│  0.6┤                       ○            ╱                                 │
│    │                  ○              ╱ ← Model curve                       │
│  0.4┤            ○               ╱                                         │
│    │       ○                 ╱                                             │
│  0.2┤  ○                  ╱                                                │
│    │                   ╱                                                   │
│  0.0┼────────────────────────────────────────────────→                     │
│    0.0    0.2    0.4    0.6    0.8    1.0                                  │
│                  Predicted Probability                                      │
│                                                                             │
│  How to read:                                                               │
│  1. Group predictions into bins (e.g., 0-0.1, 0.1-0.2, ...)                │
│  2. For each bin, compute actual positive rate                             │
│  3. Plot (mean predicted, actual rate) for each bin                        │
│  4. Perfect calibration = diagonal line                                    │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
BRIER SCORE - THE MSE FOR PROBABILITIES
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  BRIER SCORE                                                                │
│                                                                             │
│                    1   n                                                    │
│  Brier Score = ─────  Σ  (pᵢ - yᵢ)²                                        │
│                  n   i=1                                                    │
│                                                                             │
│  Where:                                                                     │
│    pᵢ = predicted probability for instance i                               │
│    yᵢ = actual outcome (0 or 1) for instance i                             │
│    n  = number of instances                                                 │
│                                                                             │
│  Range: [0, 1]                                                              │
│  Lower is better (0 = perfect predictions)                                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  EXAMPLE                                                           │    │
│  │                                                                     │    │
│  │  Instance │ Predicted │ Actual │ Squared Error                     │    │
│  │  ─────────┼───────────┼────────┼──────────────                      │    │
│  │     1     │    0.9    │   1    │  (0.9-1)² = 0.01                  │    │
│  │     2     │    0.8    │   1    │  (0.8-1)² = 0.04                  │    │
│  │     3     │    0.3    │   0    │  (0.3-0)² = 0.09                  │    │
│  │     4     │    0.2    │   0    │  (0.2-0)² = 0.04                  │    │
│  │  ─────────┴───────────┴────────┴──────────────                      │    │
│  │  Brier Score = (0.01+0.04+0.09+0.04)/4 = 0.045                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Interpretation:                                                            │
│  - 0.00 - 0.10: Excellent calibration                                       │
│  - 0.10 - 0.20: Good calibration                                            │
│  - 0.20 - 0.25: Moderate (random baseline for 50/50)                        │
│  - > 0.25: Poor calibration                                                 │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
BRIER SCORE DECOMPOSITION
==============================================================================

The Brier score can be decomposed into three components:

┌─────────────────────────────────────────────────────────────────────────────┐
│  BRIER SCORE = UNCERTAINTY - RESOLUTION + RELIABILITY                      │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  UNCERTAINTY (UNC)                                                    │  │
│  │  ─────────────────                                                    │  │
│  │  Inherent unpredictability of the outcomes                            │  │
│  │                                                                       │  │
│  │  UNC = ȳ(1 - ȳ)                                                       │  │
│  │                                                                       │  │
│  │  Where ȳ = overall positive rate                                      │  │
│  │  Maximum when ȳ = 0.5 (most uncertain)                                │  │
│  │  This is NOT controlled by the model                                  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  RESOLUTION (RES) - Higher is better!                                 │  │
│  │  ────────────────                                                     │  │
│  │  How well predictions separate positive/negative cases                │  │
│  │                                                                       │  │
│  │        1   K                                                          │  │
│  │  RES = ─── Σ  nₖ(ȳₖ - ȳ)²                                             │  │
│  │        n  k=1                                                         │  │
│  │                                                                       │  │
│  │  Where ȳₖ = actual positive rate in bin k                             │  │
│  │  High resolution = predictions vary with actual outcomes              │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  RELIABILITY (REL) - Lower is better!                                 │  │
│  │  ─────────────────                                                    │  │
│  │  How well predicted probabilities match actual rates                  │  │
│  │                                                                       │  │
│  │        1   K                                                          │  │
│  │  REL = ─── Σ  nₖ(p̄ₖ - ȳₖ)²                                            │  │
│  │        n  k=1                                                         │  │
│  │                                                                       │  │
│  │  Where p̄ₖ = mean predicted probability in bin k                       │  │
│  │  Low reliability = well-calibrated predictions                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Key insight: We want HIGH resolution and LOW reliability                   │
│               (Good discrimination AND good calibration)                    │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
EXPECTED CALIBRATION ERROR (ECE)
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  EXPECTED CALIBRATION ERROR (ECE)                                           │
│                                                                             │
│  Measures average calibration error across probability bins:                │
│                                                                             │
│        M   nₘ                                                               │
│  ECE = Σ  ──── |acc(Bₘ) - conf(Bₘ)|                                        │
│       m=1  n                                                                │
│                                                                             │
│  Where:                                                                     │
│    M = number of bins                                                       │
│    Bₘ = set of samples in bin m                                             │
│    nₘ = number of samples in bin m                                          │
│    n = total samples                                                        │
│    acc(Bₘ) = accuracy in bin m (actual positive rate)                       │
│    conf(Bₘ) = confidence in bin m (mean predicted prob)                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ECE COMPUTATION EXAMPLE                                           │    │
│  │                                                                     │    │
│  │  Bin    │ n_bin │ Mean Pred │ Actual Rate │ |Diff| │ Weighted     │    │
│  │  ───────┼───────┼───────────┼─────────────┼────────┼──────────     │    │
│  │  0.0-0.2│  100  │   0.12    │    0.08     │  0.04  │ 100×0.04     │    │
│  │  0.2-0.4│  150  │   0.31    │    0.35     │  0.04  │ 150×0.04     │    │
│  │  0.4-0.6│  200  │   0.48    │    0.52     │  0.04  │ 200×0.04     │    │
│  │  0.6-0.8│  180  │   0.72    │    0.68     │  0.04  │ 180×0.04     │    │
│  │  0.8-1.0│  120  │   0.88    │    0.92     │  0.04  │ 120×0.04     │    │
│  │  ───────┴───────┴───────────┴─────────────┴────────┴──────────     │    │
│  │  ECE = (100+150+200+180+120)×0.04 / 750 = 0.04                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Interpretation:                                                            │
│  - ECE < 0.01: Excellent                                                    │
│  - ECE < 0.05: Good                                                         │
│  - ECE < 0.10: Acceptable                                                   │
│  - ECE > 0.10: Poor (consider calibration post-processing)                  │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
MAXIMUM CALIBRATION ERROR (MCE)
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  MAXIMUM CALIBRATION ERROR (MCE)                                            │
│                                                                             │
│  Worst-case calibration error across all bins:                              │
│                                                                             │
│  MCE = max |acc(Bₘ) - conf(Bₘ)|                                            │
│         m                                                                   │
│                                                                             │
│  Why MCE matters:                                                           │
│  - ECE can hide large errors in low-population bins                         │
│  - MCE captures the worst miscalibration                                    │
│  - Important for high-stakes decisions                                      │
│                                                                             │
│  Example:                                                                   │
│  - Bin 0.9-1.0: Only 5 samples, but 100% admitted when model says 95%      │
│  - ECE contribution: small (low weight)                                     │
│  - MCE: This bin has |1.0 - 0.95| = 0.05 error                             │
│                                                                             │
│  For admission prediction, MCE matters because:                             │
│  - Students making decisions need reliable probabilities                    │
│  - A 90% prediction should really mean ~90% chance                          │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
CALIBRATION FOR MULTI-CLASS (Future Extension)
==============================================================================

For multi-class problems (e.g., Admit/Waitlist/Reject):

┌─────────────────────────────────────────────────────────────────────────────┐
│  MULTI-CLASS ECE                                                            │
│                                                                             │
│  For each class c:                                                          │
│    ECE_c = Standard ECE treating class c as positive                        │
│                                                                             │
│  Overall:                                                                   │
│    ECE_overall = (1/C) × Σ ECE_c                                            │
│                                                                             │
│  Or use confidence-based binning:                                           │
│    Bin by max(p₁, p₂, ..., p_C) regardless of which class                   │
└─────────────────────────────────────────────────────────────────────────────┘


Author: Grade Prediction Team
Course Context: STA257 (Probability), CSC148 (OOP)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class CalibrationConfig:
    """
    Configuration for calibration metric computation.

    ┌─────────────────────────────────────────────────────────────┐
    │  CALIBRATION CONFIGURATION                                  │
    │                                                             │
    │  n_bins: int (default 10)                                   │
    │    Number of bins for reliability diagram                   │
    │    More bins = finer granularity but noisier estimates      │
    │                                                             │
    │  strategy: str ('uniform' or 'quantile')                    │
    │    'uniform': Equal-width bins [0-0.1, 0.1-0.2, ...]        │
    │    'quantile': Equal-count bins (adaptive widths)           │
    │                                                             │
    │  min_samples_per_bin: int (default 10)                      │
    │    Minimum samples for reliable bin statistics              │
    │    Bins with fewer samples excluded from MCE                │
    └─────────────────────────────────────────────────────────────┘
    """
    n_bins: int = 10
    strategy: str = 'uniform'  # 'uniform' or 'quantile'
    min_samples_per_bin: int = 10


@dataclass
class CalibrationResult:
    """
    Results from calibration analysis.

    ┌─────────────────────────────────────────────────────────────┐
    │  CALIBRATION RESULTS                                        │
    │                                                             │
    │  Scalar Metrics:                                            │
    │  - brier_score: Overall Brier score                         │
    │  - ece: Expected Calibration Error                          │
    │  - mce: Maximum Calibration Error                           │
    │                                                             │
    │  Brier Decomposition:                                       │
    │  - uncertainty: Data uncertainty component                  │
    │  - resolution: Model's discrimination ability               │
    │  - reliability: Calibration quality (lower is better)       │
    │                                                             │
    │  Bin-level Data (for reliability diagram):                  │
    │  - bin_edges: Boundaries of probability bins                │
    │  - bin_counts: Number of samples per bin                    │
    │  - bin_accuracies: Actual positive rate per bin             │
    │  - bin_confidences: Mean predicted probability per bin      │
    └─────────────────────────────────────────────────────────────┘
    """
    brier_score: float
    ece: float
    mce: float
    uncertainty: float
    resolution: float
    reliability: float
    bin_edges: np.ndarray
    bin_counts: np.ndarray
    bin_accuracies: np.ndarray
    bin_confidences: np.ndarray


@dataclass
class BrierDecomposition:
    """
    Components of Brier score decomposition.

    ┌─────────────────────────────────────────────────────────────┐
    │  BRIER DECOMPOSITION COMPONENTS                             │
    │                                                             │
    │  Brier = Uncertainty - Resolution + Reliability             │
    │                                                             │
    │  For a good model:                                          │
    │  - Resolution ≈ Uncertainty (captures all possible info)   │
    │  - Reliability ≈ 0 (well-calibrated)                        │
    │  - Brier ≈ 0                                                │
    │                                                             │
    │  Skill Score = (Brier_ref - Brier) / Brier_ref              │
    │  Where Brier_ref is Brier score of climatological forecast  │
    └─────────────────────────────────────────────────────────────┘
    """
    uncertainty: float
    resolution: float
    reliability: float
    brier: float


# =============================================================================
# CORE CALIBRATION FUNCTIONS
# =============================================================================

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Brier score (MSE for probability predictions).

    ┌─────────────────────────────────────────────────────────────┐
    │  BRIER SCORE FORMULA                                        │
    │                                                             │
    │            1   n                                            │
    │  Brier = ─────  Σ  (pᵢ - yᵢ)²                              │
    │            n   i=1                                          │
    │                                                             │
    │  Implementation Steps:                                      │
    │  1. Compute squared differences: (y_prob - y_true)²        │
    │  2. Take mean over all samples                             │
    │                                                             │
    │  Edge Cases:                                                │
    │  - y_prob outside [0,1]: clip to valid range               │
    │  - Empty arrays: raise ValueError                          │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels, shape (n,), values in {0, 1}
        y_prob: Predicted probabilities, shape (n,), values in [0, 1]

    Returns:
        Brier score in range [0, 1], lower is better

    Example:
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.3, 0.2])
        score = brier_score(y_true, y_prob)  # ≈ 0.045
    """
    pass


def brier_score_decomposition(y_true: np.ndarray,
                               y_prob: np.ndarray,
                               n_bins: int = 10) -> BrierDecomposition:
    """
    Decompose Brier score into uncertainty, resolution, and reliability.

    ┌─────────────────────────────────────────────────────────────┐
    │  IMPLEMENTATION STEPS                                       │
    │                                                             │
    │  1. Compute uncertainty:                                    │
    │     ȳ = mean(y_true)                                        │
    │     UNC = ȳ × (1 - ȳ)                                       │
    │                                                             │
    │  2. Bin predictions into K bins                             │
    │                                                             │
    │  3. For each bin k:                                         │
    │     - nₖ = count of samples in bin                          │
    │     - ȳₖ = mean of y_true in bin (actual rate)              │
    │     - p̄ₖ = mean of y_prob in bin (predicted prob)           │
    │                                                             │
    │  4. Compute resolution:                                     │
    │     RES = (1/n) × Σₖ nₖ × (ȳₖ - ȳ)²                        │
    │                                                             │
    │  5. Compute reliability:                                    │
    │     REL = (1/n) × Σₖ nₖ × (p̄ₖ - ȳₖ)²                        │
    │                                                             │
    │  6. Verify: Brier = UNC - RES + REL                         │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for decomposition

    Returns:
        BrierDecomposition with uncertainty, resolution, reliability
    """
    pass


def expected_calibration_error(y_true: np.ndarray,
                                y_prob: np.ndarray,
                                n_bins: int = 10,
                                strategy: str = 'uniform') -> float:
    """
    Compute Expected Calibration Error (ECE).

    ┌─────────────────────────────────────────────────────────────┐
    │  ECE COMPUTATION                                            │
    │                                                             │
    │        M   nₘ                                               │
    │  ECE = Σ  ──── |acc(Bₘ) - conf(Bₘ)|                        │
    │       m=1  n                                                │
    │                                                             │
    │  Implementation Steps:                                      │
    │  1. Create bins based on strategy:                          │
    │     - 'uniform': [0, 0.1, 0.2, ..., 1.0]                   │
    │     - 'quantile': bins with equal sample counts            │
    │                                                             │
    │  2. Assign each sample to a bin based on y_prob            │
    │                                                             │
    │  3. For each bin m:                                         │
    │     - acc(Bₘ) = mean(y_true) in bin                        │
    │     - conf(Bₘ) = mean(y_prob) in bin                       │
    │     - contribution = nₘ × |acc - conf|                      │
    │                                                             │
    │  4. ECE = sum of contributions / total samples              │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        strategy: 'uniform' or 'quantile' binning

    Returns:
        ECE value in range [0, 1], lower is better
    """
    pass


def maximum_calibration_error(y_true: np.ndarray,
                               y_prob: np.ndarray,
                               n_bins: int = 10,
                               min_samples: int = 10) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    ┌─────────────────────────────────────────────────────────────┐
    │  MCE COMPUTATION                                            │
    │                                                             │
    │  MCE = max |acc(Bₘ) - conf(Bₘ)|                            │
    │         m                                                   │
    │                                                             │
    │  Only includes bins with at least min_samples observations  │
    │  (empty or sparse bins are unreliable)                      │
    │                                                             │
    │  Implementation Steps:                                      │
    │  1. Create bins (uniform spacing)                           │
    │  2. For each bin with nₘ ≥ min_samples:                     │
    │     - Compute |acc - conf|                                  │
    │  3. Return maximum across valid bins                        │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        min_samples: Minimum samples for a bin to be considered

    Returns:
        MCE value in range [0, 1]
    """
    pass


# =============================================================================
# RELIABILITY DIAGRAM DATA
# =============================================================================

def compute_reliability_diagram(y_true: np.ndarray,
                                 y_prob: np.ndarray,
                                 n_bins: int = 10,
                                 strategy: str = 'uniform'
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute data for reliability diagram (calibration curve).

    ┌─────────────────────────────────────────────────────────────┐
    │  RELIABILITY DIAGRAM DATA                                   │
    │                                                             │
    │  Returns four arrays:                                       │
    │  1. bin_edges: Bin boundaries [0, 0.1, 0.2, ...]           │
    │  2. bin_centers: Midpoint of each bin (x-axis)             │
    │  3. bin_accuracies: Actual positive rate per bin (y-axis)  │
    │  4. bin_counts: Samples per bin (for confidence intervals) │
    │                                                             │
    │  Plotting:                                                  │
    │  - Plot bin_centers vs bin_accuracies                       │
    │  - Add diagonal line y=x for perfect calibration           │
    │  - Size points by bin_counts (more samples = more reliable) │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        strategy: 'uniform' or 'quantile'

    Returns:
        Tuple of (bin_edges, bin_centers, bin_accuracies, bin_counts)

    Implementation Steps:
        1. Create bin edges based on strategy
        2. Assign samples to bins using np.digitize
        3. For each bin:
           - Compute accuracy = mean(y_true)
           - Compute confidence = mean(y_prob)
           - Count samples
        4. Handle empty bins (return NaN)
    """
    pass


def calibration_curve(y_true: np.ndarray,
                       y_prob: np.ndarray,
                       n_bins: int = 10,
                       strategy: str = 'uniform'
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute calibration curve (fraction of positives vs mean predicted).

    Simplified version of compute_reliability_diagram returning just
    the (mean_predicted, fraction_positive) pairs.

    ┌─────────────────────────────────────────────────────────────┐
    │  OUTPUT FORMAT                                              │
    │                                                             │
    │  fraction_positives: [0.08, 0.22, 0.45, 0.68, 0.89]        │
    │  mean_predicted:     [0.10, 0.25, 0.48, 0.72, 0.90]        │
    │                                                             │
    │  For perfectly calibrated model:                            │
    │  fraction_positives ≈ mean_predicted (on the diagonal)     │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        strategy: 'uniform' or 'quantile'

    Returns:
        Tuple of (mean_predicted, fraction_positives) arrays
    """
    pass


# =============================================================================
# FULL CALIBRATION ANALYSIS
# =============================================================================

def full_calibration_analysis(y_true: np.ndarray,
                               y_prob: np.ndarray,
                               config: Optional[CalibrationConfig] = None
                               ) -> CalibrationResult:
    """
    Perform complete calibration analysis.

    ┌─────────────────────────────────────────────────────────────┐
    │  FULL CALIBRATION ANALYSIS                                  │
    │                                                             │
    │  Computes all calibration metrics in one pass:              │
    │  1. Brier score and decomposition                           │
    │  2. ECE (Expected Calibration Error)                        │
    │  3. MCE (Maximum Calibration Error)                         │
    │  4. Reliability diagram data                                │
    │                                                             │
    │  Returns CalibrationResult dataclass with all metrics       │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        config: CalibrationConfig (uses defaults if None)

    Returns:
        CalibrationResult with all computed metrics

    Implementation Steps:
        1. Validate inputs
        2. Compute Brier score
        3. Compute Brier decomposition
        4. Compute ECE
        5. Compute MCE
        6. Compute reliability diagram data
        7. Package into CalibrationResult
    """
    pass


# =============================================================================
# CALIBRATION COMPARISON
# =============================================================================

def compare_calibration(models: Dict[str, Tuple[np.ndarray, np.ndarray]],
                         config: Optional[CalibrationConfig] = None
                         ) -> Dict[str, CalibrationResult]:
    """
    Compare calibration across multiple models.

    ┌─────────────────────────────────────────────────────────────┐
    │  MODEL COMPARISON                                           │
    │                                                             │
    │  Input format:                                              │
    │  {                                                          │
    │    'Logistic': (y_true, y_prob_logistic),                  │
    │    'Hazard': (y_true, y_prob_hazard),                       │
    │    'Embedding': (y_true, y_prob_embedding)                  │
    │  }                                                          │
    │                                                             │
    │  Output:                                                    │
    │  {                                                          │
    │    'Logistic': CalibrationResult(...),                      │
    │    'Hazard': CalibrationResult(...),                        │
    │    'Embedding': CalibrationResult(...)                      │
    │  }                                                          │
    │                                                             │
    │  Useful for:                                                │
    │  - Comparing calibration across model types                 │
    │  - Selecting best-calibrated model                          │
    │  - Overlay reliability diagrams                             │
    └─────────────────────────────────────────────────────────────┘

    Args:
        models: Dict of model_name → (y_true, y_prob)
        config: CalibrationConfig for all models

    Returns:
        Dict of model_name → CalibrationResult
    """
    pass


# =============================================================================
# CONFIDENCE INTERVALS
# =============================================================================

def calibration_confidence_intervals(y_true: np.ndarray,
                                      y_prob: np.ndarray,
                                      n_bins: int = 10,
                                      confidence_level: float = 0.95,
                                      method: str = 'wilson'
                                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute confidence intervals for reliability diagram.

    ┌─────────────────────────────────────────────────────────────┐
    │  CONFIDENCE INTERVALS FOR BIN ACCURACIES                   │
    │                                                             │
    │  Each bin accuracy is estimated from nₘ Bernoulli trials    │
    │  Standard error depends on sample size:                     │
    │                                                             │
    │           ┌─────────────────────────────────────────────┐   │
    │           │        p̂(1-p̂)                               │   │
    │  SE(p̂) = │ sqrt( ─────── )                              │   │
    │           │          n                                  │   │
    │           └─────────────────────────────────────────────┘   │
    │                                                             │
    │  Wilson Score Interval (recommended):                       │
    │  Better coverage than normal approximation for small n      │
    │                                                             │
    │  Returns:                                                   │
    │  - bin_accuracies: Point estimates                          │
    │  - ci_lower: Lower confidence bound                         │
    │  - ci_upper: Upper confidence bound                         │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        confidence_level: e.g., 0.95 for 95% CI
        method: 'wilson' (recommended) or 'normal'

    Returns:
        Tuple of (bin_accuracies, ci_lower, ci_upper)
    """
    pass


def wilson_confidence_interval(successes: int,
                                trials: int,
                                confidence_level: float = 0.95
                                ) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for a proportion.

    ┌─────────────────────────────────────────────────────────────┐
    │  WILSON SCORE INTERVAL                                      │
    │                                                             │
    │  Better than normal approximation when:                     │
    │  - n is small                                               │
    │  - p is close to 0 or 1                                     │
    │                                                             │
    │  Formula:                                                   │
    │                                                             │
    │       p̂ + z²/2n ± z√(p̂(1-p̂)/n + z²/4n²)                    │
    │  CI = ────────────────────────────────────────               │
    │                    1 + z²/n                                 │
    │                                                             │
    │  Where z = z-score for confidence level                     │
    │  (z ≈ 1.96 for 95% confidence)                              │
    └─────────────────────────────────────────────────────────────┘

    Args:
        successes: Number of positive outcomes
        trials: Total number of trials
        confidence_level: Desired confidence level

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    pass


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def hosmer_lemeshow_test(y_true: np.ndarray,
                          y_prob: np.ndarray,
                          n_groups: int = 10
                          ) -> Tuple[float, float]:
    """
    Perform Hosmer-Lemeshow goodness-of-fit test.

    ┌─────────────────────────────────────────────────────────────┐
    │  HOSMER-LEMESHOW TEST                                       │
    │                                                             │
    │  Tests null hypothesis: Model is well-calibrated            │
    │                                                             │
    │  Test statistic:                                            │
    │         g   (Oₖ - Eₖ)²                                      │
    │  χ² = Σ ───────────                                         │
    │       k=1  Eₖ(1 - πₖ)                                       │
    │                                                             │
    │  Where:                                                     │
    │    Oₖ = Observed positives in group k                       │
    │    Eₖ = Expected positives = nₖ × π̄ₖ                        │
    │    πₖ = Mean predicted probability in group k               │
    │                                                             │
    │  Under H₀: χ² ~ χ²(g-2) distribution                        │
    │                                                             │
    │  Interpretation:                                            │
    │  - Large p-value (>0.05): Fail to reject, model may be OK   │
    │  - Small p-value (<0.05): Reject, evidence of miscalibration│
    │                                                             │
    │  Caution: Large samples often reject; small samples may not │
    │  detect real miscalibration. Use alongside visual checks.   │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_groups: Number of groups for test

    Returns:
        Tuple of (chi_squared_statistic, p_value)
    """
    pass


def calibration_test(y_true: np.ndarray,
                      y_prob: np.ndarray,
                      method: str = 'spiegelhalter'
                      ) -> Tuple[float, float]:
    """
    Perform statistical test for calibration.

    ┌─────────────────────────────────────────────────────────────┐
    │  SPIEGELHALTER'S Z-TEST FOR CALIBRATION                     │
    │                                                             │
    │  Tests if Brier score is significantly different from       │
    │  that of a perfectly calibrated model.                      │
    │                                                             │
    │            Σ(pᵢ - yᵢ)(1 - 2pᵢ)                              │
    │  z = ─────────────────────────────────                      │
    │       √(Σpᵢ(1-pᵢ)(1-2pᵢ)²)                                  │
    │                                                             │
    │  Under H₀ (perfect calibration): z ~ N(0,1)                 │
    │                                                             │
    │  Interpretation:                                            │
    │  - |z| < 1.96: Calibration OK at 5% level                   │
    │  - |z| > 1.96: Evidence of miscalibration                   │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        method: 'spiegelhalter' or 'hosmer_lemeshow'

    Returns:
        Tuple of (test_statistic, p_value)
    """
    pass


# =============================================================================
# SUBGROUP CALIBRATION
# =============================================================================

def subgroup_calibration(y_true: np.ndarray,
                          y_prob: np.ndarray,
                          groups: np.ndarray,
                          config: Optional[CalibrationConfig] = None
                          ) -> Dict[Any, CalibrationResult]:
    """
    Compute calibration separately for subgroups.

    ┌─────────────────────────────────────────────────────────────┐
    │  SUBGROUP CALIBRATION ANALYSIS                              │
    │                                                             │
    │  Important for fairness: Model may be well-calibrated       │
    │  overall but miscalibrated for specific subgroups.          │
    │                                                             │
    │  Example subgroups for admission:                           │
    │  - By province (Ontario vs Quebec vs BC)                    │
    │  - By program type (STEM vs Arts)                           │
    │  - By applicant demographics                                │
    │                                                             │
    │  Output:                                                    │
    │  {                                                          │
    │    'Ontario': CalibrationResult(ece=0.03, ...),             │
    │    'Quebec': CalibrationResult(ece=0.08, ...),              │
    │    'BC': CalibrationResult(ece=0.05, ...)                   │
    │  }                                                          │
    │                                                             │
    │  Flag groups with ECE > threshold for recalibration         │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        groups: Group labels for each sample
        config: CalibrationConfig

    Returns:
        Dict of group → CalibrationResult
    """
    pass


# =============================================================================
# CALIBRATION POST-PROCESSING
# =============================================================================

def platt_scaling_params(y_true: np.ndarray,
                          y_prob: np.ndarray
                          ) -> Tuple[float, float]:
    """
    Fit Platt scaling parameters for recalibration.

    ┌─────────────────────────────────────────────────────────────┐
    │  PLATT SCALING (Sigmoid Recalibration)                      │
    │                                                             │
    │  Transform: p_calibrated = σ(A × log_odds(p) + B)           │
    │                                                             │
    │  Where:                                                     │
    │    log_odds(p) = log(p / (1-p))                             │
    │    σ(x) = 1 / (1 + exp(-x))                                 │
    │                                                             │
    │  Fit A, B by minimizing cross-entropy on validation set     │
    │                                                             │
    │  Typical results:                                           │
    │  - A ≈ 1: Model log-odds are well-scaled                    │
    │  - A > 1: Model is underconfident                           │
    │  - A < 1: Model is overconfident                            │
    │  - B shifts overall probability level                       │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels (validation set)
        y_prob: Uncalibrated predicted probabilities

    Returns:
        Tuple of (A, B) parameters for Platt scaling

    Implementation Notes:
        - Use L-BFGS or Newton's method to minimize cross-entropy
        - Regularize to avoid numerical issues near p=0 or p=1
        - MUST fit on held-out validation data, not training data
    """
    pass


def apply_platt_scaling(y_prob: np.ndarray, A: float, B: float) -> np.ndarray:
    """
    Apply Platt scaling to recalibrate probabilities.

    ┌─────────────────────────────────────────────────────────────┐
    │  APPLY PLATT TRANSFORMATION                                 │
    │                                                             │
    │  p_calibrated = σ(A × log(p/(1-p)) + B)                    │
    │                                                             │
    │  Handle edge cases:                                         │
    │  - p = 0: Set to small ε before log                        │
    │  - p = 1: Set to 1 - ε before log                          │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_prob: Uncalibrated probabilities
        A: Scale parameter from platt_scaling_params
        B: Shift parameter from platt_scaling_params

    Returns:
        Calibrated probabilities
    """
    pass


def isotonic_calibration(y_true: np.ndarray,
                          y_prob: np.ndarray
                          ) -> np.ndarray:
    """
    Fit isotonic regression for calibration.

    ┌─────────────────────────────────────────────────────────────┐
    │  ISOTONIC REGRESSION CALIBRATION                            │
    │                                                             │
    │  Non-parametric approach: Learn monotonic mapping from      │
    │  predicted probabilities to calibrated probabilities.       │
    │                                                             │
    │  Constraints:                                               │
    │  - Mapping must be non-decreasing                           │
    │  - Minimizes squared error to observed outcomes             │
    │                                                             │
    │  Advantages:                                                │
    │  - More flexible than Platt scaling                         │
    │  - No distributional assumptions                            │
    │                                                             │
    │  Disadvantages:                                             │
    │  - Can overfit with small samples                           │
    │  - May not generalize to new probability values             │
    │                                                             │
    │  Returns:                                                   │
    │  - Fitted isotonic regressor (or calibrated values)         │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels (validation set)
        y_prob: Uncalibrated predicted probabilities

    Returns:
        Calibrated probabilities

    Implementation:
        Use Pool Adjacent Violators (PAV) algorithm
    """
    pass


def temperature_scaling(y_true: np.ndarray,
                         y_logits: np.ndarray
                         ) -> float:
    """
    Fit temperature scaling parameter.

    ┌─────────────────────────────────────────────────────────────┐
    │  TEMPERATURE SCALING                                        │
    │                                                             │
    │  Simple post-processing for neural network outputs:         │
    │                                                             │
    │  p_calibrated = softmax(logits / T)                         │
    │                                                             │
    │  Where T = temperature parameter                            │
    │                                                             │
    │  For binary case:                                           │
    │  p_calibrated = σ(logit / T)                               │
    │                                                             │
    │  Effect:                                                    │
    │  - T > 1: Softens probabilities (less confident)            │
    │  - T < 1: Sharpens probabilities (more confident)           │
    │  - T = 1: No change                                         │
    │                                                             │
    │  Fit T to minimize NLL on validation set                    │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels (validation set)
        y_logits: Raw logits from neural network (NOT probabilities)

    Returns:
        Optimal temperature T
    """
    pass


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_probability_inputs(y_true: np.ndarray,
                                 y_prob: np.ndarray) -> None:
    """
    Validate inputs for calibration functions.

    Checks:
        1. y_true and y_prob have same shape
        2. y_true contains only 0 and 1
        3. y_prob values are in [0, 1]
        4. Arrays are not empty
        5. No NaN values
    """
    pass


def bin_predictions(y_prob: np.ndarray,
                     n_bins: int = 10,
                     strategy: str = 'uniform'
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign predictions to bins.

    ┌─────────────────────────────────────────────────────────────┐
    │  BINNING STRATEGIES                                         │
    │                                                             │
    │  UNIFORM:                                                   │
    │  - Equal-width bins: [0, 0.1), [0.1, 0.2), ...             │
    │  - Pro: Interpretable boundaries                            │
    │  - Con: Uneven sample sizes, empty bins possible            │
    │                                                             │
    │  QUANTILE:                                                  │
    │  - Equal-count bins (adaptive boundaries)                   │
    │  - Pro: All bins have similar sample size                   │
    │  - Con: Bin boundaries harder to interpret                  │
    │                                                             │
    │  For ECE computation, quantile often gives more stable      │
    │  estimates, especially with skewed probability distributions│
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_prob: Predicted probabilities
        n_bins: Number of bins
        strategy: 'uniform' or 'quantile'

    Returns:
        Tuple of (bin_indices, bin_edges)
    """
    pass


# =============================================================================
# TODO LIST FOR IMPLEMENTATION
# =============================================================================
"""
TODO: Implementation Checklist

CORE METRICS:
□ brier_score()
  - [ ] Implement MSE formula
  - [ ] Handle edge cases (empty input, invalid probs)
  - [ ] Validate inputs

□ brier_score_decomposition()
  - [ ] Compute uncertainty component
  - [ ] Compute resolution component
  - [ ] Compute reliability component
  - [ ] Verify decomposition sums to Brier

□ expected_calibration_error()
  - [ ] Implement uniform binning
  - [ ] Implement quantile binning
  - [ ] Compute weighted absolute errors
  - [ ] Handle empty bins

□ maximum_calibration_error()
  - [ ] Compute per-bin calibration errors
  - [ ] Find maximum across bins
  - [ ] Exclude bins with insufficient samples

RELIABILITY DIAGRAM:
□ compute_reliability_diagram()
  - [ ] Create bin edges
  - [ ] Compute bin accuracies
  - [ ] Compute bin confidences
  - [ ] Return data for plotting

□ calibration_confidence_intervals()
  - [ ] Implement Wilson score interval
  - [ ] Compute CI for each bin
  - [ ] Handle small sample sizes

STATISTICAL TESTS:
□ hosmer_lemeshow_test()
  - [ ] Implement chi-squared statistic
  - [ ] Compute p-value from chi-squared distribution
  - [ ] Handle degrees of freedom correctly

□ calibration_test() (Spiegelhalter)
  - [ ] Implement z-statistic
  - [ ] Compute two-sided p-value

SUBGROUP ANALYSIS:
□ subgroup_calibration()
  - [ ] Split data by groups
  - [ ] Compute metrics per group
  - [ ] Flag poorly calibrated groups

CALIBRATION POST-PROCESSING:
□ platt_scaling_params()
  - [ ] Fit sigmoid parameters
  - [ ] Handle numerical stability
  - [ ] Use optimization (L-BFGS)

□ isotonic_calibration()
  - [ ] Implement PAV algorithm
  - [ ] Handle ties correctly
  - [ ] Return calibrated values

□ temperature_scaling()
  - [ ] Optimize temperature parameter
  - [ ] Minimize NLL

TESTING:
□ Unit tests
  - [ ] Test with perfectly calibrated predictions
  - [ ] Test with overconfident predictions
  - [ ] Test with underconfident predictions
  - [ ] Verify decomposition identity

□ Integration tests
  - [ ] Test with model outputs
  - [ ] Verify recalibration improves ECE

DOCUMENTATION:
□ Add STA257 references for probability theory
□ Add interpretation guidelines
□ Create example reliability diagrams
"""
