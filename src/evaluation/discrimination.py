"""
Discrimination Metrics for Grade Prediction System.

==============================================================================
SYSTEM ARCHITECTURE - WHERE THIS MODULE FITS
==============================================================================

This module evaluates DISCRIMINATION - how well the model can rank instances
and separate positive from negative cases. Unlike calibration, discrimination
focuses on relative ordering rather than absolute probability values.

┌─────────────────────────────────────────────────────────────────────────────┐
│                        EVALUATION FRAMEWORK                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ┌──────────────────────────────────┐                     │
│                    │     Model Predictions            │                     │
│                    │  P(admit) for each applicant     │                     │
│                    └───────────────┬──────────────────┘                     │
│                                    │                                        │
│           ┌────────────────────────┼────────────────────────┐              │
│           │                        │                        │              │
│           ▼                        ▼                        ▼              │
│   ┌───────────────┐      ┌─────────────────┐      ┌───────────────┐        │
│   │  CALIBRATION  │      │ DISCRIMINATION  │      │  FAIRNESS     │        │
│   │               │      │ (This Module)   │      │               │        │
│   │ "Is 70%       │      │                 │      │ "Is model     │        │
│   │  really 70%?" │      │ "Can model rank │      │  fair across  │        │
│   │               │      │  applicants?"   │      │  groups?"     │        │
│   │               │      │                 │      │               │        │
│   └───────────────┘      └─────────────────┘      └───────────────┘        │
│                                    │                                        │
│                    ┌───────────────┼───────────────┐                        │
│                    │               │               │                        │
│                    ▼               ▼               ▼                        │
│             ┌───────────┐  ┌───────────┐  ┌─────────────┐                  │
│             │  ROC-AUC  │  │  PR-AUC   │  │ Lift Curves │                  │
│             │  (Rank    │  │  (Rare    │  │ (Business   │                  │
│             │  Quality) │  │  Events)  │  │  Impact)    │                  │
│             └───────────┘  └───────────┘  └─────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
ROC CURVE AND AUC FUNDAMENTALS
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  RECEIVER OPERATING CHARACTERISTIC (ROC) CURVE                              │
│                                                                             │
│  Plots True Positive Rate vs False Positive Rate at all thresholds:        │
│                                                                             │
│  True Positive Rate (Sensitivity, Recall):                                  │
│               TP         # Admitted students correctly predicted            │
│  TPR = ─────────────── = ────────────────────────────────────               │
│           TP + FN        # All admitted students                            │
│                                                                             │
│  False Positive Rate (1 - Specificity):                                     │
│               FP         # Rejected students incorrectly predicted          │
│  FPR = ─────────────── = ────────────────────────────────────               │
│           FP + TN        # All rejected students                            │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ROC CURVE                                                         │    │
│  │                                                                     │    │
│  │  TPR (Sensitivity)                                                  │    │
│  │    │                                                                │    │
│  │  1.0┤                              ╭───────○ Perfect                │    │
│  │    │                          ╭────╯       classifier              │    │
│  │  0.8┤                     ╭───╯                                     │    │
│  │    │                  ╭───╯                                         │    │
│  │  0.6┤              ╭──╯         ╱                                   │    │
│  │    │           ╭───╯         ╱ Random                               │    │
│  │  0.4┤       ╭──╯          ╱   (AUC = 0.5)                           │    │
│  │    │    ╭───╯          ╱                                            │    │
│  │  0.2┤ ╭─╯           ╱                                               │    │
│  │    │╭╯           ╱                                                  │    │
│  │  0.0○─────────────────────────────────────────→                     │    │
│  │    0.0   0.2   0.4   0.6   0.8   1.0                                │    │
│  │                    FPR (1 - Specificity)                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  AUC = Area Under the ROC Curve                                             │
│  - AUC = 0.5: Random classifier (diagonal line)                             │
│  - AUC = 1.0: Perfect classifier (upper-left corner)                        │
│  - AUC < 0.5: Worse than random (predictions inverted)                      │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
PROBABILISTIC INTERPRETATION OF ROC-AUC
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  ROC-AUC = CONCORDANCE PROBABILITY                                          │
│                                                                             │
│  P(score(positive) > score(negative))                                       │
│                                                                             │
│  If we randomly select one admitted and one rejected student,               │
│  AUC = probability that admitted student has higher predicted score.       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  EXAMPLE                                                           │    │
│  │                                                                     │    │
│  │  Admitted students:  P(admit) = [0.9, 0.8, 0.7, 0.6]                │    │
│  │  Rejected students:  P(admit) = [0.4, 0.3, 0.2, 0.1]                │    │
│  │                                                                     │    │
│  │  All pairs (admitted, rejected):                                   │    │
│  │  (0.9, 0.4): 0.9 > 0.4 ✓    (0.9, 0.3): 0.9 > 0.3 ✓  ...          │    │
│  │                                                                     │    │
│  │  Concordant pairs: All 16 pairs have higher admitted score         │    │
│  │  AUC = 16/16 = 1.0 (perfect discrimination)                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  This interpretation is useful for:                                         │
│  - Explaining model performance to stakeholders                             │
│  - Understanding what AUC measures in practice                              │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
PRECISION-RECALL CURVES
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  PR CURVES FOR IMBALANCED DATA                                              │
│                                                                             │
│  When positive class is rare, ROC-AUC can be misleading.                    │
│  PR curves focus on positive class performance.                             │
│                                                                             │
│  Precision:                                                                 │
│              TP          # Correct positive predictions                     │
│  Precision = ────── = ────────────────────────────                          │
│             TP + FP     # All positive predictions                          │
│                                                                             │
│  Recall (= Sensitivity = TPR):                                              │
│              TP          # Correct positive predictions                     │
│  Recall = ────────── = ────────────────────────────                         │
│           TP + FN       # All actual positives                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PRECISION-RECALL CURVE                                            │    │
│  │                                                                     │    │
│  │  Precision                                                          │    │
│  │    │                                                                │    │
│  │  1.0○────────○                                                      │    │
│  │    │         ╲                                                      │    │
│  │  0.8┤          ╲──○                                                 │    │
│  │    │              ╲                                                 │    │
│  │  0.6┤               ╲──○                                            │    │
│  │    │                   ╲                                            │    │
│  │  0.4┤                    ╲──○                                       │    │
│  │    │        ────────────────── Random baseline                      │    │
│  │  0.2┤                         (= positive rate)                     │    │
│  │    │                                                                │    │
│  │  0.0┼────────────────────────────────────────────→                  │    │
│  │    0.0   0.2   0.4   0.6   0.8   1.0                                │    │
│  │                     Recall                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  PR-AUC interpretation:                                                     │
│  - Random baseline = positive class proportion                              │
│  - PR-AUC = 1.0: Perfect (never misses positives, no false positives)      │
│  - PR-AUC better than ROC-AUC for imbalanced data                          │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
WHEN TO USE ROC-AUC VS PR-AUC
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  ROC-AUC vs PR-AUC DECISION GUIDE                                           │
│                                                                             │
│  Use ROC-AUC when:                                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ • Classes are balanced (roughly 50/50)                              │   │
│  │ • False positives and false negatives have similar cost             │   │
│  │ • You care about performance across all thresholds                  │   │
│  │ • Standard comparison across different models                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Use PR-AUC when:                                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ • Positive class is rare (<20% of data)                             │   │
│  │ • You care more about positive class predictions                    │   │
│  │ • False positives are costly                                        │   │
│  │ • Dataset is imbalanced (e.g., competitive programs)                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  For university admissions:                                                 │
│  - Highly competitive programs: PR-AUC (low admission rate)               │
│  - General programs: ROC-AUC (balanced outcomes)                            │
│  - Report BOTH for complete picture                                         │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
LIFT CURVES AND GAIN CHARTS
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  LIFT CURVES - BUSINESS IMPACT VISUALIZATION                               │
│                                                                             │
│  Lift = How much better is the model than random selection?                 │
│                                                                             │
│              % of positives captured at threshold                           │
│  Lift = ─────────────────────────────────────────────                       │
│              % of population selected at threshold                          │
│                                                                             │
│  Example: "Top 20% of model scores contain 60% of admits"                   │
│  Lift = 60% / 20% = 3.0                                                     │
│  (3× better than random selection)                                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  LIFT CURVE                                                        │    │
│  │                                                                     │    │
│  │  Lift                                                               │    │
│  │    │                                                                │    │
│  │  5.0┤○                                                              │    │
│  │    │ ╲                                                              │    │
│  │  4.0┤  ╲                                                            │    │
│  │    │   ╲                                                            │    │
│  │  3.0┤    ╲──○                                                       │    │
│  │    │        ╲                                                       │    │
│  │  2.0┤         ╲──○                                                  │    │
│  │    │              ╲──○                                              │    │
│  │  1.0┤─────────────────────○──────────── Random (lift=1)             │    │
│  │    │                                                                │    │
│  │  0.0┼────────────────────────────────────────────→                  │    │
│  │    0%   20%   40%   60%   80%   100%                                │    │
│  │           % of Population (by model score)                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  CUMULATIVE GAINS CHART                                                     │
│  Shows % of positives captured vs % of population                           │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  % Positives                                                       │    │
│  │  Captured                                                           │    │
│  │    │                                                                │    │
│  │ 100%┤                                 ╭────○ Perfect                │    │
│  │    │                           ╭──────╯                             │    │
│  │  80%┤                     ╭────╯     ╱                              │    │
│  │    │                 ╭────╯       ╱ Model                           │    │
│  │  60%┤            ╭───╯         ╱                                    │    │
│  │    │        ╭────╯          ╱                                       │    │
│  │  40%┤   ╭───╯            ╱                                          │    │
│  │    │╭───╯             ╱ Random                                      │    │
│  │  20%┤              ╱    (diagonal)                                  │    │
│  │    │            ╱                                                   │    │
│  │   0%○─────────────────────────────────────────→                     │    │
│  │    0%   20%   40%   60%   80%   100%                                │    │
│  │              % of Population                                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
F1 SCORE AND THRESHOLD SELECTION
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  F1 SCORE - HARMONIC MEAN OF PRECISION AND RECALL                          │
│                                                                             │
│              2 × Precision × Recall                                         │
│  F1 = ────────────────────────────────                                      │
│           Precision + Recall                                                │
│                                                                             │
│  Properties:                                                                │
│  - Range: [0, 1], higher is better                                          │
│  - Balances precision and recall                                            │
│  - F1 = 1 only when both precision and recall are 1                        │
│  - Low if either precision or recall is low                                 │
│                                                                             │
│  F1 ACROSS THRESHOLDS                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Score                                                             │    │
│  │    │                 F1                                            │    │
│  │  1.0┤            ╭───────╮                                          │    │
│  │    │           ╱         ╲ ← Optimal threshold                     │    │
│  │  0.8┤         ╱           ╲                                         │    │
│  │    │   ╭─────╯             ╲───── Precision                         │    │
│  │  0.6┤  │                                                            │    │
│  │    │  │         Recall ─────╲                                       │    │
│  │  0.4┤ │                      ╲                                      │    │
│  │    │ │                        ╲                                     │    │
│  │  0.2┤│                         ╲                                    │    │
│  │    │                            ╲                                   │    │
│  │  0.0┼────────────────────────────────────────→                      │    │
│  │    0.0   0.2   0.4   0.6   0.8   1.0                                │    │
│  │                  Threshold                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘


Author: Grade Prediction Team
Course Context: STA257 (Probability), CSC148 (OOP)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# =============================================================================
# CONFIGURATION AND RESULT DATACLASSES
# =============================================================================

@dataclass
class DiscriminationConfig:
    """
    Configuration for discrimination metric computation.

    ┌─────────────────────────────────────────────────────────────┐
    │  DISCRIMINATION CONFIGURATION                               │
    │                                                             │
    │  n_thresholds: int                                          │
    │    Number of thresholds for curve computation               │
    │    More = smoother curves, slower computation               │
    │                                                             │
    │  positive_label: int                                        │
    │    Which class is considered positive (default 1)           │
    │                                                             │
    │  lift_percentiles: List[int]                                │
    │    Percentiles for lift calculation [10, 20, 30, ...]       │
    └─────────────────────────────────────────────────────────────┘
    """
    n_thresholds: int = 1000
    positive_label: int = 1
    lift_percentiles: List[int] = field(default_factory=lambda: [10, 20, 30, 50])


@dataclass
class ROCResult:
    """
    Results from ROC curve computation.

    ┌─────────────────────────────────────────────────────────────┐
    │  ROC RESULT COMPONENTS                                      │
    │                                                             │
    │  fpr: False Positive Rates at each threshold                │
    │  tpr: True Positive Rates at each threshold                 │
    │  thresholds: Threshold values used                          │
    │  auc: Area Under the ROC Curve                              │
    │  auc_ci: Confidence interval for AUC (if computed)          │
    │  optimal_threshold: Threshold maximizing Youden's J         │
    └─────────────────────────────────────────────────────────────┘
    """
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float
    auc_ci: Optional[Tuple[float, float]] = None
    optimal_threshold: Optional[float] = None


@dataclass
class PRResult:
    """
    Results from Precision-Recall curve computation.

    ┌─────────────────────────────────────────────────────────────┐
    │  PR RESULT COMPONENTS                                       │
    │                                                             │
    │  precision: Precision at each threshold                     │
    │  recall: Recall at each threshold                           │
    │  thresholds: Threshold values used                          │
    │  auc: Area Under the PR Curve                               │
    │  ap: Average Precision (alternative to PR-AUC)              │
    │  f1_scores: F1 score at each threshold                      │
    │  optimal_threshold: Threshold maximizing F1                 │
    └─────────────────────────────────────────────────────────────┘
    """
    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray
    auc: float
    ap: float
    f1_scores: np.ndarray
    optimal_threshold: Optional[float] = None


@dataclass
class LiftResult:
    """
    Results from lift/gain analysis.

    ┌─────────────────────────────────────────────────────────────┐
    │  LIFT RESULT COMPONENTS                                     │
    │                                                             │
    │  percentiles: Population percentiles [10, 20, ...]          │
    │  lift_values: Lift at each percentile                       │
    │  gains: Cumulative gain at each percentile                  │
    │  captures: % of positives captured at each percentile       │
    └─────────────────────────────────────────────────────────────┘
    """
    percentiles: np.ndarray
    lift_values: np.ndarray
    gains: np.ndarray
    captures: np.ndarray


@dataclass
class DiscriminationResult:
    """
    Complete discrimination analysis results.
    """
    roc: ROCResult
    pr: PRResult
    lift: LiftResult
    summary: Dict[str, float]


# =============================================================================
# ROC CURVE FUNCTIONS
# =============================================================================

def compute_roc_curve(y_true: np.ndarray,
                       y_prob: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve (FPR, TPR) at various thresholds.

    ┌─────────────────────────────────────────────────────────────┐
    │  IMPLEMENTATION STEPS                                       │
    │                                                             │
    │  1. Sort predictions in descending order                    │
    │     - Indices that would sort y_prob descending             │
    │     - Reorder y_true accordingly                            │
    │                                                             │
    │  2. For each unique threshold (from high to low):           │
    │     - Classify: predict 1 if score > threshold              │
    │     - Count TP, FP, TN, FN                                  │
    │     - Compute TPR = TP / (TP + FN)                          │
    │     - Compute FPR = FP / (FP + TN)                          │
    │                                                             │
    │  3. Efficient implementation:                               │
    │     - Use cumulative sums instead of recomputing            │
    │     - Sort once, then walk through thresholds               │
    │                                                             │
    │  4. Add endpoints (0,0) and (1,1) for complete curve        │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels, shape (n,)
        y_prob: Predicted probabilities, shape (n,)

    Returns:
        Tuple of (fpr, tpr, thresholds)
        - fpr: False positive rates, shape (n_thresholds,)
        - tpr: True positive rates, shape (n_thresholds,)
        - thresholds: Threshold values, shape (n_thresholds,)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Sort by descending probability
    sorted_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]

    # Total positives and negatives
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    # Walk through sorted list accumulating TP and FP
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    tpr_vals = tps / P
    fpr_vals = fps / N

    # Find distinct threshold values (keep last occurrence of each threshold)
    distinct_indices = np.where(np.diff(y_prob_sorted))[0]
    # Also include the last index
    threshold_indices = np.concatenate([distinct_indices, [len(y_prob_sorted) - 1]])

    tpr_vals = tpr_vals[threshold_indices]
    fpr_vals = fpr_vals[threshold_indices]
    thresholds = y_prob_sorted[threshold_indices]

    # Prepend (0, 0) start point
    tpr_vals = np.concatenate([[0.0], tpr_vals])
    fpr_vals = np.concatenate([[0.0], fpr_vals])
    thresholds = np.concatenate([[thresholds[0] + 1e-10], thresholds])

    return (fpr_vals, tpr_vals, thresholds)


def roc_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve.

    ┌─────────────────────────────────────────────────────────────┐
    │  AUC COMPUTATION VIA TRAPEZOIDAL RULE                       │
    │                                                             │
    │  AUC = Σᵢ (FPRᵢ₊₁ - FPRᵢ) × (TPRᵢ₊₁ + TPRᵢ) / 2            │
    │                                                             │
    │  ┌─────────────────────────────────────────────┐            │
    │  │     ○                                       │            │
    │  │    /│                                       │            │
    │  │   / │ ← Area of trapezoid                  │            │
    │  │  ○──│                                       │            │
    │  │  │  │                                       │            │
    │  │──┴──┴───────────────────────────            │            │
    │  │  Δx = FPRᵢ₊₁ - FPRᵢ                         │            │
    │  │  Area = Δx × (TPRᵢ₊₁ + TPRᵢ) / 2           │            │
    │  └─────────────────────────────────────────────┘            │
    │                                                             │
    │  Alternative: Mann-Whitney U statistic (more direct)        │
    │                                                             │
    │  AUC = P(score(positive) > score(negative))                 │
    │      = (# concordant pairs) / (# positive × # negative)    │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities

    Returns:
        AUC score in range [0, 1]
    """
    fpr, tpr, _ = compute_roc_curve(y_true, y_prob)
    # Trapezoidal rule
    auc = float(np.trapz(tpr, fpr))
    return auc


def roc_auc_confidence_interval(y_true: np.ndarray,
                                 y_prob: np.ndarray,
                                 confidence_level: float = 0.95,
                                 n_bootstrap: int = 1000
                                 ) -> Tuple[float, float, float]:
    """
    Compute confidence interval for ROC-AUC using bootstrap.

    ┌─────────────────────────────────────────────────────────────┐
    │  BOOTSTRAP CONFIDENCE INTERVAL                              │
    │                                                             │
    │  1. Resample (y_true, y_prob) with replacement B times      │
    │  2. Compute AUC for each bootstrap sample                   │
    │  3. Use percentile method for CI:                           │
    │     - Lower = (1 - confidence) / 2 percentile               │
    │     - Upper = 1 - (1 - confidence) / 2 percentile           │
    │                                                             │
    │  Alternative: DeLong's method (parametric, faster)          │
    │  But bootstrap is more general and assumption-free          │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        confidence_level: e.g., 0.95 for 95% CI
        n_bootstrap: Number of bootstrap samples

    Returns:
        Tuple of (auc, ci_lower, ci_upper)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Compute the point estimate AUC
    auc = roc_auc_score(y_true, y_prob)

    # Bootstrap
    rng = np.random.RandomState(42)
    n = len(y_true)
    bootstrap_aucs = []

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]

        # Need at least one positive and one negative
        if np.sum(y_true_boot == 1) == 0 or np.sum(y_true_boot == 0) == 0:
            continue

        bootstrap_aucs.append(roc_auc_score(y_true_boot, y_prob_boot))

    bootstrap_aucs = np.array(bootstrap_aucs)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(bootstrap_aucs, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_aucs, 100 * (1 - alpha / 2)))

    return (auc, ci_lower, ci_upper)


def optimal_threshold_roc(y_true: np.ndarray,
                           y_prob: np.ndarray,
                           method: str = 'youden'
                           ) -> float:
    """
    Find optimal classification threshold from ROC curve.

    ┌─────────────────────────────────────────────────────────────┐
    │  THRESHOLD SELECTION METHODS                                │
    │                                                             │
    │  YOUDEN'S J STATISTIC (default):                            │
    │  J = TPR - FPR = Sensitivity + Specificity - 1              │
    │  Maximize J to find optimal threshold                       │
    │                                                             │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │  Youden's J visualized:                             │    │
    │  │                                                     │    │
    │  │  TPR │                    ○ optimal                 │    │
    │  │      │                  ╱  ← Maximum vertical       │    │
    │  │      │               ╱     distance from diagonal   │    │
    │  │      │            ╱                                 │    │
    │  │      │         ╱                                    │    │
    │  │      │      ╱                                       │    │
    │  │      └────╱─────────────────────→ FPR              │    │
    │  └─────────────────────────────────────────────────────┘    │
    │                                                             │
    │  COST-BASED:                                                │
    │  Minimize: cost_FP × FPR + cost_FN × FNR                    │
    │                                                             │
    │  CLOSEST TO (0,1):                                          │
    │  Minimize: sqrt(FPR² + (1-TPR)²)                            │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        method: 'youden', 'closest', or 'cost'

    Returns:
        Optimal threshold value
    """
    fpr, tpr, thresholds = compute_roc_curve(y_true, y_prob)

    if method == 'youden':
        # Maximize TPR - FPR (Youden's J statistic)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
    elif method == 'closest':
        # Minimize distance to (0, 1) -- the perfect corner
        distances = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
        best_idx = np.argmin(distances)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'youden' or 'closest'.")

    return float(thresholds[best_idx])


# =============================================================================
# PRECISION-RECALL FUNCTIONS
# =============================================================================

def compute_pr_curve(y_true: np.ndarray,
                      y_prob: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Precision-Recall curve at various thresholds.

    ┌─────────────────────────────────────────────────────────────┐
    │  IMPLEMENTATION STEPS                                       │
    │                                                             │
    │  1. Sort predictions in descending order                    │
    │                                                             │
    │  2. For each threshold (high to low):                       │
    │     - TP = count(y_true=1 AND y_prob > threshold)           │
    │     - FP = count(y_true=0 AND y_prob > threshold)           │
    │     - FN = count(y_true=1 AND y_prob ≤ threshold)           │
    │     - Precision = TP / (TP + FP)                            │
    │     - Recall = TP / (TP + FN)                               │
    │                                                             │
    │  3. Handle edge cases:                                      │
    │     - If no predictions above threshold: precision = 1      │
    │     - If no positives: recall = 0                           │
    │                                                             │
    │  4. Add endpoint (precision=pos_rate, recall=1)             │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities

    Returns:
        Tuple of (precision, recall, thresholds)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Sort by descending probability
    sorted_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]

    P = np.sum(y_true == 1)

    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    precision_vals = tps / (tps + fps)
    recall_vals = tps / P

    # Find distinct threshold values
    distinct_indices = np.where(np.diff(y_prob_sorted))[0]
    threshold_indices = np.concatenate([distinct_indices, [len(y_prob_sorted) - 1]])

    precision_vals = precision_vals[threshold_indices]
    recall_vals = recall_vals[threshold_indices]
    thresholds = y_prob_sorted[threshold_indices]

    # Prepend start point: precision=1, recall=0
    precision_vals = np.concatenate([[1.0], precision_vals])
    recall_vals = np.concatenate([[0.0], recall_vals])
    thresholds = np.concatenate([[thresholds[0] + 1e-10], thresholds])

    return (precision_vals, recall_vals, thresholds)


def pr_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Area Under the Precision-Recall Curve.

    ┌─────────────────────────────────────────────────────────────┐
    │  PR-AUC COMPUTATION                                         │
    │                                                             │
    │  Use trapezoidal rule on (recall, precision) curve          │
    │                                                             │
    │  AUC = Σᵢ (Rᵢ₊₁ - Rᵢ) × (Pᵢ₊₁ + Pᵢ) / 2                    │
    │                                                             │
    │  Where R = recall, P = precision                            │
    │                                                             │
    │  Note: Curve should be monotonically interpolated           │
    │  to avoid underestimating the area                          │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities

    Returns:
        PR-AUC score in range [0, 1]
    """
    precision, recall, _ = compute_pr_curve(y_true, y_prob)
    # Sort by recall for proper integration
    sort_idx = np.argsort(recall)
    recall_sorted = recall[sort_idx]
    precision_sorted = precision[sort_idx]
    auc = float(np.trapz(precision_sorted, recall_sorted))
    return auc


def average_precision_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Average Precision (AP).

    ┌─────────────────────────────────────────────────────────────┐
    │  AVERAGE PRECISION                                          │
    │                                                             │
    │  AP = Σₙ (Rₙ - Rₙ₋₁) × Pₙ                                   │
    │                                                             │
    │  Where the sum is over all thresholds where recall changes  │
    │                                                             │
    │  AP vs PR-AUC:                                              │
    │  - AP: Weighted sum of precisions at each threshold         │
    │  - PR-AUC: Area under interpolated curve                    │
    │  - Often similar, AP sometimes preferred                    │
    │                                                             │
    │  AP is equivalent to PR-AUC when using step interpolation   │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities

    Returns:
        Average precision score
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Sort by descending probability
    sorted_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[sorted_indices]

    P = np.sum(y_true == 1)

    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    precision_vals = tps / (tps + fps)
    recall_vals = tps / P

    # AP = sum((R_n - R_{n-1}) * P_n) over all positions where a positive appears
    # Prepend recall=0
    recall_with_zero = np.concatenate([[0.0], recall_vals])
    delta_recall = np.diff(recall_with_zero)

    ap = float(np.sum(delta_recall * precision_vals))
    return ap


def f1_at_threshold(y_true: np.ndarray,
                     y_prob: np.ndarray,
                     threshold: float = 0.5) -> float:
    """
    Compute F1 score at a specific threshold.

    ┌─────────────────────────────────────────────────────────────┐
    │  F1 SCORE COMPUTATION                                       │
    │                                                             │
    │  1. Binarize predictions: y_pred = (y_prob > threshold)     │
    │  2. Compute confusion matrix elements                       │
    │  3. Precision = TP / (TP + FP)                              │
    │  4. Recall = TP / (TP + FN)                                 │
    │  5. F1 = 2 × Precision × Recall / (Precision + Recall)     │
    │                                                             │
    │  Handle edge cases:                                         │
    │  - If TP = 0: F1 = 0                                        │
    │  - If TP + FP = 0: Precision undefined → F1 = 0             │
    └─────────────────────────────────────────────────────────────┘
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0

    f1 = 2.0 * precision * recall / (precision + recall)
    return float(f1)


def optimal_threshold_f1(y_true: np.ndarray,
                          y_prob: np.ndarray
                          ) -> Tuple[float, float]:
    """
    Find threshold that maximizes F1 score.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities

    Returns:
        Tuple of (optimal_threshold, max_f1_score)
    """
    # Try many thresholds between 0 and 1
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_f1 = -1.0
    best_threshold = 0.5

    for t in thresholds:
        f1 = f1_at_threshold(y_true, y_prob, threshold=t)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return (float(best_threshold), float(best_f1))


# =============================================================================
# LIFT AND GAINS
# =============================================================================

def compute_lift_curve(y_true: np.ndarray,
                        y_prob: np.ndarray,
                        n_bins: int = 10
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute lift curve.

    ┌─────────────────────────────────────────────────────────────┐
    │  LIFT CURVE COMPUTATION                                     │
    │                                                             │
    │  1. Sort by predicted probability (descending)              │
    │  2. Divide into n_bins equal groups                         │
    │  3. For each group (top k%):                                │
    │     - Compute actual positive rate in group                 │
    │     - Compute overall positive rate                         │
    │     - Lift = group_rate / overall_rate                      │
    │                                                             │
    │  Example:                                                   │
    │  - Overall positive rate: 30%                               │
    │  - Top 10% has 90% positives                                │
    │  - Lift at 10% = 0.90 / 0.30 = 3.0                          │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for lift calculation

    Returns:
        Tuple of (percentiles, lift_values)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Sort descending by probability
    sorted_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[sorted_indices]

    n = len(y_true)
    overall_rate = np.mean(y_true)

    percentiles = []
    lift_values = []

    bin_size = n // n_bins
    if bin_size < 1:
        bin_size = 1

    for i in range(n_bins):
        end = min((i + 1) * bin_size, n)
        if end == 0:
            continue
        # Cumulative: top (i+1) bins
        group = y_true_sorted[:end]
        group_rate = np.mean(group)
        lift = group_rate / overall_rate if overall_rate > 0 else 0.0
        pct = end / n * 100
        percentiles.append(pct)
        lift_values.append(lift)

    return (np.array(percentiles), np.array(lift_values))


def compute_gains_curve(y_true: np.ndarray,
                         y_prob: np.ndarray,
                         n_points: int = 100
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cumulative gains curve.

    ┌─────────────────────────────────────────────────────────────┐
    │  CUMULATIVE GAINS COMPUTATION                               │
    │                                                             │
    │  1. Sort by predicted probability (descending)              │
    │  2. For each percentage of population (1%, 2%, ...):        │
    │     - Count positives in that top percentage                │
    │     - Compute cumulative percentage of all positives        │
    │                                                             │
    │  Returns:                                                   │
    │  - population_percentages: [0.01, 0.02, ..., 1.0]           │
    │  - gains: [% of positives captured at each level]           │
    │                                                             │
    │  Perfect model: Gains = min(1, pop_pct / pos_rate)          │
    │  Random model: Gains = pop_pct (diagonal line)              │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_points: Number of points on the curve

    Returns:
        Tuple of (population_percentages, gains)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Sort descending by probability
    sorted_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[sorted_indices]

    n = len(y_true)
    total_positives = np.sum(y_true)

    if total_positives == 0:
        # No positives at all
        pop_pcts = np.linspace(1.0 / n_points, 1.0, n_points)
        gains = np.zeros(n_points)
        return (pop_pcts, gains)

    cumulative_positives = np.cumsum(y_true_sorted)

    # Sample at n_points evenly spaced fractions of the population
    indices = np.linspace(1, n, n_points, dtype=int)
    # Ensure we don't go out of bounds
    indices = np.clip(indices, 1, n)

    pop_pcts = indices / n
    gains = cumulative_positives[indices - 1] / total_positives

    return (pop_pcts, gains)


def lift_at_percentile(y_true: np.ndarray,
                        y_prob: np.ndarray,
                        percentile: int = 10) -> float:
    """
    Compute lift at a specific percentile.

    Example: lift_at_percentile(y, p, 10) returns lift at top 10%.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        percentile: Top percentile to consider (1-100)

    Returns:
        Lift value (ratio of positive rate in top percentile to overall rate)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Sort descending by probability
    sorted_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[sorted_indices]

    n = len(y_true)
    overall_rate = np.mean(y_true)

    # Take top percentile% of samples
    k = max(1, int(np.ceil(n * percentile / 100.0)))
    top_group = y_true_sorted[:k]
    group_rate = np.mean(top_group)

    if overall_rate == 0:
        return 0.0

    lift = group_rate / overall_rate
    return float(lift)


# =============================================================================
# CONFUSION MATRIX UTILITIES
# =============================================================================

def confusion_matrix_at_threshold(y_true: np.ndarray,
                                   y_prob: np.ndarray,
                                   threshold: float = 0.5
                                   ) -> np.ndarray:
    """
    Compute confusion matrix at a specific threshold.

    ┌─────────────────────────────────────────────────────────────┐
    │  CONFUSION MATRIX                                           │
    │                                                             │
    │                    Predicted                                │
    │                  Neg     Pos                                │
    │              ┌─────────┬─────────┐                          │
    │  Actual  Neg │   TN    │   FP    │                          │
    │              ├─────────┼─────────┤                          │
    │         Pos  │   FN    │   TP    │                          │
    │              └─────────┴─────────┘                          │
    │                                                             │
    │  Returns: [[TN, FP], [FN, TP]]                              │
    └─────────────────────────────────────────────────────────────┘
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    return np.array([[tn, fp], [fn, tp]])


def classification_metrics_at_threshold(y_true: np.ndarray,
                                         y_prob: np.ndarray,
                                         threshold: float = 0.5
                                         ) -> Dict[str, float]:
    """
    Compute all classification metrics at a threshold.

    Returns:
        Dictionary with keys:
        - 'accuracy': Overall accuracy
        - 'precision': Positive predictive value
        - 'recall': Sensitivity, true positive rate
        - 'specificity': True negative rate
        - 'f1': F1 score
        - 'mcc': Matthews correlation coefficient
        - 'balanced_accuracy': (TPR + TNR) / 2
    """
    cm = confusion_matrix_at_threshold(y_true, y_prob, threshold)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    balanced_accuracy = (recall + specificity) / 2.0

    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    denom = np.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1': float(f1),
        'mcc': float(mcc),
        'balanced_accuracy': float(balanced_accuracy),
    }


# =============================================================================
# FULL DISCRIMINATION ANALYSIS
# =============================================================================

def full_discrimination_analysis(y_true: np.ndarray,
                                  y_prob: np.ndarray,
                                  config: Optional[DiscriminationConfig] = None
                                  ) -> DiscriminationResult:
    """
    Perform complete discrimination analysis.

    ┌─────────────────────────────────────────────────────────────┐
    │  FULL DISCRIMINATION ANALYSIS                               │
    │                                                             │
    │  Computes:                                                  │
    │  1. ROC curve and AUC                                       │
    │  2. PR curve and AUC                                        │
    │  3. Lift and gains curves                                   │
    │  4. Optimal thresholds                                      │
    │  5. Summary statistics                                      │
    │                                                             │
    │  Returns DiscriminationResult with all components           │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        config: DiscriminationConfig

    Returns:
        DiscriminationResult with ROC, PR, and lift results
    """
    if config is None:
        config = DiscriminationConfig()

    # ROC
    fpr, tpr, roc_thresholds = compute_roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    opt_thresh_roc = optimal_threshold_roc(y_true, y_prob)

    roc_result = ROCResult(
        fpr=fpr,
        tpr=tpr,
        thresholds=roc_thresholds,
        auc=auc,
        optimal_threshold=opt_thresh_roc,
    )

    # PR
    precision, recall, pr_thresholds = compute_pr_curve(y_true, y_prob)
    pr_auc = pr_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    opt_thresh_f1, best_f1 = optimal_threshold_f1(y_true, y_prob)

    # Compute F1 scores at each PR threshold
    f1_scores = np.array([
        f1_at_threshold(y_true, y_prob, t) for t in pr_thresholds
    ])

    pr_result = PRResult(
        precision=precision,
        recall=recall,
        thresholds=pr_thresholds,
        auc=pr_auc,
        ap=ap,
        f1_scores=f1_scores,
        optimal_threshold=opt_thresh_f1,
    )

    # Lift
    n_bins = len(config.lift_percentiles)
    percentiles_arr, lift_values = compute_lift_curve(y_true, y_prob, n_bins=n_bins)
    pop_pcts, gains = compute_gains_curve(y_true, y_prob)

    # Captures at each percentile
    captures = np.array([
        _capture_at_percentile(y_true, y_prob, p)
        for p in config.lift_percentiles
    ])

    lift_result = LiftResult(
        percentiles=np.array(config.lift_percentiles, dtype=float),
        lift_values=lift_values,
        gains=gains,
        captures=captures,
    )

    # Summary
    summary = {
        'roc_auc': auc,
        'pr_auc': pr_auc,
        'average_precision': ap,
        'optimal_threshold_youden': opt_thresh_roc,
        'optimal_threshold_f1': opt_thresh_f1,
        'best_f1': best_f1,
    }

    return DiscriminationResult(
        roc=roc_result,
        pr=pr_result,
        lift=lift_result,
        summary=summary,
    )


def _capture_at_percentile(y_true, y_prob, percentile):
    """Helper: fraction of positives captured in top percentile%."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    sorted_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[sorted_indices]
    n = len(y_true)
    total_pos = np.sum(y_true)
    if total_pos == 0:
        return 0.0
    k = max(1, int(np.ceil(n * percentile / 100.0)))
    captured = np.sum(y_true_sorted[:k])
    return float(captured / total_pos)


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def compare_discrimination(models: Dict[str, Tuple[np.ndarray, np.ndarray]],
                            config: Optional[DiscriminationConfig] = None
                            ) -> Dict[str, Dict[str, float]]:
    """
    Compare discrimination metrics across multiple models.

    Args:
        models: Dict of model_name → (y_true, y_prob)
        config: DiscriminationConfig

    Returns:
        Dict of model_name → metrics_dict
        metrics_dict contains: roc_auc, pr_auc, lift_10, lift_20, etc.
    """
    if config is None:
        config = DiscriminationConfig()

    results = {}
    for name, (y_true, y_prob) in models.items():
        metrics = {
            'roc_auc': roc_auc_score(y_true, y_prob),
            'pr_auc': pr_auc_score(y_true, y_prob),
        }
        for p in config.lift_percentiles:
            metrics[f'lift_{p}'] = lift_at_percentile(y_true, y_prob, p)
        results[name] = metrics

    return results


def compare_roc_curves(models: Dict[str, Tuple[np.ndarray, np.ndarray]]
                        ) -> Dict[str, ROCResult]:
    """
    Compute ROC curves for multiple models (for overlay plotting).

    Args:
        models: Dict of model_name → (y_true, y_prob)

    Returns:
        Dict of model_name → ROCResult
    """
    results = {}
    for name, (y_true, y_prob) in models.items():
        fpr, tpr, thresholds = compute_roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        results[name] = ROCResult(
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            auc=auc,
        )
    return results


def delong_test(y_true: np.ndarray,
                 y_prob1: np.ndarray,
                 y_prob2: np.ndarray
                 ) -> Tuple[float, float]:
    """
    DeLong test for comparing two ROC curves.

    ┌─────────────────────────────────────────────────────────────┐
    │  DELONG TEST FOR AUC COMPARISON                             │
    │                                                             │
    │  Tests null hypothesis: AUC1 = AUC2                         │
    │  Uses variance estimate based on Mann-Whitney statistics    │
    │                                                             │
    │  Returns:                                                   │
    │  - z_statistic: Standardized difference                     │
    │  - p_value: Two-sided p-value                               │
    │                                                             │
    │  If p < 0.05: Significant difference between models         │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels (same for both models)
        y_prob1: Probabilities from model 1
        y_prob2: Probabilities from model 2

    Returns:
        Tuple of (z_statistic, p_value)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob1 = np.asarray(y_prob1, dtype=float)
    y_prob2 = np.asarray(y_prob2, dtype=float)

    # Separate positive and negative indices
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    m = len(pos_idx)  # number of positives
    n = len(neg_idx)  # number of negatives

    # Compute placement values for each model
    # For positives: V_i = fraction of negatives with score < score_i
    # For negatives: V_j = fraction of positives with score > score_j

    def placement_values(y_prob):
        """Compute structural components for DeLong variance."""
        # For each positive, count fraction of negatives scored lower
        pos_scores = y_prob[pos_idx]
        neg_scores = y_prob[neg_idx]

        # V10: for each positive, fraction of negatives with lower score
        v10 = np.zeros(m)
        for i in range(m):
            v10[i] = np.mean(pos_scores[i] > neg_scores) + 0.5 * np.mean(pos_scores[i] == neg_scores)

        # V01: for each negative, fraction of positives with higher score
        v01 = np.zeros(n)
        for j in range(n):
            v01[j] = np.mean(pos_scores > neg_scores[j]) + 0.5 * np.mean(pos_scores == neg_scores[j])

        return v10, v01

    v10_1, v01_1 = placement_values(y_prob1)
    v10_2, v01_2 = placement_values(y_prob2)

    auc1 = roc_auc_score(y_true, y_prob1)
    auc2 = roc_auc_score(y_true, y_prob2)

    # Covariance matrix of (AUC1, AUC2)
    # S10 = cov of placement values among positives
    # S01 = cov of placement values among negatives
    s10 = np.cov(v10_1, v10_2)[0, 1] if m > 1 else 0.0
    s01 = np.cov(v01_1, v01_2)[0, 1] if n > 1 else 0.0

    # Variance of (AUC1 - AUC2)
    var10_1 = np.var(v10_1, ddof=1) if m > 1 else 0.0
    var10_2 = np.var(v10_2, ddof=1) if m > 1 else 0.0
    var01_1 = np.var(v01_1, ddof=1) if n > 1 else 0.0
    var01_2 = np.var(v01_2, ddof=1) if n > 1 else 0.0

    var_auc_diff = (
        (var10_1 + var10_2 - 2 * s10) / m +
        (var01_1 + var01_2 - 2 * s01) / n
    )

    if var_auc_diff <= 0:
        # If variance is zero, AUCs are identical
        return (0.0, 1.0)

    se = np.sqrt(var_auc_diff)
    z = (auc1 - auc2) / se

    # Two-sided p-value using standard normal CDF
    # Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    from math import erf, sqrt
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))

    return (float(z), float(p_value))


# =============================================================================
# SUBGROUP ANALYSIS
# =============================================================================

def subgroup_discrimination(y_true: np.ndarray,
                             y_prob: np.ndarray,
                             groups: np.ndarray,
                             config: Optional[DiscriminationConfig] = None
                             ) -> Dict[Any, Dict[str, float]]:
    """
    Compute discrimination metrics separately for subgroups.

    ┌─────────────────────────────────────────────────────────────┐
    │  SUBGROUP DISCRIMINATION ANALYSIS                           │
    │                                                             │
    │  Important for fairness evaluation:                         │
    │  - Does model discriminate equally across groups?           │
    │  - Are there groups where model performs poorly?            │
    │                                                             │
    │  Example output:                                            │
    │  {                                                          │
    │    'Ontario': {'roc_auc': 0.85, 'pr_auc': 0.72, ...},       │
    │    'Quebec': {'roc_auc': 0.82, 'pr_auc': 0.68, ...},        │
    │    'BC': {'roc_auc': 0.88, 'pr_auc': 0.75, ...}             │
    │  }                                                          │
    └─────────────────────────────────────────────────────────────┘

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        groups: Group labels for each sample
        config: DiscriminationConfig

    Returns:
        Dict of group → metrics_dict
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    groups = np.asarray(groups)

    if config is None:
        config = DiscriminationConfig()

    unique_groups = np.unique(groups)
    results = {}

    for group in unique_groups:
        mask = groups == group
        y_true_g = y_true[mask]
        y_prob_g = y_prob[mask]

        # Need at least one positive and one negative to compute metrics
        if np.sum(y_true_g == 1) == 0 or np.sum(y_true_g == 0) == 0:
            results[group] = {
                'roc_auc': float('nan'),
                'pr_auc': float('nan'),
                'n_samples': int(np.sum(mask)),
                'n_positive': int(np.sum(y_true_g == 1)),
                'n_negative': int(np.sum(y_true_g == 0)),
            }
            continue

        metrics = {
            'roc_auc': roc_auc_score(y_true_g, y_prob_g),
            'pr_auc': pr_auc_score(y_true_g, y_prob_g),
            'n_samples': int(np.sum(mask)),
            'n_positive': int(np.sum(y_true_g == 1)),
            'n_negative': int(np.sum(y_true_g == 0)),
        }
        results[group] = metrics

    return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_discrimination_inputs(y_true: np.ndarray,
                                    y_prob: np.ndarray) -> bool:
    """
    Validate inputs for discrimination functions.

    Checks:
        1. Same length arrays
        2. y_true is binary (0, 1)
        3. y_prob in [0, 1]
        4. At least one positive and one negative case
        5. No NaN values

    Returns:
        True if inputs are valid.

    Raises:
        ValueError: If any validation check fails.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Check for NaN values
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_prob)):
        raise ValueError("Inputs contain NaN values.")

    # Same length
    if len(y_true) != len(y_prob):
        raise ValueError(
            f"y_true and y_prob must have the same length, "
            f"got {len(y_true)} and {len(y_prob)}."
        )

    # y_true is binary
    unique_labels = set(np.unique(y_true))
    if not unique_labels.issubset({0.0, 1.0}):
        raise ValueError(
            f"y_true must be binary (0 or 1), got unique values: {unique_labels}."
        )

    # y_prob in [0, 1]
    if np.any(y_prob < 0) or np.any(y_prob > 1):
        raise ValueError("y_prob values must be in [0, 1].")

    # At least one positive and one negative
    if np.sum(y_true == 1) == 0:
        raise ValueError("y_true must contain at least one positive case (1).")
    if np.sum(y_true == 0) == 0:
        raise ValueError("y_true must contain at least one negative case (0).")

    return True


def rank_order_statistics(y_true: np.ndarray,
                           y_prob: np.ndarray
                           ) -> Dict[str, float]:
    """
    Compute rank-order statistics.

    Returns:
        - concordance: Fraction of concordant pairs
        - discordance: Fraction of discordant pairs
        - ties: Fraction of tied pairs
        - somers_d: Somers' D statistic (2 × AUC - 1)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    pos_scores = y_prob[pos_idx]
    neg_scores = y_prob[neg_idx]

    concordant = 0
    discordant = 0
    tied = 0

    for ps in pos_scores:
        concordant += int(np.sum(ps > neg_scores))
        discordant += int(np.sum(ps < neg_scores))
        tied += int(np.sum(ps == neg_scores))

    total_pairs = len(pos_scores) * len(neg_scores)
    if total_pairs == 0:
        return {
            'concordance': 0.0,
            'discordance': 0.0,
            'ties': 0.0,
            'somers_d': 0.0,
        }

    auc = roc_auc_score(y_true, y_prob)

    return {
        'concordance': float(concordant / total_pairs),
        'discordance': float(discordant / total_pairs),
        'ties': float(tied / total_pairs),
        'somers_d': float(2 * auc - 1),
    }


# =============================================================================
# TODO LIST FOR IMPLEMENTATION
# =============================================================================
"""
TODO: Implementation Checklist

ROC CURVE:
□ compute_roc_curve()
  - [ ] Sort predictions efficiently
  - [ ] Compute TPR/FPR at each threshold
  - [ ] Handle ties in predictions
  - [ ] Add (0,0) and (1,1) endpoints

□ roc_auc_score()
  - [ ] Implement trapezoidal integration
  - [ ] Alternative: Mann-Whitney U approach
  - [ ] Handle edge cases

□ roc_auc_confidence_interval()
  - [ ] Implement bootstrap sampling
  - [ ] Compute percentile-based CI
  - [ ] Consider DeLong method alternative

□ optimal_threshold_roc()
  - [ ] Implement Youden's J
  - [ ] Implement closest to (0,1)
  - [ ] Add cost-based method

PR CURVE:
□ compute_pr_curve()
  - [ ] Compute precision/recall at thresholds
  - [ ] Handle zero-division cases
  - [ ] Proper interpolation

□ pr_auc_score()
  - [ ] Implement area calculation
  - [ ] Use proper interpolation

□ average_precision_score()
  - [ ] Implement step-wise AP
  - [ ] Handle all thresholds

□ optimal_threshold_f1()
  - [ ] Compute F1 at all thresholds
  - [ ] Return maximum

LIFT AND GAINS:
□ compute_lift_curve()
  - [ ] Sort by probability
  - [ ] Compute lift per bin
  - [ ] Handle edge cases

□ compute_gains_curve()
  - [ ] Cumulative positive counts
  - [ ] Normalize to percentages

STATISTICAL TESTS:
□ delong_test()
  - [ ] Compute variance estimates
  - [ ] Implement z-test
  - [ ] Return p-value

SUBGROUP ANALYSIS:
□ subgroup_discrimination()
  - [ ] Split by groups
  - [ ] Compute metrics per group
  - [ ] Handle small groups

TESTING:
□ Unit tests
  - [ ] Test with perfect predictions
  - [ ] Test with random predictions
  - [ ] Test with inverted predictions
  - [ ] Verify AUC = 0.5 for random

□ Integration tests
  - [ ] Test with model outputs
  - [ ] Compare with sklearn metrics

DOCUMENTATION:
□ Add interpretation guidelines
□ Include example visualizations
□ Reference STA257 concepts
"""
