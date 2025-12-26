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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


def compare_roc_curves(models: Dict[str, Tuple[np.ndarray, np.ndarray]]
                        ) -> Dict[str, ROCResult]:
    """
    Compute ROC curves for multiple models (for overlay plotting).

    Args:
        models: Dict of model_name → (y_true, y_prob)

    Returns:
        Dict of model_name → ROCResult
    """
    pass


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
    pass


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
    pass


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_discrimination_inputs(y_true: np.ndarray,
                                    y_prob: np.ndarray) -> None:
    """
    Validate inputs for discrimination functions.

    Checks:
        1. Same length arrays
        2. y_true is binary (0, 1)
        3. y_prob in [0, 1]
        4. At least one positive and one negative case
        5. No NaN values
    """
    pass


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
    pass


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
