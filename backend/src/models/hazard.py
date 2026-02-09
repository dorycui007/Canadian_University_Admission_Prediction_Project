"""
Discrete-Time Hazard Model for Decision Timing
===============================================

This module implements a discrete-time survival analysis model to predict
WHEN students receive admission decisions, not just whether they're admitted.
This extends logistic regression to handle time-to-event data.

STATISTICAL REFERENCE: Discrete-time survival analysis, hazard models

================================================================================
                        SYSTEM ARCHITECTURE CONTEXT
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    WHERE THIS MODULE FITS                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │                     logistic.py (IRLS)                                   │
    │                           │                                              │
    │                           ▼                                              │
    │                 ┌─────────────────┐                                     │
    │                 │  [THIS MODULE]  │                                     │
    │                 │   hazard.py     │                                     │
    │                 │  Decision Timing│                                     │
    │                 └────────┬────────┘                                     │
    │                          │                                               │
    │                          ▼                                               │
    │              ┌─────────────────────────┐                                │
    │              │    OUTPUT PREDICTION    │                                │
    │              │                         │                                │
    │              │  "72% admission chance  │                                │
    │              │   Decision likely in    │                                │
    │              │   week 8-12 (March)"    │                                │
    │              │                         │                                │
    │              └─────────────────────────┘                                │
    │                                                                          │
    │  QUESTION THIS ANSWERS:                                                  │
    │  ──────────────────────                                                  │
    │  Given that a student applies, when will they hear back?                │
    │                                                                          │
    │  APPLICATIONS:                                                           │
    │  ─────────────                                                           │
    │  • "You'll likely hear by mid-March"                                    │
    │  • "Rolling admissions: expect 2-3 week response"                       │
    │  • "Most competitive programs decide in April"                          │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    DISCRETE-TIME SURVIVAL ANALYSIS
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  THE TIMING PREDICTION PROBLEM                                           │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  DATA STRUCTURE:                                                         │
    │  ────────────────                                                        │
    │  For each application:                                                   │
    │  • t = week number when decision received (1, 2, 3, ..., T)            │
    │  • x = features (average, university, program, etc.)                    │
    │                                                                          │
    │  TIMELINE VISUALIZATION:                                                 │
    │  ────────────────────────                                                │
    │                                                                          │
    │      Apply    Week 1  Week 2  Week 3  ...  Week T                       │
    │        │        │       │       │           │                            │
    │        ▼        ▼       ▼       ▼           ▼                            │
    │      ──┼────────┼───────┼───────┼─────...───┼──►                        │
    │        │        │       │       │           │                            │
    │        │        │       │       ●           │  ← Decision at Week 3     │
    │        │        │       │    Decision       │                            │
    │        │        │       │                   │                            │
    │                                                                          │
    │  HAZARD FUNCTION:                                                        │
    │  ─────────────────                                                       │
    │  h(t) = P(decision at time t | no decision before t)                    │
    │                                                                          │
    │  This is the probability of getting a decision at time t,               │
    │  GIVEN that you haven't received one yet.                               │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    MODEL SPECIFICATION
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  DISCRETE-TIME LOGISTIC HAZARD MODEL                                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  The hazard at time t given features x:                                  │
    │                                                                          │
    │      h(t|x) = σ(α_t + xᵀβ)                                             │
    │                                                                          │
    │  WHERE:                                                                  │
    │  ──────                                                                  │
    │  • α_t = baseline hazard for time period t (time-specific intercepts)  │
    │  • β = covariate effects (same across all time periods)                 │
    │  • σ = sigmoid function                                                 │
    │                                                                          │
    │  KEY INSIGHT:                                                            │
    │  ────────────                                                            │
    │  This is just logistic regression on a specially structured dataset!    │
    │                                                                          │
    │  DATA TRANSFORMATION:                                                    │
    │  ─────────────────────                                                   │
    │  Original: One row per application with decision time t                 │
    │                                                                          │
    │      app_id │ avg  │ uni  │ prog │ decision_week                        │
    │      ───────┼──────┼──────┼──────┼─────────────                          │
    │      001    │ 92   │ UofT │ CS   │ 3                                    │
    │                                                                          │
    │  Expanded: One row per (application, time period) until decision        │
    │                                                                          │
    │      app_id │ week │ avg  │ uni  │ prog │ decision_this_week            │
    │      ───────┼──────┼──────┼──────┼──────┼───────────────────             │
    │      001    │  1   │ 92   │ UofT │ CS   │ 0                             │
    │      001    │  2   │ 92   │ UofT │ CS   │ 0                             │
    │      001    │  3   │ 92   │ UofT │ CS   │ 1  ← decision happened        │
    │                                                                          │
    │  Then fit standard logistic regression on expanded data!                │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    SURVIVAL AND PROBABILITY FUNCTIONS
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  FROM HAZARD TO SURVIVAL                                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  SURVIVAL FUNCTION:                                                      │
    │  ───────────────────                                                     │
    │  S(t|x) = P(no decision by time t) = Π_{s=1}^{t} [1 - h(s|x)]          │
    │                                                                          │
    │  INTERPRETATION: Probability of still waiting at time t                 │
    │                                                                          │
    │  EVENT PROBABILITY:                                                      │
    │  ───────────────────                                                     │
    │  f(t|x) = P(decision at exactly time t) = S(t-1|x) × h(t|x)            │
    │                                                                          │
    │  INTERPRETATION: Probability that week t is THE decision week           │
    │                                                                          │
    │  VISUALIZATION:                                                          │
    │  ───────────────                                                         │
    │                                                                          │
    │      Survival S(t)              Event Probability f(t)                   │
    │                                                                          │
    │      1.0 │●                          │                                   │
    │          │ ╲                         │    ●                              │
    │          │  ●                        │   ╱ ╲                             │
    │          │   ╲                       │  ●   ●                            │
    │          │    ●                      │ ╱     ╲                           │
    │          │     ╲                     │●       ●──●                       │
    │      0.0 │──────●───────            └────────────────►                  │
    │          └────────────► t              1  2  3  4  5  t                 │
    │                                                                          │
    │  Most decisions happen in the "peak" of f(t).                           │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    PREDICTION OUTPUTS
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  WHAT WE CAN PREDICT                                                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  1. SURVIVAL CURVE:                                                      │
    │     S(t|x) for each week t                                              │
    │     "Probability still waiting at week t"                               │
    │                                                                          │
    │  2. EXPECTED DECISION TIME:                                              │
    │     E[T|x] = Σ_t t × f(t|x)                                            │
    │     "Average week of decision"                                          │
    │                                                                          │
    │  3. MEDIAN DECISION TIME:                                                │
    │     t* such that S(t*|x) = 0.5                                          │
    │     "50% chance of decision by week t*"                                 │
    │                                                                          │
    │  4. DECISION WINDOW:                                                     │
    │     [t₁₀, t₉₀] such that P(t₁₀ ≤ T ≤ t₉₀) = 0.80                      │
    │     "80% of decisions happen between week t₁₀ and t₉₀"                 │
    │                                                                          │
    │  EXAMPLE OUTPUT:                                                         │
    │  ────────────────                                                        │
    │  "For UofT CS with 92% average:                                         │
    │   • Expected decision: Week 10 (early March)                            │
    │   • 50% chance by: Week 8                                               │
    │   • 80% window: Weeks 6-14 (Feb-Apr)"                                  │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

# Will import from logistic once implemented
from .logistic import LogisticModel, sigmoid


@dataclass
class TimingPrediction:
    """
    Prediction output for decision timing.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  TIMING PREDICTION OUTPUT                                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Contains all timing-related predictions for one application.           │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Attributes:
        survival_curve: S(t) for each time period
        hazard_curve: h(t) for each time period
        event_probs: f(t) = probability of decision at exactly time t
        expected_time: E[T] = mean decision time
        median_time: t where S(t) = 0.5
        ci_80_lower: 10th percentile of decision time
        ci_80_upper: 90th percentile of decision time
    """
    survival_curve: np.ndarray
    hazard_curve: np.ndarray
    event_probs: np.ndarray
    expected_time: float
    median_time: float
    ci_80_lower: float
    ci_80_upper: float

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"Expected decision time: {self.expected_time:.1f}\n"
            f"Median decision time: {self.median_time:.1f}\n"
            f"80% CI: [{self.ci_80_lower:.1f}, {self.ci_80_upper:.1f}]"
        )


def expand_to_person_period(
    X: np.ndarray,
    event_times: np.ndarray,
    max_time: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expand data to person-period format for discrete-time survival analysis.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  PERSON-PERIOD DATA TRANSFORMATION                                       │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  INPUT (one row per application):                                        │
    │  ──────────────────────────────────                                      │
    │      X: features for each application                                   │
    │      event_times: week of decision for each application                 │
    │                                                                          │
    │  OUTPUT (one row per application-week until decision):                  │
    │  ────────────────────────────────────────────────────────                │
    │      X_expanded: features repeated for each at-risk week               │
    │      y_expanded: 1 if decision this week, 0 otherwise                  │
    │      time_dummies: one-hot encoding of time period                     │
    │                                                                          │
    │  EXAMPLE:                                                                │
    │  ────────                                                                │
    │  Application with event_time=3:                                         │
    │                                                                          │
    │      Week │ At Risk │ Event │ y                                        │
    │      ─────┼─────────┼───────┼───                                        │
    │        1  │   Yes   │  No   │ 0                                         │
    │        2  │   Yes   │  No   │ 0                                         │
    │        3  │   Yes   │  Yes  │ 1  ← decision happened                   │
    │        4  │   No    │  -    │ (not in dataset)                         │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Original feature matrix (n_samples, n_features)
        event_times: Decision time for each sample (n_samples,)
        max_time: Maximum time period to consider

    Returns:
        Tuple of (X_expanded, y_expanded, time_dummies)

    Example:
        >>> X = np.array([[92, 1, 0], [88, 0, 1]])  # 2 applications
        >>> event_times = np.array([3, 2])  # decisions at week 3, 2
        >>> X_exp, y_exp, times = expand_to_person_period(X, event_times, max_time=5)
        >>> X_exp.shape  # (3 + 2 = 5 rows for the two applications)
        (5, 3)

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Calculate total rows needed: sum of event_times
    2. Initialize empty arrays
    3. For each sample i:
           For t in range(1, event_times[i] + 1):
               Append X[i] to X_expanded
               Append (t == event_times[i]) to y_expanded
               Append one-hot for t to time_dummies
    4. Return arrays
    """
    n_samples, n_features = X.shape
    total_rows = int(np.sum(event_times))

    X_expanded = np.zeros((total_rows, n_features))
    y_expanded = np.zeros(total_rows)
    time_indices = np.zeros(total_rows, dtype=int)

    row = 0
    for i in range(n_samples):
        for t in range(1, int(event_times[i]) + 1):
            X_expanded[row] = X[i]
            y_expanded[row] = 1.0 if t == event_times[i] else 0.0
            time_indices[row] = t
            row += 1

    return X_expanded, y_expanded, time_indices


def create_time_dummies(
    time_indices: np.ndarray,
    max_time: int
) -> np.ndarray:
    """
    Create one-hot encoding for time periods.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  TIME DUMMY VARIABLES                                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  For time t ∈ {1, 2, ..., T}:                                           │
    │                                                                          │
    │      time_dummies = [I(t=1), I(t=2), ..., I(t=T)]                       │
    │                                                                          │
    │  This captures the baseline hazard α_t for each time period.            │
    │                                                                          │
    │  Example for t=3 with T=5:                                              │
    │      [0, 0, 1, 0, 0]                                                    │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        time_indices: Time period for each row (1-indexed)
        max_time: Total number of time periods

    Returns:
        One-hot encoded time dummies (n_rows, max_time)

    IMPLEMENTATION:
    ────────────────
    dummies = np.zeros((len(time_indices), max_time))
    for i, t in enumerate(time_indices):
        dummies[i, t - 1] = 1  # 0-indexed
    return dummies
    """
    dummies = np.zeros((len(time_indices), max_time))
    for i, t in enumerate(time_indices):
        dummies[i, t - 1] = 1.0
    return dummies


class HazardModel:
    """
    Discrete-time hazard model for decision timing prediction.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  HAZARD MODEL FOR TIMING                                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  MODEL:                                                                  │
    │      h(t|x) = sigmoid(α_t + xᵀβ)                                       │
    │                                                                          │
    │  WHERE:                                                                  │
    │  • α_t = time-specific baseline hazard (one per week)                  │
    │  • β = covariate effects on hazard                                      │
    │                                                                          │
    │  TRAINING:                                                               │
    │  • Expand data to person-period format                                  │
    │  • Fit logistic regression on expanded data                             │
    │                                                                          │
    │  PREDICTION:                                                             │
    │  • Compute hazard for each time period                                  │
    │  • Convert to survival curve and event probabilities                    │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Example:
        >>> model = HazardModel(max_time=20, lambda_=0.1)
        >>> model.fit(X_train, decision_weeks_train)
        >>> timing = model.predict_timing(X_new)
        >>> print(f"Expected decision: Week {timing.expected_time:.1f}")
    """

    def __init__(
        self,
        max_time: int = 20,
        lambda_: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-6
    ):
        """
        Initialize hazard model.

        Args:
            max_time: Maximum time periods to consider (e.g., 20 weeks)
            lambda_: Ridge regularization strength
            max_iter: Maximum IRLS iterations
            tol: Convergence tolerance

        IMPLEMENTATION:
        ────────────────
        Store parameters.
        self.baseline_hazards = None  # α_t coefficients
        self.covariate_effects = None  # β coefficients
        self._is_fitted = False
        """
        self.max_time = max_time
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        self.baseline_hazards = None
        self.covariate_effects = None
        self._is_fitted = False

    @property
    def name(self) -> str:
        """Model name."""
        return "Discrete-Time Hazard Model"

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted

    def fit(
        self,
        X: np.ndarray,
        event_times: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> 'HazardModel':
        """
        Fit the discrete-time hazard model.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  FITTING PROCEDURE                                                   │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  1. Expand to person-period format                                  │
        │  2. Create time dummies for baseline hazard                         │
        │  3. Combine: X_full = [time_dummies, X_expanded]                    │
        │  4. Fit logistic regression on (X_full, y_expanded)                │
        │  5. Extract baseline hazards and covariate effects                  │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            X: Feature matrix (n_applications, n_features)
            event_times: Decision week for each application (n_applications,)
            sample_weight: Optional weights

        Returns:
            self

        IMPLEMENTATION STEPS:
        ─────────────────────
        1. X_exp, y_exp, times = expand_to_person_period(X, event_times, self.max_time)
        2. time_dummies = create_time_dummies(times, self.max_time)
        3. X_full = np.hstack([time_dummies, X_exp])
        4. Fit LogisticModel on (X_full, y_exp) WITHOUT intercept
           (time dummies act as time-specific intercepts)
        5. self.baseline_hazards = coefficients[:max_time]
        6. self.covariate_effects = coefficients[max_time:]
        7. self._is_fitted = True
        8. Return self
        """
        X_exp, y_exp, times = expand_to_person_period(X, event_times, self.max_time)
        time_dummies = create_time_dummies(times, self.max_time)
        X_full = np.hstack([time_dummies, X_exp])

        logistic = LogisticModel(
            lambda_=self.lambda_,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=False
        )
        logistic.fit(X_full, y_exp)

        coefficients = logistic.coefficients
        self.baseline_hazards = coefficients[:self.max_time]
        self.covariate_effects = coefficients[self.max_time:]
        self._is_fitted = True
        return self

    def predict_hazard(
        self,
        X: np.ndarray,
        times: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict hazard h(t|x) for each sample at each time.

        Args:
            X: Feature matrix (n_samples, n_features)
            times: Time periods to predict for (default: 1 to max_time)

        Returns:
            Hazard values (n_samples, n_times)

        IMPLEMENTATION:
        ────────────────
        if times is None:
            times = np.arange(1, self.max_time + 1)

        linear = self.baseline_hazards[times - 1] + X @ self.covariate_effects
        return sigmoid(linear)
        """
        if not self._is_fitted:
            return None

        if times is None:
            times = np.arange(1, self.max_time + 1)

        # baseline_hazards[times - 1] has shape (n_times,)
        # X @ covariate_effects has shape (n_samples,)
        # We need shape (n_samples, n_times)
        linear = self.baseline_hazards[times - 1] + (X @ self.covariate_effects)[:, np.newaxis]
        return sigmoid(linear)

    def predict_survival(self, X: np.ndarray) -> np.ndarray:
        """
        Predict survival curve S(t|x) for each sample.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  SURVIVAL FUNCTION                                                   │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │      S(t|x) = Π_{s=1}^{t} [1 - h(s|x)]                              │
        │                                                                      │
        │  Computed as cumulative product of (1 - hazard).                    │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Survival curves (n_samples, max_time)

        IMPLEMENTATION:
        ────────────────
        hazard = self.predict_hazard(X)  # (n, T)
        survival = np.cumprod(1 - hazard, axis=1)
        return survival
        """
        if not self._is_fitted:
            return None
        hazard = self.predict_hazard(X)
        survival = np.cumprod(1 - hazard, axis=1)
        return survival

    def predict_event_prob(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of decision at each time.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  EVENT PROBABILITY                                                   │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │      f(t|x) = P(decision at t) = S(t-1|x) × h(t|x)                 │
        │                                                                      │
        │  Where S(0|x) = 1 (everyone starts without decision).              │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            X: Feature matrix

        Returns:
            Event probabilities (n_samples, max_time)

        IMPLEMENTATION:
        ────────────────
        hazard = self.predict_hazard(X)  # (n, T)
        survival = self.predict_survival(X)  # (n, T)

        # f(t) = S(t-1) * h(t)
        # For t=1: S(0) = 1
        S_lagged = np.hstack([np.ones((X.shape[0], 1)), survival[:, :-1]])
        event_prob = S_lagged * hazard
        return event_prob
        """
        if not self._is_fitted:
            return None
        hazard = self.predict_hazard(X)
        survival = self.predict_survival(X)

        S_lagged = np.hstack([np.ones((X.shape[0], 1)), survival[:, :-1]])
        event_prob = S_lagged * hazard
        return event_prob

    def predict_timing(self, X: np.ndarray) -> List[TimingPrediction]:
        """
        Generate full timing predictions for each sample.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  COMPLETE TIMING PREDICTION                                          │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  For each application, compute:                                     │
        │  • Survival curve S(t)                                              │
        │  • Hazard curve h(t)                                                │
        │  • Event probabilities f(t)                                         │
        │  • Expected time E[T]                                               │
        │  • Median time                                                      │
        │  • 80% confidence interval                                          │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            List of TimingPrediction objects, one per sample

        IMPLEMENTATION:
        ────────────────
        hazard = self.predict_hazard(X)
        survival = self.predict_survival(X)
        event_prob = self.predict_event_prob(X)

        predictions = []
        times = np.arange(1, self.max_time + 1)

        for i in range(X.shape[0]):
            # Expected time
            expected = np.sum(times * event_prob[i])

            # Median: first t where S(t) <= 0.5
            median_idx = np.searchsorted(-survival[i], -0.5)
            median = times[min(median_idx, len(times) - 1)]

            # 80% CI from event probability distribution
            cumulative = np.cumsum(event_prob[i])
            lower_idx = np.searchsorted(cumulative, 0.10)
            upper_idx = np.searchsorted(cumulative, 0.90)
            lower = times[lower_idx]
            upper = times[min(upper_idx, len(times) - 1)]

            predictions.append(TimingPrediction(
                survival_curve=survival[i],
                hazard_curve=hazard[i],
                event_probs=event_prob[i],
                expected_time=expected,
                median_time=median,
                ci_80_lower=lower,
                ci_80_upper=upper
            ))

        return predictions
        """
        if not self._is_fitted:
            return None
        hazard = self.predict_hazard(X)
        survival = self.predict_survival(X)
        event_prob = self.predict_event_prob(X)

        predictions = []
        times = np.arange(1, self.max_time + 1)

        for i in range(X.shape[0]):
            # Expected time
            expected = np.sum(times * event_prob[i])

            # Median: first t where S(t) <= 0.5
            median_idx = np.searchsorted(-survival[i], -0.5)
            median = float(times[min(median_idx, len(times) - 1)])

            # 80% CI from event probability distribution
            cumulative = np.cumsum(event_prob[i])
            lower_idx = np.searchsorted(cumulative, 0.10)
            upper_idx = np.searchsorted(cumulative, 0.90)
            lower = float(times[min(lower_idx, len(times) - 1)])
            upper = float(times[min(upper_idx, len(times) - 1)])

            predictions.append(TimingPrediction(
                survival_curve=survival[i],
                hazard_curve=hazard[i],
                event_probs=event_prob[i],
                expected_time=expected,
                median_time=median,
                ci_80_lower=lower,
                ci_80_upper=upper
            ))

        return predictions

    def predict_expected_time(self, X: np.ndarray) -> np.ndarray:
        """
        Predict expected decision time for each sample.

        Args:
            X: Feature matrix

        Returns:
            Expected times (n_samples,)

        IMPLEMENTATION:
        ────────────────
        event_prob = self.predict_event_prob(X)
        times = np.arange(1, self.max_time + 1)
        return event_prob @ times
        """
        if not self._is_fitted:
            return None
        event_prob = self.predict_event_prob(X)
        times = np.arange(1, self.max_time + 1)
        return event_prob @ times

    def predict_median_time(self, X: np.ndarray) -> np.ndarray:
        """
        Predict median decision time for each sample.

        Args:
            X: Feature matrix

        Returns:
            Median times (n_samples,)
        """
        if not self._is_fitted:
            return None
        survival = self.predict_survival(X)
        times = np.arange(1, self.max_time + 1)
        medians = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            # First t where survival <= 0.5
            idx = np.searchsorted(-survival[i], -0.5)
            medians[i] = times[min(idx, len(times) - 1)]
        return medians

    def get_baseline_hazard_curve(self) -> np.ndarray:
        """
        Get the estimated baseline hazard for each time period.

        Returns:
            Baseline hazards α_t converted to probabilities
            via sigmoid(α_t) for the "average" application.
        """
        if not self._is_fitted:
            return None
        return sigmoid(self.baseline_hazards)

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters for serialization."""
        return {
            'baseline_hazards': self.baseline_hazards.tolist() if self.baseline_hazards is not None else None,
            'covariate_effects': self.covariate_effects.tolist() if self.covariate_effects is not None else None,
            'max_time': self.max_time,
            'lambda_': self.lambda_,
            'max_iter': self.max_iter,
            'tol': self.tol,
            '_is_fitted': self._is_fitted,
        }

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters from serialization."""
        if 'baseline_hazards' in params and params['baseline_hazards'] is not None:
            self.baseline_hazards = np.array(params['baseline_hazards'])
        if 'covariate_effects' in params and params['covariate_effects'] is not None:
            self.covariate_effects = np.array(params['covariate_effects'])
        if 'max_time' in params:
            self.max_time = params['max_time']
        if 'lambda_' in params:
            self.lambda_ = params['lambda_']
        if 'max_iter' in params:
            self.max_iter = params['max_iter']
        if 'tol' in params:
            self.tol = params['tol']
        if '_is_fitted' in params:
            self._is_fitted = params['_is_fitted']


# =============================================================================
#                           TODO LIST
# =============================================================================
"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TODO                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  HIGH PRIORITY (core model):                                                 │
│  ────────────────────────────                                                │
│  [ ] Implement expand_to_person_period() - data transformation             │
│  [ ] Implement create_time_dummies()                                        │
│  [ ] Implement HazardModel.__init__()                                       │
│  [ ] Implement HazardModel.fit()                                            │
│  [ ] Implement HazardModel.predict_hazard()                                 │
│  [ ] Implement HazardModel.predict_survival()                               │
│                                                                              │
│  MEDIUM PRIORITY (predictions):                                              │
│  ──────────────────────────────                                              │
│  [ ] Implement predict_event_prob()                                         │
│  [ ] Implement predict_timing() with full TimingPrediction                  │
│  [ ] Implement predict_expected_time()                                      │
│  [ ] Implement predict_median_time()                                        │
│                                                                              │
│  LOW PRIORITY (analysis):                                                    │
│  ─────────────────────────                                                   │
│  [ ] Implement get_baseline_hazard_curve()                                  │
│  [ ] Add covariate effect interpretation                                    │
│  [ ] Implement TimingPrediction.summary()                                   │
│                                                                              │
│  TESTING:                                                                    │
│  ────────                                                                    │
│  [ ] Verify survival curves are monotonically decreasing                    │
│  [ ] Check event probabilities sum to ~1                                    │
│  [ ] Test on synthetic data with known timing patterns                      │
│  [ ] Verify expected time is reasonable                                     │
│                                                                              │
│  EXTENSIONS:                                                                 │
│  ───────────                                                                 │
│  [ ] Add time-varying covariates                                            │
│  [ ] Add frailty/random effects for programs                                │
│  [ ] Compare to Kaplan-Meier estimates                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    print("Discrete-Time Hazard Model")
    print("=" * 50)
    print()
    print("Predicts WHEN decisions happen, not just IF admitted.")
    print()
    print("Key outputs:")
    print("  • Survival curve S(t): probability still waiting")
    print("  • Expected time E[T]: average decision week")
    print("  • Median time: 50% chance by this week")
    print("  • 80% window: most decisions in this range")
