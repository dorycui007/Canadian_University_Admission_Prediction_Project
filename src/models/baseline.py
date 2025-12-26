"""
Beta-Binomial Baseline Model
=============================

This module implements a Bayesian baseline model using Beta-Binomial conjugate
priors. This provides sensible predictions even for programs with very few
observations by "borrowing strength" from prior beliefs.

STATISTICAL REFERENCE: Beta-Binomial Conjugate Analysis, Bayesian Inference

================================================================================
                        SYSTEM ARCHITECTURE CONTEXT
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    WHERE THIS MODULE FITS                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │        Raw CSV ──► MongoDB ──► Feature Extraction                        │
    │                                       │                                  │
    │                                       ▼                                  │
    │                         ┌───────────────────────────┐                    │
    │                         │      [THIS MODULE]        │                    │
    │                         │     Beta-Binomial         │                    │
    │                         │      baseline.py          │                    │
    │                         └─────────────┬─────────────┘                    │
    │                                       │                                  │
    │                    PURPOSE: Quick, sensible baseline predictions        │
    │                    that handle sparse data gracefully                    │
    │                                       │                                  │
    │                                       ▼                                  │
    │                      ┌───────────────────────────┐                       │
    │                      │   Compare to advanced     │                       │
    │                      │   models (Logistic,       │                       │
    │                      │   Embeddings, etc.)       │                       │
    │                      └───────────────────────────┘                       │
    │                                                                          │
    │  ROLE IN PROJECT:                                                        │
    │  ─────────────────                                                       │
    │  • Establish baseline performance (Day 4)                               │
    │  • Handle programs with < 10 applications                               │
    │  • Provide uncertainty quantification (credible intervals)              │
    │  • Benchmark for more complex models to beat                            │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    THE BETA-BINOMIAL MODEL
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  BAYESIAN APPROACH TO ADMISSION RATES                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  PROBLEM:                                                                │
    │  ────────                                                                │
    │  For each (university, program) pair, estimate P(admit).                │
    │                                                                          │
    │  Simple approach: P̂ = admits / total                                    │
    │                                                                          │
    │  BUT WHAT IF:                                                            │
    │  • Queens Computing has only 2 applications (1 admit, 1 reject)?        │
    │  • MLE says P(admit) = 0.50, but is that reliable?                      │
    │                                                                          │
    │  BAYESIAN SOLUTION:                                                      │
    │  ───────────────────                                                     │
    │  Start with a PRIOR belief, update with data.                           │
    │                                                                          │
    │  Prior:     P(admit) ~ Beta(α₀, β₀)                                    │
    │  Data:      k admissions out of n applications                          │
    │  Posterior: P(admit) ~ Beta(α₀ + k, β₀ + n - k)                        │
    │                                                                          │
    │  POSTERIOR MEAN:                                                         │
    │                                                                          │
    │               α₀ + k          α₀           n           k                │
    │      E[P] = ──────────── = ────────── × ─────── + ─────── × ──────      │
    │             α₀ + β₀ + n    α₀ + β₀ + n    n      α₀+β₀+n     n          │
    │                              ↑                      ↑                    │
    │                           prior                  data                   │
    │                           weight                 weight                 │
    │                                                                          │
    │  With more data, the data weight increases (sensible!)                  │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    PRIOR SELECTION
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  CHOOSING THE PRIOR Beta(α₀, β₀)                                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  The prior parameters encode our beliefs BEFORE seeing program data.    │
    │                                                                          │
    │  INTERPRETATION:                                                         │
    │  ────────────────                                                        │
    │  α₀ = "pseudo-counts" of admits                                         │
    │  β₀ = "pseudo-counts" of rejects                                        │
    │  α₀ + β₀ = "strength" of prior (equivalent sample size)                │
    │                                                                          │
    │  COMMON CHOICES:                                                         │
    │  ────────────────                                                        │
    │                                                                          │
    │  Beta(1, 1):  Uniform (no prior preference)                             │
    │      ┌────────────────────────────────────┐                             │
    │      │  ████████████████████████████████  │ p(θ)                        │
    │      └────────────────────────────────────┘                             │
    │       0                                   1   θ                          │
    │                                                                          │
    │  Beta(2, 2):  Slight preference for middle values                       │
    │      ┌────────────────────────────────────┐                             │
    │      │          ████████████              │ p(θ)                        │
    │      │        ██            ██            │                              │
    │      │      ██                ██          │                              │
    │      └────────────────────────────────────┘                             │
    │       0                                   1   θ                          │
    │                                                                          │
    │  Beta(5, 10): Prior belief ~33% admission rate (informative)            │
    │      ┌────────────────────────────────────┐                             │
    │      │      ████████                      │ p(θ)                        │
    │      │    ██        ██                    │                              │
    │      │  ██            ██                  │                              │
    │      └────────────────────────────────────┘                             │
    │       0              ↑                    1   θ                          │
    │                     0.33                                                 │
    │                                                                          │
    │  FOR THIS PROJECT:                                                       │
    │  ──────────────────                                                      │
    │  • Compute overall admission rate across all programs                   │
    │  • Set prior mean = overall rate                                        │
    │  • Set prior strength based on desired shrinkage                        │
    │    (e.g., α₀ + β₀ = 10 means prior worth 10 observations)             │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    HIERARCHICAL STRUCTURE
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  BORROWING STRENGTH ACROSS LEVELS                                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  HIERARCHY:                                                              │
    │  ──────────                                                              │
    │                                                                          │
    │      Level 0 (Global):    Overall admission rate                        │
    │           │                                                              │
    │           ├── Level 1 (University):                                      │
    │           │       │                                                      │
    │           │       ├── UofT rate                                         │
    │           │       ├── Waterloo rate                                     │
    │           │       ├── McMaster rate                                     │
    │           │       └── ...                                                │
    │           │                                                              │
    │           └── Level 2 (Program within University):                      │
    │                   │                                                      │
    │                   ├── UofT CS rate                                      │
    │                   ├── UofT Engineering rate                              │
    │                   ├── Waterloo CS rate                                  │
    │                   └── ...                                                │
    │                                                                          │
    │  SHRINKAGE FLOW:                                                         │
    │  ────────────────                                                        │
    │  Programs with little data → shrink toward university rate             │
    │  Universities with little data → shrink toward global rate             │
    │                                                                          │
    │  EXAMPLE:                                                                │
    │  ────────                                                                │
    │  Queens Computing (2 apps): shrinks heavily toward Queens average       │
    │  UofT CS (500 apps): estimates dominated by actual data                 │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    CREDIBLE INTERVALS
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  UNCERTAINTY QUANTIFICATION                                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Unlike point estimates, Beta posteriors give DISTRIBUTIONS.            │
    │                                                                          │
    │  80% CREDIBLE INTERVAL:                                                  │
    │  ───────────────────────                                                 │
    │  [q₁₀, q₉₀] such that P(q₁₀ ≤ θ ≤ q₉₀) = 0.80                         │
    │                                                                          │
    │  VISUALIZATION:                                                          │
    │  ───────────────                                                         │
    │                                                                          │
    │  Many observations (tight):          Few observations (wide):           │
    │                                                                          │
    │      p(θ)│      ████                     p(θ)│  ████████████████        │
    │          │     ██  ██                        │██                ██      │
    │          │    ██    ██                       │                    ██    │
    │          │  ██        ██                     │                      ██  │
    │          └──┼─────┼──────► θ                 └──┼───────────────┼───► θ │
    │            q10   q90                           q10              q90     │
    │            narrow CI                           wide CI                   │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  • Display to users: "65% chance [52%, 78%]"                            │
    │  • Flag uncertain predictions (wide CI → need more data)                │
    │  • Decision-making under uncertainty                                     │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field

# Import base model (once implemented)
# from .base import BaseModel, ModelConfig, EvaluationMetrics


@dataclass
class BetaPrior:
    """
    Beta distribution prior parameters.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  BETA PRIOR CONFIGURATION                                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Beta(α, β) has:                                                        │
    │  • Mean = α / (α + β)                                                   │
    │  • Variance = αβ / [(α+β)²(α+β+1)]                                     │
    │  • Mode = (α-1) / (α+β-2) for α,β > 1                                  │
    │                                                                          │
    │  Effective sample size = α + β (prior "strength")                       │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Attributes:
        alpha: Shape parameter α (pseudo-count of successes)
        beta: Shape parameter β (pseudo-count of failures)
    """
    alpha: float = 1.0
    beta: float = 1.0

    @property
    def mean(self) -> float:
        """Prior mean = α / (α + β)."""
        pass

    @property
    def variance(self) -> float:
        """Prior variance."""
        pass

    @property
    def strength(self) -> float:
        """Effective sample size = α + β."""
        pass

    @classmethod
    def from_mean_strength(cls, mean: float, strength: float) -> 'BetaPrior':
        """
        Create prior from desired mean and strength.

        Args:
            mean: Desired prior mean (between 0 and 1)
            strength: Effective sample size (α + β)

        Returns:
            BetaPrior with specified mean and strength

        Example:
            >>> prior = BetaPrior.from_mean_strength(mean=0.4, strength=10)
            >>> prior.alpha, prior.beta
            (4.0, 6.0)

        IMPLEMENTATION:
        ────────────────
        alpha = mean * strength
        beta = (1 - mean) * strength
        return cls(alpha=alpha, beta=beta)
        """
        pass


@dataclass
class ProgramPosterior:
    """
    Beta posterior for a single program.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  POSTERIOR PARAMETERS                                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Prior:     Beta(α₀, β₀)                                               │
    │  Data:      k admits, n-k rejects                                       │
    │  Posterior: Beta(α₀ + k, β₀ + n - k)                                   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Attributes:
        program_id: Unique identifier for program
        university: University name
        program_name: Program name
        alpha_post: Posterior α = prior_α + admits
        beta_post: Posterior β = prior_β + rejects
        n_observations: Total applications observed
        n_admits: Number of admissions
    """
    program_id: str
    university: str
    program_name: str
    alpha_post: float
    beta_post: float
    n_observations: int
    n_admits: int

    @property
    def posterior_mean(self) -> float:
        """
        Posterior mean admission probability.

        E[P|data] = α_post / (α_post + β_post)
        """
        pass

    @property
    def posterior_mode(self) -> float:
        """
        Posterior mode (MAP estimate).

        Mode = (α_post - 1) / (α_post + β_post - 2)
        Only valid if α_post, β_post > 1.
        """
        pass

    @property
    def posterior_variance(self) -> float:
        """Posterior variance of admission probability."""
        pass

    def credible_interval(self, level: float = 0.80) -> Tuple[float, float]:
        """
        Compute credible interval for admission probability.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  CREDIBLE INTERVAL COMPUTATION                                       │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  For level = 0.80:                                                  │
        │  • Lower = 10th percentile of Beta posterior                        │
        │  • Upper = 90th percentile of Beta posterior                        │
        │                                                                      │
        │  Uses: scipy.stats.beta.ppf()                                       │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            level: Credibility level (default: 0.80 for 80% CI)

        Returns:
            Tuple of (lower_bound, upper_bound)

        Example:
            >>> posterior = ProgramPosterior(...)
            >>> lower, upper = posterior.credible_interval(0.80)
            >>> print(f"80% CI: [{lower:.2%}, {upper:.2%}]")

        IMPLEMENTATION:
        ────────────────
        from scipy.stats import beta
        alpha_tail = (1 - level) / 2
        lower = beta.ppf(alpha_tail, self.alpha_post, self.beta_post)
        upper = beta.ppf(1 - alpha_tail, self.alpha_post, self.beta_post)
        return (lower, upper)
        """
        pass


class BaselineModel:
    """
    Beta-Binomial baseline model for admission prediction.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  BETA-BINOMIAL BASELINE                                                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  APPROACH:                                                               │
    │  ─────────                                                               │
    │  1. Compute global admission rate as prior mean                         │
    │  2. For each program, compute Beta posterior                            │
    │  3. Predict using posterior mean                                        │
    │                                                                          │
    │  FEATURES USED:                                                          │
    │  ───────────────                                                         │
    │  • University (categorical)                                             │
    │  • Program (categorical)                                                │
    │  • Average (continuous) - optionally binned                            │
    │                                                                          │
    │  Does NOT use complex feature interactions - that's for LogisticModel. │
    │                                                                          │
    │  USE CASES:                                                              │
    │  ──────────                                                              │
    │  • Programs with very few observations (< 10)                          │
    │  • Quick baseline to beat                                               │
    │  • Uncertainty quantification via credible intervals                    │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Example:
        >>> model = BaselineModel(prior_strength=10)
        >>> model.fit(X, y, program_ids)
        >>> probs = model.predict_proba(X_new, program_ids_new)
        >>> lower, upper = model.predict_interval(X_new, program_ids_new)
    """

    def __init__(
        self,
        prior_strength: float = 10.0,
        use_hierarchical: bool = True
    ):
        """
        Initialize Beta-Binomial baseline model.

        Args:
            prior_strength: Effective sample size of prior (α₀ + β₀).
                           Higher = more shrinkage toward global mean.
            use_hierarchical: If True, use university-level priors.
                             If False, use global prior only.

        IMPLEMENTATION:
        ────────────────
        1. Store prior_strength
        2. Store use_hierarchical
        3. Initialize empty dict for posteriors: self.posteriors = {}
        4. Set self._is_fitted = False
        5. self.global_prior = None
        """
        pass

    @property
    def name(self) -> str:
        """Model name for display."""
        pass

    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        pass

    @property
    def n_params(self) -> int:
        """
        Number of parameters = 2 per program (α, β posteriors).
        """
        pass

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        program_ids: np.ndarray,
        university_ids: Optional[np.ndarray] = None
    ) -> 'BaselineModel':
        """
        Fit Beta posteriors for each program.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  FITTING PROCEDURE                                                   │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  1. COMPUTE GLOBAL PRIOR:                                           │
        │     global_rate = sum(y) / len(y)                                   │
        │     global_prior = Beta.from_mean_strength(global_rate, strength)   │
        │                                                                      │
        │  2. (If hierarchical) COMPUTE UNIVERSITY PRIORS:                    │
        │     For each university u:                                          │
        │         uni_rate = mean(y[university == u])                        │
        │         uni_prior[u] = shrink toward global                        │
        │                                                                      │
        │  3. COMPUTE PROGRAM POSTERIORS:                                      │
        │     For each program p:                                             │
        │         k = sum(y[program == p])  # admits                          │
        │         n = sum(program == p)     # total                           │
        │         prior = uni_prior[uni_of_p] or global_prior                │
        │         posterior[p] = Beta(prior.α + k, prior.β + n - k)          │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            X: Feature matrix (may be ignored, uses only categoricals)
            y: Binary outcomes (0 = reject, 1 = admit)
            program_ids: Program identifier for each sample
            university_ids: University identifier for each sample (optional)

        Returns:
            self

        IMPLEMENTATION STEPS:
        ─────────────────────
        1. Compute global admission rate
        2. Create global prior with self.prior_strength
        3. For each unique program:
               a. Get indices where program_ids == program
               b. Count admits k = sum(y[indices])
               c. Count total n = len(indices)
               d. Create ProgramPosterior with updated α, β
               e. Store in self.posteriors[program]
        4. Set self._is_fitted = True
        5. Return self
        """
        pass

    def predict_proba(
        self,
        X: np.ndarray,
        program_ids: np.ndarray
    ) -> np.ndarray:
        """
        Predict admission probabilities using posterior means.

        Args:
            X: Feature matrix (may be ignored)
            program_ids: Program identifier for each sample

        Returns:
            Probability predictions of shape (n_samples,)

        Raises:
            RuntimeError: If model is not fitted
            KeyError: If program_id not seen during training

        IMPLEMENTATION:
        ────────────────
        probs = np.zeros(len(program_ids))
        for i, prog_id in enumerate(program_ids):
            if prog_id in self.posteriors:
                probs[i] = self.posteriors[prog_id].posterior_mean
            else:
                # Unseen program: use global prior mean
                probs[i] = self.global_prior.mean
        return probs
        """
        pass

    def predict_interval(
        self,
        X: np.ndarray,
        program_ids: np.ndarray,
        level: float = 0.80
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict credible intervals for admission probabilities.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  UNCERTAINTY QUANTIFICATION                                          │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  Returns lower and upper bounds of credible interval.               │
        │                                                                      │
        │  Example output for 3 predictions:                                  │
        │  lower: [0.45, 0.62, 0.31]                                         │
        │  upper: [0.72, 0.85, 0.58]                                         │
        │                                                                      │
        │  Display: "65% [45%, 72%]"                                          │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            X: Feature matrix
            program_ids: Program identifier for each sample
            level: Credibility level (default: 0.80)

        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays

        IMPLEMENTATION:
        ────────────────
        lower = np.zeros(len(program_ids))
        upper = np.zeros(len(program_ids))
        for i, prog_id in enumerate(program_ids):
            posterior = self.posteriors.get(prog_id, self._global_posterior())
            lower[i], upper[i] = posterior.credible_interval(level)
        return lower, upper
        """
        pass

    def get_program_summary(self, program_id: str) -> Dict[str, Any]:
        """
        Get detailed summary for a specific program.

        Returns:
            Dictionary with:
            - 'program_id': str
            - 'university': str
            - 'program_name': str
            - 'n_observations': int
            - 'n_admits': int
            - 'observed_rate': float (raw k/n)
            - 'posterior_mean': float (shrunk estimate)
            - 'ci_80_lower': float
            - 'ci_80_upper': float
            - 'shrinkage': float (how much estimate was shrunk)

        Example:
            >>> summary = model.get_program_summary("uoft_cs")
            >>> print(f"{summary['program_name']}: {summary['posterior_mean']:.1%}")
            >>> print(f"  Observed: {summary['observed_rate']:.1%}")
            >>> print(f"  Shrinkage: {summary['shrinkage']:.1%}")
        """
        pass

    def rank_programs_by_difficulty(self) -> List[Tuple[str, float, float]]:
        """
        Rank all programs by admission difficulty.

        Returns:
            List of (program_id, posterior_mean, n_observations)
            sorted by posterior_mean ascending (hardest first)

        Example:
            >>> rankings = model.rank_programs_by_difficulty()
            >>> for rank, (prog, rate, n) in enumerate(rankings[:5], 1):
            ...     print(f"{rank}. {prog}: {rate:.1%} ({n} apps)")

        IMPLEMENTATION:
        ────────────────
        programs = [(pid, post.posterior_mean, post.n_observations)
                    for pid, post in self.posteriors.items()]
        return sorted(programs, key=lambda x: x[1])
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters for serialization."""
        pass

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters from serialization."""
        pass


def compute_shrinkage_factor(
    n_observations: int,
    prior_strength: float
) -> float:
    """
    Compute how much the posterior shrinks toward the prior.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  SHRINKAGE FACTOR                                                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Posterior mean = shrink × prior_mean + (1-shrink) × data_mean          │
    │                                                                          │
    │  shrinkage = prior_strength / (prior_strength + n_observations)          │
    │                                                                          │
    │  Examples:                                                               │
    │  ─────────                                                               │
    │  • n=0:   shrinkage = 1.0 (100% prior, no data)                        │
    │  • n=10:  shrinkage = 10/(10+10) = 0.5 (50% each)                      │
    │  • n=100: shrinkage = 10/(10+100) = 0.09 (9% prior)                    │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        n_observations: Number of observations for this group
        prior_strength: Effective sample size of prior

    Returns:
        Shrinkage factor in [0, 1]
    """
    pass


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
│  [ ] Implement BetaPrior dataclass with mean, variance, strength            │
│  [ ] Implement BetaPrior.from_mean_strength() class method                  │
│  [ ] Implement ProgramPosterior dataclass                                   │
│  [ ] Implement ProgramPosterior.credible_interval()                         │
│  [ ] Implement BaselineModel.__init__()                                     │
│  [ ] Implement BaselineModel.fit()                                          │
│  [ ] Implement BaselineModel.predict_proba()                                │
│                                                                              │
│  MEDIUM PRIORITY (features):                                                 │
│  ────────────────────────────                                                │
│  [ ] Implement predict_interval() for uncertainty                           │
│  [ ] Implement hierarchical priors (university level)                       │
│  [ ] Implement get_program_summary()                                        │
│  [ ] Implement rank_programs_by_difficulty()                                │
│                                                                              │
│  LOW PRIORITY (enhancements):                                                │
│  ─────────────────────────────                                               │
│  [ ] Add average-based binning (e.g., rates by average bucket)             │
│  [ ] Add trend adjustment (rate changing over years)                        │
│  [ ] Implement model serialization                                          │
│                                                                              │
│  TESTING:                                                                    │
│  ────────                                                                    │
│  [ ] Verify posterior mean shrinks toward prior for small n                 │
│  [ ] Verify posterior mean approaches data mean for large n                 │
│  [ ] Check credible intervals have correct coverage                          │
│  [ ] Compare predictions to simple k/n estimates                            │
│                                                                              │
│  EXTENSIONS (optional):                                                      │
│  ───────────────────────                                                     │
│  [ ] Empirical Bayes: estimate prior strength from data                     │
│  [ ] Full hierarchical model with MCMC/variational inference                │
│  [ ] Time-varying rates with exponential smoothing                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    print("Beta-Binomial Baseline Model")
    print("=" * 50)
    print()
    print("Bayesian approach to handling sparse program data.")
    print()
    print("Key concepts:")
    print("  • Beta prior encodes belief before seeing data")
    print("  • Posterior combines prior with observations")
    print("  • Posterior mean shrinks toward prior for small n")
    print("  • Credible intervals quantify uncertainty")
    print()
    print("After implementing, try:")
    print("  >>> model = BaselineModel(prior_strength=10)")
    print("  >>> model.fit(X, y, program_ids)")
    print("  >>> probs = model.predict_proba(X_test, program_ids_test)")
