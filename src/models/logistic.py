"""
IRLS Logistic Regression Module
================================

This module implements logistic regression from scratch using Iteratively
Reweighted Least Squares (IRLS). This is THE core algorithm for binary
classification in this project, connecting MAT223 linear algebra directly
to machine learning.

MAT223 REFERENCE: Section 4.8 (Approximating Solutions), applied to logistic loss
CSC148 REFERENCE: OOP design patterns, iterator protocol

================================================================================
                        SYSTEM ARCHITECTURE CONTEXT
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    WHERE THIS MODULE FITS                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  ┌───────────────────────────────────────────────────────────────────┐  │
    │  │                   MATH FOUNDATION                                  │  │
    │  │                                                                    │  │
    │  │   vectors.py ─► projections.py ─► qr.py ─► ridge.py              │  │
    │  │                                              │                     │  │
    │  │                                              ▼                     │  │
    │  │                                    ┌─────────────────┐            │  │
    │  │                                    │  [THIS MODULE]  │            │  │
    │  │                                    │   logistic.py   │            │  │
    │  │                                    │      IRLS       │            │  │
    │  │                                    └────────┬────────┘            │  │
    │  │                                             │                      │  │
    │  └─────────────────────────────────────────────┼──────────────────────┘  │
    │                                                │                         │
    │                              ┌─────────────────┼─────────────────┐       │
    │                              ▼                 ▼                 ▼       │
    │                       ┌───────────┐     ┌───────────┐     ┌───────────┐ │
    │                       │ hazard.py │     │embeddings │     │ attention │ │
    │                       │(uses IRLS)│     │   .py     │     │   .py     │ │
    │                       └───────────┘     └───────────┘     └───────────┘ │
    │                                                                          │
    │  IRLS is the CORE algorithm:                                            │
    │  • Each iteration solves weighted ridge regression                       │
    │  • Connects least squares (MAT223) to classification (ML)               │
    │  • Used by hazard model for timing predictions                          │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    LOGISTIC REGRESSION OVERVIEW
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  THE LOGISTIC MODEL                                                      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  MODEL:                                                                  │
    │  ──────                                                                  │
    │      P(Y=1 | x) = σ(xᵀβ) = 1 / (1 + exp(-xᵀβ))                         │
    │                                                                          │
    │  WHERE:                                                                  │
    │  ──────                                                                  │
    │  • x = feature vector (avg, is_UofT, is_CS, ...)                        │
    │  • β = coefficients (learned from data)                                 │
    │  • σ(z) = sigmoid function                                              │
    │                                                                          │
    │  SIGMOID FUNCTION:                                                       │
    │  ──────────────────                                                      │
    │                                                                          │
    │      σ(z) │           ╭───────────── 1                                  │
    │           │         ╱                                                    │
    │       0.5 ├───────●                                                      │
    │           │      ╱                                                       │
    │         0 ├─────╱──────────────────                                     │
    │           └────────────────────────► z                                   │
    │              -4  -2   0   2   4                                          │
    │                                                                          │
    │  Properties:                                                             │
    │  • σ(z) ∈ (0, 1) for all z ∈ ℝ                                         │
    │  • σ(0) = 0.5                                                           │
    │  • σ(-z) = 1 - σ(z)                                                     │
    │  • σ'(z) = σ(z)(1 - σ(z))                                              │
    │                                                                          │
    │  WHY LOGISTIC (not linear regression)?                                   │
    │  ──────────────────────────────────────                                  │
    │  Linear: P̂ = xᵀβ could be > 1 or < 0 (invalid probability!)           │
    │  Logistic: P̂ = σ(xᵀβ) is always in (0, 1) ✓                           │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    MAXIMUM LIKELIHOOD ESTIMATION
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  LOG-LIKELIHOOD AND LOSS                                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  LIKELIHOOD:                                                             │
    │  ───────────                                                             │
    │  For n independent observations:                                         │
    │                                                                          │
    │      L(β) = Πᵢ pᵢ^yᵢ (1-pᵢ)^(1-yᵢ)                                    │
    │                                                                          │
    │  Where pᵢ = σ(xᵢᵀβ)                                                    │
    │                                                                          │
    │  LOG-LIKELIHOOD (easier to work with):                                   │
    │  ───────────────────────────────────────                                 │
    │                                                                          │
    │      ℓ(β) = Σᵢ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]                        │
    │           = Σᵢ [yᵢ xᵢᵀβ - log(1 + exp(xᵢᵀβ))]                         │
    │                                                                          │
    │  NEGATIVE LOG-LIKELIHOOD (LOSS):                                         │
    │  ─────────────────────────────────                                       │
    │                                                                          │
    │      L(β) = -ℓ(β) = Σᵢ [-yᵢ xᵢᵀβ + log(1 + exp(xᵢᵀβ))]               │
    │                                                                          │
    │  With ridge regularization:                                              │
    │                                                                          │
    │      L_reg(β) = L(β) + (λ/2)||β||²                                      │
    │                                                                          │
    │  GOAL: Find β that MINIMIZES L_reg(β)                                   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    GRADIENT AND HESSIAN
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  DERIVATIVES OF LOG-LIKELIHOOD                                           │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  GRADIENT (first derivative):                                            │
    │  ─────────────────────────────                                           │
    │                                                                          │
    │      ∇ℓ(β) = Σᵢ (yᵢ - pᵢ) xᵢ = Xᵀ(y - p)                              │
    │                                                                          │
    │  With regularization:                                                    │
    │                                                                          │
    │      ∇L_reg(β) = -Xᵀ(y - p) + λβ                                       │
    │                = Xᵀ(p - y) + λβ                                         │
    │                                                                          │
    │  INTERPRETATION:                                                         │
    │  ────────────────                                                        │
    │  • (yᵢ - pᵢ) is the "residual" for sample i                            │
    │  • Gradient is 0 when predictions match outcomes                        │
    │                                                                          │
    │                                                                          │
    │  HESSIAN (second derivative):                                            │
    │  ──────────────────────────────                                          │
    │                                                                          │
    │      ∇²ℓ(β) = -Σᵢ pᵢ(1-pᵢ) xᵢxᵢᵀ = -XᵀWX                              │
    │                                                                          │
    │  Where W = diag(p₁(1-p₁), p₂(1-p₂), ..., pₙ(1-pₙ))                     │
    │                                                                          │
    │  With regularization:                                                    │
    │                                                                          │
    │      ∇²L_reg(β) = XᵀWX + λI                                            │
    │                                                                          │
    │  PROPERTIES:                                                             │
    │  ───────────                                                             │
    │  • W is diagonal with wᵢᵢ = pᵢ(1-pᵢ) ∈ (0, 0.25]                       │
    │  • Hessian is POSITIVE SEMIDEFINITE → loss is CONVEX!                  │
    │  • Convexity means gradient descent finds global optimum                │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    IRLS ALGORITHM
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  ITERATIVELY REWEIGHTED LEAST SQUARES                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  KEY INSIGHT:                                                            │
    │  ────────────                                                            │
    │  Newton's method for logistic regression is equivalent to               │
    │  solving a WEIGHTED LEAST SQUARES problem at each step!                 │
    │                                                                          │
    │  NEWTON UPDATE:                                                          │
    │  ───────────────                                                         │
    │      β_new = β_old - [∇²L]⁻¹ ∇L                                        │
    │            = β_old - (XᵀWX + λI)⁻¹[Xᵀ(p-y) + λβ_old]                  │
    │                                                                          │
    │  Rearranging gives:                                                      │
    │                                                                          │
    │      (XᵀWX + λI) β_new = XᵀWz                                          │
    │                                                                          │
    │  Where z = "working response":                                           │
    │                                                                          │
    │      zᵢ = xᵢᵀβ_old + (yᵢ - pᵢ)/wᵢ                                     │
    │                                                                          │
    │  THIS IS RIDGE REGRESSION with weights W and response z!                │
    │                                                                          │
    │                                                                          │
    │  ALGORITHM:                                                              │
    │  ──────────                                                              │
    │                                                                          │
    │      Input: X, y, λ, max_iter, tol                                      │
    │      Initialize: β = 0                                                   │
    │                                                                          │
    │      For iteration = 1, 2, ..., max_iter:                               │
    │          1. Compute probabilities: p = sigmoid(Xβ)                      │
    │          2. Compute weights: W = diag(p(1-p))                           │
    │          3. Compute working response: z = Xβ + (y-p)/W                  │
    │          4. Solve weighted ridge: β_new = (XᵀWX + λI)⁻¹XᵀWz           │
    │          5. Check convergence: if ||β_new - β|| < tol, stop            │
    │          6. Update: β = β_new                                           │
    │                                                                          │
    │      Return: β                                                           │
    │                                                                          │
    │                                                                          │
    │  VISUALIZATION OF CONVERGENCE:                                           │
    │  ───────────────────────────────                                         │
    │                                                                          │
    │      Loss │                                                              │
    │           │╲                                                             │
    │           │ ╲                                                            │
    │           │  ╲                                                           │
    │           │   ╲                                                          │
    │           │    ╲                                                         │
    │           │     ●───●───●───●  ← converged                              │
    │           └────────────────────► Iteration                              │
    │            1   2   3   4   5                                             │
    │                                                                          │
    │  Typically converges in 5-15 iterations (quadratic convergence).        │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    NUMERICAL CONSIDERATIONS
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  HANDLING NUMERICAL ISSUES                                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  1. SIGMOID OVERFLOW:                                                    │
    │     ─────────────────                                                    │
    │     exp(700) overflows! But σ(700) ≈ 1.                                 │
    │                                                                          │
    │     Solution: Stable sigmoid implementation                              │
    │     σ(z) = 1/(1+exp(-z))     if z ≥ 0                                  │
    │          = exp(z)/(1+exp(z))  if z < 0                                  │
    │                                                                          │
    │  2. LOG(0) PROBLEMS:                                                     │
    │     ──────────────────                                                   │
    │     log(p) when p = 0 is -∞!                                            │
    │                                                                          │
    │     Solution: Clip probabilities                                        │
    │     p = np.clip(p, 1e-15, 1 - 1e-15)                                   │
    │                                                                          │
    │  3. DIVISION BY ZERO IN WEIGHTS:                                         │
    │     ─────────────────────────────                                        │
    │     w = p(1-p) → 0 when p → 0 or p → 1                                 │
    │                                                                          │
    │     Solution: Clip weights                                               │
    │     w = np.clip(w, 1e-10, 0.25)                                        │
    │                                                                          │
    │  4. PERFECT SEPARATION:                                                  │
    │     ────────────────────                                                 │
    │     If classes are perfectly separable, β → ∞                           │
    │                                                                          │
    │     Solution: Ridge regularization λ > 0                                │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass


class IRLSConvergenceError(Exception):
    """Raised when IRLS fails to converge."""
    pass


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  STABLE SIGMOID                                                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      σ(z) = 1 / (1 + exp(-z))                                           │
    │                                                                          │
    │  NUMERICAL STABILITY:                                                    │
    │  ─────────────────────                                                   │
    │  For z >> 0: exp(-z) ≈ 0, so σ(z) ≈ 1     (stable)                     │
    │  For z << 0: exp(-z) → ∞, so 1/(1+∞) → 0  (overflow risk!)             │
    │                                                                          │
    │  SOLUTION:                                                               │
    │  ─────────                                                               │
    │  Use different formulas for positive and negative z:                    │
    │                                                                          │
    │      σ(z) = 1/(1+exp(-z))     if z ≥ 0                                 │
    │           = exp(z)/(1+exp(z))  if z < 0                                 │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        z: Input array of any shape

    Returns:
        Sigmoid values in (0, 1), same shape as input

    Example:
        >>> sigmoid(np.array([0, 1, -1, 100, -100]))
        array([0.5, 0.731..., 0.268..., 1.0, 0.0])

    IMPLEMENTATION:
    ────────────────
    result = np.zeros_like(z, dtype=float)
    pos_mask = z >= 0
    result[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
    result[~pos_mask] = np.exp(z[~pos_mask]) / (1 + np.exp(z[~pos_mask]))
    return result
    """
    pass


def log_loss(
    y: np.ndarray,
    p: np.ndarray,
    eps: float = 1e-15
) -> float:
    """
    Compute negative log-likelihood (cross-entropy loss).

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  LOG LOSS                                                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      L = -(1/n) Σᵢ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]                    │
    │                                                                          │
    │  INTERPRETATION:                                                         │
    │  ────────────────                                                        │
    │  • L = 0 means perfect predictions (p=1 when y=1, p=0 when y=0)        │
    │  • L = ln(2) ≈ 0.693 is baseline for random guessing (p=0.5)           │
    │  • L → ∞ for confident wrong predictions (p=1 when y=0)                │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        y: True labels (0 or 1), shape (n,)
        p: Predicted probabilities, shape (n,)
        eps: Small constant for numerical stability

    Returns:
        Average log loss (lower is better)

    IMPLEMENTATION:
    ────────────────
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    """
    pass


def compute_gradient(
    X: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    beta: np.ndarray,
    lambda_: float
) -> np.ndarray:
    """
    Compute gradient of regularized log loss.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  GRADIENT                                                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      ∇L = Xᵀ(p - y) + λβ                                               │
    │                                                                          │
    │  WHERE:                                                                  │
    │  ──────                                                                  │
    │  • p = sigmoid(Xβ) = current predictions                                │
    │  • (p - y) = residuals                                                  │
    │  • λβ = regularization term                                             │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix (n, p)
        y: Labels (n,)
        p: Current probabilities (n,)
        beta: Current coefficients (p,)
        lambda_: Regularization strength

    Returns:
        Gradient vector (p,)

    IMPLEMENTATION:
    ────────────────
    return X.T @ (p - y) + lambda_ * beta
    """
    pass


def compute_hessian(
    X: np.ndarray,
    p: np.ndarray,
    lambda_: float
) -> np.ndarray:
    """
    Compute Hessian of regularized log loss.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  HESSIAN                                                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      H = XᵀWX + λI                                                     │
    │                                                                          │
    │  WHERE:                                                                  │
    │  ──────                                                                  │
    │  • W = diag(p(1-p)) = diagonal weight matrix                           │
    │  • λI = regularization                                                  │
    │                                                                          │
    │  EFFICIENT COMPUTATION:                                                  │
    │  ───────────────────────                                                 │
    │  Don't form the full W matrix! Use:                                     │
    │      XᵀWX = (W^½X)ᵀ(W^½X)                                             │
    │           = (X * sqrt(w))ᵀ @ (X * sqrt(w))                             │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix (n, p)
        p: Current probabilities (n,)
        lambda_: Regularization strength

    Returns:
        Hessian matrix (p, p)

    IMPLEMENTATION:
    ────────────────
    w = p * (1 - p)
    w = np.clip(w, 1e-10, 0.25)  # Numerical stability
    X_weighted = X * np.sqrt(w)[:, np.newaxis]
    H = X_weighted.T @ X_weighted + lambda_ * np.eye(X.shape[1])
    return H
    """
    pass


def irls_step(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    lambda_: float
) -> np.ndarray:
    """
    Perform one IRLS iteration.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  SINGLE IRLS ITERATION                                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  1. Compute probabilities: p = sigmoid(Xβ)                              │
    │  2. Compute weights: w = p(1-p)                                         │
    │  3. Compute working response: z = Xβ + (y-p)/w                          │
    │  4. Solve weighted ridge: β_new = (XᵀWX + λI)⁻¹XᵀWz                   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix (n, p)
        y: Labels (n,)
        beta: Current coefficients (p,)
        lambda_: Regularization strength

    Returns:
        Updated coefficients (p,)

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. eta = X @ beta                     # Linear predictor
    2. p = sigmoid(eta)                    # Probabilities
    3. p = np.clip(p, 1e-10, 1-1e-10)     # Clip for stability
    4. w = p * (1 - p)                     # Weights
    5. w = np.clip(w, 1e-10, 0.25)         # Clip weights
    6. z = eta + (y - p) / w               # Working response
    7. # Solve weighted ridge:
       # (XᵀWX + λI) β = XᵀWz
       from ..math.ridge import weighted_ridge_solve
       return weighted_ridge_solve(X, z, w, lambda_)
    """
    pass


class LogisticModel:
    """
    Logistic regression via IRLS with ridge regularization.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  IRLS LOGISTIC REGRESSION                                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  MODEL:                                                                  │
    │      P(Y=1|x) = sigmoid(xᵀβ)                                           │
    │                                                                          │
    │  TRAINING:                                                               │
    │      Minimize: -log_likelihood + (λ/2)||β||²                           │
    │      Using: IRLS (Newton's method as weighted least squares)            │
    │                                                                          │
    │  PREDICTION:                                                             │
    │      p = sigmoid(X @ beta)                                              │
    │                                                                          │
    │  COEFFICIENTS:                                                           │
    │      β_j > 0: Feature j increases P(admit)                              │
    │      β_j < 0: Feature j decreases P(admit)                              │
    │      |β_j| large: Feature j has strong effect                          │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Example:
        >>> model = LogisticModel(lambda_=0.1, max_iter=100)
        >>> model.fit(X_train, y_train)
        >>> probs = model.predict_proba(X_test)
        >>> print(f"Coefficients: {model.coefficients}")
    """

    def __init__(
        self,
        lambda_: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        verbose: bool = False
    ):
        """
        Initialize logistic regression model.

        Args:
            lambda_: Ridge regularization strength (≥ 0)
            max_iter: Maximum IRLS iterations
            tol: Convergence tolerance for coefficient change
            fit_intercept: If True, add intercept column to X
            verbose: If True, print convergence information

        IMPLEMENTATION:
        ────────────────
        Store all parameters as instance attributes.
        Initialize:
            self.coefficients = None
            self._is_fitted = False
            self.training_history = {'loss': [], 'grad_norm': []}
            self.n_iter_ = 0
        """
        pass

    @property
    def name(self) -> str:
        """Model name."""
        pass

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        pass

    @property
    def n_features(self) -> Optional[int]:
        """Number of features (including intercept if fitted)."""
        pass

    @property
    def n_params(self) -> int:
        """Number of learnable parameters."""
        pass

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        Add intercept column (column of 1s) to X.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  INTERCEPT COLUMN                                                    │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  Before:                After:                                       │
        │  ┌─────────────┐        ┌───────────────────┐                       │
        │  │ x₁₁  x₁₂   │        │ 1  x₁₁  x₁₂      │                       │
        │  │ x₂₁  x₂₂   │   →    │ 1  x₂₁  x₂₂      │                       │
        │  │ x₃₁  x₃₂   │        │ 1  x₃₁  x₃₂      │                       │
        │  └─────────────┘        └───────────────────┘                       │
        │                                                                      │
        │  This allows the model to learn a bias term β₀.                     │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        IMPLEMENTATION:
        ────────────────
        return np.column_stack([np.ones(X.shape[0]), X])
        """
        pass

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> 'LogisticModel':
        """
        Fit logistic regression using IRLS.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  IRLS TRAINING LOOP                                                  │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  For iter = 1, 2, ..., max_iter:                                    │
        │      1. p = sigmoid(X @ beta)                                       │
        │      2. w = p * (1 - p)                                             │
        │      3. z = X @ beta + (y - p) / w                                  │
        │      4. beta_new = weighted_ridge_solve(X, z, w, lambda)            │
        │      5. if ||beta_new - beta|| < tol: converged!                   │
        │      6. beta = beta_new                                             │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            X: Design matrix (n_samples, n_features)
            y: Binary labels (n_samples,)
            sample_weight: Optional sample weights (n_samples,)

        Returns:
            self

        Raises:
            IRLSConvergenceError: If max_iter reached without convergence

        IMPLEMENTATION STEPS:
        ─────────────────────
        1. If fit_intercept, X = self._add_intercept(X)
        2. Initialize beta = np.zeros(X.shape[1])
        3. For iter in range(max_iter):
               a. beta_new = irls_step(X, y, beta, lambda_)
               b. diff = np.linalg.norm(beta_new - beta)
               c. Compute loss, gradient norm; store in history
               d. if verbose: print progress
               e. if diff < tol: break
               f. beta = beta_new
        4. self.coefficients = beta
        5. self._is_fitted = True
        6. self.n_iter_ = iter + 1
        7. Return self
        """
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict admission probabilities.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Probabilities (n_samples,)

        Raises:
            RuntimeError: If model not fitted

        IMPLEMENTATION:
        ────────────────
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        if self.fit_intercept:
            X = self._add_intercept(X)
        return sigmoid(X @ self.coefficients)
        """
        pass

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary outcomes.

        Args:
            X: Feature matrix
            threshold: Decision threshold (default 0.5)

        Returns:
            Binary predictions (0 or 1)
        """
        pass

    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get feature importance (coefficient magnitudes).

        ┌─────────────────────────────────────────────────────────────────────┐
        │  FEATURE IMPORTANCE                                                  │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  For logistic regression, importance = |β_j|                        │
        │                                                                      │
        │  CAUTION:                                                            │
        │  ────────                                                            │
        │  • Only meaningful if features are standardized!                    │
        │  • Otherwise, scale affects coefficient magnitude                   │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            feature_names: Optional list of feature names

        Returns:
            Dict mapping feature name → importance (sorted by importance)

        Example:
            >>> importance = model.get_feature_importance(['avg', 'is_cs'])
            >>> for name, imp in importance.items():
            ...     print(f"{name}: {imp:.3f}")
        """
        pass

    def get_odds_ratios(
        self,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get odds ratios for each feature.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  ODDS RATIOS                                                         │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  Odds = P(Y=1) / P(Y=0)                                             │
        │  Odds Ratio = exp(βⱼ)                                               │
        │                                                                      │
        │  INTERPRETATION:                                                     │
        │  ────────────────                                                    │
        │  OR = 2.0: One unit increase in feature DOUBLES the odds           │
        │  OR = 0.5: One unit increase HALVES the odds                        │
        │  OR = 1.0: Feature has no effect                                    │
        │                                                                      │
        │  Example: βⱼ = 0.1 for avg                                          │
        │  OR = exp(0.1) = 1.105                                              │
        │  "Each 1-point increase in avg increases odds by 10.5%"            │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            feature_names: Optional list of feature names

        Returns:
            Dict mapping feature name → odds ratio
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters for serialization."""
        pass

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters from serialization."""
        pass


# =============================================================================
#                           TODO LIST
# =============================================================================
"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TODO                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  HIGH PRIORITY (core IRLS):                                                  │
│  ──────────────────────────                                                  │
│  [ ] Implement sigmoid() with numerical stability                           │
│  [ ] Implement log_loss() for monitoring                                    │
│  [ ] Implement compute_gradient() - for diagnostics                        │
│  [ ] Implement irls_step() - single iteration                              │
│  [ ] Implement LogisticModel.fit() - main training loop                    │
│  [ ] Implement LogisticModel.predict_proba()                               │
│                                                                              │
│  MEDIUM PRIORITY (features):                                                 │
│  ────────────────────────────                                                │
│  [ ] Implement _add_intercept()                                             │
│  [ ] Implement predict() with threshold                                     │
│  [ ] Implement get_feature_importance()                                     │
│  [ ] Implement get_odds_ratios()                                            │
│                                                                              │
│  LOW PRIORITY (enhancements):                                                │
│  ─────────────────────────────                                               │
│  [ ] Add sample weights support                                             │
│  [ ] Add convergence diagnostics                                            │
│  [ ] Implement line search for step size                                    │
│  [ ] Add early stopping based on validation loss                            │
│                                                                              │
│  TESTING (CRITICAL):                                                         │
│  ────────────────────                                                        │
│  [ ] Compare to sklearn.linear_model.LogisticRegression                     │
│  [ ] Verify coefficients match to 4 decimal places                          │
│  [ ] Test convergence on synthetic data                                     │
│  [ ] Test numerical stability with extreme values                           │
│  [ ] Verify gradient computation against numerical gradient                 │
│                                                                              │
│  VERIFICATION CHECKLIST:                                                     │
│  ────────────────────────                                                    │
│  [ ] sigmoid(0) == 0.5                                                      │
│  [ ] sigmoid(large positive) ≈ 1.0                                         │
│  [ ] sigmoid(large negative) ≈ 0.0                                         │
│  [ ] Coefficients shrink with larger lambda                                 │
│  [ ] Loss decreases each IRLS iteration                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    print("IRLS Logistic Regression Module")
    print("=" * 50)
    print()
    print("This is the CORE algorithm for binary classification.")
    print()
    print("Key insight: Newton's method = weighted least squares!")
    print()
    print("After implementing, verify against sklearn:")
    print("  >>> from sklearn.linear_model import LogisticRegression")
    print("  >>> sklearn_model = LogisticRegression(C=1/lambda_)")
    print("  >>> sklearn_model.fit(X, y)")
    print("  >>> our_model = LogisticModel(lambda_=lambda_)")
    print("  >>> our_model.fit(X, y)")
    print("  >>> np.allclose(our_model.coefficients, sklearn_model.coef_, atol=1e-4)")
    print("  True")
