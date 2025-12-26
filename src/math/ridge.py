"""
Ridge Regression Solver Module
==============================

This module implements ridge regression, which adds L2 regularization to
least squares. Ridge is THE solution to ill-conditioned design matrices
and is the core of each IRLS iteration in logistic regression.

MAT223 REFERENCE: Sections 3.6 (Applications of Eigenvalues), 4.8 (Approximating
Solutions), combined with regularization theory.

================================================================================
                        SYSTEM ARCHITECTURE CONTEXT
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    WHERE THIS MODULE FITS                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │        vectors.py ──► matrices.py ──► projections.py                    │
    │                                              │                           │
    │                              ┌───────────────┴───────────────┐           │
    │                              ▼                               ▼           │
    │                        ┌──────────┐                    ┌──────────┐     │
    │                        │   qr.py  │                    │  svd.py  │     │
    │                        └────┬─────┘                    └────┬─────┘     │
    │                             │                               │            │
    │                             └───────────────┬───────────────┘            │
    │                                             ▼                            │
    │                                   ┌─────────────────┐                    │
    │                                   │  [THIS MODULE]  │                    │
    │                                   │    ridge.py     │                    │
    │                                   └────────┬────────┘                    │
    │                                            │                             │
    │                          ┌─────────────────┼─────────────────┐           │
    │                          ▼                 ▼                 ▼           │
    │                  ┌────────────┐    ┌────────────┐    ┌────────────┐     │
    │                  │ logistic.py│    │ baseline.py│    │ hazard.py  │     │
    │                  │   (IRLS)   │    │            │    │            │     │
    │                  └────────────┘    └────────────┘    └────────────┘     │
    │                                                                          │
    │  Ridge is used in EVERY weighted least squares step of IRLS!            │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    WHY RIDGE REGRESSION?
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  THE PROBLEM WITH ORDINARY LEAST SQUARES                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  OLS Problem: min_β ||Xβ - y||²                                         │
    │  Solution:    β̂ = (XᵀX)⁻¹Xᵀy                                          │
    │                                                                          │
    │  ISSUES WHEN XᵀX IS ILL-CONDITIONED:                                    │
    │  ─────────────────────────────────────                                   │
    │                                                                          │
    │  1. Near-collinearity:                                                   │
    │     ┌─────────────────────────────────────────────────────────┐         │
    │     │  Feature 1 (grade_11): [85, 86, 88, 90, 92, 93]         │         │
    │     │  Feature 2 (grade_12): [86, 87, 89, 91, 93, 94]         │         │
    │     │                                                          │         │
    │     │  These are almost perfectly correlated!                  │         │
    │     │  Small changes in data → HUGE changes in β               │         │
    │     └─────────────────────────────────────────────────────────┘         │
    │                                                                          │
    │  2. Rare categories:                                                     │
    │     ┌─────────────────────────────────────────────────────────┐         │
    │     │  is_queens_computing: [0, 0, 0, ..., 0, 1, 0]           │         │
    │     │                                                          │         │
    │     │  Only 1 student applied! Column is nearly zero.          │         │
    │     │  XᵀX is nearly singular.                                 │         │
    │     └─────────────────────────────────────────────────────────┘         │
    │                                                                          │
    │  3. High-dimensional:                                                    │
    │     ┌─────────────────────────────────────────────────────────┐         │
    │     │  p > n (more features than samples)                      │         │
    │     │                                                          │         │
    │     │  Infinitely many solutions exist!                        │         │
    │     │  XᵀX is not invertible.                                 │         │
    │     └─────────────────────────────────────────────────────────┘         │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    RIDGE REGRESSION: THE SOLUTION
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  RIDGE REGRESSION                                                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Ridge Problem:                                                          │
    │      min_β ||Xβ - y||² + λ||β||²                                        │
    │            ───────────   ────────                                        │
    │            fit data      regularization (penalty on large β)             │
    │                                                                          │
    │  Ridge Solution:                                                         │
    │      β_λ = (XᵀX + λI)⁻¹Xᵀy                                             │
    │                  ───                                                     │
    │                  This makes the matrix ALWAYS invertible!                │
    │                                                                          │
    │  DERIVATION:                                                             │
    │  ───────────                                                             │
    │  Taking derivative and setting to zero:                                  │
    │      ∇[||Xβ - y||² + λ||β||²] = 0                                       │
    │      2Xᵀ(Xβ - y) + 2λβ = 0                                              │
    │      XᵀXβ + λβ = Xᵀy                                                    │
    │      (XᵀX + λI)β = Xᵀy                                                  │
    │                                                                          │
    │  WHY λI HELPS:                                                           │
    │  ──────────────                                                          │
    │  If eigenvalues of XᵀX are: λ₁, λ₂, ..., λₚ                            │
    │  Then eigenvalues of XᵀX + λI are: λ₁+λ, λ₂+λ, ..., λₚ+λ              │
    │                                                                          │
    │  Even if λₚ ≈ 0, we now have λₚ + λ > 0!                               │
    │                                                                          │
    │      Original:           With Ridge:                                     │
    │      λₚ = 0.0001         λₚ + λ = 0.0001 + 0.1 = 0.1001                │
    │      cond = λ₁/λₚ       cond = λ₁/(λₚ+λ)                               │
    │           = 100000            ≈ 100 (much better!)                       │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    GEOMETRIC INTERPRETATION
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  RIDGE AS CONSTRAINED OPTIMIZATION                                       │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Equivalent formulation:                                                 │
    │      min_β ||Xβ - y||²   subject to   ||β||² ≤ t                        │
    │                                                                          │
    │  VISUALIZATION (2D coefficient space):                                   │
    │  ──────────────────────────────────────                                  │
    │                                                                          │
    │           β₂ ▲                                                           │
    │              │                                                           │
    │              │        ╭───────────╮                                     │
    │              │       ╱  OLS       ╲                                     │
    │              │      ╱  contours    ╲  (elliptical loss contours)        │
    │              │     │      ●        │  ← OLS solution (unconstrained)    │
    │              │      ╲    β̂_OLS    ╱                                     │
    │              │       ╲           ╱                                      │
    │              │     ╭──●────────╮     ← Ridge solution (on circle)       │
    │              │    │  β̂_λ      │                                        │
    │          ────┼────│───────────│──────► β₁                               │
    │              │    │           │                                          │
    │              │     ╰──────────╯                                          │
    │              │         ↑                                                 │
    │              │    ||β||² ≤ t  (L2 ball constraint)                      │
    │              │                                                           │
    │                                                                          │
    │  The ridge solution is where:                                            │
    │  • Loss contours are tangent to the constraint ball                      │
    │  • Larger λ → smaller ball → more shrinkage                             │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    WEIGHTED RIDGE (FOR IRLS)
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  WEIGHTED RIDGE REGRESSION                                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  In IRLS for logistic regression, each step solves:                      │
    │                                                                          │
    │      min_β Σᵢ wᵢ(zᵢ - xᵢᵀβ)² + λ||β||²                                │
    │                                                                          │
    │  Where:                                                                  │
    │  • wᵢ = pᵢ(1-pᵢ)           (variance of Bernoulli outcome)             │
    │  • zᵢ = xᵢᵀβ_old + (yᵢ-pᵢ)/wᵢ  (working response)                    │
    │  • pᵢ = sigmoid(xᵢᵀβ_old)  (current probability estimate)              │
    │                                                                          │
    │  Matrix form:                                                            │
    │      min_β ||W^½(Xβ - z)||² + λ||β||²                                   │
    │                                                                          │
    │  Normal equations:                                                       │
    │      (XᵀWX + λI)β = XᵀWz                                               │
    │                                                                          │
    │  INTERPRETATION OF WEIGHTS:                                              │
    │  ───────────────────────────                                             │
    │                                                                          │
    │      wᵢ = pᵢ(1-pᵢ)                                                      │
    │                                                                          │
    │      pᵢ     │  wᵢ   │  Interpretation                                   │
    │      ───────┼───────┼──────────────────────────────────                 │
    │      0.50   │ 0.25  │ Maximum weight (most uncertain)                   │
    │      0.10   │ 0.09  │ Lower weight (pretty confident reject)            │
    │      0.90   │ 0.09  │ Lower weight (pretty confident admit)             │
    │      0.01   │ 0.01  │ Very low weight (near-certain reject)             │
    │      0.99   │ 0.01  │ Very low weight (near-certain admit)              │
    │                                                                          │
    │  Uncertain predictions get MORE weight in the next iteration!           │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    CHOOSING λ
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  SELECTING REGULARIZATION STRENGTH                                       │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  METHODS:                                                                │
    │  ────────                                                                │
    │                                                                          │
    │  1. Cross-validation (gold standard):                                    │
    │     • Try λ ∈ [10⁻⁴, 10⁻³, 10⁻², 10⁻¹, 1, 10, 100]                    │
    │     • For each λ, compute CV error                                       │
    │     • Choose λ with lowest CV error                                      │
    │                                                                          │
    │  2. Grid relative to data scale:                                         │
    │     • λ ~ ||XᵀX||_F / p (average eigenvalue magnitude)                  │
    │     • Start with λ = 0.01 × this, go up to 100 × this                   │
    │                                                                          │
    │  3. Leave-one-out CV (efficient formula):                                │
    │     • LOO-CV can be computed in O(np²) for ridge!                       │
    │     • No need to refit n times                                           │
    │                                                                          │
    │  BIAS-VARIANCE TRADEOFF:                                                 │
    │  ────────────────────────                                                │
    │                                                                          │
    │      Error │                                                             │
    │            │   ╲                                                         │
    │            │    ╲  Total Error                                          │
    │            │     ╲                                                       │
    │            │      ╲    ╱                                                │
    │            │       ╲  ╱                                                 │
    │            │        ╲╱   optimal λ                                      │
    │            │        ╱╲                                                  │
    │            │       ╱  ╲ Bias²                                           │
    │            │      ╱    ╲                                                │
    │            │  Variance  ╲                                               │
    │            └────────────────────────► λ                                 │
    │               ↑              ↑                                           │
    │            underfit       overfit                                        │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, List, Union


class RidgeConvergenceError(Exception):
    """Raised when ridge regression fails to converge."""
    pass


class IllConditionedError(Exception):
    """Raised when matrix is too ill-conditioned even with regularization."""
    pass


def ridge_solve(
    X: np.ndarray,
    y: np.ndarray,
    lambda_: float
) -> np.ndarray:
    """
    Solve ridge regression: (XᵀX + λI)β = Xᵀy

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  RIDGE REGRESSION SOLVER                                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      β_λ = (XᵀX + λI)⁻¹Xᵀy                                             │
    │                                                                          │
    │  APPROACH (numerically stable):                                          │
    │  ──────────────────────────────                                          │
    │  Instead of computing the inverse, solve the linear system:              │
    │      (XᵀX + λI)β = Xᵀy                                                  │
    │                                                                          │
    │  Using Cholesky decomposition:                                           │
    │  1. A = XᵀX + λI  (symmetric positive definite!)                        │
    │  2. L = cholesky(A)  where A = LLᵀ                                      │
    │  3. Solve Lz = Xᵀy  (forward substitution)                              │
    │  4. Solve Lᵀβ = z   (backward substitution)                             │
    │                                                                          │
    │  WHY CHOLESKY:                                                           │
    │  ──────────────                                                          │
    │  • XᵀX + λI is always SPD (symmetric positive definite) for λ > 0      │
    │  • Cholesky is 2× faster than LU for SPD matrices                       │
    │  • More numerically stable than computing inverse                        │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix of shape (n, p)
        y: Target vector of shape (n,)
        lambda_: Regularization parameter (must be ≥ 0)

    Returns:
        Ridge coefficients of shape (p,)

    Raises:
        ValueError: If lambda_ < 0 or dimensions don't match
        IllConditionedError: If matrix is singular even with regularization

    Example:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([1, 2, 3])
        >>> beta = ridge_solve(X, y, lambda_=0.1)
        >>> print(beta.shape)
        (2,)

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Validate: lambda_ >= 0, X.shape[0] == y.shape[0]
    2. Compute XtX = X.T @ X
    3. Add regularization: XtX_reg = XtX + lambda_ * np.eye(p)
    4. Compute Xty = X.T @ y
    5. Solve using Cholesky:
       L = np.linalg.cholesky(XtX_reg)
       z = np.linalg.solve(L, Xty)          # Forward substitution
       beta = np.linalg.solve(L.T, z)       # Backward substitution
       OR simply: np.linalg.solve(XtX_reg, Xty)
    6. Return beta
    """
    pass


def ridge_solve_qr(
    X: np.ndarray,
    y: np.ndarray,
    lambda_: float
) -> np.ndarray:
    """
    Solve ridge regression using augmented QR factorization.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  RIDGE VIA AUGMENTED QR                                                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  KEY INSIGHT:                                                            │
    │  Ridge regression is equivalent to OLS on an augmented system:           │
    │                                                                          │
    │      ┌───────┐       ┌───┐                                              │
    │      │   X   │       │ y │                                              │
    │      │───────│ β  ≈  │───│                                              │
    │      │ √λ I  │       │ 0 │                                              │
    │      └───────┘       └───┘                                              │
    │                                                                          │
    │  Stacking √λ I below X makes it full rank even if X isn't!              │
    │                                                                          │
    │  DERIVATION:                                                             │
    │  ───────────                                                             │
    │  Minimize: ||Xβ - y||² + ||√λ β - 0||²                                  │
    │          = ||Xβ - y||² + λ||β||²                                        │
    │                                                                          │
    │  This is exactly ridge regression!                                       │
    │                                                                          │
    │  ADVANTAGE:                                                              │
    │  ──────────                                                              │
    │  • Uses QR on augmented matrix (numerically stable)                      │
    │  • Avoids forming XᵀX (better conditioning)                             │
    │  • Works well when n >> p                                                │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix of shape (n, p)
        y: Target vector of shape (n,)
        lambda_: Regularization parameter (≥ 0)

    Returns:
        Ridge coefficients of shape (p,)

    Example:
        >>> X = np.random.randn(100, 10)
        >>> y = np.random.randn(100)
        >>> beta = ridge_solve_qr(X, y, lambda_=1.0)

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. n, p = X.shape
    2. X_aug = np.vstack([X, np.sqrt(lambda_) * np.eye(p)])
    3. y_aug = np.concatenate([y, np.zeros(p)])
    4. Solve via QR: Q, R = np.linalg.qr(X_aug)
    5. beta = np.linalg.solve(R, Q.T @ y_aug)
    6. Return beta
    """
    pass


def weighted_ridge_solve(
    X: np.ndarray,
    z: np.ndarray,
    W: Union[np.ndarray, float],
    lambda_: float
) -> np.ndarray:
    """
    Solve weighted ridge regression: (XᵀWX + λI)β = XᵀWz

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  WEIGHTED RIDGE REGRESSION                                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Objective: min_β Σᵢ wᵢ(xᵢᵀβ - zᵢ)² + λ||β||²                         │
    │                                                                          │
    │  Solution: (XᵀWX + λI)β = XᵀWz                                         │
    │                                                                          │
    │  WHERE W = diag(w₁, w₂, ..., wₙ)                                        │
    │                                                                          │
    │  THIS IS THE CORE COMPUTATION IN IRLS!                                   │
    │  ─────────────────────────────────────                                   │
    │  Each iteration of IRLS calls this with:                                 │
    │  • wᵢ = pᵢ(1-pᵢ)                                                        │
    │  • zᵢ = working response                                                │
    │                                                                          │
    │  NUMERICAL CONSIDERATION:                                                │
    │  ────────────────────────                                                │
    │  When wᵢ is very small (pᵢ near 0 or 1):                               │
    │  • That observation has little influence                                 │
    │  • wᵢ = pᵢ(1-pᵢ) ≈ 0 for pᵢ ≈ 0 or pᵢ ≈ 1                            │
    │  • Need to handle near-zero weights carefully                            │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix of shape (n, p)
        z: Working response of shape (n,)
        W: Weights - either (n,) array or (n, n) diagonal matrix or scalar
        lambda_: Regularization parameter (≥ 0)

    Returns:
        Weighted ridge coefficients of shape (p,)

    Raises:
        ValueError: If dimensions don't match or lambda_ < 0

    Example:
        >>> X = np.array([[1, 0], [0, 1], [1, 1]])
        >>> z = np.array([1, 2, 2.5])
        >>> W = np.array([1.0, 0.5, 0.25])  # Weights
        >>> beta = weighted_ridge_solve(X, z, W, lambda_=0.1)

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Validate inputs
    2. If W is scalar, W = W * np.ones(n)
    3. If W is 1D, use it directly; if 2D diagonal, extract diagonal
    4. Compute XtWX = X.T @ (W[:, np.newaxis] * X)  # Efficient: avoid forming diag(W)
    5. Add regularization: XtWX_reg = XtWX + lambda_ * np.eye(p)
    6. Compute XtWz = X.T @ (W * z)
    7. Solve and return: np.linalg.solve(XtWX_reg, XtWz)
    """
    pass


def ridge_loocv(
    X: np.ndarray,
    y: np.ndarray,
    lambda_: float
) -> float:
    """
    Compute Leave-One-Out Cross-Validation error efficiently.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  EFFICIENT LOO-CV FOR RIDGE                                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  NAIVE APPROACH (expensive):                                             │
    │  ────────────────────────────                                            │
    │  For i = 1 to n:                                                         │
    │      Fit ridge on all data except i                                      │
    │      Predict for i                                                       │
    │      Compute squared error                                               │
    │                                                                          │
    │  Cost: O(n × cost of ridge) = O(n × np²) = O(n²p²)                      │
    │                                                                          │
    │  EFFICIENT FORMULA (magic!):                                             │
    │  ─────────────────────────────                                           │
    │                                                                          │
    │      LOOCV = (1/n) Σᵢ [(yᵢ - ŷᵢ) / (1 - hᵢᵢ)]²                        │
    │                                                                          │
    │  Where:                                                                  │
    │  • ŷ = X @ β_λ (fitted values from full model)                          │
    │  • hᵢᵢ = diagonal of H_λ = X(XᵀX + λI)⁻¹Xᵀ (leverage)                 │
    │                                                                          │
    │  Cost: O(np²) - same as fitting once!                                   │
    │                                                                          │
    │  WHY THIS WORKS:                                                         │
    │  ────────────────                                                        │
    │  hᵢᵢ measures how much observation i "pulls" the fit toward itself.    │
    │  Dividing by (1 - hᵢᵢ) corrects for this influence.                    │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix of shape (n, p)
        y: Target vector of shape (n,)
        lambda_: Regularization parameter (≥ 0)

    Returns:
        Leave-one-out cross-validation mean squared error

    Example:
        >>> X = np.random.randn(100, 10)
        >>> y = X @ np.random.randn(10) + 0.1 * np.random.randn(100)
        >>> loocv_01 = ridge_loocv(X, y, lambda_=0.1)
        >>> loocv_10 = ridge_loocv(X, y, lambda_=10.0)

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. beta = ridge_solve(X, y, lambda_)
    2. y_hat = X @ beta
    3. residuals = y - y_hat
    4. Compute leverage: H = X @ np.linalg.solve(X.T @ X + lambda_ * I, X.T)
    5. h_diag = np.diag(H)  # Or more efficient: see below
    6. loo_residuals = residuals / (1 - h_diag)
    7. return np.mean(loo_residuals ** 2)

    EFFICIENT LEVERAGE COMPUTATION:
    ───────────────────────────────
    # Using SVD of X for efficiency:
    U, s, V = np.linalg.svd(X, full_matrices=False)
    shrinkage = s**2 / (s**2 + lambda_)
    h_diag = np.sum((U ** 2) * shrinkage, axis=1)
    """
    pass


def ridge_cv(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: List[float],
    cv_folds: int = 5
) -> Tuple[float, np.ndarray]:
    """
    Select optimal λ using k-fold cross-validation.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  K-FOLD CROSS-VALIDATION FOR λ SELECTION                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  PROCEDURE:                                                              │
    │  ──────────                                                              │
    │  For each λ in lambdas:                                                  │
    │      For each fold k = 1, ..., K:                                        │
    │          Train on folds ≠ k                                              │
    │          Test on fold k                                                  │
    │          Record MSE                                                      │
    │      Average MSE across folds                                            │
    │  Select λ with lowest average MSE                                        │
    │                                                                          │
    │  VISUALIZATION:                                                          │
    │  ───────────────                                                         │
    │                                                                          │
    │      CV Error │                                                          │
    │               │     ╲                                                    │
    │               │      ╲                                                   │
    │               │       ╲     ← underfitting                               │
    │               │        ╲                                                 │
    │               │         ╲                                                │
    │               │          ●  ← optimal λ                                  │
    │               │         ╱                                                │
    │               │        ╱                                                 │
    │               │       ╱   ← overfitting                                  │
    │               └──────────────────────────────► log(λ)                   │
    │                  small λ          large λ                               │
    │                                                                          │
    │  COMMON λ GRIDS:                                                         │
    │  ────────────────                                                        │
    │  • np.logspace(-4, 4, 50)  # 50 values from 10⁻⁴ to 10⁴               │
    │  • np.logspace(-2, 2, 20)  # Smaller range for quick tuning             │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix of shape (n, p)
        y: Target vector of shape (n,)
        lambdas: List of λ values to try
        cv_folds: Number of cross-validation folds (default: 5)

    Returns:
        Tuple of (best_lambda, cv_scores) where:
        - best_lambda: λ with lowest CV error
        - cv_scores: Array of CV errors for each λ

    Example:
        >>> X = np.random.randn(200, 20)
        >>> y = X @ np.random.randn(20) + 0.5 * np.random.randn(200)
        >>> lambdas = np.logspace(-3, 3, 20)
        >>> best_lambda, cv_scores = ridge_cv(X, y, lambdas)
        >>> print(f"Best lambda: {best_lambda:.4f}")

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. n = X.shape[0]
    2. Create fold indices: np.random.permutation(n) split into cv_folds groups
    3. cv_scores = np.zeros(len(lambdas))
    4. For i, lambda_ in enumerate(lambdas):
           fold_scores = []
           For each fold:
               X_train, y_train = data excluding fold
               X_val, y_val = data in fold
               beta = ridge_solve(X_train, y_train, lambda_)
               mse = np.mean((y_val - X_val @ beta)**2)
               fold_scores.append(mse)
           cv_scores[i] = np.mean(fold_scores)
    5. best_idx = np.argmin(cv_scores)
    6. return lambdas[best_idx], cv_scores
    """
    pass


def ridge_path(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: List[float]
) -> np.ndarray:
    """
    Compute ridge coefficients for a sequence of λ values.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  RIDGE REGULARIZATION PATH                                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Shows how coefficients shrink as λ increases.                           │
    │                                                                          │
    │  VISUALIZATION (coefficient paths):                                      │
    │  ────────────────────────────────────                                    │
    │                                                                          │
    │      β │                                                                 │
    │        │ ────────                                                        │
    │        │          ╲                                                      │
    │        │           ╲ β₁                                                 │
    │        │            ╲                                                    │
    │      0 ┼─────────────╲───────────────────────                           │
    │        │              ╲╱                                                 │
    │        │            ╱                                                    │
    │        │          ╱  β₂                                                 │
    │        │  ──────╱                                                        │
    │        └────────────────────────────────────► log(λ)                    │
    │           OLS                            All zeros                       │
    │          solution                                                        │
    │                                                                          │
    │  INTERPRETATION:                                                         │
    │  ────────────────                                                        │
    │  • Left (λ → 0): Coefficients approach OLS solution                     │
    │  • Right (λ → ∞): Coefficients shrink toward zero                       │
    │  • Coefficients with larger |β| shrink slower                           │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  • Visualize feature importance stability                                │
    │  • Identify which features are "robust" (stay nonzero)                   │
    │  • Choose λ based on when certain features disappear                     │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix of shape (n, p)
        y: Target vector of shape (n,)
        lambdas: Sequence of λ values (should be sorted)

    Returns:
        Array of shape (len(lambdas), p) with coefficients for each λ

    Example:
        >>> X = np.random.randn(100, 5)
        >>> y = X @ np.array([1, 0.5, 0, -0.5, 0.1]) + 0.1 * np.random.randn(100)
        >>> lambdas = np.logspace(-3, 3, 50)
        >>> path = ridge_path(X, y, lambdas)
        >>> path.shape
        (50, 5)

    IMPLEMENTATION:
    ────────────────
    return np.array([ridge_solve(X, y, lam) for lam in lambdas])

    EFFICIENT IMPLEMENTATION (using SVD):
    ─────────────────────────────────────
    Compute SVD once, then for each λ:
    1. U, s, V = np.linalg.svd(X, full_matrices=False)
    2. d = s / (s**2 + lambda_)  # Shrinkage factors
    3. beta = V.T @ (d * (U.T @ y))
    """
    pass


def ridge_gcv(
    X: np.ndarray,
    y: np.ndarray,
    lambda_: float
) -> float:
    """
    Compute Generalized Cross-Validation score.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  GENERALIZED CROSS-VALIDATION (GCV)                                      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      GCV(λ) = (1/n) ||y - ŷ||² / (1 - tr(H_λ)/n)²                      │
    │                                                                          │
    │  WHERE:                                                                  │
    │  ──────                                                                  │
    │  • ŷ = X @ β_λ (fitted values)                                          │
    │  • H_λ = X(XᵀX + λI)⁻¹Xᵀ (smoother matrix)                             │
    │  • tr(H_λ) = effective degrees of freedom = Σᵢ σᵢ²/(σᵢ²+λ)            │
    │                                                                          │
    │  INTERPRETATION:                                                         │
    │  ────────────────                                                        │
    │  GCV is an approximation to LOO-CV that's even faster to compute.       │
    │  Instead of individual leverage values hᵢᵢ, it uses their average.     │
    │                                                                          │
    │  COMPARISON TO LOOCV:                                                    │
    │  ─────────────────────                                                   │
    │  LOOCV: (1/n) Σᵢ [(yᵢ - ŷᵢ)/(1 - hᵢᵢ)]²                               │
    │  GCV:   (1/n) ||y - ŷ||² / (1 - avg(hᵢᵢ))²                             │
    │                                                                          │
    │  GCV replaces individual hᵢᵢ with their average!                        │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix of shape (n, p)
        y: Target vector of shape (n,)
        lambda_: Regularization parameter (≥ 0)

    Returns:
        GCV score (lower is better)

    Example:
        >>> X = np.random.randn(100, 10)
        >>> y = np.random.randn(100)
        >>> gcv_01 = ridge_gcv(X, y, lambda_=0.1)
        >>> gcv_10 = ridge_gcv(X, y, lambda_=10.0)

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. n = X.shape[0]
    2. beta = ridge_solve(X, y, lambda_)
    3. y_hat = X @ beta
    4. rss = np.sum((y - y_hat)**2)  # Residual sum of squares
    5. # Compute effective df using SVD
       s = np.linalg.svd(X, compute_uv=False)
       df = np.sum(s**2 / (s**2 + lambda_))
    6. gcv = (rss / n) / (1 - df/n)**2
    7. return gcv
    """
    pass


def standardize_features(
    X: np.ndarray,
    center: bool = True,
    scale: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features to have zero mean and unit variance.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  FEATURE STANDARDIZATION                                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  WHY STANDARDIZE FOR RIDGE?                                              │
    │  ───────────────────────────                                             │
    │                                                                          │
    │  Ridge penalty is ||β||² = β₁² + β₂² + ... + βₚ²                       │
    │                                                                          │
    │  If features have different scales:                                      │
    │  • Feature in [0, 100] will have small β (to keep predictions right)    │
    │  • Feature in [0, 1] will have large β                                  │
    │  • Penalty hits the large β harder, even if both features matter!       │
    │                                                                          │
    │  SOLUTION:                                                               │
    │  ─────────                                                               │
    │  Standardize so all features have mean=0, std=1.                         │
    │  Then ridge penalizes all coefficients equally.                          │
    │                                                                          │
    │  TRANSFORMATION:                                                         │
    │  ────────────────                                                        │
    │      X_std[:, j] = (X[:, j] - mean_j) / std_j                           │
    │                                                                          │
    │  RECOVERING ORIGINAL COEFFICIENTS:                                       │
    │  ─────────────────────────────────                                       │
    │      β_orig = β_std / stds                                               │
    │      intercept_orig = y_mean - β_orig @ means                           │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix of shape (n, p)
        center: If True, subtract column means
        scale: If True, divide by column standard deviations

    Returns:
        Tuple of (X_standardized, means, stds)

    Example:
        >>> X = np.array([[1, 100], [2, 200], [3, 300]])
        >>> X_std, means, stds = standardize_features(X)
        >>> np.allclose(X_std.mean(axis=0), 0)
        True
        >>> np.allclose(X_std.std(axis=0), 1)
        True

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. means = X.mean(axis=0) if center else np.zeros(p)
    2. stds = X.std(axis=0) if scale else np.ones(p)
    3. stds = np.where(stds < 1e-10, 1.0, stds)  # Avoid division by zero
    4. X_std = (X - means) / stds
    5. return X_std, means, stds
    """
    pass


def ridge_effective_df(
    X: np.ndarray,
    lambda_: float
) -> float:
    """
    Compute effective degrees of freedom for ridge regression.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  EFFECTIVE DEGREES OF FREEDOM                                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      df(λ) = tr(H_λ) = Σᵢ σᵢ² / (σᵢ² + λ)                             │
    │                                                                          │
    │  WHERE σᵢ are the singular values of X.                                 │
    │                                                                          │
    │  INTERPRETATION:                                                         │
    │  ────────────────                                                        │
    │  • λ = 0:   df = p (full model complexity)                              │
    │  • λ → ∞:   df → 0 (null model)                                         │
    │  • Between: 0 < df < p (regularized)                                    │
    │                                                                          │
    │  USE CASES:                                                              │
    │  ──────────                                                              │
    │  • Model comparison (similar to counting parameters)                     │
    │  • AIC/BIC computation for ridge                                         │
    │  • Understanding regularization strength                                 │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix of shape (n, p)
        lambda_: Regularization parameter (≥ 0)

    Returns:
        Effective degrees of freedom (between 0 and p)

    Example:
        >>> X = np.random.randn(100, 10)
        >>> ridge_effective_df(X, lambda_=0)  # Returns 10 (full model)
        >>> ridge_effective_df(X, lambda_=1e6)  # Returns ~0 (null model)

    IMPLEMENTATION:
    ────────────────
    s = np.linalg.svd(X, compute_uv=False)
    return np.sum(s**2 / (s**2 + lambda_))
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
│  HIGH PRIORITY (core solvers):                                               │
│  ──────────────────────────────                                              │
│  [ ] Implement ridge_solve() - main solver using Cholesky                   │
│  [ ] Implement weighted_ridge_solve() - for IRLS (CRITICAL!)               │
│  [ ] Implement standardize_features() - preprocessing                       │
│                                                                              │
│  MEDIUM PRIORITY (model selection):                                          │
│  ──────────────────────────────────                                          │
│  [ ] Implement ridge_loocv() - efficient LOO-CV                             │
│  [ ] Implement ridge_gcv() - GCV score                                       │
│  [ ] Implement ridge_cv() - k-fold CV for λ selection                       │
│  [ ] Implement ridge_effective_df() - model complexity                      │
│                                                                              │
│  LOW PRIORITY (analysis/visualization):                                      │
│  ────────────────────────────────────                                        │
│  [ ] Implement ridge_path() - regularization path                           │
│  [ ] Implement ridge_solve_qr() - alternative solver                        │
│                                                                              │
│  TESTING:                                                                    │
│  ────────                                                                    │
│  [ ] Compare ridge_solve() to sklearn.linear_model.Ridge                    │
│  [ ] Verify LOOCV formula against brute-force LOO                           │
│  [ ] Test weighted ridge against manual weighted least squares              │
│  [ ] Verify effective df formula                                            │
│  [ ] Test that λ=0 recovers OLS solution                                    │
│  [ ] Test that λ→∞ gives β→0                                               │
│                                                                              │
│  NUMERICAL EDGE CASES:                                                       │
│  ──────────────────────                                                      │
│  [ ] Handle λ = 0 (falls back to OLS)                                       │
│  [ ] Handle zero columns in X                                               │
│  [ ] Handle near-zero weights in weighted ridge                              │
│  [ ] Add warning for high condition number even after regularization        │
│                                                                              │
│  VISUALIZATION IDEAS:                                                        │
│  ─────────────────────                                                       │
│  [ ] Plot coefficient paths (beta vs log(lambda))                           │
│  [ ] Plot CV error vs lambda (with error bars)                              │
│  [ ] Visualize shrinkage: compare OLS vs ridge coefficients                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    print("Ridge Regression Solver Module")
    print("=" * 50)
    print()
    print("Ridge regression: min ||Xβ - y||² + λ||β||²")
    print()
    print("Key benefits:")
    print("  1. Makes XᵀX + λI always invertible")
    print("  2. Reduces overfitting by shrinking coefficients")
    print("  3. Stabilizes solutions for ill-conditioned problems")
    print()
    print("After implementing, try:")
    print("  >>> X = np.random.randn(100, 50)  # p > n: normally problematic!")
    print("  >>> y = np.random.randn(100)")
    print("  >>> beta = ridge_solve(X, y, lambda_=1.0)  # Ridge handles it!")
