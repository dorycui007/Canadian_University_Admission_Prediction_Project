"""
Projections and Least Squares Module
=====================================

This module implements orthogonal projections and least squares solutions,
which are THE CORE of linear regression and form the foundation for IRLS
in logistic regression.

MAT223 REFERENCE: Sections 4.2.2 (Projections onto Lines), 4.7.1 (Projections
onto Subspaces), 4.8 (Approximating Solutions / Least Squares)

================================================================================
                        SYSTEM ARCHITECTURE CONTEXT
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    WHERE THIS MODULE FITS                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │                    vectors.py ──► matrices.py                            │
    │                                       │                                  │
    │                                       ▼                                  │
    │                             ┌─────────────────┐                          │
    │                             │  [THIS MODULE]  │                          │
    │                             │ projections.py  │                          │
    │                             └────────┬────────┘                          │
    │                                      │                                   │
    │           ┌──────────────────────────┼──────────────────────────┐        │
    │           ▼                          ▼                          ▼        │
    │      ┌─────────┐              ┌─────────────┐            ┌──────────┐   │
    │      │  qr.py  │              │  ridge.py   │            │ models/  │   │
    │      │ (better │              │(regularized │            │logistic  │   │
    │      │ solver) │              │  solver)    │            │  .py     │   │
    │      └─────────┘              └─────────────┘            └──────────┘   │
    │                                                                          │
    │  Projection is the GEOMETRIC FOUNDATION of least squares:               │
    │  Finding β̂ such that Xβ̂ is the closest point in col(X) to y           │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    THE FUNDAMENTAL IDEA: PROJECTION
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  PROJECTION: Finding the Closest Point                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Given: Target vector y, Subspace S (the column space of X)             │
    │  Find:  The closest point in S to y                                      │
    │                                                                          │
    │                          y                                               │
    │                          ●                                               │
    │                         /│                                               │
    │                        / │                                               │
    │                       /  │ r = y - ŷ (residual)                         │
    │                      /   │                                               │
    │                     /    │                                               │
    │   ──────────────────●────┼──────────────────  Subspace S = col(X)       │
    │                    ŷ = Xβ̂                                               │
    │                   (projection)                                           │
    │                                                                          │
    │  KEY INSIGHT:                                                            │
    │  ────────────                                                            │
    │  The projection ŷ is the unique point in S such that:                   │
    │                                                                          │
    │      r = y - ŷ  is ORTHOGONAL to S                                      │
    │                                                                          │
    │  This orthogonality is WHY ŷ is the closest point!                      │
    │  (Any other point in S would be farther by Pythagorean theorem)         │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    PROJECTION ONTO A LINE (1D SUBSPACE)
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  PROJECTION ONTO VECTOR a                                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  The projection of y onto the line spanned by a:                         │
    │                                                                          │
    │               (y · a)                                                    │
    │      proj_a(y) = ───── · a                                               │
    │               (a · a)                                                    │
    │                                                                          │
    │  VISUAL:                                                                 │
    │           ▲                                                              │
    │           │  y                                                           │
    │           ● ╱                                                            │
    │           │╱                                                             │
    │           │                                                              │
    │  ─────────●─────────────────►  line spanned by a                        │
    │          proj_a(y)                                                       │
    │                                                                          │
    │  The scalar (y·a)/(a·a) tells us "how much of a" we need.               │
    │                                                                          │
    │  PROJECT CONNECTION:                                                     │
    │  ────────────────────                                                    │
    │  This is the simplest case of least squares!                             │
    │  When X has only ONE column a, the solution is:                          │
    │                                                                          │
    │           y · a                                                          │
    │      β̂ = ─────                                                          │
    │           a · a                                                          │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    PROJECTION ONTO SUBSPACES (MULTIPLE COLUMNS)
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  PROJECTION ONTO col(X)                                                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  When X has multiple columns, col(X) is a multi-dimensional subspace.   │
    │                                                                          │
    │  VISUAL (X has 2 columns → col(X) is a plane):                          │
    │                                                                          │
    │              y ●                                                         │
    │               ╲│                                                         │
    │                │ r = y - Xβ̂  (residual)                                 │
    │                │                                                         │
    │     ┌──────────┼───────────┐                                            │
    │     │    ●     │           │  col(X) = plane                            │
    │     │   Xβ̂    │           │  spanned by                                 │
    │     │         ╱╲           │  columns of X                               │
    │     │        ╱  ╲          │                                            │
    │     └───────╱────╲─────────┘                                            │
    │           col₁   col₂                                                    │
    │                                                                          │
    │  THE ORTHOGONALITY CONDITION:                                            │
    │  ─────────────────────────────                                           │
    │  r must be orthogonal to EVERY column of X:                              │
    │                                                                          │
    │      col₁ᵀr = 0                                                         │
    │      col₂ᵀr = 0                                                         │
    │      ...                                                                 │
    │      colₚᵀr = 0                                                         │
    │                                                                          │
    │  In matrix form:  Xᵀr = 0  ⟺  Xᵀ(y - Xβ̂) = 0                          │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    THE NORMAL EQUATIONS
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  DERIVATION OF NORMAL EQUATIONS                                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  From orthogonality: Xᵀ(y - Xβ̂) = 0                                    │
    │                                                                          │
    │  Expand:  Xᵀy - XᵀXβ̂ = 0                                               │
    │                                                                          │
    │  Rearrange:  XᵀXβ̂ = Xᵀy                                                │
    │                                                                          │
    │  This is THE NORMAL EQUATIONS!                                           │
    │                                                                          │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │                                                                  │    │
    │  │                   XᵀXβ̂ = Xᵀy                                    │    │
    │  │                                                                  │    │
    │  │  If XᵀX is invertible (X has full column rank):                 │    │
    │  │                                                                  │    │
    │  │                   β̂ = (XᵀX)⁻¹Xᵀy                               │    │
    │  │                                                                  │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    │                                                                          │
    │  COMPONENTS:                                                             │
    │  ───────────                                                             │
    │  • XᵀX: (p×p) Gram matrix - captures feature correlations              │
    │  • Xᵀy: (p×1) cross-product - captures feature-target relationships    │
    │  • β̂:  (p×1) optimal coefficients                                      │
    │                                                                          │
    │  WARNING:                                                                │
    │  ────────                                                                │
    │  DON'T actually compute (XᵀX)⁻¹! It's numerically unstable.            │
    │  Use QR factorization (qr.py) or ridge (ridge.py) instead.              │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    LEAST SQUARES PROBLEM
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  LEAST SQUARES: min_β ||Xβ - y||²                                       │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  PROBLEM:                                                                │
    │  ────────                                                                │
    │  Find β that minimizes the sum of squared residuals:                     │
    │                                                                          │
    │      min_β Σᵢ (xᵢᵀβ - yᵢ)² = min_β ||Xβ - y||²                         │
    │                                                                          │
    │  SOLUTION (via projection):                                              │
    │  ──────────────────────────                                              │
    │  The minimizer β̂ satisfies the normal equations:                        │
    │                                                                          │
    │      XᵀXβ̂ = Xᵀy                                                        │
    │                                                                          │
    │  GEOMETRIC INTERPRETATION:                                               │
    │  ──────────────────────────                                              │
    │  The prediction Xβ̂ is the orthogonal projection of y onto col(X).      │
    │  The residual r = y - Xβ̂ is orthogonal to col(X).                      │
    │                                                                          │
    │  ┌──────────────────────────────────────────────────────────────────┐   │
    │  │                                                                   │   │
    │  │     ||y||² = ||Xβ̂||² + ||r||²    (Pythagorean theorem!)          │   │
    │  │                                                                   │   │
    │  │                   y                                               │   │
    │  │                   ●                                               │   │
    │  │                  /│                                               │   │
    │  │           ||y|| / │ ||r||                                        │   │
    │  │                /  │                                               │   │
    │  │               /   │                                               │   │
    │  │    ─────────●─────┴────────  col(X)                              │   │
    │  │           ||Xβ̂||                                                 │   │
    │  │                                                                   │   │
    │  └──────────────────────────────────────────────────────────────────┘   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    CONNECTION TO IRLS (LOGISTIC REGRESSION)
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  WEIGHTED LEAST SQUARES IN IRLS                                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  In IRLS for logistic regression, each iteration solves:                 │
    │                                                                          │
    │      min_β Σᵢ wᵢ(zᵢ - xᵢᵀβ)²                                           │
    │                                                                          │
    │  Where:                                                                  │
    │  • wᵢ = pᵢ(1-pᵢ)           (weight - variance of Bernoulli)            │
    │  • zᵢ = xᵢᵀβ_old + (yᵢ-pᵢ)/wᵢ  (working response)                     │
    │  • pᵢ = sigmoid(xᵢᵀβ_old)  (current probability estimate)              │
    │                                                                          │
    │  This is equivalent to:                                                  │
    │                                                                          │
    │      min_β ||W^½(Xβ - z)||²                                             │
    │                                                                          │
    │  With normal equations:                                                  │
    │                                                                          │
    │      (XᵀWX)β = XᵀWz                                                    │
    │                                                                          │
    │  With ridge regularization:                                              │
    │                                                                          │
    │      (XᵀWX + λI)β = XᵀWz                                               │
    │                                                                          │
    │  THE PROJECTION IS THE SAME - just weighted!                             │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional

# Import from our modules (once implemented)
# from .vectors import dot, norm, is_orthogonal
# from .matrices import gram_matrix, matrix_vector_multiply


def project_onto_vector(y: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Project vector y onto the line spanned by vector a.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  PROJECTION ONTO A LINE                                                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │                  (y · a)                                                 │
    │      proj_a(y) = ─────── × a                                            │
    │                  (a · a)                                                 │
    │                                                                          │
    │  VISUAL:                                                                 │
    │           ▲                                                              │
    │           │  y ●                                                         │
    │           │   ╱│                                                         │
    │           │  ╱ │ r = y - proj(y)                                        │
    │           │ ╱  │                                                         │
    │  ─────────●────┴───────────────►  line spanned by a                     │
    │         proj_a(y)                                                        │
    │                                                                          │
    │  The scalar coefficient:                                                 │
    │      c = (y·a)/(a·a)                                                    │
    │                                                                          │
    │  Is "how many a's" we need to get as close to y as possible.            │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        y: Vector to project, shape (n,)
        a: Vector defining the line, shape (n,)

    Returns:
        Projection of y onto span{a}, shape (n,)

    Raises:
        ValueError: If a is the zero vector

    Example:
        >>> y = np.array([3, 4])
        >>> a = np.array([1, 0])
        >>> project_onto_vector(y, a)
        array([3., 0.])  # Projects onto x-axis

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Compute dot_ya = np.dot(y, a)
    2. Compute dot_aa = np.dot(a, a)
    3. Check dot_aa > 1e-10, else raise ValueError
    4. Return (dot_ya / dot_aa) * a

    MAT223 Reference: Section 4.2.2
    """
    dot_ya = np.dot(y, a)
    dot_aa = np.dot(a, a)
    if dot_aa < 1e-10:
        raise ValueError("Cannot project onto the zero vector")
    return (dot_ya / dot_aa) * a


def project_onto_subspace(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Project vector y onto the column space of matrix X.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  PROJECTION ONTO col(X)                                                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Given orthonormal basis Q for col(X):                                   │
    │                                                                          │
    │      proj_{col(X)}(y) = Q(Qᵀy)                                          │
    │                                                                          │
    │  Or equivalently (via normal equations):                                 │
    │                                                                          │
    │      proj_{col(X)}(y) = X(XᵀX)⁻¹Xᵀy = Xβ̂                               │
    │                                                                          │
    │  VISUAL (col(X) is a plane):                                             │
    │                                                                          │
    │              y ●                                                         │
    │               ╲│                                                         │
    │                │ r = y - proj(y)                                         │
    │                │                                                         │
    │     ┌──────────●───────────┐                                            │
    │     │        proj(y)       │                                            │
    │     │                      │  col(X)                                     │
    │     │                      │                                            │
    │     └──────────────────────┘                                            │
    │                                                                          │
    │  KEY PROPERTY:                                                           │
    │      r = y - proj(y) is ORTHOGONAL to every column of X                │
    │      i.e., Xᵀr = 0                                                      │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        y: Vector to project, shape (n,)
        X: Matrix whose column space defines the subspace, shape (n, p)

    Returns:
        Projection of y onto col(X), shape (n,)

    Raises:
        ValueError: If dimensions don't match
        LinAlgError: If X is rank-deficient

    Example:
        >>> X = np.array([[1, 0], [0, 1], [0, 0]])  # xy-plane in 3D
        >>> y = np.array([1, 2, 3])
        >>> project_onto_subspace(y, X)
        array([1., 2., 0.])  # z-component removed

    IMPLEMENTATION STEPS (using QR - more stable):
    ───────────────────────────────────────────────
    1. Compute Q, R = np.linalg.qr(X)
    2. Q contains orthonormal basis for col(X)
    3. Projection = Q @ (Q.T @ y)

    ALTERNATIVE (less stable, for understanding):
    1. Compute beta = solve_normal_equations(X, y)
    2. Return X @ beta

    MAT223 Reference: Section 4.7.1
    """
    Q, R = np.linalg.qr(X)
    return Q @ (Q.T @ y)


def compute_residual(y: np.ndarray, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Compute the residual vector r = y - Xβ.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  RESIDUAL VECTOR                                                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      r = y - Xβ = y - ŷ                                                 │
    │                                                                          │
    │  The residual is the "error" or "unexplained part" of y.                 │
    │                                                                          │
    │  PROPERTIES OF LEAST SQUARES RESIDUAL:                                   │
    │  ──────────────────────────────────────                                  │
    │  When β = β̂ (optimal coefficients):                                     │
    │                                                                          │
    │  1. r ⊥ col(X):  Xᵀr = 0                                                │
    │  2. Σrᵢ = 0 if X has intercept column                                   │
    │  3. ||r||² is minimized                                                 │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  • Check orthogonality to verify solution                                │
    │  • Compute loss: ||r||²                                                 │
    │  • Residual plots for diagnostics                                        │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        y: Target vector, shape (n,)
        X: Design matrix, shape (n, p)
        beta: Coefficient vector, shape (p,)

    Returns:
        Residual vector, shape (n,)

    Example:
        >>> X = np.array([[1, 1], [1, 2], [1, 3]])
        >>> beta = np.array([1, 1])  # y = 1 + x
        >>> y = np.array([2.1, 3.0, 3.9])  # Noisy observations
        >>> compute_residual(y, X, beta)
        array([ 0.1,  0. , -0.1])

    IMPLEMENTATION:
    ────────────────
    return y - X @ beta
    """
    return y - X @ beta


def verify_orthogonality(X: np.ndarray, r: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Verify that residual r is orthogonal to all columns of X.

    This is the DEFINING PROPERTY of least squares solution!
    If Xᵀr ≠ 0, then β is NOT the optimal solution.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  ORTHOGONALITY VERIFICATION                                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  For least squares solution, we must have:                               │
    │                                                                          │
    │      Xᵀr = 0  ⟺  each column of X is orthogonal to r                   │
    │                                                                          │
    │      ┌     ┐   ┌───┐   ┌───┐                                            │
    │      │ col₁ᵀ│   │ r │   │ 0 │                                            │
    │      │ col₂ᵀ│ × │   │ = │ 0 │                                            │
    │      │  ⋮  │   │   │   │ ⋮ │                                            │
    │      │ colₚᵀ│   │   │   │ 0 │                                            │
    │      └     ┘   └───┘   └───┘                                            │
    │                                                                          │
    │  This is how we KNOW the projection is correct!                          │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix, shape (n, p)
        r: Residual vector, shape (n,)
        tol: Tolerance for numerical comparison

    Returns:
        True if r is orthogonal to col(X), False otherwise

    Example:
        >>> X = np.array([[1, 0], [0, 1], [1, 1]])
        >>> y = np.array([1, 2, 3])
        >>> beta = solve_normal_equations(X, y)
        >>> r = compute_residual(y, X, beta)
        >>> verify_orthogonality(X, r)
        True

    IMPLEMENTATION:
    ────────────────
    1. Compute Xtr = X.T @ r
    2. Check np.allclose(Xtr, 0, atol=tol)
    """
    Xtr = X.T @ r
    return bool(np.allclose(Xtr, 0, atol=tol))


def solve_normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve the normal equations XᵀXβ = Xᵀy directly.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  NORMAL EQUATIONS SOLVER (NAIVE)                                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      XᵀXβ = Xᵀy                                                         │
    │                                                                          │
    │  APPROACH:                                                               │
    │  ─────────                                                               │
    │  1. Compute A = XᵀX  (p×p Gram matrix)                                  │
    │  2. Compute b = Xᵀy  (p×1 cross-product)                                │
    │  3. Solve Aβ = b using np.linalg.solve                                  │
    │                                                                          │
    │  ⚠️ WARNING: NUMERICAL STABILITY ISSUES ⚠️                               │
    │  ─────────────────────────────────────────                               │
    │  • cond(XᵀX) = cond(X)² - condition number is SQUARED!                 │
    │  • If X is ill-conditioned, solution may be garbage                      │
    │  • Prefer QR factorization (solve_via_qr in qr.py)                      │
    │                                                                          │
    │  USE THIS ONLY FOR:                                                      │
    │  ───────────────────                                                     │
    │  • Learning/understanding the normal equations                           │
    │  • Well-conditioned problems (cond(X) < 100)                            │
    │  • Comparison/verification against QR solution                           │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix, shape (n, p), must have full column rank
        y: Target vector, shape (n,)

    Returns:
        Least squares solution β̂, shape (p,)

    Raises:
        ValueError: If dimensions don't match
        LinAlgError: If XᵀX is singular

    Example:
        >>> X = np.array([[1, 1], [1, 2], [1, 3]])
        >>> y = np.array([2, 3, 4])
        >>> solve_normal_equations(X, y)
        array([1., 1.])  # y = 1 + 1*x

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Validate X.shape[0] == y.shape[0]
    2. Compute XtX = X.T @ X
    3. Compute Xty = X.T @ y
    4. Solve using np.linalg.solve(XtX, Xty)
    5. Return solution

    MAT223 Reference: Section 4.8
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)


def solve_weighted_normal_equations(
    X: np.ndarray,
    z: np.ndarray,
    W: np.ndarray,
    lambda_: float = 0.0
) -> np.ndarray:
    """
    Solve weighted normal equations with optional ridge regularization.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  WEIGHTED LEAST SQUARES (for IRLS)                                       │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Problem: min_β ||W^½(Xβ - z)||²  + λ||β||²                             │
    │                                                                          │
    │  Normal equations:                                                       │
    │                                                                          │
    │      (XᵀWX + λI)β = XᵀWz                                               │
    │                                                                          │
    │  WHERE:                                                                  │
    │  ──────                                                                  │
    │  • W = diag(w₁, w₂, ..., wₙ) is diagonal weight matrix                  │
    │  • z is the "working response" vector                                    │
    │  • λ is ridge regularization parameter                                   │
    │                                                                          │
    │  IN IRLS FOR LOGISTIC REGRESSION:                                        │
    │  ──────────────────────────────────                                      │
    │  • wᵢ = pᵢ(1-pᵢ)  where pᵢ = sigmoid(xᵢᵀβ_old)                        │
    │  • zᵢ = xᵢᵀβ_old + (yᵢ - pᵢ)/wᵢ                                       │
    │                                                                          │
    │  The weights reflect the VARIANCE of each observation.                   │
    │  Higher variance → lower weight → less influence on fit.                 │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix, shape (n, p)
        z: Working response, shape (n,)
        W: Weight matrix, shape (n, n) diagonal, or (n,) vector of weights
        lambda_: Ridge regularization parameter (≥ 0)

    Returns:
        Solution β, shape (p,)

    Raises:
        ValueError: If dimensions don't match or lambda_ < 0

    Example:
        >>> X = np.array([[1, 1], [1, 2], [1, 3]])
        >>> z = np.array([2, 3, 4])
        >>> W = np.array([1, 1, 0.5])  # Less weight on third observation
        >>> solve_weighted_normal_equations(X, z, W, lambda_=0.01)

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. If W is 1D, convert to diagonal matrix: W_diag = np.diag(W)
    2. Compute XtWX = X.T @ W_diag @ X
    3. Add regularization: XtWX_reg = XtWX + lambda_ * np.eye(p)
    4. Compute XtWz = X.T @ W_diag @ z
    5. Solve and return: np.linalg.solve(XtWX_reg, XtWz)
    """
    if W.ndim == 1:
        W_diag = np.diag(W)
    else:
        W_diag = W
    p = X.shape[1]
    XtWX = X.T @ W_diag @ X
    XtWX_reg = XtWX + lambda_ * np.eye(p)
    XtWz = X.T @ W_diag @ z
    return np.linalg.solve(XtWX_reg, XtWz)


def compute_hat_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute the hat matrix H = X(XᵀX)⁻¹Xᵀ.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  HAT MATRIX (PROJECTION MATRIX)                                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      H = X(XᵀX)⁻¹Xᵀ                                                    │
    │                                                                          │
    │  The hat matrix "puts a hat" on y:                                       │
    │                                                                          │
    │      ŷ = Hy = X(XᵀX)⁻¹Xᵀy = Xβ̂                                        │
    │                                                                          │
    │  PROPERTIES:                                                             │
    │  ───────────                                                             │
    │  • H is symmetric: H = Hᵀ                                               │
    │  • H is idempotent: H² = H (projecting twice = projecting once)         │
    │  • Eigenvalues are 0 or 1                                                │
    │  • trace(H) = p (rank of X)                                              │
    │  • hᵢᵢ ∈ [1/n, 1] - diagonal elements are "leverage"                   │
    │                                                                          │
    │  LEVERAGE (DIAGONAL ELEMENTS):                                           │
    │  ──────────────────────────────                                          │
    │  • hᵢᵢ measures how much observation i "pulls" the fit                  │
    │  • High leverage points (hᵢᵢ near 1) have unusual x values              │
    │  • Useful for identifying influential observations                       │
    │                                                                          │
    │  ⚠️ WARNING:                                                             │
    │  ───────────                                                             │
    │  H is n×n - don't compute for large n! Use only for diagnostics.        │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix, shape (n, p), full column rank

    Returns:
        Hat matrix, shape (n, n)

    Example:
        >>> X = np.array([[1, 0], [0, 1], [1, 1]])
        >>> H = compute_hat_matrix(X)
        >>> y = np.array([1, 2, 3])
        >>> np.allclose(H @ y, project_onto_subspace(y, X))
        True

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Q, R = np.linalg.qr(X)
    2. H = Q @ Q.T (more stable than X @ inv(X.T @ X) @ X.T)
    """
    Q, R = np.linalg.qr(X)
    return Q @ Q.T


def compute_leverage(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage values (diagonal of hat matrix).

    Leverage measures how "unusual" each observation's x values are.
    High leverage points have outsized influence on the regression.

    Args:
        X: Design matrix, shape (n, p)

    Returns:
        Leverage values, shape (n,) with values in [1/n, 1]

    Example:
        >>> X = np.array([[1, 1], [1, 2], [1, 10]])  # Third point is outlier
        >>> compute_leverage(X)
        array([0.46, 0.33, 0.90])  # Third point has high leverage

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Q, R = np.linalg.qr(X)
    2. H_diag = np.sum(Q ** 2, axis=1)  # Efficient: don't form full H!
    3. Return H_diag
    """
    Q, R = np.linalg.qr(X)
    return np.sum(Q ** 2, axis=1)


def projection_matrix_onto_complement(X: np.ndarray) -> np.ndarray:
    """
    Compute the projection matrix onto the orthogonal complement of col(X).

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  PROJECTION ONTO ORTHOGONAL COMPLEMENT                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      M = I - H = I - X(XᵀX)⁻¹Xᵀ                                        │
    │                                                                          │
    │  M projects onto the ORTHOGONAL COMPLEMENT of col(X).                    │
    │                                                                          │
    │  My = y - Hy = y - ŷ = r (the residual!)                                │
    │                                                                          │
    │  PROPERTIES:                                                             │
    │  ───────────                                                             │
    │  • M is symmetric: M = Mᵀ                                               │
    │  • M is idempotent: M² = M                                              │
    │  • MX = 0 (columns of X map to zero)                                    │
    │  • trace(M) = n - p (dimension of orthogonal complement)                 │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix, shape (n, p)

    Returns:
        Annihilator matrix M, shape (n, n)

    IMPLEMENTATION:
    ────────────────
    1. Compute H = compute_hat_matrix(X)
    2. Return np.eye(n) - H
    """
    n = X.shape[0]
    H = compute_hat_matrix(X)
    return np.eye(n) - H


def sum_of_squared_residuals(y: np.ndarray, X: np.ndarray, beta: np.ndarray) -> float:
    """
    Compute the sum of squared residuals (SSR / RSS).

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  SUM OF SQUARED RESIDUALS                                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      SSR = Σᵢ (yᵢ - x̂ᵢ)² = ||y - Xβ||² = rᵀr                          │
    │                                                                          │
    │  This is the LOSS FUNCTION we minimize in least squares!                 │
    │                                                                          │
    │  RELATED QUANTITIES:                                                     │
    │  ────────────────────                                                    │
    │  • SST = Σ(yᵢ - ȳ)²  (total sum of squares)                            │
    │  • SSR = Σ(yᵢ - ŷᵢ)² (residual sum of squares) ← this function        │
    │  • SSE = Σ(ŷᵢ - ȳ)²  (explained sum of squares)                        │
    │  • SST = SSR + SSE                                                       │
    │  • R² = 1 - SSR/SST                                                     │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        y: Target vector, shape (n,)
        X: Design matrix, shape (n, p)
        beta: Coefficient vector, shape (p,)

    Returns:
        Sum of squared residuals (scalar ≥ 0)

    Example:
        >>> X = np.array([[1, 1], [1, 2], [1, 3]])
        >>> beta = np.array([1, 1])
        >>> y = np.array([2.1, 3.0, 3.9])
        >>> sum_of_squared_residuals(y, X, beta)
        0.02

    IMPLEMENTATION:
    ────────────────
    r = y - X @ beta
    return float(r @ r)  # or np.sum(r**2)
    """
    r = y - X @ beta
    return float(r @ r)


def r_squared(y: np.ndarray, X: np.ndarray, beta: np.ndarray) -> float:
    """
    Compute the coefficient of determination R².

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  R-SQUARED                                                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      R² = 1 - SSR/SST = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²                  │
    │                                                                          │
    │  INTERPRETATION:                                                         │
    │  ────────────────                                                        │
    │  • R² = 0: Model explains nothing (ŷ = ȳ for all)                       │
    │  • R² = 1: Model explains everything (ŷ = y exactly)                    │
    │  • R² = 0.7: Model explains 70% of variance in y                        │
    │                                                                          │
    │  ⚠️ WARNING FOR CLASSIFICATION:                                          │
    │  ───────────────────────────────                                         │
    │  R² is for REGRESSION (continuous y).                                    │
    │  For binary classification, use:                                         │
    │  • AUC-ROC                                                               │
    │  • Brier score                                                           │
    │  • Calibration error                                                     │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        y: Target vector, shape (n,)
        X: Design matrix, shape (n, p)
        beta: Coefficient vector, shape (p,)

    Returns:
        R-squared value in [0, 1] (can be negative for very bad models)

    IMPLEMENTATION:
    ────────────────
    1. Compute SSR = sum_of_squared_residuals(y, X, beta)
    2. Compute SST = np.sum((y - np.mean(y))**2)
    3. Return 1 - SSR/SST
    """
    ssr = sum_of_squared_residuals(y, X, beta)
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ssr / sst


# =============================================================================
#                           TODO LIST
# =============================================================================
"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TODO                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  HIGH PRIORITY (core least squares):                                         │
│  ────────────────────────────────────                                        │
│  [ ] Implement project_onto_vector() - 1D projection                        │
│  [ ] Implement project_onto_subspace() - multi-dimensional projection       │
│  [ ] Implement compute_residual() - for loss computation                    │
│  [ ] Implement verify_orthogonality() - debugging/verification              │
│  [ ] Implement solve_normal_equations() - naive least squares               │
│                                                                              │
│  HIGH PRIORITY (for IRLS):                                                   │
│  ─────────────────────────                                                   │
│  [ ] Implement solve_weighted_normal_equations() - weighted LS              │
│                                                                              │
│  MEDIUM PRIORITY (diagnostics):                                              │
│  ──────────────────────────────                                              │
│  [ ] Implement sum_of_squared_residuals() - loss value                      │
│  [ ] Implement r_squared() - model quality metric                           │
│  [ ] Implement compute_leverage() - influential point detection             │
│                                                                              │
│  LOW PRIORITY (specialized):                                                 │
│  ───────────────────────────                                                 │
│  [ ] Implement compute_hat_matrix() - full projection matrix               │
│  [ ] Implement projection_matrix_onto_complement() - for residuals          │
│                                                                              │
│  TESTING:                                                                    │
│  ────────                                                                    │
│  [ ] Write tests verifying Xᵀr = 0 for all solutions                       │
│  [ ] Compare solutions against np.linalg.lstsq                              │
│  [ ] Test with ill-conditioned matrices (should fail gracefully)            │
│  [ ] Test weighted least squares against sklearn                             │
│                                                                              │
│  AFTER COMPLETING qr.py:                                                     │
│  ─────────────────────────                                                   │
│  [ ] Update project_onto_subspace() to use QR                               │
│  [ ] Add solve_via_qr() as preferred solver                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    print("Projections and Least Squares Module")
    print("=" * 50)
    print("Implement the functions above and then run tests!")
    print()
    print("KEY VERIFICATION after implementation:")
    print("  For any least squares solution β̂:")
    print("  >>> r = y - X @ beta_hat")
    print("  >>> print(X.T @ r)  # Should be ≈ [0, 0, ..., 0]")
