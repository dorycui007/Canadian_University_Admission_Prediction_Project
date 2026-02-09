"""
QR Factorization Module
========================

This module implements QR factorization, the PREFERRED method for solving
least squares problems due to its superior numerical stability.

MAT223 REFERENCE: Sections 4.7 (Orthonormal Bases), 4.7.2 (Gram-Schmidt)

================================================================================
                        WHY QR INSTEAD OF NORMAL EQUATIONS?
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  THE STABILITY PROBLEM                                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  NORMAL EQUATIONS: XᵀXβ = Xᵀy                                          │
    │                                                                          │
    │  Problem: cond(XᵀX) = cond(X)²                                          │
    │                                                                          │
    │  EXAMPLE:                                                                │
    │  ────────                                                                │
    │  If cond(X) = 10³ (mildly ill-conditioned):                             │
    │     → cond(XᵀX) = 10⁶ (severely ill-conditioned!)                      │
    │     → Lose ~6 digits of precision                                        │
    │     → Solution may be garbage                                            │
    │                                                                          │
    │  QR FACTORIZATION: X = QR                                               │
    │                                                                          │
    │  Advantage: cond(R) = cond(X)  (condition number preserved!)            │
    │                                                                          │
    │  Using the same example:                                                 │
    │     → cond(R) = 10³ (not squared)                                       │
    │     → Only lose ~3 digits of precision                                   │
    │     → Solution is reliable                                               │
    │                                                                          │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │                                                                  │    │
    │  │  NEVER COMPUTE (XᵀX)⁻¹ DIRECTLY!                               │    │
    │  │  USE QR FACTORIZATION INSTEAD.                                   │    │
    │  │                                                                  │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                        WHAT IS QR FACTORIZATION?
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  QR FACTORIZATION                                                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Any m×n matrix X (with m ≥ n) can be factored as:                       │
    │                                                                          │
    │      X = QR                                                              │
    │                                                                          │
    │  WHERE:                                                                  │
    │  ──────                                                                  │
    │  • Q is m×n with ORTHONORMAL columns: QᵀQ = I                           │
    │  • R is n×n UPPER TRIANGULAR                                             │
    │                                                                          │
    │  VISUAL:                                                                 │
    │      ┌───────────┐     ┌───────────┐   ┌───────────┐                    │
    │      │           │     │           │   │ * * * * * │  ← upper triangular │
    │      │           │     │           │   │   * * * * │                     │
    │      │     X     │  =  │     Q     │ × │     * * * │                     │
    │      │           │     │ (orthonorm│   │       * * │                     │
    │      │           │     │  columns) │   │         * │                     │
    │      │           │     │           │   └───────────┘                    │
    │      └───────────┘     └───────────┘         R                          │
    │        (m × n)           (m × n)           (n × n)                      │
    │                                                                          │
    │  WHY ORTHONORMAL Q IS MAGICAL:                                           │
    │  ──────────────────────────────                                          │
    │  • QᵀQ = I  means Q is like a rotation (preserves lengths/angles)      │
    │  • Q⁻¹ = Qᵀ  (transpose IS the inverse - easy to compute!)             │
    │  • Projections are simple: proj_{col(Q)}(y) = QQᵀy                      │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                        SOLVING LEAST SQUARES VIA QR
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  LEAST SQUARES VIA QR                                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Problem: Find β minimizing ||Xβ - y||²                                 │
    │                                                                          │
    │  STEP 1: Factor X = QR                                                   │
    │  ─────────────────────                                                   │
    │      Q: (m×n) with QᵀQ = I                                              │
    │      R: (n×n) upper triangular                                           │
    │                                                                          │
    │  STEP 2: Transform the problem                                           │
    │  ─────────────────────────────                                           │
    │      ||Xβ - y||² = ||QRβ - y||²                                         │
    │                  = ||Rβ - Qᵀy||²    (because Q preserves lengths)       │
    │                                                                          │
    │  STEP 3: Solve Rβ = Qᵀy                                                 │
    │  ────────────────────────                                                │
    │      • Compute c = Qᵀy                                                  │
    │      • Solve Rβ = c by BACK SUBSTITUTION                                │
    │                                                                          │
    │  WHY BACK SUBSTITUTION IS EASY:                                          │
    │  ────────────────────────────────                                        │
    │  R is upper triangular:                                                  │
    │      ┌                     ┐   ┌    ┐     ┌    ┐                        │
    │      │ r₁₁ r₁₂ r₁₃ ... r₁ₙ│   │ β₁ │     │ c₁ │                        │
    │      │  0  r₂₂ r₂₃ ... r₂ₙ│   │ β₂ │     │ c₂ │                        │
    │      │  0   0  r₃₃ ... r₃ₙ│ × │ β₃ │  =  │ c₃ │                        │
    │      │  ⋮   ⋮   ⋮   ⋱   ⋮ │   │ ⋮  │     │ ⋮  │                        │
    │      │  0   0   0  ...rₙₙ │   │ βₙ │     │ cₙ │                        │
    │      └                     ┘   └    ┘     └    ┘                        │
    │                                                                          │
    │  Solve from bottom up:                                                   │
    │      βₙ = cₙ / rₙₙ                                                      │
    │      βₙ₋₁ = (cₙ₋₁ - rₙ₋₁,ₙβₙ) / rₙ₋₁,ₙ₋₁                             │
    │      ...and so on                                                        │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    HOUSEHOLDER REFLECTIONS
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  HOUSEHOLDER REFLECTION (used in QR)                                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  A Householder reflection is a matrix of the form:                       │
    │                                                                          │
    │      H = I - 2vvᵀ    where ||v|| = 1                                    │
    │                                                                          │
    │  PROPERTIES:                                                             │
    │  ───────────                                                             │
    │  • H is symmetric: H = Hᵀ                                               │
    │  • H is orthogonal: HᵀH = I                                             │
    │  • H is its own inverse: H² = I                                         │
    │  • H is a REFLECTION across the hyperplane perpendicular to v           │
    │                                                                          │
    │  VISUAL (2D):                                                            │
    │                                                                          │
    │           ▲                                                              │
    │           │      ● x                                                     │
    │           │     /                                                        │
    │           │    /                                                         │
    │    ───────┼───/────────  hyperplane ⊥ to v                              │
    │           │  /                                                           │
    │           │ /                                                            │
    │           │● Hx  (reflection of x)                                       │
    │           │                                                              │
    │           │                                                              │
    │                                                                          │
    │  USE IN QR:                                                              │
    │  ──────────                                                              │
    │  Choose v to reflect a column of X onto a multiple of e₁ = [1,0,...,0]  │
    │  This zeros out all entries below the diagonal in that column!          │
    │                                                                          │
    │      x = [*, *, *, *]ᵀ  →  Hx = [||x||, 0, 0, 0]ᵀ                       │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    ALGORITHM OVERVIEW
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  HOUSEHOLDER QR ALGORITHM                                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Start with X (m×n matrix)                                               │
    │                                                                          │
    │  ITERATION j = 1, 2, ..., n:                                             │
    │  ─────────────────────────────                                           │
    │  1. Look at column j, from row j down: x = X[j:, j]                     │
    │                                                                          │
    │  2. Find Householder vector v that reflects x to ||x||e₁:               │
    │         v = x + sign(x₁)||x||e₁                                         │
    │         v = v / ||v||                                                   │
    │                                                                          │
    │  3. Apply reflection to X[j:, j:]:                                      │
    │         X[j:, j:] = X[j:, j:] - 2v(vᵀX[j:, j:])                        │
    │                                                                          │
    │  4. Accumulate Q:                                                        │
    │         Q = Q × Hⱼ                                                      │
    │                                                                          │
    │  After n iterations:                                                     │
    │  • X has become R (upper triangular)                                     │
    │  • Q = H₁H₂...Hₙ  (product of reflections)                              │
    │                                                                          │
    │  VISUAL PROGRESS:                                                        │
    │                                                                          │
    │  Initial X:       After H₁:       After H₂:       After H₃:            │
    │  ┌─────────┐      ┌─────────┐     ┌─────────┐     ┌─────────┐          │
    │  │ * * * * │      │ * * * * │     │ * * * * │     │ * * * * │          │
    │  │ * * * * │  →   │ 0 * * * │  →  │ 0 * * * │  →  │ 0 * * * │          │
    │  │ * * * * │      │ 0 * * * │     │ 0 0 * * │     │ 0 0 * * │          │
    │  │ * * * * │      │ 0 * * * │     │ 0 0 * * │     │ 0 0 0 * │          │
    │  └─────────┘      └─────────┘     └─────────┘     └─────────┘          │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


def householder_vector(x: np.ndarray) -> np.ndarray:
    """
    Compute the Householder vector v such that Hx = ||x||e₁.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  HOUSEHOLDER VECTOR                                                      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Given x, find v such that:                                              │
    │                                                                          │
    │      Hx = (I - 2vvᵀ)x = ±||x||e₁                                        │
    │                                                                          │
    │  FORMULA:                                                                │
    │  ────────                                                                │
    │      v = x + sign(x₁)||x||e₁                                            │
    │      v = v / ||v||                                                      │
    │                                                                          │
    │  NOTE: We use sign(x₁) for numerical stability:                         │
    │  • Avoids cancellation when x₁ ≈ -||x||                                 │
    │  • Convention: sign(0) = 1                                               │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        x: Vector to reflect, shape (k,)

    Returns:
        Normalized Householder vector v, shape (k,)

    Example:
        >>> x = np.array([3.0, 4.0])
        >>> v = householder_vector(x)
        >>> H = np.eye(2) - 2 * np.outer(v, v)
        >>> H @ x  # Should be [±5, 0]

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Compute norm_x = np.linalg.norm(x)
    2. If norm_x < 1e-10, return x / norm_x (or handle zero case)
    3. sign_x1 = 1 if x[0] >= 0 else -1
    4. Create e1 = [1, 0, 0, ...] with same shape as x
    5. v = x + sign_x1 * norm_x * e1
    6. Return v / np.linalg.norm(v)

    MAT223 Reference: Section 4.7
    """
    x = np.asarray(x, dtype=float)
    norm_x = np.linalg.norm(x)
    if norm_x < 1e-10:
        # Zero vector edge case: return a unit vector e1
        e1 = np.zeros_like(x)
        e1[0] = 1.0
        return e1
    sign_x1 = 1.0 if x[0] >= 0 else -1.0
    e1 = np.zeros_like(x)
    e1[0] = 1.0
    v = x + sign_x1 * norm_x * e1
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-10:
        # This can happen if x is already aligned with -e1
        return e1
    return v / v_norm


def apply_householder(H_v: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Apply Householder reflection H = I - 2vvᵀ to matrix A efficiently.

    Instead of forming H explicitly (expensive!), we compute:

        HA = A - 2v(vᵀA)

    This is O(mn) instead of O(m²n) for explicit matrix multiply.

    Args:
        H_v: Householder vector v, shape (m,)
        A: Matrix to transform, shape (m, n)

    Returns:
        Transformed matrix HA, shape (m, n)

    IMPLEMENTATION:
    ────────────────
    vTA = v.T @ A              # (n,) - project columns onto v
    return A - 2 * np.outer(v, vTA)  # subtract 2× projection
    """
    vTA = H_v @ A  # (n,) - dot product of v with each column of A
    return A - 2.0 * np.outer(H_v, vTA)


def qr_householder(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute QR factorization using Householder reflections.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  HOUSEHOLDER QR FACTORIZATION                                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Factor A = QR where:                                                    │
    │  • Q is m×m orthogonal (QᵀQ = I)                                        │
    │  • R is m×n upper triangular                                             │
    │                                                                          │
    │  ALGORITHM:                                                              │
    │  ──────────                                                              │
    │  for j = 0 to min(m,n)-1:                                                │
    │      x = R[j:, j]                # Column j from row j down             │
    │      v = householder_vector(x)   # Find reflection vector               │
    │      R[j:, j:] -= 2v(vᵀR[j:, j:]) # Apply to remaining submatrix       │
    │      Q[:, j:] -= 2(Q[:, j:]v)vᵀ   # Accumulate Q                        │
    │                                                                          │
    │  COST:                                                                   │
    │  ─────                                                                   │
    │  • O(2mn² - 2n³/3) flops for m > n                                      │
    │  • About 2x the cost of computing XᵀX, but much more stable!           │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        A: Matrix to factor, shape (m, n), m ≥ n

    Returns:
        Tuple (Q, R) where:
        - Q: Orthogonal matrix, shape (m, m)
        - R: Upper triangular, shape (m, n)

    Example:
        >>> A = np.array([[1, 1], [1, 2], [1, 3]])
        >>> Q, R = qr_householder(A)
        >>> np.allclose(Q @ R, A)  # Verify factorization
        True
        >>> np.allclose(Q.T @ Q, np.eye(3))  # Q is orthogonal
        True

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. m, n = A.shape
    2. Q = np.eye(m)  # Start with identity
    3. R = A.copy().astype(float)  # Will transform into R
    4. for j in range(min(m, n)):
           x = R[j:, j]
           v = householder_vector(x)
           # Apply to R[j:, j:]
           R[j:, j:] = apply_householder(v, R[j:, j:].T).T  # or explicit formula
           # Apply to Q[:, j:]
           ...update Q...
    5. Return Q, R

    NOTE: The above is conceptual. See implementation hints for efficiency.

    MAT223 Reference: Section 4.7
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    for j in range(min(m, n)):
        x = R[j:, j].copy()
        v = householder_vector(x)
        # Apply H to R[j:, j:]
        R[j:, j:] = apply_householder(v, R[j:, j:])
        # Apply H to Q[:, j:]  (Q = Q @ H_j, so Q[:, j:] = Q[:, j:] @ H)
        # H is symmetric, so Q @ H = Q @ (I - 2vv^T)
        # Q[:, j:] = Q[:, j:] - 2 * (Q[:, j:] @ v) @ v^T
        Qv = Q[:, j:] @ v  # shape (m,)
        Q[:, j:] = Q[:, j:] - 2.0 * np.outer(Qv, v)
    return Q, R


def qr_reduced(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the REDUCED (thin) QR factorization.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  REDUCED VS FULL QR                                                      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  FULL QR: A = QR                                                         │
    │    • Q: m×m orthogonal                                                   │
    │    • R: m×n upper triangular                                             │
    │                                                                          │
    │      ┌─────────┐   ┌───────────────┐   ┌─────┐                          │
    │      │         │   │               │   │ R₁  │ ← n×n upper triangular   │
    │      │    A    │ = │   Q₁  │  Q₂   │ × │─────│                          │
    │      │         │   │               │   │  0  │ ← (m-n)×n zeros          │
    │      └─────────┘   └───────────────┘   └─────┘                          │
    │        (m×n)         (m×n)  (m×m-n)      (m×n)                          │
    │                                                                          │
    │  REDUCED QR: A = Q₁R₁                                                   │
    │    • Q₁: m×n with orthonormal columns                                   │
    │    • R₁: n×n upper triangular                                            │
    │                                                                          │
    │      ┌─────────┐   ┌─────────┐   ┌─────────┐                            │
    │      │         │   │         │   │         │                            │
    │      │    A    │ = │   Q₁    │ × │   R₁    │                            │
    │      │         │   │         │   │         │                            │
    │      └─────────┘   └─────────┘   └─────────┘                            │
    │        (m×n)         (m×n)         (n×n)                                │
    │                                                                          │
    │  For least squares, we only need Q₁ and R₁!                             │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        A: Matrix to factor, shape (m, n), m ≥ n

    Returns:
        Tuple (Q1, R1) where:
        - Q1: Matrix with orthonormal columns, shape (m, n)
        - R1: Upper triangular, shape (n, n)

    Example:
        >>> A = np.array([[1, 1], [1, 2], [1, 3]])
        >>> Q1, R1 = qr_reduced(A)
        >>> Q1.shape  # (3, 2)
        >>> R1.shape  # (2, 2)
        >>> np.allclose(Q1 @ R1, A)
        True

    IMPLEMENTATION (using numpy):
    ──────────────────────────────
    Q, R = np.linalg.qr(A, mode='reduced')
    return Q, R
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    k = min(m, n)
    Q_full, R_full = qr_householder(A)
    Q1 = Q_full[:, :k]
    R1 = R_full[:k, :]
    return Q1, R1


def back_substitution(R: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve upper triangular system Rx = b by back substitution.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  BACK SUBSTITUTION                                                       │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Given upper triangular R and vector b, solve Rx = b:                    │
    │                                                                          │
    │      ┌                     ┐   ┌    ┐     ┌    ┐                        │
    │      │ r₁₁ r₁₂ r₁₃ ... r₁ₙ│   │ x₁ │     │ b₁ │                        │
    │      │  0  r₂₂ r₂₃ ... r₂ₙ│   │ x₂ │     │ b₂ │                        │
    │      │  0   0  r₃₃ ... r₃ₙ│ × │ x₃ │  =  │ b₃ │                        │
    │      │  ⋮   ⋮   ⋮   ⋱   ⋮ │   │ ⋮  │     │ ⋮  │                        │
    │      │  0   0   0  ... rₙₙ│   │ xₙ │     │ bₙ │                        │
    │      └                     ┘   └    ┘     └    ┘                        │
    │                                                                          │
    │  ALGORITHM (solve from bottom up):                                       │
    │  ──────────────────────────────────                                      │
    │      xₙ = bₙ / rₙₙ                                                      │
    │      xₙ₋₁ = (bₙ₋₁ - rₙ₋₁,ₙ·xₙ) / rₙ₋₁,ₙ₋₁                            │
    │      xᵢ = (bᵢ - Σⱼ>ᵢ rᵢⱼ·xⱼ) / rᵢᵢ                                    │
    │                                                                          │
    │  COST: O(n²) operations                                                  │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        R: Upper triangular matrix, shape (n, n)
        b: Right-hand side, shape (n,)

    Returns:
        Solution x, shape (n,)

    Raises:
        ValueError: If R has zero on diagonal (singular)

    Example:
        >>> R = np.array([[2, 1], [0, 3]])
        >>> b = np.array([5, 6])
        >>> back_substitution(R, b)
        array([1., 2.])  # x = [1, 2] because 2*1 + 1*2 = 4... wait, check

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. n = len(b)
    2. x = np.zeros(n)
    3. for i in range(n-1, -1, -1):  # n-1 down to 0
           if abs(R[i, i]) < 1e-10:
               raise ValueError("Singular matrix")
           x[i] = (b[i] - R[i, i+1:] @ x[i+1:]) / R[i, i]
    4. return x
    """
    R = np.asarray(R, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) < 1e-10:
            raise ValueError("Singular matrix: zero diagonal element in R")
        x[i] = (b[i] - R[i, i+1:] @ x[i+1:]) / R[i, i]
    return x


def solve_via_qr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve least squares problem using QR factorization.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  LEAST SQUARES VIA QR                                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Problem: min_β ||Xβ - y||²                                             │
    │                                                                          │
    │  SOLUTION:                                                               │
    │  ─────────                                                               │
    │  1. Compute X = Q₁R₁ (reduced QR)                                        │
    │  2. Compute c = Q₁ᵀy                                                    │
    │  3. Solve R₁β = c by back substitution                                  │
    │                                                                          │
    │  WHY THIS WORKS:                                                         │
    │  ────────────────                                                        │
    │  ||Xβ - y||² = ||Q₁R₁β - y||²                                          │
    │              = ||R₁β - Q₁ᵀy||² + ||y - Q₁Q₁ᵀy||²                       │
    │                ↑                  ↑                                      │
    │                minimize this      fixed (doesn't depend on β)            │
    │                                                                          │
    │  Minimum achieved when R₁β = Q₁ᵀy                                       │
    │                                                                          │
    │  NUMERICAL ADVANTAGE:                                                    │
    │  ─────────────────────                                                   │
    │  • cond(R) = cond(X) (not squared!)                                     │
    │  • No need to form XᵀX                                                  │
    │  • Stable even for ill-conditioned X                                     │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix, shape (n, p), must have full column rank
        y: Target vector, shape (n,)

    Returns:
        Least squares solution β̂, shape (p,)

    Raises:
        ValueError: If X is rank-deficient

    Example:
        >>> X = np.array([[1, 1], [1, 2], [1, 3]])
        >>> y = np.array([2, 3, 4])
        >>> solve_via_qr(X, y)
        array([1., 1.])  # y = 1 + 1*x

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Q, R = qr_reduced(X)  # or np.linalg.qr(X, mode='reduced')
    2. c = Q.T @ y
    3. beta = back_substitution(R, c)
    4. return beta

    ALTERNATIVE (using numpy directly):
        return np.linalg.lstsq(X, y, rcond=None)[0]

    MAT223 Reference: Section 4.8
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Q, R = qr_reduced(X)
    c = Q.T @ y
    beta = back_substitution(R, c)
    return beta


def solve_weighted_via_qr(
    X: np.ndarray,
    z: np.ndarray,
    weights: np.ndarray,
    lambda_: float = 0.0
) -> np.ndarray:
    """
    Solve weighted least squares with ridge regularization using QR.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  WEIGHTED LEAST SQUARES VIA QR                                           │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Problem: min_β ||W^½(Xβ - z)||² + λ||β||²                              │
    │                                                                          │
    │  APPROACH (standard trick):                                              │
    │  ──────────────────────────                                              │
    │  Transform to ordinary least squares:                                    │
    │                                                                          │
    │      X̃ = W^½ X                                                          │
    │      z̃ = W^½ z                                                          │
    │                                                                          │
    │  For ridge, augment:                                                     │
    │                                                                          │
    │      ┌───────┐        ┌───┐                                             │
    │      │   X̃   │   β    │ z̃ │                                             │
    │      │───────│   →    │───│                                             │
    │      │ √λ I  │        │ 0 │                                             │
    │      └───────┘        └───┘                                             │
    │                                                                          │
    │  Then solve via standard QR!                                             │
    │                                                                          │
    │  FOR IRLS:                                                               │
    │  ─────────                                                               │
    │  This is called at each IRLS iteration with:                             │
    │  • weights = p(1-p)                                                      │
    │  • z = working response                                                  │
    │  • lambda_ = regularization                                              │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix, shape (n, p)
        z: Working response, shape (n,)
        weights: Diagonal weights, shape (n,)
        lambda_: Ridge parameter (≥ 0)

    Returns:
        Solution β, shape (p,)

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. sqrt_w = np.sqrt(weights)
    2. X_tilde = sqrt_w[:, None] * X  # Weight rows of X
    3. z_tilde = sqrt_w * z           # Weight z
    4. If lambda_ > 0:
           # Augment with ridge
           X_aug = np.vstack([X_tilde, np.sqrt(lambda_) * np.eye(p)])
           z_aug = np.concatenate([z_tilde, np.zeros(p)])
       Else:
           X_aug, z_aug = X_tilde, z_tilde
    5. return solve_via_qr(X_aug, z_aug)
    """
    X = np.asarray(X, dtype=float)
    z = np.asarray(z, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n, p = X.shape
    sqrt_w = np.sqrt(weights)
    X_tilde = sqrt_w[:, None] * X
    z_tilde = sqrt_w * z
    if lambda_ > 0:
        X_aug = np.vstack([X_tilde, np.sqrt(lambda_) * np.eye(p)])
        z_aug = np.concatenate([z_tilde, np.zeros(p)])
    else:
        X_aug = X_tilde
        z_aug = z_tilde
    return solve_via_qr(X_aug, z_aug)


def check_qr_factorization(A: np.ndarray, Q: np.ndarray, R: np.ndarray,
                           tol: float = 1e-10) -> dict:
    """
    Verify properties of a QR factorization.

    This is a TESTING function to verify your implementation.

    Args:
        A: Original matrix
        Q: Orthogonal factor
        R: Upper triangular factor
        tol: Tolerance for numerical checks

    Returns:
        Dictionary with verification results:
        - 'factorization_correct': bool - A ≈ QR
        - 'Q_orthogonal': bool - QᵀQ ≈ I
        - 'R_upper_triangular': bool - R is upper triangular
        - 'errors': dict with specific error magnitudes

    IMPLEMENTATION:
    ────────────────
    1. Check ||A - QR|| < tol
    2. Check ||QᵀQ - I|| < tol
    3. Check all entries below diagonal of R are < tol
    4. Return results dict
    """
    A = np.asarray(A, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)

    # Check factorization: A ≈ QR
    reconstruction_error = np.linalg.norm(A - Q @ R)
    factorization_correct = reconstruction_error < tol

    # Check Q orthogonality: Q^T Q ≈ I
    m = Q.shape[1]
    orthogonality_error = np.linalg.norm(Q.T @ Q - np.eye(m))
    q_orthogonal = orthogonality_error < tol

    # Check R is upper triangular
    r_upper_triangular = True
    for i in range(R.shape[0]):
        for j in range(min(i, R.shape[1])):
            if abs(R[i, j]) > tol:
                r_upper_triangular = False
                break
        if not r_upper_triangular:
            break

    return {
        'factorization_correct': factorization_correct,
        'Q_orthogonal': q_orthogonal,
        'R_upper_triangular': r_upper_triangular,
        'errors': {
            'reconstruction_error': reconstruction_error,
            'orthogonality_error': orthogonality_error,
        }
    }


# =============================================================================
#                           TODO LIST
# =============================================================================
"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TODO                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  HIGH PRIORITY (core QR solver):                                             │
│  ────────────────────────────────                                            │
│  [ ] Implement householder_vector() - reflection computation                │
│  [ ] Implement back_substitution() - solve triangular system                │
│  [ ] Implement solve_via_qr() - main least squares solver                   │
│                                                                              │
│  MEDIUM PRIORITY (full implementation):                                      │
│  ──────────────────────────────────────                                      │
│  [ ] Implement apply_householder() - efficient matrix update                │
│  [ ] Implement qr_householder() - from-scratch QR                           │
│  [ ] Implement qr_reduced() - thin QR wrapper                               │
│  [ ] Implement solve_weighted_via_qr() - for IRLS                           │
│                                                                              │
│  LOW PRIORITY (verification):                                                │
│  ─────────────────────────────                                               │
│  [ ] Implement check_qr_factorization() - testing helper                    │
│                                                                              │
│  TESTING:                                                                    │
│  ────────                                                                    │
│  [ ] Compare solve_via_qr with np.linalg.lstsq                              │
│  [ ] Compare qr_householder with np.linalg.qr                               │
│  [ ] Test on ill-conditioned matrices                                        │
│  [ ] Verify numerical stability vs normal equations                          │
│                                                                              │
│  OPTIONAL ENHANCEMENTS:                                                      │
│  ───────────────────────                                                     │
│  [ ] Implement Givens rotations (alternative to Householder)                │
│  [ ] Add column pivoting for rank-deficient matrices                        │
│  [ ] GPU acceleration via PyTorch                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    print("QR Factorization Module")
    print("=" * 50)
    print("Implement the functions above and then run tests!")
    print()
    print("Key verification after implementation:")
    print("  >>> X = np.random.randn(100, 5)")
    print("  >>> y = np.random.randn(100)")
    print("  >>> beta_qr = solve_via_qr(X, y)")
    print("  >>> beta_np = np.linalg.lstsq(X, y, rcond=None)[0]")
    print("  >>> np.allclose(beta_qr, beta_np)  # Should be True!")
