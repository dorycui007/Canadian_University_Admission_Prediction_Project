"""
Matrix Operations Module for University Admissions Prediction System
=====================================================================

This module implements fundamental matrix operations from scratch, building
on vector operations to enable design matrix manipulation and linear algebra
computations critical for ML models.

MAT223 REFERENCE: Sections 1.3-1.4 (Matrix Operations), 4.6 (Rank)

================================================================================
                        SYSTEM ARCHITECTURE CONTEXT
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    WHERE THIS MODULE FITS                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │                         vectors.py                                       │
    │                             │                                            │
    │                             ▼                                            │
    │   CSV Data ──► MongoDB ──► [THIS MODULE] ──► projections.py             │
    │                            matrices.py            │                      │
    │                                │                  │                      │
    │                                ▼                  ▼                      │
    │                           ┌─────────────┐    ┌─────────────┐            │
    │                           │   Matrix    │    │    QR       │            │
    │                           │ Operations: │    │ Factorization│           │
    │                           │ • multiply  │    │    (qr.py)   │           │
    │                           │ • rank      │    └─────────────┘            │
    │                           │ • condition │                                │
    │                           └─────────────┘                                │
    │                                  │                                       │
    │                                  ▼                                       │
    │                           Design Matrix X                                │
    │                        (features/design_matrix.py)                       │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                        MATRIX FUNDAMENTALS
================================================================================

A matrix is a 2D array of numbers with m rows and n columns:

                    ┌                     ┐
                    │ a₁₁  a₁₂  ...  a₁ₙ │
                A = │ a₂₁  a₂₂  ...  a₂ₙ │  ∈ ℝᵐˣⁿ
                    │  ⋮    ⋮    ⋱    ⋮  │
                    │ aₘ₁  aₘ₂  ...  aₘₙ │
                    └                     ┘

    DESIGN MATRIX in this project:
    ───────────────────────────────

    ┌──────────────────────────────────────────────────────────────────────────┐
    │                        DESIGN MATRIX X                                    │
    │                     (n_samples × n_features)                              │
    ├──────────────────────────────────────────────────────────────────────────┤
    │                                                                           │
    │     Features:  avg   is_UofT  is_Waterloo  is_CS  is_Eng  ...            │
    │                 ↓       ↓         ↓          ↓       ↓                   │
    │              ┌─────┬───────┬───────────┬───────┬───────┬─────┐           │
    │   Student 1  │ 92.5│   1   │     0     │   1   │   0   │ ... │           │
    │   Student 2  │ 87.3│   0   │     1     │   0   │   1   │ ... │           │
    │   Student 3  │ 95.1│   1   │     0     │   1   │   0   │ ... │           │
    │      ⋮       │  ⋮  │   ⋮   │     ⋮     │   ⋮   │   ⋮   │  ⋮  │           │
    │   Student n  │ 88.9│   0   │     0     │   0   │   1   │ ... │           │
    │              └─────┴───────┴───────────┴───────┴───────┴─────┘           │
    │                                                                           │
    │   For this project: n ≈ 4,900 students, p ≈ 250 features                 │
    │                                                                           │
    └──────────────────────────────────────────────────────────────────────────┘

================================================================================
                        MATRIX MULTIPLICATION
================================================================================

Matrix multiplication Xβ is how we compute predictions:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  MATRIX-VECTOR MULTIPLICATION: Xβ                                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │         X            ×         β          =        ŷ                    │
    │    (n × p)               (p × 1)              (n × 1)                   │
    │                                                                          │
    │  ┌─────────────┐       ┌─────┐           ┌─────┐                        │
    │  │ x₁₁ ... x₁ₚ│       │ β₁  │           │ ŷ₁  │  ← x₁ᵀβ                │
    │  │ x₂₁ ... x₂ₚ│   ×   │ β₂  │     =     │ ŷ₂  │  ← x₂ᵀβ                │
    │  │  ⋮       ⋮ │       │  ⋮  │           │  ⋮  │                         │
    │  │ xₙ₁ ... xₙₚ│       │ βₚ  │           │ ŷₙ  │  ← xₙᵀβ                │
    │  └─────────────┘       └─────┘           └─────┘                        │
    │                                                                          │
    │  Each prediction ŷᵢ = xᵢᵀβ = Σⱼ xᵢⱼβⱼ (dot product of row i with β)    │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  • Linear predictions: logit = Xβ                                        │
    │  • Gradient computation: ∇L = Xᵀ(p - y)                                 │
    │  • Hessian: H = XᵀWX (W is diagonal weight matrix)                      │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                        RANK AND LINEAR INDEPENDENCE
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  COLUMN RANK                                                             │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  rank(X) = number of linearly independent columns                        │
    │          = dimension of column space col(X)                              │
    │                                                                          │
    │  FULL COLUMN RANK: rank(X) = p (number of columns)                       │
    │  ─────────────────                                                       │
    │  • XᵀX is invertible                                                    │
    │  • Least squares has unique solution                                     │
    │  • This is what we NEED for our models                                   │
    │                                                                          │
    │  RANK DEFICIENT: rank(X) < p                                             │
    │  ───────────────                                                         │
    │  • Some columns are linear combinations of others                        │
    │  • XᵀX is singular (no inverse)                                         │
    │  • Infinitely many least squares solutions                               │
    │                                                                          │
    │  COMMON CAUSE IN THIS PROJECT:                                           │
    │  ──────────────────────────────                                          │
    │  One-hot encoding with intercept causes rank deficiency!                 │
    │                                                                          │
    │      intercept + is_UofT + is_Waterloo + is_McGill = 1 (always)         │
    │                                                                          │
    │  Solution: DROP ONE REFERENCE CATEGORY per group                         │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                        CONDITION NUMBER
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  CONDITION NUMBER                                                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      cond(A) = ||A|| · ||A⁻¹|| = σₘₐₓ / σₘᵢₙ                           │
    │                                                                          │
    │  WHERE:                                                                  │
    │  • σₘₐₓ = largest singular value of A                                   │
    │  • σₘᵢₙ = smallest singular value of A                                  │
    │                                                                          │
    │  INTERPRETATION:                                                         │
    │  ────────────────                                                        │
    │  • cond(A) ≈ 1:       Well-conditioned (stable computations)            │
    │  • cond(A) ≈ 10⁶:    Poorly conditioned (6 digits of precision lost)   │
    │  • cond(A) = ∞:       Singular matrix (no inverse exists)               │
    │                                                                          │
    │  VISUAL: Ellipse Stretching                                              │
    │  ───────────────────────────                                             │
    │                                                                          │
    │      Well-conditioned (cond ≈ 1):     Ill-conditioned (cond >> 1):      │
    │                                                                          │
    │           ○                                    ────────────              │
    │          (circle-ish)                        (very stretched)            │
    │                                                                          │
    │  WHY IT MATTERS:                                                         │
    │  ────────────────                                                        │
    │  • High cond(XᵀX) → small input changes cause HUGE output changes      │
    │  • Normal equations SQUARE the condition number: cond(XᵀX) = cond(X)²  │
    │  • QR factorization preserves condition number: cond(R) = cond(X)       │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  • Monitor condition number of design matrix                             │
    │  • If cond > 10⁶, add regularization (ridge) or remove features        │
    │  • Ridge: cond(XᵀX + λI) ≤ (σₘₐₓ² + λ) / λ  (bounded!)                │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                        TRANSPOSE PROPERTIES
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  TRANSPOSE: Aᵀ                                                           │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      (Aᵀ)ᵢⱼ = Aⱼᵢ    (swap rows and columns)                           │
    │                                                                          │
    │  VISUAL:                                                                 │
    │  ───────                                                                 │
    │      ┌─────────┐         ┌─────────┐                                    │
    │      │ 1  2  3 │   →     │ 1  4  7 │                                    │
    │  A = │ 4  5  6 │   Aᵀ =  │ 2  5  8 │                                    │
    │      │ 7  8  9 │         │ 3  6  9 │                                    │
    │      └─────────┘         └─────────┘                                    │
    │        (3×3)               (3×3)                                        │
    │                                                                          │
    │  KEY PROPERTIES:                                                         │
    │  ───────────────                                                         │
    │  • (Aᵀ)ᵀ = A                                                            │
    │  • (AB)ᵀ = BᵀAᵀ              (reverse order!)                          │
    │  • (A + B)ᵀ = Aᵀ + Bᵀ                                                  │
    │  • (αA)ᵀ = αAᵀ                                                         │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  • XᵀX: Gram matrix (p×p) for normal equations                         │
    │  • Xᵀy: Cross-product for least squares                                 │
    │  • XᵀWX: Weighted Gram matrix for IRLS                                  │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                        SYMMETRIC MATRICES
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  SYMMETRIC MATRIX: A = Aᵀ                                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  A matrix is symmetric if it equals its transpose: aᵢⱼ = aⱼᵢ           │
    │                                                                          │
    │  EXAMPLES IN THIS PROJECT:                                               │
    │  ─────────────────────────                                               │
    │  • XᵀX   (Gram matrix, always symmetric)                                │
    │  • XᵀWX  (Hessian for logistic regression)                              │
    │  • Covariance matrices                                                   │
    │                                                                          │
    │  WHY SYMMETRY MATTERS:                                                   │
    │  ─────────────────────                                                   │
    │  • Symmetric matrices have REAL eigenvalues                              │
    │  • Can be factored as A = QΛQᵀ (spectral decomposition)                 │
    │  • More efficient storage and computation                                │
    │                                                                          │
    │  SYMMETRIC POSITIVE DEFINITE (SPD):                                      │
    │  ────────────────────────────────                                        │
    │  A is SPD if: symmetric AND xᵀAx > 0 for all x ≠ 0                      │
    │                                                                          │
    │  • All eigenvalues are POSITIVE                                          │
    │  • Has unique Cholesky factorization: A = LLᵀ                           │
    │  • XᵀX is SPD when X has full column rank                               │
    │  • XᵀX + λI is ALWAYS SPD for λ > 0 (ridge regularization!)            │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
"""

from __future__ import annotations
import numpy as np
from typing import Union, List, Tuple, Optional

# Type alias for matrices
Matrix = Union[np.ndarray, List[List[float]]]


def _ensure_numpy_matrix(M: Matrix) -> np.ndarray:
    """
    Convert input to 2D numpy array if not already.

    Args:
        M: Input matrix as nested list or numpy array

    Returns:
        2D numpy array representation

    Example:
        >>> _ensure_numpy_matrix([[1, 2], [3, 4]])
        array([[1., 2.],
               [3., 4.]])

    IMPLEMENTATION HINT:
    ────────────────────
    • Check if M is np.ndarray using isinstance()
    • If yes, ensure it's 2D with .reshape() or .astype(float)
    • If no, create with np.array(M, dtype=float)
    • Verify ndim == 2
    """
    pass


def transpose(A: Matrix) -> np.ndarray:
    """
    Compute the transpose of a matrix.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  TRANSPOSE                                                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      (Aᵀ)ᵢⱼ = Aⱼᵢ                                                       │
    │                                                                          │
    │  Shape transformation: (m × n) → (n × m)                                 │
    │                                                                          │
    │  VISUAL:                                                                 │
    │      ┌───────┐           ┌─────┐                                        │
    │      │ 1 2 3 │    →      │ 1 4 │                                        │
    │  A = │ 4 5 6 │   Aᵀ =    │ 2 5 │                                        │
    │      └───────┘           │ 3 6 │                                        │
    │        (2×3)             └─────┘                                        │
    │                           (3×2)                                         │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        A: Matrix of shape (m, n)

    Returns:
        Transposed matrix of shape (n, m)

    Example:
        >>> transpose([[1, 2, 3], [4, 5, 6]])
        array([[1., 4.],
               [2., 5.],
               [3., 6.]])

    IMPLEMENTATION:
    ────────────────
    return A.T  (numpy's transpose attribute)

    MAT223 Reference: Section 1.4
    """
    pass


def matrix_multiply(A: Matrix, B: Matrix) -> np.ndarray:
    """
    Multiply two matrices A × B.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  MATRIX MULTIPLICATION                                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      (AB)ᵢⱼ = Σₖ AᵢₖBₖⱼ   (row i of A dot with column j of B)          │
    │                                                                          │
    │  DIMENSION RULE:                                                         │
    │  ───────────────                                                         │
    │      A: (m × n)  ×  B: (n × p)  =  C: (m × p)                           │
    │                ↑                                                         │
    │         must match!                                                      │
    │                                                                          │
    │  VISUAL:                                                                 │
    │      ┌─────────┐     ┌─────┐     ┌─────┐                                │
    │      │ row₁──→ │  ×  │ ↓   │  =  │c₁₁ │  c₁₁ = row₁ · col₁            │
    │      │ row₂──→ │     │col₁│     │c₂₁ │                                 │
    │      └─────────┘     └─────┘     └─────┘                                │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  • Xβ: Design matrix × coefficients → predictions                        │
    │  • XᵀX: Gram matrix for normal equations                                │
    │  • XᵀWX: Hessian for IRLS                                               │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        A: First matrix of shape (m, n)
        B: Second matrix of shape (n, p)

    Returns:
        Product matrix of shape (m, p)

    Raises:
        ValueError: If inner dimensions don't match

    Example:
        >>> A = [[1, 2], [3, 4]]
        >>> B = [[5, 6], [7, 8]]
        >>> matrix_multiply(A, B)
        array([[19., 22.],
               [43., 50.]])

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Convert both to numpy arrays
    2. Check A.shape[1] == B.shape[0], raise ValueError if not
    3. Return A @ B (numpy's matrix multiplication operator)

    MAT223 Reference: Section 1.4.4
    """
    pass


def matrix_vector_multiply(A: Matrix, x: np.ndarray) -> np.ndarray:
    """
    Multiply a matrix by a vector: Ax.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  MATRIX-VECTOR PRODUCT                                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      y = Ax    where A: (m×n), x: (n,), y: (m,)                         │
    │                                                                          │
    │      yᵢ = Σⱼ Aᵢⱼxⱼ = (row i of A) · x                                  │
    │                                                                          │
    │  TWO INTERPRETATIONS:                                                    │
    │  ────────────────────                                                    │
    │                                                                          │
    │  1. ROW PICTURE: Each output is dot product of row with x               │
    │                                                                          │
    │      ┌─────────┐   ┌───┐     ┌───────────┐                              │
    │      │ row₁ ─→ │   │ x │     │ row₁ · x  │                              │
    │      │ row₂ ─→ │ × │   │  =  │ row₂ · x  │                              │
    │      │ row₃ ─→ │   │   │     │ row₃ · x  │                              │
    │      └─────────┘   └───┘     └───────────┘                              │
    │                                                                          │
    │  2. COLUMN PICTURE: Output is linear combination of columns             │
    │                                                                          │
    │      Ax = x₁·(col₁) + x₂·(col₂) + ... + xₙ·(colₙ)                       │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  • Xβ: Predictions for all samples                                       │
    │  • Xᵀr: Gradient computation (r = residual)                             │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        A: Matrix of shape (m, n)
        x: Vector of shape (n,)

    Returns:
        Vector of shape (m,)

    Raises:
        ValueError: If dimensions don't match

    Example:
        >>> A = [[1, 2], [3, 4], [5, 6]]
        >>> x = [1, 2]
        >>> matrix_vector_multiply(A, x)
        array([ 5., 11., 17.])

    IMPLEMENTATION:
    ────────────────
    1. Validate A.shape[1] == len(x)
    2. Return A @ x
    """
    pass


def gram_matrix(X: Matrix) -> np.ndarray:
    """
    Compute the Gram matrix XᵀX.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  GRAM MATRIX: XᵀX                                                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      G = XᵀX   where X: (n×p), G: (p×p)                                 │
    │                                                                          │
    │      Gᵢⱼ = (col i)ᵀ(col j) = dot product of columns i and j            │
    │                                                                          │
    │  PROPERTIES:                                                             │
    │  ───────────                                                             │
    │  • Always SYMMETRIC: G = Gᵀ                                             │
    │  • Always POSITIVE SEMI-DEFINITE: xᵀGx ≥ 0 for all x                   │
    │  • POSITIVE DEFINITE iff X has full column rank                         │
    │  • Diagonal entries: Gᵢᵢ = ||col i||² (squared norm of column)          │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  • Normal equations: XᵀXβ = Xᵀy                                        │
    │  • Checking conditioning: cond(XᵀX)                                     │
    │  • Ridge regression: (XᵀX + λI)β = Xᵀy                                 │
    │                                                                          │
    │  WARNING:                                                                │
    │  ────────                                                                │
    │  Computing XᵀX SQUARES the condition number!                            │
    │  Use QR factorization instead for numerical stability.                   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix of shape (n, p)

    Returns:
        Gram matrix of shape (p, p)

    Example:
        >>> X = [[1, 2], [3, 4], [5, 6]]
        >>> gram_matrix(X)
        array([[35., 44.],
               [44., 56.]])

    IMPLEMENTATION:
    ────────────────
    return X.T @ X
    """
    pass


def compute_rank(A: Matrix, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  MATRIX RANK                                                             │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  rank(A) = number of linearly independent columns                        │
    │          = number of linearly independent rows (same!)                   │
    │          = dimension of column space                                     │
    │          = number of non-zero singular values                            │
    │                                                                          │
    │  FULL RANK:                                                              │
    │  ──────────                                                              │
    │  • For (m×n) matrix: max possible rank = min(m, n)                       │
    │  • Full column rank: rank = n (all columns independent)                  │
    │  • Full row rank: rank = m (all rows independent)                        │
    │                                                                          │
    │  RANK DEFICIENCY DETECTION:                                              │
    │  ───────────────────────────                                             │
    │                                                                          │
    │      Singular Values:  σ₁ ≥ σ₂ ≥ ... ≥ σₖ > tol > σₖ₊₁ ≈ 0              │
    │                                          ↑                               │
    │                                    rank = k                              │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  • Check if design matrix X has full column rank                         │
    │  • Diagnose multicollinearity (rank < p means some features redundant)  │
    │  • Verify one-hot encoding is done correctly (dropped reference)         │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        A: Matrix to analyze
        tol: Threshold below which singular values are considered zero

    Returns:
        Integer rank of the matrix

    Example:
        >>> A = [[1, 0], [0, 1], [1, 1]]  # Full column rank
        >>> compute_rank(A)
        2
        >>> B = [[1, 2], [2, 4]]  # Rank deficient (row 2 = 2*row 1)
        >>> compute_rank(B)
        1

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Compute SVD: np.linalg.svd(A)
    2. Get singular values (the 's' in u, s, vh = svd(A))
    3. Count how many singular values are > tol
    4. Return that count

    MAT223 Reference: Section 1.3.2, 4.6
    """
    pass


def check_full_column_rank(X: Matrix) -> bool:
    """
    Check if a matrix has full column rank.

    This is CRITICAL for least squares: if X doesn't have full column rank,
    XᵀX is singular and the normal equations have no unique solution.

    Args:
        X: Matrix of shape (n, p)

    Returns:
        True if rank(X) = p (number of columns), False otherwise

    Example:
        >>> X = [[1, 0], [0, 1], [1, 1]]
        >>> check_full_column_rank(X)
        True

    IMPLEMENTATION:
    ────────────────
    return compute_rank(X) == X.shape[1]
    """
    pass


def condition_number(A: Matrix) -> float:
    """
    Compute the condition number of a matrix.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  CONDITION NUMBER                                                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      cond(A) = σₘₐₓ / σₘᵢₙ                                              │
    │                                                                          │
    │  INTERPRETATION:                                                         │
    │  ────────────────                                                        │
    │  • cond ≈ 1:       Well-conditioned (numerically stable)                │
    │  • cond ≈ 10ᵏ:     Lose ~k digits of precision in computations          │
    │  • cond = ∞:       Singular matrix (σₘᵢₙ = 0)                           │
    │                                                                          │
    │  RULES OF THUMB:                                                         │
    │  ────────────────                                                        │
    │  • cond < 10²:     Excellent                                             │
    │  • cond < 10⁴:     Good                                                  │
    │  • cond < 10⁶:     Acceptable (use with caution)                         │
    │  • cond > 10⁸:     Problematic (add regularization!)                     │
    │                                                                          │
    │  WARNING FOR NORMAL EQUATIONS:                                           │
    │  ──────────────────────────────                                          │
    │                                                                          │
    │      cond(XᵀX) = cond(X)²                                               │
    │                                                                          │
    │  If cond(X) = 10³, then cond(XᵀX) = 10⁶!                                │
    │  This is why we use QR instead of normal equations.                      │
    │                                                                          │
    │  RIDGE REGULARIZATION HELPS:                                             │
    │  ────────────────────────────                                            │
    │                                                                          │
    │      cond(XᵀX + λI) ≤ (σₘₐₓ² + λ) / λ                                  │
    │                                                                          │
    │  By adding λI, we bound the condition number!                            │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        A: Matrix to analyze

    Returns:
        Condition number (≥ 1, or inf if singular)

    Example:
        >>> A = [[1, 0], [0, 1]]  # Identity, well-conditioned
        >>> condition_number(A)
        1.0
        >>> B = [[1, 1], [1, 1.0001]]  # Nearly singular
        >>> condition_number(B)  # Very large!
        ~40000

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Compute SVD: u, s, vh = np.linalg.svd(A)
    2. s contains singular values in descending order
    3. If s[-1] < 1e-15, return np.inf (singular)
    4. Return s[0] / s[-1]

    MAT223 Reference: Section 4.6
    """
    pass


def is_symmetric(A: Matrix, tol: float = 1e-10) -> bool:
    """
    Check if a matrix is symmetric (A = Aᵀ).

    Args:
        A: Square matrix to check
        tol: Tolerance for numerical comparison

    Returns:
        True if symmetric, False otherwise

    Example:
        >>> A = [[1, 2], [2, 3]]
        >>> is_symmetric(A)
        True

    IMPLEMENTATION:
    ────────────────
    1. Check A is square: A.shape[0] == A.shape[1]
    2. Check np.allclose(A, A.T, atol=tol)
    """
    pass


def is_positive_definite(A: Matrix) -> bool:
    """
    Check if a matrix is positive definite.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  POSITIVE DEFINITE (PD)                                                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  A is PD iff: xᵀAx > 0 for all x ≠ 0                                   │
    │                                                                          │
    │  EQUIVALENT CONDITIONS:                                                  │
    │  ──────────────────────                                                  │
    │  • All eigenvalues are positive                                          │
    │  • Cholesky decomposition exists (A = LLᵀ)                              │
    │  • All leading principal minors are positive                             │
    │                                                                          │
    │  WHY IT MATTERS:                                                         │
    │  ────────────────                                                        │
    │  • PD matrices are invertible                                            │
    │  • Convex optimization: Hessian is PD at minimum                         │
    │  • Logistic regression Hessian XᵀWX is PD (W has positive diagonals)   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        A: Square matrix to check

    Returns:
        True if positive definite, False otherwise

    Example:
        >>> A = [[2, -1], [-1, 2]]  # PD
        >>> is_positive_definite(A)
        True

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Check symmetry first
    2. Compute eigenvalues: np.linalg.eigvalsh(A) (for symmetric matrices)
    3. Return True if all eigenvalues > 0 (or > small tolerance)
    """
    pass


def diagonal_matrix(diag_elements: np.ndarray) -> np.ndarray:
    """
    Create a diagonal matrix from a vector.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  DIAGONAL MATRIX                                                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      diag([d₁, d₂, d₃]) =  ┌─────────────┐                              │
    │                            │ d₁  0   0   │                              │
    │                            │  0  d₂  0   │                              │
    │                            │  0   0  d₃  │                              │
    │                            └─────────────┘                              │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  • Weight matrix W in IRLS: W = diag(p(1-p))                            │
    │  • Regularization: λI = λ × diag(1,1,...,1)                             │
    │  • Scaling transformations                                               │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        diag_elements: Vector of diagonal elements

    Returns:
        Diagonal matrix of shape (n, n)

    Example:
        >>> diagonal_matrix([1, 2, 3])
        array([[1., 0., 0.],
               [0., 2., 0.],
               [0., 0., 3.]])

    IMPLEMENTATION:
    ────────────────
    return np.diag(diag_elements)
    """
    pass


def identity(n: int) -> np.ndarray:
    """
    Create an n×n identity matrix.

    The identity matrix I satisfies: IA = AI = A for any conformable matrix A.

    Args:
        n: Size of the identity matrix

    Returns:
        n×n identity matrix

    Example:
        >>> identity(3)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])

    IMPLEMENTATION:
    ────────────────
    return np.eye(n)
    """
    pass


def add_ridge(XtX: Matrix, lambda_: float) -> np.ndarray:
    """
    Add ridge regularization to a Gram matrix: XᵀX + λI.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  RIDGE REGULARIZATION                                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      (XᵀX + λI)β = Xᵀy                                                 │
    │                                                                          │
    │  EFFECT:                                                                 │
    │  ───────                                                                 │
    │  • Makes the matrix ALWAYS invertible (even if XᵀX is singular)        │
    │  • Reduces condition number                                              │
    │  • Shrinks coefficient estimates toward zero                             │
    │                                                                          │
    │  EIGENVALUE VIEW:                                                        │
    │  ─────────────────                                                       │
    │  If XᵀX has eigenvalues λ₁, λ₂, ..., λₚ                                │
    │  Then XᵀX + λI has eigenvalues λ₁+λ, λ₂+λ, ..., λₚ+λ                   │
    │                                                                          │
    │  All eigenvalues shift up by λ, so:                                      │
    │  • Smallest eigenvalue is now ≥ λ (no longer near zero!)                │
    │  • Matrix is guaranteed SPD                                              │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        XtX: Gram matrix of shape (p, p)
        lambda_: Regularization parameter (must be ≥ 0)

    Returns:
        Regularized matrix XᵀX + λI

    Raises:
        ValueError: If lambda_ < 0

    Example:
        >>> XtX = [[1, 0.5], [0.5, 1]]
        >>> add_ridge(XtX, 0.1)
        array([[1.1, 0.5],
               [0.5, 1.1]])

    IMPLEMENTATION:
    ────────────────
    1. Validate lambda_ >= 0
    2. Get size p from XtX.shape[0]
    3. Return XtX + lambda_ * np.eye(p)
    """
    pass


def outer_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the outer product of two vectors: xyᵀ.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  OUTER PRODUCT: xyᵀ                                                      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      x: (m,)  y: (n,)  →  xyᵀ: (m × n)                                  │
    │                                                                          │
    │      (xyᵀ)ᵢⱼ = xᵢyⱼ                                                     │
    │                                                                          │
    │  VISUAL:                                                                 │
    │      ┌───┐                   ┌─────────────────┐                        │
    │      │ x₁│                   │ x₁y₁  x₁y₂  x₁y₃│                        │
    │  x = │ x₂│   y = [y₁ y₂ y₃] │ x₂y₁  x₂y₂  x₂y₃│                        │
    │      │ x₃│         →         │ x₃y₁  x₃y₂  x₃y₃│                        │
    │      └───┘                   └─────────────────┘                        │
    │                                                                          │
    │  PROPERTIES:                                                             │
    │  ───────────                                                             │
    │  • rank(xyᵀ) = 1 (always rank 1!)                                       │
    │  • xyᵀ is symmetric iff x and y are parallel                            │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  • Householder reflection: I - 2vvᵀ (in QR factorization)               │
    │  • Covariance update: Σ += (x - μ)(x - μ)ᵀ                              │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        x: First vector of shape (m,)
        y: Second vector of shape (n,)

    Returns:
        Outer product matrix of shape (m, n)

    Example:
        >>> outer_product([1, 2], [3, 4, 5])
        array([[ 3.,  4.,  5.],
               [ 6.,  8., 10.]])

    IMPLEMENTATION:
    ────────────────
    return np.outer(x, y)
    """
    pass


def trace(A: Matrix) -> float:
    """
    Compute the trace of a square matrix (sum of diagonal elements).

    The trace has nice properties:
    • tr(A + B) = tr(A) + tr(B)
    • tr(αA) = α·tr(A)
    • tr(AB) = tr(BA)  (cyclic property)
    • tr(A) = sum of eigenvalues

    Args:
        A: Square matrix

    Returns:
        Sum of diagonal elements

    Example:
        >>> trace([[1, 2], [3, 4]])
        5.0

    IMPLEMENTATION:
    ────────────────
    return np.trace(A)
    """
    pass


def frobenius_norm(A: Matrix) -> float:
    """
    Compute the Frobenius norm of a matrix.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  FROBENIUS NORM                                                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      ||A||_F = √(Σᵢⱼ aᵢⱼ²) = √(tr(AᵀA))                                │
    │                                                                          │
    │  This is the matrix equivalent of the vector L2 norm.                    │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  • Convergence check: ||β_new - β_old||_F < tolerance                   │
    │  • Matrix approximation error in SVD truncation                          │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        A: Matrix

    Returns:
        Frobenius norm (scalar ≥ 0)

    Example:
        >>> frobenius_norm([[1, 2], [3, 4]])
        5.477...  # sqrt(1 + 4 + 9 + 16) = sqrt(30)

    IMPLEMENTATION:
    ────────────────
    return np.linalg.norm(A, 'fro')
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
│  HIGH PRIORITY (needed for IRLS):                                            │
│  ─────────────────────────────────                                           │
│  [ ] Implement _ensure_numpy_matrix() helper                                 │
│  [ ] Implement transpose() - basic operation                                 │
│  [ ] Implement matrix_multiply() - for Xβ predictions                       │
│  [ ] Implement matrix_vector_multiply() - for efficiency                     │
│  [ ] Implement gram_matrix() - for XᵀX                                      │
│  [ ] Implement diagonal_matrix() - for weight matrix W                       │
│  [ ] Implement identity() - for regularization                               │
│  [ ] Implement add_ridge() - for XᵀX + λI                                   │
│                                                                              │
│  MEDIUM PRIORITY (diagnostics):                                              │
│  ──────────────────────────────                                              │
│  [ ] Implement compute_rank() - verify design matrix                         │
│  [ ] Implement check_full_column_rank() - quick check                        │
│  [ ] Implement condition_number() - numerical stability check               │
│  [ ] Implement is_symmetric() - verify Gram matrices                         │
│  [ ] Implement is_positive_definite() - verify PD for solver                │
│                                                                              │
│  LOW PRIORITY (utilities):                                                   │
│  ─────────────────────────                                                   │
│  [ ] Implement outer_product() - for Householder QR                         │
│  [ ] Implement trace() - for diagnostics                                     │
│  [ ] Implement frobenius_norm() - for convergence checks                    │
│                                                                              │
│  TESTING:                                                                    │
│  ────────                                                                    │
│  [ ] Write unit tests in tests/test_math/test_matrices.py                   │
│  [ ] Verify against numpy.linalg functions                                   │
│  [ ] Test edge cases: singular matrices, empty matrices                      │
│  [ ] Test with actual design matrix structure from project                   │
│                                                                              │
│  ENHANCEMENTS (OPTIONAL):                                                    │
│  ─────────────────────────                                                   │
│  [ ] Add block matrix operations                                             │
│  [ ] Implement efficient symmetric matrix storage                            │
│  [ ] Add sparse matrix support (scipy.sparse)                                │
│  [ ] GPU acceleration with PyTorch tensors                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    print("Matrix Operations Module")
    print("=" * 50)
    print("Implement the functions above and then run tests!")
    print()
    print("After implementation, test with:")
    print("  >>> A = [[1, 2], [3, 4]]")
    print("  >>> B = [[5, 6], [7, 8]]")
    print("  >>> C = matrix_multiply(A, B)")
    print("  >>> print(C)  # Should be [[19, 22], [43, 50]]")
