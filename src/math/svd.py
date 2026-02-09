"""
Singular Value Decomposition (SVD) Module
==========================================

This module implements SVD and low-rank approximation, providing the mathematical
foundation for understanding embeddings, dimensionality reduction, and matrix
conditioning analysis.

MAT223 REFERENCE: Sections 3.4 (Eigenvalues/Eigenvectors), 3.6 (Applications)

================================================================================
                        SYSTEM ARCHITECTURE CONTEXT
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    WHERE THIS MODULE FITS                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │        vectors.py ──► matrices.py ──► projections.py ──► qr.py          │
    │                                              │                           │
    │                                              ▼                           │
    │                                    ┌─────────────────┐                   │
    │                                    │  [THIS MODULE]  │                   │
    │                                    │     svd.py      │                   │
    │                                    └────────┬────────┘                   │
    │                                             │                            │
    │           ┌─────────────────────────────────┼────────────────────────┐   │
    │           ▼                                 ▼                        ▼   │
    │    ┌─────────────┐                  ┌─────────────┐           ┌─────────┐│
    │    │  ridge.py   │                  │ embeddings  │           │  Model  ││
    │    │(shrinkage   │                  │    .py      │           │Diagnos- ││
    │    │ analysis)   │                  │(low-rank)   │           │  tics   ││
    │    └─────────────┘                  └─────────────┘           └─────────┘│
    │                                                                          │
    │  SVD reveals the fundamental structure of matrices:                      │
    │  • Singular values = "importance" of each dimension                     │
    │  • Low-rank approximation = embeddings                                   │
    │  • Condition number = numerical stability                                │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    THE SVD DECOMPOSITION
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  SINGULAR VALUE DECOMPOSITION                                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Any matrix A (m × n) can be decomposed as:                              │
    │                                                                          │
    │                    A = U Σ Vᵀ                                            │
    │                                                                          │
    │  WHERE:                                                                  │
    │  ──────                                                                  │
    │  • U (m × m): Left singular vectors (orthonormal columns)               │
    │  • Σ (m × n): Diagonal matrix with singular values σ₁ ≥ σ₂ ≥ ... ≥ 0   │
    │  • V (n × n): Right singular vectors (orthonormal columns)               │
    │                                                                          │
    │  MATRIX SHAPES:                                                          │
    │  ───────────────                                                         │
    │                                                                          │
    │      ┌─────────┐     ┌─────┐   ┌───────────┐   ┌─────────┐             │
    │      │         │     │     │   │σ₁         │   │         │ᵀ            │
    │      │    A    │  =  │  U  │ × │  σ₂       │ × │    V    │             │
    │      │  (m×n)  │     │(m×m)│   │    ⋱     │   │  (n×n)  │             │
    │      │         │     │     │   │      σₖ   │   │         │             │
    │      └─────────┘     └─────┘   └───────────┘   └─────────┘             │
    │                                    (m×n)                                 │
    │                                                                          │
    │  ECONOMIC (THIN) SVD:                                                    │
    │  ─────────────────────                                                   │
    │  When m > n (more rows than columns), we can use thin SVD:               │
    │                                                                          │
    │      A = U_thin Σ_thin Vᵀ                                               │
    │          (m×n)   (n×n)   (n×n)                                           │
    │                                                                          │
    │  This saves memory and computation.                                      │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    GEOMETRIC INTERPRETATION
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  SVD AS ROTATION → STRETCH → ROTATION                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  The SVD reveals that ANY linear transformation can be decomposed:       │
    │                                                                          │
    │      A = U Σ Vᵀ                                                         │
    │          │ │ │                                                           │
    │          │ │ └── Vᵀ: Rotate input (align with "natural" axes)           │
    │          │ └──── Σ:  Stretch along each axis by σᵢ                      │
    │          └────── U:  Rotate output (to final orientation)                │
    │                                                                          │
    │  VISUAL (2D example):                                                    │
    │                                                                          │
    │    Original   ──Vᵀ──►   Aligned    ──Σ──►   Stretched  ──U──►   Final   │
    │                                                                          │
    │       ◯               ◯                  ◯──────────◯                 ⬬  │
    │      ╱ ╲             │ │                │          │                ╱ ╲  │
    │     ╱   ╲     ──►    │ │       ──►      │          │       ──►     ╱   ╲ │
    │    ╱     ╲           │ │                │          │              ╱     ╲│
    │   ◯───────◯         ◯─◯               ◯──────────◯            ⬬───────⬬ │
    │                                                                          │
    │   Unit circle     Rotated        Ellipse (scaled)    Final ellipse       │
    │                                                                          │
    │  The singular values σᵢ are the SEMI-AXES of the output ellipse!        │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    LOW-RANK APPROXIMATION
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  BEST RANK-k APPROXIMATION (Eckart-Young Theorem)                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  The best rank-k approximation to A (in Frobenius norm) is:              │
    │                                                                          │
    │      A_k = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ                                             │
    │                                                                          │
    │  VISUAL:                                                                 │
    │  ───────                                                                 │
    │                                                                          │
    │      A = σ₁ u₁v₁ᵀ + σ₂ u₂v₂ᵀ + σ₃ u₃v₃ᵀ + ... + σᵣ uᵣvᵣᵀ             │
    │          ─────────   ─────────   ─────────         ─────────             │
    │          rank-1      rank-1      rank-1            rank-1                │
    │          component   component   component         component             │
    │                                                                          │
    │      ┌──────────────────────────────────────────────────────────┐       │
    │      │                                                           │       │
    │      │  A_2 = σ₁ u₁v₁ᵀ + σ₂ u₂v₂ᵀ   (best rank-2 approximation)│       │
    │      │        ───────────────────────                            │       │
    │      │        Keep only top 2 singular values/vectors            │       │
    │      │                                                           │       │
    │      │  Error = ||A - A_2||_F = √(σ₃² + σ₄² + ... + σᵣ²)        │       │
    │      │                                                           │       │
    │      └──────────────────────────────────────────────────────────┘       │
    │                                                                          │
    │  PROJECT CONNECTION:                                                     │
    │  ────────────────────                                                    │
    │  This is EXACTLY what embeddings do!                                     │
    │                                                                          │
    │  Full interaction matrix (universities × programs):                      │
    │      A = U_uni × Σ × V_prog^T                                           │
    │                                                                          │
    │  Low-rank approximation:                                                 │
    │      A ≈ U_uni[:,:k] × Σ[:k,:k] × V_prog[:,:k]^T                        │
    │                                                                          │
    │  The truncated U gives university embeddings!                            │
    │  The truncated V gives program embeddings!                               │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    SVD AND CONDITIONING
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  CONDITION NUMBER FROM SVD                                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  The condition number of a matrix A is:                                  │
    │                                                                          │
    │      κ(A) = σ_max / σ_min = σ₁ / σᵣ                                    │
    │                                                                          │
    │  WHERE:                                                                  │
    │  ──────                                                                  │
    │  • σ₁ = largest singular value                                          │
    │  • σᵣ = smallest nonzero singular value                                 │
    │                                                                          │
    │  INTERPRETATION:                                                         │
    │  ────────────────                                                        │
    │  κ(A) ≈ 1:     Well-conditioned (numerically stable)                    │
    │  κ(A) > 100:   Moderately ill-conditioned                               │
    │  κ(A) > 10⁶:   Severely ill-conditioned (expect numerical issues)      │
    │  κ(A) = ∞:     Singular (no inverse exists)                             │
    │                                                                          │
    │  SINGULAR VALUE SPECTRUM VISUALIZATION:                                  │
    │  ───────────────────────────────────────                                 │
    │                                                                          │
    │      Well-conditioned:           Ill-conditioned:                        │
    │                                                                          │
    │      σ│ ████                     σ│ ████████████████                    │
    │       │ ███                       │ █                                    │
    │       │ ██                        │                                      │
    │       │ █                         │    (big gap!)                        │
    │       └───────                    └───────────────                       │
    │        1 2 3 4 (index)             1         n (index)                   │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  Before solving XᵀXβ = Xᵀy, check:                                      │
    │  • If κ(X) > 1000, use ridge regularization                             │
    │  • Ridge adds λ to each σᵢ², improving conditioning                     │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    RIDGE REGRESSION VIA SVD
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  SVD VIEW OF RIDGE REGRESSION                                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Ridge solution:  β_λ = (XᵀX + λI)⁻¹Xᵀy                                │
    │                                                                          │
    │  Using SVD of X = UΣVᵀ:                                                 │
    │                                                                          │
    │      β_λ = Σᵢ (σᵢ² / (σᵢ² + λ)) × (uᵢᵀy) × vᵢ                        │
    │                ─────────────────                                         │
    │                 shrinkage factor                                         │
    │                                                                          │
    │  SHRINKAGE VISUALIZATION:                                                │
    │  ─────────────────────────                                               │
    │                                                                          │
    │      shrinkage factor  │                                                 │
    │      σᵢ²/(σᵢ²+λ)      │  ┌───────────────── 1.0 (no shrinkage)         │
    │                        │ ╱                                               │
    │               1.0 ─────┼────────╮                                        │
    │                        │        │╲                                       │
    │               0.5 ─────┼────────┼─╲───────                               │
    │                        │        │  ╲                                     │
    │               0.0 ─────┼────────┼───╲───── 0.0 (full shrinkage)         │
    │                        └────────┴────────► σᵢ                            │
    │                             λ                                            │
    │                                                                          │
    │  INTERPRETATION:                                                         │
    │  ────────────────                                                        │
    │  • Directions with large σᵢ (strong signal): little shrinkage           │
    │  • Directions with small σᵢ (weak signal/noise): heavy shrinkage        │
    │  • Ridge "trusts" strong patterns, shrinks weak ones                    │
    │                                                                          │
    │  This is why ridge regression:                                           │
    │  1. Prevents overfitting (shrinks noisy directions)                      │
    │  2. Improves conditioning (adds λ to all σᵢ²)                           │
    │  3. Provides unique solution even when X is rank-deficient               │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Union


def compute_svd(
    A: np.ndarray,
    full_matrices: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Singular Value Decomposition of matrix A.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  SVD COMPUTATION                                                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      A = U Σ Vᵀ                                                         │
    │                                                                          │
    │  With full_matrices=False (economic/thin SVD):                          │
    │      A (m×n) = U (m×k) × Σ (k×k) × Vᵀ (k×n)                            │
    │      where k = min(m, n)                                                 │
    │                                                                          │
    │  With full_matrices=True:                                                │
    │      A (m×n) = U (m×m) × Σ (m×n) × Vᵀ (n×n)                            │
    │                                                                          │
    │  NOTE: We return V, not Vᵀ (like numpy.linalg.svd)                      │
    │  So A ≈ U @ np.diag(s) @ V.T                                            │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        A: Input matrix of shape (m, n)
        full_matrices: If True, return full U and V; if False, return thin SVD

    Returns:
        Tuple of (U, s, V) where:
        - U: Left singular vectors
        - s: Singular values (1D array, sorted descending)
        - V: Right singular vectors (NOT transposed)

    Example:
        >>> A = np.array([[1, 2], [3, 4], [5, 6]])
        >>> U, s, V = compute_svd(A)
        >>> reconstructed = U @ np.diag(s) @ V.T
        >>> np.allclose(A, reconstructed)
        True

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Call np.linalg.svd(A, full_matrices=full_matrices)
    2. Note: numpy returns (U, s, Vh) where Vh = Vᵀ
    3. Return (U, s, Vh.T) so we return V, not Vᵀ

    MAT223 Reference: Section 3.4, 3.6
    """
    U, s, Vh = np.linalg.svd(A, full_matrices=full_matrices)
    return U, s, Vh.T


def singular_values(A: np.ndarray) -> np.ndarray:
    """
    Compute only the singular values of matrix A.

    More efficient than full SVD when you don't need U or V.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  SINGULAR VALUES                                                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  σ₁ ≥ σ₂ ≥ ... ≥ σₖ ≥ 0   where k = min(m, n)                          │
    │                                                                          │
    │  PROPERTIES:                                                             │
    │  ───────────                                                             │
    │  • σᵢ = √(eigenvalue of AᵀA) = √(eigenvalue of AAᵀ)                   │
    │  • σ₁ = ||A||₂ (operator/spectral norm)                                 │
    │  • Σσᵢ² = ||A||_F² (squared Frobenius norm)                            │
    │  • rank(A) = number of nonzero singular values                          │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  • Condition number = σ₁ / σᵣ                                           │
    │  • Decide embedding dimension (where σᵢ drops off)                      │
    │  • Measure "effective rank" (number of significant σᵢ)                  │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        A: Input matrix of shape (m, n)

    Returns:
        Array of singular values, sorted descending

    Example:
        >>> A = np.array([[3, 0], [0, 4]])
        >>> singular_values(A)
        array([4., 3.])

    IMPLEMENTATION:
    ────────────────
    return np.linalg.svd(A, compute_uv=False)
    """
    return np.linalg.svd(A, compute_uv=False)


def condition_number(A: np.ndarray) -> float:
    """
    Compute the condition number of matrix A using SVD.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  CONDITION NUMBER                                                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      κ(A) = σ_max / σ_min = σ₁ / σᵣ                                    │
    │                                                                          │
    │  INTERPRETATION:                                                         │
    │  ────────────────                                                        │
    │                                                                          │
    │      κ(A) ≈ 1       │ Excellent - numerically stable                    │
    │      κ(A) ≈ 10      │ Good - minor issues possible                      │
    │      κ(A) ≈ 100     │ Moderate - some precision loss                    │
    │      κ(A) ≈ 10⁶     │ Poor - significant precision loss                 │
    │      κ(A) ≈ 10¹⁵    │ Near-singular - results unreliable               │
    │      κ(A) = ∞       │ Singular - no inverse exists                      │
    │                                                                          │
    │  RULE OF THUMB:                                                          │
    │  ───────────────                                                         │
    │  With 64-bit floats, expect to lose log₁₀(κ) digits of precision.      │
    │  If κ = 10⁶, you have ~10 reliable digits (out of ~16).                │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  Before solving least squares:                                           │
    │  1. Compute κ(X) or κ(XᵀX)                                              │
    │  2. If κ > 1000, add ridge regularization                               │
    │  3. Document conditioning in model diagnostics                           │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        A: Input matrix of shape (m, n)

    Returns:
        Condition number (σ_max / σ_min), or np.inf if singular

    Example:
        >>> A = np.array([[1, 0], [0, 1]])  # Identity - perfect conditioning
        >>> condition_number(A)
        1.0
        >>> B = np.array([[1, 1], [1, 1.0001]])  # Nearly singular
        >>> condition_number(B) > 1000
        True

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. s = singular_values(A)
    2. if s[-1] < 1e-15: return np.inf  # singular
    3. return s[0] / s[-1]
    """
    s = singular_values(A)
    if len(s) == 0 or s[-1] < 1e-15:
        return np.inf
    return float(s[0] / s[-1])


def matrix_rank(A: np.ndarray, tol: Optional[float] = None) -> int:
    """
    Compute the numerical rank of a matrix using SVD.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  NUMERICAL RANK                                                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  rank(A) = number of singular values above tolerance                     │
    │                                                                          │
    │  WHY "NUMERICAL" RANK?                                                   │
    │  ──────────────────────                                                  │
    │  Due to floating-point errors, theoretically zero singular values       │
    │  may become small but nonzero (e.g., 1e-16).                            │
    │                                                                          │
    │  Default tolerance: max(m, n) × ε × σ_max                               │
    │  where ε ≈ 2.2e-16 (machine epsilon for float64)                        │
    │                                                                          │
    │  VISUAL:                                                                 │
    │  ───────                                                                 │
    │      σ│ ████                                                            │
    │       │ ███                                                              │
    │       │ ██                                                               │
    │       │ █                                                                │
    │       │────────── tolerance ───────────                                  │
    │       │ ▪▪ (below tolerance: considered zero)                           │
    │       └────────────────────────────►                                     │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  • Verify design matrix has full column rank                             │
    │  • Detect collinearity issues                                            │
    │  • Determine appropriate embedding dimension                              │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for treating singular values as zero.
             If None, uses default: max(m,n) * eps * σ_max

    Returns:
        Numerical rank of the matrix

    Example:
        >>> A = np.array([[1, 2, 3], [2, 4, 6]])  # Second row = 2 × first
        >>> matrix_rank(A)
        1

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. s = singular_values(A)
    2. if tol is None:
           tol = max(A.shape) * np.finfo(A.dtype).eps * s[0]
    3. return np.sum(s > tol)
    """
    s = singular_values(A)
    if len(s) == 0:
        return 0
    if tol is None:
        tol = max(A.shape) * np.finfo(A.dtype).eps * s[0]
    return int(np.sum(s > tol))


def low_rank_approximation(
    A: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the best rank-k approximation of matrix A.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  LOW-RANK APPROXIMATION (Eckart-Young Theorem)                           │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  A_k = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ = U_k Σ_k V_kᵀ                                 │
    │                                                                          │
    │  THEOREM:                                                                │
    │  ────────                                                                │
    │  A_k is the BEST rank-k approximation to A in both:                      │
    │  • Frobenius norm: ||A - A_k||_F                                        │
    │  • Spectral norm:  ||A - A_k||_2                                        │
    │                                                                          │
    │  VISUALIZATION:                                                          │
    │  ───────────────                                                         │
    │                                                                          │
    │      Full A (rank r):                                                    │
    │      ┌───────────────────────────────────────┐                          │
    │      │███████████████████████████████████████│                          │
    │      │███████████████████████████████████████│                          │
    │      │███████████████████████████████████████│                          │
    │      └───────────────────────────────────────┘                          │
    │                                                                          │
    │      Rank-k approximation (k < r):                                       │
    │      ┌─────────┐   ┌─────┐   ┌─────────┐ᵀ                               │
    │      │         │   │σ₁   │   │         │                                 │
    │      │   U_k   │ × │  σ₂ │ × │   V_k   │                                 │
    │      │  (m×k)  │   │   ⋱│   │  (n×k)  │                                 │
    │      │         │   │    σₖ│   │         │                                 │
    │      └─────────┘   └─────┘   └─────────┘                                │
    │                     (k×k)                                                │
    │                                                                          │
    │  COMPRESSION RATIO: k(m + n + 1) / (mn)                                  │
    │  Example: 1000×500 matrix, k=10 → 3% of original storage!               │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        A: Input matrix of shape (m, n)
        k: Target rank (must be ≤ min(m, n))

    Returns:
        Tuple of (U_k, s_k, V_k) where:
        - U_k: shape (m, k) - first k left singular vectors
        - s_k: shape (k,) - first k singular values
        - V_k: shape (n, k) - first k right singular vectors

    Raises:
        ValueError: If k > min(m, n) or k < 1

    Example:
        >>> A = np.random.randn(100, 50)
        >>> U_k, s_k, V_k = low_rank_approximation(A, k=10)
        >>> A_approx = U_k @ np.diag(s_k) @ V_k.T
        >>> A_approx.shape
        (100, 50)

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Validate 1 ≤ k ≤ min(m, n)
    2. U, s, V = compute_svd(A)
    3. Return (U[:, :k], s[:k], V[:, :k])
    """
    m, n = A.shape
    max_k = min(m, n)
    if k < 1 or k > max_k:
        raise ValueError(
            f"k must be between 1 and min(m, n)={max_k}, got k={k}"
        )
    U, s, V = compute_svd(A, full_matrices=False)
    return U[:, :k], s[:k], V[:, :k]


def reconstruct_from_svd(
    U: np.ndarray,
    s: np.ndarray,
    V: np.ndarray
) -> np.ndarray:
    """
    Reconstruct a matrix from its SVD components.

    A = U @ diag(s) @ V.T

    Args:
        U: Left singular vectors, shape (m, k)
        s: Singular values, shape (k,)
        V: Right singular vectors, shape (n, k)

    Returns:
        Reconstructed matrix of shape (m, n)

    Example:
        >>> A = np.array([[1, 2], [3, 4]])
        >>> U, s, V = compute_svd(A)
        >>> A_reconstructed = reconstruct_from_svd(U, s, V)
        >>> np.allclose(A, A_reconstructed)
        True

    IMPLEMENTATION:
    ────────────────
    return U @ np.diag(s) @ V.T
    """
    return U @ np.diag(s) @ V.T


def approximation_error(
    A: np.ndarray,
    k: int,
    norm: str = 'fro'
) -> float:
    """
    Compute the error of rank-k approximation.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  APPROXIMATION ERROR                                                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Frobenius norm error:                                                   │
    │      ||A - A_k||_F = √(σ_{k+1}² + σ_{k+2}² + ... + σ_r²)               │
    │                                                                          │
    │  Spectral norm error:                                                    │
    │      ||A - A_k||_2 = σ_{k+1}                                            │
    │                                                                          │
    │  Relative error:                                                         │
    │      ||A - A_k||_F / ||A||_F = √(Σᵢ>ₖ σᵢ²) / √(Σᵢ σᵢ²)               │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  Choose embedding dimension k by looking at:                             │
    │  • Where relative error becomes acceptable (e.g., < 10%)                │
    │  • Where σᵢ drops sharply ("elbow" in singular value plot)             │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        A: Input matrix of shape (m, n)
        k: Rank of approximation
        norm: 'fro' for Frobenius, 'spectral' for spectral/operator norm

    Returns:
        Approximation error value

    Example:
        >>> A = np.diag([10, 5, 2, 0.1])
        >>> approximation_error(A, k=2, norm='fro')  # √(2² + 0.1²) ≈ 2.0
        >>> approximation_error(A, k=2, norm='spectral')  # σ₃ = 2
        2.0

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. s = singular_values(A)
    2. if norm == 'spectral': return s[k] if k < len(s) else 0.0
    3. if norm == 'fro': return np.sqrt(np.sum(s[k:]**2))
    """
    s = singular_values(A)
    if norm == 'spectral':
        return float(s[k]) if k < len(s) else 0.0
    elif norm == 'fro':
        return float(np.sqrt(np.sum(s[k:] ** 2)))
    else:
        raise ValueError(f"Unknown norm: {norm}. Use 'fro' or 'spectral'.")


def explained_variance_ratio(A: np.ndarray) -> np.ndarray:
    """
    Compute the variance explained by each singular component.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  EXPLAINED VARIANCE RATIO                                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      ratio_i = σᵢ² / Σⱼ σⱼ²                                            │
    │                                                                          │
    │  INTERPRETATION:                                                         │
    │  ────────────────                                                        │
    │  • ratio_i tells you what fraction of ||A||_F² is captured by rank-1    │
    │    component σᵢ uᵢ vᵢᵀ                                                  │
    │  • Cumulative sum tells you total variance captured by rank-k approx    │
    │                                                                          │
    │  VISUALIZATION:                                                          │
    │  ───────────────                                                         │
    │                                                                          │
    │      Variance explained │                                                │
    │                    100% ┼────────────────────────── ●────────●──────●   │
    │                         │                      ●                         │
    │                     80% ┼                 ●                              │
    │                         │            ●                                   │
    │                     60% ┼       ●                                        │
    │                         │  ●                                             │
    │                     40% ┼                                                │
    │                         │                                                │
    │                     20% ┼                                                │
    │                         └────────────────────────────────────────────► k │
    │                           1   2   3   4   5   6   7   8                  │
    │                                                                          │
    │  "Elbow" in the cumulative plot suggests good k choice.                  │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        A: Input matrix of shape (m, n)

    Returns:
        Array of variance ratios, one per singular value, summing to 1

    Example:
        >>> A = np.diag([10, 5, 2, 1])
        >>> evr = explained_variance_ratio(A)
        >>> evr[0]  # First component explains 100/(100+25+4+1) ≈ 0.77
        0.769...
        >>> np.sum(evr[:2])  # First 2 explain (100+25)/130 ≈ 0.96

    IMPLEMENTATION:
    ────────────────
    1. s = singular_values(A)
    2. s_squared = s ** 2
    3. return s_squared / np.sum(s_squared)
    """
    s = singular_values(A)
    s_squared = s ** 2
    return s_squared / np.sum(s_squared)


def choose_rank(
    A: np.ndarray,
    variance_threshold: float = 0.95
) -> int:
    """
    Choose the rank k that explains at least variance_threshold of variance.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  AUTOMATIC RANK SELECTION                                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Strategy: Choose smallest k such that:                                  │
    │      Σᵢ₌₁ᵏ σᵢ² / Σⱼ σⱼ² ≥ threshold                                   │
    │                                                                          │
    │  Common thresholds:                                                      │
    │  ───────────────────                                                     │
    │  • 0.80 - Aggressive compression, some information loss                 │
    │  • 0.90 - Balanced compression                                           │
    │  • 0.95 - Conservative compression (default)                             │
    │  • 0.99 - Minimal compression, near-lossless                             │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  For embeddings:                                                         │
    │  • Start with threshold=0.95                                             │
    │  • If too many dimensions, try 0.90                                      │
    │  • Validate on downstream task (prediction quality)                      │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        A: Input matrix of shape (m, n)
        variance_threshold: Minimum fraction of variance to explain (0 to 1)

    Returns:
        Chosen rank k (always at least 1)

    Example:
        >>> A = np.diag([10, 5, 2, 0.1])
        >>> choose_rank(A, variance_threshold=0.95)
        2  # First 2 singular values explain >95% of variance

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. evr = explained_variance_ratio(A)
    2. cumulative = np.cumsum(evr)
    3. k = np.searchsorted(cumulative, variance_threshold) + 1
    4. return min(k, len(evr))  # Can't exceed number of singular values
    """
    evr = explained_variance_ratio(A)
    cumulative = np.cumsum(evr)
    k = int(np.searchsorted(cumulative, variance_threshold)) + 1
    return min(k, len(evr))


def svd_solve(
    A: np.ndarray,
    b: np.ndarray,
    rcond: Optional[float] = None
) -> np.ndarray:
    """
    Solve Ax = b using SVD (Moore-Penrose pseudoinverse).

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  SVD-BASED LEAST SQUARES                                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Given A = UΣVᵀ, the minimum-norm least squares solution is:            │
    │                                                                          │
    │      x = VΣ⁺Uᵀb                                                        │
    │                                                                          │
    │  Where Σ⁺ is the pseudoinverse of Σ:                                   │
    │      Σ⁺ᵢᵢ = 1/σᵢ if σᵢ > rcond × σ_max, else 0                        │
    │                                                                          │
    │  PROPERTIES:                                                             │
    │  ───────────                                                             │
    │  • Works for ANY matrix (rectangular, rank-deficient, etc.)             │
    │  • Returns minimum-norm solution when multiple solutions exist           │
    │  • Numerically stable for ill-conditioned problems                       │
    │                                                                          │
    │  COMPARISON:                                                             │
    │  ───────────                                                             │
    │  Normal equations:  (AᵀA)⁻¹Aᵀb  - squares condition number!            │
    │  QR factorization:  R⁻¹Qᵀb     - stable, but needs full rank           │
    │  SVD:               VΣ⁺Uᵀb     - most stable, works always             │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        A: Coefficient matrix, shape (m, n)
        b: Right-hand side, shape (m,) or (m, k)
        rcond: Cutoff for small singular values. Values below rcond * σ_max
               are treated as zero. Default: machine epsilon × max(m, n)

    Returns:
        Solution x of shape (n,) or (n, k)

    Example:
        >>> A = np.array([[1, 0], [0, 1], [1, 1]])  # Overdetermined
        >>> b = np.array([1, 2, 2.5])
        >>> x = svd_solve(A, b)
        >>> np.allclose(A @ x, b, atol=0.5)  # Least squares solution

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. U, s, V = compute_svd(A)
    2. if rcond is None: rcond = max(A.shape) * np.finfo(A.dtype).eps
    3. threshold = rcond * s[0]
    4. s_inv = np.where(s > threshold, 1/s, 0)
    5. return V @ (s_inv[:, np.newaxis] * (U.T @ b))  # Handle broadcasting
    """
    U, s, V = compute_svd(A, full_matrices=False)
    if rcond is None:
        rcond = max(A.shape) * np.finfo(A.dtype).eps
    threshold = rcond * s[0]
    s_inv = np.where(s > threshold, 1.0 / s, 0.0)
    # U.T @ b may be 1D or 2D; use broadcasting
    Utb = U.T @ b
    if Utb.ndim == 1:
        return V @ (s_inv * Utb)
    else:
        return V @ (s_inv[:, np.newaxis] * Utb)


def ridge_via_svd(
    X: np.ndarray,
    y: np.ndarray,
    lambda_: float
) -> np.ndarray:
    """
    Solve ridge regression using SVD.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  RIDGE REGRESSION VIA SVD                                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Ridge objective: min ||Xβ - y||² + λ||β||²                             │
    │                                                                          │
    │  SVD solution:                                                           │
    │      β_λ = Σᵢ (σᵢ² / (σᵢ² + λ)) × (uᵢᵀy) × vᵢ                        │
    │            = V × diag(σᵢ/(σᵢ²+λ)) × Uᵀy                                │
    │                                                                          │
    │  SHRINKAGE FACTOR:                                                       │
    │  ─────────────────                                                       │
    │      fᵢ = σᵢ² / (σᵢ² + λ)                                              │
    │                                                                          │
    │  • When σᵢ >> √λ:  fᵢ ≈ 1 (no shrinkage)                               │
    │  • When σᵢ << √λ:  fᵢ ≈ 0 (full shrinkage)                             │
    │  • When σᵢ = √λ:   fᵢ = 0.5 (half shrinkage)                           │
    │                                                                          │
    │  ADVANTAGES OF SVD FORM:                                                 │
    │  ────────────────────────                                                │
    │  • Understand geometry of shrinkage                                      │
    │  • Compute solutions for multiple λ efficiently                          │
    │  • Analyze which directions are shrunk                                   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix, shape (n, p)
        y: Target vector, shape (n,)
        lambda_: Ridge regularization parameter (≥ 0)

    Returns:
        Ridge regression coefficients, shape (p,)

    Example:
        >>> X = np.array([[1, 0.9], [2, 1.8], [3, 2.7]])  # Collinear
        >>> y = np.array([1, 2, 3])
        >>> beta_ridge = ridge_via_svd(X, y, lambda_=0.1)

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. U, s, V = compute_svd(X)
    2. shrinkage = s / (s**2 + lambda_)  # Note: s/(s²+λ), not s²/(s²+λ)
    3. Uty = U.T @ y
    4. return V @ (shrinkage * Uty)

    MAT223 Reference: Section 3.6
    """
    U, s, V = compute_svd(X, full_matrices=False)
    shrinkage = s / (s ** 2 + lambda_)
    Uty = U.T @ y
    return V @ (shrinkage * Uty)


def effective_degrees_of_freedom(
    X: np.ndarray,
    lambda_: float
) -> float:
    """
    Compute effective degrees of freedom for ridge regression.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  EFFECTIVE DEGREES OF FREEDOM                                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      df(λ) = Σᵢ σᵢ² / (σᵢ² + λ) = tr(X(XᵀX + λI)⁻¹Xᵀ)                │
    │                                                                          │
    │  INTERPRETATION:                                                         │
    │  ────────────────                                                        │
    │  • λ = 0:   df = p (full model, OLS)                                    │
    │  • λ → ∞:   df → 0 (null model, all shrunk to zero)                    │
    │  • λ > 0:   0 < df < p (regularized model)                              │
    │                                                                          │
    │  VISUALIZATION:                                                          │
    │  ───────────────                                                         │
    │      df │                                                                │
    │       p ┼─●                                                              │
    │         │  ╲                                                             │
    │         │   ╲                                                            │
    │     p/2 ┼────●                                                           │
    │         │     ╲                                                          │
    │         │      ╲                                                         │
    │       0 ┼───────────●─────────────────────────►                         │
    │         │    √λ*    λ                                                   │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  • Model complexity measure for comparison                               │
    │  • AIC/BIC calculation: AIC = 2×df + n×log(SSR/n)                       │
    │  • Understand shrinkage effect                                           │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix, shape (n, p)
        lambda_: Ridge regularization parameter (≥ 0)

    Returns:
        Effective degrees of freedom (between 0 and p)

    Example:
        >>> X = np.random.randn(100, 10)
        >>> effective_degrees_of_freedom(X, lambda_=0)
        10.0  # Full model
        >>> effective_degrees_of_freedom(X, lambda_=1000)
        0.5...  # Heavily regularized

    IMPLEMENTATION:
    ────────────────
    s = singular_values(X)
    return np.sum(s**2 / (s**2 + lambda_))
    """
    s = singular_values(X)
    return float(np.sum(s ** 2 / (s ** 2 + lambda_)))


def truncated_svd(
    A: np.ndarray,
    k: int,
    n_iter: int = 5,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute randomized truncated SVD for large matrices.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  RANDOMIZED SVD (for large matrices)                                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  ALGORITHM (Halko, Martinsson, Tropp 2011):                              │
    │  ──────────────────────────────────────────                              │
    │  1. Draw random matrix Ω (n × k)                                        │
    │  2. Form Y = A × Ω                       (project to random subspace)   │
    │  3. Orthonormalize: Q, _ = QR(Y)         (get orthonormal basis)        │
    │  4. Form B = Qᵀ × A                      (project A onto Q)             │
    │  5. SVD of small B: U_B, s, V = svd(B)                                  │
    │  6. Return U = Q × U_B, s, V                                            │
    │                                                                          │
    │  COMPLEXITY:                                                             │
    │  ───────────                                                             │
    │  Full SVD:       O(mn × min(m,n))                                       │
    │  Randomized:     O(mn × k)                                              │
    │                                                                          │
    │  For 10000×5000 matrix with k=50:                                       │
    │  Full SVD:       ~250 billion operations                                 │
    │  Randomized:     ~2.5 billion operations (100× faster!)                 │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  When computing embeddings for large interaction matrices:               │
    │  • Universities × Programs (50 × 200) - full SVD is fine                │
    │  • User × Item (10000 × 50000) - need randomized SVD                    │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        A: Input matrix of shape (m, n)
        k: Number of singular values/vectors to compute
        n_iter: Number of power iterations (increases accuracy)
        random_state: Seed for reproducibility

    Returns:
        Tuple of (U_k, s_k, V_k) - truncated SVD components

    Example:
        >>> A = np.random.randn(1000, 500)
        >>> U, s, V = truncated_svd(A, k=10)
        >>> U.shape, s.shape, V.shape
        ((1000, 10), (10,), (500, 10))

    IMPLEMENTATION STEPS (simplified):
    ──────────────────────────────────
    For learning, you can just wrap sklearn:
        from sklearn.utils.extmath import randomized_svd
        U, s, Vt = randomized_svd(A, n_components=k, n_iter=n_iter,
                                   random_state=random_state)
        return U, s, Vt.T

    For from-scratch implementation:
    1. Set random seed if provided
    2. Omega = np.random.randn(n, k)
    3. Y = A @ Omega
    4. for _ in range(n_iter):  # Power iterations
           Y = A @ (A.T @ Y)
    5. Q, _ = np.linalg.qr(Y)
    6. B = Q.T @ A
    7. U_B, s, Vt = np.linalg.svd(B, full_matrices=False)
    8. U = Q @ U_B
    9. return U[:, :k], s[:k], Vt[:k, :].T
    """
    m, n = A.shape
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random.RandomState()

    # Step 1: Random projection
    Omega = rng.randn(n, k)
    Y = A @ Omega

    # Step 2: Power iterations for improved accuracy
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)

    # Step 3: QR factorization to get orthonormal basis
    Q, _ = np.linalg.qr(Y)

    # Step 4: Project A onto the low-dimensional subspace
    B = Q.T @ A

    # Step 5: SVD of the small matrix B
    U_B, s, Vt = np.linalg.svd(B, full_matrices=False)

    # Step 6: Recover U in the original space
    U = Q @ U_B

    return U[:, :k], s[:k], Vt[:k, :].T


# =============================================================================
#                           TODO LIST
# =============================================================================
"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TODO                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  HIGH PRIORITY (core functionality):                                         │
│  ────────────────────────────────────                                        │
│  [ ] Implement compute_svd() - basic SVD wrapper                            │
│  [ ] Implement singular_values() - just the values                          │
│  [ ] Implement condition_number() - for stability checking                  │
│  [ ] Implement matrix_rank() - for design matrix validation                 │
│  [ ] Implement low_rank_approximation() - for embeddings                    │
│                                                                              │
│  MEDIUM PRIORITY (analysis & diagnostics):                                   │
│  ─────────────────────────────────────────                                   │
│  [ ] Implement explained_variance_ratio() - for choosing k                  │
│  [ ] Implement choose_rank() - automatic rank selection                     │
│  [ ] Implement approximation_error() - quantify compression loss            │
│  [ ] Implement reconstruct_from_svd() - verify decomposition                │
│                                                                              │
│  MEDIUM PRIORITY (ridge regression):                                         │
│  ────────────────────────────────────                                        │
│  [ ] Implement ridge_via_svd() - understand shrinkage                       │
│  [ ] Implement effective_degrees_of_freedom() - model complexity            │
│  [ ] Implement svd_solve() - general least squares                          │
│                                                                              │
│  LOW PRIORITY (advanced):                                                    │
│  ─────────────────────────                                                   │
│  [ ] Implement truncated_svd() - for large matrices                         │
│                                                                              │
│  TESTING:                                                                    │
│  ────────                                                                    │
│  [ ] Verify SVD: A ≈ U @ diag(s) @ V.T                                      │
│  [ ] Verify orthonormality: U.T @ U ≈ I, V.T @ V ≈ I                        │
│  [ ] Compare condition_number() to np.linalg.cond()                          │
│  [ ] Verify low_rank_approximation error formula                             │
│  [ ] Test ridge_via_svd against sklearn Ridge                                │
│                                                                              │
│  VISUALIZATION IDEAS:                                                        │
│  ─────────────────────                                                       │
│  [ ] Plot singular value spectrum (scree plot)                               │
│  [ ] Plot cumulative variance explained                                      │
│  [ ] Visualize SVD as rotation-stretch-rotation (2D example)                │
│  [ ] Plot shrinkage factors for different λ                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    print("SVD and Low-Rank Approximation Module")
    print("=" * 50)
    print()
    print("Key concepts to understand:")
    print("  1. SVD decomposes ANY matrix as A = UΣVᵀ")
    print("  2. Singular values reveal matrix 'importance' per dimension")
    print("  3. Low-rank approximation ≈ embeddings")
    print("  4. Condition number = numerical stability measure")
    print()
    print("After implementing, try:")
    print("  >>> A = np.random.randn(100, 50)")
    print("  >>> U, s, V = compute_svd(A)")
    print("  >>> print(f'Condition number: {condition_number(A):.2f}')")
    print("  >>> print(f'Rank: {matrix_rank(A)}')")
