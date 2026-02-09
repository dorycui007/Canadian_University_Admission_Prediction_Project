"""
Vector Operations Module for University Admissions Prediction System
====================================================================

This module implements fundamental vector operations from scratch, serving as
the mathematical foundation for all ML computations in this project.

MAT223 REFERENCE: Sections 4.1 (Vectors), 4.2 (Dot Product and Projections)

================================================================================
                        SYSTEM ARCHITECTURE CONTEXT
================================================================================

    ┌──────────────────────────────────────────────────────────────────────────┐
    │                    WHERE THIS MODULE FITS                                │
    ├──────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   CSV Data ──► MongoDB ──► [THIS MODULE] ──► Models ──► Predictions      │
    │                               vectors.py                                 │
    │                                  │                                       │
    │                                  ▼                                       │
    │                           ┌─────────────┐                                │
    │                           │ Core Vector │                                │
    │                           │ Operations: │                                │
    │                           │ • dot()     │                                │
    │                           │ • norm()    │                                │
    │                           │ • angle()   │                                │
    │                           │ • cosine()  │                                │
    │                           └─────────────┘                                │
    │                                  │                                       │
    │                    ┌─────────────┼─────────────┐                         │
    │                    ▼             ▼             ▼                         │
    │              projections.py  matrices.py   models/                       │
    │                                                                          │
    └──────────────────────────────────────────────────────────────────────────┘

================================================================================
                        VECTOR FUNDAMENTALS
================================================================================

A vector is an ordered tuple of numbers representing a point or direction in
n-dimensional space:

                    x = (x₁, x₂, ..., xₙ) ∈ ℝⁿ

    VISUAL: 2D Vector
    ─────────────────
           y
           ▲
           │      ╱ → x = (3, 4)
           │    ╱
           │  ╱
      4 ───┼─●
           │╱│
           └─┼───────────► x
             3

    In this project, each student application is a FEATURE VECTOR:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Student Application Feature Vector                                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  x = [ 92.5,  1,  0,  0, ...,  1,  0,  0, ... ]                         │
    │        │      │   │   │        │   │   │                                │
    │        │      └───┴───┴────────┴───┴───┴──── One-hot encoded            │
    │        │      is_UofT  is_Waterloo  is_CS  is_Eng  (categorical vars)   │
    │        │                                                                │
    │        └─ Normalized average (continuous feature)                       │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                        DOT PRODUCT: THE CORE OPERATION
================================================================================

The dot product (inner product) is the fundamental operation in ML:

                    x · y = x^T y = Σᵢ xᵢyᵢ

    VISUAL: Dot Product Geometry
    ────────────────────────────
           ▲ y
           │  ╲
           │   ╲   x · y = ||x|| ||y|| cos(θ)
           │  θ ╲
           │─────●──────► x
           │

    When θ = 0°:   x · y = ||x|| ||y||     (same direction, max similarity)
    When θ = 90°:  x · y = 0               (orthogonal, no similarity)
    When θ = 180°: x · y = -||x|| ||y||    (opposite, max dissimilarity)

    PROJECT USE:
    ─────────────
    • Predictions: ŷ = x^T β  (dot product of features with coefficients)
    • Similarity:  sim(prog_a, prog_b) = embed_a · embed_b
    • Attention:   score = query · key

================================================================================
                        COMPONENT INTERACTIONS
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  HOW VECTOR OPERATIONS FLOW THROUGH THE SYSTEM                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  1. FEATURE EXTRACTION (features/encoders.py)                           │
    │     ┌─────────────────────────────────────────────────────────────┐     │
    │     │ Raw: avg=92.5, uni="UofT", prog="CS"                        │     │
    │     │                        │                                    │     │
    │     │                        ▼                                    │     │
    │     │ Vector: [92.5, 1, 0, 0, ..., 1, 0, 0, ...]                  │     │
    │     └─────────────────────────────────────────────────────────────┘     │
    │                              │                                          │
    │                              ▼                                          │
    │  2. PREDICTION (models/logistic.py)                                     │
    │     ┌─────────────────────────────────────────────────────────────┐     │
    │     │ logit = dot(x, beta)  ◄── uses vectors.dot()                │     │
    │     │ prob = sigmoid(logit)                                       │     │
    │     └─────────────────────────────────────────────────────────────┘     │
    │                              │                                          │
    │                              ▼                                          │
    │  3. SIMILARITY (models/embeddings.py)                                   │
    │     ┌─────────────────────────────────────────────────────────────┐     │
    │     │ sim = cosine_similarity(embed_a, embed_b)  ◄── uses         │     │
    │     │                                            vectors.cosine() │     │
    │     └─────────────────────────────────────────────────────────────┘     │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                        MATHEMATICAL THEOREMS
================================================================================

    CAUCHY-SCHWARZ INEQUALITY:
    ──────────────────────────
    For any vectors x, y ∈ ℝⁿ:

        |x · y| ≤ ||x|| · ||y||

    Equality holds iff x and y are parallel (one is a scalar multiple of other)

    WHY IT MATTERS: Guarantees cosine similarity is in [-1, 1]


    TRIANGLE INEQUALITY:
    ────────────────────
    For any vectors x, y ∈ ℝⁿ:

        ||x + y|| ≤ ||x|| + ||y||

    WHY IT MATTERS: Distance metrics are valid (satisfy metric axioms)

================================================================================
                        IMPLEMENTATION GUIDE
================================================================================

    NUMPY HINTS:
    ────────────
    • np.array(list) - Convert list to numpy array
    • np.sum(x * y)  - Element-wise multiply then sum (dot product)
    • np.sqrt(...)   - Square root
    • np.abs(...)    - Absolute value
    • np.clip(x, a, b) - Clamp values to range [a, b]
    • x.shape        - Get dimensions of array

    TESTING YOUR IMPLEMENTATIONS:
    ─────────────────────────────
    After implementing each function, verify against numpy:

        >>> import numpy as np
        >>> x, y = np.array([1, 2, 3]), np.array([4, 5, 6])
        >>> assert np.isclose(dot(x, y), np.dot(x, y))
        >>> assert np.isclose(norm(x), np.linalg.norm(x))

================================================================================
"""

from __future__ import annotations
import numpy as np
from typing import Union, List, Tuple
import math

# Type alias for vectors - can be numpy array or list
Vector = Union[np.ndarray, List[float]]


def _ensure_numpy(v: Vector) -> np.ndarray:  # Private Method
    """
    Convert input to numpy array if not already.

    This helper ensures all functions work with both Python lists and numpy arrays,
    providing flexibility for callers while maintaining consistent internal processing.

    Args:
        v: Input vector as list or numpy array

    Returns:
        Numpy array representation of the vector

    Example:
        >>> _ensure_numpy([1, 2, 3])
        array([1., 2., 3.])

    IMPLEMENTATION HINT:
    ────────────────────
    • Check if v is already np.ndarray using isinstance()
    • If yes, convert to float dtype with .astype(float)
    • If no, create new array with np.array(v, dtype=float)
    """
    if isinstance(v, np.ndarray):
        return v.astype(float)
    else:
        return np.array(v, dtype=float)


def add(x: Vector, y: Vector) -> np.ndarray:
    """
    Add two vectors element-wise.

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  VECTOR ADDITION                                                         │
    ├──────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      x = [x₁, x₂, ..., xₙ]                                               │
    │      y = [y₁, y₂, ..., yₙ]                                               │
    │  x + y = [x₁+y₁, x₂+y₂, ..., xₙ+yₙ]                                      │
    │                                                                          │
    │  VISUAL (2D):                                                            │
    │                     ╱→ x + y                                             │
    │                   ╱                                                      │
    │            y →  ╱                                                        │
    │               ╱                                                          │
    │             ●─────────→ x                                                │
    │           origin                                                         │
    │                                                                          │
    │  TIP: Vector addition follows the "parallelogram rule"                   │
    │                                                                          │
    └──────────────────────────────────────────────────────────────────────────┘

    Args:
        x: First vector of shape (n,)
        y: Second vector of shape (n,)
        ("shape" of a vector refers to its dimensions - specifically how many elements\
           are inside. shape(n, ) means it is a 1-dimensional array or a flat list)
        Examples:
            v1 = np.shape([1, 2, 3])
            v1.shape => (3, ) because it is a vector in R^3 and has 3 elements in it

    Returns:
        Sum vector of shape (n,)

    Raises:
        ValueError: If vectors have different dimensions

    Example:
        >>> add([1, 2, 3], [4, 5, 6])
        array([5., 7., 9.])

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Convert both inputs to numpy arrays using _ensure_numpy()
    2. Check that shapes match, raise ValueError if not
    3. Return element-wise sum (numpy handles this with + operator)

    MAT223 Reference: Section 4.1 - Vector Addition
    """
    # Ensure both arguments are numpy arrays
    x = _ensure_numpy(x)
    y = _ensure_numpy(y)

    # Check if the shape match using .shape()
    if x.shape != y.shape:
        raise ValueError(f"Add: Shape mismatch: {x.shape} and {y.shape}")

    # Vector addition
    # x + y = [x_1 + y_1, x_2 + y_2, x_3 + y_3, ..., x_n + y_n]
    return x + y  


def scale(alpha: float, x: Vector) -> np.ndarray:
    """
    Multiply a vector by a scalar.

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  SCALAR MULTIPLICATION                                                   │
    ├──────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      αx = [αx₁, αx₂, ..., αxₙ]                                           │
    │                                                                          │
    │  VISUAL (2D):                                                            │
    │                                                                          │
    │      α > 1:  Stretches the vector                                        │
    │      ●─────────→ x                                                       │
    │      ●─────────────────→ 2x                                              │
    │                                                                          │
    │      0 < α < 1:  Shrinks the vector                                      │
    │      ●─────────→ x                                                       │
    │      ●────→ 0.5x                                                         │
    │                                                                          │
    │      α < 0:  Reverses direction                                          │
    │      ←─────────● x                                                       │
    │      ─────────→                                                          │
    │                -x                                                        │
    │                                                                          │
    └──────────────────────────────────────────────────────────────────────────┘

    Args:
        alpha: Scalar multiplier
        x: Vector of shape (n,)

    Returns:
        Scaled vector of shape (n,)

    Example:
        >>> scale(2.0, [1, 2, 3])
        array([2., 4., 6.])

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Convert x to numpy array
    2. Return alpha * x (numpy broadcasts scalar multiplication)

    MAT223 Reference: Section 4.1 - Scalar Multiplication
    """
    x = _ensure_numpy(x)

    return alpha * x


def linear_combination(coeffs: List[float], vectors: List[Vector]) -> np.ndarray:
    """
    Compute a linear combination of vectors.

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  LINEAR COMBINATION                                                      │
    ├──────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      α₁v₁ + α₂v₂ + ... + αₖvₖ                                             │
    │                                                                          │
    │  PROJECT CONNECTION:                                                     │
    │  ────────────────────                                                    │
    │  Model prediction is a linear combination of features:                   │
    │                                                                          │
    │      ŷ = β₁·(avg) + β₂·(is_UofT) + β₃·(is_Waterloo) + ...                │
    │        = Σᵢ βᵢxᵢ                                                         │
    │        = x^T β  (dot product form)                                       │
    │                                                                          │
    │  This is THE fundamental operation in linear models!                     │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        coeffs: List of scalar coefficients [α₁, α₂, ..., αₖ]
        vectors: List of vectors [v₁, v₂, ..., vₖ]

    Returns:
        Resulting vector from the linear combination

    Raises:
        ValueError: If number of coefficients doesn't match number of vectors
        ValueError: If vectors have different dimensions

    Example:
        >>> v1 = [1, 0]
        >>> v2 = [0, 1]
        >>> linear_combination([3, 4], [v1, v2])
        array([3., 4.])

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Validate len(coeffs) == len(vectors)
    2. Convert all vectors to numpy arrays
    3. Check all vectors have same shape
    4. Initialize result as zeros with correct shape
    5. Loop: result += coeff * vector for each pair
    6. Return result

    MAT223 Reference: Section 4.4.1 - Span
    """
    if len(coeffs) != len(vectors):
        raise ValueError(f"Number of coefficients ({len(coeffs)}) must match number of vectors ({len(vectors)})")

    vecs = [_ensure_numpy(v) for v in vectors]

    for i in range(1, len(vecs)):
        if vecs[i].shape != vecs[0].shape:
            raise ValueError(f"All vectors must have the same shape. Vector 0 has shape {vecs[0].shape}, vector {i} has shape {vecs[i].shape}")

    result = np.zeros_like(vecs[0])
    for c, v in zip(coeffs, vecs):
        result = result + c * v
    return result


def dot(x: Vector, y: Vector) -> float:
    """
    Compute the dot product (inner product) of two vectors.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  DOT PRODUCT                                                             │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      x · y = x^T y = Σᵢ xᵢyᵢ = x₁y₁ + x₂y₂ + ... + xₙyₙ                │
    │                                                                          │
    │  GEOMETRIC INTERPRETATION:                                               │
    │  ─────────────────────────                                               │
    │                                                                          │
    │      x · y = ||x|| ||y|| cos(θ)                                         │
    │                                                                          │
    │            ▲ y                                                           │
    │            │  ╲                                                          │
    │            │   ╲ θ = angle between x and y                               │
    │            │    ╲                                                        │
    │            └─────●──────────► x                                          │
    │                                                                          │
    │  PROPERTIES:                                                             │
    │  ───────────                                                             │
    │  • Commutative: x · y = y · x                                           │
    │  • Distributive: x · (y + z) = x · y + x · z                            │
    │  • Scalar: (αx) · y = α(x · y)                                          │
    │  • Self-dot: x · x = ||x||²                                             │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  • Predictions: ŷᵢ = xᵢ^T β                                             │
    │  • Attention scores: score = query · key                                 │
    │  • Embedding similarity: sim(a, b) = embed_a · embed_b                  │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        x: First vector of shape (n,)
        y: Second vector of shape (n,)

    Returns:
        Scalar dot product value

    Raises:
        ValueError: If vectors have different dimensions

    Example:
        >>> dot([1, 2, 3], [4, 5, 6])
        32.0

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Convert both to numpy arrays
    2. Validate shapes match
    3. Compute element-wise product: x * y
    4. Sum all elements: np.sum(...)
    5. Return as float

    ALTERNATIVE (once you understand the above):
        return float(np.dot(x, y))

    MAT223 Reference: Section 4.2 - Dot Product
    """
    x = _ensure_numpy(x)
    y = _ensure_numpy(y)

    if x.shape != y.shape:
        raise ValueError(f"Dot: Shape mismatch: {x.shape} and {y.shape}")
    if x.size == 0:
        raise ValueError("Dot: Cannot compute dot product of empty vectors")

    return float(np.sum(x * y))


def norm(x: Vector, p: int = 2) -> float:
    """
    Compute the Lp norm of a vector.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  VECTOR NORMS                                                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  L2 (Euclidean) Norm - DEFAULT:                                         │
    │  ─────────────────────────────                                           │
    │      ||x||₂ = √(Σᵢ xᵢ²) = √(x · x)                                      │
    │                                                                          │
    │      This is the "length" of a vector in standard geometry.             │
    │                                                                          │
    │      VISUAL (2D):                                                        │
    │           y                                                              │
    │           ▲                                                              │
    │           │    ╱→ x = (3, 4)                                             │
    │           │  ╱                                                           │
    │        4 ─┼─●    ||x||₂ = √(3² + 4²) = √25 = 5                          │
    │           │╱                                                             │
    │           └──────────► x                                                 │
    │               3                                                          │
    │                                                                          │
    │  L1 (Manhattan) Norm:                                                    │
    │  ────────────────────                                                    │
    │      ||x||₁ = Σᵢ |xᵢ|                                                   │
    │                                                                          │
    │      ||x||₁ = |3| + |4| = 7  (walk along grid lines)                    │
    │                                                                          │
    │  L∞ (Max) Norm:                                                          │
    │  ──────────────                                                          │
    │      ||x||∞ = max(|xᵢ|)                                                 │
    │                                                                          │
    │      ||x||∞ = max(3, 4) = 4                                             │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  • L2 norm: Regularization (||β||₂² in ridge regression)                │
    │  • L2 norm: Gradient magnitude (for learning rate scaling)              │
    │  • L2 norm: Normalization (unit vectors for cosine similarity)          │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        x: Vector of shape (n,)
        p: Which Lp norm to compute (default: 2 for Euclidean)

    Returns:
        Scalar norm value (always non-negative)

    Example:
        >>> norm([3, 4])
        5.0
        >>> norm([3, 4], p=1)
        7.0

    IMPLEMENTATION STEPS:
    ─────────────────────
    For L2 (p=2): sqrt(sum of squares)
        np.sqrt(np.sum(x ** 2))

    For L1 (p=1): sum of absolute values
        np.sum(np.abs(x))

    For L∞ (p=np.inf): max absolute value
        np.max(np.abs(x))

    General Lp:
        (sum(|x_i|^p))^(1/p)

    MAT223 Reference: Section 4.2 - Norms
    """
    x = _ensure_numpy(x)

    if p == 2:
        return float(np.sqrt(np.sum(x ** 2)))
    elif p == 1:
        return float(np.sum(np.abs(x)))
    elif p == np.inf:
        return float(np.max(np.abs(x))) if x.size > 0 else 0.0
    else:
        return float(np.sum(np.abs(x) ** p) ** (1.0 / p))


def distance(x: Vector, y: Vector, p: int = 2) -> float:
    """
    Compute the Lp distance between two vectors.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  VECTOR DISTANCE                                                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      d(x, y) = ||x - y||ₚ                                               │
    │                                                                          │
    │  VISUAL (2D):                                                            │
    │           y                                                              │
    │           ▲                                                              │
    │           │    ● y                                                       │
    │           │   /│                                                         │
    │           │ d/ │                                                         │
    │           │/   │                                                         │
    │           ●────┘ x                                                       │
    │           └──────────► x                                                 │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  • Loss function: ||y - ŷ||² = sum of squared errors                    │
    │  • Nearest neighbors: find similar programs by embedding distance       │
    │  • Gradient descent: step size proportional to ||∇L||                   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        x: First vector of shape (n,)
        y: Second vector of shape (n,)
        p: Which Lp distance to compute (default: 2)

    Returns:
        Scalar distance value (always non-negative)

    Example:
        >>> distance([0, 0], [3, 4])
        5.0

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Validate shapes match
    2. Compute difference vector: x - y
    3. Return norm of difference: norm(x - y, p)

    MAT223 Reference: Section 4.2
    """
    x = _ensure_numpy(x)
    y = _ensure_numpy(y)

    if x.shape != y.shape:
        raise ValueError(f"Distance: Shape mismatch: {x.shape} and {y.shape}")

    return norm(x - y, p)


def normalize(x: Vector) -> np.ndarray:
    """
    Normalize a vector to unit length (L2 norm = 1).

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  UNIT VECTOR                                                             │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      x̂ = x / ||x||                                                      │
    │                                                                          │
    │  The unit vector points in the same direction as x but has length 1.    │
    │                                                                          │
    │  VISUAL:                                                                 │
    │           ▲                                                              │
    │           │    ╱→ x = (3, 4), ||x|| = 5                                 │
    │           │  ╱                                                           │
    │           │╱                                                             │
    │           ●──→ x̂ = (0.6, 0.8), ||x̂|| = 1                               │
    │           │                                                              │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  • Cosine similarity: normalize then dot product                         │
    │  • Embedding normalization: before storing in Weaviate                   │
    │  • Attention: softmax normalizes weights to sum to 1                     │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        x: Vector of shape (n,)

    Returns:
        Unit vector of shape (n,) with ||x̂|| = 1

    Raises:
        ValueError: If x is the zero vector (cannot normalize)

    Example:
        >>> normalize([3, 4])
        array([0.6, 0.8])

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Compute x_norm = norm(x)
    2. If x_norm < 1e-10, raise ValueError (zero vector)
    3. Return x / x_norm

    MAT223 Reference: Section 4.2
    """
    x = _ensure_numpy(x)
    x_norm = norm(x)
    if x_norm < 1e-10:
        raise ValueError("Cannot normalize the zero vector")
    return x / x_norm


def angle(x: Vector, y: Vector, in_degrees: bool = False) -> float:
    """
    Compute the angle between two vectors.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  ANGLE BETWEEN VECTORS                                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      cos(θ) = (x · y) / (||x|| ||y||)                                   │
    │      θ = arccos((x · y) / (||x|| ||y||))                                │
    │                                                                          │
    │  VISUAL:                                                                 │
    │           ▲ y                                                            │
    │           │  ╲                                                           │
    │           │   ╲                                                          │
    │           │  θ ╲                                                         │
    │           └─────●──────────► x                                           │
    │                                                                          │
    │  INTERPRETATION:                                                         │
    │  ────────────────                                                        │
    │  θ = 0°:   Vectors point same direction (parallel)                       │
    │  θ = 90°:  Vectors are orthogonal (perpendicular)                        │
    │  θ = 180°: Vectors point opposite directions (anti-parallel)             │
    │                                                                          │
    │  PROJECT CONNECTION:                                                     │
    │  ────────────────────                                                    │
    │  • Similar programs have small angle between embeddings                  │
    │  • Orthogonal residuals in least squares: θ(residual, X) = 90°          │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        x: First vector of shape (n,)
        y: Second vector of shape (n,)
        in_degrees: If True, return angle in degrees; otherwise radians

    Returns:
        Angle between vectors (radians by default)

    Example:
        >>> angle([1, 0], [0, 1], in_degrees=True)
        90.0
        >>> angle([1, 0], [1, 0], in_degrees=True)
        0.0

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Compute cos_theta = dot(x,y) / (norm(x) * norm(y))
    2. Clamp cos_theta to [-1, 1] using np.clip (numerical safety)
    3. Compute theta = math.acos(cos_theta)
    4. If in_degrees, convert: math.degrees(theta)
    5. Return theta

    MAT223 Reference: Section 4.2
    """
    x = _ensure_numpy(x)
    y = _ensure_numpy(y)

    x_norm = norm(x)
    y_norm = norm(y)

    if x_norm < 1e-10 or y_norm < 1e-10:
        raise ValueError("Cannot compute angle with zero vector")

    cos_theta = dot(x, y) / (x_norm * y_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = math.acos(cos_theta)

    if in_degrees:
        return math.degrees(theta)
    return theta


def cosine_similarity(x: Vector, y: Vector) -> float:
    """
    Compute the cosine similarity between two vectors.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  COSINE SIMILARITY                                                       │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      cos_sim(x, y) = (x · y) / (||x|| ||y||) = cos(θ)                   │
    │                                                                          │
    │  RANGE: [-1, 1]                                                          │
    │  ───────────────                                                         │
    │   1:  Identical direction (most similar)                                 │
    │   0:  Orthogonal (no similarity)                                         │
    │  -1:  Opposite direction (most dissimilar)                               │
    │                                                                          │
    │  WHY COSINE INSTEAD OF DOT PRODUCT?                                      │
    │  ───────────────────────────────────                                     │
    │  Cosine similarity is SCALE INVARIANT:                                   │
    │                                                                          │
    │      cos_sim([1, 2], [2, 4]) = 1.0  (same direction)                    │
    │      dot([1, 2], [2, 4]) = 10       (magnitude-dependent)                │
    │                                                                          │
    │  PROJECT USE:                                                            │
    │  ────────────                                                            │
    │  • Compare program embeddings: "How similar is UofT CS to Waterloo CS?"  │
    │  • Weaviate uses cosine similarity for vector search by default          │
    │  • Attention scores before softmax normalization                         │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        x: First vector of shape (n,)
        y: Second vector of shape (n,)

    Returns:
        Cosine similarity in range [-1, 1]

    Raises:
        ValueError: If either vector is zero

    Example:
        >>> cosine_similarity([1, 0], [1, 0])
        1.0
        >>> cosine_similarity([1, 0], [0, 1])
        0.0
        >>> cosine_similarity([1, 0], [-1, 0])
        -1.0

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Compute x_norm = norm(x), y_norm = norm(y)
    2. If either < 1e-10, raise ValueError
    3. Return dot(x, y) / (x_norm * y_norm)

    MAT223 Reference: Section 4.2
    """
    x = _ensure_numpy(x)
    y = _ensure_numpy(y)

    x_norm = norm(x)
    y_norm = norm(y)

    if x_norm < 1e-10 or y_norm < 1e-10:
        raise ValueError("Cannot compute cosine similarity with zero vector")

    return dot(x, y) / (x_norm * y_norm)


def is_orthogonal(x: Vector, y: Vector, tol: float = 1e-10) -> bool:
    """
    Check if two vectors are orthogonal (perpendicular).

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  ORTHOGONALITY                                                           │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      x ⊥ y  ⟺  x · y = 0                                               │
    │                                                                          │
    │  VISUAL:                                                                 │
    │           ▲ y                                                            │
    │           │                                                              │
    │           │                                                              │
    │           │                                                              │
    │           └───────────────► x                                            │
    │                                                                          │
    │  PROJECT USE (CRITICAL):                                                 │
    │  ─────────────────────────                                               │
    │  In least squares, the RESIDUAL must be orthogonal to ALL columns of X: │
    │                                                                          │
    │      r = y - Xβ̂                                                         │
    │      r ⊥ col(X)  ⟺  X^T r = 0                                          │
    │                                                                          │
    │  This is the DEFINING PROPERTY of least squares solution!                │
    │                                                                          │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │           y                                                      │    │
    │  │           ▲                                                      │    │
    │  │           │  ╲                                                   │    │
    │  │           │   ╲ r (residual)                                     │    │
    │  │           │    ╲                                                 │    │
    │  │  col(X): ─┼─────●───────────  (projection Xβ̂ is here)           │    │
    │  │    plane  │     └── ŷ = Xβ̂                                      │    │
    │  │           │                                                      │    │
    │  │           r ⊥ plane (orthogonal to column space)                 │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        x: First vector of shape (n,)
        y: Second vector of shape (n,)
        tol: Tolerance for numerical comparison

    Returns:
        True if vectors are orthogonal, False otherwise

    Example:
        >>> is_orthogonal([1, 0], [0, 1])
        True
        >>> is_orthogonal([1, 1], [1, 1])
        False

    IMPLEMENTATION:
    ────────────────
    Return abs(dot(x, y)) < tol

    MAT223 Reference: Section 4.2, 4.7.1
    """
    return abs(dot(x, y)) < tol


def is_unit(x: Vector, tol: float = 1e-10) -> bool:
    """
    Check if a vector has unit length (norm = 1).

    Args:
        x: Vector to check
        tol: Tolerance for numerical comparison

    Returns:
        True if ||x|| ≈ 1, False otherwise

    Example:
        >>> is_unit([0.6, 0.8])
        True
        >>> is_unit([3, 4])
        False

    IMPLEMENTATION:
    ────────────────
    Return abs(norm(x) - 1.0) < tol
    """
    return abs(norm(x) - 1.0) < tol


def verify_cauchy_schwarz(x: Vector, y: Vector) -> Tuple[bool, float, float]:
    """
    Verify the Cauchy-Schwarz inequality: |x · y| ≤ ||x|| ||y||

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  CAUCHY-SCHWARZ INEQUALITY                                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      |x · y| ≤ ||x|| · ||y||                                            │
    │                                                                          │
    │  Equality holds if and only if x and y are PARALLEL                      │
    │  (one is a scalar multiple of the other)                                 │
    │                                                                          │
    │  WHY IT MATTERS:                                                         │
    │  ────────────────                                                        │
    │  • Guarantees cosine similarity is bounded: -1 ≤ cos_sim ≤ 1            │
    │  • Foundation for triangle inequality                                    │
    │  • Proves projection is "closest point"                                  │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        x: First vector
        y: Second vector

    Returns:
        Tuple of (is_satisfied, |x·y|, ||x||·||y||)

    Example:
        >>> verify_cauchy_schwarz([1, 2], [3, 4])
        (True, 11.0, 12.20655...)

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Compute dot_abs = abs(dot(x, y))
    2. Compute norm_product = norm(x) * norm(y)
    3. is_satisfied = dot_abs <= norm_product + 1e-10
    4. Return (is_satisfied, dot_abs, norm_product)
    """
    dot_abs = abs(dot(x, y))
    norm_product = norm(x) * norm(y)
    is_satisfied = dot_abs <= norm_product + 1e-10
    return (is_satisfied, dot_abs, norm_product)


# =============================================================================
#                           TODO LIST
# =============================================================================
"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TODO                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  HIGH PRIORITY:                                                              │
│  ─────────────                                                               │
│  [ ] Implement _ensure_numpy() helper function                               │
│  [ ] Implement add() - vector addition                                       │
│  [ ] Implement scale() - scalar multiplication                               │
│  [ ] Implement dot() - dot product (CRITICAL for predictions)               │
│  [ ] Implement norm() - L2 norm (CRITICAL for normalization)                │
│                                                                              │
│  MEDIUM PRIORITY:                                                            │
│  ────────────────                                                            │
│  [ ] Implement linear_combination() - for understanding Xβ                  │
│  [ ] Implement distance() - for loss computation                             │
│  [ ] Implement normalize() - for cosine similarity                           │
│  [ ] Implement cosine_similarity() - for embedding comparisons              │
│                                                                              │
│  LOW PRIORITY:                                                               │
│  ─────────────                                                               │
│  [ ] Implement angle() - for visualization/debugging                         │
│  [ ] Implement is_orthogonal() - for verifying least squares                │
│  [ ] Implement is_unit() - for checking normalized vectors                   │
│  [ ] Implement verify_cauchy_schwarz() - for theorem verification           │
│                                                                              │
│  TESTING:                                                                    │
│  ────────                                                                    │
│  [ ] Write unit tests in tests/test_math/test_vectors.py                    │
│  [ ] Verify all functions against numpy equivalents                          │
│  [ ] Test edge cases: zero vectors, very large/small values                  │
│  [ ] Property-based tests with hypothesis library                            │
│                                                                              │
│  ENHANCEMENTS (OPTIONAL):                                                    │
│  ─────────────────────────                                                   │
│  [ ] Add GPU support via PyTorch tensors                                     │
│  [ ] Implement batch operations for multiple vectors                         │
│  [ ] Add sparse vector support for one-hot encodings                         │
│  [ ] Add outer product: x ⊗ y = xyᵀ                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    # Quick self-test - uncomment after implementing functions
    print("Testing vector operations...")
    print("Implement the functions above and then run this test!")

    # Example test code (uncomment after implementation):
    # x = [1, 2, 3]
    # y = [4, 5, 6]
    #
    # print(f"x = {x}, y = {y}")
    # print(f"x + y = {add(x, y)}")
    # print(f"2 * x = {scale(2, x)}")
    # print(f"x · y = {dot(x, y)}")  # Should be 32
    # print(f"||x|| = {norm(x)}")    # Should be ~3.74
    #
    # # Verify Cauchy-Schwarz
    # satisfied, lhs, rhs = verify_cauchy_schwarz(x, y)
    # print(f"Cauchy-Schwarz: |x·y| = {lhs:.2f} ≤ ||x||·||y|| = {rhs:.2f}: {satisfied}")
