"""
Mathematical Visualizations for Grade Prediction System
========================================================

This module provides visualizations for mathematical concepts and intermediate
results in the prediction pipeline. It serves three critical purposes:
1. Educational: Visualize MAT223 concepts (projections, SVD, eigenvalues)
2. Debugging: Diagnose numerical issues (conditioning, convergence)
3. Interpretation: Understand what the model learned (embeddings, weights)

MAT223 REFERENCES:
    - Section 4.2: Projections (plot_vector_projection)
    - Section 3.4: Eigenvalues (plot_eigenvalue_spectrum)
    - Section 3.6: SVD Applications (plot_svd_decomposition)
    - Section 4.8: Least Squares (plot_ridge_path)

CSC148 REFERENCES:
    - Section 3.1-3.4: OOP and dataclasses (PlotConfig)
    - Section 1.3-1.5: Function design recipe (all functions)

==============================================================================
                    SYSTEM ARCHITECTURE - WHERE THIS MODULE FITS
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                     VISUALIZATION PIPELINE OVERVIEW                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│   │   src/math  │────▶│ THIS MODULE │────▶│   Outputs   │                   │
│   │             │     │  math_viz   │     │             │                   │
│   │ vectors.py  │     │             │     │ • PNG files │                   │
│   │ matrices.py │     │ Matplotlib  │     │ • Interactive│                  │
│   │ projections │     │   Plotly    │     │   figures   │                   │
│   │ qr.py       │     │             │     │ • Notebooks │                   │
│   │ svd.py      │     │             │     │             │                   │
│   └─────────────┘     └─────────────┘     └─────────────┘                   │
│                             │                                                │
│                             ▼                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    VISUALIZATION CATEGORIES                          │   │
│   ├─────────────────────────────────────────────────────────────────────┤   │
│   │                                                                      │   │
│   │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │   │
│   │  │  LINEAR ALGEBRA │  │   REGRESSION    │  │   EMBEDDINGS    │      │   │
│   │  │                 │  │                 │  │                 │      │   │
│   │  │ • Projections   │  │ • Ridge path    │  │ • t-SNE/UMAP    │      │   │
│   │  │ • SVD panels    │  │ • IRLS converg. │  │ • Similarity    │      │   │
│   │  │ • Eigenvalues   │  │ • Residuals     │  │   heatmaps      │      │   │
│   │  │ • Condition #   │  │ • Coefficients  │  │ • Clusters      │      │   │
│   │  └─────────────────┘  └─────────────────┘  └─────────────────┘      │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

==============================================================================
                    MODULE DEPENDENCIES
==============================================================================

    ┌──────────────────────────────────────────────────────────────────────┐
    │                        IMPORT HIERARCHY                               │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                       │
    │     Standard Library          Third-Party           Project Modules   │
    │    ─────────────────        ─────────────         ─────────────────   │
    │    │ dataclasses   │        │ numpy      │        │ src/math/*    │   │
    │    │ typing        │        │ matplotlib │        │               │   │
    │    │               │        │ plotly     │        │ (for testing) │   │
    │    │               │        │ sklearn    │        │               │   │
    │    │               │        │ (TSNE)     │        │               │   │
    │    └───────────────┘        └────────────┘        └───────────────┘   │
    │                                                                       │
    └──────────────────────────────────────────────────────────────────────┘

==============================================================================
                    USAGE EXAMPLES
==============================================================================

    Example 1: Visualize Vector Projection (MAT223 Day 1)
    ──────────────────────────────────────────────────────
    >>> import numpy as np
    >>> from src.visualization.math_viz import plot_vector_projection
    >>>
    >>> v = np.array([3, 4])  # Vector to project
    >>> u = np.array([1, 0])  # Vector to project onto (x-axis)
    >>>
    >>> fig, ax = plot_vector_projection(v, u)
    >>> plt.show()

    This visualizes: proj_u(v) = (v·u / u·u) * u = (3/1) * [1,0] = [3, 0]


    Example 2: Debug Ridge Regression (MAT223 Day 6-7)
    ──────────────────────────────────────────────────
    >>> lambdas = np.logspace(-4, 4, 50)
    >>> coefficients = []  # Computed from ridge_solve for each lambda
    >>>
    >>> for lam in lambdas:
    ...     beta = ridge_solve(X, y, lam)
    ...     coefficients.append(beta)
    >>>
    >>> plot_ridge_path(lambdas, np.array(coefficients), feature_names)
    >>> plt.show()


    Example 3: Understand Program Embeddings (Day 12)
    ─────────────────────────────────────────────────
    >>> # After training embedding model
    >>> embeddings = model.prog_embedding.weight.detach().numpy()
    >>> program_names = ["CS", "EE", "Biology", "Commerce", ...]
    >>>
    >>> plot_embeddings_2d(embeddings, labels=program_categories)
    >>> plot_similarity_heatmap(embeddings, program_names)

==============================================================================
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np


# =============================================================================
#                           CONFIGURATION
# =============================================================================

@dataclass
class PlotConfig:
    """
    Configuration for plot styling and output.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        PLOT CONFIGURATION                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   ┌───────────────────┐                                                  │
    │   │   PlotConfig      │                                                  │
    │   ├───────────────────┤                                                  │
    │   │ figsize: (10, 6)  │◄─── Default figure dimensions (width, height)    │
    │   │ dpi: 100          │◄─── Resolution for saved figures                 │
    │   │ style: 'seaborn'  │◄─── Matplotlib style preset                      │
    │   │ colormap: 'viridis│◄─── Default colormap for heatmaps               │
    │   │ save_format: 'png'│◄─── Output format for saved figures              │
    │   │ font_size: 12     │◄─── Base font size for labels                    │
    │   │ title_size: 14    │◄─── Font size for titles                         │
    │   │ line_width: 2.0   │◄─── Default line width for plots                 │
    │   │ marker_size: 8    │◄─── Default marker size for scatter              │
    │   │ alpha: 0.7        │◄─── Default transparency                         │
    │   └───────────────────┘                                                  │
    │                                                                          │
    │   USAGE:                                                                 │
    │   ──────                                                                 │
    │   config = PlotConfig(figsize=(12, 8), dpi=150)                         │
    │   plot_vector_projection(v, u, config=config)                            │
    │                                                                          │
    │   CSC148 CONNECTION:                                                     │
    │   ───────────────────                                                    │
    │   This is a @dataclass - Python's built-in way to create classes        │
    │   that are primarily containers for data. The @dataclass decorator      │
    │   automatically generates __init__, __repr__, and __eq__ methods.        │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Representation Invariants:
        - figsize[0] > 0 and figsize[1] > 0
        - dpi > 0
        - 0.0 <= alpha <= 1.0
        - font_size > 0 and title_size > 0

    Attributes:
        figsize: Tuple of (width, height) in inches
        dpi: Dots per inch for raster output
        style: Matplotlib style name (try 'ggplot', 'dark_background', etc.)
        colormap: Matplotlib colormap name for heatmaps
        save_format: File format for saving ('png', 'pdf', 'svg')
        font_size: Base font size for axis labels
        title_size: Font size for plot titles
        line_width: Default line width for line plots
        marker_size: Default marker size for scatter plots
        alpha: Default transparency (0=transparent, 1=opaque)
    """
    figsize: Tuple[int, int] = (10, 6)
    dpi: int = 100
    style: str = 'seaborn-v0_8-whitegrid'
    colormap: str = 'viridis'
    save_format: str = 'png'
    font_size: int = 12
    title_size: int = 14
    line_width: float = 2.0
    marker_size: int = 8
    alpha: float = 0.7

    # Color palette for multi-line plots
    colors: List[str] = field(default_factory=lambda: [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
    ])


# =============================================================================
#                    LINEAR ALGEBRA VISUALIZATIONS
# =============================================================================

def plot_vector_projection(v: np.ndarray,
                           u: np.ndarray,
                           ax: Optional[Any] = None,
                           config: Optional[PlotConfig] = None) -> Tuple[Any, Any]:
    """
    Visualize vector projection in 2D.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     PROJECTION VISUALIZATION                             │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   MATHEMATICAL DEFINITION (MAT223 Section 4.2.2):                        │
    │   ─────────────────────────────────────────────                          │
    │                                                                          │
    │       proj_u(v) = (v · u / u · u) × u                                   │
    │                                                                          │
    │   This is the "shadow" of v onto the line spanned by u.                 │
    │                                                                          │
    │   VISUAL OUTPUT:                                                         │
    │   ──────────────                                                         │
    │                                                                          │
    │              ↑ v (original vector)                                       │
    │             /│                                                           │
    │            / │                                                           │
    │           /  │ r = v - proj_u(v)                                        │
    │          /   │     (residual, shown dashed)                              │
    │         /    │                                                           │
    │        ●─────●──────────────────────→ u                                  │
    │      origin  proj_u(v)                                                   │
    │              (projection point)                                          │
    │                                                                          │
    │   KEY PROPERTY: r ⊥ u (residual is perpendicular to u)                  │
    │                                                                          │
    │   PROJECT CONNECTION:                                                    │
    │   ───────────────────                                                    │
    │   In least squares regression:                                           │
    │   • v = y (target vector)                                                │
    │   • u = column space of X (design matrix)                                │
    │   • proj = Xβ̂ (predictions)                                             │
    │   • r = y - Xβ̂ (residuals)                                              │
    │                                                                          │
    │   The residuals are ORTHOGONAL to the column space!                      │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        v: Vector to project, shape (2,) for 2D visualization
        u: Vector to project onto, shape (2,)
        ax: Matplotlib axes object. If None, creates new figure.
        config: PlotConfig for styling. Uses defaults if None.

    Returns:
        Tuple of (figure, axes) matplotlib objects

    Raises:
        ValueError: If vectors are not 2D
        ValueError: If u is the zero vector

    Example:
        >>> import numpy as np
        >>> v = np.array([3, 4])
        >>> u = np.array([1, 0])  # x-axis
        >>> fig, ax = plot_vector_projection(v, u)
        >>> # Shows: v=(3,4), u=(1,0), proj=(3,0), residual=(0,4)

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Import matplotlib.pyplot as plt
    2. Create figure if ax is None: fig, ax = plt.subplots()
    3. Compute projection: proj = (np.dot(v, u) / np.dot(u, u)) * u
    4. Compute residual: r = v - proj
    5. Draw arrows using ax.annotate() or ax.arrow()
       - v: from origin to v, color blue
       - u: from origin to u (extended), color gray
       - proj: from origin to proj, color green
       - r: from proj to v, dashed red line
    6. Add right angle marker at proj point
    7. Add labels and legend
    8. Set equal aspect ratio: ax.set_aspect('equal')

    MATPLOTLIB ARROW SYNTAX:
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle='->', color='blue'))
    """
    pass


def plot_projection_onto_subspace(y: np.ndarray,
                                   X: np.ndarray,
                                   config: Optional[PlotConfig] = None) -> Tuple[Any, Any]:
    """
    Visualize projection of y onto column space of X (3D for 2-column X).

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   SUBSPACE PROJECTION (3D)                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   When X has 2 columns, col(X) is a PLANE in R^n.                       │
    │   We project y onto this plane.                                          │
    │                                                                          │
    │                     y (target vector)                                    │
    │                     ●                                                    │
    │                    /│                                                    │
    │                   / │                                                    │
    │                  /  │ r = y - Xβ̂                                        │
    │                 /   │                                                    │
    │       col(X)  /    │                                                    │
    │       ┌──────●─────┤                                                    │
    │       │     Xβ̂     │                                                    │
    │       │  (projection)                                                    │
    │       │             │                                                    │
    │       └─────────────┘                                                    │
    │           plane                                                          │
    │                                                                          │
    │   NORMAL EQUATIONS: X^T(y - Xβ̂) = 0                                     │
    │   This says: residual r is orthogonal to ALL columns of X               │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        y: Target vector, shape (n,) where n >= 3 for 3D visualization
        X: Design matrix, shape (n, 2) for 2D subspace
        config: Plot configuration

    Returns:
        Tuple of (figure, axes) for 3D plot

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Use Axes3D from mpl_toolkits.mplot3d
    2. Solve least squares: beta = np.linalg.lstsq(X, y)[0]
    3. Compute projection: proj = X @ beta
    4. Create meshgrid for the plane col(X)
    5. Plot plane as surface with transparency
    6. Plot y, proj, and residual vector
    """
    pass


def plot_svd_decomposition(A: np.ndarray,
                           config: Optional[PlotConfig] = None) -> Tuple[Any, Any]:
    """
    Visualize SVD decomposition A = UΣV^T.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     SVD VISUALIZATION                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   SINGULAR VALUE DECOMPOSITION (MAT223 Section 3.6):                    │
    │   ──────────────────────────────────────────────────                    │
    │                                                                          │
    │       A = U Σ V^T                                                        │
    │                                                                          │
    │   Where:                                                                 │
    │   • U is m×m orthogonal (columns are left singular vectors)              │
    │   • Σ is m×n diagonal (singular values σ₁ ≥ σ₂ ≥ ... ≥ 0)               │
    │   • V^T is n×n orthogonal (rows are right singular vectors)             │
    │                                                                          │
    │   MULTI-PANEL FIGURE:                                                    │
    │   ────────────────────                                                   │
    │                                                                          │
    │   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐                 │
    │   │    A    │ = │    U    │ × │    Σ    │ × │   V^T   │                 │
    │   │  (m×n)  │   │  (m×m)  │   │  (m×n)  │   │  (n×n)  │                 │
    │   │         │   │ ortho-  │   │ diagonal│   │ ortho-  │                 │
    │   │ heatmap │   │ gonal   │   │ singular│   │ gonal   │                 │
    │   │         │   │         │   │ values  │   │         │                 │
    │   └─────────┘   └─────────┘   └─────────┘   └─────────┘                 │
    │                                                                          │
    │   ┌─────────────────────────────────────────────────────┐               │
    │   │              SINGULAR VALUE SPECTRUM                 │               │
    │   │   σ                                                  │               │
    │   │   │ ██                                               │               │
    │   │   │ ██ ██                                            │               │
    │   │   │ ██ ██ ██                                         │               │
    │   │   │ ██ ██ ██ ██ ▪▪ ▪▪ ▪▪ ▪▪                          │               │
    │   │   └────────────────────────────────→ index           │               │
    │   │                     ↑                                │               │
    │   │              effective rank                          │               │
    │   └─────────────────────────────────────────────────────┘               │
    │                                                                          │
    │   PROJECT CONNECTION:                                                    │
    │   ───────────────────                                                    │
    │   • Singular values reveal effective rank of design matrix               │
    │   • Small σᵢ → ill-conditioning → need regularization                   │
    │   • SVD of X gives low-rank approximation for embeddings                 │
    │   • Condition number = σ_max / σ_min                                    │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        A: Matrix to decompose, shape (m, n)
        config: Plot configuration

    Returns:
        Tuple of (figure, axes_array) where axes_array has 5 subplots

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Compute SVD: U, s, Vt = np.linalg.svd(A, full_matrices=False)
    2. Create figure with GridSpec for 2 rows
       - Top row: 4 panels (A, U, Σ, V^T)
       - Bottom row: 1 panel (singular value spectrum)
    3. Use imshow() for heatmaps with diverging colormap
    4. Use bar() for singular value spectrum
    5. Mark effective rank threshold (e.g., σ > 1e-10)
    6. Add annotations for matrix dimensions
    """
    pass


def plot_singular_values(singular_values: np.ndarray,
                         threshold: Optional[float] = None,
                         ax: Optional[Any] = None,
                         config: Optional[PlotConfig] = None) -> Tuple[Any, Any]:
    """
    Plot singular value spectrum.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    SINGULAR VALUE SPECTRUM                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   This plot is crucial for diagnosing matrix conditioning.               │
    │                                                                          │
    │   σ (log scale)                                                          │
    │   │                                                                      │
    │   │ ██                                                                   │
    │   │ ██ ██                                                                │
    │   │ ██ ██ ██                                                             │
    │   │ ██ ██ ██ ██                                                          │
    │   │ ██ ██ ██ ██ ██                                                       │
    │   │ ██ ██ ██ ██ ██ ██ ▪▪ ▪▪ ▪▪ ▪▪                                        │
    │   │──────────────────────────────────────── threshold (dashed)           │
    │   │                               ▪▪ ▪▪ ▪▪ ▪▪                            │
    │   └────────────────────────────────────────────→ index                   │
    │      1   2   3   4   5   6   7   8   9  10  11  12                       │
    │                        │                                                 │
    │                        └── effective rank = 6                            │
    │                                                                          │
    │   INTERPRETATION:                                                        │
    │   ────────────────                                                       │
    │   • Sharp dropoff → well-conditioned, clear rank                        │
    │   • Gradual decay → may need regularization                             │
    │   • Values near zero → rank deficiency                                  │
    │                                                                          │
    │   CONDITION NUMBER:                                                      │
    │   ──────────────────                                                     │
    │   κ(A) = σ_max / σ_min                                                  │
    │                                                                          │
    │   κ < 10:      well-conditioned                                         │
    │   κ ~ 10³:     mildly ill-conditioned                                   │
    │   κ > 10⁶:     severely ill-conditioned, need regularization            │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        singular_values: Array of singular values (sorted descending)
        threshold: Optional threshold line for effective rank
        ax: Matplotlib axes
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Create bar chart: ax.bar(range(len(s)), s)
    2. Set y-axis to log scale: ax.set_yscale('log')
    3. Add threshold line if provided: ax.axhline(threshold, linestyle='--')
    4. Label with condition number: σ[0]/σ[-1]
    5. Highlight effective rank (count σ > threshold)
    """
    pass


def plot_eigenvalue_spectrum(eigenvalues: np.ndarray,
                             matrix_name: str = "A",
                             ax: Optional[Any] = None,
                             config: Optional[PlotConfig] = None) -> Tuple[Any, Any]:
    """
    Plot eigenvalue spectrum for symmetric matrices.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    EIGENVALUE SPECTRUM                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   For symmetric matrices (like X^T X), eigenvalues are REAL.            │
    │                                                                          │
    │   λ (eigenvalue)                                                         │
    │   │                                                                      │
    │   │  ●  λ_max                                                           │
    │   │                                                                      │
    │   │     ●                                                                │
    │   │        ●                                                             │
    │   │           ●                                                          │
    │   │              ●                                                       │
    │   │                 ●  ●  ●                                              │
    │   │─────────────────────────────●──●──●───────→                          │
    │   │                                    λ_min                             │
    │   │                                                                      │
    │   └─────────────────────────────────────────────→ index                  │
    │                                                                          │
    │   FOR POSITIVE DEFINITE MATRICES (like X^T X + λI):                     │
    │   ─────────────────────────────────────────────────                      │
    │   All eigenvalues > 0, which guarantees:                                 │
    │   • Unique solution to least squares                                     │
    │   • Invertibility                                                        │
    │   • Stability of QR decomposition                                       │
    │                                                                          │
    │   PROJECT USE:                                                           │
    │   ─────────────                                                          │
    │   • X^T X eigenvalues show feature dependencies                          │
    │   • Ridge: (X^T X + λI) shifts all eigenvalues up by λ                  │
    │   • Condition number κ = λ_max / λ_min                                  │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        eigenvalues: Array of eigenvalues (will be sorted)
        matrix_name: Name for title (e.g., "X^T X")
        ax: Matplotlib axes
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)
    """
    pass


def plot_condition_number(matrices: List[np.ndarray],
                          labels: List[str],
                          config: Optional[PlotConfig] = None) -> Tuple[Any, Any]:
    """
    Compare condition numbers across matrices.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                  CONDITION NUMBER COMPARISON                             │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Shows how regularization improves conditioning.                        │
    │                                                                          │
    │   κ(A) = ||A|| × ||A⁻¹|| = σ_max / σ_min                                │
    │                                                                          │
    │   log₁₀(κ)                                                               │
    │   │                                                                      │
    │   │ ████ 10¹²  ← ill-conditioned (bad!)                                 │
    │   │ ████                                                                 │
    │   │ ████                                                                 │
    │   │ ████                                                                 │
    │   │ ████  ████ 10⁶  ← moderate (caution)                                │
    │   │ ████  ████                                                           │
    │   │ ████  ████        ████ 10²  ← well-conditioned (good!)              │
    │   │ ████  ████        ████                                               │
    │   │ ████  ████        ████                                               │
    │   └──────────────────────────────────────→                               │
    │     X^T X   X^T X      X^T X              X^T X                          │
    │    (raw)   + λ=0.01   + λ=0.1            + λ=1.0                         │
    │                                                                          │
    │   INTERPRETATION:                                                        │
    │   ────────────────                                                       │
    │   κ < 10¹:   Well-conditioned, safe to invert                           │
    │   κ ~ 10³:   Moderate, may have numerical issues                        │
    │   κ > 10⁶:   Ill-conditioned, use regularization!                       │
    │   κ = ∞:     Singular matrix, not invertible                            │
    │                                                                          │
    │   WHY RIDGE HELPS:                                                       │
    │   ─────────────────                                                      │
    │   κ(X^T X + λI) = (σ²_max + λ) / (σ²_min + λ)                          │
    │                                                                          │
    │   Adding λ to all eigenvalues brings them closer together,               │
    │   reducing the condition number.                                         │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        matrices: List of matrices to compare
        labels: Names for each matrix
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Compute condition number: np.linalg.cond(A)
    2. Use log scale: ax.set_yscale('log')
    3. Add horizontal lines at κ = 10³ and κ = 10⁶
    4. Color bars by severity (green/yellow/red)
    """
    pass


def plot_gram_schmidt_process(vectors: np.ndarray,
                              config: Optional[PlotConfig] = None) -> Tuple[Any, Any]:
    """
    Animate or show steps of Gram-Schmidt orthogonalization.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    GRAM-SCHMIDT PROCESS                                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Transform arbitrary basis {v₁, v₂, ...} into orthonormal {q₁, q₂,...} │
    │                                                                          │
    │   ALGORITHM (MAT223 Section 4.7.2):                                      │
    │   ─────────────────────────────────                                      │
    │                                                                          │
    │   Step 1: q₁ = v₁ / ||v₁||                                              │
    │                                                                          │
    │   Step 2: w₂ = v₂ - (v₂·q₁)q₁     (remove q₁ component)                 │
    │           q₂ = w₂ / ||w₂||         (normalize)                           │
    │                                                                          │
    │   Step k: wₖ = vₖ - Σⱼ<ₖ (vₖ·qⱼ)qⱼ  (remove all previous components)   │
    │           qₖ = wₖ / ||wₖ||                                               │
    │                                                                          │
    │   VISUALIZATION (2D):                                                    │
    │   ────────────────────                                                   │
    │                                                                          │
    │       v₂                   q₂ (orthogonal to q₁)                         │
    │       ↗                    ↑                                             │
    │      /                     │                                             │
    │     /                      │                                             │
    │    /                       │                                             │
    │   ●───────→ v₁   →   ●─────┴────→ q₁                                    │
    │                                                                          │
    │   PROJECT CONNECTION:                                                    │
    │   ───────────────────                                                    │
    │   Gram-Schmidt is the foundation of QR factorization!                    │
    │   Q = [q₁ | q₂ | ... | qₙ] has orthonormal columns.                     │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        vectors: Matrix where columns are vectors to orthogonalize
        config: Plot configuration

    Returns:
        Tuple of (figure, axes) showing before/after
    """
    pass


# =============================================================================
#                    REGRESSION VISUALIZATIONS
# =============================================================================

def plot_ridge_path(lambdas: np.ndarray,
                    coefficients: np.ndarray,
                    feature_names: Optional[List[str]] = None,
                    optimal_lambda: Optional[float] = None,
                    config: Optional[PlotConfig] = None) -> Tuple[Any, Any]:
    """
    Plot ridge coefficient paths as λ varies.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        RIDGE PATH                                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   As regularization λ increases, coefficients shrink toward zero.        │
    │                                                                          │
    │   β (coefficient)                                                        │
    │   │                                                                      │
    │   │  ═════════════════ GPA (strong, stable)                             │
    │   │   ══════════════════════                                            │
    │   │    ══════════════════════════                                       │
    │   │     ═══════════════════════════════ UofT                            │
    │   0├────────────────────────────────────────────────→                    │
    │   │              │                                                       │
    │   │              │ optimal λ                                             │
    │   │  ────────────┼─────────────────────────────────                      │
    │   │              │     ══════════════════ McGill                        │
    │   │              │                                                       │
    │   │              │                                                       │
    │   └──────────────┴───────────────────────────────────→ λ (log scale)    │
    │        10⁻⁴     10⁻²     1      10²      10⁴                            │
    │                                                                          │
    │   INTERPRETATION:                                                        │
    │   ────────────────                                                       │
    │   • Coefficients that stay large → robust, important features            │
    │   • Coefficients that shrink quickly → less stable, may be noise        │
    │   • All coefficients → 0 as λ → ∞                                       │
    │   • OLS solution at λ = 0 (leftmost)                                    │
    │                                                                          │
    │   MATHEMATICAL INSIGHT:                                                  │
    │   ─────────────────────                                                  │
    │   β_ridge = (X^T X + λI)⁻¹ X^T y                                        │
    │                                                                          │
    │   As λ increases:                                                        │
    │   • λI term dominates → β shrinks                                       │
    │   • Bias increases, variance decreases                                  │
    │   • Condition number improves                                           │
    │                                                                          │
    │   CHOOSING λ:                                                            │
    │   ────────────                                                           │
    │   Use cross-validation to find optimal λ (vertical line in plot)        │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        lambdas: Array of regularization values, shape (n_lambdas,)
        coefficients: Coefficient matrix, shape (n_lambdas, n_features)
        feature_names: Optional names for legend
        optimal_lambda: If provided, draws vertical line at this λ
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)

    Preconditions:
        - len(lambdas) == coefficients.shape[0]
        - lambdas are sorted in ascending order

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Create semilog-x plot: ax.semilogx(lambdas, coefficients[:, i])
    2. Loop over features, plot each as a line
    3. Add horizontal line at y=0
    4. Add vertical line at optimal_lambda if provided
    5. Add legend (consider placing outside plot if many features)
    6. Label axes: x="λ (regularization strength)", y="Coefficient value"
    """
    pass


def plot_irls_convergence(losses: List[float],
                          beta_norms: Optional[List[float]] = None,
                          tolerance: Optional[float] = None,
                          config: Optional[PlotConfig] = None) -> Tuple[Any, Any]:
    """
    Plot IRLS convergence (loss and optionally coefficient changes).

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     IRLS CONVERGENCE                                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   IRLS = Iteratively Reweighted Least Squares                           │
    │   This is a Newton-like method for logistic regression.                  │
    │                                                                          │
    │   PANEL 1: Loss over iterations                                          │
    │   ─────────────────────────────────                                      │
    │                                                                          │
    │   Loss (negative log-likelihood)                                         │
    │   │╲                                                                     │
    │   │ ╲                                                                    │
    │   │  ╲                                                                   │
    │   │   ╲                                                                  │
    │   │    ╲___                                                              │
    │   │        ──────────────────── converged                                │
    │   │                                                                      │
    │   └─────────────────────────────────────→ Iteration                      │
    │     1    2    3    4    5    6    7    8                                 │
    │                                                                          │
    │   PANEL 2: Coefficient change per iteration                              │
    │   ─────────────────────────────────────────                              │
    │                                                                          │
    │   ||β_new - β_old||                                                      │
    │   │ ●                                                                    │
    │   │  ●                                                                   │
    │   │   ●                                                                  │
    │   │    ●                                                                 │
    │   │     ●                                                                │
    │   │      ●  ●  ●                                                         │
    │   │─────────────────────── tolerance threshold                           │
    │   └─────────────────────────────────────→ Iteration                      │
    │                                                                          │
    │   TYPICAL BEHAVIOR:                                                      │
    │   ──────────────────                                                     │
    │   • Rapid decrease in first 3-5 iterations (Newton convergence)          │
    │   • Convergence in 5-15 iterations for well-conditioned problems         │
    │   • Slow convergence → ill-conditioning or wrong learning rate           │
    │   • Oscillation → numerical instability, increase λ                      │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        losses: List of loss values per iteration
        beta_norms: Optional list of ||β_new - β_old|| per iteration
        tolerance: Convergence threshold to show as horizontal line
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Create 1x2 subplot grid (or 1x1 if no beta_norms)
    2. Plot loss: ax.plot(range(len(losses)), losses)
    3. Add horizontal tolerance line if provided
    4. Mark final iteration where convergence achieved
    5. Use log scale for y-axis if values span many orders of magnitude
    """
    pass


def plot_residuals(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   feature_values: Optional[np.ndarray] = None,
                   config: Optional[PlotConfig] = None) -> Tuple[Any, Any]:
    """
    Diagnostic residual plots (4-panel grid).

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     RESIDUAL DIAGNOSTICS                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   ┌─────────────────────────────┐  ┌─────────────────────────────┐      │
    │   │  Residuals vs Fitted        │  │  Normal Q-Q Plot            │      │
    │   │                             │  │                             │      │
    │   │      ·   ·                  │  │              ···            │      │
    │   │    ·  · ·  ·   ·            │  │           ···               │      │
    │   │  ─────────────────────      │  │        ···                  │      │
    │   │    ·  ·  ·   ·   ·          │  │      ···                    │      │
    │   │      ·   ·                  │  │   ···                       │      │
    │   │                             │  │  ·                          │      │
    │   │  Should show no pattern     │  │  Points follow diagonal     │      │
    │   │  (random scatter around 0)  │  │  if normally distributed    │      │
    │   └─────────────────────────────┘  └─────────────────────────────┘      │
    │                                                                          │
    │   ┌─────────────────────────────┐  ┌─────────────────────────────┐      │
    │   │  Histogram of Residuals     │  │  Scale-Location             │      │
    │   │                             │  │                             │      │
    │   │          ▄▄▄                │  │      · ·                    │      │
    │   │        ▄█████▄              │  │    ·  · · ·  ·              │      │
    │   │      ▄█████████▄            │  │  ·  ·  · ·  ·  ·            │      │
    │   │    ▄█████████████▄          │  │    ·  ·  ·   ·              │      │
    │   │                             │  │      ·  ·                   │      │
    │   │  Should be bell-shaped      │  │  Should show constant       │      │
    │   │  (approximately normal)     │  │  spread (homoscedasticity)  │      │
    │   └─────────────────────────────┘  └─────────────────────────────┘      │
    │                                                                          │
    │   WHAT TO LOOK FOR:                                                      │
    │   ──────────────────                                                     │
    │   • Pattern in residuals vs fitted → nonlinearity, add features          │
    │   • Fan shape (spread increases) → heteroscedasticity                   │
    │   • Heavy tails in Q-Q → outliers, consider robust regression           │
    │   • Clusters → missing categorical variable                             │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        y_true: Actual values
        y_pred: Predicted values
        feature_values: Optional feature for "Residuals vs Feature" plot
        config: Plot configuration

    Returns:
        Tuple of (figure, axes) with 2x2 subplot grid

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Compute residuals: r = y_true - y_pred
    2. Create 2x2 subplot grid
    3. Panel 1: Scatter(y_pred, r) with horizontal line at 0
    4. Panel 2: Use scipy.stats.probplot for Q-Q
    5. Panel 3: plt.hist(r, bins='auto')
    6. Panel 4: Scatter(y_pred, sqrt(abs(r))) for scale-location
    """
    pass


def plot_coefficient_importance(coefficients: np.ndarray,
                                feature_names: List[str],
                                top_k: Optional[int] = None,
                                config: Optional[PlotConfig] = None) -> Tuple[Any, Any]:
    """
    Horizontal bar chart of coefficient magnitudes.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   FEATURE IMPORTANCE                                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Shows which features have the largest impact on predictions.           │
    │                                                                          │
    │           Feature Importance (|coefficient|)                             │
    │                                                                          │
    │     Top 6 Average  ██████████████████████████████  +2.34                │
    │       is_UofT_CS   █████████████████████  +1.75                         │
    │    is_Waterloo_SE  ████████████████████  +1.62                          │
    │      is_McGill_EE  ██████████████  +1.21                                │
    │       is_Queen_CS  ███████████  +0.89                                   │
    │     is_Western_CS  ████████  -0.72                                      │
    │                    ◄──────────────────────────────────►                 │
    │                         Coefficient Value                                │
    │                                                                          │
    │   INTERPRETATION:                                                        │
    │   ────────────────                                                       │
    │   • Large positive → increases P(admit)                                 │
    │   • Large negative → decreases P(admit)                                 │
    │   • Near zero → little impact (or redundant)                            │
    │                                                                          │
    │   For logistic regression:                                               │
    │   • β = +1 means odds multiply by e¹ ≈ 2.7 for each unit increase      │
    │   • β = -1 means odds divide by e¹ ≈ 2.7                                │
    │                                                                          │
    │   CAUTION:                                                               │
    │   ────────                                                               │
    │   Feature importance depends on feature scaling!                         │
    │   Compare standardized coefficients for fair comparison.                 │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        coefficients: Array of coefficient values
        feature_names: Names corresponding to each coefficient
        top_k: If provided, show only top k by magnitude
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)

    Preconditions:
        - len(coefficients) == len(feature_names)

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Sort by absolute value: sorted_idx = np.argsort(np.abs(coefficients))[::-1]
    2. Take top_k if specified
    3. Use horizontal bar: ax.barh(range(n), coefs[sorted_idx])
    4. Color by sign: positive=blue, negative=red
    5. Add value labels at end of each bar
    6. Invert y-axis so largest is at top
    """
    pass


# =============================================================================
#                    EMBEDDING VISUALIZATIONS
# =============================================================================

def plot_embeddings_2d(embeddings: np.ndarray,
                       labels: Optional[np.ndarray] = None,
                       names: Optional[List[str]] = None,
                       method: str = 'tsne',
                       perplexity: float = 30.0,
                       config: Optional[PlotConfig] = None) -> Tuple[Any, Any]:
    """
    Plot embeddings in 2D using dimensionality reduction.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │               EMBEDDING SPACE VISUALIZATION                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Reduce high-dimensional embeddings to 2D for visualization.            │
    │                                                                          │
    │   METHODS:                                                               │
    │   ────────                                                               │
    │   • t-SNE: Good for local structure, clusters                           │
    │   • UMAP: Preserves more global structure, faster                       │
    │   • PCA: Linear projection, good for seeing variance                    │
    │                                                                          │
    │   EXAMPLE OUTPUT:                                                        │
    │   ────────────────                                                       │
    │                                                                          │
    │              Engineering Programs                                        │
    │                  ○ ○ ○                                                   │
    │                 ○ ○ ○ ○                                                  │
    │               ○ ○ ○ ○ ○                                                  │
    │                                                                          │
    │                            ● ● ● ●  Business                             │
    │                           ● ● ● ● ●                                      │
    │                            ● ● ●                                         │
    │          □ □ □                                                           │
    │         □ □ □ □  Sciences                        ◇ ◇ ◇                   │
    │        □ □ □ □ □                                ◇ ◇ ◇ ◇  Arts            │
    │         □ □ □                                    ◇ ◇                     │
    │                                                                          │
    │   INTERPRETATION:                                                        │
    │   ────────────────                                                       │
    │   • Close points → similar programs (model learned this!)                │
    │   • Clusters → model discovered program categories                       │
    │   • Outliers → unique programs or data issues                           │
    │                                                                          │
    │   PROJECT CONNECTION:                                                    │
    │   ───────────────────                                                    │
    │   • University embeddings: Do similar schools cluster?                   │
    │   • Program embeddings: Are CS programs near Engineering?               │
    │   • Validates that model learned meaningful representations              │
    │                                                                          │
    │   HYPERPARAMETERS:                                                       │
    │   ─────────────────                                                      │
    │   t-SNE perplexity: Controls local vs global structure                   │
    │   • Low (5-10): Tight clusters, may miss global structure               │
    │   • Medium (30): Good default                                           │
    │   • High (50+): More global structure, looser clusters                   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        embeddings: (n, d) embedding matrix
        labels: Optional cluster/category labels for coloring
        names: Optional names to annotate points
        method: 'tsne', 'umap', or 'pca'
        perplexity: t-SNE perplexity parameter
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Import: from sklearn.manifold import TSNE
    2. Reduce: reduced = TSNE(n_components=2, perplexity=perplexity).fit_transform(embeddings)
    3. Scatter plot: ax.scatter(reduced[:, 0], reduced[:, 1], c=labels)
    4. Add colorbar or legend for labels
    5. Optionally annotate points with names
    6. Note: t-SNE is stochastic, set random_state for reproducibility
    """
    pass


def plot_similarity_heatmap(embeddings: np.ndarray,
                            labels: List[str],
                            similarity_metric: str = 'cosine',
                            cluster: bool = True,
                            config: Optional[PlotConfig] = None) -> Tuple[Any, Any]:
    """
    Heatmap of pairwise similarities between embeddings.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    SIMILARITY HEATMAP                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Shows which programs/universities are similar to each other.           │
    │                                                                          │
    │             CS    EE    ME    Bio   Eng   Math  Comm                     │
    │         ┌─────────────────────────────────────────────┐                 │
    │    CS   │ ██    ██    ▓▓    ░░    ██    ██    ░░    │                 │
    │    EE   │ ██    ██    ▓▓    ░░    ██    ▓▓    ░░    │                 │
    │    ME   │ ▓▓    ▓▓    ██    ░▒    ██    ▓▓    ░░    │                 │
    │    Bio  │ ░░    ░░    ░▒    ██    ░▒    ░░    ░░    │                 │
    │    Eng  │ ██    ██    ██    ░▒    ██    ▓▓    ░░    │                 │
    │    Math │ ██    ▓▓    ▓▓    ░░    ▓▓    ██    ░░    │                 │
    │    Comm │ ░░    ░░    ░░    ░░    ░░    ░░    ██    │                 │
    │         └─────────────────────────────────────────────┘                 │
    │                                                                          │
    │   Legend: ██ = 1.0 (identical)                                          │
    │           ▓▓ = 0.7 (similar)                                            │
    │           ▒▒ = 0.4 (moderate)                                           │
    │           ░░ = 0.1 (different)                                          │
    │                                                                          │
    │   OBSERVATIONS:                                                          │
    │   ──────────────                                                         │
    │   • CS is similar to EE, Eng, Math (STEM cluster)                       │
    │   • Bio is different from most others                                   │
    │   • Commerce stands alone (Business cluster)                            │
    │                                                                          │
    │   WITH CLUSTERING:                                                       │
    │   ─────────────────                                                      │
    │   Reorder rows/columns to group similar programs together.               │
    │   Uses hierarchical clustering on similarity matrix.                     │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        embeddings: (n, d) embedding matrix
        labels: Names for each embedding
        similarity_metric: 'cosine', 'euclidean', or 'dot'
        cluster: Whether to reorder by hierarchical clustering
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Compute similarity matrix:
       - cosine: from sklearn.metrics.pairwise import cosine_similarity
       - euclidean: use negative distance
    2. If cluster: use scipy.cluster.hierarchy.linkage and dendrogram
    3. Use seaborn.heatmap for better visualization
    4. Add colorbar with appropriate label
    5. Rotate x-labels for readability
    """
    pass


def plot_embedding_clusters(embeddings: np.ndarray,
                            cluster_labels: np.ndarray,
                            cluster_names: Optional[List[str]] = None,
                            config: Optional[PlotConfig] = None) -> Tuple[Any, Any]:
    """
    Visualize clusters in embedding space with centroids.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   EMBEDDING CLUSTERS                                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Shows cluster assignments and centroids.                               │
    │                                                                          │
    │                  ★ STEM Centroid                                         │
    │              ○ ○ ○                                                       │
    │             ○ ○ ○ ○                                                      │
    │            ○ ○ ○ ○ ○                                                     │
    │             ○ ○ ○                                                        │
    │                                                                          │
    │                           ★ Business Centroid                            │
    │                          ● ● ●                                           │
    │                         ● ● ● ●                                          │
    │                          ● ●                                             │
    │                                                                          │
    │           ★ Science Centroid               ★ Arts Centroid               │
    │          □ □ □ □                           ◇ ◇ ◇                         │
    │         □ □ □ □ □                         ◇ ◇ ◇ ◇                        │
    │          □ □ □                             ◇ ◇                           │
    │                                                                          │
    │   USE CASES:                                                             │
    │   ───────────                                                            │
    │   • Validate clustering quality                                          │
    │   • Understand how model groups programs                                 │
    │   • Find outliers (points far from centroid)                            │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        embeddings: (n, d) embedding matrix
        cluster_labels: Cluster assignment for each point
        cluster_names: Optional names for each cluster
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)
    """
    pass


def plot_attention_weights(attention_matrix: np.ndarray,
                           query_labels: List[str],
                           key_labels: List[str],
                           config: Optional[PlotConfig] = None) -> Tuple[Any, Any]:
    """
    Visualize attention weights from transformer-style attention.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    ATTENTION WEIGHTS                                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Shows what the model "attends to" when making predictions.             │
    │                                                                          │
    │   Query: "UofT CS"                                                       │
    │                                                                          │
    │   Program          Attention Weight                                      │
    │   ──────────────────────────────────────────────                        │
    │   Waterloo CS      ████████████████████  0.35                           │
    │   UofT Eng         ██████████████  0.25                                 │
    │   McGill CS        ███████████  0.20                                    │
    │   Queen's CS       ██████  0.12                                         │
    │   Western Bus      ████  0.08                                           │
    │                                                                          │
    │   INTERPRETATION:                                                        │
    │   ────────────────                                                       │
    │   When predicting for UofT CS, the model looks at:                       │
    │   • Other CS programs (especially Waterloo CS)                           │
    │   • Related UofT programs (Engineering)                                  │
    │   • Less attention to unrelated programs (Western Business)              │
    │                                                                          │
    │   This helps explain predictions: "Students with your profile            │
    │   had similar outcomes at these programs..."                             │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        attention_matrix: (n_queries, n_keys) attention weights
        query_labels: Labels for query (rows)
        key_labels: Labels for keys (columns)
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)
    """
    pass


# =============================================================================
#                    MATRIX VISUALIZATIONS
# =============================================================================

def plot_matrix_heatmap(matrix: np.ndarray,
                        row_labels: Optional[List[str]] = None,
                        col_labels: Optional[List[str]] = None,
                        title: str = 'Matrix Heatmap',
                        annotate: bool = False,
                        config: Optional[PlotConfig] = None) -> Tuple[Any, Any]:
    """
    Generic matrix heatmap visualization.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    MATRIX HEATMAP                                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   General-purpose visualization for any 2D matrix.                       │
    │                                                                          │
    │   USE CASES:                                                             │
    │   ───────────                                                            │
    │   • Correlation matrices (features)                                      │
    │   • Weight matrices (neural network layers)                              │
    │   • Confusion matrices (classification)                                  │
    │   • Design matrix structure (X)                                          │
    │   • Covariance matrices (X^T X)                                          │
    │                                                                          │
    │   OPTIONS:                                                               │
    │   ────────                                                               │
    │   • annotate=True: Show values in each cell                             │
    │   • Diverging colormap for centered data (e.g., correlations)           │
    │   • Sequential colormap for positive data (e.g., counts)                │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        matrix: 2D array to visualize
        row_labels: Labels for rows
        col_labels: Labels for columns
        title: Plot title
        annotate: Whether to show values in cells
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Use ax.imshow(matrix) or seaborn.heatmap()
    2. Add colorbar: plt.colorbar()
    3. If annotate, loop and add text to each cell
    4. Rotate x-labels if many columns
    5. Use diverging colormap if data is centered (cmap='RdBu_r')
    """
    pass


def plot_correlation_matrix(X: np.ndarray,
                            feature_names: List[str],
                            threshold: float = 0.7,
                            config: Optional[PlotConfig] = None) -> Tuple[Any, Any]:
    """
    Correlation matrix with hierarchical clustering and threshold highlighting.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   CORRELATION MATRIX                                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Identifies multicollinearity in features.                              │
    │                                                                          │
    │            avg   g11   g12   UofT  UW    CS    EE                       │
    │         ┌──────────────────────────────────────────┐                    │
    │    avg  │ 1.0   0.9   0.95  0.1   0.1   0.1   0.1 │ ← highly           │
    │    g11  │ 0.9   1.0   0.85  0.1   0.1   0.1   0.1 │   correlated!      │
    │    g12  │ 0.95  0.85  1.0   0.1   0.1   0.1   0.1 │                    │
    │    UofT │ 0.1   0.1   0.1   1.0  -0.4   0.2   0.1 │                    │
    │    UW   │ 0.1   0.1   0.1  -0.4   1.0   0.3   0.2 │ ← negative         │
    │    CS   │ 0.1   0.1   0.1   0.2   0.3   1.0   0.7 │   (makes sense)    │
    │    EE   │ 0.1   0.1   0.1   0.1   0.2   0.7   1.0 │                    │
    │         └──────────────────────────────────────────┘                    │
    │                                                                          │
    │   HIGH CORRELATION (|r| > 0.7) HIGHLIGHTED:                             │
    │   • avg ↔ g11 ↔ g12: Expected, all grade measures                       │
    │   • CS ↔ EE: Related programs often applied together                    │
    │                                                                          │
    │   IMPLICATIONS:                                                          │
    │   ──────────────                                                         │
    │   • Multicollinearity → unstable coefficients                           │
    │   • May need to drop redundant features                                 │
    │   • Ridge regression handles this better than OLS                       │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: Names for each feature
        threshold: Highlight correlations above this value
        config: Plot configuration

    Returns:
        Tuple of (figure, axes)

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Compute correlation: corr = np.corrcoef(X.T)
    2. Use seaborn.clustermap for hierarchical clustering
    3. Add mask for |r| < threshold if desired
    4. Use diverging colormap centered at 0 (RdBu_r)
    5. Add annotations for high correlations
    """
    pass


# =============================================================================
#                    UTILITY FUNCTIONS
# =============================================================================

def save_figure(fig: Any,
                filename: str,
                config: Optional[PlotConfig] = None) -> None:
    """
    Save figure with proper settings.

    Args:
        fig: Matplotlib figure object
        filename: Output filename (extension determines format if not in config)
        config: Plot configuration for format and DPI

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Get config or use defaults
    2. Determine format from filename extension or config
    3. fig.savefig(filename, dpi=config.dpi, bbox_inches='tight')
    4. Print confirmation message
    """
    pass


def create_subplot_grid(n_plots: int,
                        ncols: int = 2,
                        config: Optional[PlotConfig] = None) -> Tuple[Any, Any]:
    """
    Create a grid of subplots.

    Args:
        n_plots: Total number of subplots needed
        ncols: Number of columns
        config: Plot configuration

    Returns:
        (fig, axes) tuple where axes is 2D array

    IMPLEMENTATION HINTS:
    ─────────────────────
    1. Compute nrows: nrows = ceil(n_plots / ncols)
    2. Create figure: fig, axes = plt.subplots(nrows, ncols, figsize=...)
    3. Flatten axes for easy iteration
    4. Hide unused axes if n_plots < nrows * ncols
    """
    pass


def set_style(config: Optional[PlotConfig] = None) -> None:
    """
    Set matplotlib style from config.

    IMPLEMENTATION:
    ────────────────
    import matplotlib.pyplot as plt
    plt.style.use(config.style)
    plt.rcParams.update({'font.size': config.font_size})
    """
    pass


# =============================================================================
#                              TODO LIST
# =============================================================================
"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TODO LIST                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  HIGH PRIORITY (Core functionality):                                         │
│  ──────────────────────────────────                                         │
│  [ ] plot_vector_projection() - Essential for MAT223 Day 1                  │
│      - [ ] Draw arrows using matplotlib annotations                          │
│      - [ ] Show projection and residual with different colors               │
│      - [ ] Add right-angle marker                                            │
│      - [ ] Include legend and labels                                         │
│                                                                              │
│  [ ] plot_svd_decomposition() - Essential for understanding embeddings      │
│      - [ ] Create 4-panel figure for U, Σ, V^T, and spectrum                │
│      - [ ] Add colorbars and dimension annotations                          │
│      - [ ] Highlight effective rank on spectrum                              │
│                                                                              │
│  [ ] plot_ridge_path() - Essential for hyperparameter tuning                │
│      - [ ] Semilog x-axis for lambda                                        │
│      - [ ] Color-coded coefficient lines                                     │
│      - [ ] Vertical line for optimal lambda                                  │
│      - [ ] Legend with feature names                                         │
│                                                                              │
│  MEDIUM PRIORITY (Model diagnostics):                                        │
│  ────────────────────────────────────                                        │
│  [ ] plot_irls_convergence() - Debug training                               │
│      - [ ] Loss curve with convergence threshold                            │
│      - [ ] Optional coefficient change plot                                  │
│                                                                              │
│  [ ] plot_residuals() - Model diagnostics                                   │
│      - [ ] 4-panel diagnostic grid                                          │
│      - [ ] Q-Q plot for normality check                                      │
│                                                                              │
│  [ ] plot_singular_values() - Conditioning analysis                         │
│      - [ ] Bar chart with log scale                                         │
│      - [ ] Threshold line for effective rank                                 │
│      - [ ] Condition number annotation                                       │
│                                                                              │
│  [ ] plot_condition_number() - Compare regularization effects               │
│      - [ ] Bar chart comparing matrices                                      │
│      - [ ] Color by severity (green/yellow/red)                             │
│                                                                              │
│  LOWER PRIORITY (Embeddings, enhancement):                                   │
│  ─────────────────────────────────────────                                   │
│  [ ] plot_embeddings_2d() - Visualize learned representations               │
│      - [ ] Integrate t-SNE and UMAP                                         │
│      - [ ] Colored by category                                              │
│      - [ ] Optional point annotations                                        │
│                                                                              │
│  [ ] plot_similarity_heatmap() - Program/university similarities            │
│      - [ ] Compute cosine similarity matrix                                  │
│      - [ ] Hierarchical clustering option                                    │
│      - [ ] Annotated heatmap                                                 │
│                                                                              │
│  [ ] plot_attention_weights() - Attention visualization                     │
│      - [ ] Heatmap of attention matrix                                       │
│      - [ ] Bar chart for single query                                        │
│                                                                              │
│  [ ] plot_coefficient_importance() - Feature importance                     │
│      - [ ] Horizontal bar chart                                              │
│      - [ ] Color by sign (positive/negative)                                │
│      - [ ] Top-k filtering                                                   │
│                                                                              │
│  UTILITY FUNCTIONS:                                                          │
│  ──────────────────                                                          │
│  [ ] save_figure() - Export plots                                            │
│  [ ] create_subplot_grid() - Helper for multi-panel figures                 │
│  [ ] set_style() - Apply consistent styling                                  │
│                                                                              │
│  TESTING:                                                                    │
│  ────────                                                                    │
│  [ ] Create test matrices with known SVD                                    │
│  [ ] Test with simple 2D vectors for projection                             │
│  [ ] Visual regression tests (compare output images)                        │
│  [ ] Test edge cases (singular matrices, zero vectors)                      │
│                                                                              │
│  DOCUMENTATION:                                                              │
│  ───────────────                                                             │
│  [ ] Add example notebook demonstrating all visualizations                  │
│  [ ] Screenshot gallery in docs/                                            │
│  [ ] Add interactivity with plotly for notebooks                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    print("Mathematical Visualizations Module")
    print("=" * 50)
    print("This module provides visualizations for:")
    print("  - Linear algebra concepts (projections, SVD)")
    print("  - Regression diagnostics (ridge path, residuals)")
    print("  - Embedding analysis (t-SNE, similarity heatmaps)")
    print()
    print("Import and use individual functions, e.g.:")
    print("  from src.visualization.math_viz import plot_vector_projection")
    print()
    print("See docstrings for detailed usage instructions.")
