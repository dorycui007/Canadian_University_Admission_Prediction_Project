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
import math
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    config = config or PlotConfig()
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize)
    else:
        fig = ax.figure

    # Compute projection and residual
    proj = (np.dot(v, u) / np.dot(u, u)) * u
    residual = v - proj

    # Normalise u for drawing the direction line
    u_norm = u / np.linalg.norm(u)
    # Extend u direction line beyond the vectors for context
    all_pts = np.array([v, proj, [0, 0]])
    extent = max(np.max(np.abs(all_pts)) * 1.3, 1.0)

    # Draw u direction as thin gray line
    ax.plot([-extent * u_norm[0], extent * u_norm[0]],
            [-extent * u_norm[1], extent * u_norm[1]],
            color='gray', linewidth=0.8, linestyle='-', alpha=0.4, label='u direction')

    # Draw v arrow (blue)
    ax.annotate('', xy=(v[0], v[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=config.line_width))
    ax.text(v[0] * 1.05, v[1] * 1.05, 'v', fontsize=config.font_size, color='#1f77b4')

    # Draw projection arrow (green)
    ax.annotate('', xy=(proj[0], proj[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=config.line_width))
    ax.text(proj[0] + 0.1, proj[1] - 0.3, r'proj$_u$(v)', fontsize=config.font_size - 1, color='#2ca02c')

    # Draw residual (dashed red)
    ax.annotate('', xy=(v[0], v[1]), xytext=(proj[0], proj[1]),
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=config.line_width, linestyle='dashed'))
    mid_r = proj + residual * 0.5
    ax.text(mid_r[0] + 0.1, mid_r[1], 'r', fontsize=config.font_size, color='#d62728')

    # Right-angle marker
    marker_size_px = 0.15 * extent
    # Build a small square at the projection point
    perp = np.array([-u_norm[1], u_norm[0]])
    p1 = proj + marker_size_px * u_norm
    p2 = proj + marker_size_px * u_norm + marker_size_px * perp
    p3 = proj + marker_size_px * perp
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', linewidth=1)
    ax.plot([p2[0], p3[0]], [p2[1], p3[1]], color='black', linewidth=1)

    ax.set_aspect('equal')
    ax.set_title('Vector Projection', fontsize=config.title_size)
    ax.set_xlabel('x', fontsize=config.font_size)
    ax.set_ylabel('y', fontsize=config.font_size)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    return (fig, ax)


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
    config = config or PlotConfig()

    fig = plt.figure(figsize=config.figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Solve least squares
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    proj = X @ beta
    residual = y - proj

    # Create plane mesh from column space of X
    # Use the two columns of X as basis vectors for the plane
    col1 = X[:, 0]
    col2 = X[:, 1]

    # Create a meshgrid in parameter space
    t = np.linspace(-1.5, 1.5, 10)
    s = np.linspace(-1.5, 1.5, 10)
    T, S = np.meshgrid(t, s)

    # We can only draw a plane in 3D for the first 3 components
    n = min(3, len(y))
    plane_x = T * col1[0] + S * col2[0]
    plane_y = T * col1[1] + S * col2[1]
    plane_z = T * col1[2] + S * col2[2] if n > 2 else T * 0

    ax.plot_surface(plane_x, plane_y, plane_z, alpha=0.2, color='cyan')

    # Plot origin
    ax.scatter([0], [0], [0], color='black', s=50, zorder=5)

    # Plot y vector
    ax.quiver(0, 0, 0, y[0], y[1], y[2] if n > 2 else 0,
              color='#1f77b4', arrow_length_ratio=0.1, linewidth=config.line_width, label='y')

    # Plot projection
    ax.quiver(0, 0, 0, proj[0], proj[1], proj[2] if n > 2 else 0,
              color='#2ca02c', arrow_length_ratio=0.1, linewidth=config.line_width, label=r'$X\hat{\beta}$')

    # Plot residual
    p2 = proj[2] if n > 2 else 0
    r2 = residual[2] if n > 2 else 0
    ax.quiver(proj[0], proj[1], p2, residual[0], residual[1], r2,
              color='#d62728', arrow_length_ratio=0.1, linewidth=config.line_width,
              linestyle='dashed', label='residual')

    ax.set_title('Projection onto Column Space', fontsize=config.title_size)
    ax.set_xlabel('x', fontsize=config.font_size)
    ax.set_ylabel('y', fontsize=config.font_size)
    ax.set_zlabel('z', fontsize=config.font_size)
    ax.legend(fontsize=config.font_size - 2)

    return (fig, ax)


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
    config = config or PlotConfig()

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    Sigma = np.diag(s)

    fig = plt.figure(figsize=(config.figsize[0] * 1.5, config.figsize[1] * 1.5))

    # Top row: 4 heatmap panels
    ax1 = fig.add_subplot(2, 4, 1)
    ax2 = fig.add_subplot(2, 4, 2)
    ax3 = fig.add_subplot(2, 4, 3)
    ax4 = fig.add_subplot(2, 4, 4)
    # Bottom row: singular value bar chart spanning all columns
    ax5 = fig.add_subplot(2, 1, 2)

    # Panel 1: A
    im1 = ax1.imshow(A, cmap='RdBu_r', aspect='auto')
    ax1.set_title(f'A ({A.shape[0]}x{A.shape[1]})', fontsize=config.font_size)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Panel 2: U
    im2 = ax2.imshow(U, cmap='RdBu_r', aspect='auto')
    ax2.set_title(f'U ({U.shape[0]}x{U.shape[1]})', fontsize=config.font_size)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Panel 3: Sigma (diagonal)
    im3 = ax3.imshow(Sigma, cmap='viridis', aspect='auto')
    ax3.set_title(f'Sigma ({Sigma.shape[0]}x{Sigma.shape[1]})', fontsize=config.font_size)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Panel 4: Vt
    im4 = ax4.imshow(Vt, cmap='RdBu_r', aspect='auto')
    ax4.set_title(f'V^T ({Vt.shape[0]}x{Vt.shape[1]})', fontsize=config.font_size)
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # Bottom panel: singular value bar chart
    indices = np.arange(len(s))
    ax5.bar(indices, s, color='#1f77b4', alpha=config.alpha)
    ax5.set_xlabel('Index', fontsize=config.font_size)
    ax5.set_ylabel('Singular Value', fontsize=config.font_size)
    ax5.set_title('Singular Value Spectrum', fontsize=config.title_size)
    if len(s) > 0 and s[-1] > 0:
        ax5.set_yscale('log')

    axes = np.array([ax1, ax2, ax3, ax4, ax5])
    fig.suptitle('SVD Decomposition: A = U Sigma V^T', fontsize=config.title_size + 2)
    fig.tight_layout()

    return (fig, axes)


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
    config = config or PlotConfig()
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize)
    else:
        fig = ax.figure

    indices = np.arange(len(singular_values))
    ax.bar(indices, singular_values, color='#1f77b4', alpha=config.alpha)
    ax.set_yscale('log')

    if threshold is not None:
        ax.axhline(threshold, linestyle='--', color='#d62728', linewidth=config.line_width,
                   label=f'Threshold = {threshold}')
        effective_rank = np.sum(singular_values > threshold)
        ax.set_title(f'Singular Values (effective rank = {effective_rank})', fontsize=config.title_size)
    else:
        ax.set_title('Singular Values', fontsize=config.title_size)

    ax.set_xlabel('Index', fontsize=config.font_size)
    ax.set_ylabel('Singular Value (log scale)', fontsize=config.font_size)

    # Annotate condition number
    if len(singular_values) > 0 and singular_values[-1] > 0:
        cond = singular_values[0] / singular_values[-1]
        ax.text(0.95, 0.95, f'cond = {cond:.2e}', transform=ax.transAxes,
                ha='right', va='top', fontsize=config.font_size - 1,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if threshold is not None:
        ax.legend(fontsize=config.font_size - 2)

    return (fig, ax)


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
    config = config or PlotConfig()
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize)
    else:
        fig = ax.figure

    sorted_eig = np.sort(eigenvalues)[::-1]
    indices = np.arange(len(sorted_eig))

    ax.plot(indices, sorted_eig, 'o-', color='#1f77b4',
            linewidth=config.line_width, markersize=config.marker_size, alpha=config.alpha)

    ax.set_xlabel('Index', fontsize=config.font_size)
    ax.set_ylabel('Eigenvalue', fontsize=config.font_size)
    ax.set_title(f'Eigenvalue Spectrum of {matrix_name}', fontsize=config.title_size)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')

    return (fig, ax)


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
    config = config or PlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize)

    cond_numbers = [np.linalg.cond(m) for m in matrices]

    # Color by severity
    colors = []
    for c in cond_numbers:
        if c < 1e3:
            colors.append('#2ca02c')  # green -- well-conditioned
        elif c < 1e6:
            colors.append('#ff7f0e')  # yellow/orange -- moderate
        else:
            colors.append('#d62728')  # red -- ill-conditioned

    x_pos = np.arange(len(labels))
    ax.bar(x_pos, cond_numbers, color=colors, alpha=config.alpha)
    ax.set_yscale('log')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=config.font_size)
    ax.set_ylabel('Condition Number (log scale)', fontsize=config.font_size)
    ax.set_title('Condition Number Comparison', fontsize=config.title_size)

    # Reference lines
    ax.axhline(1e3, linestyle='--', color='#ff7f0e', linewidth=1, alpha=0.6, label='Moderate (1e3)')
    ax.axhline(1e6, linestyle='--', color='#d62728', linewidth=1, alpha=0.6, label='Ill-conditioned (1e6)')
    ax.legend(fontsize=config.font_size - 2)
    ax.grid(True, alpha=0.3, axis='y')

    return (fig, ax)


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
    config = config or PlotConfig()

    # vectors: shape (n_vectors, dim) -- each row is a vector
    # We'll work with 2D for visualization (use first 2 components)
    vecs = np.array(vectors, dtype=float)
    n_vecs = vecs.shape[0]

    # Gram-Schmidt orthogonalization
    Q = np.zeros_like(vecs, dtype=float)
    for i in range(n_vecs):
        q = vecs[i].copy()
        for j in range(i):
            q = q - np.dot(vecs[i], Q[j]) * Q[j]
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            q = q / norm
        Q[i] = q

    fig, axes = plt.subplots(1, 2, figsize=(config.figsize[0] * 1.2, config.figsize[1]))

    # Panel 1: Original vectors
    ax1 = axes[0]
    for i in range(n_vecs):
        v = vecs[i]
        color = config.colors[i % len(config.colors)]
        ax1.annotate('', xy=(v[0], v[1]), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color=color, lw=config.line_width))
        ax1.text(v[0] * 1.05, v[1] * 1.05, f'v{i+1}', fontsize=config.font_size, color=color)
    ax1.set_aspect('equal')
    ax1.set_title('Original Vectors', fontsize=config.title_size)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)

    # Panel 2: Orthogonalized vectors
    ax2 = axes[1]
    for i in range(n_vecs):
        q = Q[i]
        color = config.colors[i % len(config.colors)]
        ax2.annotate('', xy=(q[0], q[1]), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color=color, lw=config.line_width))
        ax2.text(q[0] * 1.05, q[1] * 1.05, f'q{i+1}', fontsize=config.font_size, color=color)
    ax2.set_aspect('equal')
    ax2.set_title('Orthonormalized Vectors', fontsize=config.title_size)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)

    fig.suptitle('Gram-Schmidt Process', fontsize=config.title_size + 2)
    fig.tight_layout()

    return (fig, axes)


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
    config = config or PlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize)

    n_features = coefficients.shape[1]
    for i in range(n_features):
        label = feature_names[i] if feature_names is not None else f'Feature {i}'
        color = config.colors[i % len(config.colors)]
        ax.semilogx(lambdas, coefficients[:, i], color=color,
                     linewidth=config.line_width, alpha=config.alpha, label=label)

    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')

    if optimal_lambda is not None:
        ax.axvline(optimal_lambda, color='#d62728', linewidth=config.line_width,
                   linestyle='--', alpha=0.8, label=f'Optimal lambda = {optimal_lambda}')

    ax.set_xlabel('lambda (regularization strength)', fontsize=config.font_size)
    ax.set_ylabel('Coefficient value', fontsize=config.font_size)
    ax.set_title('Ridge Coefficient Path', fontsize=config.title_size)
    ax.legend(fontsize=config.font_size - 2, loc='best')
    ax.grid(True, alpha=0.3)

    return (fig, ax)


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
    config = config or PlotConfig()

    if beta_norms is not None:
        fig, axes = plt.subplots(1, 2, figsize=(config.figsize[0] * 1.2, config.figsize[1]))
        ax_loss = axes[0]
        ax_beta = axes[1]
    else:
        fig, ax_loss = plt.subplots(figsize=config.figsize)
        axes = ax_loss

    # Panel 1: Loss
    iterations = list(range(1, len(losses) + 1))
    ax_loss.plot(iterations, losses, 'o-', color='#1f77b4',
                 linewidth=config.line_width, markersize=config.marker_size, alpha=config.alpha)
    ax_loss.set_xlabel('Iteration', fontsize=config.font_size)
    ax_loss.set_ylabel('Loss', fontsize=config.font_size)
    ax_loss.set_title('IRLS Loss Convergence', fontsize=config.title_size)
    ax_loss.grid(True, alpha=0.3)

    if tolerance is not None:
        ax_loss.axhline(tolerance, linestyle='--', color='#d62728',
                        linewidth=config.line_width, alpha=0.7, label=f'Tolerance = {tolerance}')
        ax_loss.legend(fontsize=config.font_size - 2)

    # Panel 2: Coefficient norms
    if beta_norms is not None:
        iters_beta = list(range(1, len(beta_norms) + 1))
        ax_beta.plot(iters_beta, beta_norms, 's-', color='#ff7f0e',
                     linewidth=config.line_width, markersize=config.marker_size, alpha=config.alpha)
        ax_beta.set_xlabel('Iteration', fontsize=config.font_size)
        ax_beta.set_ylabel('||beta_new - beta_old||', fontsize=config.font_size)
        ax_beta.set_title('Coefficient Change', fontsize=config.title_size)
        ax_beta.grid(True, alpha=0.3)
        if tolerance is not None:
            ax_beta.axhline(tolerance, linestyle='--', color='#d62728',
                            linewidth=config.line_width, alpha=0.7, label=f'Tolerance = {tolerance}')
            ax_beta.legend(fontsize=config.font_size - 2)

    fig.tight_layout()
    return (fig, axes)


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
    config = config or PlotConfig()

    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(config.figsize[0] * 1.2, config.figsize[1] * 1.2))

    # Panel 1: Residuals vs Fitted
    ax1 = axes[0, 0]
    ax1.scatter(y_pred, residuals, alpha=config.alpha, s=config.marker_size ** 2, color='#1f77b4')
    ax1.axhline(0, color='#d62728', linewidth=config.line_width, linestyle='--')
    ax1.set_xlabel('Fitted values', fontsize=config.font_size)
    ax1.set_ylabel('Residuals', fontsize=config.font_size)
    ax1.set_title('Residuals vs Fitted', fontsize=config.title_size)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Q-Q plot (no scipy dependency)
    ax2 = axes[0, 1]
    std_res = residuals.copy()
    res_std = np.std(residuals)
    if res_std > 0:
        std_res = (residuals - np.mean(residuals)) / res_std
    sorted_residuals = np.sort(std_res)
    # Theoretical quantiles from a standard normal
    rng = np.random.default_rng(42)
    theoretical = np.sort(rng.standard_normal(len(residuals)))
    ax2.scatter(theoretical, sorted_residuals, alpha=config.alpha,
                s=config.marker_size ** 2, color='#1f77b4')
    # Reference diagonal line
    min_val = min(theoretical.min(), sorted_residuals.min())
    max_val = max(theoretical.max(), sorted_residuals.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=config.line_width)
    ax2.set_xlabel('Theoretical Quantiles', fontsize=config.font_size)
    ax2.set_ylabel('Sample Quantiles', fontsize=config.font_size)
    ax2.set_title('Normal Q-Q', fontsize=config.title_size)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Histogram of residuals
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins='auto', color='#1f77b4', alpha=config.alpha, edgecolor='black')
    ax3.set_xlabel('Residuals', fontsize=config.font_size)
    ax3.set_ylabel('Frequency', fontsize=config.font_size)
    ax3.set_title('Histogram of Residuals', fontsize=config.title_size)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Scale-Location
    ax4 = axes[1, 1]
    sqrt_abs_res = np.sqrt(np.abs(std_res))
    ax4.scatter(y_pred, sqrt_abs_res, alpha=config.alpha,
                s=config.marker_size ** 2, color='#1f77b4')
    ax4.set_xlabel('Fitted values', fontsize=config.font_size)
    ax4.set_ylabel('sqrt(|Standardized Residuals|)', fontsize=config.font_size)
    ax4.set_title('Scale-Location', fontsize=config.title_size)
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Residual Diagnostics', fontsize=config.title_size + 2)
    fig.tight_layout()

    return (fig, axes)


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
    config = config or PlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize)

    coefficients = np.asarray(coefficients)
    # Sort by absolute value descending
    sorted_idx = np.argsort(np.abs(coefficients))[::-1]

    if top_k is not None:
        sorted_idx = sorted_idx[:top_k]

    sorted_coefs = coefficients[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]

    # Color by sign
    colors = ['#1f77b4' if c >= 0 else '#d62728' for c in sorted_coefs]

    y_pos = np.arange(len(sorted_coefs))
    ax.barh(y_pos, sorted_coefs, color=colors, alpha=config.alpha)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=config.font_size)
    ax.invert_yaxis()  # largest at top
    ax.set_xlabel('Coefficient Value', fontsize=config.font_size)
    ax.set_title('Feature Importance (|coefficient|)', fontsize=config.title_size)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (val, name) in enumerate(zip(sorted_coefs, sorted_names)):
        ax.text(val + 0.01 * np.sign(val) * np.max(np.abs(sorted_coefs)),
                i, f'{val:+.2f}', va='center', fontsize=config.font_size - 2)

    fig.tight_layout()
    return (fig, ax)


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
    config = config or PlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize)

    # PCA via numpy SVD (no sklearn dependency)
    embeddings = np.asarray(embeddings, dtype=float)
    centered = embeddings - embeddings.mean(axis=0)
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    # First 2 principal components
    reduced = centered @ Vt[:2].T

    if labels is not None:
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        for i, lab in enumerate(unique_labels):
            mask = labels == lab
            color = config.colors[i % len(config.colors)]
            ax.scatter(reduced[mask, 0], reduced[mask, 1],
                       c=color, label=str(lab), alpha=config.alpha,
                       s=config.marker_size ** 2)
        ax.legend(fontsize=config.font_size - 2)
    else:
        ax.scatter(reduced[:, 0], reduced[:, 1], alpha=config.alpha,
                   s=config.marker_size ** 2, color='#1f77b4')

    # Optionally annotate points
    if names is not None:
        for i, name in enumerate(names):
            ax.annotate(name, (reduced[i, 0], reduced[i, 1]),
                        fontsize=config.font_size - 3, alpha=0.8)

    ax.set_xlabel('Component 1', fontsize=config.font_size)
    ax.set_ylabel('Component 2', fontsize=config.font_size)
    ax.set_title(f'Embeddings 2D (PCA)', fontsize=config.title_size)
    ax.grid(True, alpha=0.3)

    return (fig, ax)


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
    config = config or PlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize)

    embeddings = np.asarray(embeddings, dtype=float)

    # Compute cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    normed = embeddings / norms
    sim_matrix = normed @ normed.T

    n = len(labels)
    im = ax.imshow(sim_matrix, cmap='viridis', aspect='auto', vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label='Cosine Similarity')

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=config.font_size - 2)
    ax.set_yticklabels(labels, fontsize=config.font_size - 2)
    ax.set_title('Similarity Heatmap (Cosine)', fontsize=config.title_size)

    fig.tight_layout()
    return (fig, ax)


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
    config = config or PlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize)

    embeddings = np.asarray(embeddings, dtype=float)
    cluster_labels = np.asarray(cluster_labels)

    # PCA via numpy SVD
    centered = embeddings - embeddings.mean(axis=0)
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    reduced = centered @ Vt[:2].T

    unique_clusters = np.unique(cluster_labels)
    for i, cl in enumerate(unique_clusters):
        mask = cluster_labels == cl
        color = config.colors[i % len(config.colors)]
        label = cluster_names[i] if cluster_names is not None and i < len(cluster_names) else f'Cluster {cl}'
        ax.scatter(reduced[mask, 0], reduced[mask, 1],
                   c=color, label=label, alpha=config.alpha,
                   s=config.marker_size ** 2)

        # Compute and plot centroid
        centroid = reduced[mask].mean(axis=0)
        ax.scatter(centroid[0], centroid[1], c=color, marker='*',
                   s=config.marker_size ** 2 * 4, edgecolors='black', linewidths=1, zorder=5)

    ax.set_xlabel('Component 1', fontsize=config.font_size)
    ax.set_ylabel('Component 2', fontsize=config.font_size)
    ax.set_title('Embedding Clusters (PCA)', fontsize=config.title_size)
    ax.legend(fontsize=config.font_size - 2)
    ax.grid(True, alpha=0.3)

    return (fig, ax)


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
    config = config or PlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize)

    attention_matrix = np.asarray(attention_matrix, dtype=float)

    im = ax.imshow(attention_matrix, cmap='viridis', aspect='auto')
    fig.colorbar(im, ax=ax, label='Attention Weight')

    n_queries = attention_matrix.shape[0]
    n_keys = attention_matrix.shape[1]

    ax.set_xticks(np.arange(n_keys))
    ax.set_yticks(np.arange(n_queries))
    ax.set_xticklabels(key_labels, rotation=45, ha='right', fontsize=config.font_size - 2)
    ax.set_yticklabels(query_labels, fontsize=config.font_size - 2)

    # Annotate cells with values
    for i in range(n_queries):
        for j in range(n_keys):
            ax.text(j, i, f'{attention_matrix[i, j]:.2f}',
                    ha='center', va='center', fontsize=config.font_size - 2,
                    color='white' if attention_matrix[i, j] < 0.5 else 'black')

    ax.set_title('Attention Weights', fontsize=config.title_size)
    fig.tight_layout()

    return (fig, ax)


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
    config = config or PlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize)

    matrix = np.asarray(matrix, dtype=float)
    im = ax.imshow(matrix, cmap=config.colormap, aspect='auto')
    fig.colorbar(im, ax=ax)

    nrows, ncols = matrix.shape

    if row_labels is not None:
        ax.set_yticks(np.arange(nrows))
        ax.set_yticklabels(row_labels, fontsize=config.font_size - 2)
    if col_labels is not None:
        ax.set_xticks(np.arange(ncols))
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=config.font_size - 2)

    if annotate:
        for i in range(nrows):
            for j in range(ncols):
                ax.text(j, i, f'{matrix[i, j]:.2f}',
                        ha='center', va='center', fontsize=config.font_size - 3,
                        color='white' if abs(matrix[i, j]) > (matrix.max() + matrix.min()) / 2 else 'black')

    ax.set_title(title, fontsize=config.title_size)
    fig.tight_layout()

    return (fig, ax)


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
    config = config or PlotConfig()
    fig, ax = plt.subplots(figsize=config.figsize)

    X = np.asarray(X, dtype=float)
    corr = np.corrcoef(X.T)

    im = ax.imshow(corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label='Correlation')

    n = len(feature_names)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=config.font_size - 2)
    ax.set_yticklabels(feature_names, fontsize=config.font_size - 2)

    # Annotate cells, highlight those above threshold
    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            if i != j and abs(val) >= threshold:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=config.font_size - 3, fontweight='bold',
                        color='white' if abs(val) > 0.5 else 'black')
            else:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=config.font_size - 3,
                        color='white' if abs(val) > 0.5 else 'black')

    ax.set_title(f'Correlation Matrix (threshold={threshold})', fontsize=config.title_size)
    fig.tight_layout()

    return (fig, ax)


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
    config = config or PlotConfig()
    if fig is not None:
        fig.savefig(filename, dpi=config.dpi, bbox_inches='tight')
    return filename


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
    config = config or PlotConfig()

    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(config.figsize[0], config.figsize[1] * nrows / 2))

    # Ensure axes is always 2D
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # Hide unused axes
    for idx in range(n_plots, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    fig.tight_layout()
    return (fig, axes)


def set_style(config: Optional[PlotConfig] = None) -> None:
    """
    Set matplotlib style from config.

    IMPLEMENTATION:
    ────────────────
    import matplotlib.pyplot as plt
    plt.style.use(config.style)
    plt.rcParams.update({'font.size': config.font_size})
    """
    config = config or PlotConfig()
    try:
        plt.style.use(config.style)
    except (OSError, ValueError):
        pass  # style not available, use defaults
    plt.rcParams.update({'font.size': config.font_size})
    return config


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
