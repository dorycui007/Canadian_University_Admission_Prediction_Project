"""
Unit Tests for Math Visualization Module
==========================================

Tests for src/visualization/math_viz.py -- PlotConfig dataclass
and all math/linear algebra plotting functions.

All functions are implemented and return (fig, ax) tuples.
Tests verify return types, figure/axes structure, and basic properties.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.figure
from dataclasses import fields

from src.visualization.math_viz import (
    PlotConfig,
    plot_vector_projection,
    plot_projection_onto_subspace,
    plot_svd_decomposition,
    plot_singular_values,
    plot_eigenvalue_spectrum,
    plot_condition_number,
    plot_gram_schmidt_process,
    plot_ridge_path,
    plot_irls_convergence,
    plot_residuals,
    plot_coefficient_importance,
    plot_embeddings_2d,
    plot_similarity_heatmap,
    plot_embedding_clusters,
    plot_attention_weights,
    plot_matrix_heatmap,
    plot_correlation_matrix,
    save_figure,
    create_subplot_grid,
    set_style,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to avoid memory leaks."""
    yield
    plt.close('all')


# ==========================================================================
#  TestPlotConfig
# ==========================================================================
class TestPlotConfig:
    """Tests for the PlotConfig dataclass."""

    def test_default_values(self):
        """Default field values match the specification."""
        cfg = PlotConfig()
        assert cfg.figsize == (10, 6)
        assert cfg.dpi == 100
        assert cfg.style == "seaborn-v0_8-whitegrid"
        assert cfg.colormap == "viridis"
        assert cfg.font_size == 12
        assert cfg.line_width == pytest.approx(2.0)
        assert cfg.alpha == pytest.approx(0.7)

    def test_custom_values(self):
        """Custom values override defaults correctly."""
        cfg = PlotConfig(
            figsize=(16, 9),
            dpi=300,
            style="ggplot",
            colormap="plasma",
            font_size=14,
            line_width=3.0,
            alpha=0.5,
        )
        assert cfg.figsize == (16, 9)
        assert cfg.dpi == 300
        assert cfg.style == "ggplot"
        assert cfg.colormap == "plasma"
        assert cfg.font_size == 14
        assert cfg.line_width == pytest.approx(3.0)
        assert cfg.alpha == pytest.approx(0.5)

    def test_field_names(self):
        """All expected field names are present on the dataclass."""
        expected = {
            "figsize",
            "dpi",
            "style",
            "colormap",
            "save_format",
            "font_size",
            "title_size",
            "line_width",
            "marker_size",
            "alpha",
            "colors",
        }
        actual = {f.name for f in fields(PlotConfig)}
        assert expected == actual


# ==========================================================================
#  TestLinearAlgebraViz
# ==========================================================================
class TestLinearAlgebraViz:
    """Tests for linear algebra visualization functions."""

    def test_plot_vector_projection(self):
        v = np.array([3, 4])
        u = np.array([1, 0])
        fig, ax = plot_vector_projection(v, u)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_projection_onto_subspace(self):
        y = np.array([1.0, 2.0, 3.0])
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        fig, ax = plot_projection_onto_subspace(y, X)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_svd_decomposition(self):
        A = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [7.0, 8.0, 9.0]])
        fig, axes = plot_svd_decomposition(A)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(axes) == 5  # 4 heatmaps + 1 bar chart

    def test_plot_singular_values(self):
        singular_values = np.array([10.0, 5.0, 2.0, 0.5, 0.01])
        fig, ax = plot_singular_values(singular_values)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_eigenvalue_spectrum(self):
        eigenvalues = np.array([8.0, 4.0, 2.0, 0.5, 0.1])
        fig, ax = plot_eigenvalue_spectrum(eigenvalues)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_condition_number(self):
        m1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        m2 = np.array([[1.0, 0.99], [0.99, 1.0]])
        matrices = [m1, m2]
        labels = ["Identity", "Near-singular"]
        fig, ax = plot_condition_number(matrices, labels)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_gram_schmidt_process(self):
        vectors = np.array([[1.0, 0.5],
                             [0.0, 1.0],
                             [1.0, 1.0]])
        fig, axes = plot_gram_schmidt_process(vectors)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert axes is not None


# ==========================================================================
#  TestRegressionViz
# ==========================================================================
class TestRegressionViz:
    """Tests for regression visualization functions."""

    def test_plot_ridge_path(self):
        lambdas = np.logspace(-3, 3, 10)
        coefficients = np.random.randn(10, 4)
        fig, ax = plot_ridge_path(lambdas, coefficients)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_ridge_path_with_feature_names(self):
        lambdas = np.logspace(-3, 3, 10)
        coefficients = np.random.randn(10, 3)
        feature_names = ["GPA", "SAT", "Extracurricular"]
        fig, ax = plot_ridge_path(lambdas, coefficients, feature_names=feature_names)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_irls_convergence(self):
        losses = [1.0, 0.5, 0.3, 0.25, 0.24, 0.239]
        fig, ax = plot_irls_convergence(losses)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_residuals(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.3, 2.8, 4.2, 4.9])
        fig, axes = plot_residuals(y_true, y_pred)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert axes is not None

    def test_plot_coefficient_importance(self):
        coefficients = np.array([2.3, -1.1, 0.8, -0.3])
        feature_names = ["GPA", "Province_ON", "is_CS", "Year"]
        fig, ax = plot_coefficient_importance(coefficients, feature_names)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None


# ==========================================================================
#  TestEmbeddingViz
# ==========================================================================
class TestEmbeddingViz:
    """Tests for embedding visualization functions."""

    def test_plot_embeddings_2d(self):
        embeddings = np.random.randn(20, 8)
        fig, ax = plot_embeddings_2d(embeddings)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_similarity_heatmap(self):
        embeddings = np.random.randn(5, 4)
        labels = ["CS", "EE", "Math", "Bio", "Comm"]
        fig, ax = plot_similarity_heatmap(embeddings, labels)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_embedding_clusters(self):
        embeddings = np.random.randn(15, 6)
        cluster_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 0, 1, 2])
        fig, ax = plot_embedding_clusters(embeddings, cluster_labels)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_attention_weights(self):
        attention_matrix = np.array([
            [0.5, 0.3, 0.2],
            [0.1, 0.6, 0.3],
            [0.2, 0.2, 0.6],
        ])
        query_labels = ["UofT CS", "McGill EE", "UW SE"]
        key_labels = ["UofT CS", "McGill EE", "UW SE"]
        fig, ax = plot_attention_weights(attention_matrix, query_labels, key_labels)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None


# ==========================================================================
#  TestMatrixViz
# ==========================================================================
class TestMatrixViz:
    """Tests for matrix visualization functions."""

    def test_plot_matrix_heatmap(self):
        matrix = np.array([[1.0, 0.5, 0.2],
                            [0.5, 1.0, 0.3],
                            [0.2, 0.3, 1.0]])
        fig, ax = plot_matrix_heatmap(matrix)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_plot_correlation_matrix(self):
        X = np.random.randn(50, 4)
        feature_names = ["GPA", "SAT", "Extra", "Year"]
        fig, ax = plot_correlation_matrix(X, feature_names)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None


# ==========================================================================
#  TestUtilityFunctions
# ==========================================================================
class TestUtilityFunctions:
    """Tests for utility helper functions."""

    def test_save_figure(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        output_path = str(tmp_path / "test_output.png")
        result = save_figure(fig, output_path)
        assert result == output_path

    def test_save_figure_none_fig(self, tmp_path):
        output_path = str(tmp_path / "test_output.png")
        result = save_figure(None, output_path)
        assert result == output_path

    def test_create_subplot_grid(self):
        fig, axes = create_subplot_grid(n_plots=4)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert axes is not None
        # Should be a 2D array with shape (2, 2) for 4 plots
        assert axes.shape == (2, 2)

    def test_create_subplot_grid_odd(self):
        fig, axes = create_subplot_grid(n_plots=3, ncols=2)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert axes.shape == (2, 2)
        # Last cell should be hidden
        assert not axes[1, 1].get_visible()

    def test_set_style(self):
        result = set_style()
        assert isinstance(result, PlotConfig)

    def test_set_style_custom(self):
        cfg = PlotConfig(font_size=16)
        result = set_style(cfg)
        assert result is cfg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
