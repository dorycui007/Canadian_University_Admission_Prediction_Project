"""
Visualization Package — Plots and Figures
===========================================

Matplotlib-based visualizations for linear algebra concepts
and model evaluation metrics.

Modules:
    math_viz - Vector projections, SVD, ridge paths, embeddings, attention maps
    eval_viz - ROC curves, calibration diagrams, lift charts, confusion matrices
"""

from .math_viz import (
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

from .eval_viz import (
    EvalPlotConfig,
    plot_reliability_diagram,
    plot_calibration_comparison,
    plot_brier_decomposition,
    plot_expected_calibration_error,
    plot_roc_curve,
    plot_roc_comparison,
    plot_pr_curve,
    plot_lift_chart,
    plot_cumulative_gains,
    plot_confusion_matrix,
    plot_threshold_metrics,
    plot_precision_recall_tradeoff,
    plot_subgroup_performance,
    plot_fairness_comparison,
    plot_performance_over_time,
    plot_prediction_distribution,
    compute_calibration_curve,
    compute_brier_decomposition,
)

__all__ = [
    # math_viz
    "PlotConfig",
    "plot_vector_projection",
    "plot_projection_onto_subspace",
    "plot_svd_decomposition",
    "plot_singular_values",
    "plot_eigenvalue_spectrum",
    "plot_condition_number",
    "plot_gram_schmidt_process",
    "plot_ridge_path",
    "plot_irls_convergence",
    "plot_residuals",
    "plot_coefficient_importance",
    "plot_embeddings_2d",
    "plot_similarity_heatmap",
    "plot_embedding_clusters",
    "plot_attention_weights",
    "plot_matrix_heatmap",
    "plot_correlation_matrix",
    "save_figure",
    "create_subplot_grid",
    "set_style",
    # eval_viz
    "EvalPlotConfig",
    "plot_reliability_diagram",
    "plot_calibration_comparison",
    "plot_brier_decomposition",
    "plot_expected_calibration_error",
    "plot_roc_curve",
    "plot_roc_comparison",
    "plot_pr_curve",
    "plot_lift_chart",
    "plot_cumulative_gains",
    "plot_confusion_matrix",
    "plot_threshold_metrics",
    "plot_precision_recall_tradeoff",
    "plot_subgroup_performance",
    "plot_fairness_comparison",
    "plot_performance_over_time",
    "plot_prediction_distribution",
    "compute_calibration_curve",
    "compute_brier_decomposition",
]
