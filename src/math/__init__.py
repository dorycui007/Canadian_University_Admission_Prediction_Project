"""
Math Package — Linear Algebra from Scratch
============================================

Pure-Python/NumPy implementations of linear algebra operations
for the admission prediction pipeline.  No sklearn, no scipy
for the core computations.

Modules:
    vectors     - Vector operations (dot, norm, projection basics)
    matrices    - Matrix operations (multiply, transpose, Gram, rank)
    projections - Projection and least-squares solvers
    qr          - Householder QR factorization
    svd         - Singular value decomposition utilities
    ridge       - Ridge regression solvers and cross-validation
"""

from .vectors import (
    Vector,
    add,
    scale,
    linear_combination,
    dot,
    norm,
    distance,
    normalize,
    angle,
    cosine_similarity,
    is_orthogonal,
    is_unit,
    verify_cauchy_schwarz,
)

from .matrices import (
    Matrix,
    transpose,
    matrix_multiply,
    matrix_vector_multiply,
    gram_matrix,
    compute_rank,
    check_full_column_rank,
    condition_number,
    is_symmetric,
    is_positive_definite,
    diagonal_matrix,
    identity,
    add_ridge,
    outer_product,
    trace,
    frobenius_norm,
)

from .projections import (
    project_onto_vector,
    project_onto_subspace,
    compute_residual,
    verify_orthogonality,
    solve_normal_equations,
    solve_weighted_normal_equations,
    compute_hat_matrix,
    compute_leverage,
    projection_matrix_onto_complement,
    sum_of_squared_residuals,
    r_squared,
)

from .qr import (
    householder_vector,
    apply_householder,
    qr_householder,
    qr_reduced,
    back_substitution,
    solve_via_qr,
    solve_weighted_via_qr,
    check_qr_factorization,
)

from .svd import (
    compute_svd,
    singular_values,
    condition_number as svd_condition_number,
    matrix_rank,
    low_rank_approximation,
    reconstruct_from_svd,
    approximation_error,
    explained_variance_ratio,
    choose_rank,
    svd_solve,
    ridge_via_svd,
    effective_degrees_of_freedom,
    truncated_svd,
)

from .ridge import (
    RidgeConvergenceError,
    IllConditionedError,
    ridge_solve,
    ridge_solve_qr,
    weighted_ridge_solve,
    ridge_loocv,
    ridge_cv,
    ridge_path,
    ridge_gcv,
    standardize_features,
    ridge_effective_df,
)

__all__ = [
    # vectors
    "Vector",
    "add",
    "scale",
    "linear_combination",
    "dot",
    "norm",
    "distance",
    "normalize",
    "angle",
    "cosine_similarity",
    "is_orthogonal",
    "is_unit",
    "verify_cauchy_schwarz",
    # matrices
    "Matrix",
    "transpose",
    "matrix_multiply",
    "matrix_vector_multiply",
    "gram_matrix",
    "compute_rank",
    "check_full_column_rank",
    "condition_number",
    "is_symmetric",
    "is_positive_definite",
    "diagonal_matrix",
    "identity",
    "add_ridge",
    "outer_product",
    "trace",
    "frobenius_norm",
    # projections
    "project_onto_vector",
    "project_onto_subspace",
    "compute_residual",
    "verify_orthogonality",
    "solve_normal_equations",
    "solve_weighted_normal_equations",
    "compute_hat_matrix",
    "compute_leverage",
    "projection_matrix_onto_complement",
    "sum_of_squared_residuals",
    "r_squared",
    # qr
    "householder_vector",
    "apply_householder",
    "qr_householder",
    "qr_reduced",
    "back_substitution",
    "solve_via_qr",
    "solve_weighted_via_qr",
    "check_qr_factorization",
    # svd
    "compute_svd",
    "singular_values",
    "svd_condition_number",
    "matrix_rank",
    "low_rank_approximation",
    "reconstruct_from_svd",
    "approximation_error",
    "explained_variance_ratio",
    "choose_rank",
    "svd_solve",
    "ridge_via_svd",
    "effective_degrees_of_freedom",
    "truncated_svd",
    # ridge
    "RidgeConvergenceError",
    "IllConditionedError",
    "ridge_solve",
    "ridge_solve_qr",
    "weighted_ridge_solve",
    "ridge_loocv",
    "ridge_cv",
    "ridge_path",
    "ridge_gcv",
    "standardize_features",
    "ridge_effective_df",
]
