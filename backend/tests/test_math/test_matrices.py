"""
Unit Tests for Matrix Operations Module
========================================

Comprehensive tests for src/math/matrices.py validating all matrix
operations used in the grade prediction pipeline.

MAT223 References: Sections 1.3-1.4 (Matrix Operations), 4.6 (Rank)
"""

import pytest
import numpy as np

from src.math.matrices import (
    _ensure_numpy_matrix,
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


# =============================================================================
#                     _ensure_numpy_matrix TESTS
# =============================================================================

class TestEnsureNumpyMatrix:
    """Tests for the _ensure_numpy_matrix helper function."""

    def test_nested_list_to_ndarray(self):
        """Convert a nested Python list to a 2D numpy array."""
        result = _ensure_numpy_matrix([[1, 2], [3, 4]])
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        np.testing.assert_array_almost_equal(result, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_ndarray_passthrough(self, identity_3x3):
        """Pass through an existing 2D numpy array."""
        result = _ensure_numpy_matrix(identity_3x3)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, identity_3x3)

    def test_float_dtype(self):
        """Ensure integer lists are converted to float arrays."""
        result = _ensure_numpy_matrix([[1, 2], [3, 4]])
        assert result.dtype in (np.float64, np.float32)

    def test_shape_preserved(self):
        """Input shape is preserved after conversion."""
        result = _ensure_numpy_matrix([[1, 2, 3], [4, 5, 6]])
        assert result.shape == (2, 3)


# =============================================================================
#                         TRANSPOSE TESTS
# =============================================================================

class TestTranspose:
    """Tests for matrix transpose: transpose(A)."""

    def test_basic_transpose(self):
        """Transpose swaps rows and columns."""
        A = [[1, 2, 3], [4, 5, 6]]
        result = transpose(A)
        expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_shape_change(self, tall_matrix_5x3):
        """Transpose of (5,3) has shape (3,5)."""
        result = transpose(tall_matrix_5x3)
        assert result.shape == (3, 5)

    def test_double_transpose_identity(self, square_matrix_3x3):
        """(A^T)^T = A."""
        np.testing.assert_array_almost_equal(
            transpose(transpose(square_matrix_3x3)), square_matrix_3x3
        )

    def test_symmetric_matrix_transpose(self, symmetric_pd_matrix):
        """Symmetric matrix equals its own transpose."""
        np.testing.assert_array_almost_equal(
            transpose(symmetric_pd_matrix), symmetric_pd_matrix
        )


# =============================================================================
#                      MATRIX MULTIPLY TESTS
# =============================================================================

class TestMatrixMultiply:
    """Tests for matrix multiplication: matrix_multiply(A, B)."""

    def test_basic_multiplication(self):
        """[[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]."""
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        result = matrix_multiply(A, B)
        np.testing.assert_array_almost_equal(result, [[19, 22], [43, 50]])

    def test_identity_left(self, square_matrix_3x3, identity_3x3):
        """I * A = A."""
        result = matrix_multiply(identity_3x3, square_matrix_3x3)
        np.testing.assert_array_almost_equal(result, square_matrix_3x3)

    def test_associativity(self):
        """(AB)C = A(BC)."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        C = np.array([[9.0, 10.0], [11.0, 12.0]])
        lhs = matrix_multiply(matrix_multiply(A, B), C)
        rhs = matrix_multiply(A, matrix_multiply(B, C))
        np.testing.assert_array_almost_equal(lhs, rhs)

    def test_dimension_mismatch_raises(self):
        """Incompatible inner dimensions should raise ValueError."""
        A = [[1, 2, 3]]  # 1x3
        B = [[1, 2], [3, 4]]  # 2x2
        with pytest.raises((ValueError, Exception)):
            matrix_multiply(A, B)

    def test_transpose_of_product(self):
        """(AB)^T = B^T A^T."""
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        B = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        lhs = transpose(matrix_multiply(A, B))
        rhs = matrix_multiply(transpose(B), transpose(A))
        np.testing.assert_array_almost_equal(lhs, rhs)


# =============================================================================
#                   MATRIX VECTOR MULTIPLY TESTS
# =============================================================================

class TestMatrixVectorMultiply:
    """Tests for matrix-vector multiplication: matrix_vector_multiply(A, x)."""

    def test_basic_mat_vec(self):
        """[[1,2],[3,4],[5,6]] @ [1,2] = [5, 11, 17]."""
        A = [[1, 2], [3, 4], [5, 6]]
        x = np.array([1.0, 2.0])
        result = matrix_vector_multiply(A, x)
        np.testing.assert_array_almost_equal(result, [5, 11, 17])

    def test_identity_times_vector(self, identity_3x3):
        """Ix = x."""
        x = np.array([7.0, 8.0, 9.0])
        result = matrix_vector_multiply(identity_3x3, x)
        np.testing.assert_array_almost_equal(result, x)

    def test_output_shape(self, tall_matrix_5x3):
        """(5x3) @ (3,) yields shape (5,)."""
        x = np.array([1.0, 2.0, 3.0])
        result = matrix_vector_multiply(tall_matrix_5x3, x)
        assert result.shape == (5,)

    def test_dimension_mismatch_raises(self):
        """Incompatible dimensions should raise ValueError."""
        A = [[1, 2, 3], [4, 5, 6]]
        x = np.array([1.0, 2.0])
        with pytest.raises((ValueError, Exception)):
            matrix_vector_multiply(A, x)

    def test_zero_vector_gives_zero(self, square_matrix_3x3):
        """A * 0 = 0."""
        zero = np.zeros(3)
        result = matrix_vector_multiply(square_matrix_3x3, zero)
        np.testing.assert_array_almost_equal(result, np.zeros(3))


# =============================================================================
#                       GRAM MATRIX TESTS
# =============================================================================

class TestGramMatrix:
    """Tests for Gram matrix computation: gram_matrix(X) = X^T X."""

    def test_basic_gram(self):
        """X^T X for known X."""
        X = [[1, 2], [3, 4], [5, 6]]
        result = gram_matrix(X)
        np.testing.assert_array_almost_equal(result, [[35, 44], [44, 56]])

    def test_gram_is_symmetric(self, tall_matrix_5x3):
        """X^T X is always symmetric."""
        G = gram_matrix(tall_matrix_5x3)
        np.testing.assert_array_almost_equal(G, G.T)

    def test_identity_gram(self, identity_3x3):
        """I^T I = I."""
        G = gram_matrix(identity_3x3)
        np.testing.assert_array_almost_equal(G, identity_3x3)

    def test_gram_positive_semidefinite(self, tall_matrix_5x3):
        """X^T X is positive semi-definite: eigenvalues >= 0."""
        G = gram_matrix(tall_matrix_5x3)
        eigenvalues = np.linalg.eigvalsh(G)
        assert np.all(eigenvalues >= -1e-10)


# =============================================================================
#                       COMPUTE RANK TESTS
# =============================================================================

class TestComputeRank:
    """Tests for rank computation: compute_rank(A)."""

    def test_identity_rank(self, identity_3x3):
        """rank(I_3) = 3."""
        assert compute_rank(identity_3x3) == 3

    def test_rank_deficient(self, rank_deficient_matrix):
        """Rank of [[1,2],[2,4],[3,6]] is 1 (col2 = 2*col1)."""
        assert compute_rank(rank_deficient_matrix) == 1

    def test_full_column_rank_tall(self, plane_basis):
        """3x2 matrix with independent columns has rank 2."""
        assert compute_rank(plane_basis) == 2

    def test_zero_matrix_rank(self):
        """Zero matrix has rank 0."""
        Z = np.zeros((3, 3))
        assert compute_rank(Z) == 0

    def test_rank_one_outer_product(self):
        """Rank of outer product xy^T is 1."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0])
        M = np.outer(x, y)
        assert compute_rank(M) == 1


# =============================================================================
#                    CHECK FULL COLUMN RANK TESTS
# =============================================================================

class TestCheckFullColumnRank:
    """Tests for check_full_column_rank(X)."""

    def test_full_rank_identity(self, identity_3x3):
        """Identity matrix has full column rank."""
        assert check_full_column_rank(identity_3x3) is True

    def test_rank_deficient(self, rank_deficient_matrix):
        """Rank-deficient matrix does NOT have full column rank."""
        assert check_full_column_rank(rank_deficient_matrix) is False

    def test_tall_full_rank(self, plane_basis):
        """3x2 matrix with independent columns has full column rank."""
        assert check_full_column_rank(plane_basis) is True

    def test_one_hot_with_intercept_deficient(self):
        """One-hot columns plus intercept are rank-deficient."""
        X = np.array([
            [1, 1, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
        ], dtype=float)
        assert check_full_column_rank(X) is False


# =============================================================================
#                     CONDITION NUMBER TESTS
# =============================================================================

class TestConditionNumber:
    """Tests for condition_number(A)."""

    def test_identity_condition(self, identity_3x3):
        """cond(I) = 1."""
        assert np.isclose(condition_number(identity_3x3), 1.0)

    def test_ill_conditioned(self, ill_conditioned_matrix):
        """Ill-conditioned matrix has very large condition number."""
        cond = condition_number(ill_conditioned_matrix)
        assert cond > 1e6

    def test_orthogonal_condition(self, orthonormal_basis_3d):
        """Orthogonal matrix has condition number 1."""
        assert np.isclose(condition_number(orthonormal_basis_3d), 1.0)

    def test_diagonal_condition(self):
        """cond(diag(a, b)) = max(|a|,|b|) / min(|a|,|b|)."""
        D = np.array([[10.0, 0.0], [0.0, 2.0]])
        assert np.isclose(condition_number(D), 5.0)

    def test_singular_matrix_inf(self):
        """Singular matrix has infinite condition number."""
        S = np.array([[1.0, 2.0], [2.0, 4.0]])
        assert condition_number(S) == np.inf


# =============================================================================
#                       IS SYMMETRIC TESTS
# =============================================================================

class TestIsSymmetric:
    """Tests for is_symmetric(A)."""

    def test_symmetric_pd(self, symmetric_pd_matrix):
        """Known symmetric matrix is detected as symmetric."""
        assert is_symmetric(symmetric_pd_matrix) is True

    def test_identity_symmetric(self, identity_3x3):
        """Identity is symmetric."""
        assert is_symmetric(identity_3x3) is True

    def test_non_symmetric(self, upper_triangular_3x3):
        """Upper triangular (non-diagonal) is not symmetric."""
        assert is_symmetric(upper_triangular_3x3) is False

    def test_near_symmetric_within_tolerance(self):
        """Matrix that is symmetric within tolerance."""
        A = np.array([[1.0, 2.0], [2.0 + 1e-12, 1.0]])
        assert is_symmetric(A, tol=1e-10) is True

    def test_not_symmetric_outside_tolerance(self):
        """Matrix that exceeds tolerance is not symmetric."""
        A = np.array([[1.0, 2.0], [2.1, 1.0]])
        assert is_symmetric(A) is False


# =============================================================================
#                    IS POSITIVE DEFINITE TESTS
# =============================================================================

class TestIsPositiveDefinite:
    """Tests for is_positive_definite(A)."""

    def test_pd_matrix(self, symmetric_pd_matrix):
        """Known SPD matrix is detected as positive definite."""
        assert is_positive_definite(symmetric_pd_matrix) is True

    def test_identity_pd(self, identity_3x3):
        """Identity matrix is positive definite."""
        assert is_positive_definite(identity_3x3) is True

    def test_negative_definite_not_pd(self):
        """-I is not positive definite."""
        A = -np.eye(2)
        assert is_positive_definite(A) is False

    def test_singular_not_pd(self):
        """Singular matrix is not positive definite."""
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        assert is_positive_definite(A) is False

    def test_gram_full_rank_pd(self, tall_matrix_5x3):
        """X^T X is PD when X has full column rank."""
        G = gram_matrix(tall_matrix_5x3)
        assert is_positive_definite(G) is True


# =============================================================================
#                   DIAGONAL AND IDENTITY TESTS
# =============================================================================

class TestDiagonalAndIdentity:
    """Tests for diagonal_matrix and identity construction."""

    def test_basic_diagonal(self):
        """diag([1,2,3]) creates correct diagonal matrix."""
        result = diagonal_matrix([1, 2, 3])
        expected = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
        np.testing.assert_array_almost_equal(result, expected)

    def test_off_diagonals_zero(self):
        """All off-diagonal elements are zero."""
        D = diagonal_matrix([10.0, 20.0, 30.0])
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert np.isclose(D[i, j], 0.0)

    def test_identity_values(self):
        """3x3 identity has ones on diagonal, zeros elsewhere."""
        I = identity(3)
        np.testing.assert_array_almost_equal(I, np.eye(3))

    def test_identity_shape(self):
        """identity(n) has shape (n, n)."""
        assert identity(5).shape == (5, 5)


# =============================================================================
#                        ADD RIDGE TESTS
# =============================================================================

class TestAddRidge:
    """Tests for add_ridge(XtX, lambda_)."""

    def test_basic_ridge(self):
        """XtX + 0.1*I shifts diagonal by 0.1."""
        XtX = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = add_ridge(XtX, 0.1)
        expected = np.array([[1.1, 0.5], [0.5, 1.1]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_lambda(self, symmetric_pd_matrix):
        """lambda=0 returns original matrix."""
        result = add_ridge(symmetric_pd_matrix, 0.0)
        np.testing.assert_array_almost_equal(result, symmetric_pd_matrix)

    def test_negative_lambda_raises(self, symmetric_pd_matrix):
        """Negative lambda should raise ValueError."""
        with pytest.raises((ValueError, Exception)):
            add_ridge(symmetric_pd_matrix, -0.1)

    def test_ridge_improves_condition(self, ill_conditioned_matrix):
        """Ridge regularization reduces condition number."""
        cond_before = condition_number(ill_conditioned_matrix)
        ridged = add_ridge(ill_conditioned_matrix, 1.0)
        cond_after = condition_number(ridged)
        assert cond_after < cond_before

    def test_ridge_makes_invertible(self):
        """Singular XtX + lambda*I becomes invertible."""
        singular = np.array([[1.0, 2.0], [2.0, 4.0]])
        ridged = add_ridge(singular, 0.01)
        det = np.linalg.det(ridged)
        assert abs(det) > 1e-10

    def test_ridge_shifts_eigenvalues(self):
        """Adding lambda shifts all eigenvalues by lambda."""
        A = np.array([[3.0, 1.0], [1.0, 2.0]])
        lam = 0.5
        eig_orig = np.sort(np.linalg.eigvalsh(A))
        eig_ridge = np.sort(np.linalg.eigvalsh(add_ridge(A, lam)))
        np.testing.assert_array_almost_equal(eig_ridge, eig_orig + lam)


# =============================================================================
#                      OUTER PRODUCT TESTS
# =============================================================================

class TestOuterProduct:
    """Tests for outer_product(x, y)."""

    def test_basic_outer(self):
        """[1,2] outer [3,4,5] = [[3,4,5],[6,8,10]]."""
        result = outer_product([1, 2], [3, 4, 5])
        expected = np.array([[3, 4, 5], [6, 8, 10]], dtype=float)
        np.testing.assert_array_almost_equal(result, expected)

    def test_outer_rank_one(self):
        """Outer product always has rank 1 (for nonzero vectors)."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0])
        M = outer_product(x, y)
        assert compute_rank(M) == 1

    def test_outer_symmetric_when_same_vector(self):
        """xx^T is symmetric."""
        x = np.array([1.0, 2.0, 3.0])
        M = outer_product(x, x)
        np.testing.assert_array_almost_equal(M, M.T)

    def test_outer_element_formula(self):
        """(xy^T)_{ij} = x_i * y_j."""
        x = np.array([2.0, 3.0])
        y = np.array([5.0, 7.0, 11.0])
        M = outer_product(x, y)
        for i in range(len(x)):
            for j in range(len(y)):
                assert np.isclose(M[i, j], x[i] * y[j])


# =============================================================================
#                     TRACE AND FROBENIUS NORM TESTS
# =============================================================================

class TestTraceAndFrobeniusNorm:
    """Tests for trace and frobenius_norm scalar diagnostics."""

    def test_basic_trace(self):
        """tr([[1,2],[3,4]]) = 1 + 4 = 5."""
        assert np.isclose(trace([[1, 2], [3, 4]]), 5.0)

    def test_identity_trace(self, identity_3x3):
        """tr(I_n) = n."""
        assert np.isclose(trace(identity_3x3), 3.0)

    def test_trace_equals_sum_of_eigenvalues(self, symmetric_pd_matrix):
        """tr(A) = sum of eigenvalues for symmetric A."""
        eigs = np.linalg.eigvalsh(symmetric_pd_matrix)
        assert np.isclose(trace(symmetric_pd_matrix), np.sum(eigs))

    def test_trace_cyclic_property(self):
        """tr(AB) = tr(BA)."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        AB = matrix_multiply(A, B)
        BA = matrix_multiply(B, A)
        assert np.isclose(trace(AB), trace(BA))

    def test_basic_frobenius(self):
        """||[[1,2],[3,4]]||_F = sqrt(1+4+9+16) = sqrt(30)."""
        result = frobenius_norm([[1, 2], [3, 4]])
        assert np.isclose(result, np.sqrt(30.0))

    def test_identity_frobenius(self, identity_3x3):
        """||I_n||_F = sqrt(n)."""
        assert np.isclose(frobenius_norm(identity_3x3), np.sqrt(3.0))

    def test_frobenius_equals_sqrt_trace_AtA(self, square_matrix_3x3):
        """||A||_F = sqrt(tr(A^T A))."""
        A = square_matrix_3x3
        expected = np.sqrt(trace(matrix_multiply(transpose(A), A)))
        assert np.isclose(frobenius_norm(A), expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
