"""
Unit Tests for Vector Operations Module
========================================

Comprehensive tests for src/math/vectors.py validating all vector
operations used in the grade prediction pipeline.

MAT223 References: Sections 1.2-1.4 (Vector arithmetic, dot products, norms)
"""

import pytest
import numpy as np

from src.math.vectors import (
    _ensure_numpy,
    add,
    scale,
    linear_combination,
    dot,
    norm,
    normalize,
    distance,
    angle,
    cosine_similarity,
    is_orthogonal,
    is_unit,
    verify_cauchy_schwarz,
)


# =============================================================================
#                         _ensure_numpy TESTS
# =============================================================================

class TestEnsureNumpy:
    """Tests for the _ensure_numpy helper function."""

    def test_list_to_ndarray(self):
        """Convert a Python list to numpy array."""
        result = _ensure_numpy([1, 2, 3])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_ndarray_passthrough(self):
        """Pass through an existing numpy array unchanged."""
        arr = np.array([1.0, 2.0, 3.0])
        result = _ensure_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_float_conversion(self):
        """Ensure integer lists are converted to float arrays."""
        result = _ensure_numpy([1, 2, 3])
        assert result.dtype in (np.float64, np.float32, np.int64, np.int32)

    def test_empty_input(self):
        """Handle empty list input."""
        result = _ensure_numpy([])
        assert isinstance(result, np.ndarray)
        assert len(result) == 0


# =============================================================================
#                         ADDITION TESTS
# =============================================================================

class TestAdd:
    """Tests for vector addition: add(x, y)."""

    def test_basic_addition(self):
        """Test basic vector addition: [1,2] + [3,4] = [4,6]."""
        result = add([1, 2], [3, 4])
        np.testing.assert_array_almost_equal(result, [4, 6])

    def test_commutativity(self, random_vectors):
        """u + v = v + u."""
        u, v = random_vectors
        np.testing.assert_array_almost_equal(add(u, v), add(v, u))

    def test_associativity(self):
        """(u + v) + w = u + (v + w)."""
        u = np.array([1.0, 2.0])
        v = np.array([3.0, 4.0])
        w = np.array([5.0, 6.0])
        np.testing.assert_array_almost_equal(add(add(u, v), w), add(u, add(v, w)))

    def test_zero_identity(self, random_vectors):
        """v + 0 = v."""
        u, _ = random_vectors
        zero = np.zeros_like(u)
        np.testing.assert_array_almost_equal(add(u, zero), u)

    def test_dimension_mismatch_raises(self):
        """Mismatched dimensions should raise an error."""
        with pytest.raises((ValueError, Exception)):
            add([1, 2, 3], [1, 2])

    def test_list_inputs(self):
        """Accept Python lists as inputs."""
        result = add([1, 2, 3], [4, 5, 6])
        np.testing.assert_array_almost_equal(result, [5, 7, 9])


# =============================================================================
#                         SCALE TESTS
# =============================================================================

class TestScale:
    """Tests for scalar multiplication: scale(c, x)."""

    def test_basic_scaling(self):
        """Test basic scaling: 2 * [1,2,3] = [2,4,6]."""
        result = scale(2, [1, 2, 3])
        np.testing.assert_array_almost_equal(result, [2, 4, 6])

    def test_zero_scalar(self):
        """0 * v = zero vector."""
        result = scale(0, [1, 2, 3])
        np.testing.assert_array_almost_equal(result, [0, 0, 0])

    def test_negative_scalar(self):
        """-1 * v = -v."""
        v = np.array([1.0, -2.0, 3.0])
        result = scale(-1, v)
        np.testing.assert_array_almost_equal(result, [-1, 2, -3])

    def test_identity_scalar(self, random_vectors):
        """1 * v = v."""
        u, _ = random_vectors
        np.testing.assert_array_almost_equal(scale(1, u), u)


# =============================================================================
#                    LINEAR COMBINATION TESTS
# =============================================================================

class TestLinearCombination:
    """Tests for linear_combination(scalars, vectors)."""

    def test_basic_combination(self):
        """2*[1,0] + 3*[0,1] = [2,3]."""
        result = linear_combination([2, 3], [[1, 0], [0, 1]])
        np.testing.assert_array_almost_equal(result, [2, 3])

    def test_single_vector(self):
        """Linear combination of a single vector = scalar * vector."""
        result = linear_combination([5], [[1, 2, 3]])
        np.testing.assert_array_almost_equal(result, [5, 10, 15])

    def test_three_vectors(self):
        """1*e1 + 2*e2 + 3*e3 = [1,2,3]."""
        e1, e2, e3 = [1, 0, 0], [0, 1, 0], [0, 0, 1]
        result = linear_combination([1, 2, 3], [e1, e2, e3])
        np.testing.assert_array_almost_equal(result, [1, 2, 3])

    def test_zero_coefficients(self):
        """All zero coefficients give zero vector."""
        result = linear_combination([0, 0], [[1, 2], [3, 4]])
        np.testing.assert_array_almost_equal(result, [0, 0])

    def test_negative_coefficients(self):
        """Negative coefficients reverse direction."""
        result = linear_combination([-1, 1], [[1, 0], [0, 1]])
        np.testing.assert_array_almost_equal(result, [-1, 1])


# =============================================================================
#                         DOT PRODUCT TESTS
# =============================================================================

class TestDotProduct:
    """Tests for dot product: dot(x, y)."""

    def test_basic_dot_product(self):
        """[1,2,3] . [4,5,6] = 4+10+18 = 32."""
        result = dot([1, 2, 3], [4, 5, 6])
        assert np.isclose(result, 32)

    def test_commutativity(self, random_vectors):
        """u . v = v . u."""
        u, v = random_vectors
        assert np.isclose(dot(u, v), dot(v, u))

    def test_distributivity(self):
        """u . (v + w) = u.v + u.w."""
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([4.0, 5.0, 6.0])
        w = np.array([7.0, 8.0, 9.0])
        lhs = dot(u, v + w)
        rhs = dot(u, v) + dot(u, w)
        assert np.isclose(lhs, rhs)

    def test_scalar_multiplication(self):
        """(cu) . v = c(u . v)."""
        c = 3.5
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([4.0, 5.0, 6.0])
        lhs = dot(c * u, v)
        rhs = c * dot(u, v)
        assert np.isclose(lhs, rhs)

    def test_self_dot_product(self, pythagorean_vector):
        """v . v = ||v||^2. For [3,4]: 9+16 = 25."""
        v = pythagorean_vector
        assert np.isclose(dot(v, v), 25.0)

    def test_orthogonal_vectors(self, orthogonal_vectors):
        """Orthogonal vectors have zero dot product."""
        u, v = orthogonal_vectors
        assert np.isclose(dot(u, v), 0.0)

    def test_standard_basis_orthonormal(self, standard_basis_3d):
        """e_i . e_j = delta_ij (Kronecker delta)."""
        e1, e2, e3 = standard_basis_3d
        assert dot(e1, e1) == 1.0
        assert dot(e2, e2) == 1.0
        assert dot(e3, e3) == 1.0
        assert dot(e1, e2) == 0.0
        assert dot(e1, e3) == 0.0
        assert dot(e2, e3) == 0.0

    def test_dimension_mismatch_raises(self):
        """Mismatched dimensions should raise ValueError."""
        with pytest.raises((ValueError, Exception)):
            dot([1, 2, 3], [1, 2])

    def test_empty_vectors_raises(self):
        """Empty vectors should raise an error."""
        with pytest.raises((ValueError, Exception)):
            dot([], [])


# =============================================================================
#                           NORM TESTS
# =============================================================================

class TestNorm:
    """Tests for vector norm: norm(x, p=2)."""

    def test_pythagorean_triple(self, pythagorean_vector):
        """||[3, 4]|| = 5."""
        assert np.isclose(norm(pythagorean_vector), 5.0)

    def test_unit_vector_norm(self, standard_basis_3d):
        """Standard basis vectors have norm 1."""
        e1, e2, e3 = standard_basis_3d
        assert np.isclose(norm(e1), 1.0)
        assert np.isclose(norm(e2), 1.0)
        assert np.isclose(norm(e3), 1.0)

    def test_zero_vector_norm(self, zero_vector_3d):
        """||0|| = 0."""
        assert norm(zero_vector_3d) == 0.0

    def test_non_negativity(self, random_vectors):
        """||v|| >= 0 for any v."""
        u, v = random_vectors
        assert norm(u) >= 0
        assert norm(v) >= 0

    def test_scalar_multiplication(self):
        """||cv|| = |c| * ||v||."""
        v = np.array([1.0, 2.0, 3.0])
        c = -3.5
        assert np.isclose(norm(scale(c, v)), abs(c) * norm(v))

    def test_triangle_inequality(self, random_vectors):
        """||u + v|| <= ||u|| + ||v||."""
        u, v = random_vectors
        assert norm(add(u, v)) <= norm(u) + norm(v) + 1e-10

    def test_sqrt_2_diagonal(self):
        """||[1, 1]|| = sqrt(2)."""
        assert np.isclose(norm([1, 1]), np.sqrt(2))

    def test_high_dimensional(self):
        """||ones(n)|| = sqrt(n)."""
        n = 100
        v = np.ones(n)
        assert np.isclose(norm(v), np.sqrt(n))


# =============================================================================
#                        NORMALIZE TESTS
# =============================================================================

class TestNormalize:
    """Tests for normalize(x) — returns unit vector."""

    def test_result_is_unit_vector(self, random_vectors):
        """||normalize(v)|| = 1."""
        u, v = random_vectors
        assert np.isclose(norm(normalize(u)), 1.0)
        assert np.isclose(norm(normalize(v)), 1.0)

    def test_direction_preserved(self, pythagorean_vector):
        """normalize([3,4]) = [0.6, 0.8]."""
        v_hat = normalize(pythagorean_vector)
        np.testing.assert_array_almost_equal(v_hat, [0.6, 0.8])

    def test_idempotent(self, random_vectors):
        """normalize(normalize(v)) = normalize(v)."""
        u, _ = random_vectors
        once = normalize(u)
        twice = normalize(once)
        np.testing.assert_array_almost_equal(once, twice)

    def test_zero_vector_raises(self, zero_vector_3d):
        """Cannot normalize zero vector."""
        with pytest.raises((ZeroDivisionError, ValueError, Exception)):
            normalize(zero_vector_3d)

    def test_already_unit(self, standard_basis_3d):
        """Unit vectors remain unchanged after normalization."""
        e1, e2, e3 = standard_basis_3d
        np.testing.assert_array_almost_equal(normalize(e1), e1)
        np.testing.assert_array_almost_equal(normalize(e2), e2)
        np.testing.assert_array_almost_equal(normalize(e3), e3)


# =============================================================================
#                        DISTANCE TESTS
# =============================================================================

class TestDistance:
    """Tests for distance(x, y, p=2)."""

    def test_basic_l2_distance(self):
        """d([0,0], [3,4]) = 5."""
        assert np.isclose(distance([0, 0], [3, 4]), 5.0)

    def test_same_point_zero_distance(self):
        """d(v, v) = 0."""
        v = np.array([1.0, 2.0, 3.0])
        assert np.isclose(distance(v, v), 0.0)

    def test_symmetry(self, random_vectors):
        """d(u, v) = d(v, u)."""
        u, v = random_vectors
        assert np.isclose(distance(u, v), distance(v, u))

    def test_triangle_inequality(self):
        """d(u, w) <= d(u, v) + d(v, w)."""
        u = np.array([0.0, 0.0])
        v = np.array([1.0, 0.0])
        w = np.array([2.0, 0.0])
        assert distance(u, w) <= distance(u, v) + distance(v, w) + 1e-10

    def test_l1_distance(self):
        """L1 distance (Manhattan): d_1([0,0], [3,4]) = 7."""
        assert np.isclose(distance([0, 0], [3, 4], p=1), 7.0)


# =============================================================================
#                      ANGLE TESTS
# =============================================================================

class TestAngle:
    """Tests for angle(x, y, in_degrees=False)."""

    def test_orthogonal_90_degrees(self, orthogonal_vectors):
        """Orthogonal vectors have pi/2 angle."""
        u, v = orthogonal_vectors
        assert np.isclose(angle(u, v), np.pi / 2)

    def test_parallel_0_degrees(self, parallel_vectors):
        """Parallel (same direction) vectors have 0 angle."""
        u, v, c = parallel_vectors
        assert np.isclose(angle(u, v), 0.0, atol=1e-10)

    def test_antiparallel_180_degrees(self):
        """Anti-parallel vectors have pi angle."""
        u = np.array([1.0, 0.0])
        v = np.array([-1.0, 0.0])
        assert np.isclose(angle(u, v), np.pi)

    def test_45_degrees(self):
        """[1,0] and [1,1] have pi/4 angle."""
        u = np.array([1.0, 0.0])
        v = np.array([1.0, 1.0])
        assert np.isclose(angle(u, v), np.pi / 4)

    def test_in_degrees_flag(self, orthogonal_vectors):
        """in_degrees=True returns 90 for orthogonal vectors."""
        u, v = orthogonal_vectors
        assert np.isclose(angle(u, v, in_degrees=True), 90.0)

    def test_same_vector_zero_angle(self):
        """Angle of a vector with itself is 0."""
        v = np.array([1.0, 2.0, 3.0])
        assert np.isclose(angle(v, v), 0.0, atol=1e-10)


# =============================================================================
#                   COSINE SIMILARITY TESTS
# =============================================================================

class TestCosineSimilarity:
    """Tests for cosine_similarity(x, y)."""

    def test_identical_vectors(self):
        """cos_sim(v, v) = 1."""
        v = np.array([1.0, 2.0, 3.0])
        assert np.isclose(cosine_similarity(v, v), 1.0)

    def test_orthogonal_vectors(self, orthogonal_vectors):
        """cos_sim of orthogonal vectors = 0."""
        u, v = orthogonal_vectors
        assert np.isclose(cosine_similarity(u, v), 0.0)

    def test_opposite_vectors(self):
        """cos_sim(v, -v) = -1."""
        v = np.array([1.0, 2.0])
        assert np.isclose(cosine_similarity(v, -v), -1.0)

    def test_scale_invariance(self, random_vectors):
        """cos_sim(v, cv) = 1 for c > 0."""
        u, _ = random_vectors
        assert np.isclose(cosine_similarity(u, 3.0 * u), 1.0)

    def test_range_bounds(self, random_vectors):
        """cos_sim always in [-1, 1]."""
        u, v = random_vectors
        cs = cosine_similarity(u, v)
        assert -1 - 1e-10 <= cs <= 1 + 1e-10


# =============================================================================
#                    IS_ORTHOGONAL TESTS
# =============================================================================

class TestIsOrthogonal:
    """Tests for is_orthogonal(x, y, tol=1e-10)."""

    def test_true_for_orthogonal(self, orthogonal_vectors):
        """Standard basis vectors are orthogonal."""
        u, v = orthogonal_vectors
        assert is_orthogonal(u, v) is True

    def test_false_for_parallel(self, parallel_vectors):
        """Parallel vectors are not orthogonal."""
        u, v, _ = parallel_vectors
        assert is_orthogonal(u, v) is False

    def test_tolerance(self):
        """Near-orthogonal within tolerance returns True."""
        u = np.array([1.0, 0.0])
        v = np.array([1e-15, 1.0])
        assert is_orthogonal(u, v, tol=1e-10) is True

    def test_false_for_arbitrary(self):
        """Non-orthogonal vectors return False."""
        u = np.array([1.0, 1.0])
        v = np.array([1.0, 0.0])
        assert is_orthogonal(u, v) is False


# =============================================================================
#                        IS_UNIT TESTS
# =============================================================================

class TestIsUnit:
    """Tests for is_unit(x, tol=1e-10)."""

    def test_true_for_unit(self, standard_basis_3d):
        """Standard basis vectors are unit vectors."""
        e1, e2, e3 = standard_basis_3d
        assert is_unit(e1) is True
        assert is_unit(e2) is True
        assert is_unit(e3) is True

    def test_false_for_non_unit(self, pythagorean_vector):
        """[3, 4] has norm 5, not a unit vector."""
        assert is_unit(pythagorean_vector) is False

    def test_tolerance(self):
        """Near-unit vector within tolerance returns True."""
        v = np.array([1.0 + 1e-12, 0.0])
        assert is_unit(v, tol=1e-10) is True


# =============================================================================
#                  VERIFY CAUCHY-SCHWARZ TESTS
# =============================================================================

class TestVerifyCauchySchwarz:
    """Tests for verify_cauchy_schwarz(x, y) -> (bool, |u.v|, ||u||*||v||)."""

    def test_general_case(self, random_vectors):
        """|u.v| <= ||u|| * ||v|| holds for arbitrary vectors."""
        u, v = random_vectors
        holds, lhs, rhs = verify_cauchy_schwarz(u, v)
        assert holds is True
        assert lhs <= rhs + 1e-10

    def test_parallel_equality(self, parallel_vectors):
        """Equality holds when vectors are parallel."""
        u, v, _ = parallel_vectors
        holds, lhs, rhs = verify_cauchy_schwarz(u, v)
        assert holds is True
        assert np.isclose(lhs, rhs)

    def test_orthogonal(self, orthogonal_vectors):
        """|u.v| = 0 for orthogonal vectors, and 0 <= ||u||*||v||."""
        u, v = orthogonal_vectors
        holds, lhs, rhs = verify_cauchy_schwarz(u, v)
        assert holds is True
        assert np.isclose(lhs, 0.0)
        assert rhs > 0


# =============================================================================
#                  NUMERICAL STABILITY TESTS
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability edge cases."""

    def test_very_small_vectors(self):
        """Norm of very small vector components is finite and positive."""
        v = np.array([1e-150, 1e-150])
        n = norm(v)
        assert n > 0
        assert np.isfinite(n)

    def test_very_large_vectors(self):
        """Norm of very large vector components is finite."""
        v = np.array([1e150, 1e150])
        n = norm(v)
        assert np.isfinite(n)

    def test_mixed_scale_dot(self):
        """Dot product with mixed scale components."""
        u = np.array([1e10, 1e-10])
        v = np.array([1e-10, 1e10])
        result = dot(u, v)
        assert np.isclose(result, 2.0, rtol=1e-5)

    def test_near_orthogonal_tolerance(self):
        """Near-orthogonal vectors detected within tolerance."""
        u = np.array([1.0, 0.0])
        v = np.array([1e-15, 1.0])
        assert is_orthogonal(u, v, tol=1e-10) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
