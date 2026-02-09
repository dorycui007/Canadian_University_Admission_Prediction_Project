"""
Math module conftest.py — Shared fixtures for all math tests.
"""

import pytest
import numpy as np


# ──── Vector fixtures ────

@pytest.fixture
def zero_vector_2d():
    """Zero vector in R^2."""
    return np.zeros(2)


@pytest.fixture
def zero_vector_3d():
    """Zero vector in R^3."""
    return np.zeros(3)


@pytest.fixture
def standard_basis_2d():
    """Standard basis vectors in R^2: (e1, e2)."""
    return np.array([1.0, 0.0]), np.array([0.0, 1.0])


@pytest.fixture
def standard_basis_3d():
    """Standard basis vectors in R^3: (e1, e2, e3)."""
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    e3 = np.array([0.0, 0.0, 1.0])
    return e1, e2, e3


@pytest.fixture
def pythagorean_vector():
    """Classic 3-4-5 Pythagorean triple: [3, 4] with ||v|| = 5."""
    return np.array([3.0, 4.0])


@pytest.fixture
def parallel_vectors():
    """Parallel vectors: (u, v, c) where v = c * u."""
    u = np.array([1.0, 2.0, 3.0])
    c = 2.0
    v = c * u
    return u, v, c


@pytest.fixture
def orthogonal_vectors():
    """Orthogonal vectors in R^2: (u, v) where u . v = 0."""
    return np.array([1.0, 0.0]), np.array([0.0, 1.0])


@pytest.fixture
def random_vectors(rng):
    """Two random vectors in R^5 for property testing."""
    return rng.standard_normal(5), rng.standard_normal(5)


# ──── Matrix fixtures ────

@pytest.fixture
def identity_3x3():
    """3x3 identity matrix."""
    return np.eye(3)


@pytest.fixture
def tall_matrix_5x3(rng):
    """5x3 tall matrix with full column rank."""
    return rng.standard_normal((5, 3))


@pytest.fixture
def square_matrix_3x3():
    """3x3 invertible matrix."""
    return np.array([[1.0, 2.0, 3.0],
                     [0.0, 4.0, 5.0],
                     [1.0, 0.0, 6.0]])


@pytest.fixture
def symmetric_pd_matrix():
    """2x2 symmetric positive definite matrix."""
    return np.array([[4.0, 2.0],
                     [2.0, 3.0]])


@pytest.fixture
def upper_triangular_3x3():
    """3x3 upper triangular matrix with positive diagonal."""
    return np.array([[2.0, 1.0, 3.0],
                     [0.0, 4.0, 5.0],
                     [0.0, 0.0, 6.0]])


@pytest.fixture
def ill_conditioned_matrix():
    """2x2 matrix with very high condition number."""
    return np.array([[1.0, 1.0],
                     [1.0, 1.0 + 1e-10]])


@pytest.fixture
def rank_deficient_matrix():
    """3x2 rank-1 matrix (columns are linearly dependent)."""
    return np.array([[1.0, 2.0],
                     [2.0, 4.0],
                     [3.0, 6.0]])


@pytest.fixture
def plane_basis():
    """3x2 matrix whose columns span a plane in R^3."""
    return np.array([[1.0, 0.0],
                     [0.0, 1.0],
                     [1.0, 1.0]])


@pytest.fixture
def orthonormal_basis_3d():
    """3x3 orthonormal matrix (identity)."""
    return np.eye(3)
