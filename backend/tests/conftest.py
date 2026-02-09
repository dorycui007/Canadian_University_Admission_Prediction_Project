"""
Root conftest.py — Shared fixtures for the entire test suite.
"""

import pytest
import numpy as np
import os


# ──── Random seed control ────

@pytest.fixture
def rng():
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


# ──── Tolerance constants ────

@pytest.fixture
def atol():
    """Absolute tolerance for float comparisons."""
    return 1e-10


@pytest.fixture
def rtol():
    """Relative tolerance for float comparisons."""
    return 1e-8


# ──── Project paths ────

@pytest.fixture
def project_root():
    """Return project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def data_dir(project_root):
    """Return data directory path."""
    return os.path.join(project_root, "data")


@pytest.fixture
def mappings_dir(data_dir):
    """Return data/mappings directory path."""
    return os.path.join(data_dir, "mappings")


@pytest.fixture
def raw_data_dir(data_dir):
    """Return data/raw directory path."""
    return os.path.join(data_dir, "raw")


# ──── Small design matrix fixtures ────

@pytest.fixture
def small_design_matrix(rng):
    """10x3 design matrix with intercept column, full rank."""
    n = 10
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    return X


@pytest.fixture
def binary_target(rng):
    """Binary target vector of length 10."""
    return rng.choice([0, 1], size=10, p=[0.4, 0.6]).astype(float)
