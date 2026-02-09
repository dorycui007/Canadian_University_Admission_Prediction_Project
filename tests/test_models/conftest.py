"""
Models module conftest.py — Shared fixtures for model tests.
"""

import pytest
import numpy as np


# ──── Classification data fixtures ────

@pytest.fixture
def default_config():
    """Default ModelConfig instance."""
    from src.models.base import ModelConfig
    return ModelConfig()


@pytest.fixture
def custom_config():
    """Custom ModelConfig with non-default values."""
    from src.models.base import ModelConfig
    return ModelConfig(
        lambda_=0.5,
        max_iter=200,
        tol=1e-8,
        random_state=42,
        extra_params={'solver': 'qr'},
    )


@pytest.fixture
def binary_classification_data(rng):
    """
    100 samples, 5 features, linearly separable binary classification data.
    Returns (X, y).
    """
    n = 100
    p = 5
    X = rng.standard_normal((n, p))
    true_beta = np.array([1.0, -0.5, 0.3, 0.0, 0.8])
    logits = X @ true_beta
    probs = 1 / (1 + np.exp(-logits))
    y = (rng.random(n) < probs).astype(float)
    return X, y


@pytest.fixture
def perfectly_separable_data():
    """
    Perfectly separable data — classes don't overlap.
    Returns (X, y).
    """
    X = np.vstack([
        np.column_stack([np.linspace(2, 5, 50), np.zeros(50)]),
        np.column_stack([np.linspace(-5, -2, 50), np.zeros(50)]),
    ])
    y = np.concatenate([np.ones(50), np.zeros(50)])
    return X, y


@pytest.fixture
def imbalanced_classification_data(rng):
    """
    Imbalanced dataset: 90% negatives, 10% positives.
    Returns (X, y).
    """
    n = 200
    p = 3
    X = rng.standard_normal((n, p))
    y = np.zeros(n)
    y[:20] = 1.0  # 10% positive
    # Shuffle
    idx = rng.permutation(n)
    return X[idx], y[idx]
