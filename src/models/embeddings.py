"""
Embedding Model for Categorical Variables
==========================================

This module implements learned embeddings for universities and programs,
replacing sparse one-hot encoding with dense, learned representations.
This is the bridge between traditional ML and neural network approaches.

REFERENCE: Entity Embeddings of Categorical Variables (fast.ai, Kaggle winners)

================================================================================
                        SYSTEM ARCHITECTURE CONTEXT
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    WHERE THIS MODULE FITS                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │        ┌───────────────────────────────────────────────────────┐        │
    │        │                 TRADITIONAL APPROACH                   │        │
    │        │                                                        │        │
    │        │   University: "UofT"  →  [1, 0, 0, ..., 0]  (50 dims) │        │
    │        │   Program:    "CS"    →  [1, 0, 0, ..., 0] (200 dims) │        │
    │        │                                                        │        │
    │        │   Total: 250 dimensions, SPARSE                        │        │
    │        └───────────────────────────────────────────────────────┘        │
    │                              │                                           │
    │                              ▼                                           │
    │        ┌───────────────────────────────────────────────────────┐        │
    │        │               [THIS MODULE]                            │        │
    │        │              EMBEDDING APPROACH                        │        │
    │        │                                                        │        │
    │        │   University: "UofT"  →  [0.2, -0.5, ..., 0.8] (16 d) │        │
    │        │   Program:    "CS"    →  [0.7, 0.1, ..., -0.3] (32 d) │        │
    │        │                                                        │        │
    │        │   Total: 48 dimensions, DENSE                          │        │
    │        └───────────────────────────────────────────────────────┘        │
    │                              │                                           │
    │                              ▼                                           │
    │        ┌───────────────────────────────────────────────────────┐        │
    │        │                  BENEFITS                              │        │
    │        │                                                        │        │
    │        │   • Fewer parameters (16+32 vs 250)                   │        │
    │        │   • Similar programs get similar embeddings            │        │
    │        │   • Embeddings exportable to Weaviate for similarity  │        │
    │        │   • Natural input for attention mechanism             │        │
    │        │                                                        │        │
    │        └───────────────────────────────────────────────────────┘        │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    EMBEDDING FUNDAMENTALS
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  WHAT ARE EMBEDDINGS?                                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  An embedding is a LEARNED mapping from categorical IDs to vectors.     │
    │                                                                          │
    │  EMBEDDING LOOKUP:                                                       │
    │  ──────────────────                                                      │
    │                                                                          │
    │      ┌────────────────────────────────────────────────────────┐         │
    │      │           Embedding Matrix E (n_categories × d)        │         │
    │      ├────────────────────────────────────────────────────────┤         │
    │      │                                                         │         │
    │      │  ID 0 (UofT)     →  [ 0.2, -0.5,  0.1, ...,  0.8 ]     │         │
    │      │  ID 1 (Waterloo) →  [ 0.3, -0.4,  0.2, ...,  0.7 ]     │         │
    │      │  ID 2 (McMaster) →  [-0.1,  0.6, -0.3, ...,  0.2 ]     │         │
    │      │  ...             →  ...                                 │         │
    │      │                                                         │         │
    │      └────────────────────────────────────────────────────────┘         │
    │                                                                          │
    │  Given category ID, return corresponding row of E.                      │
    │  This is equivalent to: one_hot(ID) @ E                                 │
    │                                                                          │
    │                                                                          │
    │  LEARNING EMBEDDINGS:                                                    │
    │  ─────────────────────                                                   │
    │  The embedding vectors are LEARNED during training!                     │
    │  • Initialize randomly                                                   │
    │  • Backpropagate gradients through lookup                               │
    │  • Update to minimize prediction loss                                    │
    │                                                                          │
    │  After training, similar categories have similar embeddings.            │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    EMBEDDING DIMENSION SIZING
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  CHOOSING EMBEDDING DIMENSION                                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  RULE OF THUMB (fast.ai):                                               │
    │  ─────────────────────────                                               │
    │                                                                          │
    │      d = min(600, round(1.6 × n_categories^0.56))                       │
    │                                                                          │
    │  FOR THIS PROJECT:                                                       │
    │  ──────────────────                                                      │
    │                                                                          │
    │      n_universities ≈ 50                                                │
    │      d_uni = min(600, 1.6 × 50^0.56) ≈ 12-16                           │
    │                                                                          │
    │      n_programs ≈ 200                                                   │
    │      d_prog = min(600, 1.6 × 200^0.56) ≈ 25-32                         │
    │                                                                          │
    │  INTUITION:                                                              │
    │  ──────────                                                              │
    │  • Too small: can't capture relationships                               │
    │  • Too large: overfitting, slow training                                │
    │  • The formula balances expressiveness vs. regularization               │
    │                                                                          │
    │  COMPARISON:                                                             │
    │  ───────────                                                             │
    │                                                                          │
    │      Method          │ Parameters │ Captures Similarity?                │
    │      ────────────────┼────────────┼──────────────────────               │
    │      One-hot         │ 250        │ No (orthogonal)                     │
    │      Embeddings      │ 16+32 = 48 │ Yes (learned)                       │
    │      Reduction       │ 81%        │                                      │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    CONNECTION TO SVD
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  EMBEDDINGS AS LOW-RANK MATRIX FACTORIZATION                             │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Consider the interaction matrix:                                        │
    │                                                                          │
    │      A[u, p] = P(admit | university u, program p)                       │
    │                                                                          │
    │  This is a (n_universities × n_programs) matrix.                        │
    │                                                                          │
    │  FULL PARAMETERIZATION:                                                  │
    │  ───────────────────────                                                 │
    │  Learn A directly: 50 × 200 = 10,000 parameters                        │
    │                                                                          │
    │  LOW-RANK APPROXIMATION (SVD view):                                      │
    │  ─────────────────────────────────                                       │
    │      A ≈ U @ Σ @ Vᵀ                                                    │
    │                                                                          │
    │  Where:                                                                  │
    │  • U (50 × k): university embeddings                                    │
    │  • V (200 × k): program embeddings                                      │
    │  • Σ (k × k): diagonal scaling                                          │
    │                                                                          │
    │  Parameters: 50×k + 200×k = 250k                                        │
    │  With k=16: 250×16 = 4,000 parameters (60% reduction!)                  │
    │                                                                          │
    │  EMBEDDING PREDICTION:                                                   │
    │  ─────────────────────                                                   │
    │      score(u, p) = u_embed ⋅ p_embed = Uᵤ ⋅ Vₚ                        │
    │                                                                          │
    │  Similar to matrix factorization in recommender systems!                │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    PYTORCH IMPLEMENTATION NOTES
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  USING nn.Embedding IN PYTORCH                                           │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  PyTorch provides nn.Embedding for efficient lookup:                    │
    │                                                                          │
    │      import torch.nn as nn                                              │
    │                                                                          │
    │      # Create embedding layer                                           │
    │      uni_embed = nn.Embedding(num_universities, embed_dim)              │
    │      prog_embed = nn.Embedding(num_programs, embed_dim)                 │
    │                                                                          │
    │      # Look up embeddings (input: integer IDs)                          │
    │      uni_vec = uni_embed(uni_ids)    # (batch, embed_dim)               │
    │      prog_vec = prog_embed(prog_ids) # (batch, embed_dim)               │
    │                                                                          │
    │      # Combine (e.g., concatenate)                                      │
    │      combined = torch.cat([uni_vec, prog_vec], dim=1)                   │
    │                                                                          │
    │      # Then pass through linear layers for prediction                    │
    │                                                                          │
    │  GRADIENTS:                                                              │
    │  ──────────                                                              │
    │  Gradients flow through the lookup, updating only the rows              │
    │  corresponding to categories in the current batch.                      │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

# PyTorch imports (for actual implementation)
# import torch
# import torch.nn as nn


@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding dimensions.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  EMBEDDING CONFIGURATION                                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Stores dimensions and metadata for embedding layers.                   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Attributes:
        n_universities: Number of unique universities
        n_programs: Number of unique programs
        uni_embed_dim: Dimension of university embeddings
        prog_embed_dim: Dimension of program embeddings
    """
    n_universities: int
    n_programs: int
    uni_embed_dim: int = 16
    prog_embed_dim: int = 32

    @classmethod
    def from_data(
        cls,
        university_ids: np.ndarray,
        program_ids: np.ndarray
    ) -> 'EmbeddingConfig':
        """
        Create config from data, using dimension sizing formula.

        Args:
            university_ids: Array of university IDs in training data
            program_ids: Array of program IDs in training data

        Returns:
            EmbeddingConfig with computed dimensions

        IMPLEMENTATION:
        ────────────────
        n_uni = len(np.unique(university_ids))
        n_prog = len(np.unique(program_ids))

        # fast.ai formula
        uni_dim = min(600, round(1.6 * n_uni ** 0.56))
        prog_dim = min(600, round(1.6 * n_prog ** 0.56))

        return cls(n_uni, n_prog, uni_dim, prog_dim)
        """
        n_uni = len(np.unique(university_ids))
        n_prog = len(np.unique(program_ids))

        # fast.ai formula
        uni_dim = compute_embedding_dim(n_uni)
        prog_dim = compute_embedding_dim(n_prog)

        return cls(n_uni, n_prog, uni_dim, prog_dim)

    @property
    def total_embed_dim(self) -> int:
        """Total dimension when embeddings are concatenated."""
        return self.uni_embed_dim + self.prog_embed_dim


def compute_embedding_dim(n_categories: int) -> int:
    """
    Compute recommended embedding dimension for given number of categories.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  EMBEDDING DIMENSION FORMULA                                             │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      d = min(600, round(1.6 × n^0.56))                                  │
    │                                                                          │
    │  Examples:                                                               │
    │  ─────────                                                               │
    │      n = 10   →  d = 6                                                  │
    │      n = 50   →  d = 14                                                 │
    │      n = 100  →  d = 20                                                 │
    │      n = 500  →  d = 44                                                 │
    │      n = 1000 →  d = 64                                                 │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        n_categories: Number of unique categories

    Returns:
        Recommended embedding dimension

    IMPLEMENTATION:
    ────────────────
    return min(600, round(1.6 * n_categories ** 0.56))
    """
    return min(600, round(1.6 * n_categories ** 0.56))


class CategoryEncoder:
    """
    Maps categorical values to integer indices for embedding lookup.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  CATEGORY ENCODING                                                       │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Embeddings use integer indices, not strings!                           │
    │                                                                          │
    │      "UofT"     →  0                                                    │
    │      "Waterloo" →  1                                                    │
    │      "McMaster" →  2                                                    │
    │      ...                                                                │
    │                                                                          │
    │  This class handles the mapping and tracks unknown categories.          │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Example:
        >>> encoder = CategoryEncoder()
        >>> encoder.fit(['UofT', 'Waterloo', 'McMaster'])
        >>> encoder.transform(['UofT', 'McMaster', 'Unknown'])
        array([0, 2, 3])  # Unknown gets special index
    """

    def __init__(self, unknown_token: str = '<UNK>'):
        """
        Initialize encoder.

        Args:
            unknown_token: Token to use for unseen categories
        """
        self.unknown_token = unknown_token
        self.category_to_idx = None
        self.idx_to_category = None
        self.n_categories = None
        self.unknown_idx = None

    def fit(self, categories: np.ndarray) -> 'CategoryEncoder':
        """
        Learn the mapping from categories to indices.

        Args:
            categories: Array of category values

        Returns:
            self

        IMPLEMENTATION:
        ────────────────
        unique = np.unique(categories)
        self.category_to_idx = {cat: idx for idx, cat in enumerate(unique)}
        self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx.items()}
        self.n_categories = len(unique) + 1  # +1 for unknown
        self.unknown_idx = len(unique)
        return self
        """
        unique = sorted(np.unique(categories))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(unique)}
        self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx.items()}
        self.n_categories = len(unique) + 1  # +1 for unknown
        self.unknown_idx = len(unique)
        return self

    def transform(self, categories: np.ndarray) -> np.ndarray:
        """
        Transform categories to indices.

        Args:
            categories: Array of category values

        Returns:
            Array of integer indices

        IMPLEMENTATION:
        ────────────────
        return np.array([
            self.category_to_idx.get(cat, self.unknown_idx)
            for cat in categories
        ])
        """
        return np.array([
            self.category_to_idx.get(cat, self.unknown_idx)
            for cat in categories
        ])

    def inverse_transform(self, indices: np.ndarray) -> np.ndarray:
        """Transform indices back to categories."""
        return np.array([
            self.idx_to_category.get(int(idx), self.unknown_token)
            for idx in indices
        ])


class EmbeddingModel:
    """
    Embedding-based model for admission prediction.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  EMBEDDING MODEL ARCHITECTURE                                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  INPUT:                                                                  │
    │  ──────                                                                  │
    │      university_id (int) → Embedding → [u₁, u₂, ..., uₖ]               │
    │      program_id (int)    → Embedding → [p₁, p₂, ..., pₘ]               │
    │      continuous features → directly   → [avg, ...]                      │
    │                                                                          │
    │  ARCHITECTURE:                                                           │
    │  ─────────────                                                           │
    │                                                                          │
    │      ┌─────────────┐                                                    │
    │      │ uni_id = 3  │                                                    │
    │      └──────┬──────┘                                                    │
    │             ▼                                                            │
    │      ┌─────────────┐       ┌─────────────┐                              │
    │      │ Uni Embed   │       │ Continuous  │                              │
    │      │ (50 × 16)   │       │ Features    │                              │
    │      └──────┬──────┘       │ [avg, ...]  │                              │
    │             │              └──────┬──────┘                              │
    │             ▼                     │                                      │
    │      ┌─────────────┐              │                                      │
    │      │ Prog Embed  │              │                                      │
    │      │ (200 × 32)  │              │                                      │
    │      └──────┬──────┘              │                                      │
    │             │                     │                                      │
    │             └──────────┬──────────┘                                      │
    │                        ▼                                                 │
    │                 ┌─────────────┐                                         │
    │                 │ Concatenate │                                         │
    │                 │ [16+32+1]   │                                         │
    │                 └──────┬──────┘                                         │
    │                        ▼                                                 │
    │                 ┌─────────────┐                                         │
    │                 │   Linear    │                                         │
    │                 │   Layer     │                                         │
    │                 └──────┬──────┘                                         │
    │                        ▼                                                 │
    │                 ┌─────────────┐                                         │
    │                 │   Sigmoid   │                                         │
    │                 │  P(admit)   │                                         │
    │                 └─────────────┘                                         │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Example:
        >>> model = EmbeddingModel(config)
        >>> model.fit(X, y, uni_ids, prog_ids)
        >>> probs = model.predict_proba(X_new, uni_ids_new, prog_ids_new)
        >>> uni_embeddings = model.get_university_embeddings()
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        hidden_dims: Optional[List[int]] = None,
        lambda_: float = 0.01,
        learning_rate: float = 0.001,
        n_epochs: int = 100,
        batch_size: int = 32
    ):
        """
        Initialize embedding model.

        Args:
            config: EmbeddingConfig with dimensions
            hidden_dims: Optional hidden layer dimensions (e.g., [64, 32])
            lambda_: Weight decay (L2 regularization)
            learning_rate: Adam learning rate
            n_epochs: Training epochs
            batch_size: Mini-batch size

        IMPLEMENTATION:
        ────────────────
        Store all parameters.
        Initialize encoders:
            self.uni_encoder = CategoryEncoder()
            self.prog_encoder = CategoryEncoder()
        Initialize model to None (created in fit)
            self._model = None
        """
        self.config = config
        self.hidden_dims = hidden_dims or []
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.uni_encoder = CategoryEncoder()
        self.prog_encoder = CategoryEncoder()

        self._uni_embeddings = None
        self._prog_embeddings = None
        self._weights = []
        self._biases = []
        self._is_fitted = False
        self._n_continuous = None
        self._model = None

    @property
    def name(self) -> str:
        """Model name."""
        return "Embedding-Based Admission Model"

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted

    @property
    def n_params(self) -> int:
        """
        Total number of learnable parameters.

        CALCULATION:
        ─────────────
        uni_embed_params = n_universities × uni_embed_dim
        prog_embed_params = n_programs × prog_embed_dim
        linear_params = depends on hidden_dims
        """
        # Embedding parameters
        uni_embed_params = (self.config.n_universities + 1) * self.config.uni_embed_dim
        prog_embed_params = (self.config.n_programs + 1) * self.config.prog_embed_dim
        total = uni_embed_params + prog_embed_params

        # Network weight parameters
        if self._weights:
            for w, b in zip(self._weights, self._biases):
                total += w.size + b.size
        else:
            # Estimate from config when model not yet built
            n_continuous = self._n_continuous if self._n_continuous is not None else 0
            input_dim = self.config.total_embed_dim + n_continuous
            for h_dim in self.hidden_dims:
                total += input_dim * h_dim + h_dim  # weight + bias
                input_dim = h_dim
            total += input_dim * 1 + 1  # output layer weight + bias

        return total

    def _build_model(self, n_continuous: int):
        """
        Build the PyTorch model.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  MODEL CONSTRUCTION                                                  │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  Components:                                                         │
        │  • uni_embedding: nn.Embedding(n_uni, uni_dim)                      │
        │  • prog_embedding: nn.Embedding(n_prog, prog_dim)                   │
        │  • fc_layers: nn.Sequential of Linear + ReLU                        │
        │  • output: nn.Linear(last_dim, 1) + sigmoid                        │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            n_continuous: Number of continuous features

        IMPLEMENTATION (PyTorch):
        ─────────────────────────
        import torch.nn as nn

        class EmbedNet(nn.Module):
            def __init__(self, config, n_continuous, hidden_dims):
                super().__init__()
                self.uni_embed = nn.Embedding(config.n_universities + 1,
                                               config.uni_embed_dim)
                self.prog_embed = nn.Embedding(config.n_programs + 1,
                                                config.prog_embed_dim)

                input_dim = config.total_embed_dim + n_continuous

                layers = []
                for h_dim in (hidden_dims or []):
                    layers.append(nn.Linear(input_dim, h_dim))
                    layers.append(nn.ReLU())
                    input_dim = h_dim

                layers.append(nn.Linear(input_dim, 1))
                self.fc = nn.Sequential(*layers)

            def forward(self, uni_ids, prog_ids, x_continuous):
                uni_vec = self.uni_embed(uni_ids)
                prog_vec = self.prog_embed(prog_ids)
                combined = torch.cat([uni_vec, prog_vec, x_continuous], dim=1)
                return torch.sigmoid(self.fc(combined))
        """
        self._n_continuous = n_continuous

        # Initialize embedding matrices with small random values (+1 for unknown token)
        self._uni_embeddings = np.random.randn(
            self.config.n_universities + 1, self.config.uni_embed_dim
        ) * 0.01
        self._prog_embeddings = np.random.randn(
            self.config.n_programs + 1, self.config.prog_embed_dim
        ) * 0.01

        # Build fully-connected layers
        input_dim = self.config.total_embed_dim + n_continuous
        self._weights = []
        self._biases = []

        for h_dim in self.hidden_dims:
            # Xavier-like initialization
            self._weights.append(np.random.randn(input_dim, h_dim) * np.sqrt(2.0 / input_dim))
            self._biases.append(np.zeros(h_dim))
            input_dim = h_dim

        # Output layer
        self._weights.append(np.random.randn(input_dim, 1) * np.sqrt(2.0 / input_dim))
        self._biases.append(np.zeros(1))

        self._model = True  # Flag indicating model is built

    def _forward(self, uni_idx: np.ndarray, prog_idx: np.ndarray, X_cont: np.ndarray):
        """
        Forward pass through the network.

        Returns:
            Tuple of (output probabilities, list of (pre_activation, post_activation) for each hidden layer)
        """
        from .logistic import sigmoid

        # Embedding lookup
        uni_embed = self._uni_embeddings[uni_idx]   # (batch, uni_embed_dim)
        prog_embed = self._prog_embeddings[prog_idx]  # (batch, prog_embed_dim)

        # Concatenate embeddings with continuous features
        combined = np.concatenate([uni_embed, prog_embed, X_cont], axis=1)

        # Forward through hidden layers
        activations = []  # Store (input, pre_relu, post_relu) for backprop
        h = combined
        for i in range(len(self._weights) - 1):
            z = h @ self._weights[i] + self._biases[i]
            a = np.maximum(z, 0)  # ReLU
            activations.append((h, z, a))
            h = a

        # Output layer
        z_out = h @ self._weights[-1] + self._biases[-1]  # (batch, 1)
        probs = sigmoid(z_out.ravel())  # (batch,)

        return probs, activations, h, combined

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        university_ids: np.ndarray,
        program_ids: np.ndarray,
        validation_data: Optional[Tuple] = None
    ) -> 'EmbeddingModel':
        """
        Train the embedding model.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  TRAINING LOOP                                                       │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  1. Encode categorical IDs to integers                              │
        │  2. Build PyTorch model                                             │
        │  3. Create DataLoader for batching                                  │
        │  4. For each epoch:                                                 │
        │        For each batch:                                              │
        │            a. Forward pass                                          │
        │            b. Compute BCE loss                                      │
        │            c. Backward pass                                         │
        │            d. Optimizer step                                        │
        │        If validation: compute validation loss                       │
        │  5. Save best model                                                 │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            X: Continuous features (n_samples, n_continuous)
            y: Binary labels (n_samples,)
            university_ids: University identifiers (strings)
            program_ids: Program identifiers (strings)
            validation_data: Optional (X_val, y_val, uni_val, prog_val)

        Returns:
            self
        """
        from .logistic import sigmoid

        # 1. Encode categorical IDs
        self.uni_encoder.fit(university_ids)
        self.prog_encoder.fit(program_ids)
        uni_idx = self.uni_encoder.transform(university_ids)
        prog_idx = self.prog_encoder.transform(program_ids)

        # Handle 1D X
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        n_continuous = X.shape[1]

        # 2. Build model
        self._build_model(n_continuous)

        y = y.astype(float)

        # 3. Training loop with mini-batch SGD
        for epoch in range(self.n_epochs):
            # Shuffle data
            perm = np.random.permutation(n_samples)
            X_shuf = X[perm]
            y_shuf = y[perm]
            uni_shuf = uni_idx[perm]
            prog_shuf = prog_idx[perm]

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]
                uni_batch = uni_shuf[start:end]
                prog_batch = prog_shuf[start:end]
                batch_size = end - start

                # Forward pass
                probs, activations, last_hidden, combined = self._forward(
                    uni_batch, prog_batch, X_batch
                )

                # Clip for numerical stability
                probs = np.clip(probs, 1e-15, 1 - 1e-15)

                # Backward pass: compute gradient of BCE loss
                # dL/dz_out = probs - y  (derivative of BCE w.r.t. pre-sigmoid logits)
                d_out = (probs - y_batch).reshape(-1, 1)  # (batch, 1)

                # Gradient for output layer
                d_w_out = last_hidden.T @ d_out / batch_size + self.lambda_ * self._weights[-1]
                d_b_out = np.mean(d_out, axis=0)

                # Backprop through hidden layers
                d_hidden_grads_w = []
                d_hidden_grads_b = []
                d_prev = d_out @ self._weights[-1].T  # (batch, last_hidden_dim)

                for i in range(len(self._weights) - 2, -1, -1):
                    h_in, z, a = activations[i]
                    # ReLU derivative
                    d_relu = d_prev * (z > 0).astype(float)
                    d_w = h_in.T @ d_relu / batch_size + self.lambda_ * self._weights[i]
                    d_b = np.mean(d_relu, axis=0)
                    d_hidden_grads_w.insert(0, d_w)
                    d_hidden_grads_b.insert(0, d_b)
                    if i > 0:
                        d_prev = d_relu @ self._weights[i].T

                    # Gradient for combined input (needed for embedding gradients)
                    if i == 0:
                        d_combined = d_relu @ self._weights[i].T  # (batch, combined_dim)

                # If no hidden layers, gradient flows directly from output
                if len(self._weights) == 1:
                    d_combined = d_out @ self._weights[-1].T  # (batch, combined_dim)

                # Extract embedding gradients from d_combined
                uni_dim = self.config.uni_embed_dim
                prog_dim = self.config.prog_embed_dim
                d_uni_embed = d_combined[:, :uni_dim]  # (batch, uni_dim)
                d_prog_embed = d_combined[:, uni_dim:uni_dim + prog_dim]  # (batch, prog_dim)

                # Update embedding matrices: only update rows that were used
                for j in range(batch_size):
                    self._uni_embeddings[uni_batch[j]] -= self.learning_rate * d_uni_embed[j] / batch_size
                    self._prog_embeddings[prog_batch[j]] -= self.learning_rate * d_prog_embed[j] / batch_size

                # Update network weights
                for i, (d_w, d_b) in enumerate(zip(d_hidden_grads_w, d_hidden_grads_b)):
                    self._weights[i] -= self.learning_rate * d_w
                    self._biases[i] -= self.learning_rate * d_b

                # Update output layer
                self._weights[-1] -= self.learning_rate * d_w_out
                self._biases[-1] -= self.learning_rate * d_b_out

        self._is_fitted = True
        return self

    def predict_proba(
        self,
        X: np.ndarray,
        university_ids: np.ndarray,
        program_ids: np.ndarray
    ) -> np.ndarray:
        """
        Predict admission probabilities.

        Args:
            X: Continuous features
            university_ids: University identifiers
            program_ids: Program identifiers

        Returns:
            Probabilities (n_samples,)
        """
        if not self._is_fitted:
            return None

        # Handle 1D X
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        uni_idx = self.uni_encoder.transform(university_ids)
        prog_idx = self.prog_encoder.transform(program_ids)

        probs, _, _, _ = self._forward(uni_idx, prog_idx, X)
        return probs

    def get_university_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Get learned university embeddings.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  EMBEDDING EXTRACTION                                                │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  Returns dict mapping university name to embedding vector.          │
        │                                                                      │
        │  Use for:                                                           │
        │  • Storing in Weaviate for similarity search                        │
        │  • Visualizing university clusters                                  │
        │  • Analyzing learned relationships                                  │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        Returns:
            Dict mapping university name → embedding vector

        Example:
            >>> embeddings = model.get_university_embeddings()
            >>> uoft_vec = embeddings['University of Toronto']
            >>> print(uoft_vec.shape)  # (16,)
        """
        if self._uni_embeddings is None:
            return None
        return {
            name: self._uni_embeddings[idx].copy()
            for name, idx in self.uni_encoder.category_to_idx.items()
        }

    def get_program_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Get learned program embeddings.

        Returns:
            Dict mapping program name → embedding vector
        """
        if self._prog_embeddings is None:
            return None
        return {
            name: self._prog_embeddings[idx].copy()
            for name, idx in self.prog_encoder.category_to_idx.items()
        }

    def find_similar_programs(
        self,
        program: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find programs most similar to given program.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  SIMILARITY SEARCH                                                   │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  Uses cosine similarity between embeddings:                         │
        │                                                                      │
        │      sim(a, b) = (a ⋅ b) / (||a|| × ||b||)                         │
        │                                                                      │
        │  Example output for "UofT CS":                                      │
        │      1. Waterloo CS (0.92)                                          │
        │      2. UofT ECE (0.85)                                             │
        │      3. McGill CS (0.83)                                            │
        │      ...                                                            │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            program: Program to find similar programs for
            top_k: Number of similar programs to return

        Returns:
            List of (program_name, similarity_score) tuples
        """
        embeddings = self.get_program_embeddings()
        if embeddings is None or program not in embeddings:
            return []

        target = embeddings[program]
        target_norm = np.linalg.norm(target)
        if target_norm == 0:
            return []

        similarities = []
        for name, vec in embeddings.items():
            if name == program:
                continue
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue
            sim = np.dot(target, vec) / (target_norm * vec_norm)
            similarities.append((name, float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def find_similar_universities(
        self,
        university: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find universities most similar to given university."""
        embeddings = self.get_university_embeddings()
        if embeddings is None or university not in embeddings:
            return []

        target = embeddings[university]
        target_norm = np.linalg.norm(target)
        if target_norm == 0:
            return []

        similarities = []
        for name, vec in embeddings.items():
            if name == university:
                continue
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue
            sim = np.dot(target, vec) / (target_norm * vec_norm)
            similarities.append((name, float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def export_embeddings_for_weaviate(self) -> Dict[str, Any]:
        """
        Export embeddings in format suitable for Weaviate import.

        Returns:
            Dict with 'universities' and 'programs' keys,
            each containing list of {id, name, embedding} dicts
        """
        uni_embeddings = self.get_university_embeddings()
        prog_embeddings = self.get_program_embeddings()

        if uni_embeddings is None or prog_embeddings is None:
            return None

        universities = []
        for name, vec in uni_embeddings.items():
            universities.append({
                'id': self.uni_encoder.category_to_idx[name],
                'name': name,
                'embedding': vec.tolist(),
            })

        programs = []
        for name, vec in prog_embeddings.items():
            programs.append({
                'id': self.prog_encoder.category_to_idx[name],
                'name': name,
                'embedding': vec.tolist(),
            })

        return {
            'universities': universities,
            'programs': programs,
        }

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters for serialization."""
        params = {
            'config': {
                'n_universities': self.config.n_universities,
                'n_programs': self.config.n_programs,
                'uni_embed_dim': self.config.uni_embed_dim,
                'prog_embed_dim': self.config.prog_embed_dim,
            },
            'hidden_dims': self.hidden_dims,
            'lambda_': self.lambda_,
            'learning_rate': self.learning_rate,
            'n_epochs': self.n_epochs,
            'batch_size': self.batch_size,
            'is_fitted': self._is_fitted,
        }
        if self._uni_embeddings is not None:
            params['uni_embeddings'] = self._uni_embeddings.tolist()
        if self._prog_embeddings is not None:
            params['prog_embeddings'] = self._prog_embeddings.tolist()
        if self._weights:
            params['weights'] = [w.tolist() for w in self._weights]
            params['biases'] = [b.tolist() for b in self._biases]
        if self.uni_encoder.category_to_idx is not None:
            params['uni_category_to_idx'] = self.uni_encoder.category_to_idx
        if self.prog_encoder.category_to_idx is not None:
            params['prog_category_to_idx'] = self.prog_encoder.category_to_idx
        return params

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters from serialization."""
        if 'config' in params:
            cfg = params['config']
            self.config = EmbeddingConfig(**cfg)
        if 'hidden_dims' in params:
            self.hidden_dims = params['hidden_dims']
        if 'lambda_' in params:
            self.lambda_ = params['lambda_']
        if 'learning_rate' in params:
            self.learning_rate = params['learning_rate']
        if 'n_epochs' in params:
            self.n_epochs = params['n_epochs']
        if 'batch_size' in params:
            self.batch_size = params['batch_size']
        if 'uni_embeddings' in params:
            self._uni_embeddings = np.array(params['uni_embeddings'])
        if 'prog_embeddings' in params:
            self._prog_embeddings = np.array(params['prog_embeddings'])
        if 'weights' in params:
            self._weights = [np.array(w) for w in params['weights']]
            self._biases = [np.array(b) for b in params['biases']]
        if 'uni_category_to_idx' in params:
            self.uni_encoder.category_to_idx = params['uni_category_to_idx']
            self.uni_encoder.idx_to_category = {v: k for k, v in params['uni_category_to_idx'].items()}
            self.uni_encoder.n_categories = len(params['uni_category_to_idx']) + 1
            self.uni_encoder.unknown_idx = len(params['uni_category_to_idx'])
        if 'prog_category_to_idx' in params:
            self.prog_encoder.category_to_idx = params['prog_category_to_idx']
            self.prog_encoder.idx_to_category = {v: k for k, v in params['prog_category_to_idx'].items()}
            self.prog_encoder.n_categories = len(params['prog_category_to_idx']) + 1
            self.prog_encoder.unknown_idx = len(params['prog_category_to_idx'])
        if params.get('is_fitted', False):
            self._is_fitted = True


# =============================================================================
#                           TODO LIST
# =============================================================================
"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TODO                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  HIGH PRIORITY (core embedding):                                             │
│  ────────────────────────────────                                            │
│  [ ] Implement compute_embedding_dim()                                      │
│  [ ] Implement CategoryEncoder.fit() and transform()                        │
│  [ ] Implement EmbeddingConfig.from_data()                                  │
│  [ ] Implement EmbeddingModel._build_model() in PyTorch                     │
│  [ ] Implement EmbeddingModel.fit() training loop                           │
│  [ ] Implement EmbeddingModel.predict_proba()                               │
│                                                                              │
│  MEDIUM PRIORITY (embedding analysis):                                       │
│  ──────────────────────────────────────                                      │
│  [ ] Implement get_university_embeddings()                                  │
│  [ ] Implement get_program_embeddings()                                     │
│  [ ] Implement find_similar_programs() with cosine similarity              │
│  [ ] Implement find_similar_universities()                                  │
│                                                                              │
│  LOW PRIORITY (integration):                                                 │
│  ────────────────────────────                                                │
│  [ ] Implement export_embeddings_for_weaviate()                             │
│  [ ] Add embedding visualization utilities                                  │
│  [ ] Implement model serialization                                          │
│                                                                              │
│  TESTING:                                                                    │
│  ────────                                                                    │
│  [ ] Verify embedding dimensions match formula                              │
│  [ ] Test CategoryEncoder handles unknown categories                        │
│  [ ] Compare predictions to one-hot logistic baseline                       │
│  [ ] Verify similar programs have high cosine similarity                    │
│                                                                              │
│  ADVANCED:                                                                   │
│  ─────────                                                                   │
│  [ ] Add pre-trained embedding initialization                               │
│  [ ] Implement embedding regularization (orthogonality)                     │
│  [ ] Add embedding dropout                                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    print("Embedding Model for Categorical Variables")
    print("=" * 50)
    print()
    print("Replaces sparse one-hot encoding with dense learned vectors.")
    print()
    print("Benefits:")
    print("  • Fewer parameters (dimensionality reduction)")
    print("  • Similar categories get similar embeddings")
    print("  • Embeddings usable for similarity search")
    print("  • Natural input for attention mechanisms")
