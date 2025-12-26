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
        pass

    @property
    def total_embed_dim(self) -> int:
        """Total dimension when embeddings are concatenated."""
        pass


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
    pass


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
        pass

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
        pass

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
        pass

    def inverse_transform(self, indices: np.ndarray) -> np.ndarray:
        """Transform indices back to categories."""
        pass


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
        pass

    @property
    def name(self) -> str:
        """Model name."""
        pass

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

    def get_program_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Get learned program embeddings.

        Returns:
            Dict mapping program name → embedding vector
        """
        pass

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
        pass

    def find_similar_universities(
        self,
        university: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find universities most similar to given university."""
        pass

    def export_embeddings_for_weaviate(self) -> Dict[str, Any]:
        """
        Export embeddings in format suitable for Weaviate import.

        Returns:
            Dict with 'universities' and 'programs' keys,
            each containing list of {id, name, embedding} dicts
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters for serialization."""
        pass

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters from serialization."""
        pass


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
