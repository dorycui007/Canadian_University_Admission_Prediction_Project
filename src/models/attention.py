"""
Attention Mechanism Module
===========================

This module implements attention mechanisms for interpretable predictions.
Attention allows the model to "focus" on relevant programs when making
predictions, providing explainability.

REFERENCE: Attention Is All You Need (Vaswani et al.), TabNet, SAINT

================================================================================
                        SYSTEM ARCHITECTURE CONTEXT
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    WHERE THIS MODULE FITS                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │                      embeddings.py                                       │
    │                           │                                              │
    │                           ▼                                              │
    │                 ┌─────────────────┐                                     │
    │                 │  [THIS MODULE]  │                                     │
    │                 │  attention.py   │                                     │
    │                 └────────┬────────┘                                     │
    │                          │                                               │
    │                          ▼                                               │
    │              ┌─────────────────────────┐                                │
    │              │    INTERPRETABLE        │                                │
    │              │    PREDICTIONS          │                                │
    │              │                         │                                │
    │              │  "72% admission chance  │                                │
    │              │   Model focused on:     │                                │
    │              │   • Waterloo CS (35%)   │ ← Attention weights           │
    │              │   • UofT CS (25%)       │                                │
    │              │   • McGill CS (15%)     │                                │
    │              │   • ..."                │                                │
    │              │                         │                                │
    │              └─────────────────────────┘                                │
    │                                                                          │
    │  ATTENTION ANSWERS:                                                      │
    │  ───────────────────                                                     │
    │  "Which similar programs did the model consider?"                       │
    │  "What patterns influenced this prediction?"                            │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    ATTENTION FUNDAMENTALS
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  SCALED DOT-PRODUCT ATTENTION                                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Given:                                                                  │
    │  • Query Q: What we're looking for                                      │
    │  • Key K: What we're searching through                                  │
    │  • Value V: What we want to retrieve                                    │
    │                                                                          │
    │  FORMULA:                                                                │
    │  ────────                                                                │
    │                                                                          │
    │      Attention(Q, K, V) = softmax(QKᵀ / √d) × V                        │
    │                                                                          │
    │  WHERE:                                                                  │
    │  ──────                                                                  │
    │  • QKᵀ: similarity scores between query and all keys                   │
    │  • √d: scaling factor (prevents dot products from growing too large)    │
    │  • softmax: converts scores to probabilities (sum to 1)                 │
    │  • × V: weighted sum of values                                          │
    │                                                                          │
    │  VISUALIZATION:                                                          │
    │  ───────────────                                                         │
    │                                                                          │
    │      Query (what I'm applying for)                                      │
    │         │                                                                │
    │         ▼                                                                │
    │      ┌─────────────────────────────────────────────────┐                │
    │      │                                                  │                │
    │      │   Key 1 ────────────► Score 1 ─── 0.35 ────┐    │                │
    │      │   (Waterloo CS)                            │    │                │
    │      │                                            │    │                │
    │      │   Key 2 ────────────► Score 2 ─── 0.25 ────┼────┼──► Weighted   │
    │      │   (UofT CS)                                │    │     Sum       │
    │      │                                            │    │                │
    │      │   Key 3 ────────────► Score 3 ─── 0.15 ────┤    │                │
    │      │   (McGill CS)                              │    │                │
    │      │                                            │    │                │
    │      │   ...                 ...       Attention  │    │                │
    │      │                                 Weights    │    │                │
    │      │                                            │    │                │
    │      └────────────────────────────────────────────┴────┘                │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    SELF-ATTENTION FOR PROGRAMS
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  PROGRAM-LEVEL SELF-ATTENTION                                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  IDEA:                                                                   │
    │  ─────                                                                   │
    │  When predicting for program P, look at other programs and              │
    │  aggregate information from similar ones.                                │
    │                                                                          │
    │  EXAMPLE:                                                                │
    │  ────────                                                                │
    │  Predicting for "Queens Computing":                                      │
    │                                                                          │
    │      Queens Computing (query) attends to:                               │
    │      • Waterloo CS (0.30) - similar program                             │
    │      • UofT CS (0.25) - similar program                                 │
    │      • Western CS (0.20) - similar university tier                      │
    │      • Queens ECE (0.15) - same university                              │
    │      • Others (0.10)                                                    │
    │                                                                          │
    │  The model "borrows" information from related programs!                 │
    │  This helps with sparse data (few observations for Queens Computing).  │
    │                                                                          │
    │  ARCHITECTURE:                                                           │
    │  ─────────────                                                           │
    │                                                                          │
    │      ┌─────────────────┐                                                │
    │      │ Program Embed   │                                                │
    │      │ (target prog)   │                                                │
    │      └────────┬────────┘                                                │
    │               │                                                          │
    │               ▼                                                          │
    │      ┌─────────────────┐        ┌─────────────────┐                     │
    │      │  Query = W_q ×  │        │ All Program     │                     │
    │      │  embed          │        │ Embeddings      │                     │
    │      └────────┬────────┘        └────────┬────────┘                     │
    │               │                          │                               │
    │               │                          ▼                               │
    │               │                 ┌─────────────────┐                     │
    │               │                 │ Keys = W_k ×    │                     │
    │               │                 │ embeddings      │                     │
    │               │                 └────────┬────────┘                     │
    │               │                          │                               │
    │               └──────────► QKᵀ ◄─────────┘                              │
    │                             │                                            │
    │                             ▼                                            │
    │                    ┌─────────────────┐                                  │
    │                    │ Attention Wts   │                                  │
    │                    │ softmax(QKᵀ/√d) │                                  │
    │                    └────────┬────────┘                                  │
    │                             │                                            │
    │                             ▼                                            │
    │                    ┌─────────────────┐                                  │
    │                    │ Context Vector  │                                  │
    │                    │ = Σ wᵢ × Vᵢ     │                                  │
    │                    └─────────────────┘                                  │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    MULTI-HEAD ATTENTION
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  MULTIPLE ATTENTION HEADS                                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Instead of one attention, use h parallel attention "heads":            │
    │                                                                          │
    │      head_1: Attends to competitiveness                                 │
    │      head_2: Attends to program type (CS vs Eng)                        │
    │      head_3: Attends to university prestige                             │
    │      head_4: Attends to geographic region                               │
    │      ...                                                                │
    │                                                                          │
    │  Each head learns different relationships!                               │
    │                                                                          │
    │  FORMULA:                                                                │
    │  ────────                                                                │
    │                                                                          │
    │      MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_o            │
    │                                                                          │
    │      where head_i = Attention(Q×W_q^i, K×W_k^i, V×W_v^i)               │
    │                                                                          │
    │  VISUALIZATION:                                                          │
    │  ───────────────                                                         │
    │                                                                          │
    │         Input                                                            │
    │           │                                                              │
    │      ┌────┴────┬────────┬────────┐                                      │
    │      ▼         ▼        ▼        ▼                                      │
    │   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                                  │
    │   │Head 1│ │Head 2│ │Head 3│ │Head 4│                                  │
    │   └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘                                  │
    │      │        │        │        │                                        │
    │      └────────┴────┬───┴────────┘                                       │
    │                    │                                                     │
    │                    ▼                                                     │
    │             ┌────────────┐                                              │
    │             │ Concatenate│                                              │
    │             └──────┬─────┘                                              │
    │                    ▼                                                     │
    │             ┌────────────┐                                              │
    │             │  Linear    │                                              │
    │             │   W_o      │                                              │
    │             └────────────┘                                              │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from .logistic import sigmoid

# PyTorch imports (for actual implementation)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled dot-product attention.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  ATTENTION COMPUTATION                                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V                      │
    │                                                                          │
    │  SHAPES:                                                                 │
    │  ───────                                                                 │
    │  Q: (batch, n_queries, d_k)                                             │
    │  K: (batch, n_keys, d_k)                                                │
    │  V: (batch, n_keys, d_v)                                                │
    │  Output: (batch, n_queries, d_v)                                        │
    │  Attention weights: (batch, n_queries, n_keys)                          │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        Q: Query tensor (batch, n_queries, d_k)
        K: Key tensor (batch, n_keys, d_k)
        V: Value tensor (batch, n_keys, d_v)
        mask: Optional mask for attention (0 = attend, -inf = ignore)

    Returns:
        Tuple of (output, attention_weights)

    IMPLEMENTATION:
    ────────────────
    d_k = Q.shape[-1]

    # Compute attention scores
    scores = Q @ K.transpose(-2, -1) / np.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        scores = scores + mask  # mask should be 0 or -inf

    # Softmax to get attention weights
    weights = softmax(scores, axis=-1)

    # Weighted sum of values
    output = weights @ V

    return output, weights
    """
    d_k = Q.shape[-1]

    # Compute attention scores
    scores = Q @ np.swapaxes(K, -2, -1) / np.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        scores = scores + mask  # mask should be 0 or -inf

    # Softmax to get attention weights
    weights = softmax(scores, axis=-1)

    # Weighted sum of values
    output = weights @ V

    return output, weights


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  SOFTMAX FUNCTION                                                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │      softmax(x)_i = exp(x_i) / Σ_j exp(x_j)                            │
    │                                                                          │
    │  NUMERICAL STABILITY:                                                    │
    │  ─────────────────────                                                   │
    │  Subtract max before exp to prevent overflow:                           │
    │                                                                          │
    │      softmax(x) = softmax(x - max(x))                                   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        x: Input array
        axis: Axis to apply softmax along

    Returns:
        Softmax probabilities (sum to 1 along axis)

    IMPLEMENTATION:
    ────────────────
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


@dataclass
class AttentionOutput:
    """
    Output from attention mechanism with interpretability.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  ATTENTION OUTPUT                                                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Stores both the computed representation AND the attention weights      │
    │  for interpretability.                                                  │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Attributes:
        context: The computed context vector
        attention_weights: Weights showing what the model focused on
        key_names: Names corresponding to each key (for interpretability)
    """
    context: np.ndarray
    attention_weights: np.ndarray
    key_names: Optional[List[str]] = None

    def top_k_attention(self, k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top-k items by attention weight.

        Returns:
            List of (name, weight) tuples, sorted by weight descending

        Example:
            >>> output.top_k_attention(3)
            [('Waterloo CS', 0.35), ('UofT CS', 0.25), ('McGill CS', 0.15)]
        """
        weights = self.attention_weights
        indices = np.argsort(-weights)[:k]
        return [
            (self.key_names[i] if self.key_names else str(i), float(weights[i]))
            for i in indices
        ]

    def visualize(self) -> str:
        """Generate ASCII visualization of attention weights."""
        items = self.top_k_attention(10)
        lines = []
        bar_width = 40
        for name, weight in items:
            filled = int(weight * bar_width)
            bar = "#" * filled + " " * (bar_width - filled)
            lines.append(f"{name}: [{bar}] {weight:.4f}")
        return "\n".join(lines)


class SelfAttentionLayer:
    """
    Self-attention layer for program embeddings.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  SELF-ATTENTION LAYER                                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Allows programs to attend to each other.                               │
    │                                                                          │
    │  For a batch of programs, each program looks at all other programs     │
    │  and computes a weighted average of their representations.              │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Example:
        >>> layer = SelfAttentionLayer(embed_dim=32, n_heads=4)
        >>> layer.build()
        >>> output = layer.forward(program_embeddings)
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize self-attention layer.

        Args:
            embed_dim: Dimension of input embeddings
            n_heads: Number of attention heads
            dropout: Dropout rate

        IMPLEMENTATION:
        ────────────────
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.dropout = dropout

        # Weights initialized in build()
        self.W_q = None
        self.W_k = None
        self.W_v = None
        self.W_o = None
        """
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.dropout = dropout

        # Weights initialized in build()
        self.W_q = None
        self.W_k = None
        self.W_v = None
        self.W_o = None

    def build(self):
        """
        Initialize weight matrices.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  WEIGHT MATRICES                                                     │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  W_q: (embed_dim, embed_dim) - projects to queries                 │
        │  W_k: (embed_dim, embed_dim) - projects to keys                    │
        │  W_v: (embed_dim, embed_dim) - projects to values                  │
        │  W_o: (embed_dim, embed_dim) - projects concatenated heads         │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        IMPLEMENTATION:
        ────────────────
        scale = np.sqrt(2.0 / self.embed_dim)
        self.W_q = np.random.randn(self.embed_dim, self.embed_dim) * scale
        self.W_k = np.random.randn(self.embed_dim, self.embed_dim) * scale
        self.W_v = np.random.randn(self.embed_dim, self.embed_dim) * scale
        self.W_o = np.random.randn(self.embed_dim, self.embed_dim) * scale
        """
        scale = np.sqrt(2.0 / self.embed_dim)
        self.W_q = np.random.randn(self.embed_dim, self.embed_dim) * scale
        self.W_k = np.random.randn(self.embed_dim, self.embed_dim) * scale
        self.W_v = np.random.randn(self.embed_dim, self.embed_dim) * scale
        self.W_o = np.random.randn(self.embed_dim, self.embed_dim) * scale

    def forward(
        self,
        x: np.ndarray,
        return_attention: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Forward pass through self-attention.

        Args:
            x: Input embeddings (batch, seq_len, embed_dim)
            return_attention: If True, also return attention weights

        Returns:
            Tuple of (output, attention_weights or None)

        IMPLEMENTATION:
        ────────────────
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        Q = Q.transpose(0, 2, 1, 3)  # (batch, heads, seq, head_dim)
        # Same for K, V

        # Compute attention
        output, weights = scaled_dot_product_attention(Q, K, V)

        # Reshape back and project
        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(batch_size, seq_len, self.embed_dim)
        output = output @ self.W_o

        if return_attention:
            return output, weights
        return output, None
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        Q = Q.transpose(0, 2, 1, 3)  # (batch, heads, seq, head_dim)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        K = K.transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        V = V.transpose(0, 2, 1, 3)

        # Compute attention
        output, weights = scaled_dot_product_attention(Q, K, V)

        # Reshape back and project
        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(batch_size, seq_len, self.embed_dim)
        output = output @ self.W_o

        if return_attention:
            return output, weights
        return output, None


class CrossAttentionLayer:
    """
    Cross-attention between student application and all programs.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  CROSS-ATTENTION                                                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Query: Student's target program embedding                              │
    │  Keys/Values: All program embeddings                                    │
    │                                                                          │
    │  "For this student's program, which other programs are relevant?"       │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Example:
        >>> layer = CrossAttentionLayer(embed_dim=32, n_heads=4)
        >>> context = layer.forward(target_prog_embed, all_prog_embeds)
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 4
    ):
        """Initialize cross-attention layer."""
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # Xavier initialization
        scale = np.sqrt(2.0 / embed_dim)
        self.W_q = np.random.randn(embed_dim, embed_dim) * scale
        self.W_k = np.random.randn(embed_dim, embed_dim) * scale
        self.W_v = np.random.randn(embed_dim, embed_dim) * scale
        self.W_o = np.random.randn(embed_dim, embed_dim) * scale

    def forward(
        self,
        query: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        key_names: Optional[List[str]] = None
    ) -> AttentionOutput:
        """
        Forward pass through cross-attention.

        Args:
            query: Query embedding (batch, embed_dim)
            keys: Key embeddings (n_keys, embed_dim)
            values: Value embeddings (n_keys, embed_dim)
            key_names: Optional names for keys (for interpretability)

        Returns:
            AttentionOutput with context and weights
        """
        # query shape: (batch, embed_dim)
        # keys shape: (n_keys, embed_dim)
        # values shape: (n_keys, embed_dim)
        d = self.embed_dim

        Q = query @ self.W_q   # (batch, embed_dim)
        K = keys @ self.W_k    # (n_keys, embed_dim)
        V = values @ self.W_v  # (n_keys, embed_dim)

        # Compute attention scores: Q @ K.T / sqrt(d)
        scores = Q @ K.T / np.sqrt(d)  # (batch, n_keys)

        # Softmax to get attention weights
        weights = softmax(scores, axis=-1)  # (batch, n_keys)

        # Context vector: weighted sum of values
        context = weights @ V  # (batch, embed_dim)

        # Return AttentionOutput
        return AttentionOutput(
            context=context,
            attention_weights=weights[0] if query.shape[0] == 1 else weights,
            key_names=key_names,
        )


class AttentionModel:
    """
    Full attention-based model for admission prediction.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  ATTENTION MODEL ARCHITECTURE                                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  1. Look up embeddings for target program                               │
    │  2. Cross-attend to all program embeddings                              │
    │  3. Combine with continuous features                                    │
    │  4. Predict admission probability                                       │
    │                                                                          │
    │  INTERPRETABILITY:                                                       │
    │  ──────────────────                                                      │
    │  Attention weights show which programs influenced the prediction.      │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Example:
        >>> model = AttentionModel(config, n_heads=4)
        >>> model.fit(X, y, uni_ids, prog_ids)
        >>> probs, explanations = model.predict_proba(X_new, ...)
        >>> print(explanations[0].top_k_attention(3))
    """

    def __init__(
        self,
        n_universities: int,
        n_programs: int,
        uni_embed_dim: int = 16,
        prog_embed_dim: int = 32,
        n_heads: int = 4,
        lambda_: float = 0.01,
        learning_rate: float = 0.001,
        n_epochs: int = 100
    ):
        """
        Initialize attention model.

        Args:
            n_universities: Number of unique universities
            n_programs: Number of unique programs
            uni_embed_dim: University embedding dimension
            prog_embed_dim: Program embedding dimension
            n_heads: Number of attention heads
            lambda_: Weight decay
            learning_rate: Adam learning rate
            n_epochs: Training epochs
        """
        self.n_universities = n_universities
        self.n_programs = n_programs
        self.uni_embed_dim = uni_embed_dim
        self.prog_embed_dim = prog_embed_dim
        self.n_heads = n_heads
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self._is_fitted = False
        self.uni_embeddings = None
        self.prog_embeddings = None
        self.cross_attention = None
        self.output_weights = None
        self.output_bias = None
        self._attention_patterns = {}

    @property
    def name(self) -> str:
        """Model name."""
        return "Attention-Based Admission Model"

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        university_ids: np.ndarray,
        program_ids: np.ndarray,
        validation_data: Optional[Tuple] = None
    ) -> 'AttentionModel':
        """
        Train the attention model.

        Args:
            X: Continuous features
            y: Binary labels
            university_ids: University identifiers
            program_ids: Program identifiers
            validation_data: Optional validation set

        Returns:
            self
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # Initialize embeddings (Xavier)
        uni_scale = np.sqrt(2.0 / self.uni_embed_dim)
        prog_scale = np.sqrt(2.0 / self.prog_embed_dim)
        self.uni_embeddings = np.random.randn(
            self.n_universities + 1, self.uni_embed_dim
        ) * uni_scale
        self.prog_embeddings = np.random.randn(
            self.n_programs + 1, self.prog_embed_dim
        ) * prog_scale

        # Create cross-attention layer
        self.cross_attention = CrossAttentionLayer(
            embed_dim=self.prog_embed_dim,
            n_heads=self.n_heads,
        )

        # Input dim = uni_embed_dim + prog_embed_dim + n_features
        input_dim = self.uni_embed_dim + self.prog_embed_dim + n_features
        self.output_weights = np.random.randn(input_dim, 1) * 0.01
        self.output_bias = np.zeros(1)

        # Training loop with simple gradient descent
        for epoch in range(self.n_epochs):
            # Forward pass
            # Clip ids to valid range
            uni_ids_clipped = np.clip(university_ids, 0, self.n_universities)
            prog_ids_clipped = np.clip(program_ids, 0, self.n_programs)

            uni_embed = self.uni_embeddings[uni_ids_clipped]  # (n_samples, uni_embed_dim)
            prog_embed = self.prog_embeddings[prog_ids_clipped]  # (n_samples, prog_embed_dim)

            # Cross-attend program embeddings
            # For each sample, attend over all program embeddings
            attn_out = self.cross_attention.forward(
                query=prog_embed,
                keys=self.prog_embeddings,
                values=self.prog_embeddings,
            )
            context = attn_out.context  # (n_samples, prog_embed_dim)

            # Concatenate: [uni_embed, context, X]
            combined = np.concatenate([uni_embed, context, X], axis=1)  # (n_samples, input_dim)

            # Linear + sigmoid
            logits = combined @ self.output_weights + self.output_bias  # (n_samples, 1)
            probs = sigmoid(logits.ravel())  # (n_samples,)

            # BCE loss
            eps = 1e-15
            probs_clipped = np.clip(probs, eps, 1 - eps)

            # Gradient of BCE w.r.t. logits: (probs - y)
            grad_logits = (probs - y).reshape(-1, 1)  # (n_samples, 1)

            # Gradient for output_weights: combined.T @ grad_logits / n_samples
            grad_w = combined.T @ grad_logits / n_samples + self.lambda_ * self.output_weights
            grad_b = np.mean(grad_logits, axis=0)

            # Update output weights
            self.output_weights -= self.learning_rate * grad_w
            self.output_bias -= self.learning_rate * grad_b

        self._is_fitted = True
        return self

    def predict_proba(
        self,
        X: np.ndarray,
        university_ids: np.ndarray,
        program_ids: np.ndarray,
        return_attention: bool = False
    ) -> Tuple[np.ndarray, Optional[List[AttentionOutput]]]:
        """
        Predict probabilities with optional attention explanations.

        Args:
            X: Continuous features
            university_ids: University identifiers
            program_ids: Program identifiers
            return_attention: If True, return attention outputs

        Returns:
            Tuple of (probabilities, attention_outputs or None)

        Example:
            >>> probs, attn = model.predict_proba(X, uni_ids, prog_ids,
            ...                                    return_attention=True)
            >>> print(f"Probability: {probs[0]:.2%}")
            >>> print("Model focused on:")
            >>> for name, weight in attn[0].top_k_attention(3):
            ...     print(f"  {name}: {weight:.2%}")
        """
        if not self._is_fitted:
            # Initialize embeddings lazily for unfitted model
            uni_scale = np.sqrt(2.0 / self.uni_embed_dim)
            prog_scale = np.sqrt(2.0 / self.prog_embed_dim)
            self.uni_embeddings = np.random.randn(
                self.n_universities + 1, self.uni_embed_dim
            ) * uni_scale
            self.prog_embeddings = np.random.randn(
                self.n_programs + 1, self.prog_embed_dim
            ) * prog_scale
            self.cross_attention = CrossAttentionLayer(
                embed_dim=self.prog_embed_dim,
                n_heads=self.n_heads,
            )
            n_features = X.shape[1]
            input_dim = self.uni_embed_dim + self.prog_embed_dim + n_features
            self.output_weights = np.random.randn(input_dim, 1) * 0.01
            self.output_bias = np.zeros(1)

        n_samples = X.shape[0]

        # Clip ids to valid range
        uni_ids_clipped = np.clip(university_ids, 0, self.n_universities)
        prog_ids_clipped = np.clip(program_ids, 0, self.n_programs)

        # Lookup embeddings
        uni_embed = self.uni_embeddings[uni_ids_clipped]  # (n_samples, uni_embed_dim)
        prog_embed = self.prog_embeddings[prog_ids_clipped]  # (n_samples, prog_embed_dim)

        # Cross-attend program embeddings
        attn_out = self.cross_attention.forward(
            query=prog_embed,
            keys=self.prog_embeddings,
            values=self.prog_embeddings,
        )
        context = attn_out.context  # (n_samples, prog_embed_dim)

        # Concatenate: [uni_embed, context, X]
        combined = np.concatenate([uni_embed, context, X], axis=1)

        # Linear + sigmoid
        logits = combined @ self.output_weights + self.output_bias
        probs = sigmoid(logits.ravel())

        # Build attention outputs if requested
        attn_outputs = None
        if return_attention:
            attn_outputs = []
            weights = attn_out.attention_weights
            # weights could be 1D (if batch==1) or 2D
            if weights.ndim == 1:
                weights = weights.reshape(1, -1)
            for i in range(n_samples):
                w = weights[i] if i < weights.shape[0] else weights[0]
                attn_outputs.append(
                    AttentionOutput(
                        context=context[i],
                        attention_weights=w,
                        key_names=None,
                    )
                )

        return probs, attn_outputs

    def explain_prediction(
        self,
        X: np.ndarray,
        university_id: str,
        program_id: str
    ) -> Dict[str, Any]:
        """
        Generate detailed explanation for a single prediction.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  PREDICTION EXPLANATION                                              │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  Returns:                                                           │
        │  • Probability and confidence                                       │
        │  • Top programs the model attended to                               │
        │  • Contribution breakdown                                           │
        │  • Similar applications in training data                            │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            X: Continuous features for single sample
            university_id: University identifier
            program_id: Program identifier

        Returns:
            Dict with explanation components
        """
        # Make a single prediction with attention
        uni_ids = np.array([hash(university_id) % (self.n_universities + 1)])
        prog_ids = np.array([hash(program_id) % (self.n_programs + 1)])

        probs, attn_outputs = self.predict_proba(
            X, uni_ids, prog_ids, return_attention=True
        )

        explanation = {
            "probability": float(probs[0]),
            "university_id": university_id,
            "program_id": program_id,
            "attention_weights": attn_outputs[0].attention_weights if attn_outputs else None,
            "top_programs": attn_outputs[0].top_k_attention(5) if attn_outputs else [],
        }
        return explanation

    def get_attention_patterns(self) -> Dict[str, np.ndarray]:
        """
        Get learned attention patterns across all heads.

        Returns:
            Dict mapping head_idx → average attention matrix
        """
        return self._attention_patterns

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters for serialization."""
        return {
            "n_universities": self.n_universities,
            "n_programs": self.n_programs,
            "uni_embed_dim": self.uni_embed_dim,
            "prog_embed_dim": self.prog_embed_dim,
            "n_heads": self.n_heads,
            "lambda_": self.lambda_,
            "learning_rate": self.learning_rate,
            "n_epochs": self.n_epochs,
            "uni_embeddings": self.uni_embeddings,
            "prog_embeddings": self.prog_embeddings,
            "output_weights": self.output_weights,
            "output_bias": self.output_bias,
            "is_fitted": self._is_fitted,
        }

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters from serialization."""
        if "n_universities" in params:
            self.n_universities = params["n_universities"]
        if "n_programs" in params:
            self.n_programs = params["n_programs"]
        if "uni_embed_dim" in params:
            self.uni_embed_dim = params["uni_embed_dim"]
        if "prog_embed_dim" in params:
            self.prog_embed_dim = params["prog_embed_dim"]
        if "n_heads" in params:
            self.n_heads = params["n_heads"]
        if "lambda_" in params:
            self.lambda_ = params["lambda_"]
        if "learning_rate" in params:
            self.learning_rate = params["learning_rate"]
        if "n_epochs" in params:
            self.n_epochs = params["n_epochs"]
        if "uni_embeddings" in params and params["uni_embeddings"] is not None:
            self.uni_embeddings = np.array(params["uni_embeddings"])
        if "prog_embeddings" in params and params["prog_embeddings"] is not None:
            self.prog_embeddings = np.array(params["prog_embeddings"])
        if "output_weights" in params and params["output_weights"] is not None:
            self.output_weights = np.array(params["output_weights"])
        if "output_bias" in params and params["output_bias"] is not None:
            self.output_bias = np.array(params["output_bias"])
        if "is_fitted" in params:
            self._is_fitted = params["is_fitted"]


# =============================================================================
#                           TODO LIST
# =============================================================================
"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TODO                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  HIGH PRIORITY (core attention):                                             │
│  ────────────────────────────────                                            │
│  [ ] Implement softmax() with numerical stability                           │
│  [ ] Implement scaled_dot_product_attention()                               │
│  [ ] Implement SelfAttentionLayer.build()                                   │
│  [ ] Implement SelfAttentionLayer.forward()                                 │
│                                                                              │
│  MEDIUM PRIORITY (full model):                                               │
│  ──────────────────────────────                                              │
│  [ ] Implement CrossAttentionLayer.forward()                                │
│  [ ] Implement AttentionModel.fit()                                         │
│  [ ] Implement AttentionModel.predict_proba()                               │
│  [ ] Implement AttentionOutput.top_k_attention()                            │
│                                                                              │
│  LOW PRIORITY (interpretability):                                            │
│  ─────────────────────────────────                                           │
│  [ ] Implement explain_prediction()                                         │
│  [ ] Implement get_attention_patterns()                                     │
│  [ ] Implement AttentionOutput.visualize()                                  │
│  [ ] Add attention visualization utilities                                  │
│                                                                              │
│  TESTING:                                                                    │
│  ────────                                                                    │
│  [ ] Verify softmax sums to 1                                               │
│  [ ] Test attention on synthetic data                                       │
│  [ ] Verify attention weights are interpretable                             │
│  [ ] Compare to embedding-only model                                        │
│                                                                              │
│  ADVANCED (optional):                                                        │
│  ─────────────────────                                                       │
│  [ ] Add layer normalization                                                │
│  [ ] Add residual connections                                               │
│  [ ] Implement multi-layer attention (transformer blocks)                   │
│  [ ] Add positional encoding for time series                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    print("Attention Mechanism Module")
    print("=" * 50)
    print()
    print("Provides interpretable predictions by showing")
    print("which similar programs influenced the prediction.")
    print()
    print("Key output:")
    print("  • Probability: 72%")
    print("  • Model focused on:")
    print("      - Waterloo CS (35%)")
    print("      - UofT CS (25%)")
    print("      - McGill CS (15%)")
