"""
Unit Tests for Attention Model Module
======================================

Tests for src/models/attention.py -- scaled_dot_product_attention, softmax,
AttentionOutput dataclass, SelfAttentionLayer, CrossAttentionLayer, AttentionModel.

All source functions are implemented. Tests verify return types, shapes,
value ranges, and expected mathematical properties.
"""

import pytest
import numpy as np
from dataclasses import fields

from src.models.attention import (
    scaled_dot_product_attention,
    softmax,
    AttentionOutput,
    SelfAttentionLayer,
    CrossAttentionLayer,
    AttentionModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)


# =========================================================================
# TestScaledDotProductAttention
# =========================================================================
class TestScaledDotProductAttention:
    """Tests for the scaled_dot_product_attention free function."""

    def test_basic_call(self):
        Q = RNG.standard_normal((1, 3, 8))
        K = RNG.standard_normal((1, 5, 8))
        V = RNG.standard_normal((1, 5, 8))
        result = scaled_dot_product_attention(Q, K, V)
        output, weights = result
        assert output.shape == (1, 3, 8)
        assert weights.shape == (1, 3, 5)

    def test_call_with_mask(self):
        Q = RNG.standard_normal((1, 3, 8))
        K = RNG.standard_normal((1, 5, 8))
        V = RNG.standard_normal((1, 5, 8))
        mask = np.zeros((1, 3, 5))
        mask[:, :, -1] = -1e9  # mask out last key
        result = scaled_dot_product_attention(Q, K, V, mask=mask)
        output, weights = result
        assert output.shape == (1, 3, 8)
        assert weights.shape == (1, 3, 5)

    def test_attention_weights_sum_to_one(self):
        """Attention weights along the key dimension should sum to 1."""
        Q = RNG.standard_normal((2, 4, 8))
        K = RNG.standard_normal((2, 6, 8))
        V = RNG.standard_normal((2, 6, 8))
        output, weights = scaled_dot_product_attention(Q, K, V)
        # weights shape: (2, 4, 6) -- each row along last axis sums to 1
        row_sums = weights.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-7)

    def test_attention_weights_nonnegative(self):
        """All attention weights should be non-negative (softmax output)."""
        Q = RNG.standard_normal((1, 3, 8))
        K = RNG.standard_normal((1, 5, 8))
        V = RNG.standard_normal((1, 5, 8))
        _, weights = scaled_dot_product_attention(Q, K, V)
        assert np.all(weights >= 0)

    def test_masked_position_near_zero_weight(self):
        """A position masked with -1e9 should get near-zero attention weight."""
        Q = RNG.standard_normal((1, 2, 4))
        K = RNG.standard_normal((1, 3, 4))
        V = RNG.standard_normal((1, 3, 4))
        mask = np.zeros((1, 2, 3))
        mask[:, :, 2] = -1e9  # mask last key
        _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        assert np.all(weights[:, :, 2] < 1e-6)

    def test_batch_dimension(self):
        """Verify correct handling of batch dimension."""
        batch = 4
        Q = RNG.standard_normal((batch, 3, 8))
        K = RNG.standard_normal((batch, 5, 8))
        V = RNG.standard_normal((batch, 5, 16))
        output, weights = scaled_dot_product_attention(Q, K, V)
        assert output.shape == (batch, 3, 16)
        assert weights.shape == (batch, 3, 5)


# =========================================================================
# TestSoftmax
# =========================================================================
class TestSoftmax:
    """Tests for the softmax free function."""

    def test_1d_array(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert result.shape == x.shape
        assert np.isclose(result.sum(), 1.0)

    def test_2d_array(self):
        x = RNG.standard_normal((3, 4))
        result = softmax(x, axis=-1)
        assert result.shape == x.shape
        row_sums = result.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(3), atol=1e-7)

    def test_all_values_positive(self):
        """Softmax output should be strictly positive."""
        x = np.array([-100.0, 0.0, 100.0])
        result = softmax(x)
        assert np.all(result > 0)

    def test_numerical_stability_large_values(self):
        """Softmax should handle large values without overflow."""
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        assert np.isclose(result.sum(), 1.0)
        assert np.all(np.isfinite(result))

    def test_uniform_input_gives_uniform_output(self):
        """Equal inputs should produce equal softmax probabilities."""
        x = np.array([5.0, 5.0, 5.0, 5.0])
        result = softmax(x)
        np.testing.assert_allclose(result, np.ones(4) / 4, atol=1e-7)

    def test_ordering_preserved(self):
        """Larger input should produce larger softmax probability."""
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert result[0] < result[1] < result[2]

    def test_axis_0(self):
        """Softmax along axis=0 should normalize columns."""
        x = RNG.standard_normal((3, 4))
        result = softmax(x, axis=0)
        col_sums = result.sum(axis=0)
        np.testing.assert_allclose(col_sums, np.ones(4), atol=1e-7)


# =========================================================================
# TestAttentionOutput
# =========================================================================
class TestAttentionOutput:
    """Tests for the AttentionOutput dataclass."""

    def test_dataclass_fields(self):
        names = {f.name for f in fields(AttentionOutput)}
        assert "context" in names
        assert "attention_weights" in names
        assert "key_names" in names

    def test_default_key_names_is_none(self):
        ao = AttentionOutput(
            context=np.array([1.0, 2.0]),
            attention_weights=np.array([0.5, 0.5]),
        )
        assert ao.key_names is None

    def test_key_names_stored(self):
        ao = AttentionOutput(
            context=np.array([1.0]),
            attention_weights=np.array([0.3, 0.7]),
            key_names=["A", "B"],
        )
        assert ao.key_names == ["A", "B"]

    def test_top_k_attention_returns_sorted_list(self):
        ao = AttentionOutput(
            context=np.array([1.0]),
            attention_weights=np.array([0.1, 0.9]),
            key_names=["A", "B"],
        )
        result = ao.top_k_attention(k=2)
        assert isinstance(result, list)
        assert len(result) == 2
        # Should be sorted descending by weight
        assert result[0][0] == "B"
        assert result[0][1] == pytest.approx(0.9)
        assert result[1][0] == "A"
        assert result[1][1] == pytest.approx(0.1)

    def test_top_k_attention_limits_k(self):
        ao = AttentionOutput(
            context=np.array([1.0]),
            attention_weights=np.array([0.1, 0.2, 0.3, 0.4]),
            key_names=["A", "B", "C", "D"],
        )
        result = ao.top_k_attention(k=2)
        assert len(result) == 2
        # Top 2 should be D (0.4) and C (0.3)
        assert result[0][0] == "D"
        assert result[1][0] == "C"

    def test_top_k_attention_without_names(self):
        """When key_names is None, indices should be used as strings."""
        ao = AttentionOutput(
            context=np.array([1.0]),
            attention_weights=np.array([0.1, 0.9]),
        )
        result = ao.top_k_attention(k=2)
        assert result[0][0] == "1"  # index 1 has highest weight
        assert result[1][0] == "0"

    def test_visualize_returns_string(self):
        ao = AttentionOutput(
            context=np.array([1.0]),
            attention_weights=np.array([0.5, 0.5]),
            key_names=["X", "Y"],
        )
        result = ao.visualize()
        assert isinstance(result, str)
        assert "X" in result
        assert "Y" in result

    def test_visualize_contains_bars(self):
        ao = AttentionOutput(
            context=np.array([1.0]),
            attention_weights=np.array([0.8, 0.2]),
            key_names=["High", "Low"],
        )
        result = ao.visualize()
        assert "#" in result
        assert "High" in result


# =========================================================================
# TestSelfAttentionLayer
# =========================================================================
class TestSelfAttentionLayer:
    """Tests for SelfAttentionLayer."""

    def test_init(self):
        layer = SelfAttentionLayer(embed_dim=16, n_heads=4)
        assert layer.embed_dim == 16
        assert layer.n_heads == 4
        assert layer.head_dim == 4
        assert layer.dropout == 0.1

    def test_init_custom_dropout(self):
        layer = SelfAttentionLayer(embed_dim=32, n_heads=8, dropout=0.2)
        assert layer.dropout == 0.2
        assert layer.head_dim == 4

    def test_build_initializes_weights(self):
        layer = SelfAttentionLayer(embed_dim=16, n_heads=4)
        layer.build()
        assert layer.W_q is not None
        assert layer.W_k is not None
        assert layer.W_v is not None
        assert layer.W_o is not None
        assert layer.W_q.shape == (16, 16)
        assert layer.W_k.shape == (16, 16)
        assert layer.W_v.shape == (16, 16)
        assert layer.W_o.shape == (16, 16)

    def test_forward_output_shape(self):
        layer = SelfAttentionLayer(embed_dim=16, n_heads=4)
        layer.build()
        x = RNG.standard_normal((2, 6, 16))
        output, weights = layer.forward(x)
        assert output.shape == (2, 6, 16)
        assert weights is None  # return_attention=False by default

    def test_forward_return_attention(self):
        layer = SelfAttentionLayer(embed_dim=16, n_heads=4)
        layer.build()
        x = RNG.standard_normal((2, 6, 16))
        output, weights = layer.forward(x, return_attention=True)
        assert output.shape == (2, 6, 16)
        assert weights is not None
        # weights shape: (batch, n_heads, seq_len, seq_len)
        assert weights.shape == (2, 4, 6, 6)

    def test_forward_attention_weights_sum_to_one(self):
        layer = SelfAttentionLayer(embed_dim=16, n_heads=4)
        layer.build()
        x = RNG.standard_normal((1, 4, 16))
        _, weights = layer.forward(x, return_attention=True)
        # Sum along last axis (keys) should be 1 for each query
        row_sums = weights.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-6)

    def test_weights_none_before_build(self):
        layer = SelfAttentionLayer(embed_dim=8, n_heads=2)
        assert layer.W_q is None
        assert layer.W_k is None


# =========================================================================
# TestCrossAttentionLayer
# =========================================================================
class TestCrossAttentionLayer:
    """Tests for CrossAttentionLayer."""

    def test_init(self):
        layer = CrossAttentionLayer(embed_dim=16)
        assert layer.embed_dim == 16
        assert layer.n_heads == 4  # default
        assert layer.head_dim == 4
        # Weights are initialized in __init__ for CrossAttentionLayer
        assert layer.W_q is not None

    def test_init_custom_heads(self):
        layer = CrossAttentionLayer(embed_dim=32, n_heads=8)
        assert layer.n_heads == 8
        assert layer.head_dim == 4

    def test_forward_returns_attention_output(self):
        layer = CrossAttentionLayer(embed_dim=16)
        query = RNG.standard_normal((2, 16))
        keys = RNG.standard_normal((10, 16))
        values = RNG.standard_normal((10, 16))
        result = layer.forward(query, keys, values, key_names=None)
        assert isinstance(result, AttentionOutput)
        assert result.context.shape == (2, 16)
        assert result.key_names is None

    def test_forward_attention_weights_shape(self):
        layer = CrossAttentionLayer(embed_dim=16)
        query = RNG.standard_normal((3, 16))
        keys = RNG.standard_normal((8, 16))
        values = RNG.standard_normal((8, 16))
        result = layer.forward(query, keys, values)
        # For batch > 1, weights shape is (batch, n_keys)
        assert result.attention_weights.shape == (3, 8)

    def test_forward_single_query_weights_1d(self):
        layer = CrossAttentionLayer(embed_dim=16)
        query = RNG.standard_normal((1, 16))
        keys = RNG.standard_normal((5, 16))
        values = RNG.standard_normal((5, 16))
        result = layer.forward(query, keys, values)
        # For single query, attention_weights is squeezed to 1D
        assert result.attention_weights.ndim == 1
        assert result.attention_weights.shape[0] == 5

    def test_forward_with_key_names(self):
        layer = CrossAttentionLayer(embed_dim=16)
        query = RNG.standard_normal((2, 16))
        keys = RNG.standard_normal((5, 16))
        values = RNG.standard_normal((5, 16))
        names = [f"prog_{i}" for i in range(5)]
        result = layer.forward(query, keys, values, key_names=names)
        assert result.key_names == names

    def test_forward_attention_weights_sum_to_one(self):
        layer = CrossAttentionLayer(embed_dim=16)
        query = RNG.standard_normal((3, 16))
        keys = RNG.standard_normal((6, 16))
        values = RNG.standard_normal((6, 16))
        result = layer.forward(query, keys, values)
        row_sums = result.attention_weights.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-6)


# =========================================================================
# TestAttentionModel
# =========================================================================
class TestAttentionModel:
    """Tests for the full AttentionModel class."""

    # -- properties --------------------------------------------------------

    def test_name_property(self):
        model = AttentionModel(n_universities=10, n_programs=20)
        result = model.name
        assert isinstance(result, str)
        assert "Attention" in result

    def test_is_fitted_property(self):
        model = AttentionModel(n_universities=10, n_programs=20)
        assert model.is_fitted is False

    # -- fit ---------------------------------------------------------------

    def test_fit_returns_self(self):
        np.random.seed(42)
        model = AttentionModel(
            n_universities=10, n_programs=20,
            n_epochs=3, learning_rate=0.01
        )
        X = np.random.default_rng(42).standard_normal((20, 5))
        y = np.random.default_rng(42).choice([0, 1], size=20).astype(float)
        uni_ids = np.random.default_rng(42).integers(0, 10, size=20)
        prog_ids = np.random.default_rng(42).integers(0, 20, size=20)
        result = model.fit(X, y, uni_ids, prog_ids)
        assert result is model

    def test_fit_sets_is_fitted(self):
        np.random.seed(42)
        model = AttentionModel(
            n_universities=5, n_programs=10,
            n_epochs=2, learning_rate=0.01
        )
        X = np.random.default_rng(42).standard_normal((15, 3))
        y = np.random.default_rng(42).choice([0, 1], size=15).astype(float)
        uni_ids = np.random.default_rng(42).integers(0, 5, size=15)
        prog_ids = np.random.default_rng(42).integers(0, 10, size=15)
        model.fit(X, y, uni_ids, prog_ids)
        assert model.is_fitted is True

    def test_fit_initializes_embeddings(self):
        np.random.seed(42)
        model = AttentionModel(
            n_universities=5, n_programs=10,
            uni_embed_dim=8, prog_embed_dim=16,
            n_epochs=2
        )
        X = np.random.default_rng(42).standard_normal((15, 3))
        y = np.random.default_rng(42).choice([0, 1], size=15).astype(float)
        uni_ids = np.random.default_rng(42).integers(0, 5, size=15)
        prog_ids = np.random.default_rng(42).integers(0, 10, size=15)
        model.fit(X, y, uni_ids, prog_ids)
        assert model.uni_embeddings is not None
        assert model.uni_embeddings.shape == (6, 8)  # n_universities + 1
        assert model.prog_embeddings is not None
        assert model.prog_embeddings.shape == (11, 16)  # n_programs + 1

    # -- predict_proba -----------------------------------------------------

    def test_predict_proba_shape(self):
        np.random.seed(42)
        model = AttentionModel(n_universities=10, n_programs=20)
        X = RNG.standard_normal((5, 5))
        uni_ids = RNG.integers(0, 10, size=5)
        prog_ids = RNG.integers(0, 20, size=5)
        probs, attn_out = model.predict_proba(X, uni_ids, prog_ids)
        assert probs.shape == (5,)
        assert attn_out is None

    def test_predict_proba_values_in_zero_one(self):
        np.random.seed(42)
        model = AttentionModel(n_universities=10, n_programs=20)
        X = RNG.standard_normal((10, 5))
        uni_ids = RNG.integers(0, 10, size=10)
        prog_ids = RNG.integers(0, 20, size=10)
        probs, _ = model.predict_proba(X, uni_ids, prog_ids)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_predict_proba_return_attention(self):
        np.random.seed(42)
        model = AttentionModel(n_universities=10, n_programs=20)
        X = RNG.standard_normal((5, 5))
        uni_ids = RNG.integers(0, 10, size=5)
        prog_ids = RNG.integers(0, 20, size=5)
        probs, attn_out = model.predict_proba(X, uni_ids, prog_ids, return_attention=True)
        assert attn_out is not None
        assert len(attn_out) == 5
        assert isinstance(attn_out[0], AttentionOutput)
        assert attn_out[0].context.shape == (model.prog_embed_dim,)

    # -- explain_prediction ------------------------------------------------

    def test_explain_prediction_returns_dict(self):
        np.random.seed(42)
        model = AttentionModel(n_universities=10, n_programs=20)
        X = RNG.standard_normal((1, 5))
        result = model.explain_prediction(X, university_id="uni_0", program_id="prog_0")
        assert isinstance(result, dict)
        assert "probability" in result
        assert "university_id" in result
        assert "program_id" in result
        assert "attention_weights" in result
        assert "top_programs" in result

    def test_explain_prediction_probability_in_range(self):
        np.random.seed(42)
        model = AttentionModel(n_universities=10, n_programs=20)
        X = RNG.standard_normal((1, 5))
        result = model.explain_prediction(X, university_id="uni_0", program_id="prog_0")
        assert 0.0 <= result["probability"] <= 1.0

    # -- get_attention_patterns --------------------------------------------

    def test_get_attention_patterns_returns_dict(self):
        model = AttentionModel(n_universities=10, n_programs=20)
        result = model.get_attention_patterns()
        assert isinstance(result, dict)
        # Initially empty since model is not fitted with pattern tracking
        assert len(result) == 0

    # -- get_params / set_params -------------------------------------------

    def test_get_params_returns_dict(self):
        model = AttentionModel(n_universities=10, n_programs=20)
        result = model.get_params()
        assert isinstance(result, dict)
        assert result["n_universities"] == 10
        assert result["n_programs"] == 20
        assert result["is_fitted"] is False

    def test_set_params_updates_model(self):
        model = AttentionModel(n_universities=10, n_programs=20)
        model.set_params({
            "n_universities": 15,
            "n_programs": 30,
            "n_heads": 8,
            "lambda_": 0.05,
        })
        assert model.n_universities == 15
        assert model.n_programs == 30
        assert model.n_heads == 8
        assert model.lambda_ == 0.05

    def test_get_set_params_roundtrip(self):
        """Verify get_params -> set_params roundtrip preserves state."""
        np.random.seed(42)
        model1 = AttentionModel(
            n_universities=5, n_programs=10,
            n_epochs=2, learning_rate=0.01
        )
        X = np.random.default_rng(42).standard_normal((15, 3))
        y = np.random.default_rng(42).choice([0, 1], size=15).astype(float)
        uni_ids = np.random.default_rng(42).integers(0, 5, size=15)
        prog_ids = np.random.default_rng(42).integers(0, 10, size=15)
        model1.fit(X, y, uni_ids, prog_ids)

        params = model1.get_params()
        model2 = AttentionModel(n_universities=1, n_programs=1)
        model2.set_params(params)

        assert model2.n_universities == 5
        assert model2.n_programs == 10
        assert model2._is_fitted is True
        np.testing.assert_array_equal(model2.uni_embeddings, model1.uni_embeddings)
        np.testing.assert_array_equal(model2.prog_embeddings, model1.prog_embeddings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
