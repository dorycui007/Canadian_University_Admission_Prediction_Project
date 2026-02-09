"""
Unit Tests for Embedding Model Module
======================================

Tests for src/models/embeddings.py -- EmbeddingConfig dataclass,
compute_embedding_dim, CategoryEncoder, and EmbeddingModel.

All source functions are implemented. Tests verify return types, shapes,
value ranges, and expected mathematical properties.
"""

import pytest
import numpy as np
from dataclasses import fields

from src.models.embeddings import (
    EmbeddingConfig,
    compute_embedding_dim,
    CategoryEncoder,
    EmbeddingModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)


# =========================================================================
# TestEmbeddingConfig
# =========================================================================
class TestEmbeddingConfig:
    """Tests for the EmbeddingConfig dataclass."""

    def test_fields_exist(self):
        names = {f.name for f in fields(EmbeddingConfig)}
        assert "n_universities" in names
        assert "n_programs" in names
        assert "uni_embed_dim" in names
        assert "prog_embed_dim" in names

    def test_default_uni_embed_dim(self):
        cfg = EmbeddingConfig(n_universities=10, n_programs=20)
        assert cfg.uni_embed_dim == 16

    def test_default_prog_embed_dim(self):
        cfg = EmbeddingConfig(n_universities=10, n_programs=20)
        assert cfg.prog_embed_dim == 32

    def test_explicit_dims(self):
        cfg = EmbeddingConfig(n_universities=5, n_programs=10, uni_embed_dim=8, prog_embed_dim=16)
        assert cfg.uni_embed_dim == 8
        assert cfg.prog_embed_dim == 16

    def test_from_data_returns_config(self):
        uni_ids = np.array([0, 1, 2, 0, 1])
        prog_ids = np.array([0, 1, 2, 3, 4])
        result = EmbeddingConfig.from_data(uni_ids, prog_ids)
        assert isinstance(result, EmbeddingConfig)
        assert result.n_universities == 3
        assert result.n_programs == 5

    def test_from_data_uses_formula_dims(self):
        """Dimensions from from_data should match compute_embedding_dim formula."""
        uni_ids = np.array([0, 1, 2, 0, 1])
        prog_ids = np.array([0, 1, 2, 3, 4])
        result = EmbeddingConfig.from_data(uni_ids, prog_ids)
        expected_uni_dim = compute_embedding_dim(3)
        expected_prog_dim = compute_embedding_dim(5)
        assert result.uni_embed_dim == expected_uni_dim
        assert result.prog_embed_dim == expected_prog_dim

    def test_total_embed_dim_property(self):
        cfg = EmbeddingConfig(n_universities=10, n_programs=20)
        result = cfg.total_embed_dim
        assert result == 16 + 32

    def test_total_embed_dim_with_custom(self):
        cfg = EmbeddingConfig(n_universities=10, n_programs=20, uni_embed_dim=8, prog_embed_dim=24)
        assert cfg.total_embed_dim == 32


# =========================================================================
# TestComputeEmbeddingDim
# =========================================================================
class TestComputeEmbeddingDim:
    """Tests for the compute_embedding_dim free function."""

    def test_with_100_categories(self):
        result = compute_embedding_dim(100)
        # fast.ai formula: min(600, round(1.6 * 100**0.56))
        expected = min(600, round(1.6 * 100 ** 0.56))
        assert isinstance(result, (int, np.integer))
        assert result == expected

    def test_small_n(self):
        result = compute_embedding_dim(10)
        assert isinstance(result, (int, np.integer))
        assert result > 0
        expected = min(600, round(1.6 * 10 ** 0.56))
        assert result == expected

    def test_large_n_capped_at_600(self):
        result = compute_embedding_dim(1_000_000)
        assert result <= 600
        assert result == 600

    def test_monotonically_increasing(self):
        """Larger n should yield same or larger embedding dim."""
        dims = [compute_embedding_dim(n) for n in [5, 10, 50, 100, 500]]
        for i in range(len(dims) - 1):
            assert dims[i] <= dims[i + 1]

    def test_n_equals_1(self):
        """Single category should still return a positive dimension."""
        result = compute_embedding_dim(1)
        assert result >= 1


# =========================================================================
# TestCategoryEncoder
# =========================================================================
class TestCategoryEncoder:
    """Tests for the CategoryEncoder class."""

    def test_init(self):
        enc = CategoryEncoder()
        assert enc is not None
        assert enc.unknown_token == "<UNK>"

    def test_init_custom_token(self):
        enc = CategoryEncoder(unknown_token="<MISSING>")
        assert enc.unknown_token == "<MISSING>"

    def test_fit_returns_self(self):
        enc = CategoryEncoder()
        cats = np.array(["UofT", "Waterloo", "McMaster"])
        result = enc.fit(cats)
        assert result is enc

    def test_fit_sets_attributes(self):
        enc = CategoryEncoder()
        cats = np.array(["UofT", "Waterloo", "McMaster"])
        enc.fit(cats)
        assert enc.n_categories == 4  # 3 categories + 1 unknown
        assert enc.unknown_idx == 3
        assert enc.category_to_idx is not None
        assert len(enc.category_to_idx) == 3

    def test_transform_known_categories(self):
        enc = CategoryEncoder()
        cats = np.array(["UofT", "Waterloo", "McMaster"])
        enc.fit(cats)
        result = enc.transform(np.array(["UofT", "McMaster"]))
        assert isinstance(result, np.ndarray)
        assert len(result) == 2
        # Each index should be < n_categories
        assert np.all(result < enc.n_categories)

    def test_transform_preserves_identity(self):
        """Transforming the same value twice yields the same index."""
        enc = CategoryEncoder()
        cats = np.array(["A", "B", "C"])
        enc.fit(cats)
        r1 = enc.transform(np.array(["B"]))
        r2 = enc.transform(np.array(["B"]))
        assert r1[0] == r2[0]

    def test_transform_different_categories_different_indices(self):
        enc = CategoryEncoder()
        cats = np.array(["A", "B", "C"])
        enc.fit(cats)
        result = enc.transform(np.array(["A", "B", "C"]))
        assert len(set(result)) == 3  # all different indices

    def test_inverse_transform(self):
        enc = CategoryEncoder()
        cats = np.array(["A", "B", "C"])
        enc.fit(cats)
        indices = enc.transform(cats)
        recovered = enc.inverse_transform(indices)
        assert isinstance(recovered, np.ndarray)
        # Sorted unique of cats, then inverse transform should match
        assert set(recovered) == {"A", "B", "C"}

    def test_transform_unknown_category_gets_unknown_idx(self):
        enc = CategoryEncoder()
        cats = np.array(["UofT", "Waterloo"])
        enc.fit(cats)
        result = enc.transform(np.array(["Unknown"]))
        assert isinstance(result, np.ndarray)
        assert result[0] == enc.unknown_idx

    def test_inverse_transform_unknown_idx(self):
        enc = CategoryEncoder()
        cats = np.array(["A", "B"])
        enc.fit(cats)
        result = enc.inverse_transform(np.array([enc.unknown_idx]))
        assert result[0] == "<UNK>"


# =========================================================================
# TestEmbeddingModel
# =========================================================================
class TestEmbeddingModel:
    """Tests for the full EmbeddingModel class."""

    def _make_config(self):
        return EmbeddingConfig(n_universities=10, n_programs=20)

    def _fit_model(self, hidden_dims=None, n_epochs=5):
        """Helper to create and fit a model on small synthetic data."""
        config = EmbeddingConfig(n_universities=5, n_programs=10, uni_embed_dim=4, prog_embed_dim=8)
        model = EmbeddingModel(
            config=config,
            hidden_dims=hidden_dims or [],
            n_epochs=n_epochs,
            learning_rate=0.01,
            batch_size=8,
        )
        rng = np.random.default_rng(42)
        n = 30
        X = rng.standard_normal((n, 3))
        y = rng.choice([0, 1], size=n).astype(float)
        uni_ids = np.array([f"uni_{i % 5}" for i in range(n)])
        prog_ids = np.array([f"prog_{i % 10}" for i in range(n)])
        model.fit(X, y, uni_ids, prog_ids)
        return model, X, y, uni_ids, prog_ids

    # -- properties --------------------------------------------------------

    def test_name_property(self):
        model = EmbeddingModel(config=self._make_config())
        result = model.name
        assert isinstance(result, str)
        assert "Embedding" in result

    def test_is_fitted_property(self):
        model = EmbeddingModel(config=self._make_config())
        assert model.is_fitted is False

    def test_n_params_property(self):
        model = EmbeddingModel(config=self._make_config())
        result = model.n_params
        assert isinstance(result, (int, np.integer))
        assert result > 0

    # -- _build_model ------------------------------------------------------

    def test_build_model_creates_embeddings(self):
        model = EmbeddingModel(config=self._make_config())
        model._build_model(n_continuous=5)
        assert model._uni_embeddings is not None
        assert model._prog_embeddings is not None
        # n_universities + 1 for unknown token
        assert model._uni_embeddings.shape == (11, 16)
        assert model._prog_embeddings.shape == (21, 32)

    def test_build_model_creates_weights(self):
        model = EmbeddingModel(config=self._make_config(), hidden_dims=[32])
        model._build_model(n_continuous=5)
        assert len(model._weights) == 2  # 1 hidden + 1 output
        assert len(model._biases) == 2
        # Input to hidden: (16+32+5) x 32
        assert model._weights[0].shape == (53, 32)
        # Hidden to output: 32 x 1
        assert model._weights[1].shape == (32, 1)

    def test_build_model_no_hidden(self):
        model = EmbeddingModel(config=self._make_config(), hidden_dims=[])
        model._build_model(n_continuous=3)
        assert len(model._weights) == 1  # just output layer
        # Input to output: (16+32+3) x 1
        assert model._weights[0].shape == (51, 1)

    # -- fit ---------------------------------------------------------------

    def test_fit_returns_self(self):
        model, _, _, _, _ = self._fit_model()
        assert model.is_fitted is True

    def test_fit_sets_encoders(self):
        model, _, _, _, _ = self._fit_model()
        assert model.uni_encoder.n_categories is not None
        assert model.prog_encoder.n_categories is not None

    # -- predict_proba -----------------------------------------------------

    def test_predict_proba_unfitted_returns_none(self):
        model = EmbeddingModel(config=self._make_config())
        X = RNG.standard_normal((5, 5))
        uni_ids = np.array(["uni_0"] * 5)
        prog_ids = np.array(["prog_0"] * 5)
        result = model.predict_proba(X, uni_ids, prog_ids)
        assert result is None

    def test_predict_proba_after_fit(self):
        model, X, y, uni_ids, prog_ids = self._fit_model()
        probs = model.predict_proba(X[:5], uni_ids[:5], prog_ids[:5])
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (5,)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_predict_proba_unknown_category(self):
        """Model should handle unseen categories via unknown token."""
        model, X, _, _, _ = self._fit_model()
        uni_ids = np.array(["never_seen_uni"] * 3)
        prog_ids = np.array(["never_seen_prog"] * 3)
        probs = model.predict_proba(X[:3], uni_ids, prog_ids)
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (3,)
        assert np.all(np.isfinite(probs))

    # -- embedding getters -------------------------------------------------

    def test_get_university_embeddings_unfitted_returns_none(self):
        model = EmbeddingModel(config=self._make_config())
        result = model.get_university_embeddings()
        assert result is None

    def test_get_university_embeddings_after_fit(self):
        model, _, _, _, _ = self._fit_model()
        result = model.get_university_embeddings()
        assert isinstance(result, dict)
        assert len(result) == 5  # 5 unique universities
        for name, vec in result.items():
            assert isinstance(name, str)
            assert vec.shape == (4,)  # uni_embed_dim=4

    def test_get_program_embeddings_unfitted_returns_none(self):
        model = EmbeddingModel(config=self._make_config())
        result = model.get_program_embeddings()
        assert result is None

    def test_get_program_embeddings_after_fit(self):
        model, _, _, _, _ = self._fit_model()
        result = model.get_program_embeddings()
        assert isinstance(result, dict)
        assert len(result) == 10  # 10 unique programs
        for name, vec in result.items():
            assert isinstance(name, str)
            assert vec.shape == (8,)  # prog_embed_dim=8

    # -- similarity search -------------------------------------------------

    def test_find_similar_programs_unfitted_returns_empty(self):
        model = EmbeddingModel(config=self._make_config())
        result = model.find_similar_programs("UofT CS", top_k=3)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_find_similar_programs_after_fit(self):
        model, _, _, _, _ = self._fit_model()
        result = model.find_similar_programs("prog_0", top_k=3)
        assert isinstance(result, list)
        assert len(result) <= 3
        # Each entry is (name, similarity_score)
        for name, score in result:
            assert isinstance(name, str)
            assert name != "prog_0"  # Should not include itself
            assert -1.0 <= score <= 1.0  # cosine similarity range

    def test_find_similar_programs_unknown_returns_empty(self):
        model, _, _, _, _ = self._fit_model()
        result = model.find_similar_programs("nonexistent_program", top_k=3)
        assert result == []

    def test_find_similar_universities_unfitted_returns_empty(self):
        model = EmbeddingModel(config=self._make_config())
        result = model.find_similar_universities("UofT", top_k=3)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_find_similar_universities_after_fit(self):
        model, _, _, _, _ = self._fit_model()
        result = model.find_similar_universities("uni_0", top_k=3)
        assert isinstance(result, list)
        assert len(result) <= 3
        for name, score in result:
            assert isinstance(name, str)
            assert name != "uni_0"

    # -- export ------------------------------------------------------------

    def test_export_embeddings_for_weaviate_unfitted_returns_none(self):
        model = EmbeddingModel(config=self._make_config())
        result = model.export_embeddings_for_weaviate()
        assert result is None

    def test_export_embeddings_for_weaviate_after_fit(self):
        model, _, _, _, _ = self._fit_model()
        result = model.export_embeddings_for_weaviate()
        assert isinstance(result, dict)
        assert "universities" in result
        assert "programs" in result
        assert len(result["universities"]) == 5
        assert len(result["programs"]) == 10
        # Each entry should have id, name, embedding
        uni_entry = result["universities"][0]
        assert "id" in uni_entry
        assert "name" in uni_entry
        assert "embedding" in uni_entry
        assert isinstance(uni_entry["embedding"], list)

    # -- get_params / set_params -------------------------------------------

    def test_get_params_returns_dict(self):
        model = EmbeddingModel(config=self._make_config())
        result = model.get_params()
        assert isinstance(result, dict)
        assert "config" in result
        assert result["config"]["n_universities"] == 10
        assert result["config"]["n_programs"] == 20
        assert result["is_fitted"] is False

    def test_set_params_updates_model(self):
        model = EmbeddingModel(config=self._make_config())
        model.set_params({
            "config": {"n_universities": 15, "n_programs": 30, "uni_embed_dim": 8, "prog_embed_dim": 16},
            "lambda_": 0.05,
            "n_epochs": 50,
        })
        assert model.config.n_universities == 15
        assert model.config.n_programs == 30
        assert model.lambda_ == 0.05
        assert model.n_epochs == 50

    def test_get_set_params_roundtrip(self):
        """Verify get_params -> set_params roundtrip preserves state."""
        model1, _, _, _, _ = self._fit_model()
        params = model1.get_params()

        config2 = EmbeddingConfig(n_universities=1, n_programs=1)
        model2 = EmbeddingModel(config=config2)
        model2.set_params(params)

        assert model2.config.n_universities == model1.config.n_universities
        assert model2.config.n_programs == model1.config.n_programs
        assert model2._is_fitted is True
        np.testing.assert_array_equal(model2._uni_embeddings, model1._uni_embeddings)
        np.testing.assert_array_equal(model2._prog_embeddings, model1._prog_embeddings)

    def test_fit_with_hidden_dims(self):
        """Model should train successfully with hidden layers."""
        model, X, y, uni_ids, prog_ids = self._fit_model(hidden_dims=[16, 8])
        assert model.is_fitted is True
        probs = model.predict_proba(X[:5], uni_ids[:5], prog_ids[:5])
        assert probs.shape == (5,)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
