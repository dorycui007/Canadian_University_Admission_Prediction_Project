"""
Tests for src.db.weaviate_client module.
=============================================

Covers Weaviate configuration dataclasses, search/batch result dataclasses,
WeaviateClient operations, EmbeddingVectorizer, schema helpers, and
filter builders.

All source functions are implemented, so tests use real assertions.
"""

import pytest
import numpy as np

from src.db.weaviate_client import (
    WeaviateConfig,
    SchemaClass,
    SearchResult,
    BatchResult,
    WeaviateClient,
    EmbeddingVectorizer,
    create_program_schema,
    create_student_schema,
    build_filter,
    build_range_filter,
    _cosine_similarity,
    _matches_filters,
)


# =========================================================================
# TestWeaviateConfig
# =========================================================================

class TestWeaviateConfig:
    """Tests for the WeaviateConfig dataclass defaults."""

    def test_default_host(self):
        cfg = WeaviateConfig()
        assert cfg.host == "localhost"

    def test_default_port(self):
        cfg = WeaviateConfig()
        assert cfg.port == 8080

    def test_default_grpc_port(self):
        cfg = WeaviateConfig()
        assert cfg.grpc_port == 50051

    def test_default_api_key_none(self):
        cfg = WeaviateConfig()
        assert cfg.api_key is None

    def test_default_use_embedded(self):
        cfg = WeaviateConfig()
        assert cfg.use_embedded is False

    def test_default_timeout(self):
        cfg = WeaviateConfig()
        assert cfg.timeout == 60

    def test_custom_values(self):
        cfg = WeaviateConfig(
            host="weaviate.example.com", port=9090,
            grpc_port=50052, api_key="key123",
            use_embedded=True, timeout=120
        )
        assert cfg.host == "weaviate.example.com"
        assert cfg.port == 9090
        assert cfg.grpc_port == 50052
        assert cfg.api_key == "key123"
        assert cfg.use_embedded is True
        assert cfg.timeout == 120


# =========================================================================
# TestSchemaClass
# =========================================================================

class TestSchemaClass:
    """Tests for the SchemaClass dataclass."""

    def test_required_fields(self):
        sc = SchemaClass(
            name="Program",
            properties=[{"name": "title", "dataType": ["text"]}]
        )
        assert sc.name == "Program"
        assert len(sc.properties) == 1

    def test_default_vector_dimension(self):
        sc = SchemaClass(name="Test", properties=[])
        assert sc.vector_dimension == 128

    def test_default_distance_metric(self):
        sc = SchemaClass(name="Test", properties=[])
        assert sc.distance_metric == "cosine"

    def test_default_vectorizer(self):
        sc = SchemaClass(name="Test", properties=[])
        assert sc.vectorizer == "none"

    def test_default_description_none(self):
        sc = SchemaClass(name="Test", properties=[])
        assert sc.description is None

    def test_custom_values(self):
        sc = SchemaClass(
            name="StudentProfile",
            properties=[{"name": "gpa", "dataType": ["number"]}],
            vector_dimension=256, distance_metric="l2",
            vectorizer="text2vec", description="Student profiles"
        )
        assert sc.name == "StudentProfile"
        assert sc.vector_dimension == 256
        assert sc.distance_metric == "l2"
        assert sc.vectorizer == "text2vec"
        assert sc.description == "Student profiles"


# =========================================================================
# TestSearchResult
# =========================================================================

class TestSearchResult:
    """Tests for the SearchResult dataclass."""

    def test_required_fields(self):
        sr = SearchResult(
            id="uuid-1", properties={"name": "CS"},
            distance=0.15, certainty=0.92
        )
        assert sr.id == "uuid-1"
        assert sr.properties == {"name": "CS"}
        assert sr.distance == 0.15
        assert sr.certainty == 0.92

    def test_default_score_none(self):
        sr = SearchResult(
            id="uuid-2", properties={}, distance=0.1, certainty=0.95
        )
        assert sr.score is None

    def test_default_vector_none(self):
        sr = SearchResult(
            id="uuid-3", properties={}, distance=0.2, certainty=0.9
        )
        assert sr.vector is None

    def test_custom_score(self):
        sr = SearchResult(
            id="uuid-4", properties={}, distance=0.05,
            certainty=0.97, score=0.88
        )
        assert sr.score == 0.88

    def test_custom_vector(self):
        vec = np.array([0.1, 0.2, 0.3])
        sr = SearchResult(
            id="uuid-5", properties={}, distance=0.1,
            certainty=0.95, vector=vec
        )
        assert np.array_equal(sr.vector, vec)


# =========================================================================
# TestBatchResult
# =========================================================================

class TestBatchResult:
    """Tests for the BatchResult dataclass."""

    def test_required_fields(self):
        br = BatchResult(successful=10, failed=2)
        assert br.successful == 10
        assert br.failed == 2

    def test_default_errors_empty(self):
        br = BatchResult(successful=5, failed=0)
        assert br.errors == []

    def test_custom_errors(self):
        br = BatchResult(
            successful=8, failed=2,
            errors=[{"id": "x", "msg": "timeout"}]
        )
        assert len(br.errors) == 1
        assert br.errors[0]["msg"] == "timeout"


# =========================================================================
# TestHelperFunctions
# =========================================================================

class TestHelperFunctions:
    """Tests for _cosine_similarity and _matches_filters."""

    def test_cosine_similarity_identical_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert abs(_cosine_similarity(a, b) - 1.0) < 1e-9

    def test_cosine_similarity_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_similarity(a, b)) < 1e-9

    def test_cosine_similarity_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-9

    def test_cosine_similarity_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        assert _cosine_similarity(a, b) == 0.0

    def test_matches_filters_none(self):
        obj = {"properties": {"name": "CS"}}
        assert _matches_filters(obj, None) is True

    def test_matches_filters_equal(self):
        obj = {"properties": {"province": "Ontario"}}
        f = {"path": ["province"], "operator": "Equal", "valueText": "Ontario"}
        assert _matches_filters(obj, f) is True

    def test_matches_filters_equal_no_match(self):
        obj = {"properties": {"province": "BC"}}
        f = {"path": ["province"], "operator": "Equal", "valueText": "Ontario"}
        assert _matches_filters(obj, f) is False

    def test_matches_filters_greater_than_equal(self):
        obj = {"properties": {"gpa": 85.0}}
        f = {"path": ["gpa"], "operator": "GreaterThanEqual", "valueNumber": 80.0}
        assert _matches_filters(obj, f) is True

    def test_matches_filters_less_than_equal(self):
        obj = {"properties": {"gpa": 85.0}}
        f = {"path": ["gpa"], "operator": "LessThanEqual", "valueNumber": 90.0}
        assert _matches_filters(obj, f) is True

    def test_matches_filters_and_compound(self):
        obj = {"properties": {"province": "Ontario", "gpa": 90.0}}
        f = {
            "operator": "And",
            "operands": [
                {"path": ["province"], "operator": "Equal", "valueText": "Ontario"},
                {"path": ["gpa"], "operator": "GreaterThanEqual", "valueNumber": 85.0},
            ]
        }
        assert _matches_filters(obj, f) is True

    def test_matches_filters_or_compound(self):
        obj = {"properties": {"province": "BC"}}
        f = {
            "operator": "Or",
            "operands": [
                {"path": ["province"], "operator": "Equal", "valueText": "Ontario"},
                {"path": ["province"], "operator": "Equal", "valueText": "BC"},
            ]
        }
        assert _matches_filters(obj, f) is True

    def test_matches_filters_greater_than(self):
        obj = {"properties": {"gpa": 85.0}}
        f = {"path": ["gpa"], "operator": "GreaterThan", "valueNumber": 80.0}
        assert _matches_filters(obj, f) is True
        f2 = {"path": ["gpa"], "operator": "GreaterThan", "valueNumber": 85.0}
        assert _matches_filters(obj, f2) is False

    def test_matches_filters_less_than(self):
        obj = {"properties": {"gpa": 85.0}}
        f = {"path": ["gpa"], "operator": "LessThan", "valueNumber": 90.0}
        assert _matches_filters(obj, f) is True
        f2 = {"path": ["gpa"], "operator": "LessThan", "valueNumber": 85.0}
        assert _matches_filters(obj, f2) is False


# =========================================================================
# TestWeaviateClient
# =========================================================================

class TestWeaviateClient:
    """Tests for the WeaviateClient class."""

    def _make(self):
        return WeaviateClient(WeaviateConfig())

    def _make_with_programs(self):
        """Create client with Program schema and sample objects."""
        client = self._make()
        client.connect()
        schema = create_program_schema(vector_dim=3)
        client.create_schema(schema)
        # Add objects with distinct vectors
        client.add_object("Program", {
            "name": "Computer Science", "university": "UofT",
            "province": "Ontario", "admission_rate": 0.1
        }, np.array([1.0, 0.0, 0.0]))
        client.add_object("Program", {
            "name": "Data Science", "university": "UofT",
            "province": "Ontario", "admission_rate": 0.2
        }, np.array([0.9, 0.1, 0.0]))
        client.add_object("Program", {
            "name": "English Literature", "university": "McGill",
            "province": "Quebec", "admission_rate": 0.5
        }, np.array([0.0, 0.0, 1.0]))
        return client

    def test_instantiation(self):
        client = self._make()
        assert isinstance(client, WeaviateClient)

    def test_connect_sets_connected(self):
        client = self._make()
        client.connect()
        assert client._connected is True

    def test_disconnect_clears_connected(self):
        client = self._make()
        client.connect()
        client.disconnect()
        assert client._connected is False

    def test_is_ready_before_connect(self):
        client = self._make()
        assert client.is_ready() is False

    def test_is_ready_after_connect(self):
        client = self._make()
        client.connect()
        assert client.is_ready() is True

    def test_create_schema(self):
        client = self._make()
        schema = SchemaClass(name="Test", properties=[{"name": "x", "dataType": ["text"]}])
        client.create_schema(schema)
        assert client.class_exists("Test") is True

    def test_delete_schema(self):
        client = self._make()
        schema = SchemaClass(name="Test", properties=[])
        client.create_schema(schema)
        client.delete_schema("Test")
        assert client.class_exists("Test") is False

    def test_get_schema_specific_class(self):
        client = self._make()
        schema = SchemaClass(name="Test", properties=[{"name": "x", "dataType": ["text"]}])
        client.create_schema(schema)
        result = client.get_schema("Test")
        assert isinstance(result, SchemaClass)
        assert result.name == "Test"

    def test_get_schema_nonexistent_class(self):
        client = self._make()
        result = client.get_schema("Nonexistent")
        # Returns empty dict for nonexistent classes
        assert result == {}

    def test_get_schema_all(self):
        client = self._make()
        schema1 = SchemaClass(name="A", properties=[])
        schema2 = SchemaClass(name="B", properties=[])
        client.create_schema(schema1)
        client.create_schema(schema2)
        result = client.get_schema()
        assert isinstance(result, dict)
        assert "A" in result
        assert "B" in result

    def test_class_exists_true(self):
        client = self._make()
        schema = SchemaClass(name="Test", properties=[])
        client.create_schema(schema)
        assert client.class_exists("Test") is True

    def test_class_exists_false(self):
        client = self._make()
        assert client.class_exists("Nonexistent") is False

    def test_add_object_returns_uuid(self):
        client = self._make()
        vec = np.array([0.1, 0.2, 0.3])
        result = client.add_object("Program", {"name": "CS"}, vec)
        assert isinstance(result, str)

    def test_add_object_with_custom_uuid(self):
        client = self._make()
        vec = np.array([0.1, 0.2])
        result = client.add_object("Program", {"name": "Eng"}, vec,
                                   uuid="custom-uuid")
        assert result == "custom-uuid"

    def test_add_objects_batch_returns_batch_result(self):
        client = self._make()
        objs = [
            ({"name": "CS"}, np.array([0.1, 0.2])),
            ({"name": "Eng"}, np.array([0.3, 0.4])),
        ]
        result = client.add_objects_batch("Program", objs)
        assert isinstance(result, BatchResult)
        assert result.successful == 2
        assert result.failed == 0

    def test_get_object_found(self):
        client = self._make()
        vec = np.array([0.1, 0.2])
        uuid = client.add_object("Program", {"name": "CS", "university": "UofT"}, vec)
        result = client.get_object("Program", uuid)
        assert result is not None
        assert result["name"] == "CS"
        assert result["university"] == "UofT"

    def test_get_object_not_found(self):
        client = self._make()
        result = client.get_object("Program", "nonexistent-uuid")
        assert result is None

    def test_get_object_with_vector(self):
        client = self._make()
        vec = np.array([0.1, 0.2, 0.3])
        uuid = client.add_object("Program", {"name": "CS"}, vec)
        result = client.get_object("Program", uuid, include_vector=True)
        assert result is not None
        assert "vector" in result
        assert np.allclose(result["vector"], vec)

    def test_get_object_without_vector(self):
        client = self._make()
        vec = np.array([0.1, 0.2])
        uuid = client.add_object("Program", {"name": "CS"}, vec)
        result = client.get_object("Program", uuid, include_vector=False)
        assert result is not None
        assert "vector" not in result

    def test_update_object_properties(self):
        client = self._make()
        vec = np.array([0.1, 0.2])
        uuid = client.add_object("Program", {"name": "CS", "uni": "UofT"}, vec)
        client.update_object("Program", uuid, properties={"name": "Updated CS"})
        result = client.get_object("Program", uuid)
        assert result["name"] == "Updated CS"
        assert result["uni"] == "UofT"  # other properties preserved

    def test_update_object_vector(self):
        client = self._make()
        vec = np.array([0.1, 0.2])
        uuid = client.add_object("Program", {"name": "CS"}, vec)
        new_vec = np.array([0.9, 0.8])
        client.update_object("Program", uuid, vector=new_vec)
        result = client.get_object("Program", uuid, include_vector=True)
        assert np.allclose(result["vector"], new_vec)

    def test_delete_object(self):
        client = self._make()
        vec = np.array([0.1, 0.2])
        uuid = client.add_object("Program", {"name": "CS"}, vec)
        client.delete_object("Program", uuid)
        result = client.get_object("Program", uuid)
        assert result is None

    def test_vector_search_returns_list(self):
        client = self._make_with_programs()
        query = np.array([1.0, 0.0, 0.0])
        results = client.vector_search("Program", query, limit=5)
        assert isinstance(results, list)

    def test_vector_search_returns_search_results(self):
        client = self._make_with_programs()
        query = np.array([1.0, 0.0, 0.0])
        results = client.vector_search("Program", query, limit=3)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_vector_search_nearest_first(self):
        client = self._make_with_programs()
        # Query vector [1,0,0] should be closest to CS [1,0,0], then Data Sci [0.9,0.1,0]
        query = np.array([1.0, 0.0, 0.0])
        results = client.vector_search("Program", query, limit=3)
        assert results[0].properties["name"] == "Computer Science"
        assert results[1].properties["name"] == "Data Science"

    def test_vector_search_distance_increases(self):
        client = self._make_with_programs()
        query = np.array([1.0, 0.0, 0.0])
        results = client.vector_search("Program", query, limit=3)
        for i in range(len(results) - 1):
            assert results[i].distance <= results[i + 1].distance

    def test_vector_search_with_limit(self):
        client = self._make_with_programs()
        query = np.array([1.0, 0.0, 0.0])
        results = client.vector_search("Program", query, limit=1)
        assert len(results) == 1

    def test_vector_search_with_filters(self):
        client = self._make_with_programs()
        query = np.array([0.5, 0.5, 0.0])
        filters = build_filter({"province": "Ontario"})
        results = client.vector_search("Program", query, limit=10, filters=filters)
        assert len(results) == 2  # Only Ontario programs
        for r in results:
            assert r.properties["province"] == "Ontario"

    def test_vector_search_return_properties(self):
        client = self._make_with_programs()
        query = np.array([1.0, 0.0, 0.0])
        results = client.vector_search("Program", query, limit=1,
                                        return_properties=["name"])
        assert "name" in results[0].properties
        assert "university" not in results[0].properties

    def test_vector_search_return_vector(self):
        client = self._make_with_programs()
        query = np.array([1.0, 0.0, 0.0])
        results = client.vector_search("Program", query, limit=1,
                                        return_vector=True)
        assert results[0].vector is not None

    def test_vector_search_empty_collection(self):
        client = self._make()
        query = np.array([1.0, 0.0])
        results = client.vector_search("Empty", query, limit=5)
        assert results == []

    def test_hybrid_search_returns_list(self):
        client = self._make_with_programs()
        results = client.hybrid_search("Program", "Computer", limit=5)
        assert isinstance(results, list)

    def test_hybrid_search_keyword_match(self):
        client = self._make_with_programs()
        results = client.hybrid_search("Program", "Computer", limit=5)
        # "Computer" appears in "Computer Science"
        names = [r.properties["name"] for r in results]
        assert "Computer Science" in names

    def test_hybrid_search_with_vector(self):
        client = self._make_with_programs()
        vec = np.array([1.0, 0.0, 0.0])
        results = client.hybrid_search("Program", "Science",
                                        query_vector=vec, alpha=0.5, limit=5)
        assert len(results) > 0

    def test_hybrid_search_alpha_pure_keyword(self):
        client = self._make_with_programs()
        results = client.hybrid_search("Program", "English",
                                        alpha=0.0, limit=5)
        # Pure keyword: only "English Literature" matches
        names = [r.properties["name"] for r in results]
        assert "English Literature" in names

    def test_keyword_search_returns_list(self):
        client = self._make_with_programs()
        results = client.keyword_search("Program", "Computer")
        assert isinstance(results, list)

    def test_keyword_search_finds_match(self):
        client = self._make_with_programs()
        results = client.keyword_search("Program", "Computer")
        names = [r.properties["name"] for r in results]
        assert "Computer Science" in names

    def test_keyword_search_no_match(self):
        client = self._make_with_programs()
        results = client.keyword_search("Program", "Nonexistent")
        assert results == []

    def test_keyword_search_specific_properties(self):
        client = self._make_with_programs()
        # Search only in 'university' property for "UofT"
        results = client.keyword_search("Program", "UofT",
                                         properties=["university"])
        assert len(results) == 2  # Two UofT programs

    def test_keyword_search_limit(self):
        client = self._make_with_programs()
        results = client.keyword_search("Program", "Science", limit=1)
        assert len(results) <= 1

    def test_find_similar_programs(self):
        client = self._make_with_programs()
        vec = np.array([1.0, 0.0, 0.0])
        results = client.find_similar_programs(vec, limit=2)
        assert isinstance(results, list)
        assert len(results) <= 2

    def test_find_similar_programs_exclude_id(self):
        client = self._make_with_programs()
        vec = np.array([1.0, 0.0, 0.0])
        # Get the CS program UUID
        all_results = client.vector_search("Program", vec, limit=1)
        cs_uuid = all_results[0].id
        results = client.find_similar_programs(vec, limit=5,
                                                exclude_id=cs_uuid)
        ids = [r.id for r in results]
        assert cs_uuid not in ids

    def test_find_similar_programs_province_filter(self):
        client = self._make_with_programs()
        vec = np.array([0.5, 0.5, 0.0])
        results = client.find_similar_programs(vec, limit=5,
                                                province="Ontario")
        for r in results:
            assert r.properties["province"] == "Ontario"

    def test_find_matching_students(self):
        client = self._make()
        client.connect()
        schema = create_student_schema(vector_dim=2)
        client.create_schema(schema)
        client.add_object("StudentProfile", {
            "student_id": "S1", "gpa": 90.0, "province": "Ontario"
        }, np.array([1.0, 0.0]))
        client.add_object("StudentProfile", {
            "student_id": "S2", "gpa": 85.0, "province": "Ontario"
        }, np.array([0.9, 0.1]))
        query = np.array([1.0, 0.0])
        results = client.find_matching_students(query, limit=5)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_find_matching_students_with_gpa_filter(self):
        client = self._make()
        client.connect()
        schema = create_student_schema(vector_dim=2)
        client.create_schema(schema)
        client.add_object("StudentProfile", {
            "student_id": "S1", "gpa": 90.0, "province": "Ontario"
        }, np.array([1.0, 0.0]))
        client.add_object("StudentProfile", {
            "student_id": "S2", "gpa": 75.0, "province": "Ontario"
        }, np.array([0.9, 0.1]))
        query = np.array([1.0, 0.0])
        results = client.find_matching_students(query, limit=5,
                                                 min_gpa=80.0)
        assert len(results) == 1
        assert results[0].properties["student_id"] == "S1"

    def test_recommend_programs(self):
        client = self._make_with_programs()
        vec = np.array([1.0, 0.0, 0.0])
        results = client.recommend_programs(vec, interests=["CS"], limit=3)
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_recommend_programs_province_filter(self):
        client = self._make_with_programs()
        vec = np.array([0.5, 0.5, 0.5])
        results = client.recommend_programs(vec, interests=["CS"],
                                             province="Quebec", limit=5)
        for r in results:
            assert r.properties["province"] == "Quebec"

    def test_aggregate_count_all(self):
        client = self._make_with_programs()
        count = client.aggregate_count("Program")
        assert isinstance(count, int)
        assert count == 3

    def test_aggregate_count_with_filter(self):
        client = self._make_with_programs()
        filters = build_filter({"province": "Ontario"})
        count = client.aggregate_count("Program", filters=filters)
        assert count == 2

    def test_aggregate_count_empty(self):
        client = self._make()
        count = client.aggregate_count("EmptyClass")
        assert count == 0

    def test_aggregate_by_property_count(self):
        client = self._make_with_programs()
        result = client.aggregate_by_property("Program", "province")
        assert isinstance(result, dict)
        assert result.get("Ontario") == 2
        assert result.get("Quebec") == 1

    def test_aggregate_by_property_empty(self):
        client = self._make()
        result = client.aggregate_by_property("Empty", "province")
        assert result == {}


# =========================================================================
# TestEmbeddingVectorizer
# =========================================================================

class TestEmbeddingVectorizer:
    """Tests for the EmbeddingVectorizer class."""

    def test_instantiation(self):
        ev = EmbeddingVectorizer(embedding_model=None)
        assert isinstance(ev, EmbeddingVectorizer)

    def test_encode_returns_ndarray(self):
        ev = EmbeddingVectorizer(embedding_model=None)
        result = ev.encode("Computer Science at UofT")
        assert isinstance(result, np.ndarray)

    def test_encode_returns_correct_dimension(self):
        ev = EmbeddingVectorizer(embedding_model=None)
        result = ev.encode("Computer Science at UofT")
        assert result.shape == (128,)

    def test_encode_is_normalized(self):
        ev = EmbeddingVectorizer(embedding_model=None)
        result = ev.encode("Computer Science at UofT")
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-6

    def test_encode_deterministic(self):
        ev = EmbeddingVectorizer(embedding_model=None)
        v1 = ev.encode("same text")
        v2 = ev.encode("same text")
        assert np.allclose(v1, v2)

    def test_encode_different_text_different_vector(self):
        ev = EmbeddingVectorizer(embedding_model=None)
        v1 = ev.encode("text A")
        v2 = ev.encode("text B")
        assert not np.allclose(v1, v2)

    def test_encode_batch_returns_ndarray(self):
        ev = EmbeddingVectorizer(embedding_model=None)
        result = ev.encode_batch(["CS at UofT", "Eng at UBC"])
        assert isinstance(result, np.ndarray)

    def test_encode_batch_shape(self):
        ev = EmbeddingVectorizer(embedding_model=None)
        result = ev.encode_batch(["CS", "Eng", "Math"])
        assert result.shape == (3, 128)

    def test_encode_program_returns_ndarray(self):
        ev = EmbeddingVectorizer(embedding_model=None)
        result = ev.encode_program("UofT", "CS")
        assert isinstance(result, np.ndarray)
        assert result.shape == (128,)

    def test_encode_program_with_description(self):
        ev = EmbeddingVectorizer(embedding_model=None)
        v_without = ev.encode_program("UofT", "CS")
        v_with = ev.encode_program("UofT", "CS",
                                   description="Study of computation")
        # Different because description changes the text
        assert not np.allclose(v_without, v_with)

    def test_encode_program_is_deterministic(self):
        ev = EmbeddingVectorizer(embedding_model=None)
        v1 = ev.encode_program("UofT", "CS")
        v2 = ev.encode_program("UofT", "CS")
        assert np.allclose(v1, v2)


# =========================================================================
# TestSchemaHelpers
# =========================================================================

class TestSchemaHelpers:
    """Tests for schema helper functions."""

    def test_create_program_schema_returns_schema_class(self):
        result = create_program_schema()
        assert isinstance(result, SchemaClass)

    def test_create_program_schema_name(self):
        result = create_program_schema()
        assert result.name == "Program"

    def test_create_program_schema_default_dim(self):
        result = create_program_schema()
        assert result.vector_dimension == 128

    def test_create_program_schema_custom_dim(self):
        result = create_program_schema(vector_dim=256)
        assert result.vector_dimension == 256

    def test_create_program_schema_properties(self):
        result = create_program_schema()
        prop_names = [p["name"] for p in result.properties]
        assert "name" in prop_names
        assert "university" in prop_names
        assert "faculty" in prop_names
        assert "province" in prop_names
        assert "admission_rate" in prop_names

    def test_create_program_schema_distance_metric(self):
        result = create_program_schema()
        assert result.distance_metric == "cosine"

    def test_create_student_schema_returns_schema_class(self):
        result = create_student_schema()
        assert isinstance(result, SchemaClass)

    def test_create_student_schema_name(self):
        result = create_student_schema()
        assert result.name == "StudentProfile"

    def test_create_student_schema_default_dim(self):
        result = create_student_schema()
        assert result.vector_dimension == 128

    def test_create_student_schema_custom_dim(self):
        result = create_student_schema(vector_dim=64)
        assert result.vector_dimension == 64

    def test_create_student_schema_properties(self):
        result = create_student_schema()
        prop_names = [p["name"] for p in result.properties]
        assert "student_id" in prop_names
        assert "gpa" in prop_names
        assert "province" in prop_names
        assert "interests" in prop_names


# =========================================================================
# TestFilterBuilders
# =========================================================================

class TestFilterBuilders:
    """Tests for filter builder functions."""

    def test_build_filter_single_condition(self):
        result = build_filter({"province": "Ontario"})
        assert isinstance(result, dict)
        assert result["path"] == ["province"]
        assert result["operator"] == "Equal"
        assert result["valueText"] == "Ontario"

    def test_build_filter_multiple_conditions(self):
        result = build_filter({"province": "Ontario", "degree": "BSc"})
        assert isinstance(result, dict)
        assert result["operator"] == "And"
        assert len(result["operands"]) == 2

    def test_build_filter_multiple_conditions_values(self):
        result = build_filter({"province": "Ontario", "degree": "BSc"})
        paths = [op["path"][0] for op in result["operands"]]
        assert "province" in paths
        assert "degree" in paths

    def test_build_range_filter_both_bounds(self):
        result = build_range_filter("gpa", min_val=80.0, max_val=95.0)
        assert isinstance(result, dict)
        assert result["operator"] == "And"
        assert len(result["operands"]) == 2

    def test_build_range_filter_min_only(self):
        result = build_range_filter("gpa", min_val=80.0)
        assert isinstance(result, dict)
        assert result["operator"] == "GreaterThanEqual"
        assert result["valueNumber"] == 80.0
        assert result["path"] == ["gpa"]

    def test_build_range_filter_max_only(self):
        result = build_range_filter("gpa", max_val=95.0)
        assert isinstance(result, dict)
        assert result["operator"] == "LessThanEqual"
        assert result["valueNumber"] == 95.0

    def test_build_range_filter_no_bounds(self):
        result = build_range_filter("gpa")
        assert result == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
