"""
Tests for src.db.mongo module.
=============================================

Covers MongoDB configuration dataclasses, record dataclasses, ABC enforcement
for BaseMongoClient, MongoClient operations, ApplicationQueries,
BulkOperations, IndexManager, and utility functions.

All source functions are implemented, so tests use real assertions.
"""

import pytest

from src.db.mongo import (
    MongoConfig,
    QueryConfig,
    StudentRecord,
    ApplicationRecord,
    BaseMongoClient,
    MongoClient,
    ApplicationQueries,
    BulkOperations,
    IndexManager,
    build_application_query,
    document_to_record,
    record_to_document,
    _get_nested,
    _match_query,
    _apply_projection,
    _MISSING,
    _MissingSentinel,
)


# =========================================================================
# TestMongoConfig
# =========================================================================

class TestMongoConfig:
    """Tests for the MongoConfig dataclass defaults."""

    def test_default_host(self):
        cfg = MongoConfig()
        assert cfg.host == "localhost"

    def test_default_port(self):
        cfg = MongoConfig()
        assert cfg.port == 27017

    def test_default_database(self):
        cfg = MongoConfig()
        assert cfg.database == "grade_prediction"

    def test_default_username_none(self):
        cfg = MongoConfig()
        assert cfg.username is None

    def test_default_password_none(self):
        cfg = MongoConfig()
        assert cfg.password is None

    def test_default_connection_string_none(self):
        cfg = MongoConfig()
        assert cfg.connection_string is None

    def test_default_max_pool_size(self):
        cfg = MongoConfig()
        assert cfg.max_pool_size == 100

    def test_default_timeout_ms(self):
        cfg = MongoConfig()
        assert cfg.timeout_ms == 30000

    def test_default_replica_set_none(self):
        cfg = MongoConfig()
        assert cfg.replica_set is None

    def test_custom_values(self):
        cfg = MongoConfig(
            host="db.example.com", port=27018,
            database="mydb", username="admin",
            password="secret", connection_string="mongodb://...",
            max_pool_size=50, timeout_ms=5000,
            replica_set="rs0"
        )
        assert cfg.host == "db.example.com"
        assert cfg.port == 27018
        assert cfg.database == "mydb"
        assert cfg.username == "admin"
        assert cfg.password == "secret"
        assert cfg.connection_string == "mongodb://..."
        assert cfg.max_pool_size == 50
        assert cfg.timeout_ms == 5000
        assert cfg.replica_set == "rs0"


# =========================================================================
# TestQueryConfig
# =========================================================================

class TestQueryConfig:
    """Tests for the QueryConfig dataclass defaults."""

    def test_default_batch_size(self):
        qc = QueryConfig()
        assert qc.batch_size == 1000

    def test_default_allow_disk_use(self):
        qc = QueryConfig()
        assert qc.allow_disk_use is True

    def test_default_max_time_ms(self):
        qc = QueryConfig()
        assert qc.max_time_ms == 60000

    def test_default_read_preference(self):
        qc = QueryConfig()
        assert qc.read_preference == "primary"

    def test_custom_values(self):
        qc = QueryConfig(
            batch_size=500, allow_disk_use=False,
            max_time_ms=10000, read_preference="secondary"
        )
        assert qc.batch_size == 500
        assert qc.allow_disk_use is False
        assert qc.max_time_ms == 10000
        assert qc.read_preference == "secondary"


# =========================================================================
# TestStudentRecord
# =========================================================================

class TestStudentRecord:
    """Tests for the StudentRecord dataclass."""

    def test_required_fields(self):
        sr = StudentRecord(
            student_id="STU001",
            demographics={"province": "ON"},
            academics={"gpa": 90.0},
            applications=[{"university": "UofT"}]
        )
        assert sr.student_id == "STU001"
        assert sr.demographics == {"province": "ON"}
        assert sr.academics == {"gpa": 90.0}
        assert len(sr.applications) == 1

    def test_default_metadata(self):
        sr = StudentRecord(
            student_id="STU002",
            demographics={},
            academics={},
            applications=[]
        )
        assert sr.metadata == {}

    def test_custom_metadata(self):
        sr = StudentRecord(
            student_id="STU003",
            demographics={},
            academics={},
            applications=[],
            metadata={"source": "csv"}
        )
        assert sr.metadata == {"source": "csv"}


# =========================================================================
# TestApplicationRecord
# =========================================================================

class TestApplicationRecord:
    """Tests for the ApplicationRecord dataclass."""

    def test_required_fields(self):
        ar = ApplicationRecord(
            university="UofT", program="CS",
            campus="St. George", application_date="2024-01-15"
        )
        assert ar.university == "UofT"
        assert ar.program == "CS"
        assert ar.campus == "St. George"
        assert ar.application_date == "2024-01-15"

    def test_default_outcome_none(self):
        ar = ApplicationRecord(
            university="UBC", program="Eng",
            campus=None, application_date="2024-02-01"
        )
        assert ar.outcome is None

    def test_default_decision_date_none(self):
        ar = ApplicationRecord(
            university="UBC", program="Eng",
            campus=None, application_date="2024-02-01"
        )
        assert ar.decision_date is None

    def test_default_offer_details_empty(self):
        ar = ApplicationRecord(
            university="McGill", program="Math",
            campus=None, application_date="2024-01-20"
        )
        assert ar.offer_details == {}

    def test_custom_optional_fields(self):
        ar = ApplicationRecord(
            university="Waterloo", program="SE",
            campus="Main", application_date="2024-01-10",
            outcome="admitted", decision_date="2024-05-01",
            offer_details={"scholarship": True}
        )
        assert ar.outcome == "admitted"
        assert ar.decision_date == "2024-05-01"
        assert ar.offer_details == {"scholarship": True}


# =========================================================================
# TestBaseMongoClient (ABC enforcement)
# =========================================================================

class TestBaseMongoClient:
    """Tests that BaseMongoClient cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseMongoClient()

    def test_incomplete_subclass_fails(self):
        class Partial(BaseMongoClient):
            def connect(self):
                pass
            # missing disconnect, insert_one, insert_many, find, aggregate

        with pytest.raises(TypeError):
            Partial()

    def test_complete_subclass_ok(self):
        class Full(BaseMongoClient):
            def connect(self):
                pass

            def disconnect(self):
                pass

            def insert_one(self, collection, document):
                return "id"

            def insert_many(self, collection, documents):
                return []

            def find(self, collection, query, projection=None):
                return iter([])

            def aggregate(self, collection, pipeline):
                return iter([])

        inst = Full()
        assert inst is not None


# =========================================================================
# TestHelperFunctions (private helpers)
# =========================================================================

class TestHelperFunctions:
    """Tests for _get_nested, _match_query, _apply_projection."""

    def test_get_nested_simple_key(self):
        doc = {"name": "Alice"}
        assert _get_nested(doc, "name") == "Alice"

    def test_get_nested_dotted_path(self):
        doc = {"demographics": {"province": "Ontario"}}
        assert _get_nested(doc, "demographics.province") == "Ontario"

    def test_get_nested_missing_returns_sentinel(self):
        doc = {"name": "Alice"}
        result = _get_nested(doc, "missing_key")
        assert isinstance(result, _MissingSentinel)

    def test_get_nested_partial_path_missing(self):
        doc = {"a": {"b": 1}}
        result = _get_nested(doc, "a.c")
        assert isinstance(result, _MissingSentinel)

    def test_match_query_simple_equality(self):
        doc = {"name": "Alice", "age": 20}
        assert _match_query(doc, {"name": "Alice"}) is True
        assert _match_query(doc, {"name": "Bob"}) is False

    def test_match_query_dotted_path(self):
        doc = {"demographics": {"province": "Ontario"}}
        assert _match_query(doc, {"demographics.province": "Ontario"}) is True
        assert _match_query(doc, {"demographics.province": "BC"}) is False

    def test_match_query_elem_match(self):
        doc = {
            "applications": [
                {"university": "UofT", "program": "CS", "outcome": "admitted"},
                {"university": "UBC", "program": "Eng", "outcome": "rejected"},
            ]
        }
        assert _match_query(doc, {
            "applications": {"$elemMatch": {"university": "UofT", "outcome": "admitted"}}
        }) is True
        assert _match_query(doc, {
            "applications": {"$elemMatch": {"university": "UofT", "outcome": "rejected"}}
        }) is False

    def test_match_query_elem_match_missing_array(self):
        doc = {"name": "Alice"}
        assert _match_query(doc, {
            "applications": {"$elemMatch": {"university": "UofT"}}
        }) is False

    def test_match_query_empty_query_matches_all(self):
        doc = {"name": "Alice"}
        assert _match_query(doc, {}) is True

    def test_apply_projection_none_returns_full_copy(self):
        doc = {"_id": "1", "name": "Alice", "age": 20}
        result = _apply_projection(doc, None)
        assert result == doc
        assert result is not doc  # should be a copy

    def test_apply_projection_include_fields(self):
        doc = {"_id": "1", "name": "Alice", "age": 20, "email": "a@b.com"}
        result = _apply_projection(doc, {"name": 1, "age": 1})
        assert "name" in result
        assert "age" in result
        assert "_id" in result  # _id included by default
        assert "email" not in result

    def test_apply_projection_exclude_id(self):
        doc = {"_id": "1", "name": "Alice"}
        result = _apply_projection(doc, {"name": 1, "_id": 0})
        assert "name" in result
        assert "_id" not in result

    def test_apply_projection_dotted_path(self):
        doc = {"_id": "1", "demographics": {"province": "Ontario", "city": "Toronto"}}
        result = _apply_projection(doc, {"demographics.province": 1})
        assert result["demographics"]["province"] == "Ontario"


# =========================================================================
# TestMongoClient
# =========================================================================

class TestMongoClient:
    """Tests for the MongoClient concrete class."""

    def _make(self):
        return MongoClient(MongoConfig())

    def test_instantiation(self):
        client = self._make()
        assert isinstance(client, MongoClient)

    def test_instantiation_with_query_config(self):
        client = MongoClient(MongoConfig(), query_config=QueryConfig())
        assert isinstance(client, MongoClient)
        assert client.query_config.batch_size == 1000

    def test_connect_sets_connected(self):
        client = self._make()
        client.connect()
        assert client._connected is True

    def test_disconnect_clears_connected(self):
        client = self._make()
        client.connect()
        client.disconnect()
        assert client._connected is False

    def test_insert_one_returns_string_id(self):
        client = self._make()
        result = client.insert_one("students", {"student_id": "S1"})
        assert isinstance(result, str)

    def test_insert_one_increments_id(self):
        client = self._make()
        id1 = client.insert_one("students", {"student_id": "S1"})
        id2 = client.insert_one("students", {"student_id": "S2"})
        assert id1 != id2

    def test_insert_one_stores_document(self):
        client = self._make()
        client.insert_one("students", {"student_id": "S1", "gpa": 90})
        docs = list(client.find("students", {"student_id": "S1"}))
        assert len(docs) == 1
        assert docs[0]["student_id"] == "S1"
        assert docs[0]["gpa"] == 90

    def test_insert_many_returns_list_of_ids(self):
        client = self._make()
        result = client.insert_many("students", [
            {"student_id": "S1"},
            {"student_id": "S2"},
            {"student_id": "S3"},
        ])
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(r, str) for r in result)

    def test_insert_many_stores_all_documents(self):
        client = self._make()
        client.insert_many("students", [
            {"student_id": "S1"},
            {"student_id": "S2"},
        ])
        count = client.count_documents("students")
        assert count == 2

    def test_find_returns_iterator(self):
        client = self._make()
        client.insert_one("students", {"student_id": "S1", "province": "ON"})
        result = client.find("students", {"province": "ON"})
        # Should be an iterator/generator
        docs = list(result)
        assert len(docs) == 1
        assert docs[0]["student_id"] == "S1"

    def test_find_no_match_returns_empty(self):
        client = self._make()
        client.insert_one("students", {"student_id": "S1", "province": "ON"})
        docs = list(client.find("students", {"province": "BC"}))
        assert len(docs) == 0

    def test_find_with_projection(self):
        client = self._make()
        client.insert_one("students", {"student_id": "S1", "gpa": 90, "name": "Alice"})
        docs = list(client.find("students", {}, projection={"student_id": 1}))
        assert len(docs) == 1
        assert "student_id" in docs[0]
        assert "gpa" not in docs[0]

    def test_find_with_sort_ascending(self):
        client = self._make()
        client.insert_one("students", {"name": "B", "gpa": 80})
        client.insert_one("students", {"name": "A", "gpa": 90})
        client.insert_one("students", {"name": "C", "gpa": 70})
        docs = list(client.find("students", {}, sort=[("gpa", 1)]))
        gpas = [d["gpa"] for d in docs]
        assert gpas == [70, 80, 90]

    def test_find_with_sort_descending(self):
        client = self._make()
        client.insert_one("students", {"name": "B", "gpa": 80})
        client.insert_one("students", {"name": "A", "gpa": 90})
        docs = list(client.find("students", {}, sort=[("gpa", -1)]))
        gpas = [d["gpa"] for d in docs]
        assert gpas == [90, 80]

    def test_find_with_limit(self):
        client = self._make()
        for i in range(5):
            client.insert_one("students", {"name": f"S{i}"})
        docs = list(client.find("students", {}, limit=3))
        assert len(docs) == 3

    def test_find_one_returns_single_doc(self):
        client = self._make()
        client.insert_one("students", {"student_id": "S1", "name": "Alice"})
        client.insert_one("students", {"student_id": "S2", "name": "Bob"})
        result = client.find_one("students", {"student_id": "S1"})
        assert result is not None
        assert result["student_id"] == "S1"

    def test_find_one_returns_none_when_not_found(self):
        client = self._make()
        result = client.find_one("students", {"student_id": "MISSING"})
        assert result is None

    def test_aggregate_match_stage(self):
        client = self._make()
        client.insert_one("students", {"province": "ON", "gpa": 90})
        client.insert_one("students", {"province": "BC", "gpa": 85})
        pipeline = [{"$match": {"province": "ON"}}]
        results = list(client.aggregate("students", pipeline))
        assert len(results) == 1
        assert results[0]["province"] == "ON"

    def test_aggregate_unwind_stage(self):
        client = self._make()
        client.insert_one("students", {
            "name": "Alice",
            "applications": [
                {"program": "CS"},
                {"program": "Math"},
            ]
        })
        pipeline = [{"$unwind": "$applications"}]
        results = list(client.aggregate("students", pipeline))
        assert len(results) == 2
        programs = [r["applications"]["program"] for r in results]
        assert "CS" in programs
        assert "Math" in programs

    def test_aggregate_group_sum(self):
        client = self._make()
        client.insert_one("col", {"category": "A", "amount": 10})
        client.insert_one("col", {"category": "A", "amount": 20})
        client.insert_one("col", {"category": "B", "amount": 5})
        pipeline = [
            {"$group": {
                "_id": "$category",
                "total": {"$sum": "$amount"}
            }}
        ]
        results = list(client.aggregate("col", pipeline))
        totals = {r["_id"]: r["total"] for r in results}
        assert totals["A"] == 30
        assert totals["B"] == 5

    def test_aggregate_group_count(self):
        client = self._make()
        client.insert_one("col", {"category": "A"})
        client.insert_one("col", {"category": "A"})
        client.insert_one("col", {"category": "B"})
        pipeline = [
            {"$group": {
                "_id": "$category",
                "count": {"$sum": 1}
            }}
        ]
        results = list(client.aggregate("col", pipeline))
        counts = {r["_id"]: r["count"] for r in results}
        assert counts["A"] == 2
        assert counts["B"] == 1

    def test_aggregate_group_avg(self):
        client = self._make()
        client.insert_one("col", {"cat": "A", "val": 10})
        client.insert_one("col", {"cat": "A", "val": 20})
        pipeline = [
            {"$group": {
                "_id": "$cat",
                "avg_val": {"$avg": "$val"}
            }}
        ]
        results = list(client.aggregate("col", pipeline))
        assert len(results) == 1
        assert results[0]["avg_val"] == 15.0

    def test_aggregate_project_stage(self):
        client = self._make()
        client.insert_one("col", {"a": 10, "b": 20})
        pipeline = [
            {"$project": {"a": 1, "_id": 0}}
        ]
        results = list(client.aggregate("col", pipeline))
        assert len(results) == 1
        assert "a" in results[0]
        assert "_id" not in results[0]

    def test_aggregate_sort_stage(self):
        client = self._make()
        client.insert_one("col", {"val": 3})
        client.insert_one("col", {"val": 1})
        client.insert_one("col", {"val": 2})
        pipeline = [{"$sort": {"val": 1}}]
        results = list(client.aggregate("col", pipeline))
        vals = [r["val"] for r in results]
        assert vals == [1, 2, 3]

    def test_aggregate_limit_stage(self):
        client = self._make()
        for i in range(5):
            client.insert_one("col", {"val": i})
        pipeline = [{"$limit": 2}]
        results = list(client.aggregate("col", pipeline))
        assert len(results) == 2

    def test_update_one_modifies_existing(self):
        client = self._make()
        client.insert_one("students", {"student_id": "S1", "gpa": 80})
        modified = client.update_one(
            "students", {"student_id": "S1"}, {"$set": {"gpa": 95}}
        )
        assert modified == 1
        doc = client.find_one("students", {"student_id": "S1"})
        assert doc["gpa"] == 95

    def test_update_one_no_match_returns_zero(self):
        client = self._make()
        modified = client.update_one(
            "students", {"student_id": "MISSING"}, {"$set": {"gpa": 95}}
        )
        assert modified == 0

    def test_update_one_upsert_inserts_new(self):
        client = self._make()
        modified = client.update_one(
            "students", {"student_id": "S_NEW"}, {"$set": {"gpa": 88}},
            upsert=True
        )
        assert modified == 1
        doc = client.find_one("students", {"student_id": "S_NEW"})
        assert doc is not None
        assert doc["gpa"] == 88

    def test_update_many_modifies_multiple(self):
        client = self._make()
        client.insert_one("students", {"province": "ON", "flag": False})
        client.insert_one("students", {"province": "ON", "flag": False})
        client.insert_one("students", {"province": "BC", "flag": False})
        count = client.update_many(
            "students", {"province": "ON"}, {"$set": {"flag": True}}
        )
        assert count == 2
        on_docs = list(client.find("students", {"province": "ON"}))
        assert all(d["flag"] is True for d in on_docs)

    def test_delete_many_removes_matching(self):
        client = self._make()
        client.insert_one("students", {"province": "ON"})
        client.insert_one("students", {"province": "ON"})
        client.insert_one("students", {"province": "BC"})
        deleted = client.delete_many("students", {"province": "ON"})
        assert deleted == 2
        remaining = client.count_documents("students")
        assert remaining == 1

    def test_delete_many_no_match_returns_zero(self):
        client = self._make()
        client.insert_one("students", {"province": "ON"})
        deleted = client.delete_many("students", {"province": "QC"})
        assert deleted == 0

    def test_count_documents_with_query(self):
        client = self._make()
        client.insert_one("students", {"province": "ON"})
        client.insert_one("students", {"province": "ON"})
        client.insert_one("students", {"province": "BC"})
        assert client.count_documents("students", {"province": "ON"}) == 2

    def test_count_documents_no_query(self):
        client = self._make()
        client.insert_one("students", {"a": 1})
        client.insert_one("students", {"b": 2})
        assert client.count_documents("students") == 2

    def test_count_documents_empty_collection(self):
        client = self._make()
        assert client.count_documents("empty_col") == 0


# =========================================================================
# TestApplicationQueries
# =========================================================================

class TestApplicationQueries:
    """Tests for the ApplicationQueries class."""

    def _make_with_data(self):
        """Create a client with sample student data and return (aq, client)."""
        client = MongoClient(MongoConfig())
        # Insert sample students with applications
        client.insert_one("students", {
            "student_id": "S1",
            "demographics": {"province": "Ontario"},
            "academics": {"gpa_overall": 90.0},
            "applications": [
                {"university": "UofT", "program": "CS",
                 "outcome": "admitted", "application_date": "2023-09-01"},
                {"university": "UBC", "program": "Eng",
                 "outcome": "rejected", "application_date": "2023-10-01"},
            ]
        })
        client.insert_one("students", {
            "student_id": "S2",
            "demographics": {"province": "Ontario"},
            "academics": {"gpa_overall": 85.0},
            "applications": [
                {"university": "UofT", "program": "CS",
                 "outcome": "rejected", "application_date": "2023-09-15"},
            ]
        })
        client.insert_one("students", {
            "student_id": "S3",
            "demographics": {"province": "BC"},
            "academics": {"gpa_overall": 92.0},
            "applications": [
                {"university": "UofT", "program": "CS",
                 "outcome": "admitted", "application_date": "2023-08-20"},
            ]
        })
        aq = ApplicationQueries(client)
        return aq, client

    def test_instantiation(self):
        client = MongoClient(MongoConfig())
        aq = ApplicationQueries(client)
        assert isinstance(aq, ApplicationQueries)

    def test_get_admission_rates_returns_dict(self):
        aq, _ = self._make_with_data()
        result = aq.get_admission_rates(2023)
        assert isinstance(result, dict)

    def test_get_admission_rates_by_program(self):
        aq, _ = self._make_with_data()
        result = aq.get_admission_rates(2023, by="program")
        assert isinstance(result, dict)
        # CS has 4 total applications (3 CS + 1 Eng unwound)
        # but grouped by program: CS should have 3 apps (2 admitted, 1 rejected)
        if "CS" in result:
            assert 0.0 <= result["CS"] <= 1.0

    def test_get_admission_rates_by_university(self):
        aq, _ = self._make_with_data()
        result = aq.get_admission_rates(2023, by="university")
        assert isinstance(result, dict)

    def test_get_gpa_statistics_returns_dict(self):
        aq, _ = self._make_with_data()
        result = aq.get_gpa_statistics("UofT", "CS")
        assert isinstance(result, dict)
        assert "mean" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result
        assert "median" in result
        assert "count" in result

    def test_get_gpa_statistics_with_data(self):
        aq, _ = self._make_with_data()
        result = aq.get_gpa_statistics("UofT", "CS")
        # S1 (90), S2 (85), S3 (92) all applied to UofT CS
        assert result["count"] == 3
        assert result["min"] == 85.0
        assert result["max"] == 92.0

    def test_get_gpa_statistics_with_outcome_filter(self):
        aq, _ = self._make_with_data()
        result = aq.get_gpa_statistics("UofT", "CS", outcome="admitted")
        # S1 (90) and S3 (92) admitted
        assert result["count"] == 2
        assert result["min"] == 90.0
        assert result["max"] == 92.0

    def test_get_gpa_statistics_no_match(self):
        aq, _ = self._make_with_data()
        result = aq.get_gpa_statistics("Harvard", "Law")
        assert result["count"] == 0
        assert result["mean"] == 0.0

    def test_get_application_trends_returns_dict(self):
        aq, _ = self._make_with_data()
        result = aq.get_application_trends("UofT", "CS", [2022, 2023])
        assert isinstance(result, dict)
        assert 2022 in result
        assert 2023 in result

    def test_get_application_trends_counts(self):
        aq, _ = self._make_with_data()
        result = aq.get_application_trends("UofT", "CS", [2023])
        assert result[2023]["applications"] == 3
        assert result[2023]["admissions"] == 2

    def test_get_application_trends_year_no_data(self):
        aq, _ = self._make_with_data()
        result = aq.get_application_trends("UofT", "CS", [2020])
        assert result[2020]["applications"] == 0
        assert result[2020]["admissions"] == 0

    def test_get_competitive_programs_returns_list(self):
        aq, _ = self._make_with_data()
        result = aq.get_competitive_programs(top_n=10, min_applications=1)
        assert isinstance(result, list)
        # Should have entries with rate, program, university, applications
        if result:
            assert "rate" in result[0]
            assert "program" in result[0]

    def test_get_competitive_programs_min_applications_filter(self):
        aq, _ = self._make_with_data()
        # With min_applications=100, no programs should qualify with our small dataset
        result = aq.get_competitive_programs(top_n=10, min_applications=100)
        assert result == []

    def test_get_student_applications_found(self):
        aq, _ = self._make_with_data()
        result = aq.get_student_applications("S1")
        assert isinstance(result, list)
        assert len(result) == 2

    def test_get_student_applications_not_found(self):
        aq, _ = self._make_with_data()
        result = aq.get_student_applications("MISSING")
        assert result == []

    def test_find_similar_students_returns_list(self):
        aq, _ = self._make_with_data()
        result = aq.find_similar_students("S1", limit=5)
        assert isinstance(result, list)

    def test_find_similar_students_within_gpa_range(self):
        aq, _ = self._make_with_data()
        # S1 has GPA 90. Range is 85-95. S2 (85) and S3 (92) should match.
        result = aq.find_similar_students("S1", limit=10)
        ids = [d.get("student_id") for d in result]
        assert "S1" not in ids  # excludes self
        assert "S2" in ids
        assert "S3" in ids

    def test_find_similar_students_not_found(self):
        aq, _ = self._make_with_data()
        result = aq.find_similar_students("MISSING", limit=5)
        assert result == []


# =========================================================================
# TestBulkOperations
# =========================================================================

class TestBulkOperations:
    """Tests for the BulkOperations class."""

    def _make(self):
        client = MongoClient(MongoConfig())
        return BulkOperations(client, batch_size=2), client

    def test_instantiation(self):
        bo, _ = self._make()
        assert isinstance(bo, BulkOperations)

    def test_bulk_insert_returns_dict(self):
        bo, _ = self._make()
        docs = iter([{"id": "1"}, {"id": "2"}])
        result = bo.bulk_insert("students", docs)
        assert isinstance(result, dict)
        assert "inserted" in result
        assert "errors" in result

    def test_bulk_insert_counts_correct(self):
        bo, client = self._make()
        docs = iter([{"id": "1"}, {"id": "2"}, {"id": "3"}])
        result = bo.bulk_insert("students", docs)
        assert result["inserted"] == 3
        assert result["errors"] == 0

    def test_bulk_insert_batches_correctly(self):
        bo, client = self._make()
        # batch_size=2, so 5 docs should be 2 full batches + 1 remainder
        docs = iter([{"id": str(i)} for i in range(5)])
        result = bo.bulk_insert("col", docs)
        assert result["inserted"] == 5
        assert client.count_documents("col") == 5

    def test_bulk_insert_ordered(self):
        bo, _ = self._make()
        docs = iter([{"id": "1"}])
        result = bo.bulk_insert("students", docs, ordered=True)
        assert result["inserted"] == 1
        assert result["errors"] == 0

    def test_bulk_upsert_returns_dict(self):
        bo, _ = self._make()
        docs = iter([{"_id": "1", "name": "A"}, {"_id": "2", "name": "B"}])
        result = bo.bulk_upsert("students", docs, key_field="_id")
        assert isinstance(result, dict)
        assert "inserted" in result
        assert "updated" in result
        assert "errors" in result

    def test_bulk_upsert_inserts_new(self):
        bo, client = self._make()
        docs = iter([{"student_id": "S1", "name": "Alice"}])
        result = bo.bulk_upsert("students", docs, key_field="student_id")
        assert result["inserted"] == 1
        assert result["updated"] == 0

    def test_bulk_upsert_updates_existing(self):
        bo, client = self._make()
        # First insert
        client.insert_one("students", {"student_id": "S1", "name": "Alice"})
        # Now upsert with same key
        docs = iter([{"student_id": "S1", "name": "Alice Updated"}])
        result = bo.bulk_upsert("students", docs, key_field="student_id")
        assert result["updated"] == 1
        assert result["inserted"] == 0

    def test_bulk_upsert_missing_key_counts_as_error(self):
        bo, _ = self._make()
        docs = iter([{"no_key": "val"}])
        result = bo.bulk_upsert("students", docs, key_field="student_id")
        assert result["errors"] == 1


# =========================================================================
# TestIndexManager
# =========================================================================

class TestIndexManager:
    """Tests for the IndexManager class."""

    def _make(self):
        client = MongoClient(MongoConfig())
        return IndexManager(client)

    def test_instantiation(self):
        im = self._make()
        assert isinstance(im, IndexManager)

    def test_create_admission_indexes_returns_list(self):
        im = self._make()
        result = im.create_admission_indexes()
        assert isinstance(result, list)
        assert len(result) == 7  # 5 students + 2 programs indexes

    def test_create_admission_indexes_expected_names(self):
        im = self._make()
        result = im.create_admission_indexes()
        assert "student_id_1" in result
        assert "demographics.province_1" in result
        assert "applications.university_1_applications.program_1" in result
        assert "university_1_program_name_1" in result

    def test_list_indexes_after_create(self):
        im = self._make()
        im.create_admission_indexes()
        students_indexes = im.list_indexes("students")
        assert isinstance(students_indexes, list)
        assert len(students_indexes) == 5
        programs_indexes = im.list_indexes("programs")
        assert len(programs_indexes) == 2

    def test_list_indexes_empty_collection(self):
        im = self._make()
        result = im.list_indexes("nonexistent")
        assert result == []

    def test_drop_index(self):
        im = self._make()
        im.create_admission_indexes()
        im.drop_index("students", "student_id_1")
        indexes = im.list_indexes("students")
        names = [idx["name"] for idx in indexes]
        assert "student_id_1" not in names
        assert len(indexes) == 4

    def test_drop_index_nonexistent_no_error(self):
        im = self._make()
        im.create_admission_indexes()
        # Should not raise an error
        im.drop_index("students", "nonexistent_index")
        assert len(im.list_indexes("students")) == 5  # unchanged


# =========================================================================
# TestUtilityFunctions
# =========================================================================

class TestUtilityFunctions:
    """Tests for module-level utility functions."""

    def test_build_application_query_no_args(self):
        result = build_application_query()
        assert isinstance(result, dict)
        assert result == {}  # no filters means empty query

    def test_build_application_query_with_university(self):
        result = build_application_query(university="UofT")
        assert "applications" in result
        assert "$elemMatch" in result["applications"]
        assert result["applications"]["$elemMatch"]["university"] == "UofT"

    def test_build_application_query_with_program(self):
        result = build_application_query(program="CS")
        assert result["applications"]["$elemMatch"]["program"] == "CS"

    def test_build_application_query_with_outcome(self):
        result = build_application_query(outcome="admitted")
        assert result["applications"]["$elemMatch"]["outcome"] == "admitted"

    def test_build_application_query_with_year(self):
        result = build_application_query(year=2023)
        elem = result["applications"]["$elemMatch"]
        assert "application_date" in elem
        assert elem["application_date"]["$gte"] == "2023-01-01"
        assert elem["application_date"]["$lt"] == "2024-01-01"

    def test_build_application_query_with_province(self):
        result = build_application_query(province="ON")
        assert result["demographics.province"] == "ON"

    def test_build_application_query_all_args(self):
        result = build_application_query(
            university="UofT", program="CS",
            outcome="admitted", year=2023, province="ON"
        )
        assert "applications" in result
        assert "demographics.province" in result
        elem = result["applications"]["$elemMatch"]
        assert elem["university"] == "UofT"
        assert elem["program"] == "CS"
        assert elem["outcome"] == "admitted"
        assert "application_date" in elem

    def test_document_to_record(self):
        doc = {
            "student_id": "S1",
            "demographics": {"province": "ON"},
            "academics": {"gpa": 90.0},
            "applications": [{"university": "UofT"}]
        }
        result = document_to_record(doc)
        assert isinstance(result, StudentRecord)
        assert result.student_id == "S1"
        assert result.demographics == {"province": "ON"}
        assert result.academics == {"gpa": 90.0}
        assert len(result.applications) == 1

    def test_document_to_record_missing_fields(self):
        doc = {}
        result = document_to_record(doc)
        assert isinstance(result, StudentRecord)
        assert result.student_id == ""
        assert result.demographics == {}
        assert result.academics == {}
        assert result.applications == []
        assert result.metadata == {}

    def test_record_to_document(self):
        sr = StudentRecord(
            student_id="S1",
            demographics={"province": "ON"},
            academics={"gpa": 90.0},
            applications=[{"university": "UofT"}]
        )
        result = record_to_document(sr)
        assert isinstance(result, dict)
        assert result["student_id"] == "S1"
        assert result["demographics"] == {"province": "ON"}
        assert result["academics"] == {"gpa": 90.0}
        assert len(result["applications"]) == 1
        assert result["metadata"] == {}

    def test_record_to_document_roundtrip(self):
        doc = {
            "student_id": "S1",
            "demographics": {"province": "ON"},
            "academics": {"gpa": 90.0},
            "applications": [{"university": "UofT"}],
            "metadata": {"source": "csv"},
        }
        record = document_to_record(doc)
        result = record_to_document(record)
        assert result == doc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
