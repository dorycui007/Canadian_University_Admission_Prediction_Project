"""
Tests for src.db.mongo module.
=============================================

Covers MongoDB configuration dataclasses, record dataclasses, ABC enforcement
for BaseMongoClient, MongoClient stubs, ApplicationQueries stubs,
BulkOperations stubs, IndexManager stubs, and utility function stubs.

Pattern: call-then-skip -- call each stub, if it returns None pytest.skip.
For dataclasses: real assertions on defaults.
For ABCs: test cannot-instantiate with pytest.raises(TypeError).
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
# TestMongoClient
# =========================================================================

class TestMongoClient:
    """Tests for the MongoClient concrete class stubs."""

    def _make(self):
        return MongoClient(MongoConfig())

    def test_instantiation(self):
        client = self._make()
        assert isinstance(client, MongoClient)

    def test_instantiation_with_query_config(self):
        client = MongoClient(MongoConfig(), query_config=QueryConfig())
        assert isinstance(client, MongoClient)

    def test_connect_stub(self):
        client = self._make()
        result = client.connect()
        if result is not None:
            pass  # implemented

    def test_disconnect_stub(self):
        client = self._make()
        result = client.disconnect()
        if result is not None:
            pass  # implemented

    def test_insert_one_stub(self):
        client = self._make()
        result = client.insert_one("students", {"student_id": "S1"})
        if result is None:
            pytest.skip("insert_one not yet implemented")
        assert isinstance(result, str)

    def test_insert_many_stub(self):
        client = self._make()
        result = client.insert_many("students", [{"student_id": "S1"}])
        if result is None:
            pytest.skip("insert_many not yet implemented")
        assert isinstance(result, list)

    def test_find_stub(self):
        client = self._make()
        result = client.find("students", {"province": "ON"})
        if result is None:
            pytest.skip("find not yet implemented")

    def test_find_with_projection_stub(self):
        client = self._make()
        result = client.find("students", {}, projection={"student_id": 1})
        if result is None:
            pytest.skip("find not yet implemented")

    def test_find_one_stub(self):
        client = self._make()
        result = client.find_one("students", {"student_id": "S1"})
        if result is None:
            pytest.skip("find_one not yet implemented")

    def test_aggregate_stub(self):
        client = self._make()
        pipeline = [{"$match": {"province": "ON"}}]
        result = client.aggregate("students", pipeline)
        if result is None:
            pytest.skip("aggregate not yet implemented")

    def test_update_one_stub(self):
        client = self._make()
        result = client.update_one(
            "students", {"student_id": "S1"}, {"$set": {"gpa": 95}}
        )
        if result is None:
            pytest.skip("update_one not yet implemented")

    def test_update_many_stub(self):
        client = self._make()
        result = client.update_many(
            "students", {"province": "ON"}, {"$set": {"flag": True}}
        )
        if result is None:
            pytest.skip("update_many not yet implemented")

    def test_delete_many_stub(self):
        client = self._make()
        result = client.delete_many("students", {"province": "QC"})
        if result is None:
            pytest.skip("delete_many not yet implemented")

    def test_count_documents_stub(self):
        client = self._make()
        result = client.count_documents("students", {"province": "ON"})
        if result is None:
            pytest.skip("count_documents not yet implemented")

    def test_count_documents_no_query_stub(self):
        client = self._make()
        result = client.count_documents("students")
        if result is None:
            pytest.skip("count_documents not yet implemented")


# =========================================================================
# TestApplicationQueries
# =========================================================================

class TestApplicationQueries:
    """Tests for the ApplicationQueries class stubs."""

    def _make(self):
        client = MongoClient(MongoConfig())
        return ApplicationQueries(client)

    def test_instantiation(self):
        aq = self._make()
        assert isinstance(aq, ApplicationQueries)

    def test_get_admission_rates_stub(self):
        aq = self._make()
        result = aq.get_admission_rates(2023)
        if result is None:
            pytest.skip("get_admission_rates not yet implemented")
        assert isinstance(result, dict)

    def test_get_admission_rates_by_university_stub(self):
        aq = self._make()
        result = aq.get_admission_rates(2023, by="university")
        if result is None:
            pytest.skip("get_admission_rates not yet implemented")

    def test_get_gpa_statistics_stub(self):
        aq = self._make()
        result = aq.get_gpa_statistics("UofT", "CS")
        if result is None:
            pytest.skip("get_gpa_statistics not yet implemented")
        assert isinstance(result, dict)

    def test_get_gpa_statistics_with_outcome_stub(self):
        aq = self._make()
        result = aq.get_gpa_statistics("UofT", "CS", outcome="admitted")
        if result is None:
            pytest.skip("get_gpa_statistics not yet implemented")

    def test_get_application_trends_stub(self):
        aq = self._make()
        result = aq.get_application_trends("UofT", "CS", [2021, 2022, 2023])
        if result is None:
            pytest.skip("get_application_trends not yet implemented")
        assert isinstance(result, dict)

    def test_get_competitive_programs_stub(self):
        aq = self._make()
        result = aq.get_competitive_programs(top_n=10, min_applications=50)
        if result is None:
            pytest.skip("get_competitive_programs not yet implemented")
        assert isinstance(result, list)

    def test_get_student_applications_stub(self):
        aq = self._make()
        result = aq.get_student_applications("STU001")
        if result is None:
            pytest.skip("get_student_applications not yet implemented")
        assert isinstance(result, list)

    def test_find_similar_students_stub(self):
        aq = self._make()
        result = aq.find_similar_students("STU001", limit=5)
        if result is None:
            pytest.skip("find_similar_students not yet implemented")
        assert isinstance(result, list)


# =========================================================================
# TestBulkOperations
# =========================================================================

class TestBulkOperations:
    """Tests for the BulkOperations class stubs."""

    def _make(self):
        client = MongoClient(MongoConfig())
        return BulkOperations(client, batch_size=500)

    def test_instantiation(self):
        bo = self._make()
        assert isinstance(bo, BulkOperations)

    def test_bulk_insert_stub(self):
        bo = self._make()
        docs = iter([{"id": "1"}, {"id": "2"}])
        result = bo.bulk_insert("students", docs)
        if result is None:
            pytest.skip("bulk_insert not yet implemented")
        assert isinstance(result, dict)

    def test_bulk_insert_ordered_stub(self):
        bo = self._make()
        docs = iter([{"id": "1"}])
        result = bo.bulk_insert("students", docs, ordered=True)
        if result is None:
            pytest.skip("bulk_insert not yet implemented")

    def test_bulk_upsert_stub(self):
        bo = self._make()
        docs = iter([{"_id": "1", "name": "A"}, {"_id": "2", "name": "B"}])
        result = bo.bulk_upsert("students", docs, key_field="_id")
        if result is None:
            pytest.skip("bulk_upsert not yet implemented")
        assert isinstance(result, dict)


# =========================================================================
# TestIndexManager
# =========================================================================

class TestIndexManager:
    """Tests for the IndexManager class stubs."""

    def _make(self):
        client = MongoClient(MongoConfig())
        return IndexManager(client)

    def test_instantiation(self):
        im = self._make()
        assert isinstance(im, IndexManager)

    def test_create_admission_indexes_stub(self):
        im = self._make()
        result = im.create_admission_indexes()
        if result is None:
            pytest.skip("create_admission_indexes not yet implemented")
        assert isinstance(result, list)

    def test_list_indexes_stub(self):
        im = self._make()
        result = im.list_indexes("students")
        if result is None:
            pytest.skip("list_indexes not yet implemented")
        assert isinstance(result, list)

    def test_drop_index_stub(self):
        im = self._make()
        result = im.drop_index("students", "student_id_1")
        if result is not None:
            pass  # implemented


# =========================================================================
# TestUtilityFunctions
# =========================================================================

class TestUtilityFunctions:
    """Tests for module-level utility function stubs."""

    def test_build_application_query_no_args_stub(self):
        result = build_application_query()
        if result is None:
            pytest.skip("build_application_query not yet implemented")
        assert isinstance(result, dict)

    def test_build_application_query_with_university_stub(self):
        result = build_application_query(university="UofT")
        if result is None:
            pytest.skip("build_application_query not yet implemented")

    def test_build_application_query_all_args_stub(self):
        result = build_application_query(
            university="UofT", program="CS",
            outcome="admitted", year=2023, province="ON"
        )
        if result is None:
            pytest.skip("build_application_query not yet implemented")

    def test_document_to_record_stub(self):
        doc = {
            "student_id": "S1",
            "demographics": {"province": "ON"},
            "academics": {"gpa": 90.0},
            "applications": []
        }
        result = document_to_record(doc)
        if result is None:
            pytest.skip("document_to_record not yet implemented")
        assert isinstance(result, StudentRecord)

    def test_record_to_document_stub(self):
        sr = StudentRecord(
            student_id="S1",
            demographics={"province": "ON"},
            academics={"gpa": 90.0},
            applications=[]
        )
        result = record_to_document(sr)
        if result is None:
            pytest.skip("record_to_document not yet implemented")
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
