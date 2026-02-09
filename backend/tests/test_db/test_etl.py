"""
Tests for src.db.etl module.
=============================================

Covers ETL configuration dataclasses, abstract base classes (ABC enforcement),
concrete extractors, transformers, loaders, pipeline orchestrator, and
utility functions.
"""

import pytest
import json
import os

from src.db.etl import (
    ETLConfig,
    DataSourceConfig,
    ValidationResult,
    ETLStats,
    BaseExtractor,
    BaseTransformer,
    BaseLoader,
    CSVExtractor,
    JSONExtractor,
    APIExtractor,
    CleaningTransformer,
    NormalizationTransformer,
    EnrichmentTransformer,
    ValidationTransformer,
    MongoLoader,
    WeaviateLoader,
    ETLPipeline,
    create_admission_etl,
    run_incremental_load,
)


# =========================================================================
# TestETLConfig
# =========================================================================

class TestETLConfig:
    """Tests for the ETLConfig dataclass defaults."""

    def test_default_batch_size(self):
        cfg = ETLConfig()
        assert cfg.batch_size == 1000

    def test_default_error_threshold(self):
        cfg = ETLConfig()
        assert cfg.error_threshold == 100

    def test_default_validate_all(self):
        cfg = ETLConfig()
        assert cfg.validate_all is True

    def test_default_skip_existing(self):
        cfg = ETLConfig()
        assert cfg.skip_existing is True

    def test_default_dry_run(self):
        cfg = ETLConfig()
        assert cfg.dry_run is False

    def test_default_log_level(self):
        cfg = ETLConfig()
        assert cfg.log_level == "INFO"

    def test_custom_values(self):
        cfg = ETLConfig(
            batch_size=500, error_threshold=50, validate_all=False,
            skip_existing=False, dry_run=True, log_level="DEBUG"
        )
        assert cfg.batch_size == 500
        assert cfg.error_threshold == 50
        assert cfg.validate_all is False
        assert cfg.skip_existing is False
        assert cfg.dry_run is True
        assert cfg.log_level == "DEBUG"


# =========================================================================
# TestDataSourceConfig
# =========================================================================

class TestDataSourceConfig:
    """Tests for the DataSourceConfig dataclass."""

    def test_required_fields(self):
        dsc = DataSourceConfig(source_type="csv", path="test.csv")
        assert dsc.source_type == "csv"
        assert dsc.path == "test.csv"

    def test_default_encoding(self):
        dsc = DataSourceConfig(source_type="csv", path="f.csv")
        assert dsc.encoding == "utf-8"

    def test_default_delimiter(self):
        dsc = DataSourceConfig(source_type="csv", path="f.csv")
        assert dsc.delimiter == ","

    def test_default_schema_none(self):
        dsc = DataSourceConfig(source_type="csv", path="f.csv")
        assert dsc.schema is None

    def test_default_date_format(self):
        dsc = DataSourceConfig(source_type="csv", path="f.csv")
        assert dsc.date_format == "%Y-%m-%d"

    def test_custom_values(self):
        dsc = DataSourceConfig(
            source_type="json", path="data.json",
            encoding="latin-1", delimiter=";",
            schema={"col": "str"}, date_format="%d/%m/%Y"
        )
        assert dsc.source_type == "json"
        assert dsc.path == "data.json"
        assert dsc.encoding == "latin-1"
        assert dsc.delimiter == ";"
        assert dsc.schema == {"col": "str"}
        assert dsc.date_format == "%d/%m/%Y"


# =========================================================================
# TestValidationResult
# =========================================================================

class TestValidationResult:
    """Tests for the ValidationResult dataclass."""

    def test_required_is_valid(self):
        vr = ValidationResult(is_valid=True)
        assert vr.is_valid is True

    def test_default_errors_empty(self):
        vr = ValidationResult(is_valid=True)
        assert vr.errors == []

    def test_default_warnings_empty(self):
        vr = ValidationResult(is_valid=True)
        assert vr.warnings == []

    def test_default_stats_empty(self):
        vr = ValidationResult(is_valid=True)
        assert vr.stats == {}

    def test_custom_values(self):
        vr = ValidationResult(
            is_valid=False,
            errors=[{"field": "gpa", "msg": "out of range"}],
            warnings=[{"field": "name", "msg": "truncated"}],
            stats={"total": 100}
        )
        assert vr.is_valid is False
        assert len(vr.errors) == 1
        assert len(vr.warnings) == 1
        assert vr.stats["total"] == 100


# =========================================================================
# TestETLStats
# =========================================================================

class TestETLStats:
    """Tests for the ETLStats dataclass."""

    def test_default_records_read(self):
        stats = ETLStats()
        assert stats.records_read == 0

    def test_default_records_processed(self):
        stats = ETLStats()
        assert stats.records_processed == 0

    def test_default_records_inserted(self):
        stats = ETLStats()
        assert stats.records_inserted == 0

    def test_default_records_updated(self):
        stats = ETLStats()
        assert stats.records_updated == 0

    def test_default_records_skipped(self):
        stats = ETLStats()
        assert stats.records_skipped == 0

    def test_default_records_failed(self):
        stats = ETLStats()
        assert stats.records_failed == 0

    def test_default_duration(self):
        stats = ETLStats()
        assert stats.duration_seconds == 0.0

    def test_default_errors_empty(self):
        stats = ETLStats()
        assert stats.errors == []


# =========================================================================
# TestBaseExtractor (ABC enforcement)
# =========================================================================

class TestBaseExtractor:
    """Tests that BaseExtractor cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseExtractor()

    def test_incomplete_subclass_fails(self):
        class PartialExtractor(BaseExtractor):
            def extract(self):
                pass
            # missing validate_source

        with pytest.raises(TypeError):
            PartialExtractor()

    def test_complete_subclass_ok(self):
        class FullExtractor(BaseExtractor):
            def extract(self):
                yield {}

            def validate_source(self):
                return True

        inst = FullExtractor()
        assert inst is not None

    def test_get_record_count_default_returns_none(self):
        """BaseExtractor.get_record_count returns None by default."""
        class MinimalExtractor(BaseExtractor):
            def extract(self):
                yield {}

            def validate_source(self):
                return True

        ext = MinimalExtractor()
        result = ext.get_record_count()
        assert result is None


# =========================================================================
# TestBaseTransformer (ABC enforcement)
# =========================================================================

class TestBaseTransformer:
    """Tests that BaseTransformer cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseTransformer()

    def test_incomplete_subclass_fails(self):
        class Empty(BaseTransformer):
            pass

        with pytest.raises(TypeError):
            Empty()

    def test_complete_subclass_ok(self):
        class FullTransformer(BaseTransformer):
            def transform(self, record):
                return record

        inst = FullTransformer()
        assert inst is not None

    def test_transform_batch_delegates_to_transform(self):
        """transform_batch calls transform() on each record."""
        class UpperTransformer(BaseTransformer):
            def transform(self, record):
                return {k: v.upper() if isinstance(v, str) else v
                        for k, v in record.items()}

        t = UpperTransformer()
        result = t.transform_batch([{"a": "hello"}, {"b": "world"}])
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"a": "HELLO"}
        assert result[1] == {"b": "WORLD"}

    def test_transform_batch_filters_none(self):
        """transform_batch skips records where transform returns None."""
        class SkipTransformer(BaseTransformer):
            def transform(self, record):
                return None if record.get("skip") else record

        t = SkipTransformer()
        result = t.transform_batch([{"a": 1}, {"skip": True}, {"b": 2}])
        assert len(result) == 2


# =========================================================================
# TestBaseLoader (ABC enforcement)
# =========================================================================

class TestBaseLoader:
    """Tests that BaseLoader cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseLoader()

    def test_incomplete_subclass_fails(self):
        class PartialLoader(BaseLoader):
            def load(self, record):
                return True
            # missing load_batch and exists

        with pytest.raises(TypeError):
            PartialLoader()

    def test_complete_subclass_ok(self):
        class FullLoader(BaseLoader):
            def load(self, record):
                return True

            def load_batch(self, records):
                return len(records)

            def exists(self, key):
                return False

        inst = FullLoader()
        assert inst is not None


# =========================================================================
# TestCSVExtractor
# =========================================================================

class TestCSVExtractor:
    """Tests for the CSVExtractor concrete class."""

    def test_instantiation(self):
        config = DataSourceConfig(source_type="csv", path="test.csv")
        ext = CSVExtractor(config)
        assert isinstance(ext, CSVExtractor)

    def test_extract_yields_rows(self, tmp_path):
        """extract() yields dicts from CSV rows."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,gpa\nAlice,90\nBob,85\n")
        config = DataSourceConfig(source_type="csv", path=str(csv_file))
        ext = CSVExtractor(config)
        rows = list(ext.extract())
        assert len(rows) == 2
        assert rows[0]["name"] == "Alice"
        assert rows[0]["gpa"] == "90"
        assert rows[1]["name"] == "Bob"

    def test_validate_source_existing_file(self, tmp_path):
        """validate_source returns True for existing file."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col\nval\n")
        config = DataSourceConfig(source_type="csv", path=str(csv_file))
        ext = CSVExtractor(config)
        assert ext.validate_source() is True

    def test_validate_source_missing_file(self):
        """validate_source returns False for non-existent file."""
        config = DataSourceConfig(source_type="csv", path="/nonexistent/file.csv")
        ext = CSVExtractor(config)
        assert ext.validate_source() is False

    def test_get_record_count(self, tmp_path):
        """get_record_count returns number of data rows."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,gpa\nAlice,90\nBob,85\nCarol,92\n")
        config = DataSourceConfig(source_type="csv", path=str(csv_file))
        ext = CSVExtractor(config)
        assert ext.get_record_count() == 3

    def test_get_record_count_missing_file(self):
        """get_record_count returns 0 for missing file."""
        config = DataSourceConfig(source_type="csv", path="/nonexistent.csv")
        ext = CSVExtractor(config)
        assert ext.get_record_count() == 0


# =========================================================================
# TestJSONExtractor
# =========================================================================

class TestJSONExtractor:
    """Tests for the JSONExtractor concrete class."""

    def test_instantiation(self):
        config = DataSourceConfig(source_type="json", path="test.json")
        ext = JSONExtractor(config)
        assert isinstance(ext, JSONExtractor)

    def test_instantiation_with_record_path(self):
        config = DataSourceConfig(source_type="json", path="test.json")
        ext = JSONExtractor(config, record_path="data.students")
        assert isinstance(ext, JSONExtractor)

    def test_extract_json_array(self, tmp_path):
        """extract() yields items from JSON array."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps([{"name": "Alice"}, {"name": "Bob"}]))
        config = DataSourceConfig(source_type="json", path=str(json_file))
        ext = JSONExtractor(config)
        rows = list(ext.extract())
        assert len(rows) == 2
        assert rows[0]["name"] == "Alice"

    def test_extract_json_object(self, tmp_path):
        """extract() yields a single dict from JSON object."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps({"name": "Alice", "gpa": 90}))
        config = DataSourceConfig(source_type="json", path=str(json_file))
        ext = JSONExtractor(config)
        rows = list(ext.extract())
        assert len(rows) == 1
        assert rows[0]["name"] == "Alice"

    def test_extract_with_record_path(self, tmp_path):
        """extract() navigates record_path to find records."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps({
            "data": {"students": [{"name": "Alice"}, {"name": "Bob"}]}
        }))
        config = DataSourceConfig(source_type="json", path=str(json_file))
        ext = JSONExtractor(config, record_path="data.students")
        rows = list(ext.extract())
        assert len(rows) == 2

    def test_validate_source_valid_json(self, tmp_path):
        """validate_source returns True for valid JSON file."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps({"key": "value"}))
        config = DataSourceConfig(source_type="json", path=str(json_file))
        ext = JSONExtractor(config)
        assert ext.validate_source() is True

    def test_validate_source_missing_file(self):
        """validate_source returns False for non-existent file."""
        config = DataSourceConfig(source_type="json", path="/nonexistent.json")
        ext = JSONExtractor(config)
        assert ext.validate_source() is False


# =========================================================================
# TestAPIExtractor
# =========================================================================

class TestAPIExtractor:
    """Tests for the APIExtractor concrete class."""

    def test_instantiation(self):
        config = DataSourceConfig(source_type="api", path="http://test")
        ext = APIExtractor(config)
        assert isinstance(ext, APIExtractor)

    def test_instantiation_with_auth(self):
        config = DataSourceConfig(source_type="api", path="http://test")
        ext = APIExtractor(config, auth_token="tok123", page_size=50)
        assert isinstance(ext, APIExtractor)
        assert ext._auth_token == "tok123"
        assert ext._page_size == 50

    def test_extract_returns_iterator(self):
        """extract() returns a generator (iterator) object."""
        config = DataSourceConfig(source_type="api", path="http://test")
        ext = APIExtractor(config)
        result = ext.extract()
        assert hasattr(result, '__iter__')
        assert hasattr(result, '__next__')

    def test_validate_source_unreachable(self):
        """validate_source returns False for unreachable endpoint."""
        config = DataSourceConfig(source_type="api", path="http://localhost:99999/fake")
        ext = APIExtractor(config)
        assert ext.validate_source() is False


# =========================================================================
# TestCleaningTransformer
# =========================================================================

class TestCleaningTransformer:
    """Tests for the CleaningTransformer concrete class."""

    def test_instantiation_defaults(self):
        ct = CleaningTransformer()
        assert isinstance(ct, CleaningTransformer)

    def test_instantiation_with_args(self):
        ct = CleaningTransformer(
            type_map={"gpa": "float"}, null_values=["NA", "N/A"]
        )
        assert isinstance(ct, CleaningTransformer)

    def test_transform_strips_whitespace(self):
        """transform() strips leading/trailing whitespace from strings."""
        ct = CleaningTransformer()
        result = ct.transform({"name": "  Alice  ", "gpa": "90"})
        assert result is not None
        assert result["name"] == "Alice"
        assert result["gpa"] == "90"

    def test_transform_null_values(self):
        """transform() converts null sentinel strings to None."""
        ct = CleaningTransformer()
        result = ct.transform({"name": "Alice", "phone": "NA"})
        assert result["name"] == "Alice"
        assert result["phone"] is None

    def test_transform_type_conversion(self):
        """transform() converts types according to type_map."""
        ct = CleaningTransformer(type_map={"gpa": "float", "year": "int"})
        result = ct.transform({"gpa": "87.5", "year": "2024"})
        assert result["gpa"] == 87.5
        assert result["year"] == 2024

    def test_clean_string_strips_and_nulls(self):
        """_clean_string strips whitespace and detects null values."""
        ct = CleaningTransformer()
        assert ct._clean_string("  hello  ") == "hello"
        assert ct._clean_string("NA") is None
        assert ct._clean_string(None) is None
        assert ct._clean_string(42) == "42"

    def test_convert_type_float(self):
        """_convert_type converts to float."""
        ct = CleaningTransformer()
        assert ct._convert_type("87.5", "float") == 87.5

    def test_convert_type_int(self):
        """_convert_type converts to int."""
        ct = CleaningTransformer()
        assert ct._convert_type("42", "int") == 42

    def test_convert_type_bool(self):
        """_convert_type converts to bool."""
        ct = CleaningTransformer()
        assert ct._convert_type("true", "bool") is True
        assert ct._convert_type("0", "bool") is False


# =========================================================================
# TestNormalizationTransformer
# =========================================================================

class TestNormalizationTransformer:
    """Tests for the NormalizationTransformer concrete class."""

    def test_instantiation_defaults(self):
        nt = NormalizationTransformer()
        assert isinstance(nt, NormalizationTransformer)

    def test_instantiation_with_mappings(self):
        mappings = {"university": {"UofT": "University of Toronto"}}
        nt = NormalizationTransformer(mappings=mappings)
        assert isinstance(nt, NormalizationTransformer)

    def test_transform_applies_mapping(self):
        """transform() applies field mappings to record values."""
        mappings = {"university": {"UofT": "University of Toronto"}}
        nt = NormalizationTransformer(mappings=mappings)
        result = nt.transform({"university": "UofT", "program": "CS"})
        assert result is not None
        assert result["university"] == "University of Toronto"
        assert result["program"] == "CS"

    def test_transform_no_mapping_keeps_original(self):
        """transform() keeps values unchanged if no mapping exists."""
        nt = NormalizationTransformer()
        record = {"university": "UofT"}
        result = nt.transform(record)
        assert result["university"] == "UofT"

    def test_normalize_field_exact_match(self):
        """_normalize_field matches exact values."""
        mappings = {"university": {"UofT": "University of Toronto"}}
        nt = NormalizationTransformer(mappings=mappings)
        assert nt._normalize_field("university", "UofT") == "University of Toronto"

    def test_normalize_field_case_insensitive(self):
        """_normalize_field falls back to case-insensitive match."""
        mappings = {"university": {"UofT": "University of Toronto"}}
        nt = NormalizationTransformer(mappings=mappings)
        assert nt._normalize_field("university", "uoft") == "University of Toronto"

    def test_normalize_field_no_match(self):
        """_normalize_field returns original value if no match."""
        nt = NormalizationTransformer()
        result = nt._normalize_field("university", "UofT")
        assert result == "UofT"


# =========================================================================
# TestEnrichmentTransformer
# =========================================================================

class TestEnrichmentTransformer:
    """Tests for the EnrichmentTransformer concrete class."""

    def test_instantiation_defaults(self):
        et = EnrichmentTransformer()
        assert isinstance(et, EnrichmentTransformer)

    def test_instantiation_with_data(self):
        et = EnrichmentTransformer(historical_data={"UofT_CS": 0.15})
        assert isinstance(et, EnrichmentTransformer)

    def test_transform_returns_enriched_record(self):
        """transform() returns a dict with original fields preserved."""
        et = EnrichmentTransformer()
        result = et.transform({"university": "UofT", "program": "CS"})
        assert result is not None
        assert result["university"] == "UofT"
        assert result["program"] == "CS"

    def test_add_historical_rate_adds_field(self):
        """_add_historical_rate adds historical_admit_rate when match found."""
        et = EnrichmentTransformer(historical_data={"UofT_CS": 0.15})
        record = {"university": "UofT", "program": "CS"}
        et._add_historical_rate(record)
        assert record["historical_admit_rate"] == 0.15

    def test_add_historical_rate_no_match(self):
        """_add_historical_rate does not add field when no match."""
        et = EnrichmentTransformer(historical_data={"UofT_CS": 0.15})
        record = {"university": "UBC", "program": "Math"}
        et._add_historical_rate(record)
        assert "historical_admit_rate" not in record

    def test_add_date_features_parses_date(self):
        """_add_date_features extracts month and year from application_date."""
        et = EnrichmentTransformer()
        record = {"application_date": "2024-01-15"}
        et._add_date_features(record)
        assert record["application_month"] == 1
        assert record["application_year"] == 2024

    def test_add_date_features_no_date(self):
        """_add_date_features does nothing when no application_date."""
        et = EnrichmentTransformer()
        record = {"university": "UofT"}
        et._add_date_features(record)
        assert "application_month" not in record


# =========================================================================
# TestValidationTransformer
# =========================================================================

class TestValidationTransformer:
    """Tests for the ValidationTransformer concrete class."""

    def test_instantiation_defaults(self):
        vt = ValidationTransformer()
        assert isinstance(vt, ValidationTransformer)

    def test_instantiation_strict(self):
        vt = ValidationTransformer(strict=True)
        assert isinstance(vt, ValidationTransformer)

    def test_instantiation_with_rules(self):
        vt = ValidationTransformer(rules={"required": ["student_id"]})
        assert isinstance(vt, ValidationTransformer)

    def test_transform_valid_record(self):
        """transform() returns record without errors for valid input."""
        vt = ValidationTransformer()
        result = vt.transform({
            "student_id": "S001", "university": "UofT",
            "program": "CS", "gpa": 85
        })
        assert result is not None
        assert "_validation_errors" not in result

    def test_transform_missing_required_field(self):
        """transform() adds _validation_errors for missing required fields."""
        vt = ValidationTransformer()
        result = vt.transform({"student_id": "S001"})
        assert result is not None
        assert "_validation_errors" in result
        assert any("university" in e for e in result["_validation_errors"])

    def test_transform_strict_rejects_invalid(self):
        """transform() in strict mode returns None for invalid records."""
        vt = ValidationTransformer(strict=True)
        result = vt.transform({"student_id": "S001"})
        assert result is None

    def test_validate_required_all_present(self):
        """_validate_required returns empty list when all required present."""
        vt = ValidationTransformer()
        errors = vt._validate_required({
            "student_id": "S001", "university": "UofT", "program": "CS"
        })
        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_validate_required_missing(self):
        """_validate_required returns errors for missing fields."""
        vt = ValidationTransformer()
        errors = vt._validate_required({"student_id": "S001"})
        assert len(errors) == 2  # missing university and program

    def test_validate_ranges_valid_gpa(self):
        """_validate_ranges returns no errors for valid GPA."""
        vt = ValidationTransformer()
        errors = vt._validate_ranges({"gpa": 85})
        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_validate_ranges_invalid_gpa(self):
        """_validate_ranges flags out-of-range GPA."""
        vt = ValidationTransformer()
        errors = vt._validate_ranges({"gpa": 150})
        assert len(errors) == 1
        assert "out of range" in errors[0]

    def test_validate_business_rules_valid_outcome(self):
        """_validate_business_rules returns no errors for valid outcome."""
        vt = ValidationTransformer()
        errors = vt._validate_business_rules({"outcome": "admitted"})
        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_validate_business_rules_invalid_outcome(self):
        """_validate_business_rules flags invalid outcome."""
        vt = ValidationTransformer()
        errors = vt._validate_business_rules({"outcome": "maybe"})
        assert len(errors) == 1
        assert "Invalid outcome" in errors[0]


# =========================================================================
# TestMongoLoader
# =========================================================================

class TestMongoLoader:
    """Tests for the MongoLoader concrete class."""

    def _make(self):
        return MongoLoader(None, "collection", ["id"])

    def test_instantiation(self):
        loader = self._make()
        assert isinstance(loader, MongoLoader)

    def test_load_with_none_client(self):
        """load() returns True when client is None (no-op)."""
        loader = self._make()
        result = loader.load({"id": "1", "name": "Alice"})
        assert result is True

    def test_load_batch_returns_count(self):
        """load_batch() returns number of records loaded."""
        loader = self._make()
        result = loader.load_batch([{"id": "1"}, {"id": "2"}])
        assert result == 2

    def test_exists_with_none_client(self):
        """exists() returns False when client is None."""
        loader = self._make()
        result = loader.exists({"id": "1"})
        assert result is False


# =========================================================================
# TestWeaviateLoader
# =========================================================================

class TestWeaviateLoader:
    """Tests for the WeaviateLoader concrete class."""

    def _make(self):
        return WeaviateLoader(None, "ClassName")

    def test_instantiation(self):
        loader = self._make()
        assert isinstance(loader, WeaviateLoader)

    def test_instantiation_with_kwargs(self):
        loader = WeaviateLoader(None, "ClassName", embedding_model=None,
                                id_field="uuid")
        assert isinstance(loader, WeaviateLoader)

    def test_load_with_none_client(self):
        """load() returns False when client is None (add_object fails)."""
        loader = self._make()
        result = loader.load({"id": "1", "text": "hello"})
        assert result is False

    def test_load_batch_with_none_client(self):
        """load_batch() returns 0 when all loads fail (None client)."""
        loader = self._make()
        result = loader.load_batch([{"id": "1"}, {"id": "2"}])
        assert result == 0

    def test_exists_with_none_client(self):
        """exists() returns False when client is None."""
        loader = self._make()
        result = loader.exists({"id": "1"})
        assert result is False


# =========================================================================
# TestETLPipeline
# =========================================================================

class TestETLPipeline:
    """Tests for the ETLPipeline orchestrator."""

    def _make(self):
        return ETLPipeline(ETLConfig())

    def test_instantiation(self):
        pipeline = self._make()
        assert isinstance(pipeline, ETLPipeline)

    def test_add_extractor_returns_self(self):
        """add_extractor returns pipeline for chaining."""
        pipeline = self._make()
        config = DataSourceConfig(source_type="csv", path="test.csv")
        ext = CSVExtractor(config)
        result = pipeline.add_extractor(ext)
        assert result is pipeline

    def test_add_transformer_returns_self(self):
        """add_transformer returns pipeline for chaining."""
        pipeline = self._make()
        transformer = CleaningTransformer()
        result = pipeline.add_transformer(transformer)
        assert result is pipeline

    def test_add_loader_returns_self(self):
        """add_loader returns pipeline for chaining."""
        pipeline = self._make()
        loader = MongoLoader(None, "col", ["id"])
        result = pipeline.add_loader(loader)
        assert result is pipeline

    def test_method_chaining(self):
        """Pipeline supports fluent method chaining."""
        pipeline = self._make()
        result = (pipeline
                  .add_transformer(CleaningTransformer())
                  .add_loader(MongoLoader(None, "col", ["id"])))
        assert result is pipeline

    def test_run_returns_stats(self, tmp_path):
        """run() returns ETLStats with processing counts."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob\n")
        config = DataSourceConfig(source_type="csv", path=str(csv_file))
        pipeline = self._make()
        pipeline.add_extractor(CSVExtractor(config))
        pipeline.add_loader(MongoLoader(None, "col", ["id"]))
        result = pipeline.run()
        assert isinstance(result, ETLStats)
        assert result.records_read == 2
        assert result.records_inserted == 2

    def test_run_no_extractors_returns_errors(self):
        """run() with no extractors returns stats with validation errors."""
        pipeline = self._make()
        pipeline.add_loader(MongoLoader(None, "col", ["id"]))
        result = pipeline.run()
        assert isinstance(result, ETLStats)
        assert len(result.errors) > 0

    def test_validate_no_components(self):
        """validate() returns invalid when no extractors or loaders."""
        pipeline = self._make()
        result = pipeline.validate()
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False

    def test_validate_complete_pipeline(self, tmp_path):
        """validate() returns valid for properly configured pipeline."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col\nval\n")
        config = DataSourceConfig(source_type="csv", path=str(csv_file))
        pipeline = self._make()
        pipeline.add_extractor(CSVExtractor(config))
        pipeline.add_loader(MongoLoader(None, "col", ["id"]))
        result = pipeline.validate()
        assert result.is_valid is True

    def test_process_batch_no_loaders(self):
        """_process_batch with no loaders returns (0, N) failures."""
        pipeline = self._make()
        success, fail = pipeline._process_batch([{"id": "1"}, {"id": "2"}])
        assert isinstance((success, fail), tuple)
        assert success == 0
        assert fail == 2

    def test_apply_transformers_no_transformers(self):
        """_apply_transformers with no transformers returns record unchanged."""
        pipeline = self._make()
        record = {"id": "1", "name": "Alice"}
        result = pipeline._apply_transformers(record)
        assert result == record

    def test_apply_transformers_chains(self):
        """_apply_transformers applies transformers in order."""
        pipeline = self._make()
        pipeline.add_transformer(CleaningTransformer())
        pipeline.add_transformer(
            NormalizationTransformer(
                mappings={"university": {"UofT": "University of Toronto"}}
            )
        )
        result = pipeline._apply_transformers({
            "university": "  UofT  ", "program": "CS"
        })
        assert result["university"] == "University of Toronto"


# =========================================================================
# TestUtilityFunctions
# =========================================================================

class TestUtilityFunctions:
    """Tests for the module-level utility functions."""

    def test_create_admission_etl_returns_pipeline(self):
        """create_admission_etl returns a configured ETLPipeline."""
        result = create_admission_etl(None)
        assert isinstance(result, ETLPipeline)

    def test_create_admission_etl_with_optional_args(self):
        """create_admission_etl accepts optional weaviate and embedding args."""
        result = create_admission_etl(None, weaviate_client=None,
                                       embedding_model=None)
        assert isinstance(result, ETLPipeline)

    def test_run_incremental_load_returns_stats(self):
        """run_incremental_load returns ETLStats."""
        pipeline = ETLPipeline(ETLConfig())
        pipeline.add_loader(MongoLoader(None, "col", ["id"]))
        result = run_incremental_load(pipeline, "2024-01-01")
        assert isinstance(result, ETLStats)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
