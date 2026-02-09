"""
Tests for src.api.predictor module.
=============================================

Covers enums, dataclasses (defaults and custom values), PredictorService
stub methods, endpoint function stubs, and utility function stubs.

Pattern: call-then-skip -- call each stub, if it returns None pytest.skip.
For dataclasses: real assertions on defaults.
"""

import pytest
import numpy as np

from src.api.predictor import (
    PredictionLabel,
    ConfidenceInterval,
    FeatureImportance,
    SimilarProgram,
    ApplicationRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    HealthResponse,
    PredictorService,
    create_app,
    predict_endpoint,
    predict_batch_endpoint,
    health_endpoint,
    model_info_endpoint,
    log_request_middleware,
    normalize_university_name,
    normalize_program_name,
    startup_event,
    shutdown_event,
    rate_limit_middleware,
)


# =========================================================================
# TestPredictionLabel
# =========================================================================

class TestPredictionLabel:
    """Tests for the PredictionLabel enum."""

    def test_likely_admit_value(self):
        assert PredictionLabel.LIKELY_ADMIT.value == "LIKELY_ADMIT"

    def test_uncertain_value(self):
        assert PredictionLabel.UNCERTAIN.value == "UNCERTAIN"

    def test_unlikely_admit_value(self):
        assert PredictionLabel.UNLIKELY_ADMIT.value == "UNLIKELY_ADMIT"

    def test_enum_members_count(self):
        assert len(PredictionLabel) == 3

    def test_enum_from_value(self):
        assert PredictionLabel("LIKELY_ADMIT") is PredictionLabel.LIKELY_ADMIT


# =========================================================================
# TestConfidenceInterval
# =========================================================================

class TestConfidenceInterval:
    """Tests for the ConfidenceInterval dataclass."""

    def test_default_method(self):
        ci = ConfidenceInterval(lower=0.5, upper=0.9)
        assert ci.method == "asymptotic"

    def test_custom_method(self):
        ci = ConfidenceInterval(lower=0.3, upper=0.7, method="bootstrap")
        assert ci.method == "bootstrap"

    def test_lower_stored(self):
        ci = ConfidenceInterval(lower=0.25, upper=0.75)
        assert ci.lower == 0.25

    def test_upper_stored(self):
        ci = ConfidenceInterval(lower=0.25, upper=0.75)
        assert ci.upper == 0.75

    def test_all_fields(self):
        ci = ConfidenceInterval(lower=0.1, upper=0.9, method="bayesian")
        assert ci.lower == 0.1
        assert ci.upper == 0.9
        assert ci.method == "bayesian"


# =========================================================================
# TestFeatureImportance
# =========================================================================

class TestFeatureImportance:
    """Tests for the FeatureImportance dataclass."""

    def test_feature_name_stored(self):
        fi = FeatureImportance(
            feature_name="gpa", value=90.0, coefficient=0.5,
            contribution=45.0, direction="+"
        )
        assert fi.feature_name == "gpa"

    def test_value_stored(self):
        fi = FeatureImportance(
            feature_name="gpa", value=90.0, coefficient=0.5,
            contribution=45.0, direction="+"
        )
        assert fi.value == 90.0

    def test_coefficient_stored(self):
        fi = FeatureImportance(
            feature_name="gpa", value=90.0, coefficient=0.5,
            contribution=45.0, direction="+"
        )
        assert fi.coefficient == 0.5

    def test_contribution_stored(self):
        fi = FeatureImportance(
            feature_name="gpa", value=90.0, coefficient=0.5,
            contribution=45.0, direction="+"
        )
        assert fi.contribution == 45.0

    def test_direction_stored(self):
        fi = FeatureImportance(
            feature_name="gpa", value=90.0, coefficient=0.5,
            contribution=45.0, direction="-"
        )
        assert fi.direction == "-"


# =========================================================================
# TestSimilarProgram
# =========================================================================

class TestSimilarProgram:
    """Tests for the SimilarProgram dataclass."""

    def test_default_historical_admit_rate(self):
        sp = SimilarProgram(university="UofT", program="CS", similarity=0.9)
        assert sp.historical_admit_rate is None

    def test_custom_historical_admit_rate(self):
        sp = SimilarProgram(
            university="UBC", program="Eng", similarity=0.8,
            historical_admit_rate=0.35
        )
        assert sp.historical_admit_rate == 0.35

    def test_university_stored(self):
        sp = SimilarProgram(university="McGill", program="Math", similarity=0.7)
        assert sp.university == "McGill"

    def test_program_stored(self):
        sp = SimilarProgram(university="McGill", program="Math", similarity=0.7)
        assert sp.program == "Math"

    def test_similarity_stored(self):
        sp = SimilarProgram(university="McGill", program="Math", similarity=0.7)
        assert sp.similarity == 0.7


# =========================================================================
# TestApplicationRequest
# =========================================================================

class TestApplicationRequest:
    """Tests for the ApplicationRequest dataclass."""

    def test_required_fields(self):
        req = ApplicationRequest(
            top_6_average=90.0, university="UofT", program="CS"
        )
        assert req.top_6_average == 90.0
        assert req.university == "UofT"
        assert req.program == "CS"

    def test_default_country(self):
        req = ApplicationRequest(
            top_6_average=85.0, university="McGill", program="Eng"
        )
        assert req.country == "Canada"

    def test_default_optional_none(self):
        req = ApplicationRequest(
            top_6_average=85.0, university="UBC", program="Science"
        )
        assert req.grade_11_average is None
        assert req.grade_12_average is None
        assert req.province is None
        assert req.application_year is None

    def test_custom_optional_fields(self):
        req = ApplicationRequest(
            top_6_average=92.0, university="Waterloo", program="SE",
            grade_11_average=88.0, grade_12_average=91.0,
            province="ON", country="Canada", application_year=2024
        )
        assert req.grade_11_average == 88.0
        assert req.grade_12_average == 91.0
        assert req.province == "ON"
        assert req.application_year == 2024

    def test_validate_valid_request(self):
        req = ApplicationRequest(
            top_6_average=90.0, university="UofT", program="CS"
        )
        result = req.validate()
        assert isinstance(result, list)
        assert result == []

    def test_validate_invalid_average(self):
        req = ApplicationRequest(
            top_6_average=110.0, university="UofT", program="CS"
        )
        result = req.validate()
        assert isinstance(result, list)
        assert len(result) > 0
        assert any("top_6_average" in e for e in result)

    def test_validate_empty_university(self):
        req = ApplicationRequest(
            top_6_average=90.0, university="  ", program="CS"
        )
        result = req.validate()
        assert len(result) > 0
        assert any("university" in e for e in result)

    def test_validate_invalid_province(self):
        req = ApplicationRequest(
            top_6_average=90.0, university="UofT", program="CS",
            province="XX"
        )
        result = req.validate()
        assert len(result) > 0
        assert any("province" in e.lower() or "Invalid" in e for e in result)


# =========================================================================
# TestPredictionResponse
# =========================================================================

class TestPredictionResponse:
    """Tests for the PredictionResponse dataclass."""

    def test_warnings_default_empty(self):
        ci = ConfidenceInterval(lower=0.5, upper=0.8)
        resp = PredictionResponse(
            probability=0.7, confidence_interval=ci,
            prediction="LIKELY_ADMIT", feature_importance=[],
            similar_programs=[], model_version="v1.0",
            timestamp="2024-01-01T00:00:00Z"
        )
        assert resp.warnings == []

    def test_calibration_note_default_none(self):
        ci = ConfidenceInterval(lower=0.5, upper=0.8)
        resp = PredictionResponse(
            probability=0.7, confidence_interval=ci,
            prediction="LIKELY_ADMIT", feature_importance=[],
            similar_programs=[], model_version="v1.0",
            timestamp="2024-01-01T00:00:00Z"
        )
        assert resp.calibration_note is None

    def test_all_fields_stored(self):
        ci = ConfidenceInterval(lower=0.6, upper=0.85)
        fi = FeatureImportance(
            feature_name="gpa", value=92.0, coefficient=0.5,
            contribution=46.0, direction="+"
        )
        sp = SimilarProgram(university="UBC", program="CS", similarity=0.88)
        resp = PredictionResponse(
            probability=0.73, confidence_interval=ci,
            prediction="LIKELY_ADMIT", feature_importance=[fi],
            similar_programs=[sp], model_version="v1.2.0",
            timestamp="2024-01-15T10:30:00Z",
            calibration_note="Good", warnings=["low sample"]
        )
        assert resp.probability == 0.73
        assert resp.confidence_interval is ci
        assert resp.prediction == "LIKELY_ADMIT"
        assert len(resp.feature_importance) == 1
        assert len(resp.similar_programs) == 1
        assert resp.model_version == "v1.2.0"
        assert resp.calibration_note == "Good"
        assert resp.warnings == ["low sample"]


# =========================================================================
# TestBatchPredictionRequest
# =========================================================================

class TestBatchPredictionRequest:
    """Tests for the BatchPredictionRequest dataclass."""

    def test_default_return_similar_programs(self):
        req = BatchPredictionRequest(applications=[])
        assert req.return_similar_programs is False

    def test_default_return_feature_importance(self):
        req = BatchPredictionRequest(applications=[])
        assert req.return_feature_importance is True

    def test_applications_stored(self):
        app = ApplicationRequest(
            top_6_average=90.0, university="UofT", program="CS"
        )
        req = BatchPredictionRequest(applications=[app])
        assert len(req.applications) == 1
        assert req.applications[0] is app


# =========================================================================
# TestBatchPredictionResponse
# =========================================================================

class TestBatchPredictionResponse:
    """Tests for the BatchPredictionResponse dataclass."""

    def test_fields_stored(self):
        resp = BatchPredictionResponse(
            predictions=[], total_count=0, success_count=0,
            error_count=0, errors=[], processing_time_ms=12.5
        )
        assert resp.predictions == []
        assert resp.total_count == 0
        assert resp.success_count == 0
        assert resp.error_count == 0
        assert resp.errors == []
        assert resp.processing_time_ms == 12.5


# =========================================================================
# TestModelInfo
# =========================================================================

class TestModelInfo:
    """Tests for the ModelInfo dataclass."""

    def test_fields_stored(self):
        info = ModelInfo(
            version="v1.2.0", training_date="2024-01-10",
            training_samples=50000, feature_count=42,
            universities_supported=25, programs_supported=150,
            metrics={"auc_roc": 0.85}, calibration_method="platt_scaling",
            embedding_dim=128
        )
        assert info.version == "v1.2.0"
        assert info.training_date == "2024-01-10"
        assert info.training_samples == 50000
        assert info.feature_count == 42
        assert info.universities_supported == 25
        assert info.programs_supported == 150
        assert info.metrics == {"auc_roc": 0.85}
        assert info.calibration_method == "platt_scaling"
        assert info.embedding_dim == 128


# =========================================================================
# TestHealthResponse
# =========================================================================

class TestHealthResponse:
    """Tests for the HealthResponse dataclass."""

    def test_fields_stored(self):
        hr = HealthResponse(
            status="healthy", model_loaded=True,
            database_connected=True,
            embedding_service_available=True,
            timestamp="2024-01-15T10:00:00Z",
            uptime_seconds=3600.0
        )
        assert hr.status == "healthy"
        assert hr.model_loaded is True
        assert hr.database_connected is True
        assert hr.embedding_service_available is True
        assert hr.timestamp == "2024-01-15T10:00:00Z"
        assert hr.uptime_seconds == 3600.0


# =========================================================================
# TestPredictorService
# =========================================================================

class TestPredictorService:
    """Tests for the PredictorService class stubs."""

    def _make_service(self):
        return PredictorService("model.pkl", "cal.pkl", "emb.pkl", "config.yaml")

    def test_init_stub(self):
        service = self._make_service()
        assert isinstance(service, PredictorService)

    def test_predict(self):
        service = self._make_service()
        request = ApplicationRequest(
            top_6_average=90.0, university="UofT", program="CS"
        )
        result = service.predict(request)
        assert isinstance(result, PredictionResponse)
        assert 0.0 <= result.probability <= 1.0
        assert result.prediction in ("LIKELY_ADMIT", "UNCERTAIN", "UNLIKELY_ADMIT")
        assert isinstance(result.confidence_interval, ConfidenceInterval)
        assert result.confidence_interval.lower <= result.probability
        assert result.confidence_interval.upper >= result.probability
        assert result.model_version == "v1.0.0"
        assert isinstance(result.feature_importance, list)
        assert isinstance(result.similar_programs, list)

    def test_predict_invalid_raises(self):
        service = self._make_service()
        request = ApplicationRequest(
            top_6_average=150.0, university="UofT", program="CS"
        )
        with pytest.raises(ValueError):
            service.predict(request)

    def test_predict_batch(self):
        service = self._make_service()
        app = ApplicationRequest(
            top_6_average=88.0, university="McGill", program="Eng"
        )
        batch_req = BatchPredictionRequest(applications=[app])
        result = service.predict_batch(batch_req)
        assert isinstance(result, BatchPredictionResponse)
        assert result.total_count == 1
        assert result.success_count == 1
        assert result.error_count == 0
        assert len(result.predictions) == 1
        assert result.processing_time_ms >= 0

    def test_predict_batch_with_error(self):
        service = self._make_service()
        valid_app = ApplicationRequest(
            top_6_average=88.0, university="McGill", program="Eng"
        )
        invalid_app = ApplicationRequest(
            top_6_average=200.0, university="McGill", program="Eng"
        )
        batch_req = BatchPredictionRequest(applications=[valid_app, invalid_app])
        result = service.predict_batch(batch_req)
        assert result.total_count == 2
        assert result.success_count == 1
        assert result.error_count == 1

    def test_build_features(self):
        service = self._make_service()
        request = ApplicationRequest(
            top_6_average=85.0, university="UBC", program="Science"
        )
        result = service.build_features(request)
        assert isinstance(result, np.ndarray)
        assert len(result) == 8
        assert result[0] == 1.0  # bias term
        assert result[1] == pytest.approx(0.85)  # 85/100

    def test_build_features_with_province(self):
        service = self._make_service()
        request = ApplicationRequest(
            top_6_average=90.0, university="UofT", program="CS",
            province="ON"
        )
        result = service.build_features(request)
        assert result[4] == 1.0   # is_ontario
        assert result[5] == 0.0   # is_bc

    def test_compute_raw_prediction(self):
        service = self._make_service()
        x = np.array([1.0, 0.9, 0.88, 0.91, 1.0, 0.0, 0.0, 0.0])
        result = service.compute_raw_prediction(x)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_calibrate(self):
        service = self._make_service()
        result = service.calibrate(0.6)
        assert isinstance(result, float)
        assert 0.001 <= result <= 0.999

    def test_calibrate_bounds(self):
        service = self._make_service()
        # Result must be clipped to [0.001, 0.999]
        low = service.calibrate(0.0)
        high = service.calibrate(1.0)
        assert low >= 0.001
        assert high <= 0.999

    def test_compute_confidence_interval(self):
        service = self._make_service()
        x = np.array([1.0, 0.9])
        result = service.compute_confidence_interval(x, 0.7)
        assert isinstance(result, ConfidenceInterval)
        assert result.lower <= 0.7
        assert result.upper >= 0.7
        assert result.lower >= 0.0
        assert result.upper <= 1.0
        assert result.method == "approximate"

    def test_explain_prediction(self):
        service = self._make_service()
        x = np.array([1.0, 0.9, 0.88, 0.91, 1.0, 0.0, 0.0, 0.0])
        request = ApplicationRequest(
            top_6_average=90.0, university="UofT", program="CS"
        )
        result = service.explain_prediction(x, request, top_k=5)
        assert isinstance(result, list)
        assert len(result) <= 5
        for fi in result:
            assert isinstance(fi, FeatureImportance)
            assert fi.direction in ("+", "-")
            assert fi.contribution == pytest.approx(fi.value * fi.coefficient)
        # Sorted by |contribution| descending
        contribs = [abs(fi.contribution) for fi in result]
        assert contribs == sorted(contribs, reverse=True)

    def test_find_similar_programs(self):
        service = self._make_service()
        result = service.find_similar_programs("UofT", "CS", top_k=5)
        assert isinstance(result, list)
        assert len(result) <= 5
        for sp in result:
            assert isinstance(sp, SimilarProgram)
            assert isinstance(sp.university, str)
            assert isinstance(sp.program, str)
            assert 0.0 <= sp.similarity <= 1.0

    def test_get_prediction_label_likely(self):
        service = self._make_service()
        assert service.get_prediction_label(0.8) == "LIKELY_ADMIT"
        assert service.get_prediction_label(0.70) == "LIKELY_ADMIT"

    def test_get_prediction_label_uncertain(self):
        service = self._make_service()
        assert service.get_prediction_label(0.5) == "UNCERTAIN"
        assert service.get_prediction_label(0.40) == "UNCERTAIN"
        assert service.get_prediction_label(0.69) == "UNCERTAIN"

    def test_get_prediction_label_unlikely(self):
        service = self._make_service()
        assert service.get_prediction_label(0.2) == "UNLIKELY_ADMIT"
        assert service.get_prediction_label(0.39) == "UNLIKELY_ADMIT"


# =========================================================================
# TestEndpointFunctions
# =========================================================================

class TestEndpointFunctions:
    """Tests for the FastAPI endpoint function stubs."""

    def _make_service(self):
        return PredictorService("m", "c", "e", "cfg")

    def test_create_app(self):
        service = self._make_service()
        result = create_app(service)
        assert result is not None
        assert isinstance(result, dict)
        assert "title" in result
        assert "predictor_service" in result
        assert result["predictor_service"] is service

    def test_predict_endpoint(self):
        service = self._make_service()
        request = ApplicationRequest(
            top_6_average=85.0, university="McGill", program="Eng"
        )
        result = predict_endpoint(request, service)
        assert isinstance(result, PredictionResponse)
        assert 0.0 <= result.probability <= 1.0

    def test_predict_batch_endpoint(self):
        service = self._make_service()
        app = ApplicationRequest(
            top_6_average=87.0, university="Waterloo", program="Math"
        )
        batch_req = BatchPredictionRequest(applications=[app])
        result = predict_batch_endpoint(batch_req, service)
        assert isinstance(result, BatchPredictionResponse)
        assert result.total_count == 1
        assert result.success_count == 1

    def test_health_endpoint(self):
        service = self._make_service()
        result = health_endpoint(service)
        assert isinstance(result, HealthResponse)
        assert result.status == "healthy"
        assert result.model_loaded is True
        assert result.database_connected is True

    def test_model_info_endpoint(self):
        service = self._make_service()
        result = model_info_endpoint(service)
        assert isinstance(result, ModelInfo)
        assert result.version == "v1.0.0"
        assert result.training_samples == 10000
        assert result.feature_count == 8

    def test_log_request_middleware(self):
        result = log_request_middleware("dummy_request", lambda r: r)
        assert result == "dummy_request"


# =========================================================================
# TestUtilityFunctions
# =========================================================================

class TestUtilityFunctions:
    """Tests for utility function stubs."""

    def test_normalize_university_name_known(self):
        assert normalize_university_name("uoft") == "University of Toronto"
        assert normalize_university_name("UofT") == "University of Toronto"
        assert normalize_university_name("ubc") == "University of British Columbia"
        assert normalize_university_name("mcgill") == "McGill University"
        assert normalize_university_name("waterloo") == "University of Waterloo"

    def test_normalize_university_name_unknown(self):
        result = normalize_university_name("UnknownSchool123")
        assert isinstance(result, str)
        # Unknown names are returned as-is
        assert result == "UnknownSchool123"

    def test_normalize_program_name_known(self):
        assert normalize_program_name("cs") == "Computer Science"
        assert normalize_program_name("CS") == "Computer Science"
        assert normalize_program_name("comp sci") == "Computer Science"
        assert normalize_program_name("eng") == "Engineering"
        assert normalize_program_name("math") == "Mathematics"

    def test_normalize_program_name_unknown(self):
        result = normalize_program_name("UnknownProgram123")
        assert isinstance(result, str)
        assert result == "UnknownProgram123"

    def test_startup_event(self):
        # startup_event prints and returns None
        result = startup_event()
        assert result is None

    def test_shutdown_event(self):
        # shutdown_event prints and returns None
        result = shutdown_event()
        assert result is None

    def test_rate_limit_middleware(self):
        result = rate_limit_middleware("dummy_request", lambda r: r)
        assert result == "dummy_request"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
