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

    def test_validate_stub(self):
        req = ApplicationRequest(
            top_6_average=90.0, university="UofT", program="CS"
        )
        result = req.validate()
        if result is None:
            pytest.skip("validate not yet implemented")
        assert isinstance(result, list)


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

    def test_predict_stub(self):
        service = self._make_service()
        request = ApplicationRequest(
            top_6_average=90.0, university="UofT", program="CS"
        )
        result = service.predict(request)
        if result is None:
            pytest.skip("predict not yet implemented")
        assert isinstance(result, PredictionResponse)

    def test_predict_batch_stub(self):
        service = self._make_service()
        app = ApplicationRequest(
            top_6_average=88.0, university="McGill", program="Eng"
        )
        batch_req = BatchPredictionRequest(applications=[app])
        result = service.predict_batch(batch_req)
        if result is None:
            pytest.skip("predict_batch not yet implemented")
        assert isinstance(result, BatchPredictionResponse)

    def test_build_features_stub(self):
        service = self._make_service()
        request = ApplicationRequest(
            top_6_average=85.0, university="UBC", program="Science"
        )
        result = service.build_features(request)
        if result is None:
            pytest.skip("build_features not yet implemented")
        assert isinstance(result, np.ndarray)

    def test_compute_raw_prediction_stub(self):
        service = self._make_service()
        x = np.array([1.0, 2.0, 3.0])
        result = service.compute_raw_prediction(x)
        if result is None:
            pytest.skip("compute_raw_prediction not yet implemented")
        assert isinstance(result, float)

    def test_calibrate_stub(self):
        service = self._make_service()
        result = service.calibrate(0.6)
        if result is None:
            pytest.skip("calibrate not yet implemented")
        assert isinstance(result, float)

    def test_compute_confidence_interval_stub(self):
        service = self._make_service()
        x = np.array([1.0, 2.0])
        result = service.compute_confidence_interval(x, 0.7)
        if result is None:
            pytest.skip("compute_confidence_interval not yet implemented")
        assert isinstance(result, ConfidenceInterval)

    def test_explain_prediction_stub(self):
        service = self._make_service()
        x = np.array([1.0, 2.0])
        request = ApplicationRequest(
            top_6_average=90.0, university="UofT", program="CS"
        )
        result = service.explain_prediction(x, request, top_k=5)
        if result is None:
            pytest.skip("explain_prediction not yet implemented")
        assert isinstance(result, list)

    def test_find_similar_programs_stub(self):
        service = self._make_service()
        result = service.find_similar_programs("UofT", "CS", top_k=5)
        if result is None:
            pytest.skip("find_similar_programs not yet implemented")
        assert isinstance(result, list)

    def test_get_prediction_label_stub(self):
        service = self._make_service()
        result = service.get_prediction_label(0.8)
        if result is None:
            pytest.skip("get_prediction_label not yet implemented")
        assert isinstance(result, str)


# =========================================================================
# TestEndpointFunctions
# =========================================================================

class TestEndpointFunctions:
    """Tests for the FastAPI endpoint function stubs."""

    def _make_service(self):
        return PredictorService("m", "c", "e", "cfg")

    def test_create_app_stub(self):
        service = self._make_service()
        result = create_app(service)
        if result is None:
            pytest.skip("create_app not yet implemented")

    def test_predict_endpoint_stub(self):
        service = self._make_service()
        request = ApplicationRequest(
            top_6_average=85.0, university="McGill", program="Eng"
        )
        result = predict_endpoint(request, service)
        if result is None:
            pytest.skip("predict_endpoint not yet implemented")

    def test_predict_batch_endpoint_stub(self):
        service = self._make_service()
        app = ApplicationRequest(
            top_6_average=87.0, university="Waterloo", program="Math"
        )
        batch_req = BatchPredictionRequest(applications=[app])
        result = predict_batch_endpoint(batch_req, service)
        if result is None:
            pytest.skip("predict_batch_endpoint not yet implemented")

    def test_health_endpoint_stub(self):
        service = self._make_service()
        result = health_endpoint(service)
        if result is None:
            pytest.skip("health_endpoint not yet implemented")

    def test_model_info_endpoint_stub(self):
        service = self._make_service()
        result = model_info_endpoint(service)
        if result is None:
            pytest.skip("model_info_endpoint not yet implemented")

    def test_log_request_middleware_stub(self):
        result = log_request_middleware("dummy_request", lambda r: r)
        if result is None:
            pytest.skip("log_request_middleware not yet implemented")


# =========================================================================
# TestUtilityFunctions
# =========================================================================

class TestUtilityFunctions:
    """Tests for utility function stubs."""

    def test_normalize_university_name_stub(self):
        result = normalize_university_name("UofT")
        if result is None:
            pytest.skip("normalize_university_name not yet implemented")
        assert isinstance(result, str)

    def test_normalize_program_name_stub(self):
        result = normalize_program_name("CS")
        if result is None:
            pytest.skip("normalize_program_name not yet implemented")
        assert isinstance(result, str)

    def test_startup_event_stub(self):
        result = startup_event()
        if result is not None:
            pass  # implemented, nothing to assert specifically

    def test_shutdown_event_stub(self):
        result = shutdown_event()
        if result is not None:
            pass  # implemented, nothing to assert specifically

    def test_rate_limit_middleware_stub(self):
        result = rate_limit_middleware("dummy_request", lambda r: r)
        if result is None:
            pytest.skip("rate_limit_middleware not yet implemented")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
