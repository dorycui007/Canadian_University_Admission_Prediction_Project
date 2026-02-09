"""
API Package — REST Prediction Service
=======================================

FastAPI-based REST endpoints for real-time admission predictions,
batch processing, and model introspection.

Modules:
    predictor - PredictorService, request/response types, app factory
"""

from .predictor import (
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
)

__all__ = [
    "PredictionLabel",
    "ConfidenceInterval",
    "FeatureImportance",
    "SimilarProgram",
    "ApplicationRequest",
    "PredictionResponse",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "ModelInfo",
    "HealthResponse",
    "PredictorService",
    "create_app",
]
