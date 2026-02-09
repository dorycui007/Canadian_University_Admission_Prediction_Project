"""
FastAPI Predictor API for Grade Prediction System
===================================================

This module provides REST API endpoints for the grade prediction system,
enabling real-time predictions and batch processing of applications.

==============================================================================
                    SYSTEM ARCHITECTURE - WHERE THIS MODULE FITS
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                        API LAYER ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         EXTERNAL CLIENTS                             │   │
│   │                                                                      │   │
│   │   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │   │
│   │   │  Web UI  │    │ Mobile   │    │ Batch    │    │ Testing  │      │   │
│   │   │  Client  │    │   App    │    │ Scripts  │    │  Tools   │      │   │
│   │   └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘      │   │
│   │        │               │               │               │            │   │
│   └────────┼───────────────┼───────────────┼───────────────┼────────────┘   │
│            │               │               │               │                │
│            ▼               ▼               ▼               ▼                │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    THIS MODULE (predictor.py)                        │   │
│   │                                                                      │   │
│   │   ┌───────────────────────────────────────────────────────────┐     │   │
│   │   │                    FastAPI Application                    │     │   │
│   │   ├───────────────────────────────────────────────────────────┤     │   │
│   │   │                                                           │     │   │
│   │   │  ENDPOINTS:                                               │     │   │
│   │   │  ┌─────────────────┐  ┌─────────────────────────────────┐│     │   │
│   │   │  │ POST /predict   │  │ POST /predict/batch             ││     │   │
│   │   │  │ Single app      │  │ Multiple apps                   ││     │   │
│   │   │  └─────────────────┘  └─────────────────────────────────┘│     │   │
│   │   │  ┌─────────────────┐  ┌─────────────────────────────────┐│     │   │
│   │   │  │ GET /health     │  │ GET /model/info                 ││     │   │
│   │   │  │ Service status  │  │ Model metadata                  ││     │   │
│   │   │  └─────────────────┘  └─────────────────────────────────┘│     │   │
│   │   │  ┌─────────────────┐  ┌─────────────────────────────────┐│     │   │
│   │   │  │ POST /explain   │  │ GET /universities               ││     │   │
│   │   │  │ Feature import  │  │ Available options               ││     │   │
│   │   │  └─────────────────┘  └─────────────────────────────────┘│     │   │
│   │   │                                                           │     │   │
│   │   └───────────────────────────────────────────────────────────┘     │   │
│   │                              │                                       │   │
│   └──────────────────────────────┼───────────────────────────────────────┘   │
│                                  │                                           │
│                                  ▼                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      PREDICTION PIPELINE                             │   │
│   │                                                                      │   │
│   │  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐  │   │
│   │  │ Validation │──▶│  Feature   │──▶│   Model    │──▶│ Calibration│  │   │
│   │  │  (Pydantic)│   │ Engineering│   │ Prediction │   │   (Platt)  │  │   │
│   │  └────────────┘   └────────────┘   └────────────┘   └────────────┘  │   │
│   │                         │                                            │   │
│   │                         ▼                                            │   │
│   │              ┌─────────────────────┐                                │   │
│   │              │  Embedding Lookup   │                                │   │
│   │              │  (Weaviate/Local)   │                                │   │
│   │              └─────────────────────┘                                │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

==============================================================================
                    REQUEST/RESPONSE FLOW
==============================================================================

    ┌────────────────────────────────────────────────────────────────────────┐
    │                    SINGLE PREDICTION FLOW                               │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │   1. CLIENT REQUEST                                                     │
    │   ──────────────────                                                    │
    │   POST /predict                                                         │
    │   {                                                                     │
    │     "top_6_average": 92.5,                                             │
    │     "grade_11_average": 88.0,                                          │
    │     "grade_12_average": 91.0,                                          │
    │     "university": "University of Toronto",                              │
    │     "program": "Computer Science",                                      │
    │     "province": "Ontario",                                              │
    │     "country": "Canada"                                                │
    │   }                                                                     │
    │                                                                         │
    │   2. VALIDATION (Pydantic)                                              │
    │   ─────────────────────────                                             │
    │   • Check required fields                                               │
    │   • Validate data types                                                 │
    │   • Verify value ranges (0 <= average <= 100)                          │
    │   • Normalize university/program names                                  │
    │                                                                         │
    │   3. FEATURE ENGINEERING                                                │
    │   ───────────────────────                                               │
    │   • Build design matrix row                                             │
    │   • Lookup embeddings (university, program)                            │
    │   • Create interaction features                                         │
    │                                                                         │
    │   4. MODEL PREDICTION                                                   │
    │   ────────────────────                                                  │
    │   • Compute raw logit: z = X @ beta                                    │
    │   • Apply sigmoid: p_raw = 1 / (1 + exp(-z))                           │
    │                                                                         │
    │   5. CALIBRATION (Platt Scaling)                                        │
    │   ──────────────────────────────                                        │
    │   • Apply calibration: p_cal = sigmoid(a * p_raw + b)                  │
    │   • Ensures probability is well-calibrated                              │
    │                                                                         │
    │   6. RESPONSE                                                           │
    │   ─────────                                                             │
    │   {                                                                     │
    │     "probability": 0.73,                                               │
    │     "confidence_interval": {                                           │
    │       "lower": 0.65,                                                   │
    │       "upper": 0.80                                                    │
    │     },                                                                 │
    │     "prediction": "LIKELY_ADMIT",                                      │
    │     "model_version": "v1.2.0",                                         │
    │     "timestamp": "2024-01-15T10:30:00Z"                                │
    │   }                                                                     │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

==============================================================================
                    DATA MODELS (Pydantic)
==============================================================================

    ┌────────────────────────────────────────────────────────────────────────┐
    │                    INPUT/OUTPUT SCHEMAS                                 │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │   ApplicationRequest                                                    │
    │   ────────────────────                                                  │
    │   ┌─────────────────────────────────────────────────────────────┐      │
    │   │ top_6_average: float      [0, 100]    Required              │      │
    │   │ grade_11_average: float   [0, 100]    Optional              │      │
    │   │ grade_12_average: float   [0, 100]    Optional              │      │
    │   │ university: str           Valid name  Required              │      │
    │   │ program: str              Valid name  Required              │      │
    │   │ province: str             CA province Optional              │      │
    │   │ country: str              ISO code    Optional (default: CA)│      │
    │   │ application_year: int     Year        Optional              │      │
    │   └─────────────────────────────────────────────────────────────┘      │
    │                                                                         │
    │   PredictionResponse                                                    │
    │   ────────────────────                                                  │
    │   ┌─────────────────────────────────────────────────────────────┐      │
    │   │ probability: float        [0, 1]      Admission prob        │      │
    │   │ confidence_interval: CI   Lower/Upper 95% CI                │      │
    │   │ prediction: str           Label       ADMIT/REJECT/UNCERTAIN│      │
    │   │ similar_programs: List    Top-5       Similar by embedding  │      │
    │   │ feature_importance: Dict  Top-5       Key factors           │      │
    │   │ model_version: str        Version     Model identifier      │      │
    │   │ timestamp: datetime       ISO8601     Request time          │      │
    │   │ calibration_note: str     Warning     If applicable         │      │
    │   └─────────────────────────────────────────────────────────────┘      │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

==============================================================================
                    ERROR HANDLING
==============================================================================

    ┌────────────────────────────────────────────────────────────────────────┐
    │                    HTTP ERROR CODES                                     │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │   400 Bad Request                                                       │
    │   ─────────────────                                                     │
    │   • Missing required fields                                             │
    │   • Invalid data types                                                  │
    │   • Values out of range (e.g., average > 100)                          │
    │   • Unknown university/program                                          │
    │                                                                         │
    │   Example:                                                              │
    │   {                                                                     │
    │     "detail": [                                                        │
    │       {                                                                 │
    │         "loc": ["body", "top_6_average"],                              │
    │         "msg": "value must be between 0 and 100",                      │
    │         "type": "value_error"                                          │
    │       }                                                                 │
    │     ]                                                                   │
    │   }                                                                     │
    │                                                                         │
    │   404 Not Found                                                         │
    │   ──────────────                                                        │
    │   • University not in database                                          │
    │   • Program not available at specified university                       │
    │                                                                         │
    │   503 Service Unavailable                                               │
    │   ─────────────────────────                                             │
    │   • Model not loaded                                                    │
    │   • Embedding database unavailable                                      │
    │   • Configuration error                                                 │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

==============================================================================
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import numpy as np

# Note: These would be actual imports in implementation
# from fastapi import FastAPI, HTTPException, Depends, status
# from pydantic import BaseModel, Field, validator
# from contextlib import asynccontextmanager


# =============================================================================
#                              ENUMS
# =============================================================================

class PredictionLabel(Enum):
    """
    Prediction category labels based on probability thresholds.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    PREDICTION THRESHOLDS                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Probability        Label              Meaning                          │
    │   ─────────────────────────────────────────────────────────             │
    │   p >= 0.70          LIKELY_ADMIT       Strong candidate                 │
    │   0.40 <= p < 0.70   UNCERTAIN          Could go either way             │
    │   p < 0.40           UNLIKELY_ADMIT     Weak candidate                   │
    │                                                                          │
    │   ┌────────────────────────────────────────────────────────────┐        │
    │   │    0       0.40              0.70                     1    │        │
    │   │    ├────────┼────────────────┼─────────────────────────┤   │        │
    │   │    │UNLIKELY│   UNCERTAIN    │     LIKELY_ADMIT       │   │        │
    │   │    └────────┴────────────────┴─────────────────────────┘   │        │
    │   └────────────────────────────────────────────────────────────┘        │
    │                                                                          │
    │   NOTE: These thresholds should be tuned based on:                      │
    │   • Historical admission rates                                          │
    │   • Business requirements (false positive/negative costs)               │
    │   • User expectations                                                   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    LIKELY_ADMIT = "LIKELY_ADMIT"
    UNCERTAIN = "UNCERTAIN"
    UNLIKELY_ADMIT = "UNLIKELY_ADMIT"


# =============================================================================
#                         REQUEST/RESPONSE MODELS
# =============================================================================

@dataclass
class ConfidenceInterval:
    """
    Confidence interval for probability estimate.

    Computed using one of:
    1. Bootstrap (resampling training data)
    2. Asymptotic formula from logistic regression
    3. Posterior predictive from Bayesian approach

    Attributes:
        lower: Lower bound of 95% CI
        upper: Upper bound of 95% CI
        method: How CI was computed ("bootstrap", "asymptotic", "bayesian")
    """
    lower: float
    upper: float
    method: str = "asymptotic"


@dataclass
class FeatureImportance:
    """
    Feature contribution to the prediction.

    Attributes:
        feature_name: Human-readable feature name
        value: The actual feature value for this application
        coefficient: Model coefficient for this feature
        contribution: value * coefficient (contribution to log-odds)
        direction: "+" for positive, "-" for negative contribution
    """
    feature_name: str
    value: float
    coefficient: float
    contribution: float
    direction: str


@dataclass
class SimilarProgram:
    """
    A similar program based on embedding similarity.

    Attributes:
        university: University name
        program: Program name
        similarity: Cosine similarity score (0 to 1)
        historical_admit_rate: Past admission rate for this program
    """
    university: str
    program: str
    similarity: float
    historical_admit_rate: Optional[float] = None


@dataclass
class ApplicationRequest:
    """
    Request model for single application prediction.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     APPLICATION REQUEST                                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   REQUIRED FIELDS:                                                       │
    │   ─────────────────                                                      │
    │   • top_6_average: Student's top 6 course average                       │
    │   • university: Target university name                                   │
    │   • program: Target program name                                         │
    │                                                                          │
    │   OPTIONAL FIELDS (improve prediction accuracy):                         │
    │   ────────────────────────────────────────────────                       │
    │   • grade_11_average: Grade 11 average                                  │
    │   • grade_12_average: Grade 12 average                                  │
    │   • province: Student's province (location-based patterns)              │
    │   • country: Student's country (default: Canada)                        │
    │   • application_year: Year of application                               │
    │                                                                          │
    │   VALIDATION RULES:                                                      │
    │   ──────────────────                                                     │
    │   1. Averages must be in [0, 100]                                       │
    │   2. University/program must exist in database                          │
    │   3. Province must be valid Canadian province code                      │
    │                                                                          │
    │   EXAMPLE:                                                               │
    │   ─────────                                                              │
    │   {                                                                      │
    │     "top_6_average": 92.5,                                              │
    │     "university": "University of Toronto",                               │
    │     "program": "Computer Science",                                       │
    │     "grade_11_average": 88.0,                                           │
    │     "province": "ON"                                                    │
    │   }                                                                      │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Representation Invariants:
        - 0 <= top_6_average <= 100
        - 0 <= grade_11_average <= 100 (if provided)
        - 0 <= grade_12_average <= 100 (if provided)
        - university is a non-empty string
        - program is a non-empty string
    """
    top_6_average: float
    university: str
    program: str
    grade_11_average: Optional[float] = None
    grade_12_average: Optional[float] = None
    province: Optional[str] = None
    country: str = "Canada"
    application_year: Optional[int] = None

    def validate(self) -> List[str]:
        """
        Validate request fields.

        Returns:
            List of validation error messages (empty if valid)

        Implementation:
            1. Check average ranges (0-100)
            2. Check required fields are non-empty
            3. Validate province code if provided
            4. Return list of error messages
        """
        errors = []
        # Check averages in range
        for field_name, value in [('top_6_average', self.top_6_average),
                                   ('grade_11_average', self.grade_11_average),
                                   ('grade_12_average', self.grade_12_average)]:
            if value is not None and not (0 <= value <= 100):
                errors.append(f"{field_name} must be between 0 and 100")
        # Check required non-empty strings
        if not self.university or not self.university.strip():
            errors.append("university is required")
        if not self.program or not self.program.strip():
            errors.append("program is required")
        # Validate province if provided
        valid_provinces = {'ON', 'BC', 'AB', 'QC', 'MB', 'SK', 'NS', 'NB', 'NL', 'PE', 'NT', 'YT', 'NU'}
        if self.province and self.province not in valid_provinces:
            errors.append(f"Invalid province: {self.province}")
        return errors


@dataclass
class PredictionResponse:
    """
    Response model for prediction endpoint.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     PREDICTION RESPONSE                                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   CORE PREDICTION:                                                       │
    │   ─────────────────                                                      │
    │   probability: 0.73                                                      │
    │       The calibrated probability of admission.                           │
    │       Range: [0, 1]                                                      │
    │                                                                          │
    │   confidence_interval:                                                   │
    │       lower: 0.65                                                        │
    │       upper: 0.80                                                        │
    │       95% confidence interval for the probability.                       │
    │                                                                          │
    │   prediction: "LIKELY_ADMIT"                                            │
    │       Human-readable category based on threshold.                        │
    │                                                                          │
    │   INTERPRETABILITY:                                                      │
    │   ──────────────────                                                     │
    │   feature_importance: [                                                 │
    │       {"feature": "top_6_average", "contribution": +0.8},               │
    │       {"feature": "is_uoft", "contribution": -0.3},                     │
    │       ...                                                               │
    │   ]                                                                      │
    │       Top features affecting the prediction.                             │
    │                                                                          │
    │   similar_programs: [                                                   │
    │       {"university": "Waterloo", "program": "CS", "sim": 0.92},        │
    │       ...                                                               │
    │   ]                                                                      │
    │       Programs with similar admission patterns (via embeddings).         │
    │                                                                          │
    │   METADATA:                                                              │
    │   ──────────                                                             │
    │   model_version: "v1.2.0"                                               │
    │   timestamp: "2024-01-15T10:30:00Z"                                     │
    │   calibration_note: "Well-calibrated for Ontario applicants"            │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    probability: float
    confidence_interval: ConfidenceInterval
    prediction: str
    feature_importance: List[FeatureImportance]
    similar_programs: List[SimilarProgram]
    model_version: str
    timestamp: str
    calibration_note: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class BatchPredictionRequest:
    """
    Request model for batch predictions.

    Attributes:
        applications: List of application requests
        return_similar_programs: Whether to include similar programs (slower)
        return_feature_importance: Whether to include feature importance
    """
    applications: List[ApplicationRequest]
    return_similar_programs: bool = False
    return_feature_importance: bool = True


@dataclass
class BatchPredictionResponse:
    """
    Response model for batch predictions.

    Attributes:
        predictions: List of prediction responses
        total_count: Number of predictions
        success_count: Number of successful predictions
        error_count: Number of failed predictions
        errors: List of error details
        processing_time_ms: Total processing time in milliseconds
    """
    predictions: List[PredictionResponse]
    total_count: int
    success_count: int
    error_count: int
    errors: List[Dict[str, Any]]
    processing_time_ms: float


@dataclass
class ModelInfo:
    """
    Model metadata for /model/info endpoint.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        MODEL INFO                                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   version: "v1.2.0"                                                     │
    │   training_date: "2024-01-10"                                           │
    │   training_samples: 50000                                               │
    │   feature_count: 42                                                     │
    │   universities_supported: 25                                            │
    │   programs_supported: 150                                               │
    │   metrics:                                                              │
    │       auc_roc: 0.85                                                    │
    │       brier_score: 0.12                                                │
    │       ece: 0.03                                                        │
    │   calibration_method: "platt_scaling"                                  │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    version: str
    training_date: str
    training_samples: int
    feature_count: int
    universities_supported: int
    programs_supported: int
    metrics: Dict[str, float]
    calibration_method: str
    embedding_dim: int


@dataclass
class HealthResponse:
    """
    Response model for /health endpoint.
    """
    status: str  # "healthy", "degraded", "unhealthy"
    model_loaded: bool
    database_connected: bool
    embedding_service_available: bool
    timestamp: str
    uptime_seconds: float


# =============================================================================
#                         PREDICTOR SERVICE
# =============================================================================

class PredictorService:
    """
    Core prediction service that coordinates model inference.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     PREDICTOR SERVICE                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   This class encapsulates the prediction logic:                         │
    │   1. Load and manage model artifacts                                    │
    │   2. Feature engineering                                                │
    │   3. Prediction with uncertainty                                        │
    │   4. Calibration                                                        │
    │   5. Explanation generation                                             │
    │                                                                          │
    │   ARCHITECTURE:                                                          │
    │   ──────────────                                                         │
    │                                                                          │
    │   PredictorService                                                       │
    │   ├── model: LogisticRegression                                         │
    │   │   └── coefficients (β)                                              │
    │   ├── calibrator: PlattScaler                                           │
    │   │   └── calibration params (a, b)                                     │
    │   ├── embedder: EmbeddingLookup                                         │
    │   │   └── university/program embeddings                                 │
    │   ├── feature_builder: DesignMatrixBuilder                              │
    │   │   └── feature engineering logic                                     │
    │   └── config: PredictorConfig                                           │
    │       └── thresholds, settings                                          │
    │                                                                          │
    │   THREAD SAFETY:                                                         │
    │   ───────────────                                                        │
    │   This service is stateless after initialization.                        │
    │   Safe to use with FastAPI's async handlers.                            │
    │   Model coefficients are read-only.                                     │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Attributes:
        model: The trained logistic regression model
        calibrator: Platt scaling calibrator
        embedder: Embedding lookup service
        feature_builder: Design matrix builder
        config: Predictor configuration
        model_info: Model metadata
    """

    def __init__(self,
                 model_path: str,
                 calibrator_path: str,
                 embedding_path: str,
                 config_path: str):
        """
        Initialize predictor service.

        Args:
            model_path: Path to serialized model
            calibrator_path: Path to calibrator parameters
            embedding_path: Path to embedding vectors
            config_path: Path to configuration file

        Implementation:
            1. Load model coefficients from model_path
            2. Load calibration parameters (a, b)
            3. Initialize embedding lookup
            4. Load configuration
            5. Build feature name mapping
        """
        self._model_path = model_path
        self._calibrator_path = calibrator_path
        self._embedding_path = embedding_path
        self._config_path = config_path
        self._feature_names = [
            'bias', 'top_6_average', 'grade_11_average', 'grade_12_average',
            'is_ontario', 'is_bc', 'is_alberta', 'is_quebec'
        ]
        self._beta = np.array([-0.5, 0.08, 0.02, 0.03, 0.1, 0.05, 0.05, 0.08])
        self._cal_a = -1.0
        self._cal_b = 0.0
        self._model_info = ModelInfo(
            version="v1.0.0",
            training_date="2024-01-01",
            training_samples=10000,
            feature_count=8,
            universities_supported=25,
            programs_supported=150,
            metrics={"auc_roc": 0.82, "brier_score": 0.15},
            calibration_method="platt_scaling",
            embedding_dim=128
        )
        self._start_time = datetime.now()

    def predict(self, request: ApplicationRequest) -> PredictionResponse:
        """
        Make prediction for a single application.

        ┌─────────────────────────────────────────────────────────────┐
        │                  PREDICTION PIPELINE                         │
        ├─────────────────────────────────────────────────────────────┤
        │                                                              │
        │  ApplicationRequest                                          │
        │        │                                                     │
        │        ▼                                                     │
        │  ┌──────────────┐                                           │
        │  │   Validate   │◄── Check ranges, required fields          │
        │  └──────┬───────┘                                           │
        │         │                                                    │
        │         ▼                                                    │
        │  ┌──────────────┐                                           │
        │  │   Build X    │◄── Feature engineering                    │
        │  └──────┬───────┘                                           │
        │         │                                                    │
        │         ▼                                                    │
        │  ┌──────────────┐                                           │
        │  │ Raw Predict  │◄── z = X @ β, p = sigmoid(z)             │
        │  └──────┬───────┘                                           │
        │         │                                                    │
        │         ▼                                                    │
        │  ┌──────────────┐                                           │
        │  │  Calibrate   │◄── p_cal = sigmoid(a*p + b)              │
        │  └──────┬───────┘                                           │
        │         │                                                    │
        │         ▼                                                    │
        │  ┌──────────────┐                                           │
        │  │ Compute CI   │◄── Bootstrap or asymptotic                │
        │  └──────┬───────┘                                           │
        │         │                                                    │
        │         ▼                                                    │
        │  ┌──────────────┐                                           │
        │  │   Explain    │◄── Top-k features, similar programs       │
        │  └──────┬───────┘                                           │
        │         │                                                    │
        │         ▼                                                    │
        │  PredictionResponse                                          │
        │                                                              │
        └─────────────────────────────────────────────────────────────┘

        Args:
            request: Application data

        Returns:
            PredictionResponse with probability and metadata

        Raises:
            ValueError: If request validation fails
            RuntimeError: If model is not loaded
        """
        errors = request.validate()
        if errors:
            raise ValueError("; ".join(errors))
        x = self.build_features(request)
        p_raw = self.compute_raw_prediction(x)
        p_cal = self.calibrate(p_raw)
        ci = self.compute_confidence_interval(x, p_cal)
        label = self.get_prediction_label(p_cal)
        importance = self.explain_prediction(x, request)
        similar = self.find_similar_programs(request.university, request.program)
        return PredictionResponse(
            probability=round(p_cal, 4),
            confidence_interval=ci,
            prediction=label,
            feature_importance=importance,
            similar_programs=similar,
            model_version=self._model_info.version,
            timestamp=datetime.now().isoformat()
        )

    def predict_batch(self, request: BatchPredictionRequest) -> BatchPredictionResponse:
        """
        Make predictions for multiple applications.

        ┌─────────────────────────────────────────────────────────────┐
        │                  BATCH PREDICTION                            │
        ├─────────────────────────────────────────────────────────────┤
        │                                                              │
        │  For efficiency, batch prediction:                          │
        │  1. Validates all requests upfront                          │
        │  2. Builds feature matrix X for all at once                 │
        │  3. Single matrix multiply: Z = X @ β                       │
        │  4. Vectorized sigmoid and calibration                      │
        │  5. Parallelizes explanation generation                     │
        │                                                              │
        │  PERFORMANCE:                                                │
        │  ─────────────                                               │
        │  • Single prediction: ~10ms                                 │
        │  • Batch of 100: ~50ms (5x speedup per item)                │
        │  • Batch of 1000: ~200ms (50x speedup per item)             │
        │                                                              │
        │  This is because matrix multiplication is vectorized        │
        │  (numpy/BLAS operations).                                   │
        │                                                              │
        └─────────────────────────────────────────────────────────────┘

        Args:
            request: Batch request with list of applications

        Returns:
            BatchPredictionResponse with all predictions
        """
        import time
        start = time.time()
        predictions, errors = [], []
        for i, app in enumerate(request.applications):
            try:
                pred = self.predict(app)
                predictions.append(pred)
            except Exception as e:
                errors.append({"index": i, "error": str(e)})
        elapsed = (time.time() - start) * 1000
        return BatchPredictionResponse(
            predictions=predictions, total_count=len(request.applications),
            success_count=len(predictions), error_count=len(errors),
            errors=errors, processing_time_ms=round(elapsed, 2)
        )

    def build_features(self, request: ApplicationRequest) -> np.ndarray:
        """
        Build feature vector from application request.

        ┌─────────────────────────────────────────────────────────────┐
        │                  FEATURE ENGINEERING                         │
        ├─────────────────────────────────────────────────────────────┤
        │                                                              │
        │  Input: ApplicationRequest                                   │
        │                                                              │
        │  Output: Feature vector x ∈ R^d                             │
        │                                                              │
        │  FEATURE GROUPS:                                             │
        │  ────────────────                                            │
        │                                                              │
        │  1. Numerical (3 features):                                  │
        │     [top_6_avg, g11_avg, g12_avg]                           │
        │     Standardized: (x - μ) / σ                               │
        │                                                              │
        │  2. University embedding (d_univ features):                  │
        │     Lookup: embeddings[university]                          │
        │     Or: one-hot if embedding not found                      │
        │                                                              │
        │  3. Program embedding (d_prog features):                     │
        │     Lookup: embeddings[program]                             │
        │     Or: one-hot if embedding not found                      │
        │                                                              │
        │  4. Categorical (variable):                                  │
        │     Province: one-hot encoded                               │
        │     Year: normalized to [0, 1]                              │
        │                                                              │
        │  5. Interaction features:                                    │
        │     avg × is_competitive_program                            │
        │     avg × is_ontario                                        │
        │                                                              │
        │  6. Bias term:                                               │
        │     x[0] = 1 (intercept)                                    │
        │                                                              │
        └─────────────────────────────────────────────────────────────┘

        Args:
            request: Application request

        Returns:
            Feature vector as numpy array
        """
        province = request.province or ''
        features = [
            1.0,  # bias
            request.top_6_average / 100.0,  # normalized
            (request.grade_11_average or request.top_6_average) / 100.0,
            (request.grade_12_average or request.top_6_average) / 100.0,
            1.0 if province == 'ON' else 0.0,
            1.0 if province == 'BC' else 0.0,
            1.0 if province == 'AB' else 0.0,
            1.0 if province == 'QC' else 0.0,
        ]
        return np.array(features)

    def compute_raw_prediction(self, x: np.ndarray) -> float:
        """
        Compute raw (uncalibrated) prediction.

        MATHEMATICAL STEPS:
            1. z = x @ β (linear combination)
            2. p = 1 / (1 + exp(-z)) (sigmoid)

        Args:
            x: Feature vector

        Returns:
            Raw probability in [0, 1]
        """
        beta = self._beta[:len(x)] if len(x) < len(self._beta) else self._beta
        x_use = x[:len(beta)]
        z = float(np.dot(x_use, beta))
        p = 1.0 / (1.0 + np.exp(-z))
        return float(p)

    def calibrate(self, p_raw: float) -> float:
        """
        Apply Platt scaling calibration.

        ┌─────────────────────────────────────────────────────────────┐
        │                  PLATT SCALING                               │
        ├─────────────────────────────────────────────────────────────┤
        │                                                              │
        │  Platt scaling fits a sigmoid to map raw scores to          │
        │  calibrated probabilities:                                   │
        │                                                              │
        │      p_calibrated = 1 / (1 + exp(a * p_raw + b))            │
        │                                                              │
        │  Parameters (a, b) are fitted on validation set by          │
        │  minimizing cross-entropy loss.                              │
        │                                                              │
        │  TYPICAL VALUES:                                             │
        │  ────────────────                                            │
        │  • a ≈ 1.0: No scaling needed (already calibrated)         │
        │  • a > 1.0: Raw predictions too uncertain                   │
        │  • a < 1.0: Raw predictions too confident                   │
        │  • b < 0:   Shift predictions upward                        │
        │  • b > 0:   Shift predictions downward                      │
        │                                                              │
        └─────────────────────────────────────────────────────────────┘

        Args:
            p_raw: Raw probability from model

        Returns:
            Calibrated probability
        """
        z = self._cal_a * p_raw + self._cal_b
        p_cal = 1.0 / (1.0 + np.exp(-z))
        return float(np.clip(p_cal, 0.001, 0.999))

    def compute_confidence_interval(self,
                                    x: np.ndarray,
                                    p: float) -> ConfidenceInterval:
        """
        Compute confidence interval for probability.

        ┌─────────────────────────────────────────────────────────────┐
        │              CONFIDENCE INTERVAL METHODS                     │
        ├─────────────────────────────────────────────────────────────┤
        │                                                              │
        │  METHOD 1: Asymptotic (Delta Method)                         │
        │  ────────────────────────────────────                        │
        │  For logistic regression:                                    │
        │                                                              │
        │  Var(z) = x^T (X^T W X)^{-1} x                              │
        │  where W = diag(p(1-p))                                     │
        │                                                              │
        │  SE(z) = sqrt(Var(z))                                       │
        │  z_lower = z - 1.96 * SE(z)                                 │
        │  z_upper = z + 1.96 * SE(z)                                 │
        │  p_lower = sigmoid(z_lower)                                 │
        │  p_upper = sigmoid(z_upper)                                 │
        │                                                              │
        │  METHOD 2: Bootstrap                                         │
        │  ────────────────────                                        │
        │  1. Draw B bootstrap samples from training data             │
        │  2. Fit model on each sample                                │
        │  3. Predict with each model                                 │
        │  4. Take 2.5% and 97.5% percentiles                         │
        │                                                              │
        │  Trade-off:                                                  │
        │  • Asymptotic: Fast, assumes large sample                   │
        │  • Bootstrap: Slower, but more robust                       │
        │                                                              │
        └─────────────────────────────────────────────────────────────┘

        Args:
            x: Feature vector
            p: Point estimate of probability

        Returns:
            ConfidenceInterval with lower and upper bounds
        """
        se = np.sqrt(p * (1 - p) / max(self._model_info.training_samples, 1))
        z_crit = 1.96
        lower = max(0.0, p - z_crit * se)
        upper = min(1.0, p + z_crit * se)
        return ConfidenceInterval(lower=round(lower, 4), upper=round(upper, 4), method="asymptotic")

    def explain_prediction(self,
                          x: np.ndarray,
                          request: ApplicationRequest,
                          top_k: int = 5) -> List[FeatureImportance]:
        """
        Generate feature importance explanation.

        ┌─────────────────────────────────────────────────────────────┐
        │                  FEATURE IMPORTANCE                          │
        ├─────────────────────────────────────────────────────────────┤
        │                                                              │
        │  For logistic regression, contribution of feature j:        │
        │                                                              │
        │      contribution_j = x_j × β_j                             │
        │                                                              │
        │  This is the additive contribution to the log-odds.         │
        │                                                              │
        │  EXAMPLE:                                                    │
        │  ─────────                                                   │
        │  If β_gpa = 0.5 and x_gpa = 2 (standardized):              │
        │  contribution = 0.5 × 2 = 1.0                               │
        │                                                              │
        │  This means having GPA 2 std devs above mean               │
        │  adds 1.0 to log-odds (≈ 2.7x odds multiplier)             │
        │                                                              │
        │  RANKING:                                                    │
        │  ─────────                                                   │
        │  Sort by |contribution| and return top k.                   │
        │                                                              │
        └─────────────────────────────────────────────────────────────┘

        Args:
            x: Feature vector
            request: Original application request
            top_k: Number of top features to return

        Returns:
            List of FeatureImportance objects
        """
        contributions = []
        for i, (name, xi, bi) in enumerate(zip(self._feature_names, x, self._beta)):
            contrib = float(xi * bi)
            contributions.append(FeatureImportance(
                feature_name=name, value=float(xi), coefficient=float(bi),
                contribution=contrib, direction="+" if contrib >= 0 else "-"
            ))
        contributions.sort(key=lambda fi: abs(fi.contribution), reverse=True)
        return contributions[:top_k]

    def find_similar_programs(self,
                              university: str,
                              program: str,
                              top_k: int = 5) -> List[SimilarProgram]:
        """
        Find similar programs using embedding similarity.

        ┌─────────────────────────────────────────────────────────────┐
        │                EMBEDDING SIMILARITY                          │
        ├─────────────────────────────────────────────────────────────┤
        │                                                              │
        │  Given query program embedding q:                           │
        │                                                              │
        │  1. Compute cosine similarity to all programs:              │
        │     sim(q, p_i) = (q · p_i) / (||q|| ||p_i||)              │
        │                                                              │
        │  2. Sort by similarity, exclude query itself                │
        │                                                              │
        │  3. Return top-k with metadata                              │
        │                                                              │
        │  WHY THIS IS USEFUL:                                         │
        │  ─────────────────────                                       │
        │  • Shows what the model considers "similar"                 │
        │  • Helps users explore alternatives                         │
        │  • Validates embedding quality                              │
        │                                                              │
        └─────────────────────────────────────────────────────────────┘

        Args:
            university: University name
            program: Program name
            top_k: Number of similar programs to return

        Returns:
            List of SimilarProgram objects
        """
        return [
            SimilarProgram(university="University of Waterloo", program=program, similarity=0.85),
            SimilarProgram(university="McGill University", program=program, similarity=0.78),
        ][:top_k]

    def get_prediction_label(self, probability: float) -> str:
        """
        Convert probability to categorical label.

        Args:
            probability: Calibrated probability

        Returns:
            One of: "LIKELY_ADMIT", "UNCERTAIN", "UNLIKELY_ADMIT"
        """
        if probability >= 0.70:
            return PredictionLabel.LIKELY_ADMIT.value
        elif probability >= 0.40:
            return PredictionLabel.UNCERTAIN.value
        else:
            return PredictionLabel.UNLIKELY_ADMIT.value


# =============================================================================
#                         FastAPI APPLICATION
# =============================================================================

def create_app(predictor_service: PredictorService) -> Any:
    """
    Create FastAPI application with endpoints.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     API ENDPOINTS                                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   POST /predict                                                          │
    │   ─────────────                                                          │
    │   Single application prediction.                                         │
    │   Request: ApplicationRequest                                            │
    │   Response: PredictionResponse                                           │
    │                                                                          │
    │   POST /predict/batch                                                    │
    │   ────────────────────                                                   │
    │   Batch predictions (up to 1000 at once).                               │
    │   Request: BatchPredictionRequest                                        │
    │   Response: BatchPredictionResponse                                      │
    │                                                                          │
    │   POST /explain                                                          │
    │   ──────────────                                                         │
    │   Get detailed explanation for a prediction.                             │
    │   Request: ApplicationRequest                                            │
    │   Response: ExplanationResponse (extended feature importance)            │
    │                                                                          │
    │   GET /health                                                            │
    │   ────────────                                                           │
    │   Health check endpoint.                                                 │
    │   Response: HealthResponse                                               │
    │                                                                          │
    │   GET /model/info                                                        │
    │   ────────────────                                                       │
    │   Model metadata and metrics.                                            │
    │   Response: ModelInfo                                                    │
    │                                                                          │
    │   GET /universities                                                      │
    │   ──────────────────                                                     │
    │   List of supported universities.                                        │
    │   Response: List[str]                                                    │
    │                                                                          │
    │   GET /programs                                                          │
    │   ──────────────                                                         │
    │   List of programs, optionally filtered by university.                  │
    │   Query: ?university=<name>                                             │
    │   Response: List[str]                                                    │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        predictor_service: Initialized predictor service

    Returns:
        FastAPI application instance

    IMPLEMENTATION:
    ────────────────
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="Grade Prediction API",
        version="1.0.0",
        description="Predict university admission probabilities"
    )

    @app.post("/predict")
    async def predict(request: ApplicationRequest) -> PredictionResponse:
        try:
            return predictor_service.predict(request)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # ... other endpoints
    """
    app_config = {
        "title": "Grade Prediction API",
        "version": "1.0.0",
        "predictor_service": predictor_service,
        "endpoints": ["/predict", "/predict/batch", "/health", "/model/info"],
    }
    return app_config


def predict_endpoint(request: ApplicationRequest,
                     predictor: PredictorService) -> PredictionResponse:
    """
    Handler for POST /predict endpoint.

    Args:
        request: Application data from request body
        predictor: Injected predictor service

    Returns:
        PredictionResponse

    Raises:
        HTTPException(400): If validation fails
        HTTPException(404): If university/program not found
        HTTPException(503): If model not loaded

    IMPLEMENTATION:
        1. Validate request using request.validate()
        2. Call predictor.predict(request)
        3. Handle exceptions and convert to HTTP errors
    """
    return predictor.predict(request)


def predict_batch_endpoint(request: BatchPredictionRequest,
                           predictor: PredictorService) -> BatchPredictionResponse:
    """
    Handler for POST /predict/batch endpoint.

    Args:
        request: Batch request with list of applications
        predictor: Injected predictor service

    Returns:
        BatchPredictionResponse

    Raises:
        HTTPException(400): If batch is too large (>1000)

    IMPLEMENTATION:
        1. Check batch size limit
        2. Call predictor.predict_batch(request)
        3. Return results with timing
    """
    return predictor.predict_batch(request)


def health_endpoint(predictor: PredictorService) -> HealthResponse:
    """
    Handler for GET /health endpoint.

    Returns service health status.

    IMPLEMENTATION:
        1. Check if model is loaded
        2. Check database connection
        3. Check embedding service
        4. Return aggregated status
    """
    return HealthResponse(
        status="healthy", model_loaded=True, database_connected=True,
        embedding_service_available=True, timestamp=datetime.now().isoformat(),
        uptime_seconds=0.0
    )


def model_info_endpoint(predictor: PredictorService) -> ModelInfo:
    """
    Handler for GET /model/info endpoint.

    Returns model metadata and performance metrics.
    """
    return predictor._model_info


# =============================================================================
#                         MIDDLEWARE & UTILITIES
# =============================================================================

def log_request_middleware(request: Any, call_next: Any) -> Any:
    """
    Middleware to log all requests.

    Logs:
        - Timestamp
        - Endpoint
        - Request body (if JSON)
        - Response time
        - Status code

    IMPLEMENTATION:
        1. Record start time
        2. Call next middleware
        3. Compute duration
        4. Log request details
        5. Return response
    """
    import time
    start = time.time()
    response = call_next(request)
    return response


def rate_limit_middleware(request: Any, call_next: Any) -> Any:
    """
    Middleware for rate limiting.

    Limits:
        - 100 requests per minute per IP for single predictions
        - 10 requests per minute per IP for batch predictions

    IMPLEMENTATION:
        1. Extract client IP
        2. Check rate limit counter
        3. If exceeded, return 429 Too Many Requests
        4. Otherwise, increment counter and proceed
    """
    response = call_next(request)
    return response


def normalize_university_name(name: str) -> str:
    """
    Normalize university name for lookup.

    Examples:
        "UofT" -> "University of Toronto"
        "u of t" -> "University of Toronto"
        "toronto" -> "University of Toronto"

    IMPLEMENTATION:
        1. Lowercase
        2. Check alias dictionary
        3. Return canonical name or raise ValueError
    """
    aliases = {
        "uoft": "University of Toronto", "u of t": "University of Toronto",
        "toronto": "University of Toronto", "university of toronto": "University of Toronto",
        "ubc": "University of British Columbia", "university of british columbia": "University of British Columbia",
        "mcgill": "McGill University", "mcgill university": "McGill University",
        "waterloo": "University of Waterloo", "university of waterloo": "University of Waterloo",
        "queens": "Queen's University", "queen's": "Queen's University",
        "western": "Western University", "western university": "Western University",
        "uottawa": "University of Ottawa", "ottawa": "University of Ottawa",
    }
    return aliases.get(name.lower().strip(), name)


def normalize_program_name(name: str) -> str:
    """
    Normalize program name for lookup.

    Examples:
        "CS" -> "Computer Science"
        "comp sci" -> "Computer Science"
        "ECE" -> "Electrical and Computer Engineering"
    """
    aliases = {
        "cs": "Computer Science", "comp sci": "Computer Science", "computer science": "Computer Science",
        "ece": "Electrical and Computer Engineering",
        "eng": "Engineering", "engineering": "Engineering",
        "math": "Mathematics", "mathematics": "Mathematics",
        "bio": "Biology", "biology": "Biology",
        "chem": "Chemistry", "chemistry": "Chemistry",
        "phys": "Physics", "physics": "Physics",
        "se": "Software Engineering", "software engineering": "Software Engineering",
    }
    return aliases.get(name.lower().strip(), name)


# =============================================================================
#                         STARTUP/SHUTDOWN
# =============================================================================

def startup_event():
    """
    Application startup handler.

    IMPLEMENTATION:
        1. Load model from disk/S3
        2. Initialize embedding service
        3. Connect to database
        4. Warm up model (run dummy prediction)
        5. Log startup message
    """
    print("Application starting up...")


def shutdown_event():
    """
    Application shutdown handler.

    IMPLEMENTATION:
        1. Close database connections
        2. Flush logs
        3. Save any cached state
        4. Log shutdown message
    """
    print("Application shutting down...")


# =============================================================================
#                              TODO LIST
# =============================================================================
"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TODO LIST                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  HIGH PRIORITY (Core API):                                                   │
│  ─────────────────────────                                                  │
│  [ ] ApplicationRequest validation                                          │
│      - [ ] Range checks for averages                                        │
│      - [ ] Required field validation                                        │
│      - [ ] Province code validation                                         │
│                                                                              │
│  [ ] PredictorService.predict()                                             │
│      - [ ] Feature engineering                                              │
│      - [ ] Model inference                                                  │
│      - [ ] Calibration                                                      │
│      - [ ] Confidence intervals                                             │
│                                                                              │
│  [ ] create_app() FastAPI setup                                             │
│      - [ ] POST /predict endpoint                                           │
│      - [ ] GET /health endpoint                                             │
│      - [ ] Error handling                                                   │
│                                                                              │
│  MEDIUM PRIORITY (Enhanced features):                                        │
│  ─────────────────────────────────────                                       │
│  [ ] Batch prediction endpoint                                              │
│      - [ ] Vectorized prediction                                            │
│      - [ ] Error accumulation                                               │
│      - [ ] Timing metrics                                                   │
│                                                                              │
│  [ ] Feature importance explanation                                         │
│      - [ ] Contribution calculation                                         │
│      - [ ] Top-k selection                                                  │
│      - [ ] Human-readable names                                             │
│                                                                              │
│  [ ] Similar programs endpoint                                              │
│      - [ ] Embedding similarity                                             │
│      - [ ] Historical rate lookup                                           │
│                                                                              │
│  LOWER PRIORITY (Production readiness):                                      │
│  ───────────────────────────────────────                                     │
│  [ ] Logging middleware                                                     │
│  [ ] Rate limiting                                                          │
│  [ ] Request/response caching                                               │
│  [ ] OpenAPI documentation                                                  │
│  [ ] CORS configuration                                                     │
│  [ ] Authentication (API keys)                                              │
│                                                                              │
│  TESTING:                                                                    │
│  ────────                                                                    │
│  [ ] Unit tests for validation                                              │
│  [ ] Integration tests for endpoints                                        │
│  [ ] Load testing (100+ requests/sec)                                       │
│  [ ] Edge cases (missing fields, invalid values)                           │
│                                                                              │
│  DEPLOYMENT:                                                                 │
│  ───────────                                                                 │
│  [ ] Docker container setup                                                 │
│  [ ] Health check configuration                                             │
│  [ ] Environment variable handling                                          │
│  [ ] Model artifact loading from S3/GCS                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    print("FastAPI Predictor Module")
    print("=" * 50)
    print("This module provides REST API for predictions.")
    print()
    print("To run the server:")
    print("  uvicorn src.api.predictor:app --reload")
    print()
    print("API endpoints:")
    print("  POST /predict          - Single prediction")
    print("  POST /predict/batch    - Batch predictions")
    print("  GET  /health           - Health check")
    print("  GET  /model/info       - Model metadata")
    print()
    print("See docstrings for detailed API documentation.")
