"""
Abstract Base Model Module
===========================

This module defines the abstract base class for all prediction models in the
University Admissions Prediction System. All models (Baseline, Logistic, Hazard,
Embeddings) inherit from BaseModel to ensure consistent interfaces.

CSC148 REFERENCE: Sections 3.5-3.8 (Inheritance), 4.1-4.2 (ADTs)

================================================================================
                        SYSTEM ARCHITECTURE CONTEXT
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    MODEL HIERARCHY                                       │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │                         ┌─────────────────┐                              │
    │                         │   [BaseModel]   │  ← Abstract base class       │
    │                         │    (this file)  │                              │
    │                         └────────┬────────┘                              │
    │                                  │                                       │
    │        ┌─────────────┬───────────┼───────────┬─────────────┐            │
    │        ▼             ▼           ▼           ▼             ▼            │
    │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ │
    │  │ Baseline  │ │ Logistic  │ │  Hazard   │ │ Embedding │ │ Attention │ │
    │  │  Model    │ │   Model   │ │   Model   │ │   Model   │ │   Model   │ │
    │  │  (Beta)   │ │  (IRLS)   │ │ (timing)  │ │ (low-rank)│ │           │ │
    │  └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘ │
    │       │             │             │             │             │         │
    │       └─────────────┴─────────────┴─────────────┴─────────────┘         │
    │                                  │                                       │
    │                                  ▼                                       │
    │                         ┌─────────────────┐                              │
    │                         │   Predictor     │  ← Unified interface        │
    │                         │   (api/)        │                              │
    │                         └─────────────────┘                              │
    │                                                                          │
    │  All models share: fit(), predict_proba(), evaluate(), save(), load()   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    WHY ABSTRACT BASE CLASS?
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  BENEFITS OF INHERITANCE                                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  1. CONSISTENT INTERFACE                                                 │
    │     ─────────────────────                                                │
    │     All models have the same methods:                                    │
    │     • fit(X, y)           - Train the model                             │
    │     • predict_proba(X)    - Get probability predictions                 │
    │     • predict(X)          - Get binary predictions                      │
    │     • evaluate(X, y)      - Compute metrics                             │
    │                                                                          │
    │     This means the API layer doesn't need to know which model it uses!  │
    │                                                                          │
    │  2. SHARED FUNCTIONALITY                                                 │
    │     ─────────────────────                                                │
    │     Common code is implemented once in BaseModel:                        │
    │     • Input validation                                                   │
    │     • Metric computation                                                 │
    │     • Model persistence (save/load)                                      │
    │     • Logging and diagnostics                                            │
    │                                                                          │
    │  3. TYPE SAFETY                                                          │
    │     ───────────                                                          │
    │     Using abstract methods, Python will error if a subclass forgets     │
    │     to implement required methods.                                       │
    │                                                                          │
    │  4. EASY MODEL COMPARISON                                                │
    │     ──────────────────────                                               │
    │     ```python                                                            │
    │     models = [BaselineModel(), LogisticModel(), EmbeddingModel()]       │
    │     for model in models:                                                 │
    │         model.fit(X_train, y_train)                                      │
    │         print(model.name, model.evaluate(X_test, y_test))               │
    │     ```                                                                  │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    MODEL INTERFACE DESIGN
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  DATA FLOW THROUGH MODELS                                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  TRAINING:                                                               │
    │  ─────────                                                               │
    │                                                                          │
    │      Raw Data ──► Feature Engineering ──► Design Matrix X               │
    │                                               │                          │
    │                                               ▼                          │
    │                                    ┌─────────────────┐                   │
    │                                    │  model.fit(X,y) │                   │
    │                                    └────────┬────────┘                   │
    │                                             │                            │
    │                                             ▼                            │
    │                                    ┌─────────────────┐                   │
    │                                    │  Learned params │                   │
    │                                    │  (β, embeddings,│                   │
    │                                    │   α, β for Beta)│                   │
    │                                    └─────────────────┘                   │
    │                                                                          │
    │  INFERENCE:                                                              │
    │  ──────────                                                              │
    │                                                                          │
    │      New Application ──► Feature Engineering ──► Feature vector x       │
    │                                                       │                  │
    │                                                       ▼                  │
    │                                          ┌────────────────────┐          │
    │                                          │ model.predict_proba│          │
    │                                          └──────────┬─────────┘          │
    │                                                     │                    │
    │                                                     ▼                    │
    │                                              ┌─────────────┐             │
    │                                              │ P(admit) ∈  │             │
    │                                              │   [0, 1]    │             │
    │                                              └─────────────┘             │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                    EVALUATION METRICS OVERVIEW
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  METRICS COMPUTED BY evaluate()                                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  DISCRIMINATION (does model rank correctly?):                            │
    │  ───────────────────────────────────────────                             │
    │  • ROC-AUC:  Area under ROC curve [0, 1]                                │
    │              Higher = better separation of classes                       │
    │                                                                          │
    │  • PR-AUC:   Area under Precision-Recall curve                          │
    │              Better for imbalanced classes                               │
    │                                                                          │
    │  CALIBRATION (are probabilities accurate?):                              │
    │  ───────────────────────────────────────────                             │
    │  • Brier Score:  Mean squared error of probabilities [0, 1]             │
    │                  Lower = better                                          │
    │                  Brier = (1/n) Σ(pᵢ - yᵢ)²                              │
    │                                                                          │
    │  • ECE:      Expected Calibration Error                                  │
    │              Measures how well P(admit|pred=p) ≈ p                      │
    │                                                                          │
    │  VISUAL:                                                                 │
    │  ───────                                                                 │
    │                                                                          │
    │      Discrimination:          Calibration:                               │
    │                                                                          │
    │      TPR│     ╭───────        Observed│    ╱                            │
    │         │   ╱╱                    prob│  ╱  ●                           │
    │         │  ╱                          │ ╱ ●                              │
    │         │ ╱   AUC = area              │╱●                               │
    │         │╱    under curve           0 ├───────────►                      │
    │         └─────────────►               0          1                       │
    │           FPR                         Predicted prob                     │
    │                                                                          │
    │  TARGETS FOR THIS PROJECT:                                               │
    │  ──────────────────────────                                              │
    │  • AUC-ROC > 0.75                                                       │
    │  • Brier Score < 0.20                                                   │
    │  • ECE < 0.05                                                           │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class ModelConfig:
    """
    Configuration for model hyperparameters and training settings.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  MODEL CONFIGURATION                                                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  COMMON PARAMETERS:                                                      │
    │  ───────────────────                                                     │
    │  • lambda_:      Ridge regularization strength (all models)             │
    │  • max_iter:     Maximum training iterations (IRLS, gradient descent)   │
    │  • tol:          Convergence tolerance                                   │
    │  • random_state: For reproducibility                                     │
    │                                                                          │
    │  MODEL-SPECIFIC:                                                         │
    │  ────────────────                                                        │
    │  Loaded from config.yaml and passed to specific model classes.          │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Attributes:
        lambda_: Ridge regularization parameter (default: 0.1)
        max_iter: Maximum iterations for iterative algorithms (default: 100)
        tol: Convergence tolerance (default: 1e-6)
        random_state: Random seed for reproducibility (default: None)
        extra_params: Dictionary for model-specific parameters

    Example:
        >>> config = ModelConfig(lambda_=0.5, max_iter=200)
        >>> model = LogisticModel(config)
    """
    lambda_: float = 0.1
    max_iter: int = 100
    tol: float = 1e-6
    random_state: Optional[int] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingHistory:
    """
    Record of model training process for debugging and visualization.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  TRAINING HISTORY                                                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Tracks per-iteration metrics during training:                           │
    │                                                                          │
    │      Iteration │ Loss     │ Gradient Norm │ Step Size                   │
    │      ──────────┼──────────┼───────────────┼────────────                  │
    │      1         │ 0.693    │ 0.452         │ 1.0                          │
    │      2         │ 0.501    │ 0.321         │ 1.0                          │
    │      3         │ 0.423    │ 0.185         │ 1.0                          │
    │      ...       │ ...      │ ...           │ ...                          │
    │      25        │ 0.312    │ 0.0001        │ ← Converged!                │
    │                                                                          │
    │  USEFUL FOR:                                                             │
    │  ────────────                                                            │
    │  • Diagnosing convergence issues                                        │
    │  • Plotting learning curves                                             │
    │  • Choosing appropriate max_iter                                        │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Attributes:
        losses: Loss value at each iteration
        gradient_norms: Norm of gradient at each iteration
        timestamps: Time of each iteration (for profiling)
        converged: Whether training converged before max_iter
        final_iteration: Number of iterations completed
    """
    losses: List[float] = field(default_factory=list)
    gradient_norms: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    converged: bool = False
    final_iteration: int = 0


@dataclass
class EvaluationMetrics:
    """
    Container for model evaluation metrics.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  EVALUATION METRICS                                                      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  DISCRIMINATION:                                                         │
    │  ────────────────                                                        │
    │  • roc_auc:     ROC-AUC score [0.5 = random, 1.0 = perfect]            │
    │  • pr_auc:      Precision-Recall AUC                                    │
    │  • accuracy:    Classification accuracy at threshold 0.5                │
    │                                                                          │
    │  CALIBRATION:                                                            │
    │  ─────────────                                                           │
    │  • brier_score: Mean squared probability error [0 = perfect]            │
    │  • ece:         Expected Calibration Error                               │
    │  • mce:         Maximum Calibration Error                                │
    │                                                                          │
    │  LOG-LIKELIHOOD:                                                         │
    │  ────────────────                                                        │
    │  • log_loss:    Negative log-likelihood (cross-entropy)                 │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Attributes:
        roc_auc: Area under ROC curve
        pr_auc: Area under Precision-Recall curve
        brier_score: Brier score (mean squared error of probabilities)
        ece: Expected Calibration Error
        mce: Maximum Calibration Error
        log_loss: Negative log-likelihood
        accuracy: Classification accuracy at threshold 0.5
    """
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    brier_score: float = 0.0
    ece: float = 0.0
    mce: float = 0.0
    log_loss: float = 0.0
    accuracy: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary for serialization."""
        pass

    def __str__(self) -> str:
        """Pretty-print metrics summary."""
        pass


class BaseModel(ABC):
    """
    Abstract base class for all prediction models.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  ABSTRACT BASE CLASS DESIGN                                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  ABSTRACT METHODS (must be implemented by subclasses):                   │
    │  ─────────────────────────────────────────────────────                   │
    │  • fit(X, y)         - Train the model                                   │
    │  • predict_proba(X)  - Return probabilities                              │
    │  • get_params()      - Return model parameters                           │
    │  • set_params()      - Set model parameters                              │
    │                                                                          │
    │  CONCRETE METHODS (shared implementation):                               │
    │  ──────────────────────────────────────────                              │
    │  • predict(X)        - Threshold probabilities → binary                 │
    │  • evaluate(X, y)    - Compute all metrics                              │
    │  • save(path)        - Persist model to disk                            │
    │  • load(path)        - Load model from disk                             │
    │  • validate_input()  - Check X and y shapes                             │
    │                                                                          │
    │  PROPERTIES:                                                             │
    │  ───────────                                                             │
    │  • name:             Human-readable model name                           │
    │  • is_fitted:        Whether fit() has been called                       │
    │  • n_features:       Number of input features                           │
    │  • n_params:         Number of learnable parameters                     │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Example usage:
        >>> class MyModel(BaseModel):
        ...     def fit(self, X, y):
        ...         # Training implementation
        ...         pass
        ...     def predict_proba(self, X):
        ...         # Probability prediction
        ...         pass
        ...
        >>> model = MyModel()
        >>> model.fit(X_train, y_train)
        >>> probs = model.predict_proba(X_test)
        >>> metrics = model.evaluate(X_test, y_test)
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize base model with configuration.

        Args:
            config: ModelConfig instance with hyperparameters.
                    If None, uses default configuration.

        IMPLEMENTATION NOTES:
        ─────────────────────
        1. Store config (or create default)
        2. Initialize _is_fitted = False
        3. Initialize training_history as empty TrainingHistory
        4. Initialize _n_features = None
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable name for the model.

        Returns:
            Model name string (e.g., "Beta-Binomial Baseline", "IRLS Logistic")

        Example:
            >>> model = LogisticModel()
            >>> print(model.name)
            "IRLS Logistic Regression"
        """
        pass

    @property
    def is_fitted(self) -> bool:
        """
        Check if the model has been fitted.

        Returns:
            True if fit() has been called successfully, False otherwise.

        IMPORTANT: Always check is_fitted before predict_proba()!

        Example:
            >>> model = LogisticModel()
            >>> model.is_fitted
            False
            >>> model.fit(X, y)
            >>> model.is_fitted
            True
        """
        pass

    @property
    def n_features(self) -> Optional[int]:
        """
        Number of features the model expects.

        Returns:
            Number of features (columns in X), or None if not fitted.

        USED FOR: Input validation in predict_proba().
        """
        pass

    @property
    @abstractmethod
    def n_params(self) -> int:
        """
        Number of learnable parameters in the model.

        Returns:
            Total number of parameters.

        Examples:
            - Logistic regression: n_features (β coefficients)
            - Embedding model: n_universities * embed_dim + n_programs * embed_dim
            - Beta baseline: 2 per program (α, β for each)

        USED FOR: Model complexity comparison, AIC/BIC computation.
        """
        pass

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> 'BaseModel':
        """
        Fit the model to training data.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  TRAINING INTERFACE                                                  │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  INPUT:                                                              │
        │  ──────                                                              │
        │  X: Design matrix of shape (n_samples, n_features)                  │
        │     • Each row is one application                                    │
        │     • Each column is one feature (avg, is_UofT, is_CS, etc.)        │
        │                                                                      │
        │  y: Binary target of shape (n_samples,)                             │
        │     • 1 = admitted                                                   │
        │     • 0 = rejected                                                   │
        │                                                                      │
        │  sample_weight: Optional weights of shape (n_samples,)              │
        │     • Used for importance weighting                                  │
        │     • Default: equal weight for all samples                         │
        │                                                                      │
        │  OUTPUT:                                                             │
        │  ───────                                                             │
        │  Returns self (for method chaining)                                 │
        │  Side effects:                                                       │
        │  • Sets self._is_fitted = True                                      │
        │  • Sets self._n_features = X.shape[1]                               │
        │  • Stores learned parameters                                        │
        │  • Updates self.training_history                                    │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            X: Design matrix of shape (n_samples, n_features)
            y: Binary target vector of shape (n_samples,)
            sample_weight: Optional sample weights of shape (n_samples,)

        Returns:
            self (allows method chaining like model.fit(X,y).predict(X))

        Raises:
            ValueError: If X and y have incompatible shapes

        Example:
            >>> model = LogisticModel()
            >>> model.fit(X_train, y_train)
            >>> # Method chaining:
            >>> probs = LogisticModel().fit(X_train, y_train).predict_proba(X_test)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict admission probabilities.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  PROBABILITY PREDICTION                                              │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  Returns P(admit | X) for each sample.                              │
        │                                                                      │
        │  OUTPUT REQUIREMENTS:                                                │
        │  ─────────────────────                                               │
        │  • Shape: (n_samples,)                                              │
        │  • Values: All in [0, 1]                                            │
        │  • Interpretation: P(admit) for each application                    │
        │                                                                      │
        │  IMPLEMENTATION NOTES:                                               │
        │  ──────────────────────                                              │
        │  1. Check self.is_fitted, raise if False                            │
        │  2. Validate X.shape[1] == self.n_features                         │
        │  3. Compute probabilities using learned parameters                  │
        │  4. Clip to [epsilon, 1-epsilon] for numerical stability           │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Probability predictions of shape (n_samples,)

        Raises:
            RuntimeError: If model has not been fitted
            ValueError: If X has wrong number of features

        Example:
            >>> model.fit(X_train, y_train)
            >>> probs = model.predict_proba(X_test)
            >>> print(f"First 5 predictions: {probs[:5]}")
            [0.72, 0.45, 0.88, 0.12, 0.56]
        """
        pass

    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Make binary predictions by thresholding probabilities.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  BINARY PREDICTION                                                   │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │      predict(X) = 1  if  predict_proba(X) ≥ threshold               │
        │                   0  otherwise                                       │
        │                                                                      │
        │  CHOOSING THRESHOLD:                                                 │
        │  ───────────────────                                                 │
        │  • 0.5: Natural choice, treats FP and FN equally                    │
        │  • Higher: More conservative, fewer false positives                 │
        │  • Lower: More lenient, fewer false negatives                       │
        │                                                                      │
        │  For admissions predictions, we generally want PROBABILITIES        │
        │  not binary predictions. Use predict_proba() for the API.           │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            threshold: Classification threshold (default: 0.5)

        Returns:
            Binary predictions of shape (n_samples,)

        Example:
            >>> probs = model.predict_proba(X_test)
            >>> preds = model.predict(X_test, threshold=0.5)
            >>> print(preds)  # [1, 0, 1, 0, 1]

        IMPLEMENTATION:
        ────────────────
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters as a dictionary.

        Returns:
            Dictionary containing all model parameters.
            Keys are parameter names, values are parameter values.

        Example:
            >>> params = model.get_params()
            >>> print(params['coefficients'])  # For logistic regression
        """
        pass

    @abstractmethod
    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set model parameters from a dictionary.

        Used for loading saved models.

        Args:
            params: Dictionary of parameter name → value

        Raises:
            ValueError: If params dict is missing required keys
        """
        pass

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> EvaluationMetrics:
        """
        Compute all evaluation metrics on given data.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  MODEL EVALUATION                                                    │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  Computes:                                                           │
        │  ─────────                                                           │
        │  DISCRIMINATION:                                                     │
        │  • ROC-AUC          (from evaluation/discrimination.py)             │
        │  • PR-AUC           (from evaluation/discrimination.py)             │
        │  • Accuracy         (simple threshold-based)                        │
        │                                                                      │
        │  CALIBRATION:                                                        │
        │  • Brier Score      (from evaluation/calibration.py)                │
        │  • ECE              (from evaluation/calibration.py)                │
        │  • MCE              (from evaluation/calibration.py)                │
        │                                                                      │
        │  LOG-LIKELIHOOD:                                                     │
        │  • Log Loss         (negative log-likelihood)                       │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: True labels of shape (n_samples,)

        Returns:
            EvaluationMetrics dataclass with all computed metrics

        Example:
            >>> metrics = model.evaluate(X_test, y_test)
            >>> print(f"ROC-AUC: {metrics.roc_auc:.3f}")
            >>> print(f"Brier Score: {metrics.brier_score:.3f}")

        IMPLEMENTATION STEPS:
        ─────────────────────
        1. probs = self.predict_proba(X)
        2. preds = self.predict(X)
        3. Compute each metric using evaluation module functions
        4. Return EvaluationMetrics dataclass
        """
        pass

    def validate_input(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> None:
        """
        Validate input data shapes and types.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  INPUT VALIDATION                                                    │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  CHECKS:                                                             │
        │  ───────                                                             │
        │  1. X is 2D numpy array                                             │
        │  2. y (if provided) is 1D numpy array                               │
        │  3. X.shape[0] == y.shape[0]                                        │
        │  4. No NaN or Inf values                                            │
        │  5. If fitted, X.shape[1] == self.n_features                       │
        │                                                                      │
        │  RAISES:                                                             │
        │  ───────                                                             │
        │  • ValueError for shape mismatches                                  │
        │  • TypeError for wrong types                                        │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            X: Feature matrix to validate
            y: Optional target vector to validate

        Raises:
            ValueError: If validation fails
            TypeError: If inputs have wrong type

        IMPLEMENTATION STEPS:
        ─────────────────────
        1. if not isinstance(X, np.ndarray): raise TypeError
        2. if X.ndim != 2: raise ValueError("X must be 2D")
        3. if y is not None:
               if y.ndim != 1: raise ValueError("y must be 1D")
               if X.shape[0] != y.shape[0]: raise ValueError("Size mismatch")
        4. if np.any(np.isnan(X)) or np.any(np.isinf(X)):
               raise ValueError("X contains NaN or Inf")
        5. if self.is_fitted and X.shape[1] != self.n_features:
               raise ValueError(f"Expected {self.n_features} features")
        """
        pass

    def save(self, path: str) -> None:
        """
        Save model to disk.

        ┌─────────────────────────────────────────────────────────────────────┐
        │  MODEL PERSISTENCE                                                   │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  SAVED DATA:                                                         │
        │  ───────────                                                         │
        │  • Model class name (for loading)                                   │
        │  • Configuration (hyperparameters)                                  │
        │  • Learned parameters (from get_params())                           │
        │  • Training history                                                 │
        │  • Metadata (date saved, version, etc.)                             │
        │                                                                      │
        │  FORMAT: JSON or pickle (depending on implementation)               │
        │                                                                      │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            path: File path to save model

        Raises:
            RuntimeError: If model is not fitted
            IOError: If file cannot be written

        Example:
            >>> model.fit(X, y)
            >>> model.save("models/logistic_v1.json")

        IMPLEMENTATION STEPS:
        ─────────────────────
        1. Check self.is_fitted
        2. Create dict with:
           - 'model_class': self.__class__.__name__
           - 'config': asdict(self.config)
           - 'params': self.get_params()
           - 'n_features': self.n_features
           - 'saved_at': datetime.now().isoformat()
        3. Save to path (use json or pickle)
        """
        pass

    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """
        Load model from disk.

        Args:
            path: File path to load model from

        Returns:
            Loaded model instance

        Raises:
            FileNotFoundError: If path doesn't exist
            ValueError: If file is corrupted or wrong format

        Example:
            >>> model = LogisticModel.load("models/logistic_v1.json")
            >>> probs = model.predict_proba(X_new)

        IMPLEMENTATION STEPS:
        ─────────────────────
        1. Load dict from file
        2. Create new instance with saved config
        3. Call set_params() with saved parameters
        4. Set _is_fitted = True
        5. Return instance
        """
        pass

    def summary(self) -> str:
        """
        Generate human-readable model summary.

        Returns:
            Multi-line string describing the model

        Example output:
            ─────────────────────────────────────────
            Model: IRLS Logistic Regression
            Status: Fitted
            Features: 25
            Parameters: 25
            Config:
              lambda: 0.1
              max_iter: 100
              tol: 1e-6
            ─────────────────────────────────────────

        IMPLEMENTATION:
        ────────────────
        Build string using self.name, self.is_fitted, self.n_features,
        self.n_params, and self.config attributes.
        """
        pass


# =============================================================================
#                           TODO LIST
# =============================================================================
"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TODO                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  HIGH PRIORITY (core structure):                                             │
│  ────────────────────────────────                                            │
│  [ ] Implement ModelConfig dataclass                                        │
│  [ ] Implement TrainingHistory dataclass                                    │
│  [ ] Implement EvaluationMetrics dataclass with to_dict() and __str__()     │
│  [ ] Implement BaseModel.__init__()                                         │
│  [ ] Implement BaseModel.is_fitted property                                 │
│  [ ] Implement BaseModel.n_features property                                │
│                                                                              │
│  MEDIUM PRIORITY (concrete methods):                                         │
│  ────────────────────────────────────                                        │
│  [ ] Implement BaseModel.predict() - threshold probabilities               │
│  [ ] Implement BaseModel.evaluate() - compute all metrics                   │
│  [ ] Implement BaseModel.validate_input() - input checking                  │
│  [ ] Implement BaseModel.summary() - pretty print                           │
│                                                                              │
│  LOW PRIORITY (persistence):                                                 │
│  ────────────────────────────                                                │
│  [ ] Implement BaseModel.save() - serialize to disk                         │
│  [ ] Implement BaseModel.load() - deserialize from disk                     │
│                                                                              │
│  TESTING:                                                                    │
│  ────────                                                                    │
│  [ ] Test that abstract methods raise NotImplementedError                   │
│  [ ] Test validate_input catches all error cases                            │
│  [ ] Test save/load round-trip preserves model                              │
│  [ ] Test evaluate returns correct metric types                             │
│                                                                              │
│  AFTER COMPLETING SUBCLASSES:                                                │
│  ─────────────────────────────                                               │
│  [ ] Verify all subclasses implement abstract methods                       │
│  [ ] Test polymorphism (list of models, same interface)                     │
│  [ ] Create model comparison utility                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    print("Abstract Base Model Module")
    print("=" * 50)
    print()
    print("This module defines the interface for all models.")
    print("Do NOT instantiate BaseModel directly - use subclasses:")
    print()
    print("  • BaselineModel  - Beta-binomial baseline")
    print("  • LogisticModel  - IRLS logistic regression")
    print("  • HazardModel    - Discrete-time hazard")
    print("  • EmbeddingModel - Low-rank embeddings")
    print("  • AttentionModel - Attention mechanism")
