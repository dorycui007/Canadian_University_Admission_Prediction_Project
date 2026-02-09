"""
Models Package — Prediction Models
====================================

Prediction models for university admission outcomes, from simple
Bayesian baselines to logistic regression (IRLS) and attention-based
architectures.

Modules:
    base       - Abstract base class and shared dataclasses
    baseline   - Beta-Binomial conjugate baseline model
    logistic   - Logistic regression via Iteratively Reweighted Least Squares
    attention  - Self/cross-attention layers for applicant features
    embeddings - Learned entity embeddings for universities and programs
    hazard     - Discrete-time survival model for decision timing
"""

from .base import (
    BaseModel,
    ModelConfig,
    TrainingHistory,
    EvaluationMetrics,
)

from .baseline import (
    BaselineModel,
    BetaPrior,
    ProgramPosterior,
    compute_shrinkage_factor,
)

from .logistic import (
    LogisticModel,
    IRLSConvergenceError,
    sigmoid,
    log_loss,
    compute_gradient,
    compute_hessian,
    irls_step,
)

from .attention import (
    AttentionModel,
    SelfAttentionLayer,
    CrossAttentionLayer,
    AttentionOutput,
    scaled_dot_product_attention,
    softmax,
)

from .embeddings import (
    EmbeddingModel,
    EmbeddingConfig,
    CategoryEncoder,
    compute_embedding_dim,
)

from .hazard import (
    HazardModel,
    TimingPrediction,
    expand_to_person_period,
    create_time_dummies,
)

__all__ = [
    # base
    "BaseModel",
    "ModelConfig",
    "TrainingHistory",
    "EvaluationMetrics",
    # baseline
    "BaselineModel",
    "BetaPrior",
    "ProgramPosterior",
    "compute_shrinkage_factor",
    # logistic
    "LogisticModel",
    "IRLSConvergenceError",
    "sigmoid",
    "log_loss",
    "compute_gradient",
    "compute_hessian",
    "irls_step",
    # attention
    "AttentionModel",
    "SelfAttentionLayer",
    "CrossAttentionLayer",
    "AttentionOutput",
    "scaled_dot_product_attention",
    "softmax",
    # embeddings
    "EmbeddingModel",
    "EmbeddingConfig",
    "CategoryEncoder",
    "compute_embedding_dim",
    # hazard
    "HazardModel",
    "TimingPrediction",
    "expand_to_person_period",
    "create_time_dummies",
]
