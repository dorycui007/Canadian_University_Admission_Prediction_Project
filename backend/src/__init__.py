"""
University Admission Prediction System
=======================================

A from-scratch prediction system for Canadian university admissions,
built on pure linear algebra (MAT223/CSC148 foundations).

Subpackages:
    math        - Linear algebra primitives (vectors, matrices, QR, SVD, ridge)
    models      - Prediction models (Beta-Binomial, logistic IRLS, attention, embeddings, hazard)
    features    - Feature engineering (encoders, design matrix builder)
    evaluation  - Model evaluation (calibration, discrimination, temporal validation)
    utils       - Data normalization (university/program fuzzy matching)
    db          - Database layer (MongoDB, Weaviate vector DB, ETL pipeline)
    visualization - Plots for math concepts and evaluation metrics
    api         - REST API for real-time predictions
"""

__version__ = "0.1.0"
