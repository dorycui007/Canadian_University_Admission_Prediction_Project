"""
Features Package — Feature Engineering
========================================

Specialized encoders for admission data and a builder-pattern
design matrix constructor.

Modules:
    encoders      - Typed encoders (GPA, university, program, term, date, frequency, WOE)
    design_matrix - Design matrix builder with interactions and polynomial features
"""

from .encoders import (
    BaseEncoder,
    GPAEncoder,
    UniversityEncoder,
    ProgramEncoder,
    TermEncoder,
    DateEncoder,
    FrequencyEncoder,
    WOEEncoder,
    CompositeEncoder,
    GPAScaleConfig,
    TargetEncodingConfig,
    HierarchicalGrouping,
    create_admission_encoders,
    encode_admission_features,
)

from .design_matrix import (
    FeatureSpec,
    InteractionSpec,
    DesignMatrixConfig,
    FittedTransform,
    BaseFeatureTransformer,
    NumericScaler,
    OneHotEncoder,
    DummyEncoder,
    OrdinalEncoder,
    InteractionBuilder,
    DesignMatrixBuilder,
    validate_design_matrix,
    check_column_rank,
    identify_collinear_features,
    build_admission_matrix,
    create_polynomial_features,
)

__all__ = [
    # encoders
    "BaseEncoder",
    "GPAEncoder",
    "UniversityEncoder",
    "ProgramEncoder",
    "TermEncoder",
    "DateEncoder",
    "FrequencyEncoder",
    "WOEEncoder",
    "CompositeEncoder",
    "GPAScaleConfig",
    "TargetEncodingConfig",
    "HierarchicalGrouping",
    "create_admission_encoders",
    "encode_admission_features",
    # design_matrix
    "FeatureSpec",
    "InteractionSpec",
    "DesignMatrixConfig",
    "FittedTransform",
    "BaseFeatureTransformer",
    "NumericScaler",
    "OneHotEncoder",
    "DummyEncoder",
    "OrdinalEncoder",
    "InteractionBuilder",
    "DesignMatrixBuilder",
    "validate_design_matrix",
    "check_column_rank",
    "identify_collinear_features",
    "build_admission_matrix",
    "create_polynomial_features",
]
