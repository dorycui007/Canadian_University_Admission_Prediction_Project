"""
Design Matrix Builder for Grade Prediction System.

==============================================================================
SYSTEM ARCHITECTURE - WHERE THIS MODULE FITS
==============================================================================

This module constructs the design matrix (X) that feeds into ALL predictive
models. It transforms raw student/application data into a structured numeric
array suitable for linear algebra operations.

┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────────┐   │
│   │   MongoDB   │───▶│  ETL Layer   │───▶│    Raw Feature Dict         │   │
│   │   (Source)  │    │  (db/etl.py) │    │  {'gpa': 3.7, 'uni': 'UofT'}│   │
│   └─────────────┘    └──────────────┘    └──────────────┬──────────────┘   │
│                                                          │                  │
│                                                          ▼                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    DESIGN MATRIX BUILDER                            │  │
│   │                    (This Module)                                    │  │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │  │
│   │  │  Numeric    │  │  Categorical │  │ Interaction │  │  Temporal │  │  │
│   │  │  Features   │  │  Encoding    │  │   Terms     │  │  Features │  │  │
│   │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘  │  │
│   │         │                │                │                │        │  │
│   │         └────────────────┴────────────────┴────────────────┘        │  │
│   │                                    │                                 │  │
│   │                                    ▼                                 │  │
│   │                          ┌─────────────────┐                        │  │
│   │                          │   DESIGN MATRIX │                        │  │
│   │                          │   X ∈ ℝⁿˣᵖ      │                        │  │
│   │                          └─────────────────┘                        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                          │                                  │
│                 ┌────────────────────────┼────────────────────────┐        │
│                 ▼                        ▼                        ▼        │
│   ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────┐   │
│   │   LogisticModel     │  │    HazardModel      │  │  EmbeddingModel │   │
│   │   (IRLS Solver)     │  │  (Survival Timing)  │  │ (Deep Learning) │   │
│   └─────────────────────┘  └─────────────────────┘  └─────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
MAT223 CONNECTION - DESIGN MATRICES IN LINEAR REGRESSION
==============================================================================

The design matrix X encodes all predictor information for the normal equations:

    β̂ = (XᵀX)⁻¹Xᵀy     (OLS solution from MAT223)

Each row represents ONE observation, each column represents ONE feature:

    ┌─────────────────────────────────────────────────────────────┐
    │                    DESIGN MATRIX X                          │
    │                                                             │
    │             Feature 1   Feature 2   Feature 3   ...   p     │
    │           ┌─────────┬─────────┬─────────┬─────┬───────┐     │
    │   Obs 1   │  x₁₁    │   x₁₂   │   x₁₃   │ ... │  x₁ₚ  │     │
    │   Obs 2   │  x₂₁    │   x₂₂   │   x₂₃   │ ... │  x₂ₚ  │     │
    │   Obs 3   │  x₃₁    │   x₃₂   │   x₃₃   │ ... │  x₃ₚ  │     │
    │    ⋮      │   ⋮     │    ⋮    │    ⋮    │  ⋱  │   ⋮   │     │
    │   Obs n   │  xₙ₁    │   xₙ₂   │   xₙ₃   │ ... │  xₙₚ  │     │
    │           └─────────┴─────────┴─────────┴─────┴───────┘     │
    │                                                             │
    │   Dimensions: n observations × p features                   │
    │   X ∈ ℝⁿˣᵖ                                                  │
    └─────────────────────────────────────────────────────────────┘


==============================================================================
FEATURE TYPES AND ENCODING STRATEGIES
==============================================================================

1. NUMERIC FEATURES (Continuous)
   ─────────────────────────────────
   - GPA, test scores, application counts
   - Must be scaled/standardized for IRLS stability

   Raw:      [3.7, 3.2, 3.9, 2.8]
   Scaled:   [0.5, -0.3, 0.8, -1.0]   (z-score normalization)

2. CATEGORICAL FEATURES (Discrete)
   ─────────────────────────────────
   Universities, programs, provinces → One-hot or dummy encoding

   University = 'UofT'  →  [1, 0, 0, 0, 0]  (UofT, UBC, McGill, etc.)
   University = 'UBC'   →  [0, 1, 0, 0, 0]

   ┌─────────────────────────────────────────────────────────┐
   │  ONE-HOT ENCODING (Full Rank for Neural Networks)      │
   │                                                         │
   │  Category: ['A', 'B', 'C']                              │
   │                                                         │
   │  'A' → [1, 0, 0]                                        │
   │  'B' → [0, 1, 0]                                        │
   │  'C' → [0, 0, 1]                                        │
   │                                                         │
   │  Creates k columns for k categories                     │
   └─────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────┐
   │  DUMMY ENCODING (Drop-one for Linear Models)           │
   │                                                         │
   │  Category: ['A', 'B', 'C']   (Reference: 'A')           │
   │                                                         │
   │  'A' → [0, 0]    ← Reference category (implicit)       │
   │  'B' → [1, 0]                                           │
   │  'C' → [0, 1]                                           │
   │                                                         │
   │  Creates k-1 columns, avoids multicollinearity          │
   └─────────────────────────────────────────────────────────┘

3. ORDINAL FEATURES
   ─────────────────────────────────
   High school year, term number → Integer encoding with order

   Year: ['Grade 10', 'Grade 11', 'Grade 12'] → [0, 1, 2]

4. INTERACTION TERMS
   ─────────────────────────────────
   Capture non-additive effects between features

   ┌─────────────────────────────────────────────────────────┐
   │  INTERACTION: GPA × Program Type                        │
   │                                                         │
   │  Some programs weight GPA differently:                  │
   │  - Engineering: High GPA weight                         │
   │  - Arts: Moderate GPA weight                            │
   │                                                         │
   │  x_interaction = x_gpa × x_program_engineering          │
   │                                                         │
   │  This allows the model to learn program-specific        │
   │  GPA effects rather than one global effect              │
   └─────────────────────────────────────────────────────────┘

5. TEMPORAL FEATURES
   ─────────────────────────────────
   Application timing, decision patterns

   - Days until deadline
   - Application month (cyclical encoding)
   - Historical acceptance rate trends


==============================================================================
MULTICOLLINEARITY AND RANK DEFICIENCY
==============================================================================

The design matrix must have FULL COLUMN RANK for stable regression:

    rank(X) = p   (number of columns)

Problems arise when columns are linearly dependent:

    ┌─────────────────────────────────────────────────────────────┐
    │  MULTICOLLINEARITY EXAMPLE                                  │
    │                                                             │
    │  Feature 1: Total Applications                              │
    │  Feature 2: Accepted Applications                           │
    │  Feature 3: Rejected Applications                           │
    │                                                             │
    │  Problem: Feature1 = Feature2 + Feature3 (always!)          │
    │                                                             │
    │  ┌─────┬─────┬─────┐                                        │
    │  │  10 │  3  │  7  │  ← 10 = 3 + 7                          │
    │  │  15 │  5  │  10 │  ← 15 = 5 + 10                         │
    │  │   8 │  4  │  4  │  ← 8 = 4 + 4                           │
    │  └─────┴─────┴─────┘                                        │
    │                                                             │
    │  Solution: Drop one column (Feature 3)                      │
    │  Result: X has full column rank                             │
    └─────────────────────────────────────────────────────────────┘

Ridge regression (L2 penalty) also handles near-collinearity:

    β̂_ridge = (XᵀX + λI)⁻¹Xᵀy

The λI term ensures (XᵀX + λI) is always invertible.


==============================================================================
SPARSE MATRIX CONSIDERATIONS
==============================================================================

With many categorical variables, design matrices become sparse:

    ┌─────────────────────────────────────────────────────────────┐
    │  SPARSE DESIGN MATRIX (One-Hot Universities)                │
    │                                                             │
    │  100 universities → 100 columns (mostly zeros)              │
    │                                                             │
    │  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐          │
    │  │  0  │  0  │  1  │  0  │  0  │  0  │ ... │  0  │ ← UBC    │
    │  │  1  │  0  │  0  │  0  │  0  │  0  │ ... │  0  │ ← UofT   │
    │  │  0  │  0  │  0  │  0  │  1  │  0  │ ... │  0  │ ← McGill │
    │  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘          │
    │                                                             │
    │  Storage: Use scipy.sparse.csr_matrix                       │
    │  - Efficient for row slicing (batch training)               │
    │  - Memory: O(nnz) instead of O(n×p)                         │
    └─────────────────────────────────────────────────────────────┘


==============================================================================
INTERCEPT / BIAS TERM
==============================================================================

The intercept (β₀) requires a column of ones:

    y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ

    ┌─────────────────────────────────────────────────────────────┐
    │  DESIGN MATRIX WITH INTERCEPT                               │
    │                                                             │
    │         Intercept   x₁      x₂      x₃                      │
    │       ┌─────────┬───────┬───────┬───────┐                   │
    │       │    1    │  3.7  │   0   │   1   │                   │
    │       │    1    │  3.2  │   1   │   0   │                   │
    │       │    1    │  3.9  │   0   │   0   │                   │
    │       │    1    │  2.8  │   1   │   1   │                   │
    │       └─────────┴───────┴───────┴───────┘                   │
    │            ↑                                                │
    │       Column of 1s enables intercept term                   │
    └─────────────────────────────────────────────────────────────┘


==============================================================================
STANDARDIZATION AND SCALING
==============================================================================

For IRLS and ridge regression, features should be standardized:

    ┌─────────────────────────────────────────────────────────────┐
    │  Z-SCORE STANDARDIZATION                                    │
    │                                                             │
    │              x - μ                                          │
    │       z = ─────────                                         │
    │               σ                                             │
    │                                                             │
    │  Where:                                                     │
    │    μ = mean of feature                                      │
    │    σ = standard deviation of feature                        │
    │                                                             │
    │  Result: mean=0, std=1 for each feature                     │
    │                                                             │
    │  WHY?                                                       │
    │  1. IRLS convergence is faster with scaled features         │
    │  2. Ridge penalty λ‖β‖² treats all features equally         │
    │  3. Gradient descent takes uniform steps                    │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │  MIN-MAX SCALING                                            │
    │                                                             │
    │              x - min(x)                                     │
    │       z = ──────────────                                    │
    │            max(x) - min(x)                                  │
    │                                                             │
    │  Result: Values in [0, 1] range                             │
    │  Use for: Neural network inputs, bounded features           │
    └─────────────────────────────────────────────────────────────┘


==============================================================================
INTEGRATION WITH MODEL CLASSES
==============================================================================

    ┌─────────────────────────────────────────────────────────────┐
    │  DESIGN MATRIX → MODEL PIPELINE                             │
    │                                                             │
    │  1. Build matrix:                                           │
    │     builder = DesignMatrixBuilder(config)                   │
    │     X = builder.fit_transform(training_data)                │
    │                                                             │
    │  2. Train model:                                            │
    │     model = LogisticModel(ridge_lambda=0.01)                │
    │     model.fit(X, y)                                         │
    │                                                             │
    │  3. Predict (IMPORTANT: use same transformation):           │
    │     X_new = builder.transform(new_data)  # NOT fit_transform│
    │     probabilities = model.predict_proba(X_new)              │
    │                                                             │
    │  CRITICAL: The builder stores μ, σ from training data       │
    │            and applies same transformation to test data     │
    └─────────────────────────────────────────────────────────────┘


Author: Grade Prediction Team
Course Context: MAT223 (Linear Algebra), CSC148 (OOP)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
import numpy as np


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class FeatureSpec:
    """
    Specification for a single feature in the design matrix.

    ┌─────────────────────────────────────────────────────────────┐
    │  FEATURE SPECIFICATION ATTRIBUTES                           │
    │                                                             │
    │  name: str          - Column name (e.g., 'gpa', 'university')│
    │  dtype: str         - 'numeric', 'categorical', 'ordinal'   │
    │  encoding: str      - 'standard', 'onehot', 'dummy', 'ordinal'│
    │  categories: list   - Valid categories (for categorical)    │
    │  handle_unknown: str- 'error', 'ignore', 'encode'           │
    │  missing_strategy: str - 'mean', 'median', 'mode', 'drop'  │
    └─────────────────────────────────────────────────────────────┘

    Examples:
        # Numeric feature with standardization
        FeatureSpec(
            name='gpa',
            dtype='numeric',
            encoding='standard',
            missing_strategy='mean'
        )

        # Categorical with one-hot encoding
        FeatureSpec(
            name='university',
            dtype='categorical',
            encoding='onehot',
            categories=['UofT', 'UBC', 'McGill', ...],
            handle_unknown='ignore'
        )
    """
    name: str
    dtype: str  # 'numeric', 'categorical', 'ordinal'
    encoding: str = 'standard'  # 'standard', 'onehot', 'dummy', 'ordinal', 'none'
    categories: Optional[List[str]] = None
    handle_unknown: str = 'error'  # 'error', 'ignore', 'encode'
    missing_strategy: str = 'mean'  # 'mean', 'median', 'mode', 'drop', 'zero'


@dataclass
class InteractionSpec:
    """
    Specification for interaction terms between features.

    ┌─────────────────────────────────────────────────────────────┐
    │  INTERACTION TYPES                                          │
    │                                                             │
    │  Multiplicative:  x_new = x₁ × x₂                           │
    │  Polynomial:      x_new = x₁²                               │
    │  Cross-product:   x_new = x₁ × x₂ × x₃                      │
    └─────────────────────────────────────────────────────────────┘

    Examples:
        # GPA × Program interaction
        InteractionSpec(
            features=['gpa', 'program_type'],
            interaction_type='multiplicative'
        )

        # GPA squared (polynomial)
        InteractionSpec(
            features=['gpa'],
            interaction_type='polynomial',
            degree=2
        )
    """
    features: List[str]
    interaction_type: str = 'multiplicative'  # 'multiplicative', 'polynomial'
    degree: int = 2  # For polynomial interactions


@dataclass
class DesignMatrixConfig:
    """
    Complete configuration for design matrix construction.

    ┌─────────────────────────────────────────────────────────────┐
    │  DESIGN MATRIX CONFIGURATION                                │
    │                                                             │
    │  feature_specs: List[FeatureSpec]                           │
    │      - Definitions for each raw feature                     │
    │                                                             │
    │  interactions: List[InteractionSpec]                        │
    │      - Interaction terms to create                          │
    │                                                             │
    │  include_intercept: bool                                    │
    │      - Whether to add column of 1s                          │
    │                                                             │
    │  sparse_threshold: float                                    │
    │      - If sparsity > threshold, use sparse matrix           │
    │                                                             │
    │  drop_first: bool                                           │
    │      - Drop first category for dummy encoding               │
    └─────────────────────────────────────────────────────────────┘
    """
    feature_specs: List[FeatureSpec] = field(default_factory=list)
    interactions: List[InteractionSpec] = field(default_factory=list)
    include_intercept: bool = True
    sparse_threshold: float = 0.5  # Use sparse if >50% zeros
    drop_first: bool = True  # For dummy encoding


@dataclass
class FittedTransform:
    """
    Stores learned parameters from fit() for consistent transform().

    ┌─────────────────────────────────────────────────────────────┐
    │  WHY STORE FITTED PARAMETERS?                               │
    │                                                             │
    │  Training:  μ_train = 3.5,  σ_train = 0.4                   │
    │                                                             │
    │  Test sample: GPA = 3.7                                     │
    │                                                             │
    │  CORRECT:                                                   │
    │      z = (3.7 - 3.5) / 0.4 = 0.5                            │
    │      Using training μ and σ                                 │
    │                                                             │
    │  WRONG:                                                     │
    │      z = (3.7 - 3.7) / 0.0 = undefined                      │
    │      Using test μ and σ (data leakage!)                     │
    └─────────────────────────────────────────────────────────────┘
    """
    feature_name: str
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    categories: Optional[List[str]] = None
    category_to_index: Optional[Dict[str, int]] = None


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class BaseFeatureTransformer(ABC):
    """
    Abstract base class for feature transformers.

    ┌─────────────────────────────────────────────────────────────┐
    │  TRANSFORMER INTERFACE (CSC148 Pattern)                     │
    │                                                             │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │           BaseFeatureTransformer                    │    │
    │  │                  (Abstract)                         │    │
    │  │                                                     │    │
    │  │  + fit(X) → self                                    │    │
    │  │  + transform(X) → np.ndarray                        │    │
    │  │  + fit_transform(X) → np.ndarray                    │    │
    │  │  + get_feature_names() → List[str]                  │    │
    │  └─────────────────────────────────────────────────────┘    │
    │                          △                                  │
    │                          │                                  │
    │          ┌───────────────┼───────────────┐                  │
    │          │               │               │                  │
    │  ┌───────┴───────┐ ┌─────┴─────┐ ┌──────┴──────┐           │
    │  │NumericScaler  │ │OneHotEnc  │ │OrdinalEnc   │           │
    │  └───────────────┘ └───────────┘ └─────────────┘           │
    └─────────────────────────────────────────────────────────────┘

    Implementation Notes:
        1. fit() learns parameters from training data
        2. transform() applies learned parameters to any data
        3. fit_transform() is convenience for fit() + transform()
        4. NEVER call fit() on test data - causes data leakage
    """

    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseFeatureTransformer':
        """
        Learn transformation parameters from training data.

        Args:
            X: Training data, shape (n_samples, n_features)

        Returns:
            self (for method chaining)

        Implementation Steps:
            1. Compute statistics (mean, std, categories, etc.)
            2. Store in instance variables
            3. Return self
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply learned transformation to data.

        Args:
            X: Data to transform, shape (n_samples, n_features)

        Returns:
            Transformed data as numpy array

        Implementation Steps:
            1. Check that fit() was called
            2. Apply stored transformation
            3. Handle edge cases (missing values, unknown categories)
        """
        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        ┌─────────────────────────────────────────────────────────┐
        │  EQUIVALENT TO:                                         │
        │                                                         │
        │  transformer.fit(X_train)                               │
        │  X_transformed = transformer.transform(X_train)         │
        │                                                         │
        │  BUT more efficient (single pass through data)          │
        └─────────────────────────────────────────────────────────┘
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Return names of output features.

        Important for interpretability - maps column indices
        back to meaningful feature names.

        Returns:
            List of feature names after transformation

        Example:
            Input: 'university' (single categorical column)
            Output: ['university_UofT', 'university_UBC',
                     'university_McGill', ...]
        """
        pass


# =============================================================================
# NUMERIC FEATURE TRANSFORMER
# =============================================================================

class NumericScaler(BaseFeatureTransformer):
    """
    Scales numeric features using z-score or min-max normalization.

    ┌─────────────────────────────────────────────────────────────┐
    │  Z-SCORE STANDARDIZATION                                    │
    │                                                             │
    │  Training:                                                  │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │  GPA values: [3.7, 3.2, 3.9, 2.8, 3.5, 3.0]         │    │
    │  │                                                     │    │
    │  │  μ = (3.7+3.2+3.9+2.8+3.5+3.0)/6 = 3.35            │    │
    │  │  σ = sqrt(Σ(x-μ)²/n) ≈ 0.38                        │    │
    │  │                                                     │    │
    │  │  Store: μ=3.35, σ=0.38                              │    │
    │  └─────────────────────────────────────────────────────┘    │
    │                                                             │
    │  Transform (same or new data):                              │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │  x = 3.7                                            │    │
    │  │  z = (3.7 - 3.35) / 0.38 = 0.92                     │    │
    │  │                                                     │    │
    │  │  x = 2.8                                            │    │
    │  │  z = (2.8 - 3.35) / 0.38 = -1.45                    │    │
    │  └─────────────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────────────┘

    Attributes:
        method: str - 'standard' (z-score) or 'minmax'
        feature_name: str - Name of the feature being scaled
        mean_: float - Learned mean (for standard scaling)
        std_: float - Learned standard deviation
        min_: float - Learned minimum (for minmax)
        max_: float - Learned maximum

    Example:
        scaler = NumericScaler(method='standard', feature_name='gpa')
        scaler.fit(gpa_train)
        gpa_scaled = scaler.transform(gpa_test)
    """

    def __init__(self, method: str = 'standard', feature_name: str = 'numeric'):
        """
        Initialize numeric scaler.

        Args:
            method: 'standard' for z-score, 'minmax' for [0,1] scaling
            feature_name: Name for output feature

        Implementation:
            1. Store configuration
            2. Initialize fitted parameters to None
        """
        pass

    def fit(self, X: np.ndarray) -> 'NumericScaler':
        """
        Learn scaling parameters from training data.

        ┌─────────────────────────────────────────────────────────┐
        │  IMPLEMENTATION STEPS                                   │
        │                                                         │
        │  1. Flatten X to 1D if needed                           │
        │  2. Handle missing values (np.nan)                      │
        │  3. If method='standard':                               │
        │     - Compute μ = np.nanmean(X)                         │
        │     - Compute σ = np.nanstd(X)                          │
        │     - If σ ≈ 0, set σ = 1 (avoid division by zero)      │
        │  4. If method='minmax':                                 │
        │     - Compute min_ = np.nanmin(X)                       │
        │     - Compute max_ = np.nanmax(X)                       │
        │     - If max_ ≈ min_, set range = 1                     │
        │  5. Store parameters as instance variables              │
        │  6. Return self                                         │
        └─────────────────────────────────────────────────────────┘
        """
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply learned scaling to data.

        ┌─────────────────────────────────────────────────────────┐
        │  IMPLEMENTATION STEPS                                   │
        │                                                         │
        │  1. Check fit() was called (parameters exist)           │
        │  2. If method='standard':                               │
        │     - z = (X - self.mean_) / self.std_                  │
        │  3. If method='minmax':                                 │
        │     - z = (X - self.min_) / (self.max_ - self.min_)     │
        │  4. Reshape to (n, 1) for concatenation                 │
        │  5. Return transformed array                            │
        └─────────────────────────────────────────────────────────┘
        """
        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in single pass.

        More efficient than separate calls - computes statistics
        while transforming.
        """
        pass

    def get_feature_names(self) -> List[str]:
        """
        Return feature name.

        Returns:
            [self.feature_name] - single element list
        """
        pass

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Convert scaled values back to original scale.

        ┌─────────────────────────────────────────────────────────┐
        │  INVERSE TRANSFORM                                      │
        │                                                         │
        │  Standard: x = z × σ + μ                                │
        │  MinMax:   x = z × (max - min) + min                    │
        │                                                         │
        │  Useful for:                                            │
        │  - Interpreting coefficients in original units          │
        │  - Displaying predictions in human-readable form        │
        └─────────────────────────────────────────────────────────┘
        """
        pass


# =============================================================================
# CATEGORICAL ENCODERS
# =============================================================================

class OneHotEncoder(BaseFeatureTransformer):
    """
    Converts categorical features to one-hot (binary) encoding.

    ┌─────────────────────────────────────────────────────────────┐
    │  ONE-HOT ENCODING VISUALIZATION                             │
    │                                                             │
    │  Input: ['UofT', 'UBC', 'McGill', 'UofT', 'UBC']            │
    │                                                             │
    │  Unique categories: ['McGill', 'UBC', 'UofT'] (sorted)      │
    │                                                             │
    │  Output matrix:                                             │
    │           McGill  UBC  UofT                                 │
    │  ┌─────┬───────┬─────┬──────┐                               │
    │  │ 0   │   0   │  0  │  1   │  ← 'UofT'                     │
    │  │ 1   │   0   │  1  │  0   │  ← 'UBC'                      │
    │  │ 2   │   1   │  0  │  0   │  ← 'McGill'                   │
    │  │ 3   │   0   │  0  │  1   │  ← 'UofT'                     │
    │  │ 4   │   0   │  1  │  0   │  ← 'UBC'                      │
    │  └─────┴───────┴─────┴──────┘                               │
    │                                                             │
    │  Properties:                                                │
    │  - Each row sums to 1 (exactly one category)                │
    │  - k categories → k columns                                 │
    │  - Suitable for neural networks                             │
    └─────────────────────────────────────────────────────────────┘

    Attributes:
        categories_: List[str] - Unique categories learned from fit()
        category_to_idx_: Dict[str, int] - Mapping for fast lookup
        feature_name: str - Base name for generated feature columns
        handle_unknown: str - 'error', 'ignore', or 'encode'
    """

    def __init__(self, feature_name: str = 'category',
                 handle_unknown: str = 'error',
                 categories: Optional[List[str]] = None):
        """
        Initialize one-hot encoder.

        Args:
            feature_name: Base name for output columns
            handle_unknown: How to handle unseen categories
                - 'error': Raise exception
                - 'ignore': Encode as all zeros
                - 'encode': Add new column for unknown
            categories: Pre-specified categories (optional)

        Implementation:
            1. Store configuration
            2. If categories provided, create mapping
            3. Otherwise, wait for fit()
        """
        pass

    def fit(self, X: np.ndarray) -> 'OneHotEncoder':
        """
        Learn unique categories from training data.

        ┌─────────────────────────────────────────────────────────┐
        │  IMPLEMENTATION STEPS                                   │
        │                                                         │
        │  1. Flatten X to 1D array of strings                    │
        │  2. Find unique categories: np.unique(X)                │
        │  3. Sort alphabetically for reproducibility             │
        │  4. Create category → index mapping:                    │
        │     {'McGill': 0, 'UBC': 1, 'UofT': 2}                  │
        │  5. Store as instance variables                         │
        │  6. Return self                                         │
        └─────────────────────────────────────────────────────────┘
        """
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Convert categories to one-hot encoding.

        ┌─────────────────────────────────────────────────────────┐
        │  IMPLEMENTATION STEPS                                   │
        │                                                         │
        │  1. Initialize output: np.zeros((n_samples, n_cats))    │
        │  2. For each sample i:                                  │
        │     a. Get category: cat = X[i]                         │
        │     b. If cat in mapping:                               │
        │        - Get index: j = category_to_idx_[cat]           │
        │        - Set: output[i, j] = 1                          │
        │     c. If cat not in mapping:                           │
        │        - If handle_unknown='error': raise               │
        │        - If handle_unknown='ignore': leave as zeros     │
        │  3. Return output matrix                                │
        └─────────────────────────────────────────────────────────┘

        Performance Note:
            For large datasets, vectorize using np.searchsorted
            instead of Python loop.
        """
        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        pass

    def get_feature_names(self) -> List[str]:
        """
        Return generated feature names.

        Returns:
            ['feature_cat1', 'feature_cat2', ...]

        Example:
            feature_name='university', categories=['UBC', 'UofT']
            → ['university_UBC', 'university_UofT']
        """
        pass


class DummyEncoder(BaseFeatureTransformer):
    """
    Converts categorical features to dummy (k-1) encoding.

    ┌─────────────────────────────────────────────────────────────┐
    │  DUMMY ENCODING (DROP-FIRST)                                │
    │                                                             │
    │  Input: ['UofT', 'UBC', 'McGill', 'UofT', 'UBC']            │
    │                                                             │
    │  Reference category: 'McGill' (first alphabetically)        │
    │                                                             │
    │  Output matrix:                                             │
    │           UBC   UofT                                        │
    │  ┌─────┬──────┬──────┐                                      │
    │  │ 0   │  0   │  1   │  ← 'UofT'                            │
    │  │ 1   │  1   │  0   │  ← 'UBC'                             │
    │  │ 2   │  0   │  0   │  ← 'McGill' (reference: all zeros)   │
    │  │ 3   │  0   │  1   │  ← 'UofT'                            │
    │  │ 4   │  1   │  0   │  ← 'UBC'                             │
    │  └─────┴──────┴──────┘                                      │
    │                                                             │
    │  WHY DROP ONE?                                              │
    │  ─────────────                                              │
    │  With intercept: y = β₀ + β₁x_UBC + β₂x_UofT + ...          │
    │                                                             │
    │  If we included McGill dummy:                               │
    │  x_McGill + x_UBC + x_UofT = 1 (always!)                    │
    │                                                             │
    │  This is PERFECT MULTICOLLINEARITY with intercept!          │
    │  XᵀX would be singular (non-invertible)                     │
    │                                                             │
    │  Solution: Drop one category (reference)                    │
    │  Coefficients are interpreted RELATIVE to reference         │
    │                                                             │
    │  β_UBC = log-odds(UBC) - log-odds(McGill)                   │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, feature_name: str = 'category',
                 drop: str = 'first',
                 handle_unknown: str = 'error'):
        """
        Initialize dummy encoder.

        Args:
            feature_name: Base name for output columns
            drop: Which category to drop
                - 'first': Drop first alphabetically
                - 'last': Drop last alphabetically
                - Specific category name
            handle_unknown: How to handle unseen categories
        """
        pass

    def fit(self, X: np.ndarray) -> 'DummyEncoder':
        """
        Learn categories and determine reference.

        Implementation Steps:
            1. Find unique categories (sorted)
            2. Determine which to drop based on 'drop' parameter
            3. Store remaining categories
            4. Create mapping (skipping dropped category)
        """
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Convert to dummy encoding.

        Same as one-hot but with k-1 columns.
        Reference category encoded as all zeros.
        """
        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        pass

    def get_feature_names(self) -> List[str]:
        """Return feature names (excluding dropped category)."""
        pass


class OrdinalEncoder(BaseFeatureTransformer):
    """
    Converts ordinal categories to integer codes preserving order.

    ┌─────────────────────────────────────────────────────────────┐
    │  ORDINAL ENCODING                                           │
    │                                                             │
    │  Input: ['Grade 10', 'Grade 12', 'Grade 11', 'Grade 12']    │
    │                                                             │
    │  Ordering: 'Grade 10' < 'Grade 11' < 'Grade 12'             │
    │                                                             │
    │  Output: [0, 2, 1, 2]                                       │
    │                                                             │
    │  WHEN TO USE:                                               │
    │  ─────────────                                              │
    │  - Categories have natural ordering                         │
    │  - Distance between categories is meaningful                │
    │  - Want to treat as numeric (linear relationship)           │
    │                                                             │
    │  Examples:                                                  │
    │  - Education level: HS < Bachelor < Master < PhD            │
    │  - Term: Fall < Winter < Summer                             │
    │  - Satisfaction: Poor < Fair < Good < Excellent             │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, feature_name: str = 'ordinal',
                 categories: Optional[List[str]] = None):
        """
        Initialize ordinal encoder.

        Args:
            feature_name: Name for output feature
            categories: Ordered list of categories (MUST provide order)
                        First = 0, Second = 1, etc.
        """
        pass

    def fit(self, X: np.ndarray) -> 'OrdinalEncoder':
        """
        Learn or validate category ordering.

        If categories provided: validate all data in categories
        If not provided: infer ordering from data (alphabetical)
        """
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Convert categories to ordinal integers.

        Returns column of integers, shape (n, 1)
        """
        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        pass

    def get_feature_names(self) -> List[str]:
        """Return [feature_name]."""
        pass


# =============================================================================
# INTERACTION TERM BUILDER
# =============================================================================

class InteractionBuilder:
    """
    Creates interaction terms between features.

    ┌─────────────────────────────────────────────────────────────┐
    │  INTERACTION TERMS MATHEMATICS                              │
    │                                                             │
    │  Given features x₁, x₂, the interaction x₁×x₂ allows       │
    │  the effect of x₁ to depend on the value of x₂.            │
    │                                                             │
    │  Example: GPA effect on admission depends on program        │
    │                                                             │
    │  Without interaction:                                       │
    │      log-odds = β₀ + β₁(GPA) + β₂(Engineering)              │
    │      GPA effect is β₁ for ALL programs                      │
    │                                                             │
    │  With interaction:                                          │
    │      log-odds = β₀ + β₁(GPA) + β₂(Eng) + β₃(GPA×Eng)        │
    │      GPA effect is β₁ for non-engineering                   │
    │      GPA effect is β₁ + β₃ for engineering                  │
    │                                                             │
    │  ┌────────────────────────────────────────────────────┐     │
    │  │  VISUAL: Effect Modification                       │     │
    │  │                                                    │     │
    │  │  log-odds                                          │     │
    │  │     ↑                                              │     │
    │  │     │      ╱ Engineering (steeper slope)           │     │
    │  │     │    ╱                                         │     │
    │  │     │  ╱      ╱ Arts (gentler slope)               │     │
    │  │     │╱      ╱                                      │     │
    │  │     ├─────╱───────────────────→ GPA                │     │
    │  │                                                    │     │
    │  └────────────────────────────────────────────────────┘     │
    └─────────────────────────────────────────────────────────────┘

    Implementation Notes:
        - For numeric × numeric: element-wise multiplication
        - For numeric × categorical: multiply numeric by each dummy
        - For polynomial: use np.power
    """

    def __init__(self, specs: List[InteractionSpec]):
        """
        Initialize interaction builder.

        Args:
            specs: List of InteractionSpec defining terms to create
        """
        pass

    def build_interactions(self, X: np.ndarray,
                           feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Build interaction columns from existing features.

        Args:
            X: Current design matrix, shape (n, p)
            feature_names: Names of columns in X

        Returns:
            Tuple of (interaction_columns, interaction_names)

        ┌─────────────────────────────────────────────────────────┐
        │  IMPLEMENTATION STEPS                                   │
        │                                                         │
        │  For each InteractionSpec:                              │
        │  1. Find indices of specified features in X             │
        │  2. If multiplicative:                                  │
        │     - Multiply columns element-wise                     │
        │     - Name: 'feat1_x_feat2'                             │
        │  3. If polynomial:                                      │
        │     - Raise to specified power                          │
        │     - Name: 'feat1_pow2'                                │
        │  4. Collect all new columns                             │
        │  5. Return stacked array and names                      │
        └─────────────────────────────────────────────────────────┘
        """
        pass

    def _multiply_features(self, X: np.ndarray,
                           indices: List[int]) -> np.ndarray:
        """
        Element-wise multiply multiple columns.

        x_interaction = x₁ × x₂ × ... × xₖ
        """
        pass

    def _polynomial_features(self, X: np.ndarray,
                             index: int, degree: int) -> np.ndarray:
        """
        Create polynomial features up to specified degree.

        Returns columns for x², x³, ..., xᵈ
        (Original x is already in X)
        """
        pass


# =============================================================================
# MAIN DESIGN MATRIX BUILDER
# =============================================================================

class DesignMatrixBuilder:
    """
    Constructs complete design matrix from raw feature dictionaries.

    ┌─────────────────────────────────────────────────────────────┐
    │  DESIGN MATRIX BUILDER PIPELINE                             │
    │                                                             │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │ Input: List of dictionaries                          │   │
    │  │ [{'gpa': 3.7, 'university': 'UofT', 'term': 'F21'},  │   │
    │  │  {'gpa': 3.2, 'university': 'UBC', 'term': 'W22'}]   │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                         │                                    │
    │                         ▼                                    │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │ Step 1: Extract raw feature arrays                   │   │
    │  │                                                      │   │
    │  │ gpa_raw = [3.7, 3.2]                                 │   │
    │  │ uni_raw = ['UofT', 'UBC']                            │   │
    │  │ term_raw = ['F21', 'W22']                            │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                         │                                    │
    │                         ▼                                    │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │ Step 2: Apply transformers                           │   │
    │  │                                                      │   │
    │  │ NumericScaler('gpa') → [0.5, -0.3]                   │   │
    │  │ OneHotEncoder('university') → [[0,1], [1,0]]         │   │
    │  │ OrdinalEncoder('term') → [0, 1]                      │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                         │                                    │
    │                         ▼                                    │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │ Step 3: Build interaction terms                      │   │
    │  │                                                      │   │
    │  │ gpa × uni_UofT → [0.5×0, -0.3×1] = [0, -0.3]         │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                         │                                    │
    │                         ▼                                    │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │ Step 4: Add intercept (optional)                     │   │
    │  │                                                      │   │
    │  │ Prepend column of 1s                                 │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                         │                                    │
    │                         ▼                                    │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │ Step 5: Concatenate all columns                      │   │
    │  │                                                      │   │
    │  │ X = np.hstack([intercept, numeric, categorical,      │   │
    │  │                interactions])                        │   │
    │  │                                                      │   │
    │  │ Output: X ∈ ℝⁿˣᵖ                                     │   │
    │  └──────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘

    Attributes:
        config: DesignMatrixConfig with feature specifications
        transformers_: Dict[str, BaseFeatureTransformer] - fitted transformers
        feature_names_: List[str] - names of all output columns
        n_features_: int - total number of output features
        is_fitted_: bool - whether fit() has been called
    """

    def __init__(self, config: DesignMatrixConfig):
        """
        Initialize design matrix builder.

        Args:
            config: DesignMatrixConfig specifying features and transforms

        Implementation:
            1. Store configuration
            2. Initialize empty transformer dict
            3. Create transformers for each feature spec
        """
        pass

    def fit(self, data: List[Dict[str, Any]]) -> 'DesignMatrixBuilder':
        """
        Learn all transformation parameters from training data.

        Args:
            data: List of feature dictionaries

        Returns:
            self (for method chaining)

        ┌─────────────────────────────────────────────────────────┐
        │  IMPLEMENTATION STEPS                                   │
        │                                                         │
        │  1. For each feature spec:                              │
        │     a. Extract raw values from data dicts               │
        │     b. Create appropriate transformer                   │
        │     c. Call transformer.fit()                           │
        │     d. Store fitted transformer                         │
        │                                                         │
        │  2. Compute total number of output features             │
        │  3. Collect all feature names                           │
        │  4. Set is_fitted_ = True                               │
        │  5. Return self                                         │
        └─────────────────────────────────────────────────────────┘
        """
        pass

    def transform(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Transform data using fitted parameters.

        Args:
            data: List of feature dictionaries

        Returns:
            Design matrix X, shape (n_samples, n_features)

        ┌─────────────────────────────────────────────────────────┐
        │  IMPLEMENTATION STEPS                                   │
        │                                                         │
        │  1. Check is_fitted_ is True                            │
        │  2. For each feature:                                   │
        │     a. Extract raw values                               │
        │     b. Apply fitted transformer.transform()             │
        │     c. Collect transformed columns                      │
        │  3. Build interaction terms                             │
        │  4. Add intercept column if configured                  │
        │  5. Concatenate: np.hstack([intercept, ...features])    │
        │  6. Return design matrix                                │
        └─────────────────────────────────────────────────────────┘
        """
        pass

    def fit_transform(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Fit and transform training data.

        More efficient than separate calls for training data.
        """
        pass

    def get_feature_names(self) -> List[str]:
        """
        Return all output feature names in order.

        Returns:
            ['intercept', 'gpa', 'university_UBC', 'university_UofT', ...]
        """
        pass

    def get_feature_index(self, name: str) -> int:
        """
        Get column index for a feature name.

        Useful for interpreting model coefficients.

        Args:
            name: Feature name to look up

        Returns:
            Column index in design matrix
        """
        pass

    def _extract_feature(self, data: List[Dict[str, Any]],
                         name: str) -> np.ndarray:
        """
        Extract single feature from list of dictionaries.

        Handles missing values according to feature spec.
        """
        pass

    def _handle_missing(self, values: np.ndarray,
                        spec: FeatureSpec) -> np.ndarray:
        """
        Handle missing values according to strategy.

        Strategies:
        - 'mean': Replace with mean of non-missing
        - 'median': Replace with median
        - 'mode': Replace with most common
        - 'zero': Replace with 0
        - 'drop': Will be handled at row level
        """
        pass

    def sparsity_ratio(self, X: np.ndarray) -> float:
        """
        Calculate ratio of zeros in design matrix.

        Returns:
            Number of zeros / total elements

        Used to decide whether to convert to sparse format.
        """
        pass

    def to_sparse(self, X: np.ndarray):
        """
        Convert dense matrix to sparse CSR format.

        Use when sparsity_ratio > config.sparse_threshold.

        Returns:
            scipy.sparse.csr_matrix
        """
        pass


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_design_matrix(X: np.ndarray,
                           feature_names: List[str]) -> Dict[str, Any]:
    """
    Validate design matrix for common issues.

    ┌─────────────────────────────────────────────────────────────┐
    │  VALIDATION CHECKS                                          │
    │                                                             │
    │  1. Check for NaN/Inf values                                │
    │     - Report locations if found                             │
    │                                                             │
    │  2. Check for constant columns                              │
    │     - Zero variance = no predictive power                   │
    │     - Should be removed                                     │
    │                                                             │
    │  3. Check for perfect multicollinearity                     │
    │     - Columns that are linear combinations                  │
    │     - Compute rank(X) vs n_columns                          │
    │                                                             │
    │  4. Check for near-collinearity                             │
    │     - Condition number of XᵀX                               │
    │     - If > 30, regularization recommended                   │
    │                                                             │
    │  5. Check for extreme values                                │
    │     - Values beyond 3 standard deviations                   │
    │     - May indicate data errors                              │
    └─────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix to validate
        feature_names: Names for each column

    Returns:
        Dictionary with validation results:
        {
            'is_valid': bool,
            'has_nan': bool,
            'nan_locations': List[Tuple],
            'constant_columns': List[str],
            'rank': int,
            'expected_rank': int,
            'condition_number': float,
            'extreme_values': Dict[str, List]
        }
    """
    pass


def check_column_rank(X: np.ndarray) -> Tuple[int, bool]:
    """
    Check if design matrix has full column rank.

    ┌─────────────────────────────────────────────────────────────┐
    │  RANK CHECK via SVD                                         │
    │                                                             │
    │  1. Compute SVD: X = UΣVᵀ                                   │
    │  2. Count non-zero singular values                          │
    │  3. rank(X) = count(σᵢ > ε)                                 │
    │                                                             │
    │  Full column rank: rank(X) = number of columns              │
    │  Rank deficient: rank(X) < number of columns                │
    │                                                             │
    │  If rank deficient:                                         │
    │  - Some columns are linear combinations of others           │
    │  - OLS solution is not unique                               │
    │  - Need regularization or column removal                    │
    └─────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix

    Returns:
        Tuple of (rank, is_full_rank)
    """
    pass


def identify_collinear_features(X: np.ndarray,
                                 feature_names: List[str],
                                 threshold: float = 0.95) -> List[Tuple[str, str, float]]:
    """
    Find pairs of highly correlated features.

    ┌─────────────────────────────────────────────────────────────┐
    │  CORRELATION ANALYSIS                                       │
    │                                                             │
    │  Compute pairwise Pearson correlations:                     │
    │                                                             │
    │         ┌──────┬──────┬──────┐                              │
    │         │  x₁  │  x₂  │  x₃  │                              │
    │  ┌──────┼──────┼──────┼──────┤                              │
    │  │  x₁  │ 1.00 │ 0.98 │ 0.12 │ ← x₁, x₂ highly correlated  │
    │  │  x₂  │ 0.98 │ 1.00 │ 0.15 │                              │
    │  │  x₃  │ 0.12 │ 0.15 │ 1.00 │                              │
    │  └──────┴──────┴──────┴──────┘                              │
    │                                                             │
    │  Returns pairs where |correlation| > threshold              │
    └─────────────────────────────────────────────────────────────┘

    Args:
        X: Design matrix
        feature_names: Names of features
        threshold: Correlation threshold (default 0.95)

    Returns:
        List of (feature1, feature2, correlation) tuples
    """
    pass


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def build_admission_matrix(applications: List[Dict[str, Any]],
                            include_interactions: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Build standard design matrix for admission prediction.

    ┌─────────────────────────────────────────────────────────────┐
    │  ADMISSION PREDICTION FEATURES                              │
    │                                                             │
    │  Numeric (standardized):                                    │
    │  - GPA (overall and by subject)                             │
    │  - Application count                                        │
    │  - Days until deadline                                      │
    │                                                             │
    │  Categorical (dummy encoded):                               │
    │  - University                                               │
    │  - Program type                                             │
    │  - Province                                                 │
    │  - Term                                                     │
    │                                                             │
    │  Interactions:                                              │
    │  - GPA × Program type                                       │
    │  - University × Term                                        │
    └─────────────────────────────────────────────────────────────┘

    Args:
        applications: List of application dictionaries
        include_interactions: Whether to add interaction terms

    Returns:
        Tuple of (X, feature_names)
    """
    pass


def create_polynomial_features(X: np.ndarray,
                                degree: int = 2,
                                interaction_only: bool = False) -> Tuple[np.ndarray, List[str]]:
    """
    Generate polynomial features up to specified degree.

    ┌─────────────────────────────────────────────────────────────┐
    │  POLYNOMIAL EXPANSION                                       │
    │                                                             │
    │  Input: [x₁, x₂]                                            │
    │  Degree: 2                                                  │
    │                                                             │
    │  Output (interaction_only=False):                           │
    │  [1, x₁, x₂, x₁², x₁x₂, x₂²]                               │
    │                                                             │
    │  Output (interaction_only=True):                            │
    │  [1, x₁, x₂, x₁x₂]                                         │
    │                                                             │
    │  WARNING: Number of features grows rapidly!                 │
    │  For p features and degree d:                               │
    │  n_output = C(p + d, d) = (p+d)! / (p!d!)                  │
    └─────────────────────────────────────────────────────────────┘

    Args:
        X: Input features, shape (n_samples, n_features)
        degree: Maximum polynomial degree
        interaction_only: If True, only produce interaction terms

    Returns:
        Tuple of (X_poly, feature_names)
    """
    pass


# =============================================================================
# TODO LIST FOR IMPLEMENTATION
# =============================================================================
"""
TODO: Implementation Checklist

CORE TRANSFORMERS:
□ NumericScaler
  - [ ] Implement fit() with mean/std computation
  - [ ] Implement transform() with z-score formula
  - [ ] Handle edge case: std = 0 (constant feature)
  - [ ] Add inverse_transform for interpretability
  - [ ] Add minmax scaling option

□ OneHotEncoder
  - [ ] Implement fit() to learn categories
  - [ ] Implement transform() with vectorized lookup
  - [ ] Handle unknown categories (error/ignore/encode)
  - [ ] Optimize for large category counts

□ DummyEncoder
  - [ ] Extend OneHotEncoder with drop-first logic
  - [ ] Allow specifying which category to drop
  - [ ] Document reference category interpretation

□ OrdinalEncoder
  - [ ] Implement with explicit ordering
  - [ ] Validate all values in specified order
  - [ ] Handle missing values

DESIGN MATRIX BUILDER:
□ DesignMatrixBuilder
  - [ ] Implement fit() to fit all transformers
  - [ ] Implement transform() to apply transformers
  - [ ] Add intercept column handling
  - [ ] Build interaction terms after main features
  - [ ] Generate comprehensive feature names

□ InteractionBuilder
  - [ ] Implement multiplicative interactions
  - [ ] Implement polynomial features
  - [ ] Handle numeric × categorical interactions
  - [ ] Efficient computation for large matrices

VALIDATION:
□ validate_design_matrix()
  - [ ] Check for NaN/Inf values
  - [ ] Detect constant columns
  - [ ] Compute and check rank
  - [ ] Calculate condition number
  - [ ] Flag extreme outliers

□ check_column_rank()
  - [ ] Implement via SVD
  - [ ] Use appropriate numerical tolerance
  - [ ] Report which columns cause rank deficiency

□ identify_collinear_features()
  - [ ] Compute correlation matrix
  - [ ] Find pairs above threshold
  - [ ] Suggest which to remove

SPARSE MATRIX SUPPORT:
□ Sparse matrix handling
  - [ ] Implement to_sparse() using scipy.sparse
  - [ ] Modify transform() to optionally return sparse
  - [ ] Ensure compatibility with model classes

CONVENIENCE FUNCTIONS:
□ build_admission_matrix()
  - [ ] Define standard feature set for admission
  - [ ] Create default configuration
  - [ ] Include common interaction terms

□ create_polynomial_features()
  - [ ] Implement up to degree n
  - [ ] Add interaction_only option
  - [ ] Generate meaningful feature names

TESTING:
□ Unit tests
  - [ ] Test each transformer independently
  - [ ] Test fit/transform consistency
  - [ ] Test with missing values
  - [ ] Test with unknown categories
  - [ ] Verify numerical stability

□ Integration tests
  - [ ] End-to-end pipeline test
  - [ ] Test with real admission data format
  - [ ] Verify compatibility with LogisticModel

DOCUMENTATION:
□ Add examples for each transformer
□ Document numerical stability considerations
□ Add MAT223 references for theory
□ Create tutorial notebook
"""
