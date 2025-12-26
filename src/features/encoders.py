"""
Feature Encoders for Grade Prediction System.

==============================================================================
SYSTEM ARCHITECTURE - WHERE THIS MODULE FITS
==============================================================================

This module provides specialized encoding strategies for different feature
types in the admission prediction pipeline. It complements design_matrix.py
by providing domain-specific encoders for university admissions data.

┌─────────────────────────────────────────────────────────────────────────────┐
│                        ENCODER HIERARCHY                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ┌─────────────────────────────────┐                      │
│                    │     BaseEncoder (Abstract)      │                      │
│                    │                                 │                      │
│                    │  + fit(X) → self                │                      │
│                    │  + transform(X) → array         │                      │
│                    │  + inverse_transform(X)         │                      │
│                    └───────────────┬─────────────────┘                      │
│                                    │                                        │
│         ┌──────────────────────────┼──────────────────────────┐            │
│         │                          │                          │            │
│         ▼                          ▼                          ▼            │
│  ┌──────────────┐        ┌──────────────────┐       ┌─────────────────┐   │
│  │ GPAEncoder   │        │ UniversityEncoder │       │ TermEncoder     │   │
│  │              │        │                   │       │                 │   │
│  │ Handles:     │        │ Handles:          │       │ Handles:        │   │
│  │ - 4.0 scale  │        │ - 100+ unis       │       │ - Cyclical time │   │
│  │ - % scale    │        │ - Hierarchical    │       │ - F/W/S terms   │   │
│  │ - IB/AP      │        │ - Province groups │       │ - Year encoding │   │
│  └──────────────┘        └──────────────────┘       └─────────────────┘   │
│         │                          │                          │            │
│         └──────────────────────────┼──────────────────────────┘            │
│                                    ▼                                        │
│                    ┌─────────────────────────────────┐                      │
│                    │       DesignMatrixBuilder       │                      │
│                    │     (features/design_matrix.py) │                      │
│                    └─────────────────────────────────┘                      │
│                                    │                                        │
│                                    ▼                                        │
│                    ┌─────────────────────────────────┐                      │
│                    │           Models                │                      │
│                    │  LogisticModel / EmbeddingModel │                      │
│                    └─────────────────────────────────┘                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
DOMAIN-SPECIFIC ENCODING CHALLENGES
==============================================================================

University admissions data has unique encoding challenges:

┌─────────────────────────────────────────────────────────────────────────────┐
│  CHALLENGE 1: GPA SCALE INCONSISTENCY                                       │
│                                                                             │
│  Different high schools use different grading scales:                       │
│                                                                             │
│  ┌────────────────┬────────────┬─────────────────┐                          │
│  │ Scale Type     │ Range      │ Examples        │                          │
│  ├────────────────┼────────────┼─────────────────┤                          │
│  │ 4.0 Scale      │ 0.0 - 4.0  │ US high schools │                          │
│  │ Percentage     │ 0 - 100    │ Ontario HS      │                          │
│  │ Letter Grade   │ A+ to F    │ Some provinces  │                          │
│  │ IB Score       │ 1 - 45     │ IB Diploma      │                          │
│  │ AP Score       │ 1 - 5      │ Per course      │                          │
│  └────────────────┴────────────┴─────────────────┘                          │
│                                                                             │
│  Solution: Normalize all to [0, 1] range before standardization             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  CHALLENGE 2: HIGH CARDINALITY CATEGORIES                                   │
│                                                                             │
│  - 100+ Canadian universities                                               │
│  - 1000+ unique programs                                                    │
│  - Long-tail distribution (few students per small program)                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Program Application Counts (Log Scale)                             │    │
│  │                                                                     │    │
│  │  Count                                                              │    │
│  │    │                                                                │    │
│  │ 1K ┤ ██                                                             │    │
│  │    │ ██                                                             │    │
│  │500 ┤ ████                                                           │    │
│  │    │ ██████                                                         │    │
│  │100 ┤ ██████████                                                     │    │
│  │    │ ████████████████                                               │    │
│  │ 10 ┤ ████████████████████████████                                   │    │
│  │    │ ████████████████████████████████████████████████████           │    │
│  │  1 ┤─────────────────────────────────────────────────────           │    │
│  │    └──────────────────────────────────────────────────────→         │    │
│  │      Top 10      Top 50      Top 200        Long tail    Programs   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Solutions:                                                                 │
│  1. Target encoding with regularization                                     │
│  2. Hierarchical grouping (Program → Faculty → University)                  │
│  3. Embedding layers (for neural network models)                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  CHALLENGE 3: CYCLICAL TIME FEATURES                                        │
│                                                                             │
│  Application months and terms are cyclical:                                 │
│                                                                             │
│  Linear encoding (WRONG):                                                   │
│  January = 1, February = 2, ..., December = 12                              │
│                                                                             │
│  Problem: December (12) appears far from January (1)                        │
│           but they're actually adjacent!                                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    CYCLICAL ENCODING                                │    │
│  │                                                                     │    │
│  │           sin(θ)                                                    │    │
│  │             ↑                                                       │    │
│  │             │     March                                             │    │
│  │        Apr ○│ ○ Feb                                                 │    │
│  │           ╲ │ ╱                                                     │    │
│  │      May ○──┼──○ Jan    → cos(θ)                                    │    │
│  │           ╱ │ ╲                                                     │    │
│  │       Jun ○ │  ○ Dec                                                │    │
│  │             │      Nov                                              │    │
│  │             ↓                                                       │    │
│  │                                                                     │    │
│  │  θ = 2π × (month - 1) / 12                                          │    │
│  │  x = cos(θ)                                                         │    │
│  │  y = sin(θ)                                                         │    │
│  │                                                                     │    │
│  │  Now Dec and Jan are adjacent on the circle!                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
TARGET ENCODING THEORY
==============================================================================

Target encoding replaces categories with their mean target value:

┌─────────────────────────────────────────────────────────────────────────────┐
│  TARGET ENCODING (with regularization)                                      │
│                                                                             │
│  Raw data:                                                                  │
│  ┌───────────────┬──────────┬──────────┐                                    │
│  │ University    │ Admitted │  Count   │                                    │
│  ├───────────────┼──────────┼──────────┤                                    │
│  │ UofT          │   142    │   200    │  → 142/200 = 0.71                  │
│  │ UBC           │    85    │   150    │  → 85/150 = 0.57                   │
│  │ SmallUni      │     2    │     3    │  → 2/3 = 0.67 (unreliable!)        │
│  └───────────────┴──────────┴──────────┘                                    │
│                                                                             │
│  Problem: SmallUni has only 3 samples - 0.67 is noisy estimate              │
│                                                                             │
│  Solution: Shrink toward global mean (Bayesian regularization)              │
│                                                                             │
│                 n × μ_category + m × μ_global                               │
│  encoded = ─────────────────────────────────────                            │
│                        n + m                                                │
│                                                                             │
│  Where:                                                                     │
│    n = sample count for category                                            │
│    m = smoothing parameter (e.g., 10)                                       │
│    μ_category = category mean                                               │
│    μ_global = overall mean                                                  │
│                                                                             │
│  For SmallUni (n=3, m=10, μ_cat=0.67, μ_global=0.60):                       │
│  encoded = (3×0.67 + 10×0.60) / (3+10) = 0.62                               │
│                                                                             │
│  Result: SmallUni shrunk toward global mean (more reliable)                 │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
LEAVE-ONE-OUT TARGET ENCODING
==============================================================================

To prevent target leakage in cross-validation:

┌─────────────────────────────────────────────────────────────────────────────┐
│  LEAVE-ONE-OUT (LOO) ENCODING                                               │
│                                                                             │
│  Standard target encoding uses same data for computing and applying         │
│  encoding - this leaks information!                                         │
│                                                                             │
│  LOO Solution: For each row, compute category mean EXCLUDING that row       │
│                                                                             │
│  Example for UofT student i:                                                │
│  ┌───────────┬──────────────┬──────────────┐                                │
│  │ Student   │   Admitted   │   UofT_enc   │                                │
│  ├───────────┼──────────────┼──────────────┤                                │
│  │ 1 (UofT)  │      1       │  (142-1)/199 │  = 0.708 (exclude self)        │
│  │ 2 (UofT)  │      0       │  (142-0)/199 │  = 0.714 (exclude self)        │
│  │ 3 (UBC)   │      1       │   85/150     │  = 0.567 (UBC mean)            │
│  └───────────┴──────────────┴──────────────┘                                │
│                                                                             │
│  For test data: use full training category means (no LOO needed)            │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
CSC148 CONNECTION - ENCODER DESIGN PATTERNS
==============================================================================

The encoder classes follow key OOP patterns from CSC148:

┌─────────────────────────────────────────────────────────────────────────────┐
│  TEMPLATE METHOD PATTERN                                                    │
│                                                                             │
│  BaseEncoder defines the algorithm structure:                               │
│                                                                             │
│  class BaseEncoder:                                                         │
│      def fit_transform(self, X, y=None):                                    │
│          self.fit(X, y)           # Template step 1                         │
│          return self.transform(X)  # Template step 2                        │
│                                                                             │
│  Subclasses implement specific fit() and transform() logic                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  STRATEGY PATTERN                                                           │
│                                                                             │
│  Different encoders are interchangeable strategies:                         │
│                                                                             │
│  def encode_feature(data, encoder: BaseEncoder):                            │
│      # Works with any encoder subclass!                                     │
│      return encoder.fit_transform(data)                                     │
│                                                                             │
│  encode_feature(universities, OneHotEncoder())                              │
│  encode_feature(universities, TargetEncoder())                              │
│  encode_feature(universities, EmbeddingEncoder())                           │
└─────────────────────────────────────────────────────────────────────────────┘


Author: Grade Prediction Team
Course Context: MAT223 (Linear Algebra), CSC148 (OOP), STA257 (Probability)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
import numpy as np


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class GPAScaleConfig:
    """
    Configuration for GPA scale normalization.

    ┌─────────────────────────────────────────────────────────────┐
    │  GPA SCALE DEFINITIONS                                      │
    │                                                             │
    │  scale_type: 'percentage' | '4.0' | 'letter' | 'ib'        │
    │                                                             │
    │  Percentage (Ontario):                                      │
    │    min_value: 0, max_value: 100                             │
    │                                                             │
    │  4.0 Scale:                                                 │
    │    min_value: 0.0, max_value: 4.0                           │
    │    (or 4.3 for A+ schools)                                  │
    │                                                             │
    │  IB:                                                        │
    │    min_value: 1, max_value: 45 (diploma)                    │
    │    or 1-7 per course                                        │
    └─────────────────────────────────────────────────────────────┘
    """
    scale_type: str  # 'percentage', '4.0', 'letter', 'ib'
    min_value: float = 0.0
    max_value: float = 100.0
    letter_mapping: Optional[Dict[str, float]] = None


@dataclass
class TargetEncodingConfig:
    """
    Configuration for target encoding with regularization.

    ┌─────────────────────────────────────────────────────────────┐
    │  TARGET ENCODING PARAMETERS                                 │
    │                                                             │
    │  smoothing: float (default 10)                              │
    │    Higher = more shrinkage toward global mean               │
    │    Lower = more trust in category mean                      │
    │                                                             │
    │  min_samples: int (default 5)                               │
    │    Categories with fewer samples use global mean            │
    │                                                             │
    │  noise_std: float (default 0.0)                             │
    │    Add Gaussian noise to prevent overfitting                │
    │    Only applied during fit_transform, not transform         │
    └─────────────────────────────────────────────────────────────┘
    """
    smoothing: float = 10.0
    min_samples: int = 5
    noise_std: float = 0.0
    use_loo: bool = True  # Leave-one-out encoding


@dataclass
class HierarchicalGrouping:
    """
    Hierarchical structure for category grouping.

    ┌─────────────────────────────────────────────────────────────┐
    │  HIERARCHICAL CATEGORY STRUCTURE                            │
    │                                                             │
    │  Level 0 (finest):  Program                                 │
    │       │                                                     │
    │       ▼                                                     │
    │  Level 1:           Faculty/Department                      │
    │       │                                                     │
    │       ▼                                                     │
    │  Level 2:           University                              │
    │       │                                                     │
    │       ▼                                                     │
    │  Level 3 (coarsest): Province                               │
    │                                                             │
    │  Example:                                                   │
    │  "UofT CS" → "UofT Engineering" → "UofT" → "Ontario"        │
    │                                                             │
    │  For rare programs, we can back off to coarser levels       │
    └─────────────────────────────────────────────────────────────┘
    """
    levels: List[str]  # ['program', 'faculty', 'university', 'province']
    mappings: Dict[str, Dict[str, str]]  # level -> {child: parent}


# =============================================================================
# ABSTRACT BASE ENCODER
# =============================================================================

class BaseEncoder(ABC):
    """
    Abstract base class for all feature encoders.

    ┌─────────────────────────────────────────────────────────────┐
    │  ENCODER LIFECYCLE                                          │
    │                                                             │
    │  1. __init__: Configure encoder parameters                  │
    │       │                                                     │
    │       ▼                                                     │
    │  2. fit(X, y): Learn encoding from training data            │
    │       │        (y needed for target encoding)               │
    │       ▼                                                     │
    │  3. transform(X): Apply learned encoding to any data        │
    │       │                                                     │
    │       ▼                                                     │
    │  4. inverse_transform(X_enc): Reverse encoding (if possible)│
    └─────────────────────────────────────────────────────────────┘

    Implementation Notes:
        - fit() should store all learned parameters
        - transform() should NOT modify stored parameters
        - Always check is_fitted before transform()
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BaseEncoder':
        """
        Learn encoding parameters from training data.

        Args:
            X: Feature values to encode, shape (n_samples,) or (n_samples, 1)
            y: Target values (optional, needed for target encoding)

        Returns:
            self (for method chaining)

        Implementation:
            1. Validate input shapes
            2. Learn encoding parameters
            3. Store as instance attributes
            4. Set is_fitted_ = True
            5. Return self
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply learned encoding to data.

        Args:
            X: Feature values to encode

        Returns:
            Encoded values as numpy array

        Implementation:
            1. Check is_fitted_
            2. Apply stored encoding
            3. Handle unknown values
            4. Return encoded array
        """
        pass

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit and transform in one step.

        For target encoding with LOO, this is NOT equivalent to
        fit() + transform() - it computes leave-one-out values.
        """
        pass

    @abstractmethod
    def inverse_transform(self, X_encoded: np.ndarray) -> np.ndarray:
        """
        Convert encoded values back to original representation.

        Not all encoders support this (e.g., target encoding loses info).
        """
        pass

    @abstractmethod
    def get_feature_names_out(self) -> List[str]:
        """
        Return names of output features.

        For encoders that produce multiple columns (like cyclical),
        return all output column names.
        """
        pass


# =============================================================================
# GPA ENCODER
# =============================================================================

class GPAEncoder(BaseEncoder):
    """
    Encodes GPA values from various scales to normalized [0, 1] range.

    ┌─────────────────────────────────────────────────────────────┐
    │  GPA NORMALIZATION PIPELINE                                 │
    │                                                             │
    │  Input: Raw GPA values in various scales                    │
    │                                                             │
    │  Step 1: Detect or use specified scale                      │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │  If max(values) ≤ 4.5 → 4.0 scale                   │    │
    │  │  If max(values) ≤ 7   → IB course scale             │    │
    │  │  If max(values) ≤ 45  → IB diploma scale            │    │
    │  │  If max(values) ≤ 100 → Percentage scale            │    │
    │  └─────────────────────────────────────────────────────┘    │
    │                                                             │
    │  Step 2: Apply scale-specific normalization                 │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │  Percentage:  normalized = gpa / 100                │    │
    │  │  4.0 Scale:   normalized = gpa / 4.0                │    │
    │  │  IB Diploma:  normalized = (gpa - 1) / 44           │    │
    │  │  Letter:      normalized = letter_to_value(gpa)     │    │
    │  └─────────────────────────────────────────────────────┘    │
    │                                                             │
    │  Step 3: Optional standardization (z-score)                 │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │  z = (normalized - μ) / σ                           │    │
    │  │  Where μ, σ learned from training data              │    │
    │  └─────────────────────────────────────────────────────┘    │
    │                                                             │
    │  Output: Normalized GPA values                              │
    └─────────────────────────────────────────────────────────────┘

    Attributes:
        scale_config: GPAScaleConfig with scale parameters
        standardize: Whether to z-score after normalization
        mean_: Learned mean (if standardize=True)
        std_: Learned std (if standardize=True)
    """

    # Default letter grade to numeric mappings
    DEFAULT_LETTER_MAP = {
        'A+': 0.97, 'A': 0.93, 'A-': 0.90,
        'B+': 0.87, 'B': 0.83, 'B-': 0.80,
        'C+': 0.77, 'C': 0.73, 'C-': 0.70,
        'D+': 0.67, 'D': 0.63, 'D-': 0.60,
        'F': 0.50
    }

    def __init__(self, scale_config: Optional[GPAScaleConfig] = None,
                 standardize: bool = True,
                 auto_detect: bool = True):
        """
        Initialize GPA encoder.

        Args:
            scale_config: Explicit scale configuration (optional)
            standardize: Whether to z-score normalize after scaling
            auto_detect: Automatically detect scale if not specified

        Implementation:
            1. Store configuration
            2. Initialize fitted parameters to None
            3. Set is_fitted_ = False
        """
        pass

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'GPAEncoder':
        """
        Learn GPA encoding parameters from training data.

        ┌─────────────────────────────────────────────────────────┐
        │  IMPLEMENTATION STEPS                                   │
        │                                                         │
        │  1. If auto_detect and no scale_config:                 │
        │     - Analyze X to determine scale type                 │
        │     - Check max values, data range                      │
        │                                                         │
        │  2. Store detected/provided scale parameters            │
        │                                                         │
        │  3. If standardize:                                     │
        │     - Normalize all values to [0,1]                     │
        │     - Compute mean_ and std_ of normalized values       │
        │                                                         │
        │  4. Set is_fitted_ = True                               │
        │  5. Return self                                         │
        └─────────────────────────────────────────────────────────┘
        """
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform GPA values using learned parameters.

        ┌─────────────────────────────────────────────────────────┐
        │  IMPLEMENTATION STEPS                                   │
        │                                                         │
        │  1. Verify is_fitted_                                   │
        │  2. Normalize to [0, 1] using scale parameters          │
        │  3. If standardize:                                     │
        │     - Apply z-score: z = (x - mean_) / std_             │
        │  4. Reshape to (n, 1) for compatibility                 │
        │  5. Return transformed array                            │
        └─────────────────────────────────────────────────────────┘
        """
        pass

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform GPA values."""
        pass

    def inverse_transform(self, X_encoded: np.ndarray) -> np.ndarray:
        """
        Convert encoded GPA back to original scale.

        ┌─────────────────────────────────────────────────────────┐
        │  INVERSE TRANSFORM                                      │
        │                                                         │
        │  1. If standardized:                                    │
        │     - Un-standardize: x = z × std_ + mean_              │
        │  2. De-normalize based on scale:                        │
        │     - Percentage: gpa = normalized × 100                │
        │     - 4.0: gpa = normalized × 4.0                       │
        │  3. Return original scale values                        │
        └─────────────────────────────────────────────────────────┘
        """
        pass

    def get_feature_names_out(self) -> List[str]:
        """Return ['gpa_normalized']."""
        pass

    def _detect_scale(self, X: np.ndarray) -> GPAScaleConfig:
        """
        Auto-detect GPA scale from data.

        Logic:
        - max ≤ 4.5: Likely 4.0 scale
        - max ≤ 7: Likely IB course scale
        - max ≤ 45: Likely IB diploma
        - max ≤ 100: Likely percentage
        - Contains letters: Letter grade
        """
        pass

    def _normalize_percentage(self, X: np.ndarray) -> np.ndarray:
        """Normalize percentage GPA to [0, 1]."""
        pass

    def _normalize_4point(self, X: np.ndarray) -> np.ndarray:
        """Normalize 4.0 scale GPA to [0, 1]."""
        pass

    def _normalize_letter(self, X: np.ndarray) -> np.ndarray:
        """Convert letter grades to [0, 1] using mapping."""
        pass


# =============================================================================
# UNIVERSITY ENCODER
# =============================================================================

class UniversityEncoder(BaseEncoder):
    """
    Encodes university names with various strategies.

    ┌─────────────────────────────────────────────────────────────┐
    │  ENCODING STRATEGIES FOR UNIVERSITIES                       │
    │                                                             │
    │  Strategy 1: One-Hot (for linear models)                    │
    │  ──────────────────────────────────────                     │
    │  Creates k binary columns for k universities                │
    │  Pro: No assumptions about relationships                    │
    │  Con: High dimensionality, no transfer for rare unis        │
    │                                                             │
    │  Strategy 2: Target Encoding (for tree models)              │
    │  ──────────────────────────────────────────────             │
    │  Replaces university with admission rate                    │
    │  Pro: Single column, captures acceptance difficulty         │
    │  Con: Risk of leakage, needs regularization                 │
    │                                                             │
    │  Strategy 3: Hierarchical Grouping                          │
    │  ──────────────────────────────────                         │
    │  Groups universities by region/type                         │
    │  Pro: Reduces dimensionality, enables generalization        │
    │  Con: May lose fine-grained differences                     │
    │                                                             │
    │  Strategy 4: Embedding (for neural networks)                │
    │  ──────────────────────────────────────────                 │
    │  Maps to learned dense vectors                              │
    │  Pro: Learns relationships, handles rare categories         │
    │  Con: Requires more data, less interpretable                │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │  HIERARCHICAL UNIVERSITY GROUPING                           │
    │                                                             │
    │  Level 1 (Provincial):                                      │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │  Ontario: UofT, Waterloo, Western, Queens, McMaster │   │
    │  │  Quebec: McGill, Concordia, Laval, UdeM             │   │
    │  │  BC: UBC, SFU, UVic                                 │   │
    │  │  Alberta: UofA, UCalgary                            │   │
    │  │  ...                                                │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                                                             │
    │  Level 2 (Type):                                            │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │  Medical/Doctoral: UofT, UBC, McGill, Alberta       │   │
    │  │  Comprehensive: Waterloo, Simon Fraser, Victoria    │   │
    │  │  Primarily Undergrad: Mount Allison, Trent          │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                                                             │
    │  Can encode at multiple levels simultaneously               │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, strategy: str = 'target',
                 target_config: Optional[TargetEncodingConfig] = None,
                 grouping: Optional[HierarchicalGrouping] = None,
                 handle_unknown: str = 'global_mean'):
        """
        Initialize university encoder.

        Args:
            strategy: 'onehot', 'target', 'hierarchical', 'frequency'
            target_config: Config for target encoding
            grouping: Hierarchical grouping structure
            handle_unknown: 'error', 'global_mean', 'group_mean'

        Implementation:
            1. Validate strategy
            2. Store configuration
            3. Initialize encoding dictionaries
        """
        pass

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'UniversityEncoder':
        """
        Learn university encoding from training data.

        ┌─────────────────────────────────────────────────────────┐
        │  IMPLEMENTATION BY STRATEGY                             │
        │                                                         │
        │  One-Hot:                                               │
        │  1. Find unique universities                            │
        │  2. Create university → index mapping                   │
        │  3. Store n_categories for transform                    │
        │                                                         │
        │  Target Encoding:                                       │
        │  1. Group by university                                 │
        │  2. Compute mean(y) per university                      │
        │  3. Apply smoothing regularization                      │
        │  4. Store university → encoded_value mapping            │
        │                                                         │
        │  Hierarchical:                                          │
        │  1. For each grouping level                             │
        │  2. Apply target encoding or frequency encoding         │
        │  3. Store mappings for each level                       │
        │                                                         │
        │  Frequency:                                             │
        │  1. Count occurrences of each university                │
        │  2. Compute frequency = count / total                   │
        │  3. Store university → frequency mapping                │
        └─────────────────────────────────────────────────────────┘
        """
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform university names using learned encoding.

        Handles unknown universities according to handle_unknown strategy.
        """
        pass

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit and transform with leave-one-out for target encoding.

        For target encoding with use_loo=True, this computes
        leave-one-out means to prevent overfitting.
        """
        pass

    def inverse_transform(self, X_encoded: np.ndarray) -> np.ndarray:
        """
        Convert encoded values back to university names.

        Only supported for one-hot encoding.
        Target encoding is not invertible (information loss).
        """
        pass

    def get_feature_names_out(self) -> List[str]:
        """
        Return output feature names.

        One-hot: ['university_UofT', 'university_UBC', ...]
        Target: ['university_target']
        Hierarchical: ['university_target', 'province_target', ...]
        """
        pass

    def _compute_target_encoding(self, X: np.ndarray,
                                  y: np.ndarray,
                                  smoothing: float) -> Dict[str, float]:
        """
        Compute target encoding with smoothing.

        ┌─────────────────────────────────────────────────────────┐
        │  SMOOTHED TARGET ENCODING                               │
        │                                                         │
        │           n × μ_category + m × μ_global                 │
        │  encode = ─────────────────────────────────             │
        │                     n + m                               │
        │                                                         │
        │  Where:                                                 │
        │    n = category sample count                            │
        │    m = smoothing parameter                              │
        │    μ_category = category mean                           │
        │    μ_global = overall mean                              │
        └─────────────────────────────────────────────────────────┘
        """
        pass

    def _compute_loo_encoding(self, X: np.ndarray,
                               y: np.ndarray) -> np.ndarray:
        """
        Compute leave-one-out target encoding.

        For each sample, encode using category mean
        computed WITHOUT that sample.
        """
        pass


# =============================================================================
# PROGRAM ENCODER
# =============================================================================

class ProgramEncoder(BaseEncoder):
    """
    Encodes program/major with hierarchical structure.

    ┌─────────────────────────────────────────────────────────────┐
    │  PROGRAM HIERARCHY                                          │
    │                                                             │
    │  Programs have natural hierarchical structure:              │
    │                                                             │
    │  Level 0: Specific Program                                  │
    │           "Computer Science - Data Science Specialist"      │
    │                     │                                       │
    │                     ▼                                       │
    │  Level 1: Department                                        │
    │           "Computer Science"                                │
    │                     │                                       │
    │                     ▼                                       │
    │  Level 2: Faculty                                           │
    │           "Arts & Science" or "Engineering"                 │
    │                     │                                       │
    │                     ▼                                       │
    │  Level 3: Broad Category                                    │
    │           "STEM" or "Arts & Humanities"                     │
    │                                                             │
    │  For rare programs, we back off to coarser levels           │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │  BACKOFF STRATEGY FOR RARE PROGRAMS                         │
    │                                                             │
    │  Sample counts:                                             │
    │  "UofT CS Data Science" : 3 samples    ← Too few!          │
    │  "UofT CS" (department) : 150 samples  ← Use this          │
    │                                                             │
    │  Encoding:                                                  │
    │  1. Check if specific program has min_samples               │
    │  2. If not, backoff to department level                     │
    │  3. If still insufficient, backoff to faculty               │
    │  4. Continue until sufficient samples                       │
    │                                                             │
    │  This provides stable estimates for rare programs           │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, hierarchy: Optional[HierarchicalGrouping] = None,
                 min_samples_per_level: List[int] = None,
                 target_config: Optional[TargetEncodingConfig] = None):
        """
        Initialize program encoder.

        Args:
            hierarchy: Program hierarchy definition
            min_samples_per_level: Minimum samples at each level for target encoding
            target_config: Target encoding configuration
        """
        pass

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'ProgramEncoder':
        """
        Learn program encoding with hierarchical backoff.

        Implementation:
            1. For each program, find its hierarchy path
            2. Compute target encoding at each level
            3. Determine which level to use based on sample counts
            4. Store program → (level, encoded_value) mapping
        """
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform programs using hierarchical encoding.

        For unknown programs, backs off to highest known level
        in hierarchy.
        """
        pass

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform with LOO encoding."""
        pass

    def inverse_transform(self, X_encoded: np.ndarray) -> np.ndarray:
        """Not supported for hierarchical encoding."""
        pass

    def get_feature_names_out(self) -> List[str]:
        """
        Return feature names for each hierarchy level.

        Returns:
            ['program_target', 'department_target', 'faculty_target', ...]
        """
        pass


# =============================================================================
# TERM / TIME ENCODER
# =============================================================================

class TermEncoder(BaseEncoder):
    """
    Encodes academic terms with cyclical representation.

    ┌─────────────────────────────────────────────────────────────┐
    │  ACADEMIC TERM ENCODING                                     │
    │                                                             │
    │  Terms follow cyclical pattern:                             │
    │                                                             │
    │       Fall → Winter → Summer → Fall → ...                   │
    │                                                             │
    │  Simple ordinal encoding loses this structure:              │
    │  Fall=1, Winter=2, Summer=3                                 │
    │                                                             │
    │  Problem: Fall(1) appears far from Summer(3)                │
    │           but they're consecutive!                          │
    │                                                             │
    │  Solution: Cyclical encoding on unit circle                 │
    │                                                             │
    │       sin(θ)                                                │
    │         ↑                                                   │
    │         │     Winter                                        │
    │    Fall ○──────○                                            │
    │         │      │                                            │
    │    ─────┼──────┼─────→ cos(θ)                               │
    │         │      │                                            │
    │         ○──────○ Summer                                     │
    │         │                                                   │
    │         ↓                                                   │
    │                                                             │
    │  θ = 2π × term_index / n_terms                              │
    │  x = cos(θ),  y = sin(θ)                                    │
    └─────────────────────────────────────────────────────────────┘

    Attributes:
        terms: List of term names in order
        use_cyclical: Whether to use sin/cos encoding
        include_year: Whether to include year as separate feature
    """

    STANDARD_TERMS = ['Fall', 'Winter', 'Summer']

    def __init__(self, terms: Optional[List[str]] = None,
                 use_cyclical: bool = True,
                 include_year: bool = True,
                 year_base: int = 2020):
        """
        Initialize term encoder.

        Args:
            terms: Ordered list of terms (default: Fall, Winter, Summer)
            use_cyclical: Use sin/cos encoding for term
            include_year: Include year as separate feature
            year_base: Reference year for year encoding

        Implementation:
            1. Store term ordering
            2. Create term → index mapping
            3. Set encoding parameters
        """
        pass

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'TermEncoder':
        """
        Learn term encoding parameters.

        ┌─────────────────────────────────────────────────────────┐
        │  IMPLEMENTATION STEPS                                   │
        │                                                         │
        │  1. Parse term strings (e.g., "F21", "Fall 2021")       │
        │  2. Validate all terms are in known term list           │
        │  3. If include_year:                                    │
        │     - Extract years from term strings                   │
        │     - Compute year normalization parameters             │
        │  4. Store n_terms for cyclical encoding                 │
        │  5. Return self                                         │
        └─────────────────────────────────────────────────────────┘
        """
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform term strings to encoded values.

        ┌─────────────────────────────────────────────────────────┐
        │  OUTPUT COLUMNS                                         │
        │                                                         │
        │  If use_cyclical=True, include_year=True:               │
        │  - Column 0: cos(θ_term)                                │
        │  - Column 1: sin(θ_term)                                │
        │  - Column 2: normalized_year                            │
        │                                                         │
        │  Example: "Fall 2023"                                   │
        │  - term_idx = 0 (Fall)                                  │
        │  - θ = 2π × 0 / 3 = 0                                   │
        │  - cos(0) = 1, sin(0) = 0                               │
        │  - year = (2023 - 2020) / scale                         │
        │  Output: [1.0, 0.0, 0.3]                                │
        └─────────────────────────────────────────────────────────┘
        """
        pass

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform term values."""
        pass

    def inverse_transform(self, X_encoded: np.ndarray) -> np.ndarray:
        """
        Convert encoded values back to term strings.

        Uses arctan2 to recover term from sin/cos values.
        """
        pass

    def get_feature_names_out(self) -> List[str]:
        """
        Return output feature names.

        Returns:
            ['term_cos', 'term_sin', 'year'] if cyclical
            ['term_ordinal', 'year'] if not cyclical
        """
        pass

    def _parse_term_string(self, term_str: str) -> Tuple[str, int]:
        """
        Parse term string into (term_name, year).

        Handles formats:
        - "F21" → ("Fall", 2021)
        - "Fall 2021" → ("Fall", 2021)
        - "2021-Fall" → ("Fall", 2021)
        """
        pass

    def _cyclical_encode(self, indices: np.ndarray) -> np.ndarray:
        """
        Apply cyclical encoding to term indices.

        Returns array with cos and sin columns.
        """
        pass


# =============================================================================
# MONTH / DATE ENCODER
# =============================================================================

class DateEncoder(BaseEncoder):
    """
    Encodes dates with cyclical and linear components.

    ┌─────────────────────────────────────────────────────────────┐
    │  DATE FEATURE EXTRACTION                                    │
    │                                                             │
    │  From a date, we can extract:                               │
    │                                                             │
    │  1. Cyclical features (repeat each year):                   │
    │     - Month (1-12) → sin/cos                                │
    │     - Day of year (1-365) → sin/cos                         │
    │     - Day of week (0-6) → sin/cos                           │
    │                                                             │
    │  2. Linear features (trend over time):                      │
    │     - Days since reference date                             │
    │     - Year                                                  │
    │                                                             │
    │  3. Binary features:                                        │
    │     - Is weekend                                            │
    │     - Is holiday (if holiday calendar provided)             │
    │                                                             │
    │  For admission prediction, useful features:                 │
    │  - Days until deadline (countdown)                          │
    │  - Application month (cyclical - patterns repeat)           │
    │  - Year trend (admission rates may trend over time)         │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, features: List[str] = None,
                 reference_date: str = '2020-01-01'):
        """
        Initialize date encoder.

        Args:
            features: Which features to extract
                ['month_cyclical', 'day_of_year_cyclical',
                 'days_since_ref', 'year', 'is_weekend']
            reference_date: Reference for linear features
        """
        pass

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'DateEncoder':
        """
        Learn date encoding parameters.

        For standardization of linear features, learns mean/std.
        """
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extract date features from date values.

        Input can be:
        - Datetime objects
        - String dates
        - Unix timestamps
        """
        pass

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform dates."""
        pass

    def inverse_transform(self, X_encoded: np.ndarray) -> np.ndarray:
        """
        Reconstruct dates from encoded features.

        Only possible if sufficient features included.
        """
        pass

    def get_feature_names_out(self) -> List[str]:
        """Return names of extracted features."""
        pass


# =============================================================================
# FREQUENCY ENCODER
# =============================================================================

class FrequencyEncoder(BaseEncoder):
    """
    Encodes categories by their frequency in training data.

    ┌─────────────────────────────────────────────────────────────┐
    │  FREQUENCY ENCODING                                         │
    │                                                             │
    │  Replace each category with its proportion in training data │
    │                                                             │
    │  Training data:                                             │
    │  ┌───────────────┬──────────┬───────────────┐               │
    │  │ University    │  Count   │  Frequency    │               │
    │  ├───────────────┼──────────┼───────────────┤               │
    │  │ UofT          │   200    │  200/500=0.40 │               │
    │  │ UBC           │   150    │  150/500=0.30 │               │
    │  │ McGill        │   100    │  100/500=0.20 │               │
    │  │ Other         │    50    │   50/500=0.10 │               │
    │  └───────────────┴──────────┴───────────────┘               │
    │  Total: 500                                                 │
    │                                                             │
    │  Encoded values:                                            │
    │  UofT → 0.40                                                │
    │  UBC  → 0.30                                                │
    │  ...                                                        │
    │                                                             │
    │  Pros:                                                      │
    │  - Captures popularity/commonality                          │
    │  - Single column output                                     │
    │  - No target leakage (doesn't use y)                        │
    │                                                             │
    │  Cons:                                                      │
    │  - Doesn't capture relationship with target                 │
    │  - Same encoding for different rare categories              │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, normalize: bool = True,
                 handle_unknown: str = 'zero',
                 min_frequency: float = 0.0):
        """
        Initialize frequency encoder.

        Args:
            normalize: If True, frequencies sum to 1
            handle_unknown: 'zero' or 'min_frequency' for unseen categories
            min_frequency: Floor for rare category frequencies
        """
        pass

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'FrequencyEncoder':
        """
        Learn category frequencies from training data.

        Implementation:
            1. Count occurrences of each category
            2. Compute frequencies (optionally normalized)
            3. Apply min_frequency floor
            4. Store category → frequency mapping
        """
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform categories to frequencies."""
        pass

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform."""
        pass

    def inverse_transform(self, X_encoded: np.ndarray) -> np.ndarray:
        """
        Not exactly invertible - returns most likely category.

        For each frequency, returns category with closest frequency.
        """
        pass

    def get_feature_names_out(self) -> List[str]:
        """Return ['category_frequency']."""
        pass


# =============================================================================
# WEIGHT OF EVIDENCE ENCODER
# =============================================================================

class WOEEncoder(BaseEncoder):
    """
    Weight of Evidence encoder for binary classification.

    ┌─────────────────────────────────────────────────────────────┐
    │  WEIGHT OF EVIDENCE (WOE)                                   │
    │                                                             │
    │  From credit scoring, measures predictive power of category │
    │                                                             │
    │            P(X=x | Y=1)                                     │
    │  WOE(x) = ln ─────────────                                  │
    │            P(X=x | Y=0)                                     │
    │                                                             │
    │  Interpretation:                                            │
    │  - WOE > 0: Category associated with positive class         │
    │  - WOE < 0: Category associated with negative class         │
    │  - WOE = 0: Category has no predictive value                │
    │                                                             │
    │  Example:                                                   │
    │  ┌───────────────┬────────┬────────┬────────────────┐       │
    │  │ University    │ Admit  │ Reject │     WOE        │       │
    │  ├───────────────┼────────┼────────┼────────────────┤       │
    │  │ UofT          │  142   │   58   │ ln(142/58)=0.90│       │
    │  │ UBC           │   85   │   65   │ ln(85/65)=0.27 │       │
    │  │ SmallUni      │    5   │   45   │ ln(5/45)=-2.20 │       │
    │  └───────────────┴────────┴────────┴────────────────┘       │
    │                                                             │
    │  Note: Requires adjustment for class imbalance              │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │  INFORMATION VALUE (IV)                                     │
    │                                                             │
    │  Summarizes total predictive power of a feature:            │
    │                                                             │
    │  IV = Σ (P(X=x|Y=1) - P(X=x|Y=0)) × WOE(x)                  │
    │                                                             │
    │  Interpretation:                                            │
    │  - IV < 0.02: Useless predictor                             │
    │  - 0.02 ≤ IV < 0.1: Weak predictor                          │
    │  - 0.1 ≤ IV < 0.3: Medium predictor                         │
    │  - IV ≥ 0.3: Strong predictor                               │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, regularization: float = 0.5,
                 handle_unknown: str = 'zero'):
        """
        Initialize WOE encoder.

        Args:
            regularization: Add to counts to prevent log(0)
            handle_unknown: How to handle unseen categories
        """
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'WOEEncoder':
        """
        Learn WOE values from training data.

        REQUIRES binary target y.

        Implementation:
            1. For each category, count positive and negative
            2. Compute P(X=x|Y=1) and P(X=x|Y=0)
            3. Apply regularization
            4. Compute WOE = ln(P1/P0)
            5. Store category → WOE mapping
            6. Compute and store Information Value
        """
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform categories to WOE values."""
        pass

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        pass

    def inverse_transform(self, X_encoded: np.ndarray) -> np.ndarray:
        """Not invertible - WOE loses category information."""
        pass

    def get_feature_names_out(self) -> List[str]:
        """Return ['category_woe']."""
        pass

    def get_information_value(self) -> float:
        """Return computed Information Value for this feature."""
        pass


# =============================================================================
# COMPOSITE ENCODER
# =============================================================================

class CompositeEncoder(BaseEncoder):
    """
    Combines multiple encoders for the same feature.

    ┌─────────────────────────────────────────────────────────────┐
    │  COMPOSITE ENCODING                                         │
    │                                                             │
    │  Sometimes we want multiple representations:                │
    │                                                             │
    │  University → [target_enc, province_onehot, frequency]      │
    │                                                             │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │ Input: "UofT"                                         │  │
    │  │                                                       │  │
    │  │ ┌─────────────────┐                                   │  │
    │  │ │ TargetEncoder   │ → [0.71]                          │  │
    │  │ └─────────────────┘                                   │  │
    │  │         +                                             │  │
    │  │ ┌─────────────────┐                                   │  │
    │  │ │ ProvinceOneHot  │ → [1, 0, 0, 0]  (Ontario)         │  │
    │  │ └─────────────────┘                                   │  │
    │  │         +                                             │  │
    │  │ ┌─────────────────┐                                   │  │
    │  │ │ FrequencyEnc    │ → [0.40]                          │  │
    │  │ └─────────────────┘                                   │  │
    │  │                                                       │  │
    │  │ Output: [0.71, 1, 0, 0, 0, 0.40]                       │  │
    │  └───────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, encoders: List[BaseEncoder]):
        """
        Initialize composite encoder.

        Args:
            encoders: List of encoders to apply
        """
        pass

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'CompositeEncoder':
        """Fit all encoders."""
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform with all encoders and concatenate.

        Returns horizontally stacked outputs.
        """
        pass

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform with all encoders."""
        pass

    def inverse_transform(self, X_encoded: np.ndarray) -> np.ndarray:
        """Not generally supported for composite."""
        pass

    def get_feature_names_out(self) -> List[str]:
        """Concatenate feature names from all encoders."""
        pass


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_admission_encoders() -> Dict[str, BaseEncoder]:
    """
    Create standard encoder set for admission prediction.

    Returns:
        Dictionary of feature_name → encoder mappings

    ┌─────────────────────────────────────────────────────────────┐
    │  STANDARD ADMISSION ENCODERS                                │
    │                                                             │
    │  'gpa' → GPAEncoder(standardize=True)                       │
    │  'university' → UniversityEncoder(strategy='target')        │
    │  'program' → ProgramEncoder(with hierarchy)                 │
    │  'term' → TermEncoder(cyclical=True)                        │
    │  'province' → OneHotEncoder()                               │
    │  'application_date' → DateEncoder()                         │
    └─────────────────────────────────────────────────────────────┘
    """
    pass


def encode_admission_features(data: List[Dict[str, Any]],
                               target: np.ndarray,
                               encoders: Optional[Dict[str, BaseEncoder]] = None
                               ) -> Tuple[np.ndarray, List[str], Dict[str, BaseEncoder]]:
    """
    Encode all admission features using appropriate encoders.

    Args:
        data: List of application dictionaries
        target: Binary admission outcomes
        encoders: Pre-fitted encoders (optional)

    Returns:
        Tuple of (encoded_matrix, feature_names, fitted_encoders)

    If encoders not provided, creates and fits default encoders.
    If encoders provided, uses them for transform only.
    """
    pass


# =============================================================================
# TODO LIST FOR IMPLEMENTATION
# =============================================================================
"""
TODO: Implementation Checklist

GPA ENCODER:
□ GPAEncoder
  - [ ] Implement scale detection logic
  - [ ] Handle percentage normalization
  - [ ] Handle 4.0 scale normalization
  - [ ] Handle letter grade conversion
  - [ ] Implement z-score standardization
  - [ ] Store and apply fitted parameters
  - [ ] Test with mixed scale data

UNIVERSITY ENCODER:
□ UniversityEncoder
  - [ ] Implement one-hot strategy
  - [ ] Implement target encoding with smoothing
  - [ ] Implement leave-one-out for training
  - [ ] Handle unknown universities
  - [ ] Add frequency encoding option
  - [ ] Test with rare categories

PROGRAM ENCODER:
□ ProgramEncoder
  - [ ] Define program hierarchy structure
  - [ ] Implement hierarchical backoff
  - [ ] Determine optimal level per program
  - [ ] Handle unknown programs via hierarchy
  - [ ] Multi-level output option

TERM ENCODER:
□ TermEncoder
  - [ ] Parse various term string formats
  - [ ] Implement cyclical sin/cos encoding
  - [ ] Extract and encode year
  - [ ] Inverse transform via arctan2
  - [ ] Test with edge cases (year boundaries)

DATE ENCODER:
□ DateEncoder
  - [ ] Parse multiple date formats
  - [ ] Extract cyclical month features
  - [ ] Compute days since reference
  - [ ] Handle timezones appropriately

ADDITIONAL ENCODERS:
□ FrequencyEncoder
  - [ ] Count and normalize frequencies
  - [ ] Handle unknown categories
  - [ ] Apply minimum frequency floor

□ WOEEncoder
  - [ ] Compute WOE values
  - [ ] Apply Laplace smoothing
  - [ ] Calculate Information Value
  - [ ] Validate binary target

□ CompositeEncoder
  - [ ] Combine multiple encoders
  - [ ] Concatenate outputs correctly
  - [ ] Aggregate feature names

TESTING:
□ Unit tests for each encoder
  - [ ] Test fit/transform consistency
  - [ ] Test with missing values
  - [ ] Test with unseen categories
  - [ ] Verify inverse transform where applicable

□ Integration tests
  - [ ] Test with real admission data format
  - [ ] Verify compatibility with DesignMatrixBuilder
  - [ ] Test pipeline end-to-end

DOCUMENTATION:
□ Add usage examples
□ Document encoding strategies and tradeoffs
□ Add STA257 probability theory references
□ Create visualization of encodings
"""
