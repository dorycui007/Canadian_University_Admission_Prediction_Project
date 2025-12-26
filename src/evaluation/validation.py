"""
Validation Strategies for Grade Prediction System.

==============================================================================
SYSTEM ARCHITECTURE - WHERE THIS MODULE FITS
==============================================================================

This module implements data splitting and cross-validation strategies,
with particular focus on TEMPORAL SPLITS required for time-series data
like university admissions.

┌─────────────────────────────────────────────────────────────────────────────┐
│                        VALIDATION IN THE PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      FULL DATASET                                   │   │
│   │  Applications from 2019, 2020, 2021, 2022, 2023, 2024               │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                   VALIDATION STRATEGY                               │   │
│   │                   (This Module)                                     │   │
│   │                                                                     │   │
│   │  Options:                                                           │   │
│   │  • Temporal Split (recommended for time-series)                     │   │
│   │  • Expanding Window                                                 │   │
│   │  • Sliding Window                                                   │   │
│   │  • Stratified K-Fold (only if iid assumption holds)                 │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│               ┌────────────────────┼────────────────────┐                   │
│               │                    │                    │                   │
│               ▼                    ▼                    ▼                   │
│       ┌─────────────┐      ┌─────────────┐      ┌─────────────┐            │
│       │   TRAIN     │      │ VALIDATION  │      │    TEST     │            │
│       │  2019-2021  │      │    2022     │      │  2023-2024  │            │
│       └─────────────┘      └─────────────┘      └─────────────┘            │
│               │                    │                    │                   │
│               ▼                    ▼                    ▼                   │
│       ┌─────────────┐      ┌─────────────┐      ┌─────────────┐            │
│       │  Fit Model  │      │ Tune Hyper- │      │   Final     │            │
│       │             │      │ parameters  │      │  Evaluation │            │
│       └─────────────┘      └─────────────┘      └─────────────┘            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
WHY TEMPORAL SPLITTING IS CRITICAL
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  DATA LEAKAGE IN TIME-SERIES DATA                                           │
│                                                                             │
│  WRONG: Random Split                                                        │
│  ───────────────────                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Timeline: 2019 ──── 2020 ──── 2021 ──── 2022 ──── 2023            │    │
│  │                                                                     │    │
│  │  Random Split:  T  V  T  T  V  T  T  T  V  T  V  T  T  V           │    │
│  │                 │  │  │  │  │  │  │  │  │  │  │  │  │  │           │    │
│  │  Problem:       └──┼──┘  └──┼──┘  └──┼──┘  └──┼──┘  └──┘           │    │
│  │                    │        │        │        │                     │    │
│  │                    ▼        ▼        ▼        ▼                     │    │
│  │               Model trains on FUTURE data to predict PAST!         │    │
│  │               This is CHEATING - model sees the future!            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  RIGHT: Temporal Split                                                      │
│  ─────────────────────                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Timeline: 2019 ──── 2020 ──── 2021 ──── 2022 ──── 2023            │    │
│  │                                                                     │    │
│  │  Temporal Split:  TRAIN  TRAIN  TRAIN │ VALID │  TEST              │    │
│  │                                       │       │                     │    │
│  │  Model trained on past ──────────────►│◄──────│◄── Evaluated       │    │
│  │                                       │       │    on future        │    │
│  │                                       │       │                     │    │
│  │  This respects causality - no future information leaks!            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
TEMPORAL CROSS-VALIDATION STRATEGIES
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  STRATEGY 1: EXPANDING WINDOW                                               │
│                                                                             │
│  Training set grows over time:                                              │
│                                                                             │
│  Fold 1: [2019────────────][2020]                                           │
│           TRAIN              VAL                                            │
│                                                                             │
│  Fold 2: [2019────────────2020][2021]                                       │
│           TRAIN                  VAL                                        │
│                                                                             │
│  Fold 3: [2019────────────2020────────────2021][2022]                       │
│           TRAIN                                  VAL                        │
│                                                                             │
│  Fold 4: [2019────────────2020────────────2021────────────2022][2023]       │
│           TRAIN                                                VAL         │
│                                                                             │
│  Pros: Uses maximum available training data                                 │
│  Cons: Earlier folds have less training data                                │
│        Distribution shift over time not handled                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  STRATEGY 2: SLIDING WINDOW                                                 │
│                                                                             │
│  Fixed-size training window slides forward:                                 │
│                                                                             │
│  Fold 1: [2019────────────2020][2021]                                       │
│           TRAIN (2 years)       VAL                                         │
│                                                                             │
│  Fold 2:      [2020────────────2021][2022]                                  │
│                TRAIN (2 years)       VAL                                    │
│                                                                             │
│  Fold 3:           [2021────────────2022][2023]                             │
│                     TRAIN (2 years)       VAL                               │
│                                                                             │
│  Pros: Handles distribution shift (recent data only)                        │
│  Cons: Discards older data that might be useful                             │
│        Less training data per fold                                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  STRATEGY 3: BLOCKED TIME-SERIES SPLIT                                      │
│                                                                             │
│  Similar to expanding window but with gap to prevent leakage:               │
│                                                                             │
│  Fold 1: [2019────────────][GAP][2021]                                      │
│           TRAIN                   VAL                                       │
│                                                                             │
│  The GAP prevents information leakage from features that might              │
│  include recent history (e.g., 30-day rolling averages).                    │
│                                                                             │
│  Gap size should be ≥ maximum lookback window in features.                  │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
STRATIFICATION IN TEMPORAL SPLITS
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  COMBINING TEMPORAL AND STRATIFIED SAMPLING                                 │
│                                                                             │
│  Problem: Some programs have few applications per year.                     │
│  If we split purely by time, rare programs may be missing from train/val.  │
│                                                                             │
│  Solution: Within each time-based fold, ensure stratification:              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Original 2022 data:                                                │    │
│  │  Program A: 500 applications (balanced)                             │    │
│  │  Program B: 50 applications (rare)                                  │    │
│  │  Program C: 10 applications (very rare)                             │    │
│  │                                                                     │    │
│  │  After stratified validation split within 2022:                     │    │
│  │  Train 2022: A=400, B=40, C=8                                       │    │
│  │  Val 2022:   A=100, B=10, C=2                                       │    │
│  │                                                                     │    │
│  │  Each subset maintains the program distribution                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Note: For very rare categories, may need to group into "Other"            │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
HANDLING LEAKAGE FROM FEATURES
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  FEATURE-LEVEL LEAKAGE PREVENTION                                           │
│                                                                             │
│  Some features computed at prediction time may leak future info:            │
│                                                                             │
│  DANGEROUS FEATURES:                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  ✗ Program acceptance rate (computed from all data)                 │   │
│  │    → Uses future outcomes!                                          │   │
│  │                                                                     │   │
│  │  ✗ University ranking (if it changes year-to-year)                  │   │
│  │    → Future rankings shouldn't inform past predictions              │   │
│  │                                                                     │   │
│  │  ✗ Average GPA of admitted students                                 │   │
│  │    → Computed from outcomes we're trying to predict                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  SAFE APPROACH:                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  ✓ Compute historical features using ONLY past data:                │   │
│  │                                                                     │   │
│  │    For predicting 2023 outcomes:                                    │   │
│  │    - Use 2022 acceptance rates (not 2023)                           │   │
│  │    - Use 2022 rankings                                              │   │
│  │    - Use 2022 GPA statistics                                        │   │
│  │                                                                     │   │
│  │  ✓ Create "point-in-time" features:                                 │   │
│  │    historical_rate_2022 = rate as of December 2022                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
GROUP-AWARE SPLITTING
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  PREVENTING STUDENT OVERLAP BETWEEN SPLITS                                  │
│                                                                             │
│  Problem: Same student may apply multiple years.                            │
│  If student appears in both train and test, model may memorize student.     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Student Alice:                                                    │    │
│  │    2022: Applied to UofT CS (rejected)  ← in TRAIN                 │    │
│  │    2023: Applied to UofT CS (admitted)  ← in TEST                  │    │
│  │                                                                     │    │
│  │  Model might learn Alice-specific patterns from 2022               │    │
│  │  and apply them to predict her 2023 outcome!                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Solutions:                                                                 │
│  1. GroupKFold: Keep all of a student's applications in same split         │
│  2. Block students: Exclude repeat applicants from test set                │
│  3. Accept minor leakage: If repeat rate is low (<5%), may be OK           │
│                                                                             │
│  For admissions, temporal split usually dominates (students change          │
│  between years anyway), but worth monitoring repeat applicant rate.         │
└─────────────────────────────────────────────────────────────────────────────┘


Author: Grade Prediction Team
Course Context: STA257 (Probability), CSC148 (OOP)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Iterator, Generator
from abc import ABC, abstractmethod
import numpy as np


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class TemporalSplitConfig:
    """
    Configuration for temporal train/test split.

    ┌─────────────────────────────────────────────────────────────┐
    │  TEMPORAL SPLIT CONFIGURATION                               │
    │                                                             │
    │  train_end_date: Last date for training data                │
    │  val_end_date: Last date for validation data                │
    │  test_end_date: Last date for test data (optional)          │
    │                                                             │
    │  Example:                                                   │
    │  train_end_date = '2021-12-31'                              │
    │  val_end_date = '2022-12-31'                                │
    │  test_end_date = '2023-12-31'                               │
    │                                                             │
    │  Result:                                                    │
    │  Train: All data before 2022                                │
    │  Val: Data from 2022                                        │
    │  Test: Data from 2023                                       │
    └─────────────────────────────────────────────────────────────┘
    """
    train_end_date: str  # YYYY-MM-DD format
    val_end_date: str
    test_end_date: Optional[str] = None
    date_column: str = 'application_date'
    gap_days: int = 0  # Gap between train and val to prevent leakage


@dataclass
class TimeSeriesCVConfig:
    """
    Configuration for time-series cross-validation.

    ┌─────────────────────────────────────────────────────────────┐
    │  TIME-SERIES CV CONFIGURATION                               │
    │                                                             │
    │  n_splits: Number of CV folds                               │
    │  strategy: 'expanding' or 'sliding'                         │
    │  min_train_size: Minimum training set size                  │
    │  max_train_size: Maximum training size (for sliding)        │
    │  val_size: Validation set size per fold                     │
    │  gap: Gap between train and val                             │
    └─────────────────────────────────────────────────────────────┘
    """
    n_splits: int = 5
    strategy: str = 'expanding'  # 'expanding' or 'sliding'
    min_train_size: Optional[int] = None
    max_train_size: Optional[int] = None  # Only for sliding window
    val_size: Optional[int] = None
    gap: int = 0
    date_column: str = 'application_date'


@dataclass
class StratifiedConfig:
    """
    Configuration for stratification within temporal splits.

    ┌─────────────────────────────────────────────────────────────┐
    │  STRATIFICATION CONFIGURATION                               │
    │                                                             │
    │  stratify_columns: Columns to stratify on                   │
    │    e.g., ['program_type', 'province']                       │
    │                                                             │
    │  min_samples_per_group: Minimum samples for stratification  │
    │    Groups with fewer samples assigned to 'other'            │
    │                                                             │
    │  combine_rare: Whether to combine rare categories           │
    └─────────────────────────────────────────────────────────────┘
    """
    stratify_columns: List[str] = field(default_factory=list)
    min_samples_per_group: int = 5
    combine_rare: bool = True
    other_label: str = '_OTHER_'


@dataclass
class SplitResult:
    """
    Result from a train/test split operation.

    Contains:
        - train_indices: Indices for training set
        - val_indices: Indices for validation set (optional)
        - test_indices: Indices for test set (optional)
        - metadata: Additional info about the split
    """
    train_indices: np.ndarray
    val_indices: Optional[np.ndarray] = None
    test_indices: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class BaseSplitter(ABC):
    """
    Abstract base class for data splitters.

    ┌─────────────────────────────────────────────────────────────┐
    │  SPLITTER INTERFACE                                         │
    │                                                             │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │            BaseSplitter (Abstract)                  │    │
    │  │                                                     │    │
    │  │  + split(X, y, groups) → Iterator[SplitResult]      │    │
    │  │  + get_n_splits() → int                             │    │
    │  │  + validate_inputs(X, y)                            │    │
    │  └─────────────────────────────────────────────────────┘    │
    │                          △                                  │
    │                          │                                  │
    │          ┌───────────────┼───────────────┐                  │
    │          │               │               │                  │
    │  ┌───────┴───────┐ ┌─────┴─────┐ ┌──────┴──────┐           │
    │  │TemporalSplit  │ │TimeSeriesCV│ │GroupKFold   │           │
    │  └───────────────┘ └───────────┘ └─────────────┘           │
    └─────────────────────────────────────────────────────────────┘
    """

    @abstractmethod
    def split(self, X: np.ndarray,
              y: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None
              ) -> Iterator[SplitResult]:
        """
        Generate train/test splits.

        Args:
            X: Feature array or indices
            y: Target array (optional, for stratification)
            groups: Group labels (optional, for group-aware splitting)

        Yields:
            SplitResult for each fold
        """
        pass

    @abstractmethod
    def get_n_splits(self) -> int:
        """Return number of splits."""
        pass

    def validate_inputs(self, X: np.ndarray,
                        y: Optional[np.ndarray] = None) -> None:
        """
        Validate input arrays.

        Checks:
            - X is not empty
            - y has same length as X (if provided)
            - No NaN in indices
        """
        pass


# =============================================================================
# TEMPORAL SPLIT
# =============================================================================

class TemporalSplit(BaseSplitter):
    """
    Simple temporal train/val/test split.

    ┌─────────────────────────────────────────────────────────────┐
    │  TEMPORAL SPLIT USAGE                                       │
    │                                                             │
    │  splitter = TemporalSplit(config)                           │
    │  for split in splitter.split(dates):                        │
    │      train_idx = split.train_indices                        │
    │      val_idx = split.val_indices                            │
    │      test_idx = split.test_indices                          │
    │                                                             │
    │  Only yields ONE split (not cross-validation)               │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, config: TemporalSplitConfig):
        """
        Initialize temporal splitter.

        Args:
            config: TemporalSplitConfig with date boundaries
        """
        pass

    def split(self, dates: np.ndarray,
              y: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None
              ) -> Iterator[SplitResult]:
        """
        Split data based on dates.

        ┌─────────────────────────────────────────────────────────┐
        │  IMPLEMENTATION STEPS                                   │
        │                                                         │
        │  1. Parse config dates to datetime objects              │
        │  2. Parse input dates array to datetime                 │
        │  3. Find indices where:                                 │
        │     - date ≤ train_end_date → train                    │
        │     - train_end_date < date ≤ val_end_date → val       │
        │     - val_end_date < date ≤ test_end_date → test       │
        │  4. If gap_days > 0:                                    │
        │     - Exclude dates within gap from train               │
        │  5. Yield single SplitResult                            │
        └─────────────────────────────────────────────────────────┘

        Args:
            dates: Array of date strings or datetime objects
            y: Not used (for interface compatibility)
            groups: Not used

        Yields:
            Single SplitResult with train/val/test indices
        """
        pass

    def get_n_splits(self) -> int:
        """Return 1 (single split)."""
        pass


# =============================================================================
# TIME SERIES CROSS-VALIDATION
# =============================================================================

class TimeSeriesCV(BaseSplitter):
    """
    Time series cross-validation with expanding or sliding window.

    ┌─────────────────────────────────────────────────────────────┐
    │  TIME SERIES CV EXAMPLE                                     │
    │                                                             │
    │  With n_splits=4, expanding window:                         │
    │                                                             │
    │  Data: [────────────────────────────────────────────]       │
    │             ↓         ↓         ↓         ↓                 │
    │  Fold 1: [TRAIN][VAL]                                       │
    │  Fold 2: [─TRAIN─][VAL]                                     │
    │  Fold 3: [──TRAIN──][VAL]                                   │
    │  Fold 4: [───TRAIN───][VAL]                                 │
    │                                                             │
    │  With n_splits=4, sliding window:                           │
    │                                                             │
    │  Fold 1: [TRAIN][VAL]                                       │
    │  Fold 2:    [TRAIN][VAL]                                    │
    │  Fold 3:       [TRAIN][VAL]                                 │
    │  Fold 4:          [TRAIN][VAL]                              │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, config: TimeSeriesCVConfig):
        """
        Initialize time series cross-validator.

        Args:
            config: TimeSeriesCVConfig with CV parameters
        """
        pass

    def split(self, dates: np.ndarray,
              y: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None
              ) -> Iterator[SplitResult]:
        """
        Generate time series CV splits.

        ┌─────────────────────────────────────────────────────────┐
        │  IMPLEMENTATION STEPS                                   │
        │                                                         │
        │  1. Sort data by date                                   │
        │  2. Compute split points for n_splits folds             │
        │  3. For each fold i:                                    │
        │     - If expanding:                                     │
        │       train = all data up to split point i             │
        │     - If sliding:                                       │
        │       train = last max_train_size before split i       │
        │     - Apply gap between train and val                   │
        │     - val = next val_size samples after gap             │
        │  4. Yield SplitResult for each fold                     │
        └─────────────────────────────────────────────────────────┘

        Args:
            dates: Array of dates for temporal ordering
            y: Target array (optional)
            groups: Not used for basic time series CV

        Yields:
            SplitResult for each CV fold
        """
        pass

    def get_n_splits(self) -> int:
        """Return number of CV folds."""
        pass

    def _expanding_split(self, n_samples: int
                          ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate expanding window splits.

        Training set grows with each fold.
        """
        pass

    def _sliding_split(self, n_samples: int
                        ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate sliding window splits.

        Training set has fixed size, slides forward.
        """
        pass


# =============================================================================
# GROUPED TIME SERIES CV
# =============================================================================

class GroupedTimeSeriesCV(BaseSplitter):
    """
    Time series CV that respects group boundaries (e.g., students).

    ┌─────────────────────────────────────────────────────────────┐
    │  GROUPED TIME SERIES CV                                     │
    │                                                             │
    │  Ensures all samples from same group stay in same split:    │
    │                                                             │
    │  Student A: [App1, App2, App3]  → All in TRAIN or VAL      │
    │  Student B: [App1, App2]        → All in TRAIN or VAL      │
    │  Student C: [App1]              → In TRAIN or VAL           │
    │                                                             │
    │  Group assignment based on FIRST application date:          │
    │  If Student A first applied in 2021, all of A's apps       │
    │  go to whichever split contains 2021.                       │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, config: TimeSeriesCVConfig):
        """
        Initialize grouped time series CV.

        Args:
            config: TimeSeriesCVConfig with CV parameters
        """
        pass

    def split(self, dates: np.ndarray,
              y: Optional[np.ndarray] = None,
              groups: np.ndarray = None
              ) -> Iterator[SplitResult]:
        """
        Generate grouped time series CV splits.

        Args:
            dates: Array of dates
            y: Target array
            groups: Group labels (e.g., student IDs) - REQUIRED

        Yields:
            SplitResult with groups kept together
        """
        pass

    def get_n_splits(self) -> int:
        """Return number of CV folds."""
        pass

    def _get_group_first_dates(self, dates: np.ndarray,
                                groups: np.ndarray
                                ) -> Dict[Any, Any]:
        """
        Get first date for each group.

        Used to assign groups to train/val based on first appearance.
        """
        pass


# =============================================================================
# STRATIFIED TEMPORAL SPLIT
# =============================================================================

class StratifiedTemporalSplit(BaseSplitter):
    """
    Temporal split with stratification within time periods.

    ┌─────────────────────────────────────────────────────────────┐
    │  STRATIFIED TEMPORAL SPLIT                                  │
    │                                                             │
    │  Useful when you need to ensure class balance within each   │
    │  time period's subset.                                      │
    │                                                             │
    │  Example: Within 2022 data, split 80/20 while maintaining   │
    │  the same proportion of admits/rejects in both parts.       │
    │                                                             │
    │  2022 Original:  40% admits, 60% rejects                    │
    │  2022 Train:     40% admits, 60% rejects (80% of data)      │
    │  2022 Val:       40% admits, 60% rejects (20% of data)      │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, temporal_config: TemporalSplitConfig,
                 stratify_config: StratifiedConfig):
        """
        Initialize stratified temporal splitter.

        Args:
            temporal_config: Date-based splitting config
            stratify_config: Stratification config
        """
        pass

    def split(self, dates: np.ndarray,
              y: np.ndarray,
              groups: Optional[np.ndarray] = None
              ) -> Iterator[SplitResult]:
        """
        Generate stratified temporal splits.

        First splits by date, then stratifies within each period.
        """
        pass

    def get_n_splits(self) -> int:
        """Return 1."""
        pass

    def _stratified_indices(self, indices: np.ndarray,
                             y: np.ndarray,
                             test_size: float
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split indices with stratification on y.

        Returns (train_indices, test_indices) maintaining class proportions.
        """
        pass


# =============================================================================
# PURGED CROSS-VALIDATION
# =============================================================================

class PurgedKFold(BaseSplitter):
    """
    K-Fold CV with purging to prevent leakage in time-series.

    ┌─────────────────────────────────────────────────────────────┐
    │  PURGED K-FOLD                                              │
    │                                                             │
    │  Problem with standard K-Fold on time-series:               │
    │  Features computed from surrounding samples may leak info.  │
    │                                                             │
    │  Example: 30-day rolling average of acceptance rates        │
    │  If sample at day 50 is in validation, and samples at       │
    │  days 35-49 are in training, the rolling average at day 50  │
    │  contains information from training samples!                │
    │                                                             │
    │  Solution: PURGE samples near the train/val boundary        │
    │                                                             │
    │  Without purging:                                           │
    │  [─────TRAIN─────][─VAL─][─────TRAIN─────]                  │
    │                  ↑      ↑                                   │
    │                  Leakage zone!                              │
    │                                                             │
    │  With purging:                                              │
    │  [───TRAIN───][PURGE][VAL][PURGE][───TRAIN───]              │
    │              ↑                  ↑                           │
    │              Excluded from train                            │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, n_splits: int = 5,
                 purge_gap: int = 0,
                 embargo_gap: int = 0):
        """
        Initialize purged K-Fold.

        Args:
            n_splits: Number of folds
            purge_gap: Samples to purge BEFORE val (prevent leakage INTO val)
            embargo_gap: Samples to purge AFTER val (prevent leakage FROM val)
        """
        pass

    def split(self, dates: np.ndarray,
              y: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None
              ) -> Iterator[SplitResult]:
        """
        Generate purged K-Fold splits.

        For each fold, excludes samples within purge_gap and embargo_gap
        of the validation set.
        """
        pass

    def get_n_splits(self) -> int:
        """Return n_splits."""
        pass


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def temporal_train_test_split(dates: np.ndarray,
                               X: np.ndarray,
                               y: np.ndarray,
                               test_size: float = 0.2
                               ) -> Tuple[np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray]:
    """
    Simple temporal train/test split.

    ┌─────────────────────────────────────────────────────────────┐
    │  SIMPLE TEMPORAL SPLIT                                      │
    │                                                             │
    │  Splits data so that:                                       │
    │  - All test samples are chronologically after train         │
    │  - test_size determines the proportion of test data         │
    │                                                             │
    │  Example:                                                   │
    │  With test_size=0.2, oldest 80% → train, newest 20% → test  │
    └─────────────────────────────────────────────────────────────┘

    Args:
        dates: Array of dates for ordering
        X: Feature array
        y: Target array
        test_size: Proportion of data for test set

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    pass


def check_temporal_leakage(train_dates: np.ndarray,
                            test_dates: np.ndarray
                            ) -> Dict[str, Any]:
    """
    Check for potential temporal leakage.

    ┌─────────────────────────────────────────────────────────────┐
    │  LEAKAGE DETECTION                                          │
    │                                                             │
    │  Checks:                                                    │
    │  1. Any train dates after earliest test date?              │
    │  2. Any test dates before latest train date?               │
    │  3. Date range overlap percentage                           │
    │                                                             │
    │  Returns warnings if potential leakage detected             │
    └─────────────────────────────────────────────────────────────┘

    Args:
        train_dates: Training set dates
        test_dates: Test set dates

    Returns:
        Dict with:
        - 'has_leakage': bool
        - 'leaky_train_count': Number of train samples after test start
        - 'leaky_test_count': Number of test samples before train end
        - 'overlap_days': Days of overlap
    """
    pass


def get_fold_statistics(splits: List[SplitResult],
                         y: np.ndarray
                         ) -> Dict[str, List[Dict[str, float]]]:
    """
    Compute statistics for each CV fold.

    Returns statistics about class distribution, size, etc. for each fold.
    Useful for validating that splits are reasonable.
    """
    pass


def validate_temporal_consistency(train_indices: np.ndarray,
                                   test_indices: np.ndarray,
                                   dates: np.ndarray
                                   ) -> bool:
    """
    Validate that all train dates come before all test dates.

    Args:
        train_indices: Indices for training set
        test_indices: Indices for test set
        dates: Full array of dates

    Returns:
        True if temporally consistent, False if there's overlap
    """
    pass


def create_date_features(dates: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Create temporal features from dates.

    Returns:
        Dict with:
        - 'year': Year values
        - 'month': Month values
        - 'day_of_year': Day of year
        - 'is_application_season': Boolean for peak season
    """
    pass


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_admission_cv(dates: np.ndarray,
                         n_splits: int = 5,
                         strategy: str = 'expanding'
                         ) -> TimeSeriesCV:
    """
    Create standard CV splitter for admission prediction.

    ┌─────────────────────────────────────────────────────────────┐
    │  RECOMMENDED SETTINGS FOR ADMISSION PREDICTION              │
    │                                                             │
    │  - Use expanding window (more training data is better)      │
    │  - Gap of 30 days between train and val (for rolling feats) │
    │  - Validation size = 1 academic year                        │
    └─────────────────────────────────────────────────────────────┘

    Args:
        dates: Array of application dates
        n_splits: Number of CV folds
        strategy: 'expanding' or 'sliding'

    Returns:
        Configured TimeSeriesCV splitter
    """
    pass


def create_holdout_split(dates: np.ndarray,
                          holdout_years: int = 1
                          ) -> TemporalSplit:
    """
    Create simple holdout split with most recent years as test.

    Args:
        dates: Array of application dates
        holdout_years: Number of recent years for test set

    Returns:
        Configured TemporalSplit
    """
    pass


# =============================================================================
# TODO LIST FOR IMPLEMENTATION
# =============================================================================
"""
TODO: Implementation Checklist

CORE SPLITTERS:
□ TemporalSplit
  - [ ] Parse date strings to datetime
  - [ ] Handle timezone issues
  - [ ] Implement gap_days logic
  - [ ] Return proper SplitResult

□ TimeSeriesCV
  - [ ] Implement expanding window
  - [ ] Implement sliding window
  - [ ] Handle min_train_size and max_train_size
  - [ ] Apply gap between train and val
  - [ ] Edge case: not enough data for n_splits

□ GroupedTimeSeriesCV
  - [ ] Implement group-first-date logic
  - [ ] Keep groups together in splits
  - [ ] Handle groups spanning multiple years

□ StratifiedTemporalSplit
  - [ ] Combine temporal and stratified logic
  - [ ] Handle rare classes properly
  - [ ] Maintain proportions within time periods

□ PurgedKFold
  - [ ] Implement purge logic before val
  - [ ] Implement embargo logic after val
  - [ ] Handle overlapping purge zones

UTILITY FUNCTIONS:
□ temporal_train_test_split()
  - [ ] Sort by dates
  - [ ] Split at correct index
  - [ ] Return arrays in correct order

□ check_temporal_leakage()
  - [ ] Compare date ranges
  - [ ] Count overlap instances
  - [ ] Generate clear warnings

□ validate_temporal_consistency()
  - [ ] Check max(train_dates) < min(test_dates)
  - [ ] Return descriptive result

TESTING:
□ Unit tests
  - [ ] Test each splitter with synthetic data
  - [ ] Test edge cases (small data, single fold)
  - [ ] Verify temporal consistency of all splits
  - [ ] Test stratification preservation

□ Integration tests
  - [ ] Test with real admission date patterns
  - [ ] Verify no leakage in CV workflow
  - [ ] Test with model training pipeline

DOCUMENTATION:
□ Add diagrams for each strategy
□ Document date format requirements
□ Add examples for common use cases
□ Reference STA257 for probability concepts
"""
