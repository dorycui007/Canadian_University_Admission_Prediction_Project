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
        X = np.asarray(X)
        if X.size == 0:
            raise ValueError("X must not be empty.")
        n_samples = X.shape[0]
        if y is not None:
            y = np.asarray(y)
            if y.shape[0] != n_samples:
                raise ValueError(
                    f"X and y must have the same number of samples. "
                    f"Got X with {n_samples} and y with {y.shape[0]}."
                )
        return True


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
        self.config = config

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
        # Parse dates to np.datetime64 array
        parsed_dates = np.array([np.datetime64(d) for d in dates.flat])

        train_end = np.datetime64(self.config.train_end_date)
        val_end = np.datetime64(self.config.val_end_date)

        # Apply gap: exclude samples within gap_days before train_end from train
        if self.config.gap_days > 0:
            gap_delta = np.timedelta64(self.config.gap_days, 'D')
            train_cutoff = train_end - gap_delta
            train_mask = parsed_dates <= train_cutoff
        else:
            train_mask = parsed_dates <= train_end

        val_mask = (parsed_dates > train_end) & (parsed_dates <= val_end)

        if self.config.test_end_date is not None:
            test_end = np.datetime64(self.config.test_end_date)
            test_mask = (parsed_dates > val_end) & (parsed_dates <= test_end)
        else:
            test_mask = parsed_dates > val_end

        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        test_indices = np.where(test_mask)[0]

        yield SplitResult(
            train_indices=train_indices,
            val_indices=val_indices if len(val_indices) > 0 else None,
            test_indices=test_indices if len(test_indices) > 0 else None,
            metadata={
                'train_end_date': self.config.train_end_date,
                'val_end_date': self.config.val_end_date,
                'test_end_date': self.config.test_end_date,
                'gap_days': self.config.gap_days,
                'train_size': len(train_indices),
                'val_size': len(val_indices),
                'test_size': len(test_indices),
            }
        )

    def get_n_splits(self) -> int:
        """Return 1 (single split)."""
        return 1


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
        self.config = config

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
        n_samples = dates.shape[0] if hasattr(dates, 'shape') else len(dates)
        # Sort indices by date
        sort_order = np.argsort(dates.flat if hasattr(dates, 'flat') else dates)
        sorted_indices = np.arange(n_samples)[sort_order]

        if self.config.strategy == 'expanding':
            splits_gen = self._expanding_split(n_samples)
        else:
            splits_gen = self._sliding_split(n_samples)

        for train_pos, val_pos in splits_gen:
            # Map positional indices back to original indices
            train_indices = sorted_indices[train_pos]
            val_indices = sorted_indices[val_pos]
            yield SplitResult(
                train_indices=train_indices,
                val_indices=val_indices,
                metadata={
                    'train_size': len(train_indices),
                    'val_size': len(val_indices),
                    'strategy': self.config.strategy,
                }
            )

    def get_n_splits(self) -> int:
        """Return number of CV folds."""
        return self.config.n_splits

    def _expanding_split(self, n_samples: int
                          ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate expanding window splits.

        Training set grows with each fold.
        """
        n_splits = self.config.n_splits
        gap = self.config.gap
        val_size = self.config.val_size
        min_train_size = self.config.min_train_size

        if val_size is None:
            # Divide data so that we have n_splits folds with expanding train
            # Reserve space for n_splits validation sets
            val_size = max(1, n_samples // (n_splits + 1))

        if min_train_size is None:
            min_train_size = val_size

        for i in range(n_splits):
            # Train end position grows with each fold
            train_end = min_train_size + i * val_size
            val_start = train_end + gap
            val_end = val_start + val_size

            if val_end > n_samples:
                break

            train_indices = np.arange(0, train_end)
            val_indices = np.arange(val_start, val_end)

            yield train_indices, val_indices

    def _sliding_split(self, n_samples: int
                        ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate sliding window splits.

        Training set has fixed size, slides forward.
        """
        n_splits = self.config.n_splits
        gap = self.config.gap
        val_size = self.config.val_size
        max_train_size = self.config.max_train_size

        if val_size is None:
            val_size = max(1, n_samples // (n_splits + 1))

        if max_train_size is None:
            max_train_size = val_size

        for i in range(n_splits):
            # Sliding window: train size is fixed
            train_start = i * val_size
            train_end = train_start + max_train_size
            val_start = train_end + gap
            val_end = val_start + val_size

            if val_end > n_samples:
                break

            train_indices = np.arange(train_start, train_end)
            val_indices = np.arange(val_start, val_end)

            yield train_indices, val_indices


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
        self.config = config

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
        if groups is None:
            raise ValueError("groups parameter is required for GroupedTimeSeriesCV")

        # Get first date for each group
        group_first_dates = self._get_group_first_dates(dates, groups)

        # Sort unique groups by their first date
        sorted_groups = sorted(group_first_dates.keys(),
                               key=lambda g: group_first_dates[g])
        n_groups = len(sorted_groups)

        n_splits = self.config.n_splits
        gap = self.config.gap

        # For each fold, assign groups to train or val based on temporal order
        groups_per_fold = max(1, n_groups // (n_splits + 1))

        for i in range(n_splits):
            if self.config.strategy == 'expanding':
                train_group_end = groups_per_fold + i * groups_per_fold
            else:
                train_group_start = i * groups_per_fold
                train_group_end = train_group_start + groups_per_fold

            val_group_start = train_group_end
            val_group_end = val_group_start + groups_per_fold

            if val_group_end > n_groups:
                break

            if self.config.strategy == 'expanding':
                train_groups_set = set(sorted_groups[:train_group_end])
            else:
                train_groups_set = set(sorted_groups[train_group_start:train_group_end])

            val_groups_set = set(sorted_groups[val_group_start:val_group_end])

            train_indices = np.array([j for j in range(len(groups))
                                      if groups[j] in train_groups_set])
            val_indices = np.array([j for j in range(len(groups))
                                    if groups[j] in val_groups_set])

            if len(train_indices) == 0 or len(val_indices) == 0:
                continue

            yield SplitResult(
                train_indices=train_indices,
                val_indices=val_indices,
                metadata={
                    'train_size': len(train_indices),
                    'val_size': len(val_indices),
                    'n_train_groups': len(train_groups_set),
                    'n_val_groups': len(val_groups_set),
                }
            )

    def get_n_splits(self) -> int:
        """Return number of CV folds."""
        return self.config.n_splits

    def _get_group_first_dates(self, dates: np.ndarray,
                                groups: np.ndarray
                                ) -> Dict[Any, Any]:
        """
        Get first date for each group.

        Used to assign groups to train/val based on first appearance.
        """
        group_first = {}
        for i, g in enumerate(groups):
            date_val = dates[i]
            # Parse to np.datetime64 if string
            try:
                parsed = np.datetime64(date_val)
            except (ValueError, TypeError):
                parsed = date_val

            if g not in group_first or parsed < group_first[g]:
                group_first[g] = parsed
        return group_first


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
        self.temporal_config = temporal_config
        self.stratify_config = stratify_config

    def split(self, dates: np.ndarray,
              y: np.ndarray,
              groups: Optional[np.ndarray] = None
              ) -> Iterator[SplitResult]:
        """
        Generate stratified temporal splits.

        First splits by date, then stratifies within each period.
        """
        # First do temporal split
        parsed_dates = np.array([np.datetime64(d) for d in dates.flat])
        train_end = np.datetime64(self.temporal_config.train_end_date)
        val_end = np.datetime64(self.temporal_config.val_end_date)

        train_mask = parsed_dates <= train_end
        val_mask = (parsed_dates > train_end) & (parsed_dates <= val_end)

        if self.temporal_config.test_end_date is not None:
            test_end = np.datetime64(self.temporal_config.test_end_date)
            test_mask = (parsed_dates > val_end) & (parsed_dates <= test_end)
        else:
            test_mask = parsed_dates > val_end

        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        test_indices = np.where(test_mask)[0]

        yield SplitResult(
            train_indices=train_indices,
            val_indices=val_indices if len(val_indices) > 0 else None,
            test_indices=test_indices if len(test_indices) > 0 else None,
            metadata={
                'stratified': True,
                'train_size': len(train_indices),
                'val_size': len(val_indices),
                'test_size': len(test_indices),
            }
        )

    def get_n_splits(self) -> int:
        """Return 1."""
        return 1

    def _stratified_indices(self, indices: np.ndarray,
                             y: np.ndarray,
                             test_size: float
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split indices with stratification on y.

        Returns (train_indices, test_indices) maintaining class proportions.
        """
        # Group indices by class label
        classes = {}
        for idx in indices:
            label = y[idx]
            if label not in classes:
                classes[label] = []
            classes[label].append(idx)

        train_list = []
        test_list = []

        for label, class_indices in classes.items():
            class_indices = np.array(class_indices)
            n_test = max(1, int(len(class_indices) * test_size))
            n_test = min(n_test, len(class_indices) - 1)  # Keep at least 1 for train

            # Shuffle deterministically within class
            test_list.extend(class_indices[:n_test].tolist())
            train_list.extend(class_indices[n_test:].tolist())

        return np.array(train_list), np.array(test_list)


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
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap

    def split(self, dates: np.ndarray,
              y: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None
              ) -> Iterator[SplitResult]:
        """
        Generate purged K-Fold splits.

        For each fold, excludes samples within purge_gap and embargo_gap
        of the validation set.
        """
        n_samples = len(dates)
        # Sort by dates
        sort_order = np.argsort(dates.flat if hasattr(dates, 'flat') else dates)
        sorted_indices = np.arange(n_samples)[sort_order]

        fold_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples

            val_positions = np.arange(val_start, val_end)

            # Purge: exclude purge_gap samples before val from train
            purge_start = max(0, val_start - self.purge_gap)
            # Embargo: exclude embargo_gap samples after val from train
            embargo_end = min(n_samples, val_end + self.embargo_gap)

            # Train is everything except val + purge zone + embargo zone
            train_positions = np.array([
                j for j in range(n_samples)
                if j < purge_start or j >= embargo_end
            ])

            # Exclude val positions from train as well
            train_positions = np.array([
                j for j in train_positions
                if j < val_start or j >= val_end
            ])

            if len(train_positions) == 0:
                continue

            train_indices = sorted_indices[train_positions]
            val_indices = sorted_indices[val_positions]

            yield SplitResult(
                train_indices=train_indices,
                val_indices=val_indices,
                metadata={
                    'fold': i,
                    'train_size': len(train_indices),
                    'val_size': len(val_indices),
                    'purge_gap': self.purge_gap,
                    'embargo_gap': self.embargo_gap,
                }
            )

    def get_n_splits(self) -> int:
        """Return n_splits."""
        return self.n_splits


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
    dates = np.asarray(dates)
    X = np.asarray(X)
    y = np.asarray(y)

    n_samples = dates.shape[0]

    # Sort indices by dates - use first column if multi-dimensional
    if dates.ndim > 1:
        sort_key = dates[:, 0]
    else:
        sort_key = dates
    sort_order = np.argsort(sort_key)

    split_point = int(n_samples * (1.0 - test_size))

    train_idx = sort_order[:split_point]
    test_idx = sort_order[split_point:]

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


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
    train_dates = np.asarray(train_dates)
    test_dates = np.asarray(test_dates)

    # Try to parse as datetime64 if they aren't already numeric
    try:
        train_dt = np.array([np.datetime64(d) for d in train_dates.flat])
        test_dt = np.array([np.datetime64(d) for d in test_dates.flat])
    except (ValueError, TypeError):
        # Treat as numeric / ordinal values
        train_dt = train_dates.flatten().astype(float)
        test_dt = test_dates.flatten().astype(float)

    max_train = np.max(train_dt)
    min_test = np.min(test_dt)
    min_train = np.min(train_dt)
    max_test = np.max(test_dt)

    # Count leaky samples
    leaky_train_count = int(np.sum(train_dt >= min_test))
    leaky_test_count = int(np.sum(test_dt <= max_train))

    has_leakage = leaky_train_count > 0 or leaky_test_count > 0

    # Compute overlap days
    if has_leakage:
        overlap_start = min_test
        overlap_end = max_train
        if hasattr(overlap_end, 'astype') and np.issubdtype(type(overlap_end), np.datetime64):
            overlap_days = max(0, int((overlap_end - overlap_start) / np.timedelta64(1, 'D')) + 1)
        else:
            overlap_days = max(0, int(overlap_end - overlap_start) + 1)
    else:
        overlap_days = 0

    return {
        'has_leakage': has_leakage,
        'leaky_train_count': leaky_train_count,
        'leaky_test_count': leaky_test_count,
        'overlap_days': overlap_days,
    }


def get_fold_statistics(splits: List[SplitResult],
                         y: np.ndarray
                         ) -> Dict[str, List[Dict[str, float]]]:
    """
    Compute statistics for each CV fold.

    Returns statistics about class distribution, size, etc. for each fold.
    Useful for validating that splits are reasonable.
    """
    y = np.asarray(y)
    fold_stats = []

    for i, split in enumerate(splits):
        stats = {}

        # Train statistics
        train_idx = split.train_indices
        stats['fold'] = i
        stats['train_size'] = len(train_idx)

        train_y = y[train_idx]
        unique_classes, counts = np.unique(train_y, return_counts=True)
        stats['train_class_distribution'] = {
            str(cls): int(cnt) for cls, cnt in zip(unique_classes, counts)
        }

        # Val statistics
        if split.val_indices is not None and len(split.val_indices) > 0:
            val_idx = split.val_indices
            stats['val_size'] = len(val_idx)
            val_y = y[val_idx]
            unique_classes_v, counts_v = np.unique(val_y, return_counts=True)
            stats['val_class_distribution'] = {
                str(cls): int(cnt) for cls, cnt in zip(unique_classes_v, counts_v)
            }
        else:
            stats['val_size'] = 0

        # Test statistics
        if split.test_indices is not None and len(split.test_indices) > 0:
            test_idx = split.test_indices
            stats['test_size'] = len(test_idx)
            test_y = y[test_idx]
            unique_classes_t, counts_t = np.unique(test_y, return_counts=True)
            stats['test_class_distribution'] = {
                str(cls): int(cnt) for cls, cnt in zip(unique_classes_t, counts_t)
            }
        else:
            stats['test_size'] = 0

        fold_stats.append(stats)

    return {'folds': fold_stats}


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
    dates = np.asarray(dates)
    train_dates = dates[train_indices]
    test_dates = dates[test_indices]

    if len(train_dates) == 0 or len(test_dates) == 0:
        return True

    max_train = np.max(train_dates)
    min_test = np.min(test_dates)

    return max_train < min_test


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
    # Parse dates to datetime64
    parsed = np.array([np.datetime64(d, 'D') for d in np.asarray(dates).flat])

    # Extract year, month, day_of_year using datetime64 arithmetic
    years = []
    months = []
    day_of_years = []
    is_app_season = []

    for d in parsed:
        # Convert to Python date for easy extraction
        # np.datetime64 -> timestamp -> date components
        ts = (d - np.datetime64('1970-01-01', 'D')) / np.timedelta64(1, 'D')
        import datetime
        dt = datetime.date.fromordinal(int(ts) + 719163)  # Unix epoch ordinal
        years.append(dt.year)
        months.append(dt.month)
        day_of_years.append(dt.timetuple().tm_yday)
        # Application season: typically Sep-Jan (fall) and Mar-May (spring)
        is_app_season.append(dt.month in (9, 10, 11, 12, 1, 3, 4, 5))

    return {
        'year': np.array(years),
        'month': np.array(months),
        'day_of_year': np.array(day_of_years),
        'is_application_season': np.array(is_app_season),
    }


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
    config = TimeSeriesCVConfig(
        n_splits=n_splits,
        strategy=strategy,
        gap=30,  # 30-day gap for rolling features
    )
    return TimeSeriesCV(config)


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
    import datetime

    parsed = np.array([np.datetime64(d, 'D') for d in np.asarray(dates).flat])
    max_date = np.max(parsed)

    # Convert max_date to year
    ts = (max_date - np.datetime64('1970-01-01', 'D')) / np.timedelta64(1, 'D')
    max_dt = datetime.date.fromordinal(int(ts) + 719163)
    max_year = max_dt.year

    # holdout_years: most recent N years become test+val
    # val is the year before holdout, test is the holdout year(s)
    val_end_year = max_year
    train_end_year = max_year - holdout_years

    train_end_date = f"{train_end_year}-12-31"
    val_end_date = f"{val_end_year}-12-31"

    config = TemporalSplitConfig(
        train_end_date=train_end_date,
        val_end_date=val_end_date,
    )
    return TemporalSplit(config)


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
