"""
Feature Encoders for Grade Prediction System.

==============================================================================
SYSTEM ARCHITECTURE - WHERE THIS MODULE FITS
==============================================================================

This module provides specialized encoding strategies for different feature
types in the admission prediction pipeline. It complements design_matrix.py
by providing domain-specific encoders for university admissions data.

Author: Grade Prediction Team
Course Context: MAT223 (Linear Algebra), CSC148 (OOP), STA257 (Probability)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
import numpy as np
import re
from datetime import datetime, timedelta


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class GPAScaleConfig:
    """
    Configuration for GPA scale normalization.

    Attributes:
        scale_type: 'percentage', '4.0', 'letter', 'ib', 'ib_course'
        min_value: Minimum value on this scale
        max_value: Maximum value on this scale
        letter_mapping: Optional mapping from letter grades to numeric values
    """
    scale_type: str  # 'percentage', '4.0', 'letter', 'ib'
    min_value: float = 0.0
    max_value: float = 100.0
    letter_mapping: Optional[Dict[str, float]] = None


@dataclass
class TargetEncodingConfig:
    """
    Configuration for target encoding with regularization.

    Attributes:
        smoothing: Higher = more shrinkage toward global mean
        min_samples: Categories with fewer samples use global mean
        noise_std: Gaussian noise to prevent overfitting (only during fit_transform)
        use_loo: Whether to use leave-one-out encoding
    """
    smoothing: float = 10.0
    min_samples: int = 5
    noise_std: float = 0.0
    use_loo: bool = True  # Leave-one-out encoding


@dataclass
class HierarchicalGrouping:
    """
    Hierarchical structure for category grouping.

    Attributes:
        levels: List of level names, e.g. ['program', 'faculty', 'university', 'province']
        mappings: level -> {child: parent} mapping dictionaries
    """
    levels: List[str]  # ['program', 'faculty', 'university', 'province']
    mappings: Dict[str, Dict[str, str]]  # level -> {child: parent}


# =============================================================================
# ABSTRACT BASE ENCODER
# =============================================================================

class BaseEncoder(ABC):
    """
    Abstract base class for all feature encoders.

    Subclasses must implement: fit, transform, inverse_transform,
    get_feature_names_out.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BaseEncoder':
        """Learn encoding parameters from training data."""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply learned encoding to data."""
        pass

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    @abstractmethod
    def inverse_transform(self, X_encoded: np.ndarray) -> np.ndarray:
        """Convert encoded values back to original representation."""
        pass

    @abstractmethod
    def get_feature_names_out(self) -> List[str]:
        """Return names of output features."""
        pass


# =============================================================================
# GPA ENCODER
# =============================================================================

class GPAEncoder(BaseEncoder):
    """
    Encodes GPA values from various scales to normalized [0, 1] range,
    with optional z-score standardization.
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
        """
        self.scale_config = scale_config
        self.standardize = standardize
        self.auto_detect = auto_detect
        self.is_fitted_ = False
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'GPAEncoder':
        """Learn GPA encoding parameters from training data."""
        X = np.asarray(X).ravel()

        # Auto-detect or use provided scale config
        if self.scale_config is None and self.auto_detect:
            self.scale_config = self._detect_scale(X)
        elif self.scale_config is None:
            self.scale_config = GPAScaleConfig(scale_type='percentage', min_value=0.0, max_value=100.0)

        # Normalize to [0, 1]
        normalized = self._normalize(X)

        # Compute standardization parameters
        if self.standardize:
            self.mean_ = float(np.mean(normalized))
            self.std_ = float(np.std(normalized))
            if self.std_ == 0:
                self.std_ = 1.0

        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform GPA values using learned parameters."""
        if not self.is_fitted_:
            # Auto-fit on the given data if not yet fitted
            self.fit(X)

        X = np.asarray(X).ravel()
        normalized = self._normalize(X)

        if self.standardize:
            normalized = (normalized - self.mean_) / self.std_

        return normalized.reshape(-1, 1)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform GPA values."""
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X_encoded: np.ndarray) -> np.ndarray:
        """Convert encoded GPA back to original scale."""
        if not self.is_fitted_:
            # Return normalized values as-is when not fitted
            return np.asarray(X_encoded).ravel()

        X_encoded = np.asarray(X_encoded).ravel()

        # Reverse z-score if standardized
        if self.standardize:
            normalized = X_encoded * self.std_ + self.mean_
        else:
            normalized = X_encoded.copy()

        # De-normalize based on scale
        scale_type = self.scale_config.scale_type
        if scale_type == 'percentage':
            return normalized * 100.0
        elif scale_type == '4.0':
            return normalized * self.scale_config.max_value
        elif scale_type == 'ib':
            return normalized * 44.0 + 1.0
        elif scale_type == 'ib_course':
            return normalized * 6.0 + 1.0
        else:
            return normalized

    def get_feature_names_out(self) -> List[str]:
        """Return ['gpa_normalized']."""
        return ['gpa_normalized']

    def _detect_scale(self, X: np.ndarray) -> GPAScaleConfig:
        """Auto-detect GPA scale from data."""
        # Check if data contains string/letter values
        try:
            numeric_vals = X.astype(float)
        except (ValueError, TypeError):
            # Contains non-numeric (letters) - letter grade scale
            return GPAScaleConfig(scale_type='letter', min_value=0.0, max_value=1.0)

        max_val = float(np.max(numeric_vals))

        if max_val <= 4.5:
            return GPAScaleConfig(scale_type='4.0', min_value=0.0, max_value=4.0)
        elif max_val <= 7:
            return GPAScaleConfig(scale_type='ib_course', min_value=1.0, max_value=7.0)
        elif max_val <= 45:
            return GPAScaleConfig(scale_type='ib', min_value=1.0, max_value=45.0)
        else:
            return GPAScaleConfig(scale_type='percentage', min_value=0.0, max_value=100.0)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize X to [0,1] based on scale_config."""
        scale_type = self.scale_config.scale_type
        if scale_type == 'percentage':
            return self._normalize_percentage(X)
        elif scale_type == '4.0':
            return self._normalize_4point(X)
        elif scale_type == 'letter':
            return self._normalize_letter(X)
        elif scale_type == 'ib':
            return (X.astype(float) - 1.0) / 44.0
        elif scale_type == 'ib_course':
            return (X.astype(float) - 1.0) / 6.0
        else:
            return self._normalize_percentage(X)

    def _normalize_percentage(self, X: np.ndarray) -> np.ndarray:
        """Normalize percentage GPA to [0, 1]."""
        return X.astype(float) / 100.0

    def _normalize_4point(self, X: np.ndarray) -> np.ndarray:
        """Normalize 4.0 scale GPA to [0, 1]."""
        max_val = self.scale_config.max_value if self.scale_config else 4.0
        return X.astype(float) / max_val

    def _normalize_letter(self, X: np.ndarray) -> np.ndarray:
        """Convert letter grades to [0, 1] using mapping."""
        mapping = self.DEFAULT_LETTER_MAP
        if self.scale_config and self.scale_config.letter_mapping:
            mapping = self.scale_config.letter_mapping
        result = np.array([mapping.get(str(x).strip(), 0.5) for x in X], dtype=float)
        return result


# =============================================================================
# UNIVERSITY ENCODER
# =============================================================================

class UniversityEncoder(BaseEncoder):
    """
    Encodes university names with various strategies:
    onehot, target, hierarchical, frequency.
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
        """
        valid_strategies = {'onehot', 'target', 'hierarchical', 'frequency'}
        if strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}, got '{strategy}'")
        self.strategy = strategy
        self.target_config = target_config or TargetEncodingConfig()
        self.grouping = grouping
        self.handle_unknown = handle_unknown
        self.is_fitted_ = False
        self.encoding_map_ = {}
        self.categories_ = []
        self.global_mean_ = 0.0
        self.n_categories_ = 0

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'UniversityEncoder':
        """Learn university encoding from training data."""
        X = np.asarray(X).ravel()

        if self.strategy == 'onehot':
            self.categories_ = sorted(list(set(X)))
            self.encoding_map_ = {cat: i for i, cat in enumerate(self.categories_)}
            self.n_categories_ = len(self.categories_)

        elif self.strategy == 'target':
            if y is None:
                # Fall back to frequency encoding when no target available
                total = len(X)
                unique, counts = np.unique(X, return_counts=True)
                self.encoding_map_ = {cat: float(count) / total for cat, count in zip(unique, counts)}
                self.global_mean_ = 1.0 / len(unique) if len(unique) > 0 else 0.0
            else:
                y = np.asarray(y).ravel().astype(float)
                self.global_mean_ = float(np.mean(y))
                self.encoding_map_ = self._compute_target_encoding(X, y, self.target_config.smoothing)

        elif self.strategy == 'frequency':
            total = len(X)
            unique, counts = np.unique(X, return_counts=True)
            self.encoding_map_ = {cat: float(count) / total for cat, count in zip(unique, counts)}
            self.global_mean_ = 1.0 / len(unique) if len(unique) > 0 else 0.0

        elif self.strategy == 'hierarchical':
            # Treat like target encoding with grouping info available
            if y is not None:
                y = np.asarray(y).ravel().astype(float)
                self.global_mean_ = float(np.mean(y))
                self.encoding_map_ = self._compute_target_encoding(X, y, self.target_config.smoothing)
            else:
                # Fall back to frequency
                total = len(X)
                unique, counts = np.unique(X, return_counts=True)
                self.encoding_map_ = {cat: float(count) / total for cat, count in zip(unique, counts)}

        self.categories_ = sorted(list(set(X)))
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform university names using learned encoding."""
        if not self.is_fitted_:
            # Auto-fit with frequency strategy if no y available
            self.fit(X) if self.strategy != 'target' else None
            if not self.is_fitted_:
                # Target encoding requires y; use frequency fallback
                old_strategy = self.strategy
                self.strategy = 'frequency'
                self.fit(X)
                self.strategy = old_strategy

        X = np.asarray(X).ravel()

        if self.strategy == 'onehot':
            result = np.zeros((len(X), self.n_categories_))
            for i, val in enumerate(X):
                if val in self.encoding_map_:
                    result[i, self.encoding_map_[val]] = 1.0
                elif self.handle_unknown == 'error':
                    raise ValueError(f"Unknown category: {val}")
                # else: row stays all zeros
            return result
        else:
            result = np.zeros(len(X))
            for i, val in enumerate(X):
                if val in self.encoding_map_:
                    result[i] = self.encoding_map_[val]
                elif self.handle_unknown == 'global_mean':
                    result[i] = self.global_mean_
                elif self.handle_unknown == 'error':
                    raise ValueError(f"Unknown category: {val}")
                else:
                    result[i] = self.global_mean_
            return result.reshape(-1, 1)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform with leave-one-out for target encoding."""
        X = np.asarray(X).ravel()

        if self.strategy == 'target' and y is not None and self.target_config.use_loo:
            y = np.asarray(y).ravel().astype(float)
            # First do a normal fit to store encoding_map_ for later transforms
            self.fit(X, y)
            # Then compute LOO for the training data itself
            return self._compute_loo_encoding(X, y)
        else:
            self.fit(X, y)
            return self.transform(X)

    def inverse_transform(self, X_encoded: np.ndarray) -> np.ndarray:
        """Convert encoded values back to university names (onehot only)."""
        if not self.is_fitted_:
            return np.asarray(X_encoded).ravel()

        X_encoded = np.asarray(X_encoded)

        if self.strategy == 'onehot':
            if X_encoded.ndim == 2:
                indices = np.argmax(X_encoded, axis=1)
                return np.array([self.categories_[i] for i in indices])
            else:
                return np.array([self.categories_[int(i)] for i in X_encoded])
        else:
            # For target/frequency, find closest match
            inv_map = {v: k for k, v in self.encoding_map_.items()}
            result = []
            for val in X_encoded.ravel():
                closest_key = min(inv_map.keys(), key=lambda k: abs(k - val))
                result.append(inv_map[closest_key])
            return np.array(result)

    def get_feature_names_out(self) -> List[str]:
        """Return output feature names."""
        if not self.is_fitted_:
            # Return sensible defaults even when not fitted
            if self.strategy == 'onehot':
                return []
            elif self.strategy == 'target':
                return ['university_target']
            elif self.strategy == 'frequency':
                return ['university_frequency']
            elif self.strategy == 'hierarchical':
                return ['university_target']
            return ['university_encoded']

        if self.strategy == 'onehot':
            return [f'university_{cat}' for cat in self.categories_]
        elif self.strategy == 'target':
            return ['university_target']
        elif self.strategy == 'frequency':
            return ['university_frequency']
        elif self.strategy == 'hierarchical':
            return ['university_target']
        return ['university_encoded']

    def _compute_target_encoding(self, X: np.ndarray,
                                  y: np.ndarray,
                                  smoothing: float) -> Dict[str, float]:
        """Compute target encoding with smoothing."""
        global_mean = float(np.mean(y))
        encoding = {}

        unique_cats = np.unique(X)
        for cat in unique_cats:
            mask = X == cat
            n = int(np.sum(mask))
            cat_mean = float(np.mean(y[mask]))
            smoothed = (n * cat_mean + smoothing * global_mean) / (n + smoothing)
            encoding[cat] = smoothed

        return encoding

    def _compute_loo_encoding(self, X: np.ndarray,
                               y: np.ndarray) -> np.ndarray:
        """Compute leave-one-out target encoding."""
        X = np.asarray(X).ravel()
        y = np.asarray(y).ravel().astype(float)
        global_mean = float(np.mean(y))
        smoothing = self.target_config.smoothing

        # Precompute category sums and counts
        cat_sum = {}
        cat_count = {}
        for cat in np.unique(X):
            mask = X == cat
            cat_sum[cat] = float(np.sum(y[mask]))
            cat_count[cat] = int(np.sum(mask))

        result = np.zeros(len(X))
        for i in range(len(X)):
            cat = X[i]
            n = cat_count[cat]
            if n > 1:
                loo_sum = cat_sum[cat] - y[i]
                loo_count = n - 1
                loo_mean = loo_sum / loo_count
                result[i] = (loo_count * loo_mean + smoothing * global_mean) / (loo_count + smoothing)
            else:
                # Only one sample in category, use global mean
                result[i] = global_mean

        return result.reshape(-1, 1)


# =============================================================================
# PROGRAM ENCODER
# =============================================================================

class ProgramEncoder(BaseEncoder):
    """
    Encodes program/major with hierarchical structure.
    Implemented as target encoding with optional hierarchy support.
    """

    def __init__(self, hierarchy: Optional[HierarchicalGrouping] = None,
                 min_samples_per_level: List[int] = None,
                 target_config: Optional[TargetEncodingConfig] = None):
        """
        Initialize program encoder.

        Args:
            hierarchy: Program hierarchy definition
            min_samples_per_level: Minimum samples at each level
            target_config: Target encoding configuration
        """
        self.hierarchy = hierarchy
        self.min_samples_per_level = min_samples_per_level or [5]
        self.target_config = target_config or TargetEncodingConfig()
        self.is_fitted_ = False
        self.encoding_map_ = {}
        self.global_mean_ = 0.0
        self.categories_ = []

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'ProgramEncoder':
        """Learn program encoding with hierarchical backoff."""
        X = np.asarray(X).ravel()
        self.categories_ = sorted(list(set(X)))

        if y is not None:
            y = np.asarray(y).ravel().astype(float)
            self.global_mean_ = float(np.mean(y))
            smoothing = self.target_config.smoothing

            unique_cats = np.unique(X)
            for cat in unique_cats:
                mask = X == cat
                n = int(np.sum(mask))
                cat_mean = float(np.mean(y[mask]))
                smoothed = (n * cat_mean + smoothing * self.global_mean_) / (n + smoothing)
                self.encoding_map_[cat] = smoothed
        else:
            # Frequency-based fallback
            total = len(X)
            unique, counts = np.unique(X, return_counts=True)
            self.encoding_map_ = {cat: float(count) / total for cat, count in zip(unique, counts)}
            self.global_mean_ = 1.0 / len(unique) if len(unique) > 0 else 0.0

        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform programs using hierarchical encoding."""
        if not self.is_fitted_:
            self.fit(X)

        X = np.asarray(X).ravel()
        result = np.zeros(len(X))
        for i, val in enumerate(X):
            if val in self.encoding_map_:
                result[i] = self.encoding_map_[val]
            else:
                result[i] = self.global_mean_
        return result.reshape(-1, 1)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform with LOO encoding."""
        X = np.asarray(X).ravel()

        if y is not None and self.target_config.use_loo:
            y = np.asarray(y).ravel().astype(float)
            self.fit(X, y)
            # LOO encoding
            global_mean = self.global_mean_
            smoothing = self.target_config.smoothing

            cat_sum = {}
            cat_count = {}
            for cat in np.unique(X):
                mask = X == cat
                cat_sum[cat] = float(np.sum(y[mask]))
                cat_count[cat] = int(np.sum(mask))

            result = np.zeros(len(X))
            for i in range(len(X)):
                cat = X[i]
                n = cat_count[cat]
                if n > 1:
                    loo_sum = cat_sum[cat] - y[i]
                    loo_count = n - 1
                    loo_mean = loo_sum / loo_count
                    result[i] = (loo_count * loo_mean + smoothing * global_mean) / (loo_count + smoothing)
                else:
                    result[i] = global_mean

            return result.reshape(-1, 1)
        else:
            self.fit(X, y)
            return self.transform(X)

    def inverse_transform(self, X_encoded: np.ndarray) -> np.ndarray:
        """Not supported for hierarchical encoding - returns closest category."""
        X_encoded = np.asarray(X_encoded).ravel()
        if not self.encoding_map_:
            return X_encoded

        inv_map = {v: k for k, v in self.encoding_map_.items()}
        result = []
        for val in X_encoded:
            closest_key = min(inv_map.keys(), key=lambda k: abs(k - val))
            result.append(inv_map[closest_key])
        return np.array(result)

    def get_feature_names_out(self) -> List[str]:
        """Return feature names for each hierarchy level."""
        if self.hierarchy:
            return [f'{level}_target' for level in self.hierarchy.levels]
        return ['program_target']


# =============================================================================
# TERM / TIME ENCODER
# =============================================================================

class TermEncoder(BaseEncoder):
    """
    Encodes academic terms with cyclical representation.
    """

    STANDARD_TERMS = ['Fall', 'Winter', 'Summer']

    # Short-form abbreviations
    TERM_ABBREVS = {
        'F': 'Fall', 'W': 'Winter', 'S': 'Summer',
        'Fa': 'Fall', 'Wi': 'Winter', 'Su': 'Summer',
    }

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
        """
        self.terms = terms or self.STANDARD_TERMS
        self.use_cyclical = use_cyclical
        self.include_year = include_year
        self.year_base = year_base
        self.term_to_idx = {t: i for i, t in enumerate(self.terms)}
        self.n_terms = len(self.terms)
        self.is_fitted_ = False
        self.year_mean_ = 0.0
        self.year_std_ = 1.0

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'TermEncoder':
        """Learn term encoding parameters."""
        X = np.asarray(X).ravel()

        years = []
        for term_str in X:
            term_name, year = self._parse_term_string(str(term_str))
            years.append(year)

        years = np.array(years, dtype=float)
        if self.include_year and len(years) > 0:
            self.year_mean_ = float(np.mean(years))
            self.year_std_ = float(np.std(years))
            if self.year_std_ == 0:
                self.year_std_ = 1.0

        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform term strings to encoded values."""
        if not self.is_fitted_:
            self.fit(X)

        X = np.asarray(X).ravel()
        indices = []
        years = []

        for term_str in X:
            term_name, year = self._parse_term_string(str(term_str))
            idx = self.term_to_idx.get(term_name, 0)
            indices.append(idx)
            years.append(year)

        indices = np.array(indices, dtype=float)
        years = np.array(years, dtype=float)

        if self.use_cyclical:
            cyclical = self._cyclical_encode(indices)
            if self.include_year:
                year_normalized = (years - self.year_base) / max(self.year_std_, 1.0)
                return np.column_stack([cyclical, year_normalized])
            return cyclical
        else:
            ordinal = indices.reshape(-1, 1)
            if self.include_year:
                year_normalized = ((years - self.year_base) / max(self.year_std_, 1.0)).reshape(-1, 1)
                return np.column_stack([ordinal, year_normalized])
            return ordinal

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform term values."""
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X_encoded: np.ndarray) -> np.ndarray:
        """Convert encoded values back to term strings using arctan2."""
        X_encoded = np.asarray(X_encoded)
        if X_encoded.ndim == 1:
            X_encoded = X_encoded.reshape(1, -1)

        results = []
        for row in X_encoded:
            if self.use_cyclical:
                cos_val = row[0]
                sin_val = row[1]
                angle = np.arctan2(sin_val, cos_val)
                if angle < 0:
                    angle += 2 * np.pi
                idx = int(np.round(angle * self.n_terms / (2 * np.pi))) % self.n_terms
                term_name = self.terms[idx]

                if self.include_year and len(row) > 2:
                    year = int(np.round(row[2] * max(self.year_std_, 1.0) + self.year_base))
                    results.append(f"{term_name} {year}")
                else:
                    results.append(term_name)
            else:
                idx = int(np.round(row[0])) % self.n_terms
                term_name = self.terms[idx]
                if self.include_year and len(row) > 1:
                    year = int(np.round(row[1] * max(self.year_std_, 1.0) + self.year_base))
                    results.append(f"{term_name} {year}")
                else:
                    results.append(term_name)

        return np.array(results)

    def get_feature_names_out(self) -> List[str]:
        """Return output feature names."""
        if self.use_cyclical:
            names = ['term_cos', 'term_sin']
        else:
            names = ['term_ordinal']
        if self.include_year:
            names.append('year')
        return names

    def _parse_term_string(self, term_str: str) -> Tuple[str, int]:
        """
        Parse term string into (term_name, year).

        Handles: "F21", "Fall 2021", "2021-Fall", "Winter 2024"
        """
        term_str = term_str.strip()

        # Try "Fall 2023" format
        match = re.match(r'^(Fall|Winter|Summer)\s+(\d{4})$', term_str, re.IGNORECASE)
        if match:
            term_name = match.group(1).capitalize()
            # Ensure first letter uppercase
            if term_name.lower() == 'fall':
                term_name = 'Fall'
            elif term_name.lower() == 'winter':
                term_name = 'Winter'
            elif term_name.lower() == 'summer':
                term_name = 'Summer'
            year = int(match.group(2))
            return (term_name, year)

        # Try "2021-Fall" format
        match = re.match(r'^(\d{4})[-/](Fall|Winter|Summer)$', term_str, re.IGNORECASE)
        if match:
            year = int(match.group(1))
            term_name = match.group(2).capitalize()
            if term_name.lower() == 'fall':
                term_name = 'Fall'
            elif term_name.lower() == 'winter':
                term_name = 'Winter'
            elif term_name.lower() == 'summer':
                term_name = 'Summer'
            return (term_name, year)

        # Try short form "F21", "W24", "S23"
        match = re.match(r'^([FWSfws])(\d{2})$', term_str)
        if match:
            abbrev = match.group(1).upper()
            year_short = int(match.group(2))
            year = 2000 + year_short
            term_name = self.TERM_ABBREVS.get(abbrev, 'Fall')
            return (term_name, year)

        # Try longer short form "Fa21", "Wi24", "Su23"
        match = re.match(r'^(Fa|Wi|Su)(\d{2})$', term_str, re.IGNORECASE)
        if match:
            abbrev = match.group(1).capitalize()
            year_short = int(match.group(2))
            year = 2000 + year_short
            term_name = self.TERM_ABBREVS.get(abbrev, 'Fall')
            return (term_name, year)

        # Default: try to extract any term name and year
        for term in self.terms:
            if term.lower() in term_str.lower():
                # Try to find a year
                year_match = re.search(r'(\d{4})', term_str)
                if year_match:
                    return (term, int(year_match.group(1)))
                year_match = re.search(r'(\d{2})', term_str)
                if year_match:
                    return (term, 2000 + int(year_match.group(1)))
                return (term, self.year_base)

        return (self.terms[0], self.year_base)

    def _cyclical_encode(self, indices: np.ndarray) -> np.ndarray:
        """Apply cyclical encoding to term indices."""
        theta = 2.0 * np.pi * indices / self.n_terms
        return np.column_stack([np.cos(theta), np.sin(theta)])


# =============================================================================
# MONTH / DATE ENCODER
# =============================================================================

class DateEncoder(BaseEncoder):
    """
    Encodes dates with cyclical and linear components.
    """

    def __init__(self, features: List[str] = None,
                 reference_date: str = '2020-01-01'):
        """
        Initialize date encoder.

        Args:
            features: Which features to extract
                ['month_cyclical', 'days_since_ref', 'year']
            reference_date: Reference for linear features
        """
        self.features = features or ['month_cyclical', 'days_since_ref', 'year']
        self.reference_date = datetime.strptime(reference_date, '%Y-%m-%d')
        self.is_fitted_ = False
        self.days_mean_ = 0.0
        self.days_std_ = 1.0

    def _parse_date(self, date_str: str) -> datetime:
        """Parse a date string into a datetime object."""
        date_str = str(date_str).strip()
        for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', '%B %d, %Y'):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse date: {date_str}")

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'DateEncoder':
        """Learn date encoding parameters."""
        X = np.asarray(X).ravel()

        if 'days_since_ref' in self.features:
            days = []
            for d in X:
                dt = self._parse_date(str(d))
                days.append((dt - self.reference_date).days)
            days = np.array(days, dtype=float)
            self.days_mean_ = float(np.mean(days))
            self.days_std_ = float(np.std(days))
            if self.days_std_ == 0:
                self.days_std_ = 1.0

        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract date features from date values."""
        if not self.is_fitted_:
            self.fit(X)

        X = np.asarray(X).ravel()
        columns = []

        dates = [self._parse_date(str(d)) for d in X]

        if 'month_cyclical' in self.features:
            months = np.array([d.month for d in dates], dtype=float)
            theta = 2.0 * np.pi * (months - 1) / 12.0
            columns.append(np.cos(theta))
            columns.append(np.sin(theta))

        if 'day_of_year_cyclical' in self.features:
            doy = np.array([d.timetuple().tm_yday for d in dates], dtype=float)
            theta = 2.0 * np.pi * (doy - 1) / 365.0
            columns.append(np.cos(theta))
            columns.append(np.sin(theta))

        if 'days_since_ref' in self.features:
            days = np.array([(d - self.reference_date).days for d in dates], dtype=float)
            columns.append(days)

        if 'year' in self.features:
            years = np.array([d.year for d in dates], dtype=float)
            columns.append(years)

        if 'is_weekend' in self.features:
            weekend = np.array([1.0 if d.weekday() >= 5 else 0.0 for d in dates])
            columns.append(weekend)

        if len(columns) == 0:
            return np.zeros((len(X), 1))

        return np.column_stack(columns)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform dates."""
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X_encoded: np.ndarray) -> np.ndarray:
        """Reconstruct dates from encoded features (approximate)."""
        X_encoded = np.asarray(X_encoded)
        if X_encoded.ndim == 1:
            X_encoded = X_encoded.reshape(1, -1)

        results = []
        col_idx = 0

        for row in X_encoded:
            # Try to reconstruct from days_since_ref if available
            if 'month_cyclical' in self.features:
                col_idx = 2  # skip cos, sin
            if 'day_of_year_cyclical' in self.features:
                col_idx += 2
            if 'days_since_ref' in self.features:
                days = int(row[col_idx])
                dt = self.reference_date + timedelta(days=days)
                results.append(dt.strftime('%Y-%m-%d'))
            elif 'year' in self.features:
                year_idx = 0
                if 'month_cyclical' in self.features:
                    year_idx += 2
                if 'day_of_year_cyclical' in self.features:
                    year_idx += 2
                if 'days_since_ref' in self.features:
                    year_idx += 1
                year = int(row[year_idx])
                results.append(f"{year}-01-01")
            else:
                results.append("unknown")

        return np.array(results)

    def get_feature_names_out(self) -> List[str]:
        """Return names of extracted features."""
        names = []
        if 'month_cyclical' in self.features:
            names.extend(['month_cos', 'month_sin'])
        if 'day_of_year_cyclical' in self.features:
            names.extend(['day_of_year_cos', 'day_of_year_sin'])
        if 'days_since_ref' in self.features:
            names.append('days_since_ref')
        if 'year' in self.features:
            names.append('year')
        if 'is_weekend' in self.features:
            names.append('is_weekend')
        return names


# =============================================================================
# FREQUENCY ENCODER
# =============================================================================

class FrequencyEncoder(BaseEncoder):
    """
    Encodes categories by their frequency in training data.
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
        self.normalize = normalize
        self.handle_unknown = handle_unknown
        self.min_frequency = min_frequency
        self.is_fitted_ = False
        self.frequency_map_ = {}
        self.categories_ = []

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'FrequencyEncoder':
        """Learn category frequencies from training data."""
        X = np.asarray(X).ravel()
        total = len(X)
        unique, counts = np.unique(X, return_counts=True)

        self.frequency_map_ = {}
        for cat, count in zip(unique, counts):
            if self.normalize:
                freq = float(count) / total
            else:
                freq = float(count)

            # Apply min_frequency floor
            if self.min_frequency > 0 and freq < self.min_frequency:
                freq = self.min_frequency

            self.frequency_map_[cat] = freq

        self.categories_ = list(unique)
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform categories to frequencies."""
        if not self.is_fitted_:
            self.fit(X)

        X = np.asarray(X).ravel()
        result = np.zeros(len(X))

        for i, val in enumerate(X):
            if val in self.frequency_map_:
                result[i] = self.frequency_map_[val]
            elif self.handle_unknown == 'min_frequency':
                result[i] = self.min_frequency
            else:
                result[i] = 0.0

        return result.reshape(-1, 1)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X_encoded: np.ndarray) -> np.ndarray:
        """Not exactly invertible - returns most likely category."""
        X_encoded = np.asarray(X_encoded).ravel()
        inv_map = {v: k for k, v in self.frequency_map_.items()}

        result = []
        for val in X_encoded:
            if inv_map:
                closest_key = min(inv_map.keys(), key=lambda k: abs(k - val))
                result.append(inv_map[closest_key])
            else:
                result.append('unknown')

        return np.array(result)

    def get_feature_names_out(self) -> List[str]:
        """Return ['category_frequency']."""
        return ['category_frequency']


# =============================================================================
# WEIGHT OF EVIDENCE ENCODER
# =============================================================================

class WOEEncoder(BaseEncoder):
    """
    Weight of Evidence encoder for binary classification.
    WOE(x) = ln(P(X=x|Y=1) / P(X=x|Y=0))
    """

    def __init__(self, regularization: float = 0.5,
                 handle_unknown: str = 'zero'):
        """
        Initialize WOE encoder.

        Args:
            regularization: Add to counts to prevent log(0)
            handle_unknown: How to handle unseen categories
        """
        self.regularization = regularization
        self.handle_unknown = handle_unknown
        self.is_fitted_ = False
        self.woe_map_ = {}
        self.iv_ = 0.0
        self.categories_ = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'WOEEncoder':
        """Learn WOE values from training data. Requires binary target y."""
        X = np.asarray(X).ravel()
        y = np.asarray(y).ravel().astype(float)

        total_pos = float(np.sum(y == 1))
        total_neg = float(np.sum(y == 0))

        # Add regularization to prevent division by zero
        if total_pos == 0:
            total_pos = self.regularization
        if total_neg == 0:
            total_neg = self.regularization

        unique_cats = np.unique(X)
        self.woe_map_ = {}
        self.iv_ = 0.0

        for cat in unique_cats:
            mask = X == cat
            cat_pos = float(np.sum(y[mask] == 1)) + self.regularization
            cat_neg = float(np.sum(y[mask] == 0)) + self.regularization

            # Distribution of positives and negatives
            dist_pos = cat_pos / (total_pos + self.regularization * len(unique_cats))
            dist_neg = cat_neg / (total_neg + self.regularization * len(unique_cats))

            woe = np.log(dist_pos / dist_neg)
            self.woe_map_[cat] = float(woe)

            # Information Value contribution
            self.iv_ += (dist_pos - dist_neg) * woe

        self.categories_ = list(unique_cats)
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform categories to WOE values."""
        if not self.is_fitted_:
            # WOE requires y to fit; return zeros as fallback
            X = np.asarray(X).ravel()
            return np.zeros(len(X)).reshape(-1, 1)

        X = np.asarray(X).ravel()
        result = np.zeros(len(X))

        for i, val in enumerate(X):
            if val in self.woe_map_:
                result[i] = self.woe_map_[val]
            elif self.handle_unknown == 'zero':
                result[i] = 0.0
            else:
                result[i] = 0.0

        return result.reshape(-1, 1)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X_encoded: np.ndarray) -> np.ndarray:
        """Not invertible - WOE loses category information."""
        X_encoded = np.asarray(X_encoded).ravel()
        inv_map = {v: k for k, v in self.woe_map_.items()}

        result = []
        for val in X_encoded:
            if inv_map:
                closest_key = min(inv_map.keys(), key=lambda k: abs(k - val))
                result.append(inv_map[closest_key])
            else:
                result.append('unknown')

        return np.array(result)

    def get_feature_names_out(self) -> List[str]:
        """Return ['category_woe']."""
        return ['category_woe']

    def get_information_value(self) -> float:
        """Return computed Information Value for this feature."""
        return self.iv_


# =============================================================================
# COMPOSITE ENCODER
# =============================================================================

class CompositeEncoder(BaseEncoder):
    """
    Combines multiple encoders for the same feature.
    Applies each encoder independently and concatenates results horizontally.
    """

    def __init__(self, encoders: List[BaseEncoder]):
        """
        Initialize composite encoder.

        Args:
            encoders: List of encoders to apply
        """
        self.encoders = encoders
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'CompositeEncoder':
        """Fit all encoders."""
        for encoder in self.encoders:
            encoder.fit(X, y)
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform with all encoders and concatenate."""
        if not self.is_fitted_:
            self.fit(X)

        if len(self.encoders) == 0:
            return np.zeros((len(np.asarray(X).ravel()), 0))

        results = []
        for encoder in self.encoders:
            encoded = encoder.transform(X)
            if encoded.ndim == 1:
                encoded = encoded.reshape(-1, 1)
            results.append(encoded)

        return np.hstack(results)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform with all encoders."""
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X_encoded: np.ndarray) -> np.ndarray:
        """Not generally supported for composite."""
        # Return the input as-is since we can't generally invert composites
        return np.asarray(X_encoded)

    def get_feature_names_out(self) -> List[str]:
        """Concatenate feature names from all encoders."""
        names = []
        for encoder in self.encoders:
            names.extend(encoder.get_feature_names_out())
        return names


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_admission_encoders() -> Dict[str, BaseEncoder]:
    """
    Create standard encoder set for admission prediction.

    Returns:
        Dictionary of feature_name -> encoder mappings
    """
    return {
        'gpa': GPAEncoder(standardize=True, auto_detect=True),
        'university': UniversityEncoder(strategy='target'),
        'program': ProgramEncoder(),
        'term': TermEncoder(use_cyclical=True, include_year=True),
        'application_date': DateEncoder(),
    }


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
    """
    target = np.asarray(target).ravel()

    if encoders is None:
        encoders = create_admission_encoders()
        needs_fit = True
    else:
        needs_fit = False

    all_columns = []
    all_names = []

    # Extract features from data dicts
    for feature_name, encoder in encoders.items():
        # Extract this feature from all data records
        values = []
        for record in data:
            if feature_name in record:
                values.append(record[feature_name])
            else:
                values.append(None)

        X_feature = np.array(values)

        # Skip features that are all None
        if all(v is None for v in values):
            continue

        if needs_fit:
            encoded = encoder.fit_transform(X_feature, target)
        else:
            encoded = encoder.transform(X_feature)

        if encoded.ndim == 1:
            encoded = encoded.reshape(-1, 1)

        all_columns.append(encoded)
        all_names.extend(encoder.get_feature_names_out())

    if len(all_columns) == 0:
        return np.zeros((len(data), 0)), [], encoders

    encoded_matrix = np.hstack(all_columns)
    return encoded_matrix, all_names, encoders
