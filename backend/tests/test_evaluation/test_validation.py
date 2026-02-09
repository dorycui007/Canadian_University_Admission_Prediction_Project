"""
Unit Tests for Validation / Splitting Module
=============================================

Tests for src/evaluation/validation.py — TemporalSplitConfig, TimeSeriesCVConfig,
StratifiedConfig, SplitResult dataclasses, BaseSplitter ABC, and all temporal
splitting / cross-validation functions.
"""

import pytest
import numpy as np
from dataclasses import fields

from src.evaluation.validation import (
    TemporalSplitConfig, TimeSeriesCVConfig, StratifiedConfig, SplitResult,
    BaseSplitter, TemporalSplit, TimeSeriesCV, GroupedTimeSeriesCV,
    StratifiedTemporalSplit, PurgedKFold,
    temporal_train_test_split, check_temporal_leakage,
    validate_temporal_consistency, get_fold_statistics,
    create_date_features, create_holdout_split, create_admission_cv,
)


# =============================================================================
#                    TIMESERIESCVCONFIG TESTS
# =============================================================================

class TestTimeSeriesCVConfig:
    """Tests for TimeSeriesCVConfig dataclass."""

    def test_default_values(self):
        """Default config has expected values."""
        config = TimeSeriesCVConfig()
        assert config.n_splits == 5
        assert config.strategy == "expanding"
        assert config.gap == 0

    def test_has_expected_fields(self):
        """TimeSeriesCVConfig has exactly the expected fields."""
        field_names = {f.name for f in fields(TimeSeriesCVConfig)}
        assert "n_splits" in field_names
        assert "strategy" in field_names
        assert "gap" in field_names
        assert "min_train_size" in field_names
        assert "max_train_size" in field_names
        assert "val_size" in field_names
        assert "date_column" in field_names


# =============================================================================
#                    STRATIFIEDCONFIG TESTS
# =============================================================================

class TestStratifiedConfig:
    """Tests for StratifiedConfig dataclass."""

    def test_default_values(self):
        """Default config has expected values."""
        config = StratifiedConfig()
        assert config.min_samples_per_group == 5
        assert config.combine_rare is True

    def test_has_expected_fields(self):
        """StratifiedConfig has exactly the expected fields."""
        field_names = {f.name for f in fields(StratifiedConfig)}
        assert "stratify_columns" in field_names
        assert "min_samples_per_group" in field_names
        assert "combine_rare" in field_names
        assert "other_label" in field_names


# =============================================================================
#                    SPLITRESULT TESTS
# =============================================================================

class TestSplitResult:
    """Tests for SplitResult dataclass."""

    def test_has_expected_fields(self):
        """SplitResult has the expected fields."""
        field_names = {f.name for f in fields(SplitResult)}
        expected = {"train_indices", "val_indices", "test_indices", "metadata"}
        assert expected.issubset(field_names)

    def test_fields_stored_correctly(self):
        """Values passed to SplitResult are stored correctly."""
        result = SplitResult(
            train_indices=np.array([0, 1, 2]),
            val_indices=np.array([3, 4]),
            test_indices=np.array([5]),
            metadata={"split": 0},
        )
        np.testing.assert_array_equal(result.train_indices, [0, 1, 2])
        np.testing.assert_array_equal(result.val_indices, [3, 4])
        np.testing.assert_array_equal(result.test_indices, [5])
        assert result.metadata == {"split": 0}


# =============================================================================
#                    BASESPLITTER ABC TESTS
# =============================================================================

class TestBaseSplitterABC:
    """Tests for BaseSplitter abstract base class enforcement."""

    def test_cannot_instantiate_directly(self):
        """BaseSplitter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseSplitter()


# =============================================================================
#                    TEMPORAL SPLIT TESTS
# =============================================================================

class TestTemporalSplit:
    """Tests for TemporalSplit splitter."""

    def test_temporal_split_basic(self):
        """TemporalSplit produces non-overlapping train/val sets."""
        config = TemporalSplitConfig(
            train_end_date='2021-12-31', val_end_date='2022-12-31'
        )
        splitter = TemporalSplit(config)
        dates = np.array(['2021-01-01'] * 10 + ['2022-06-01'] * 10)
        splits = list(splitter.split(dates))
        assert len(splits) == 1
        split = splits[0]
        assert isinstance(split, SplitResult)
        assert len(split.train_indices) == 10
        assert len(split.val_indices) == 10

    def test_temporal_split_no_overlap(self):
        """Train and val indices do not overlap."""
        config = TemporalSplitConfig(
            train_end_date='2021-12-31', val_end_date='2022-12-31'
        )
        splitter = TemporalSplit(config)
        dates = np.array(
            ['2021-01-01'] * 5 + ['2021-06-01'] * 5 +
            ['2022-03-01'] * 5 + ['2022-09-01'] * 5
        )
        splits = list(splitter.split(dates))
        split = splits[0]
        train_set = set(split.train_indices)
        val_set = set(split.val_indices)
        assert len(train_set & val_set) == 0

    def test_temporal_split_with_test(self):
        """TemporalSplit produces test indices when test_end_date is set."""
        config = TemporalSplitConfig(
            train_end_date='2021-12-31',
            val_end_date='2022-12-31',
            test_end_date='2023-12-31',
        )
        splitter = TemporalSplit(config)
        dates = np.array(
            ['2021-06-01'] * 5 + ['2022-06-01'] * 5 + ['2023-06-01'] * 5
        )
        splits = list(splitter.split(dates))
        split = splits[0]
        assert split.train_indices is not None and len(split.train_indices) == 5
        assert split.val_indices is not None and len(split.val_indices) == 5
        assert split.test_indices is not None and len(split.test_indices) == 5

    def test_temporal_split_metadata(self):
        """TemporalSplit includes metadata about sizes."""
        config = TemporalSplitConfig(
            train_end_date='2021-12-31', val_end_date='2022-12-31'
        )
        splitter = TemporalSplit(config)
        dates = np.array(['2021-01-01'] * 10 + ['2022-06-01'] * 10)
        splits = list(splitter.split(dates))
        split = splits[0]
        assert 'train_size' in split.metadata
        assert 'val_size' in split.metadata
        assert split.metadata['train_size'] == 10
        assert split.metadata['val_size'] == 10


# =============================================================================
#                    TIME SERIES CV TESTS
# =============================================================================

class TestTimeSeriesCV:
    """Tests for TimeSeriesCV cross-validator."""

    def test_time_series_cv_generates_splits(self):
        """TimeSeriesCV generates at least one split."""
        config = TimeSeriesCVConfig(n_splits=3, strategy='expanding')
        cv = TimeSeriesCV(config)
        dates = np.array(['2021-01-01'] * 40)
        splits = list(cv.split(dates))
        assert len(splits) >= 1

    def test_time_series_cv_each_split_has_train_and_val(self):
        """Each CV split has non-empty train and val indices."""
        config = TimeSeriesCVConfig(n_splits=3, strategy='expanding')
        cv = TimeSeriesCV(config)
        dates = np.array(['2021-01-01'] * 40)
        for split in cv.split(dates):
            assert len(split.train_indices) > 0
            assert len(split.val_indices) > 0

    def test_time_series_cv_sliding(self):
        """TimeSeriesCV with sliding strategy produces splits."""
        config = TimeSeriesCVConfig(n_splits=3, strategy='sliding')
        cv = TimeSeriesCV(config)
        dates = np.array(['2021-01-01'] * 40)
        splits = list(cv.split(dates))
        assert len(splits) >= 1

    def test_train_val_no_overlap(self):
        """Train and val indices do not overlap in any fold."""
        config = TimeSeriesCVConfig(n_splits=3, strategy='expanding')
        cv = TimeSeriesCV(config)
        dates = np.array(['2021-01-01'] * 40)
        for split in cv.split(dates):
            train_set = set(split.train_indices)
            val_set = set(split.val_indices)
            assert len(train_set & val_set) == 0


# =============================================================================
#                    TEMPORAL TRAIN TEST SPLIT TESTS
# =============================================================================

class TestTemporalTrainTestSplit:
    """Tests for temporal_train_test_split(dates, X, y, test_size)."""

    def test_temporal_split_returns_four_arrays(self):
        """temporal_train_test_split returns (X_train, X_test, y_train, y_test)."""
        dates = np.arange(20)
        X = np.random.randn(20, 3)
        y = np.arange(20)
        result = temporal_train_test_split(dates, X, y)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_temporal_split_sizes(self):
        """Train and test sizes respect test_size proportion."""
        dates = np.arange(20)
        X = np.random.randn(20, 3)
        y = np.arange(20)
        X_train, X_test, y_train, y_test = temporal_train_test_split(
            dates, X, y, test_size=0.2
        )
        assert len(X_train) == 16
        assert len(X_test) == 4
        assert len(y_train) == 16
        assert len(y_test) == 4

    def test_temporal_split_no_leakage(self):
        """All training dates precede all test dates."""
        dates = np.arange(20)
        X = np.random.randn(20, 3)
        y = np.arange(20)
        X_train, X_test, y_train, y_test = temporal_train_test_split(dates, X, y)
        # y values correspond to dates, so max of train y < min of test y
        assert np.max(y_train) < np.min(y_test)

    def test_total_count_preserved(self):
        """Total number of samples is preserved across train and test."""
        dates = np.arange(20)
        X = np.random.randn(20, 3)
        y = np.arange(20)
        X_train, X_test, y_train, y_test = temporal_train_test_split(dates, X, y)
        assert len(X_train) + len(X_test) == 20


# =============================================================================
#                    CHECK TEMPORAL LEAKAGE TESTS
# =============================================================================

class TestCheckTemporalLeakage:
    """Tests for check_temporal_leakage(train_dates, test_dates)."""

    def test_no_leakage_detected(self):
        """No leakage is detected when train dates precede test dates."""
        result = check_temporal_leakage(np.arange(10), np.arange(10, 20))
        assert isinstance(result, dict)
        assert result['has_leakage'] is False
        assert result['leaky_train_count'] == 0
        assert result['leaky_test_count'] == 0

    def test_leakage_detected(self):
        """Leakage is detected when train and test dates overlap."""
        result = check_temporal_leakage(np.arange(15), np.arange(10, 20))
        assert result['has_leakage'] is True
        assert result['leaky_train_count'] > 0
        assert result['leaky_test_count'] > 0

    def test_overlap_days_reported(self):
        """Overlap days are reported when there is leakage."""
        result = check_temporal_leakage(np.arange(15), np.arange(10, 20))
        assert result['overlap_days'] > 0

    def test_no_overlap_days_when_clean(self):
        """Overlap days are 0 when there is no leakage."""
        result = check_temporal_leakage(np.arange(10), np.arange(10, 20))
        assert result['overlap_days'] == 0


# =============================================================================
#                    BASESPLITTER.VALIDATE_INPUTS TESTS
# =============================================================================

class TestBaseSplitterValidateInputs:
    """Tests for BaseSplitter.validate_inputs via a concrete subclass."""

    def test_validate_inputs_called_through_temporal_split(self):
        """BaseSplitter.validate_inputs returns True for valid inputs."""
        config = TemporalSplitConfig(
            train_end_date='2021-12-31', val_end_date='2022-12-31'
        )
        splitter = TemporalSplit(config)
        result = splitter.validate_inputs(np.arange(10).reshape(10, 1))
        assert result is True

    def test_validate_inputs_with_y(self):
        """BaseSplitter.validate_inputs accepts optional y argument."""
        config = TemporalSplitConfig(
            train_end_date='2021-12-31', val_end_date='2022-12-31'
        )
        splitter = TemporalSplit(config)
        X = np.arange(10).reshape(10, 1)
        y = np.arange(10)
        result = splitter.validate_inputs(X, y)
        assert result is True

    def test_validate_inputs_empty_raises(self):
        """Empty X array raises ValueError."""
        config = TemporalSplitConfig(
            train_end_date='2021-12-31', val_end_date='2022-12-31'
        )
        splitter = TemporalSplit(config)
        with pytest.raises(ValueError, match="empty"):
            splitter.validate_inputs(np.array([]))

    def test_validate_inputs_mismatched_y_raises(self):
        """Mismatched X and y lengths raise ValueError."""
        config = TemporalSplitConfig(
            train_end_date='2021-12-31', val_end_date='2022-12-31'
        )
        splitter = TemporalSplit(config)
        X = np.arange(10).reshape(10, 1)
        y = np.arange(5)
        with pytest.raises(ValueError, match="same number of samples"):
            splitter.validate_inputs(X, y)


# =============================================================================
#                    TEMPORAL SPLIT GET_N_SPLITS TESTS
# =============================================================================

class TestTemporalSplitGetNSplits:
    """Tests for TemporalSplit.get_n_splits."""

    def test_get_n_splits_returns_one(self):
        """TemporalSplit.get_n_splits returns 1."""
        config = TemporalSplitConfig(
            train_end_date='2021-12-31', val_end_date='2022-12-31'
        )
        splitter = TemporalSplit(config)
        result = splitter.get_n_splits()
        assert isinstance(result, int)
        assert result == 1


# =============================================================================
#                    TIMESERIESCV GET_N_SPLITS TESTS
# =============================================================================

class TestTimeSeriesCVGetNSplits:
    """Tests for TimeSeriesCV.get_n_splits."""

    def test_get_n_splits_returns_config_value(self):
        """TimeSeriesCV.get_n_splits returns the configured n_splits."""
        config = TimeSeriesCVConfig(n_splits=5)
        cv = TimeSeriesCV(config)
        result = cv.get_n_splits()
        assert isinstance(result, int)
        assert result == 5

    def test_get_n_splits_custom(self):
        """Custom n_splits value is returned."""
        config = TimeSeriesCVConfig(n_splits=3)
        cv = TimeSeriesCV(config)
        assert cv.get_n_splits() == 3


# =============================================================================
#                    TIMESERIESCV EXPANDING SPLIT TESTS
# =============================================================================

class TestTimeSeriesCVExpandingSplit:
    """Tests for TimeSeriesCV._expanding_split."""

    def test_expanding_split_generates_folds(self):
        """_expanding_split generates multiple folds."""
        config = TimeSeriesCVConfig(n_splits=3, strategy='expanding')
        cv = TimeSeriesCV(config)
        folds = list(cv._expanding_split(100))
        assert len(folds) >= 1

    def test_expanding_split_train_grows(self):
        """In expanding splits, training set grows with each fold."""
        config = TimeSeriesCVConfig(n_splits=3, strategy='expanding')
        cv = TimeSeriesCV(config)
        folds = list(cv._expanding_split(100))
        if len(folds) >= 2:
            assert len(folds[1][0]) > len(folds[0][0])


# =============================================================================
#                    TIMESERIESCV SLIDING SPLIT TESTS
# =============================================================================

class TestTimeSeriesCVSlidingSplit:
    """Tests for TimeSeriesCV._sliding_split."""

    def test_sliding_split_generates_folds(self):
        """_sliding_split generates multiple folds."""
        config = TimeSeriesCVConfig(n_splits=3, strategy='sliding')
        cv = TimeSeriesCV(config)
        folds = list(cv._sliding_split(100))
        assert len(folds) >= 1

    def test_sliding_split_fixed_train_size(self):
        """In sliding splits, training set size stays fixed."""
        config = TimeSeriesCVConfig(n_splits=3, strategy='sliding', max_train_size=20, val_size=10)
        cv = TimeSeriesCV(config)
        folds = list(cv._sliding_split(100))
        if len(folds) >= 2:
            assert len(folds[0][0]) == len(folds[1][0])


# =============================================================================
#                    GROUPED TIMESERIES CV TESTS
# =============================================================================

class TestGroupedTimeSeriesCV:
    """Tests for GroupedTimeSeriesCV."""

    def test_init(self):
        """GroupedTimeSeriesCV can be instantiated."""
        config = TimeSeriesCVConfig(n_splits=3)
        cv = GroupedTimeSeriesCV(config)
        assert cv is not None

    def test_split_generates_splits(self):
        """GroupedTimeSeriesCV.split generates at least one split."""
        config = TimeSeriesCVConfig(n_splits=2)
        cv = GroupedTimeSeriesCV(config)
        dates = np.array(
            ['2021-01-01'] * 10 + ['2022-01-01'] * 10 + ['2023-01-01'] * 10
        )
        y = np.zeros(30)
        groups = np.array(
            ['A'] * 5 + ['B'] * 5 + ['C'] * 5 + ['D'] * 5 + ['E'] * 5 + ['F'] * 5
        )
        splits = list(cv.split(dates, y, groups))
        assert len(splits) >= 1

    def test_split_requires_groups(self):
        """GroupedTimeSeriesCV.split raises if groups is None."""
        config = TimeSeriesCVConfig(n_splits=3)
        cv = GroupedTimeSeriesCV(config)
        dates = np.array(['2021-01-01'] * 10)
        with pytest.raises(ValueError, match="groups"):
            list(cv.split(dates, groups=None))

    def test_get_n_splits(self):
        """GroupedTimeSeriesCV.get_n_splits returns configured value."""
        config = TimeSeriesCVConfig(n_splits=3)
        cv = GroupedTimeSeriesCV(config)
        result = cv.get_n_splits()
        assert isinstance(result, int)
        assert result == 3

    def test_get_group_first_dates(self):
        """_get_group_first_dates returns dict mapping groups to first dates."""
        config = TimeSeriesCVConfig(n_splits=3)
        cv = GroupedTimeSeriesCV(config)
        dates = np.array(['2021-03-01', '2021-01-01', '2022-06-01', '2022-01-01'])
        groups = np.array(['A', 'A', 'B', 'B'])
        result = cv._get_group_first_dates(dates, groups)
        assert isinstance(result, dict)
        assert 'A' in result
        assert 'B' in result
        # Group A first date should be 2021-01-01 (the earlier one)
        assert result['A'] == np.datetime64('2021-01-01')
        assert result['B'] == np.datetime64('2022-01-01')


# =============================================================================
#                    STRATIFIED TEMPORAL SPLIT TESTS
# =============================================================================

class TestStratifiedTemporalSplit:
    """Tests for StratifiedTemporalSplit."""

    def test_init(self):
        """StratifiedTemporalSplit can be instantiated."""
        temporal_config = TemporalSplitConfig(
            train_end_date='2021-12-31', val_end_date='2022-12-31'
        )
        stratify_config = StratifiedConfig()
        splitter = StratifiedTemporalSplit(temporal_config, stratify_config)
        assert splitter is not None

    def test_split_produces_split_result(self):
        """StratifiedTemporalSplit.split yields SplitResult."""
        temporal_config = TemporalSplitConfig(
            train_end_date='2021-12-31', val_end_date='2022-12-31'
        )
        stratify_config = StratifiedConfig()
        splitter = StratifiedTemporalSplit(temporal_config, stratify_config)
        dates = np.array(['2021-01-01'] * 10 + ['2022-06-01'] * 10)
        y = np.array([0, 1] * 10)
        splits = list(splitter.split(dates, y))
        assert len(splits) == 1
        assert isinstance(splits[0], SplitResult)

    def test_get_n_splits(self):
        """StratifiedTemporalSplit.get_n_splits returns 1."""
        temporal_config = TemporalSplitConfig(
            train_end_date='2021-12-31', val_end_date='2022-12-31'
        )
        stratify_config = StratifiedConfig()
        splitter = StratifiedTemporalSplit(temporal_config, stratify_config)
        result = splitter.get_n_splits()
        assert isinstance(result, int)
        assert result == 1

    def test_stratified_indices(self):
        """_stratified_indices returns train and test arrays that partition indices."""
        temporal_config = TemporalSplitConfig(
            train_end_date='2021-12-31', val_end_date='2022-12-31'
        )
        stratify_config = StratifiedConfig()
        splitter = StratifiedTemporalSplit(temporal_config, stratify_config)
        indices = np.arange(20)
        y = np.array([0, 1] * 10)
        train_idx, test_idx = splitter._stratified_indices(indices, y, test_size=0.2)
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
        assert len(train_idx) + len(test_idx) == 20
        assert len(set(train_idx) & set(test_idx)) == 0


# =============================================================================
#                    PURGED KFOLD TESTS
# =============================================================================

class TestPurgedKFold:
    """Tests for PurgedKFold."""

    def test_init(self):
        """PurgedKFold can be instantiated with default args."""
        kfold = PurgedKFold()
        assert kfold is not None
        assert kfold.n_splits == 5

    def test_init_with_params(self):
        """PurgedKFold can be instantiated with custom args."""
        kfold = PurgedKFold(n_splits=3, purge_gap=5, embargo_gap=2)
        assert kfold.n_splits == 3
        assert kfold.purge_gap == 5
        assert kfold.embargo_gap == 2

    def test_split_generates_folds(self):
        """PurgedKFold.split generates multiple folds."""
        kfold = PurgedKFold(n_splits=3)
        dates = np.arange(30)
        splits = list(kfold.split(dates))
        assert len(splits) == 3

    def test_split_train_val_no_overlap(self):
        """Train and val indices do not overlap in any fold."""
        kfold = PurgedKFold(n_splits=3)
        dates = np.arange(30)
        for split in kfold.split(dates):
            train_set = set(split.train_indices)
            val_set = set(split.val_indices)
            assert len(train_set & val_set) == 0

    def test_get_n_splits(self):
        """PurgedKFold.get_n_splits returns configured n_splits."""
        kfold = PurgedKFold(n_splits=4)
        result = kfold.get_n_splits()
        assert isinstance(result, int)
        assert result == 4

    def test_purge_gap_reduces_train(self):
        """With purge_gap > 0, train set is smaller than without purging."""
        kfold_no_purge = PurgedKFold(n_splits=3, purge_gap=0, embargo_gap=0)
        kfold_purge = PurgedKFold(n_splits=3, purge_gap=3, embargo_gap=0)
        dates = np.arange(30)
        splits_no = list(kfold_no_purge.split(dates))
        splits_yes = list(kfold_purge.split(dates))
        # With purging, train size should be smaller or equal
        if len(splits_no) > 0 and len(splits_yes) > 0:
            assert len(splits_yes[0].train_indices) <= len(splits_no[0].train_indices)

    def test_split_metadata(self):
        """Each split includes metadata with fold info."""
        kfold = PurgedKFold(n_splits=3)
        dates = np.arange(30)
        for split in kfold.split(dates):
            assert 'fold' in split.metadata
            assert 'train_size' in split.metadata
            assert 'val_size' in split.metadata


# =============================================================================
#                    GET_FOLD_STATISTICS TESTS
# =============================================================================

class TestGetFoldStatistics:
    """Tests for get_fold_statistics standalone function."""

    def test_get_fold_statistics_basic(self):
        """get_fold_statistics returns a dict with 'folds' key."""
        splits = [
            SplitResult(
                train_indices=np.array([0, 1, 2]),
                val_indices=np.array([3, 4]),
                test_indices=np.array([5]),
            ),
        ]
        y = np.array([0, 1, 0, 1, 0, 1])
        result = get_fold_statistics(splits, y)
        assert isinstance(result, dict)
        assert 'folds' in result

    def test_get_fold_statistics_fold_count(self):
        """Number of fold stats matches number of splits."""
        splits = [
            SplitResult(
                train_indices=np.array([0, 1, 2]),
                val_indices=np.array([3, 4]),
            ),
            SplitResult(
                train_indices=np.array([0, 1, 2, 3]),
                val_indices=np.array([4, 5]),
            ),
        ]
        y = np.array([0, 1, 0, 1, 0, 1])
        result = get_fold_statistics(splits, y)
        assert len(result['folds']) == 2

    def test_get_fold_statistics_contains_sizes(self):
        """Each fold's stats include train_size and val_size."""
        splits = [
            SplitResult(
                train_indices=np.array([0, 1, 2]),
                val_indices=np.array([3, 4]),
                test_indices=np.array([5]),
            ),
        ]
        y = np.array([0, 1, 0, 1, 0, 1])
        result = get_fold_statistics(splits, y)
        fold = result['folds'][0]
        assert fold['train_size'] == 3
        assert fold['val_size'] == 2
        assert fold['test_size'] == 1


# =============================================================================
#                    VALIDATE_TEMPORAL_CONSISTENCY TESTS
# =============================================================================

class TestValidateTemporalConsistencyExtended:
    """Tests for validate_temporal_consistency standalone function."""

    def test_consistent_returns_true(self):
        """validate_temporal_consistency returns True for consistent split."""
        train_indices = np.array([0, 1, 2, 3])
        test_indices = np.array([4, 5, 6, 7])
        dates = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        result = validate_temporal_consistency(train_indices, test_indices, dates)
        assert result == True

    def test_inconsistent_returns_false(self):
        """validate_temporal_consistency returns False when train overlaps test dates."""
        train_indices = np.array([0, 1, 2, 5])  # includes date=6
        test_indices = np.array([3, 4])  # dates=4,5
        dates = np.array([1, 2, 3, 4, 5, 6])
        result = validate_temporal_consistency(train_indices, test_indices, dates)
        assert result == False

    def test_empty_sets_return_true(self):
        """Empty train or test returns True."""
        dates = np.array([1, 2, 3, 4])
        assert validate_temporal_consistency(np.array([], dtype=int), np.array([1, 2]), dates) == True
        assert validate_temporal_consistency(np.array([0, 1]), np.array([], dtype=int), dates) == True


# =============================================================================
#                    CREATE_DATE_FEATURES TESTS
# =============================================================================

class TestCreateDateFeatures:
    """Tests for create_date_features standalone function."""

    def test_create_date_features_returns_dict(self):
        """create_date_features returns a dict with expected keys."""
        dates = np.array(['2021-01-15', '2022-06-01', '2023-09-10'])
        result = create_date_features(dates)
        assert isinstance(result, dict)
        assert 'year' in result
        assert 'month' in result
        assert 'day_of_year' in result
        assert 'is_application_season' in result

    def test_year_values(self):
        """Year extraction is correct."""
        dates = np.array(['2021-01-15', '2022-06-01', '2023-09-10'])
        result = create_date_features(dates)
        np.testing.assert_array_equal(result['year'], [2021, 2022, 2023])

    def test_month_values(self):
        """Month extraction is correct."""
        dates = np.array(['2021-01-15', '2022-06-01', '2023-09-10'])
        result = create_date_features(dates)
        np.testing.assert_array_equal(result['month'], [1, 6, 9])

    def test_application_season(self):
        """is_application_season flags correct months."""
        # Jan is in application season, June is not, September is
        dates = np.array(['2021-01-15', '2022-06-01', '2023-09-10'])
        result = create_date_features(dates)
        assert result['is_application_season'][0] == True   # Jan - yes
        assert result['is_application_season'][1] == False   # June - no
        assert result['is_application_season'][2] == True    # Sep - yes

    def test_arrays_same_length(self):
        """All returned arrays have the same length as input."""
        dates = np.array(['2021-01-15', '2022-06-01', '2023-09-10'])
        result = create_date_features(dates)
        for key in ('year', 'month', 'day_of_year', 'is_application_season'):
            assert len(result[key]) == 3


# =============================================================================
#                    CREATE_ADMISSION_CV TESTS
# =============================================================================

class TestCreateAdmissionCV:
    """Tests for create_admission_cv standalone function."""

    def test_create_admission_cv_basic(self):
        """create_admission_cv returns a TimeSeriesCV instance."""
        dates = np.array(['2021-01-01', '2022-01-01', '2023-01-01'] * 5)
        result = create_admission_cv(dates)
        assert isinstance(result, TimeSeriesCV)

    def test_create_admission_cv_with_params(self):
        """create_admission_cv accepts n_splits and strategy parameters."""
        dates = np.array(['2021-01-01', '2022-01-01', '2023-01-01'] * 5)
        result = create_admission_cv(dates, n_splits=3, strategy='sliding')
        assert isinstance(result, TimeSeriesCV)
        assert result.config.n_splits == 3
        assert result.config.strategy == 'sliding'

    def test_create_admission_cv_default_gap(self):
        """Default admission CV has a 30-day gap."""
        dates = np.array(['2021-01-01'] * 10)
        result = create_admission_cv(dates)
        assert result.config.gap == 30


# =============================================================================
#                    CREATE_HOLDOUT_SPLIT TESTS
# =============================================================================

class TestCreateHoldoutSplit:
    """Tests for create_holdout_split standalone function."""

    def test_create_holdout_split_basic(self):
        """create_holdout_split returns a TemporalSplit instance."""
        dates = np.array(['2021-01-01', '2022-01-01', '2023-01-01'] * 5)
        result = create_holdout_split(dates)
        assert isinstance(result, TemporalSplit)

    def test_create_holdout_split_with_holdout_years(self):
        """create_holdout_split accepts holdout_years parameter."""
        dates = np.array(
            ['2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01'] * 5
        )
        result = create_holdout_split(dates, holdout_years=2)
        assert isinstance(result, TemporalSplit)

    def test_holdout_config_dates(self):
        """Holdout split has correct config date boundaries."""
        dates = np.array(['2021-06-01', '2022-06-01', '2023-06-01'] * 5)
        result = create_holdout_split(dates, holdout_years=1)
        # With max year 2023 and holdout_years=1, train_end should be 2022-12-31
        assert result.config.train_end_date == '2022-12-31'
        assert result.config.val_end_date == '2023-12-31'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
