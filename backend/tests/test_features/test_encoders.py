"""
Unit Tests for Feature Encoders Module
=======================================

Tests for src/features/encoders.py — GPAScaleConfig, TargetEncodingConfig,
HierarchicalGrouping dataclasses, BaseEncoder ABC, and all concrete encoder
classes (GPAEncoder, UniversityEncoder, ProgramEncoder, TermEncoder,
DateEncoder, FrequencyEncoder, WOEEncoder, CompositeEncoder), plus
convenience functions.
"""

import pytest
import numpy as np

from src.features.encoders import (
    GPAScaleConfig, TargetEncodingConfig, HierarchicalGrouping,
    BaseEncoder, GPAEncoder, UniversityEncoder, ProgramEncoder,
    TermEncoder, DateEncoder, FrequencyEncoder, WOEEncoder,
    CompositeEncoder, create_admission_encoders, encode_admission_features,
)


# =============================================================================
#                    GPASCALECONFIG TESTS
# =============================================================================

class TestGPAScaleConfig:
    """Tests for GPAScaleConfig dataclass."""

    def test_default_values(self):
        """Default config has min_value=0.0, max_value=100.0, letter_mapping=None."""
        config = GPAScaleConfig(scale_type='percentage')
        assert config.scale_type == 'percentage'
        assert config.min_value == 0.0
        assert config.max_value == 100.0
        assert config.letter_mapping is None

    def test_custom_values(self):
        """Custom values are stored correctly for a 4.0 scale."""
        mapping = {'A+': 4.3, 'A': 4.0, 'B': 3.0}
        config = GPAScaleConfig(
            scale_type='4.0',
            min_value=0.0,
            max_value=4.3,
            letter_mapping=mapping,
        )
        assert config.scale_type == '4.0'
        assert config.max_value == 4.3
        assert config.letter_mapping == mapping


# =============================================================================
#                    TARGETENCODINGCONFIG TESTS
# =============================================================================

class TestTargetEncodingConfig:
    """Tests for TargetEncodingConfig dataclass."""

    def test_defaults(self):
        """Default config has smoothing=10.0, min_samples=5, noise_std=0.0, use_loo=True."""
        config = TargetEncodingConfig()
        assert config.smoothing == 10.0
        assert config.min_samples == 5
        assert config.noise_std == 0.0
        assert config.use_loo is True


# =============================================================================
#                    BASEENCODER ABC TESTS
# =============================================================================

class TestBaseEncoderABC:
    """Tests for BaseEncoder abstract base class enforcement."""

    def test_cannot_instantiate_directly(self):
        """BaseEncoder cannot be instantiated because it is abstract."""
        with pytest.raises(TypeError):
            BaseEncoder()

    def test_subclass_missing_methods_raises(self):
        """A subclass that omits an abstract method cannot be instantiated."""
        class IncompleteEncoder(BaseEncoder):
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X

            # Missing: inverse_transform, get_feature_names_out

        with pytest.raises(TypeError):
            IncompleteEncoder()

    def test_fit_transform_base(self):
        """BaseEncoder.fit_transform() called directly on the base class method."""
        encoder = GPAEncoder()
        X = np.array([85.0, 90.0, 78.0])
        # Call BaseEncoder.fit_transform explicitly to cover the base class stub
        result = BaseEncoder.fit_transform(encoder, X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)


# =============================================================================
#                    GPAENCODER TESTS
# =============================================================================

class TestGPAEncoder:
    """Tests for GPAEncoder."""

    def test_init_default_params(self):
        """GPAEncoder can be created with default parameters."""
        encoder = GPAEncoder()
        # Constructor stores parameters; does not raise
        assert isinstance(encoder, BaseEncoder)
        assert encoder.standardize is True
        assert encoder.auto_detect is True
        assert encoder.is_fitted_ is False

    def test_default_letter_map_exists(self):
        """DEFAULT_LETTER_MAP class attribute contains expected keys."""
        assert isinstance(GPAEncoder.DEFAULT_LETTER_MAP, dict)
        assert 'A+' in GPAEncoder.DEFAULT_LETTER_MAP
        assert 'F' in GPAEncoder.DEFAULT_LETTER_MAP
        assert GPAEncoder.DEFAULT_LETTER_MAP['A+'] == 0.97

    def test_fit_returns_self(self):
        """fit() returns the encoder instance for method chaining."""
        encoder = GPAEncoder()
        result = encoder.fit(np.array([85.0, 90.0, 78.0]))
        assert result is encoder
        assert encoder.is_fitted_ is True

    def test_fit_sets_mean_and_std(self):
        """fit() with standardize=True computes mean_ and std_."""
        encoder = GPAEncoder(standardize=True)
        X = np.array([80.0, 90.0, 100.0])
        encoder.fit(X)
        # Normalized to [0,1] => [0.8, 0.9, 1.0], mean=0.9, std~0.0816
        assert encoder.mean_ is not None
        assert encoder.std_ is not None
        assert abs(encoder.mean_ - 0.9) < 1e-10
        expected_std = float(np.std([0.8, 0.9, 1.0]))
        assert abs(encoder.std_ - expected_std) < 1e-10

    def test_transform_percentage(self):
        """GPAEncoder.transform() normalizes percentage GPAs and z-scores them."""
        encoder = GPAEncoder(standardize=True)
        X = np.array([80.0, 90.0, 100.0])
        result = encoder.fit_transform(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)
        # Z-scored values should have mean ~0
        assert abs(np.mean(result)) < 1e-10

    def test_transform_no_standardize(self):
        """GPAEncoder with standardize=False returns [0,1] normalized values."""
        encoder = GPAEncoder(standardize=False)
        X = np.array([80.0, 90.0, 100.0])
        result = encoder.fit_transform(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)
        np.testing.assert_allclose(result.ravel(), [0.80, 0.90, 1.00])

    def test_fit_transform_method(self):
        """GPAEncoder.fit_transform() fits and transforms in one step."""
        encoder = GPAEncoder()
        X = np.array([85.0, 90.0, 78.0])
        result = encoder.fit_transform(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)

    def test_inverse_transform_percentage(self):
        """GPAEncoder.inverse_transform() reverses encoding back to percentage scale."""
        encoder = GPAEncoder(standardize=True)
        X = np.array([80.0, 90.0, 100.0])
        encoded = encoder.fit_transform(X)
        recovered = encoder.inverse_transform(encoded)
        np.testing.assert_allclose(recovered, X, atol=1e-10)

    def test_inverse_transform_not_fitted(self):
        """inverse_transform on unfitted encoder returns values as-is."""
        encoder = GPAEncoder()
        X_encoded = np.array([0.85, 0.90, 0.78])
        result = encoder.inverse_transform(X_encoded)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, X_encoded)

    def test_get_feature_names_out(self):
        """GPAEncoder.get_feature_names_out() returns ['gpa_normalized']."""
        encoder = GPAEncoder()
        result = encoder.get_feature_names_out()
        assert isinstance(result, list)
        assert result == ['gpa_normalized']

    def test_detect_scale_percentage(self):
        """_detect_scale detects percentage scale for values > 45."""
        encoder = GPAEncoder()
        X = np.array([85.0, 90.0, 78.0])
        config = encoder._detect_scale(X)
        assert isinstance(config, GPAScaleConfig)
        assert config.scale_type == 'percentage'

    def test_detect_scale_4point(self):
        """_detect_scale detects 4.0 scale for values <= 4.5."""
        encoder = GPAEncoder()
        X = np.array([3.5, 4.0, 2.8])
        config = encoder._detect_scale(X)
        assert isinstance(config, GPAScaleConfig)
        assert config.scale_type == '4.0'

    def test_detect_scale_ib(self):
        """_detect_scale detects IB scale for values in (7, 45]."""
        encoder = GPAEncoder()
        X = np.array([38.0, 42.0, 35.0])
        config = encoder._detect_scale(X)
        assert isinstance(config, GPAScaleConfig)
        assert config.scale_type == 'ib'

    def test_detect_scale_ib_course(self):
        """_detect_scale detects IB course scale for values in (4.5, 7]."""
        encoder = GPAEncoder()
        X = np.array([5.0, 6.0, 7.0])
        config = encoder._detect_scale(X)
        assert isinstance(config, GPAScaleConfig)
        assert config.scale_type == 'ib_course'

    def test_normalize_percentage(self):
        """_normalize_percentage divides by 100."""
        encoder = GPAEncoder()
        X = np.array([85.0, 90.0, 78.0])
        result = encoder._normalize_percentage(X)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [0.85, 0.90, 0.78])

    def test_normalize_4point(self):
        """_normalize_4point divides by max scale value (default 4.0)."""
        encoder = GPAEncoder()
        encoder.scale_config = GPAScaleConfig(scale_type='4.0', max_value=4.0)
        X = np.array([3.5, 4.0, 2.8])
        result = encoder._normalize_4point(X)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [3.5/4.0, 4.0/4.0, 2.8/4.0])

    def test_normalize_letter(self):
        """_normalize_letter converts letter grades using DEFAULT_LETTER_MAP."""
        encoder = GPAEncoder()
        X = np.array(['A+', 'B', 'C'])
        result = encoder._normalize_letter(X)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert result[0] == 0.97  # A+
        assert result[1] == 0.83  # B
        assert result[2] == 0.73  # C

    def test_4point_scale_end_to_end(self):
        """End-to-end encoding of 4.0 scale GPA values."""
        encoder = GPAEncoder(standardize=False)
        X = np.array([3.0, 4.0, 2.0])
        result = encoder.fit_transform(X)
        # Auto-detects as 4.0 scale, normalizes to [0,1]
        np.testing.assert_allclose(result.ravel(), [3.0/4.0, 4.0/4.0, 2.0/4.0])


# =============================================================================
#                    UNIVERSITYENCODER TESTS
# =============================================================================

class TestUniversityEncoder:
    """Tests for UniversityEncoder."""

    def test_init_default_strategy(self):
        """UniversityEncoder defaults to strategy='target'."""
        encoder = UniversityEncoder()
        assert isinstance(encoder, BaseEncoder)
        assert encoder.strategy == 'target'

    def test_invalid_strategy_raises(self):
        """UniversityEncoder raises ValueError for invalid strategy."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            UniversityEncoder(strategy='invalid')

    def test_fit_frequency_strategy(self):
        """fit() with frequency strategy computes frequency-based encoding."""
        encoder = UniversityEncoder(strategy='frequency')
        X = np.array(["UofT", "UBC", "McGill", "UofT", "UBC"])
        result = encoder.fit(X)
        assert result is encoder
        assert encoder.is_fitted_ is True
        assert 'UofT' in encoder.encoding_map_
        assert abs(encoder.encoding_map_['UofT'] - 2.0/5.0) < 1e-10
        assert abs(encoder.encoding_map_['UBC'] - 2.0/5.0) < 1e-10
        assert abs(encoder.encoding_map_['McGill'] - 1.0/5.0) < 1e-10

    def test_fit_onehot_strategy(self):
        """fit() with onehot strategy learns sorted categories."""
        encoder = UniversityEncoder(strategy='onehot')
        X = np.array(["UofT", "UBC", "McGill"])
        encoder.fit(X)
        assert encoder.is_fitted_ is True
        assert encoder.categories_ == ['McGill', 'UBC', 'UofT']
        assert encoder.n_categories_ == 3

    def test_transform_onehot(self):
        """transform() with onehot produces correct binary matrix."""
        encoder = UniversityEncoder(strategy='onehot')
        X = np.array(["UofT", "UBC", "McGill"])
        encoder.fit(X)
        result = encoder.transform(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)
        # Each row should sum to 1
        np.testing.assert_allclose(result.sum(axis=1), [1.0, 1.0, 1.0])
        # UofT is last alphabetically (index 2), UBC is index 1, McGill index 0
        np.testing.assert_array_equal(result[0], [0.0, 0.0, 1.0])  # UofT
        np.testing.assert_array_equal(result[1], [0.0, 1.0, 0.0])  # UBC
        np.testing.assert_array_equal(result[2], [1.0, 0.0, 0.0])  # McGill

    def test_transform_frequency(self):
        """transform() with frequency returns frequency values in (-1,1) column."""
        encoder = UniversityEncoder(strategy='frequency')
        X = np.array(["UofT", "UBC", "McGill", "UofT", "UBC"])
        encoder.fit(X)
        result = encoder.transform(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 1)
        # UofT freq = 2/5 = 0.4
        assert abs(result[0, 0] - 0.4) < 1e-10

    def test_fit_target_with_y(self):
        """fit() with target strategy and y uses smoothed target encoding."""
        encoder = UniversityEncoder(strategy='target')
        X = np.array(["UofT", "UBC", "McGill", "UofT", "UBC"])
        y = np.array([1, 0, 1, 1, 0])
        encoder.fit(X, y)
        assert encoder.is_fitted_ is True
        global_mean = np.mean(y)
        assert abs(encoder.global_mean_ - global_mean) < 1e-10
        # UofT: 2 samples, mean=1.0, smoothed = (2*1.0 + 10*0.6)/(2+10)
        expected_uoft = (2 * 1.0 + 10 * global_mean) / (2 + 10)
        assert abs(encoder.encoding_map_['UofT'] - expected_uoft) < 1e-10

    def test_fit_target_without_y(self):
        """fit() with target strategy but no y falls back to frequency encoding."""
        encoder = UniversityEncoder(strategy='target')
        X = np.array(["UofT", "UBC", "McGill", "UofT", "UBC"])
        encoder.fit(X)
        assert encoder.is_fitted_ is True
        # Falls back to frequency encoding
        assert abs(encoder.encoding_map_['UofT'] - 2.0/5.0) < 1e-10

    def test_fit_transform_with_loo(self):
        """fit_transform() with target strategy and y uses leave-one-out encoding."""
        encoder = UniversityEncoder(strategy='target')
        X = np.array(["UofT", "UBC", "McGill", "UofT", "UBC"])
        y = np.array([1, 0, 1, 1, 0])
        result = encoder.fit_transform(X, y)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 1)
        # McGill has only 1 sample, so LOO uses global mean
        global_mean = np.mean(y)
        assert abs(result[2, 0] - global_mean) < 1e-10

    def test_inverse_transform_onehot(self):
        """inverse_transform() with onehot recovers original categories."""
        encoder = UniversityEncoder(strategy='onehot')
        X = np.array(["UofT", "UBC", "McGill"])
        encoder.fit(X)
        encoded = encoder.transform(X)
        recovered = encoder.inverse_transform(encoded)
        assert isinstance(recovered, np.ndarray)
        np.testing.assert_array_equal(recovered, X)

    def test_inverse_transform_target(self):
        """inverse_transform() with target finds closest match."""
        encoder = UniversityEncoder(strategy='target')
        X = np.array(["UofT", "UBC", "McGill", "UofT", "UBC"])
        y = np.array([1, 0, 1, 1, 0])
        encoder.fit(X, y)
        encoded = encoder.transform(X[:3])
        recovered = encoder.inverse_transform(encoded)
        assert isinstance(recovered, np.ndarray)
        assert len(recovered) == 3

    def test_get_feature_names_out_onehot(self):
        """get_feature_names_out() with onehot returns category-based names."""
        encoder = UniversityEncoder(strategy='onehot')
        X = np.array(["UofT", "UBC", "McGill"])
        encoder.fit(X)
        result = encoder.get_feature_names_out()
        assert isinstance(result, list)
        assert result == ['university_McGill', 'university_UBC', 'university_UofT']

    def test_get_feature_names_out_target(self):
        """get_feature_names_out() with target returns ['university_target']."""
        encoder = UniversityEncoder(strategy='target')
        result = encoder.get_feature_names_out()
        assert result == ['university_target']

    def test_get_feature_names_out_frequency(self):
        """get_feature_names_out() with frequency returns ['university_frequency']."""
        encoder = UniversityEncoder(strategy='frequency')
        result = encoder.get_feature_names_out()
        assert result == ['university_frequency']

    def test_handle_unknown_global_mean(self):
        """Unknown categories use global mean when handle_unknown='global_mean'."""
        encoder = UniversityEncoder(strategy='frequency', handle_unknown='global_mean')
        X_train = np.array(["UofT", "UBC", "McGill"])
        encoder.fit(X_train)
        X_test = np.array(["UnknownUni"])
        result = encoder.transform(X_test)
        assert result.shape == (1, 1)
        assert abs(result[0, 0] - encoder.global_mean_) < 1e-10

    def test_handle_unknown_error(self):
        """Unknown categories raise ValueError when handle_unknown='error'."""
        encoder = UniversityEncoder(strategy='onehot', handle_unknown='error')
        X_train = np.array(["UofT", "UBC"])
        encoder.fit(X_train)
        with pytest.raises(ValueError, match="Unknown category"):
            encoder.transform(np.array(["McGill"]))

    def test_compute_target_encoding(self):
        """_compute_target_encoding computes smoothed target encoding."""
        encoder = UniversityEncoder()
        X = np.array(["UofT", "UBC", "McGill", "UofT", "UBC"])
        y = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
        result = encoder._compute_target_encoding(X, y, smoothing=10.0)
        assert isinstance(result, dict)
        assert len(result) == 3
        global_mean = np.mean(y)
        # UofT: n=2, mean=1.0
        expected_uoft = (2 * 1.0 + 10 * global_mean) / (2 + 10)
        assert abs(result['UofT'] - expected_uoft) < 1e-10

    def test_compute_loo_encoding(self):
        """_compute_loo_encoding computes leave-one-out target encoding."""
        encoder = UniversityEncoder()
        encoder.target_config = TargetEncodingConfig(smoothing=10.0)
        X = np.array(["UofT", "UBC", "McGill", "UofT", "UBC"])
        y = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
        result = encoder._compute_loo_encoding(X, y)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 1)
        # McGill has n=1, so LOO returns global mean
        global_mean = np.mean(y)
        assert abs(result[2, 0] - global_mean) < 1e-10


# =============================================================================
#                    PROGRAMENCODER TESTS
# =============================================================================

class TestProgramEncoder:
    """Tests for ProgramEncoder."""

    def test_init_default_params(self):
        """ProgramEncoder can be created with default parameters."""
        encoder = ProgramEncoder()
        assert isinstance(encoder, BaseEncoder)
        assert encoder.is_fitted_ is False

    def test_fit_with_target(self):
        """fit() with y computes smoothed target encoding for programs."""
        encoder = ProgramEncoder()
        X = np.array(["CS", "Engineering", "Biology", "CS", "Engineering"])
        y = np.array([1, 0, 1, 1, 0])
        result = encoder.fit(X, y)
        assert result is encoder
        assert encoder.is_fitted_ is True
        assert encoder.global_mean_ == np.mean(y)
        assert 'CS' in encoder.encoding_map_

    def test_fit_without_target(self):
        """fit() without y uses frequency-based fallback."""
        encoder = ProgramEncoder()
        X = np.array(["CS", "Engineering", "Biology"])
        result = encoder.fit(X)
        assert result is encoder
        assert encoder.is_fitted_ is True
        # Frequency: each appears once out of 3
        assert abs(encoder.encoding_map_['CS'] - 1.0/3.0) < 1e-10

    def test_transform(self):
        """transform() applies learned encoding to produce (n, 1) array."""
        encoder = ProgramEncoder()
        X = np.array(["CS", "Engineering", "Biology"])
        encoder.fit(X)
        result = encoder.transform(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)

    def test_transform_unknown_category(self):
        """transform() uses global_mean for unknown categories."""
        encoder = ProgramEncoder()
        X_train = np.array(["CS", "Engineering", "Biology"])
        encoder.fit(X_train)
        result = encoder.transform(np.array(["Physics"]))
        assert result.shape == (1, 1)
        assert abs(result[0, 0] - encoder.global_mean_) < 1e-10

    def test_fit_transform_with_loo(self):
        """fit_transform() with y uses leave-one-out encoding."""
        encoder = ProgramEncoder()
        X = np.array(["CS", "Engineering", "Biology", "CS", "Engineering"])
        y = np.array([1, 0, 1, 1, 0])
        result = encoder.fit_transform(X, y)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 1)
        # Biology has n=1, LOO returns global mean
        global_mean = np.mean(y)
        assert abs(result[2, 0] - global_mean) < 1e-10

    def test_fit_transform_without_loo(self):
        """fit_transform() without y falls back to frequency encoding."""
        encoder = ProgramEncoder()
        X = np.array(["CS", "Engineering", "Biology"])
        result = encoder.fit_transform(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)

    def test_inverse_transform(self):
        """inverse_transform() finds closest category by encoded value."""
        encoder = ProgramEncoder()
        X = np.array(["CS", "Engineering", "Biology"])
        encoder.fit(X)
        encoded = encoder.transform(X)
        recovered = encoder.inverse_transform(encoded)
        assert isinstance(recovered, np.ndarray)
        assert len(recovered) == 3

    def test_get_feature_names_out_no_hierarchy(self):
        """get_feature_names_out() returns ['program_target'] without hierarchy."""
        encoder = ProgramEncoder()
        result = encoder.get_feature_names_out()
        assert isinstance(result, list)
        assert result == ['program_target']

    def test_get_feature_names_out_with_hierarchy(self):
        """get_feature_names_out() returns level-based names with hierarchy."""
        hierarchy = HierarchicalGrouping(
            levels=['program', 'faculty'],
            mappings={'program': {'CS': 'Science'}, 'faculty': {'Science': 'UofT'}},
        )
        encoder = ProgramEncoder(hierarchy=hierarchy)
        result = encoder.get_feature_names_out()
        assert result == ['program_target', 'faculty_target']


# =============================================================================
#                    TERMENCODER TESTS
# =============================================================================

class TestTermEncoder:
    """Tests for TermEncoder."""

    def test_standard_terms_class_attribute(self):
        """STANDARD_TERMS is ['Fall', 'Winter', 'Summer']."""
        assert TermEncoder.STANDARD_TERMS == ['Fall', 'Winter', 'Summer']

    def test_init_default_params(self):
        """TermEncoder can be created with default parameters."""
        encoder = TermEncoder()
        assert isinstance(encoder, BaseEncoder)
        assert encoder.use_cyclical is True
        assert encoder.include_year is True
        assert encoder.year_base == 2020

    def test_fit_returns_self(self):
        """fit() returns the encoder instance and sets is_fitted_."""
        encoder = TermEncoder()
        result = encoder.fit(np.array(["Fall 2023", "Winter 2024"]))
        assert result is encoder
        assert encoder.is_fitted_ is True

    def test_fit_computes_year_stats(self):
        """fit() computes year_mean_ and year_std_ from parsed years."""
        encoder = TermEncoder()
        X = np.array(["Fall 2023", "Winter 2024", "Summer 2023"])
        encoder.fit(X)
        # Years: [2023, 2024, 2023], mean = 2023.333...
        expected_mean = np.mean([2023, 2024, 2023])
        assert abs(encoder.year_mean_ - expected_mean) < 1e-10

    def test_transform_cyclical_with_year(self):
        """transform() with cyclical=True, include_year=True returns 3 columns."""
        encoder = TermEncoder(use_cyclical=True, include_year=True)
        X = np.array(["Fall 2023", "Winter 2024", "Summer 2023"])
        result = encoder.fit_transform(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)  # cos, sin, year

    def test_transform_cyclical_without_year(self):
        """transform() with cyclical=True, include_year=False returns 2 columns."""
        encoder = TermEncoder(use_cyclical=True, include_year=False)
        X = np.array(["Fall 2023", "Winter 2024"])
        result = encoder.fit_transform(X)
        assert result.shape == (2, 2)  # cos, sin only

    def test_transform_ordinal_with_year(self):
        """transform() with use_cyclical=False returns ordinal + year."""
        encoder = TermEncoder(use_cyclical=False, include_year=True)
        X = np.array(["Fall 2023", "Winter 2024"])
        result = encoder.fit_transform(X)
        assert result.shape == (2, 2)  # ordinal, year

    def test_transform_ordinal_without_year(self):
        """transform() with use_cyclical=False, include_year=False returns 1 column."""
        encoder = TermEncoder(use_cyclical=False, include_year=False)
        X = np.array(["Fall 2023", "Winter 2024"])
        result = encoder.fit_transform(X)
        assert result.shape == (2, 1)  # ordinal only

    def test_cyclical_values_unit_circle(self):
        """Cyclical encoding values lie on the unit circle (cos^2 + sin^2 = 1)."""
        encoder = TermEncoder(use_cyclical=True, include_year=False)
        X = np.array(["Fall 2023", "Winter 2024", "Summer 2023"])
        result = encoder.fit_transform(X)
        for i in range(3):
            cos_val = result[i, 0]
            sin_val = result[i, 1]
            assert abs(cos_val**2 + sin_val**2 - 1.0) < 1e-10

    def test_inverse_transform_cyclical(self):
        """inverse_transform() recovers term names from cyclical encoding."""
        encoder = TermEncoder(use_cyclical=True, include_year=True)
        X = np.array(["Fall 2023", "Winter 2024", "Summer 2023"])
        encoded = encoder.fit_transform(X)
        recovered = encoder.inverse_transform(encoded)
        assert isinstance(recovered, np.ndarray)
        assert len(recovered) == 3
        # Each recovered string should contain a term name
        for term_str in recovered:
            assert any(t in str(term_str) for t in ['Fall', 'Winter', 'Summer'])

    def test_get_feature_names_out_cyclical(self):
        """get_feature_names_out() returns ['term_cos', 'term_sin', 'year'] for cyclical."""
        encoder = TermEncoder(use_cyclical=True, include_year=True)
        result = encoder.get_feature_names_out()
        assert isinstance(result, list)
        assert result == ['term_cos', 'term_sin', 'year']

    def test_get_feature_names_out_ordinal(self):
        """get_feature_names_out() returns ['term_ordinal', 'year'] for non-cyclical."""
        encoder = TermEncoder(use_cyclical=False, include_year=True)
        result = encoder.get_feature_names_out()
        assert result == ['term_ordinal', 'year']

    def test_parse_term_string_full_format(self):
        """_parse_term_string parses 'Fall 2023' format."""
        encoder = TermEncoder()
        term_name, year = encoder._parse_term_string("Fall 2023")
        assert term_name == 'Fall'
        assert year == 2023

    def test_parse_term_string_dash_format(self):
        """_parse_term_string parses '2023-Fall' format."""
        encoder = TermEncoder()
        term_name, year = encoder._parse_term_string("2023-Fall")
        assert term_name == 'Fall'
        assert year == 2023

    def test_parse_term_string_short_format(self):
        """_parse_term_string parses 'F21' short format."""
        encoder = TermEncoder()
        term_name, year = encoder._parse_term_string("F21")
        assert term_name == 'Fall'
        assert year == 2021

    def test_parse_term_string_winter_short(self):
        """_parse_term_string parses 'W24' short format."""
        encoder = TermEncoder()
        term_name, year = encoder._parse_term_string("W24")
        assert term_name == 'Winter'
        assert year == 2024

    def test_parse_term_string_longer_short_format(self):
        """_parse_term_string parses 'Fa21', 'Wi24', 'Su23' formats."""
        encoder = TermEncoder()
        name, year = encoder._parse_term_string("Fa21")
        assert name == 'Fall'
        assert year == 2021
        name, year = encoder._parse_term_string("Wi24")
        assert name == 'Winter'
        assert year == 2024
        name, year = encoder._parse_term_string("Su23")
        assert name == 'Summer'
        assert year == 2023

    def test_cyclical_encode(self):
        """_cyclical_encode produces cos/sin columns for term indices."""
        encoder = TermEncoder()
        indices = np.array([0, 1, 2])
        result = encoder._cyclical_encode(indices)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)  # cos and sin columns
        # Index 0 (Fall): angle = 0, cos=1, sin=0
        assert abs(result[0, 0] - 1.0) < 1e-10
        assert abs(result[0, 1] - 0.0) < 1e-10


# =============================================================================
#                    DATEENCODER TESTS
# =============================================================================

class TestDateEncoder:
    """Tests for DateEncoder."""

    def test_init_default_params(self):
        """DateEncoder can be created with default parameters."""
        encoder = DateEncoder()
        assert isinstance(encoder, BaseEncoder)
        assert 'month_cyclical' in encoder.features
        assert 'days_since_ref' in encoder.features
        assert 'year' in encoder.features

    def test_fit_returns_self(self):
        """fit() returns encoder and sets is_fitted_."""
        encoder = DateEncoder()
        result = encoder.fit(np.array(["2023-01-01", "2023-06-15"]))
        assert result is encoder
        assert encoder.is_fitted_ is True

    def test_transform_default_features(self):
        """transform() with default features produces 4 columns (cos, sin, days, year)."""
        encoder = DateEncoder()
        X = np.array(["2023-01-01", "2023-06-15"])
        result = encoder.fit_transform(X)
        assert isinstance(result, np.ndarray)
        # Default features: month_cyclical(2 cols) + days_since_ref(1) + year(1) = 4
        assert result.shape == (2, 4)

    def test_transform_month_cyclical(self):
        """Month cyclical encoding produces cos/sin values on unit circle."""
        encoder = DateEncoder(features=['month_cyclical'])
        X = np.array(["2023-01-01", "2023-07-01"])
        result = encoder.fit_transform(X)
        assert result.shape == (2, 2)
        # cos^2 + sin^2 = 1
        for i in range(2):
            assert abs(result[i, 0]**2 + result[i, 1]**2 - 1.0) < 1e-10

    def test_transform_days_since_ref(self):
        """days_since_ref returns correct day counts from reference date."""
        encoder = DateEncoder(features=['days_since_ref'], reference_date='2020-01-01')
        X = np.array(["2020-01-02", "2020-01-11"])
        result = encoder.fit_transform(X)
        assert result.shape == (2, 1)
        assert abs(result[0, 0] - 1.0) < 1e-10   # 1 day
        assert abs(result[1, 0] - 10.0) < 1e-10   # 10 days

    def test_transform_year_feature(self):
        """year feature extracts the year from dates."""
        encoder = DateEncoder(features=['year'])
        X = np.array(["2023-01-01", "2024-06-15"])
        result = encoder.fit_transform(X)
        assert result.shape == (2, 1)
        assert result[0, 0] == 2023.0
        assert result[1, 0] == 2024.0

    def test_fit_transform_method(self):
        """DateEncoder.fit_transform() fits and transforms in one step."""
        encoder = DateEncoder()
        X = np.array(["2023-01-01", "2023-06-15"])
        result = encoder.fit_transform(X)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 2

    def test_inverse_transform_with_days(self):
        """inverse_transform() reconstructs dates from days_since_ref."""
        encoder = DateEncoder(features=['month_cyclical', 'days_since_ref', 'year'])
        X = np.array(["2023-01-15", "2023-06-15"])
        encoded = encoder.fit_transform(X)
        recovered = encoder.inverse_transform(encoded)
        assert isinstance(recovered, np.ndarray)
        assert len(recovered) == 2

    def test_get_feature_names_out_default(self):
        """get_feature_names_out() returns names for default features."""
        encoder = DateEncoder()
        result = encoder.get_feature_names_out()
        assert isinstance(result, list)
        assert 'month_cos' in result
        assert 'month_sin' in result
        assert 'days_since_ref' in result
        assert 'year' in result

    def test_get_feature_names_out_month_only(self):
        """get_feature_names_out() returns ['month_cos', 'month_sin'] for month_cyclical."""
        encoder = DateEncoder(features=['month_cyclical'])
        result = encoder.get_feature_names_out()
        assert result == ['month_cos', 'month_sin']

    def test_parse_date_multiple_formats(self):
        """_parse_date handles multiple date formats."""
        encoder = DateEncoder()
        from datetime import datetime
        # YYYY-MM-DD
        dt = encoder._parse_date("2023-06-15")
        assert dt == datetime(2023, 6, 15)
        # MM/DD/YYYY
        dt = encoder._parse_date("06/15/2023")
        assert dt == datetime(2023, 6, 15)

    def test_parse_date_invalid_raises(self):
        """_parse_date raises ValueError for unparseable date strings."""
        encoder = DateEncoder()
        with pytest.raises(ValueError, match="Cannot parse date"):
            encoder._parse_date("not-a-date")


# =============================================================================
#                    FREQUENCYENCODER TESTS
# =============================================================================

class TestFrequencyEncoder:
    """Tests for FrequencyEncoder."""

    def test_init_default_params(self):
        """FrequencyEncoder can be created with default parameters."""
        encoder = FrequencyEncoder()
        assert isinstance(encoder, BaseEncoder)
        assert encoder.normalize is True

    def test_fit_returns_self(self):
        """fit() returns encoder and sets is_fitted_."""
        encoder = FrequencyEncoder()
        result = encoder.fit(np.array(["cat_a", "cat_b", "cat_a"]))
        assert result is encoder
        assert encoder.is_fitted_ is True

    def test_fit_normalized_frequencies(self):
        """fit() with normalize=True computes frequencies summing to ~1."""
        encoder = FrequencyEncoder(normalize=True)
        X = np.array(["cat_a", "cat_b", "cat_a"])
        encoder.fit(X)
        # cat_a: 2/3, cat_b: 1/3
        assert abs(encoder.frequency_map_['cat_a'] - 2.0/3.0) < 1e-10
        assert abs(encoder.frequency_map_['cat_b'] - 1.0/3.0) < 1e-10

    def test_fit_unnormalized_counts(self):
        """fit() with normalize=False stores raw counts."""
        encoder = FrequencyEncoder(normalize=False)
        X = np.array(["cat_a", "cat_b", "cat_a"])
        encoder.fit(X)
        assert encoder.frequency_map_['cat_a'] == 2.0
        assert encoder.frequency_map_['cat_b'] == 1.0

    def test_transform(self):
        """transform() produces correct frequency values."""
        encoder = FrequencyEncoder()
        X = np.array(["cat_a", "cat_b", "cat_a"])
        encoder.fit(X)
        result = encoder.transform(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)
        assert abs(result[0, 0] - 2.0/3.0) < 1e-10  # cat_a
        assert abs(result[1, 0] - 1.0/3.0) < 1e-10  # cat_b

    def test_transform_unknown_zero(self):
        """Unknown categories get 0 when handle_unknown='zero'."""
        encoder = FrequencyEncoder(handle_unknown='zero')
        X_train = np.array(["cat_a", "cat_b"])
        encoder.fit(X_train)
        result = encoder.transform(np.array(["unknown"]))
        assert result[0, 0] == 0.0

    def test_transform_unknown_min_frequency(self):
        """Unknown categories get min_frequency when handle_unknown='min_frequency'."""
        encoder = FrequencyEncoder(handle_unknown='min_frequency', min_frequency=0.01)
        X_train = np.array(["cat_a", "cat_b"])
        encoder.fit(X_train)
        result = encoder.transform(np.array(["unknown"]))
        assert abs(result[0, 0] - 0.01) < 1e-10

    def test_fit_transform_method(self):
        """FrequencyEncoder.fit_transform() fits and transforms in one step."""
        encoder = FrequencyEncoder()
        X = np.array(["cat_a", "cat_b", "cat_a"])
        result = encoder.fit_transform(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)

    def test_min_frequency_floor(self):
        """Rare categories are floored to min_frequency."""
        encoder = FrequencyEncoder(normalize=True, min_frequency=0.4)
        # cat_a: 1/4 = 0.25 < 0.4, cat_b: 3/4 = 0.75
        X = np.array(["cat_a", "cat_b", "cat_b", "cat_b"])
        encoder.fit(X)
        assert abs(encoder.frequency_map_['cat_a'] - 0.4) < 1e-10  # floored
        assert abs(encoder.frequency_map_['cat_b'] - 0.75) < 1e-10  # not floored

    def test_inverse_transform(self):
        """inverse_transform() finds closest matching category."""
        encoder = FrequencyEncoder()
        X = np.array(["cat_a", "cat_b", "cat_a"])
        encoder.fit(X)
        result = encoder.inverse_transform(np.array([2.0/3.0, 1.0/3.0]))
        assert isinstance(result, np.ndarray)
        assert len(result) == 2
        assert result[0] == 'cat_a'
        assert result[1] == 'cat_b'

    def test_get_feature_names_out(self):
        """get_feature_names_out() returns ['category_frequency']."""
        encoder = FrequencyEncoder()
        result = encoder.get_feature_names_out()
        assert isinstance(result, list)
        assert result == ['category_frequency']


# =============================================================================
#                    WOEENCODER TESTS
# =============================================================================

class TestWOEEncoder:
    """Tests for WOEEncoder."""

    def test_init_default_params(self):
        """WOEEncoder can be created with default parameters."""
        encoder = WOEEncoder()
        assert isinstance(encoder, BaseEncoder)
        assert encoder.regularization == 0.5
        assert encoder.iv_ == 0.0

    def test_fit_returns_self(self):
        """fit() returns encoder and sets is_fitted_."""
        encoder = WOEEncoder()
        X = np.array(["cat_a", "cat_b", "cat_a", "cat_b"])
        y = np.array([1, 0, 1, 0])
        result = encoder.fit(X, y)
        assert result is encoder
        assert encoder.is_fitted_ is True

    def test_fit_computes_woe_values(self):
        """fit() computes WOE values for each category."""
        encoder = WOEEncoder()
        X = np.array(["cat_a", "cat_b", "cat_a", "cat_b"])
        y = np.array([1, 0, 1, 0])
        encoder.fit(X, y)
        assert 'cat_a' in encoder.woe_map_
        assert 'cat_b' in encoder.woe_map_
        # cat_a has 2 positive, 0 negative (+ regularization)
        # cat_b has 0 positive, 2 negative (+ regularization)
        # WOE for cat_a should be positive, cat_b negative
        assert encoder.woe_map_['cat_a'] > 0
        assert encoder.woe_map_['cat_b'] < 0

    def test_transform(self):
        """transform() maps categories to WOE values."""
        encoder = WOEEncoder()
        X = np.array(["cat_a", "cat_b", "cat_a"])
        y = np.array([1, 0, 1])
        encoder.fit(X, y)
        result = encoder.transform(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)
        # cat_a values should be equal
        assert abs(result[0, 0] - result[2, 0]) < 1e-10

    def test_transform_unfitted_returns_zeros(self):
        """transform() on unfitted encoder returns zeros."""
        encoder = WOEEncoder()
        result = encoder.transform(np.array(["cat_a", "cat_b"]))
        assert result.shape == (2, 1)
        np.testing.assert_array_equal(result, np.zeros((2, 1)))

    def test_transform_unknown_category(self):
        """Unknown categories get 0 when handle_unknown='zero'."""
        encoder = WOEEncoder(handle_unknown='zero')
        X = np.array(["cat_a", "cat_b"])
        y = np.array([1, 0])
        encoder.fit(X, y)
        result = encoder.transform(np.array(["unknown"]))
        assert result[0, 0] == 0.0

    def test_fit_transform_method(self):
        """WOEEncoder.fit_transform() fits and transforms in one step."""
        encoder = WOEEncoder()
        X = np.array(["cat_a", "cat_b", "cat_a"])
        y = np.array([1, 0, 1])
        result = encoder.fit_transform(X, y)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)

    def test_inverse_transform(self):
        """inverse_transform() finds closest category by WOE value."""
        encoder = WOEEncoder()
        X = np.array(["cat_a", "cat_b", "cat_a"])
        y = np.array([1, 0, 1])
        encoder.fit(X, y)
        encoded = encoder.transform(X)
        recovered = encoder.inverse_transform(encoded)
        assert isinstance(recovered, np.ndarray)
        assert len(recovered) == 3

    def test_get_feature_names_out(self):
        """get_feature_names_out() returns ['category_woe']."""
        encoder = WOEEncoder()
        result = encoder.get_feature_names_out()
        assert isinstance(result, list)
        assert result == ['category_woe']

    def test_get_information_value(self):
        """get_information_value() returns a non-negative float after fitting."""
        encoder = WOEEncoder()
        X = np.array(["cat_a", "cat_b", "cat_a", "cat_b"])
        y = np.array([1, 0, 1, 0])
        encoder.fit(X, y)
        iv = encoder.get_information_value()
        assert isinstance(iv, float)
        assert iv >= 0.0

    def test_information_value_before_fit(self):
        """get_information_value() returns 0.0 before fitting."""
        encoder = WOEEncoder()
        assert encoder.get_information_value() == 0.0


# =============================================================================
#                    COMPOSITEENCODER TESTS
# =============================================================================

class TestCompositeEncoder:
    """Tests for CompositeEncoder."""

    def test_init_with_empty_list(self):
        """CompositeEncoder accepts an empty list of encoders."""
        encoder = CompositeEncoder(encoders=[])
        result = encoder.fit(np.array(["data"]))
        assert result is encoder
        assert encoder.is_fitted_ is True

    def test_fit_returns_self(self):
        """fit() returns the composite encoder."""
        encoder = CompositeEncoder(encoders=[GPAEncoder()])
        result = encoder.fit(np.array([85.0, 90.0]))
        assert result is encoder
        assert encoder.is_fitted_ is True

    def test_transform_single_encoder(self):
        """transform() with a single GPA encoder works correctly."""
        encoder = CompositeEncoder(encoders=[GPAEncoder(standardize=False)])
        X = np.array([80.0, 90.0, 100.0])
        result = encoder.fit_transform(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)
        np.testing.assert_allclose(result.ravel(), [0.80, 0.90, 1.00])

    def test_transform_multiple_encoders(self):
        """transform() concatenates results from multiple encoders."""
        gpa_enc = GPAEncoder(standardize=False)
        freq_enc = FrequencyEncoder()
        encoder = CompositeEncoder(encoders=[gpa_enc, freq_enc])
        X = np.array([80.0, 90.0, 80.0])
        result = encoder.fit_transform(X)
        assert isinstance(result, np.ndarray)
        # GPA: 1 col, Frequency: 1 col = 2 cols total
        assert result.shape == (3, 2)

    def test_transform_empty_encoders(self):
        """transform() with no encoders returns zero-width array."""
        encoder = CompositeEncoder(encoders=[])
        X = np.array([1.0, 2.0, 3.0])
        encoder.fit(X)
        result = encoder.transform(X)
        assert result.shape == (3, 0)

    def test_fit_transform_method(self):
        """CompositeEncoder.fit_transform() fits and transforms with all encoders."""
        encoder = CompositeEncoder(encoders=[GPAEncoder()])
        X = np.array([85.0, 90.0, 78.0])
        result = encoder.fit_transform(X)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 3

    def test_inverse_transform(self):
        """inverse_transform() returns the input as-is for composite."""
        encoder = CompositeEncoder(encoders=[GPAEncoder()])
        X_encoded = np.array([0.85, 0.90, 0.78])
        result = encoder.inverse_transform(X_encoded)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, X_encoded)

    def test_get_feature_names_out(self):
        """get_feature_names_out() concatenates names from all encoders."""
        gpa_enc = GPAEncoder()
        freq_enc = FrequencyEncoder()
        encoder = CompositeEncoder(encoders=[gpa_enc, freq_enc])
        result = encoder.get_feature_names_out()
        assert isinstance(result, list)
        assert result == ['gpa_normalized', 'category_frequency']

    def test_get_feature_names_out_empty(self):
        """get_feature_names_out() returns empty list for no encoders."""
        encoder = CompositeEncoder(encoders=[])
        assert encoder.get_feature_names_out() == []


# =============================================================================
#                    CONVENIENCE FUNCTIONS TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Tests for create_admission_encoders and encode_admission_features."""

    def test_create_admission_encoders(self):
        """create_admission_encoders returns a dict of 5 standard encoders."""
        result = create_admission_encoders()
        assert isinstance(result, dict)
        assert 'gpa' in result
        assert 'university' in result
        assert 'program' in result
        assert 'term' in result
        assert 'application_date' in result
        assert isinstance(result['gpa'], GPAEncoder)
        assert isinstance(result['university'], UniversityEncoder)
        assert isinstance(result['program'], ProgramEncoder)
        assert isinstance(result['term'], TermEncoder)
        assert isinstance(result['application_date'], DateEncoder)

    def test_encode_admission_features(self):
        """encode_admission_features returns (matrix, names, encoders) tuple."""
        data = [
            {'gpa': 85.0, 'university': 'UofT', 'program': 'CS',
             'term': 'Fall 2023', 'application_date': '2023-01-15'},
            {'gpa': 90.0, 'university': 'UBC', 'program': 'Engineering',
             'term': 'Winter 2024', 'application_date': '2023-06-15'},
        ]
        target = np.array([1, 0])
        matrix, names, encoders = encode_admission_features(data, target)
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape[0] == 2
        assert isinstance(names, list)
        assert len(names) == matrix.shape[1]
        assert isinstance(encoders, dict)

    def test_encode_admission_features_partial_data(self):
        """encode_admission_features works with partial feature data."""
        data = [{'gpa': 85.0}, {'gpa': 90.0}]
        target = np.array([1, 0])
        matrix, names, encoders = encode_admission_features(data, target)
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape[0] == 2

    def test_encode_admission_features_empty_data(self):
        """encode_admission_features with missing features returns empty columns."""
        data = [{'other': 'value'}, {'other': 'value2'}]
        target = np.array([1, 0])
        matrix, names, encoders = encode_admission_features(data, target)
        assert isinstance(matrix, np.ndarray)

    def test_encode_with_prefitted_encoders(self):
        """encode_admission_features uses pre-fitted encoders when provided."""
        data = [
            {'gpa': 85.0, 'university': 'UofT'},
            {'gpa': 90.0, 'university': 'UBC'},
        ]
        target = np.array([1, 0])
        # First fit
        _, _, fitted_encoders = encode_admission_features(data, target)
        # Use pre-fitted encoders
        matrix2, names2, _ = encode_admission_features(data, target, encoders=fitted_encoders)
        assert isinstance(matrix2, np.ndarray)
        assert matrix2.shape[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
