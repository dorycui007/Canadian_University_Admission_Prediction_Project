"""
Unit Tests for Design Matrix Builder Module
=============================================

Tests for src/features/design_matrix.py — FeatureSpec, InteractionSpec,
DesignMatrixConfig, FittedTransform dataclasses, BaseFeatureTransformer ABC,
concrete transformers (NumericScaler, OneHotEncoder, DummyEncoder,
OrdinalEncoder), InteractionBuilder, DesignMatrixBuilder, and validation
utilities.
"""

import pytest
import numpy as np

from src.features.design_matrix import (
    FeatureSpec, InteractionSpec, DesignMatrixConfig, FittedTransform,
    BaseFeatureTransformer, NumericScaler, OneHotEncoder, DummyEncoder,
    OrdinalEncoder, InteractionBuilder, DesignMatrixBuilder,
    validate_design_matrix, check_column_rank, identify_collinear_features,
    build_admission_matrix, create_polynomial_features,
)


# =============================================================================
#                    FEATURESPEC TESTS
# =============================================================================

class TestFeatureSpec:
    """Tests for FeatureSpec dataclass."""

    def test_defaults(self):
        """FeatureSpec has correct default values for optional fields."""
        spec = FeatureSpec(name='gpa', dtype='numeric')
        assert spec.name == 'gpa'
        assert spec.dtype == 'numeric'
        assert spec.encoding == 'standard'
        assert spec.categories is None
        assert spec.handle_unknown == 'error'
        assert spec.missing_strategy == 'mean'

    def test_custom_values(self):
        """Custom values are stored correctly for a categorical feature."""
        spec = FeatureSpec(
            name='university',
            dtype='categorical',
            encoding='onehot',
            categories=['UofT', 'UBC', 'McGill'],
            handle_unknown='ignore',
            missing_strategy='mode',
        )
        assert spec.name == 'university'
        assert spec.dtype == 'categorical'
        assert spec.encoding == 'onehot'
        assert spec.categories == ['UofT', 'UBC', 'McGill']
        assert spec.handle_unknown == 'ignore'
        assert spec.missing_strategy == 'mode'


# =============================================================================
#                    INTERACTIONSPEC TESTS
# =============================================================================

class TestInteractionSpec:
    """Tests for InteractionSpec dataclass."""

    def test_defaults(self):
        """InteractionSpec has interaction_type='multiplicative' and degree=2."""
        spec = InteractionSpec(features=['gpa', 'program_type'])
        assert spec.features == ['gpa', 'program_type']
        assert spec.interaction_type == 'multiplicative'
        assert spec.degree == 2


# =============================================================================
#                    DESIGNMATRIXCONFIG TESTS
# =============================================================================

class TestDesignMatrixConfig:
    """Tests for DesignMatrixConfig dataclass."""

    def test_defaults(self):
        """Default config has empty lists, intercept=True, sparse_threshold=0.5, drop_first=True."""
        config = DesignMatrixConfig()
        assert config.feature_specs == []
        assert config.interactions == []
        assert config.include_intercept is True
        assert config.sparse_threshold == 0.5
        assert config.drop_first is True

    def test_include_intercept_false(self):
        """include_intercept can be set to False."""
        config = DesignMatrixConfig(include_intercept=False)
        assert config.include_intercept is False

    def test_lists_default_independent(self):
        """Each instance gets its own feature_specs and interactions lists."""
        c1 = DesignMatrixConfig()
        c2 = DesignMatrixConfig()
        c1.feature_specs.append(FeatureSpec(name='x', dtype='numeric'))
        assert len(c2.feature_specs) == 0


# =============================================================================
#                    FITTEDTRANSFORM TESTS
# =============================================================================

class TestFittedTransform:
    """Tests for FittedTransform dataclass."""

    def test_defaults(self):
        """Default FittedTransform has None for all optional fields."""
        ft = FittedTransform(feature_name='gpa')
        assert ft.feature_name == 'gpa'
        assert ft.mean is None
        assert ft.std is None
        assert ft.min_val is None
        assert ft.max_val is None
        assert ft.categories is None
        assert ft.category_to_index is None


# =============================================================================
#                    BASEFEATURETRANSFORMER ABC TESTS
# =============================================================================

class TestBaseFeatureTransformerABC:
    """Tests for BaseFeatureTransformer abstract base class enforcement."""

    def test_cannot_instantiate_directly(self):
        """BaseFeatureTransformer cannot be instantiated because it is abstract."""
        with pytest.raises(TypeError):
            BaseFeatureTransformer()


# =============================================================================
#                    NUMERICSCALER TESTS
# =============================================================================

class TestNumericScaler:
    """Tests for NumericScaler."""

    def test_init_defaults(self):
        """NumericScaler initializes with correct defaults."""
        scaler = NumericScaler()
        assert scaler.method == 'standard'
        assert scaler.feature_name == 'numeric'
        assert scaler.is_fitted_ is False

    def test_fit_returns_self(self):
        """fit() returns the scaler for method chaining."""
        scaler = NumericScaler()
        result = scaler.fit(np.array([1.0, 2.0, 3.0]))
        assert result is scaler
        assert scaler.is_fitted_ is True

    def test_fit_standard_computes_mean_std(self):
        """fit() with method='standard' computes mean_ and std_."""
        scaler = NumericScaler(method='standard')
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scaler.fit(X)
        assert abs(scaler.mean_ - 3.0) < 1e-10
        expected_std = float(np.std(X))
        assert abs(scaler.std_ - expected_std) < 1e-10

    def test_fit_minmax_computes_min_max(self):
        """fit() with method='minmax' computes min_ and max_."""
        scaler = NumericScaler(method='minmax')
        X = np.array([10.0, 20.0, 30.0])
        scaler.fit(X)
        assert scaler.min_ == 10.0
        assert scaler.max_ == 30.0

    def test_fit_standard_zero_std(self):
        """fit() with constant data sets std_ to 1.0 to avoid division by zero."""
        scaler = NumericScaler(method='standard')
        X = np.array([5.0, 5.0, 5.0])
        scaler.fit(X)
        assert scaler.std_ == 1.0

    def test_transform_standard(self):
        """transform() with standard method produces z-scored values."""
        scaler = NumericScaler(method='standard')
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scaler.fit(X)
        result = scaler.transform(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 1)
        # Z-scored mean should be ~0
        assert abs(np.mean(result)) < 1e-10

    def test_transform_minmax(self):
        """transform() with minmax scales to [0, 1]."""
        scaler = NumericScaler(method='minmax')
        X = np.array([10.0, 20.0, 30.0])
        scaler.fit(X)
        result = scaler.transform(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)
        np.testing.assert_allclose(result.ravel(), [0.0, 0.5, 1.0])

    def test_transform_minmax_new_data(self):
        """transform() with minmax applies training min/max to new data."""
        scaler = NumericScaler(method='minmax')
        X_train = np.array([10.0, 20.0, 30.0])
        scaler.fit(X_train)
        X_test = np.array([15.0, 25.0])
        result = scaler.transform(X_test)
        np.testing.assert_allclose(result.ravel(), [0.25, 0.75])

    def test_fit_transform(self):
        """fit_transform() fits and transforms in one step."""
        scaler = NumericScaler(method='standard', feature_name='gpa')
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scaler.fit_transform(data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 1)
        assert abs(np.mean(result)) < 1e-10

    def test_get_feature_names(self):
        """get_feature_names() returns [feature_name]."""
        scaler = NumericScaler(method='standard', feature_name='gpa')
        result = scaler.get_feature_names()
        assert isinstance(result, list)
        assert result == ['gpa']

    def test_inverse_transform_standard(self):
        """inverse_transform() reverses z-score standardization."""
        scaler = NumericScaler(method='standard')
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scaler.fit(X)
        scaled = scaler.transform(X)
        recovered = scaler.inverse_transform(scaled)
        np.testing.assert_allclose(recovered, X, atol=1e-10)

    def test_inverse_transform_minmax(self):
        """inverse_transform() reverses min-max scaling."""
        scaler = NumericScaler(method='minmax')
        X = np.array([10.0, 20.0, 30.0])
        scaler.fit(X)
        scaled = scaler.transform(X)
        recovered = scaler.inverse_transform(scaled)
        np.testing.assert_allclose(recovered, X, atol=1e-10)

    def test_inverse_transform_not_fitted(self):
        """inverse_transform() on unfitted scaler returns values as-is."""
        scaler = NumericScaler(method='standard')
        result = scaler.inverse_transform(np.array([0.5, -0.5]))
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [0.5, -0.5])


# =============================================================================
#                    ONEHOTENCODER TESTS
# =============================================================================

class TestOneHotEncoder:
    """Tests for OneHotEncoder."""

    def test_init_defaults(self):
        """OneHotEncoder initializes with correct defaults."""
        encoder = OneHotEncoder()
        assert encoder.feature_name == 'category'
        assert encoder.handle_unknown == 'error'
        assert encoder.is_fitted_ is False

    def test_init_with_categories(self):
        """OneHotEncoder with pre-specified categories is immediately fitted."""
        encoder = OneHotEncoder(categories=['a', 'b', 'c'])
        assert encoder.is_fitted_ is True
        assert encoder.categories_ == ['a', 'b', 'c']

    def test_fit_returns_self(self):
        """fit() returns the encoder and learns sorted categories."""
        encoder = OneHotEncoder()
        result = encoder.fit(np.array(['c', 'a', 'b']))
        assert result is encoder
        assert encoder.is_fitted_ is True
        assert encoder.categories_ == ['a', 'b', 'c']

    def test_transform_produces_binary_matrix(self):
        """transform() produces a binary matrix with one 1 per row."""
        encoder = OneHotEncoder(feature_name='university')
        X = np.array(['UofT', 'UBC', 'McGill', 'UofT'])
        encoder.fit(X)
        result = encoder.transform(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 3)
        # Each row sums to 1
        np.testing.assert_array_equal(result.sum(axis=1), [1.0, 1.0, 1.0, 1.0])
        # Verify specific encoding (alphabetical: McGill=0, UBC=1, UofT=2)
        np.testing.assert_array_equal(result[0], [0.0, 0.0, 1.0])  # UofT
        np.testing.assert_array_equal(result[2], [1.0, 0.0, 0.0])  # McGill

    def test_transform_unknown_error(self):
        """transform() raises ValueError for unknown category with handle_unknown='error'."""
        encoder = OneHotEncoder(handle_unknown='error')
        encoder.fit(np.array(['a', 'b']))
        with pytest.raises(ValueError, match="Unknown category"):
            encoder.transform(np.array(['c']))

    def test_transform_unknown_ignore(self):
        """transform() encodes unknown categories as all zeros with handle_unknown='ignore'."""
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(np.array(['a', 'b']))
        result = encoder.transform(np.array(['c']))
        np.testing.assert_array_equal(result[0], [0.0, 0.0])

    def test_fit_transform(self):
        """fit_transform() fits and encodes in one call."""
        encoder = OneHotEncoder(feature_name='university')
        data = np.array(['UofT', 'UBC', 'McGill', 'UofT'])
        result = encoder.fit_transform(data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 3)

    def test_get_feature_names(self):
        """get_feature_names() returns feature_name_category for each category."""
        encoder = OneHotEncoder(feature_name='university')
        encoder.fit(np.array(['UBC', 'UofT', 'McGill']))
        result = encoder.get_feature_names()
        assert isinstance(result, list)
        assert result == ['university_McGill', 'university_UBC', 'university_UofT']

    def test_get_feature_names_unfitted(self):
        """get_feature_names() on unfitted encoder returns [feature_name]."""
        encoder = OneHotEncoder(feature_name='university')
        result = encoder.get_feature_names()
        assert result == ['university']


# =============================================================================
#                    DUMMYENCODER TESTS
# =============================================================================

class TestDummyEncoderStub:
    """Tests for DummyEncoder."""

    def test_init_defaults(self):
        """DummyEncoder initializes with correct defaults."""
        encoder = DummyEncoder()
        assert encoder.feature_name == 'category'
        assert encoder.drop == 'first'
        assert encoder.is_fitted_ is False

    def test_fit_returns_self(self):
        """fit() learns categories and drops the first alphabetically."""
        encoder = DummyEncoder(feature_name='program')
        data = np.array(['CS', 'Math', 'Physics', 'CS'])
        result = encoder.fit(data)
        assert result is encoder
        assert encoder.is_fitted_ is True
        assert encoder.dropped_category_ == 'CS'  # first alphabetically
        assert encoder.categories_ == ['Math', 'Physics']

    def test_fit_drop_last(self):
        """fit() with drop='last' drops the last alphabetical category."""
        encoder = DummyEncoder(drop='last')
        encoder.fit(np.array(['a', 'b', 'c']))
        assert encoder.dropped_category_ == 'c'
        assert encoder.categories_ == ['a', 'b']

    def test_fit_drop_specific(self):
        """fit() with drop=specific_name drops that category."""
        encoder = DummyEncoder(drop='b')
        encoder.fit(np.array(['a', 'b', 'c']))
        assert encoder.dropped_category_ == 'b'
        assert encoder.categories_ == ['a', 'c']

    def test_transform_k_minus_1_columns(self):
        """transform() produces k-1 columns (reference category is all zeros)."""
        encoder = DummyEncoder(feature_name='program')
        data = np.array(['CS', 'Math', 'Physics', 'CS'])
        encoder.fit(data)
        result = encoder.transform(data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 2)  # 3 categories - 1 = 2 columns
        # CS is dropped (reference), so CS rows should be all zeros
        np.testing.assert_array_equal(result[0], [0.0, 0.0])  # CS
        np.testing.assert_array_equal(result[3], [0.0, 0.0])  # CS

    def test_transform_reference_all_zeros(self):
        """The dropped/reference category is encoded as all zeros."""
        encoder = DummyEncoder(feature_name='uni')
        encoder.fit(np.array(['McGill', 'UBC', 'UofT']))
        # McGill is first alphabetically, so dropped
        result = encoder.transform(np.array(['McGill']))
        np.testing.assert_array_equal(result[0], [0.0, 0.0])

    def test_transform_unknown_error(self):
        """transform() raises ValueError for unknown with handle_unknown='error'."""
        encoder = DummyEncoder(handle_unknown='error')
        encoder.fit(np.array(['a', 'b', 'c']))
        with pytest.raises(ValueError, match="Unknown category"):
            encoder.transform(np.array(['d']))

    def test_fit_transform(self):
        """fit_transform() fits and transforms in one call."""
        encoder = DummyEncoder(feature_name='program')
        data = np.array(['CS', 'Math', 'Physics', 'CS'])
        result = encoder.fit_transform(data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 2)

    def test_get_feature_names(self):
        """get_feature_names() returns names excluding dropped category."""
        encoder = DummyEncoder(feature_name='program')
        encoder.fit(np.array(['CS', 'Math', 'Physics']))
        result = encoder.get_feature_names()
        assert isinstance(result, list)
        assert result == ['program_Math', 'program_Physics']

    def test_get_feature_names_unfitted(self):
        """get_feature_names() on unfitted encoder returns [feature_name]."""
        encoder = DummyEncoder(feature_name='program')
        result = encoder.get_feature_names()
        assert result == ['program']


# =============================================================================
#                    ORDINALENCODER TESTS
# =============================================================================

class TestOrdinalEncoderStub:
    """Tests for OrdinalEncoder."""

    def test_init_defaults(self):
        """OrdinalEncoder initializes with correct defaults."""
        encoder = OrdinalEncoder()
        assert encoder.feature_name == 'ordinal'
        assert encoder.is_fitted_ is False

    def test_fit_with_provided_categories(self):
        """fit() with provided categories validates data against them."""
        encoder = OrdinalEncoder(
            feature_name='education',
            categories=['HS', 'Bachelor', 'Master', 'PhD'],
        )
        data = np.array(['Bachelor', 'PhD', 'HS', 'Master'])
        result = encoder.fit(data)
        assert result is encoder
        assert encoder.is_fitted_ is True
        assert encoder.categories_ == ['HS', 'Bachelor', 'Master', 'PhD']

    def test_fit_without_categories_infers(self):
        """fit() without provided categories infers alphabetical order."""
        encoder = OrdinalEncoder(feature_name='size')
        encoder.fit(np.array(['medium', 'large', 'small']))
        assert encoder.categories_ == ['large', 'medium', 'small']

    def test_fit_unknown_category_raises(self):
        """fit() raises ValueError if data contains unknown category."""
        encoder = OrdinalEncoder(categories=['a', 'b'])
        with pytest.raises(ValueError, match="Unknown category"):
            encoder.fit(np.array(['a', 'c']))

    def test_transform_produces_ordinal_integers(self):
        """transform() converts categories to ordinal integers in (n, 1) shape."""
        encoder = OrdinalEncoder(
            feature_name='education',
            categories=['HS', 'Bachelor', 'Master', 'PhD'],
        )
        data = np.array(['Bachelor', 'PhD', 'HS', 'Master'])
        encoder.fit(data)
        result = encoder.transform(data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 1)
        np.testing.assert_allclose(result.ravel(), [1.0, 3.0, 0.0, 2.0])

    def test_transform_unknown_raises(self):
        """transform() raises ValueError for unknown category."""
        encoder = OrdinalEncoder(categories=['a', 'b', 'c'])
        encoder.fit(np.array(['a', 'b', 'c']))
        with pytest.raises(ValueError, match="Unknown category"):
            encoder.transform(np.array(['d']))

    def test_fit_transform(self):
        """fit_transform() fits and transforms in one call."""
        encoder = OrdinalEncoder(
            feature_name='education',
            categories=['HS', 'Bachelor', 'Master', 'PhD'],
        )
        data = np.array(['Bachelor', 'PhD', 'HS', 'Master'])
        result = encoder.fit_transform(data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 1)
        np.testing.assert_allclose(result.ravel(), [1.0, 3.0, 0.0, 2.0])

    def test_get_feature_names(self):
        """get_feature_names() returns [feature_name]."""
        encoder = OrdinalEncoder(feature_name='education')
        result = encoder.get_feature_names()
        assert isinstance(result, list)
        assert result == ['education']


# =============================================================================
#                    ORDINALENCODER COVERAGE TESTS
# =============================================================================

class TestOrdinalEncoderCoverage:
    """Additional coverage for OrdinalEncoder."""

    def test_fit(self):
        """OrdinalEncoder.fit() learns or validates category ordering."""
        encoder = OrdinalEncoder(
            feature_name='education',
            categories=['HS', 'Bachelor', 'Master', 'PhD'],
        )
        data = np.array(['Bachelor', 'PhD', 'HS', 'Master'])
        result = encoder.fit(data)
        assert result is encoder
        assert encoder.is_fitted_ is True
        assert encoder.category_to_idx_ == {'HS': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}

    def test_transform(self):
        """OrdinalEncoder.transform() converts categories to ordinal integers."""
        encoder = OrdinalEncoder(
            feature_name='education',
            categories=['HS', 'Bachelor', 'Master', 'PhD'],
        )
        encoder.fit(np.array(['HS', 'Bachelor', 'Master', 'PhD']))
        result = encoder.transform(np.array(['HS', 'PhD']))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 1)
        np.testing.assert_allclose(result.ravel(), [0.0, 3.0])

    def test_fit_transform(self):
        """OrdinalEncoder.fit_transform() fits and transforms in one call."""
        encoder = OrdinalEncoder(
            feature_name='education',
            categories=['HS', 'Bachelor', 'Master', 'PhD'],
        )
        data = np.array(['Bachelor', 'PhD', 'HS', 'Master'])
        result = encoder.fit_transform(data)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result.ravel(), [1.0, 3.0, 0.0, 2.0])

    def test_get_feature_names(self):
        """OrdinalEncoder.get_feature_names() returns the feature name."""
        encoder = OrdinalEncoder(
            feature_name='education',
            categories=['HS', 'Bachelor', 'Master', 'PhD'],
        )
        result = encoder.get_feature_names()
        assert isinstance(result, list)
        assert result == ['education']


# =============================================================================
#                    INTERACTIONBUILDER TESTS
# =============================================================================

class TestInteractionBuilderStub:
    """Tests for InteractionBuilder."""

    def test_build_with_no_specs(self):
        """build_interactions with empty specs returns zero-width array."""
        builder = InteractionBuilder(specs=[])
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result_cols, result_names = builder.build_interactions(X, ['f1', 'f2'])
        assert result_cols.shape == (2, 0)
        assert result_names == []

    def test_build_multiplicative(self):
        """build_interactions with multiplicative spec produces element-wise product."""
        spec = InteractionSpec(features=['f1', 'f2'], interaction_type='multiplicative')
        builder = InteractionBuilder(specs=[spec])
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result_cols, result_names = builder.build_interactions(X, ['f1', 'f2'])
        assert result_cols.shape == (3, 1)
        np.testing.assert_allclose(result_cols.ravel(), [2.0, 12.0, 30.0])
        assert result_names == ['f1_x_f2']

    def test_build_polynomial(self):
        """build_interactions with polynomial spec produces power terms."""
        spec = InteractionSpec(features=['f1'], interaction_type='polynomial', degree=3)
        builder = InteractionBuilder(specs=[spec])
        X = np.array([[2.0, 0.0], [3.0, 0.0]])
        result_cols, result_names = builder.build_interactions(X, ['f1', 'f2'])
        assert result_cols.shape == (2, 2)  # x^2 and x^3
        np.testing.assert_allclose(result_cols[:, 0], [4.0, 9.0])   # x^2
        np.testing.assert_allclose(result_cols[:, 1], [8.0, 27.0])  # x^3
        assert result_names == ['f1_pow2', 'f1_pow3']

    def test_build_unknown_feature_raises(self):
        """build_interactions raises ValueError for unknown feature name."""
        spec = InteractionSpec(features=['f1', 'unknown'], interaction_type='multiplicative')
        builder = InteractionBuilder(specs=[spec])
        X = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="Feature 'unknown' not found"):
            builder.build_interactions(X, ['f1', 'f2'])


# =============================================================================
#                    INTERACTIONBUILDER COVERAGE TESTS
# =============================================================================

class TestInteractionBuilderCoverage:
    """Cover InteractionBuilder._multiply_features and _polynomial_features."""

    def test_multiply_features(self):
        """_multiply_features multiplies columns element-wise."""
        spec = InteractionSpec(features=['f1', 'f2'], interaction_type='multiplicative')
        builder = InteractionBuilder(specs=[spec])
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = builder._multiply_features(X, [0, 1])
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [2.0, 12.0, 30.0])

    def test_multiply_three_features(self):
        """_multiply_features multiplies three columns element-wise."""
        builder = InteractionBuilder(specs=[])
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = builder._multiply_features(X, [0, 1, 2])
        np.testing.assert_allclose(result, [6.0, 120.0])

    def test_polynomial_features(self):
        """_polynomial_features creates x^2, x^3 columns."""
        builder = InteractionBuilder(specs=[])
        X = np.array([[2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
        result = builder._polynomial_features(X, index=0, degree=3)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)  # x^2 and x^3
        np.testing.assert_allclose(result[:, 0], [4.0, 9.0, 16.0])   # x^2
        np.testing.assert_allclose(result[:, 1], [8.0, 27.0, 64.0])  # x^3

    def test_polynomial_features_degree_1(self):
        """_polynomial_features with degree=1 returns zero-width array."""
        builder = InteractionBuilder(specs=[])
        X = np.array([[2.0], [3.0]])
        result = builder._polynomial_features(X, index=0, degree=1)
        assert result.shape == (2, 0)


# =============================================================================
#                    DESIGNMATRIXBUILDER TESTS
# =============================================================================

class TestDesignMatrixBuilder:
    """Tests for DesignMatrixBuilder."""

    def test_init(self):
        """DesignMatrixBuilder initializes with config."""
        config = DesignMatrixConfig()
        builder = DesignMatrixBuilder(config)
        assert builder.is_fitted_ is False
        assert builder.feature_names_ == []

    def test_fit_returns_self(self):
        """fit() returns the builder for method chaining."""
        config = DesignMatrixConfig(
            feature_specs=[FeatureSpec(name='gpa', dtype='numeric')],
        )
        builder = DesignMatrixBuilder(config)
        data = [{'gpa': 3.5}, {'gpa': 3.0}, {'gpa': 3.9}]
        result = builder.fit(data)
        assert result is builder
        assert builder.is_fitted_ is True

    def test_fit_creates_transformers(self):
        """fit() creates and fits appropriate transformers for each feature spec."""
        config = DesignMatrixConfig(
            feature_specs=[
                FeatureSpec(name='gpa', dtype='numeric'),
                FeatureSpec(name='university', dtype='categorical', encoding='onehot'),
            ],
        )
        builder = DesignMatrixBuilder(config)
        data = [
            {'gpa': 3.5, 'university': 'UofT'},
            {'gpa': 3.0, 'university': 'UBC'},
        ]
        builder.fit(data)
        assert 'gpa' in builder.transformers_
        assert 'university' in builder.transformers_
        assert isinstance(builder.transformers_['gpa'], NumericScaler)
        assert isinstance(builder.transformers_['university'], OneHotEncoder)

    def test_transform_with_intercept(self):
        """transform() with include_intercept=True prepends column of 1s."""
        config = DesignMatrixConfig(
            feature_specs=[FeatureSpec(name='gpa', dtype='numeric')],
            include_intercept=True,
        )
        builder = DesignMatrixBuilder(config)
        data = [{'gpa': 3.5}, {'gpa': 3.0}, {'gpa': 3.9}]
        builder.fit(data)
        result = builder.transform(data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)  # intercept + gpa
        # First column should be all 1s
        np.testing.assert_array_equal(result[:, 0], [1.0, 1.0, 1.0])

    def test_transform_without_intercept(self):
        """transform() with include_intercept=False has no 1s column."""
        config = DesignMatrixConfig(
            feature_specs=[FeatureSpec(name='gpa', dtype='numeric')],
            include_intercept=False,
        )
        builder = DesignMatrixBuilder(config)
        data = [{'gpa': 3.5}, {'gpa': 3.0}]
        builder.fit(data)
        result = builder.transform(data)
        assert result.shape == (2, 1)  # just gpa

    def test_transform_categorical_onehot(self):
        """transform() produces one-hot columns for categorical features."""
        config = DesignMatrixConfig(
            feature_specs=[
                FeatureSpec(name='university', dtype='categorical', encoding='onehot'),
            ],
            include_intercept=False,
        )
        builder = DesignMatrixBuilder(config)
        data = [
            {'university': 'UofT'},
            {'university': 'UBC'},
            {'university': 'McGill'},
        ]
        builder.fit(data)
        result = builder.transform(data)
        assert result.shape == (3, 3)
        # Each row sums to 1
        np.testing.assert_array_equal(result.sum(axis=1), [1.0, 1.0, 1.0])

    def test_transform_categorical_dummy(self):
        """transform() produces k-1 dummy columns for categorical features."""
        config = DesignMatrixConfig(
            feature_specs=[
                FeatureSpec(name='university', dtype='categorical', encoding='dummy'),
            ],
            include_intercept=False,
        )
        builder = DesignMatrixBuilder(config)
        data = [
            {'university': 'UofT'},
            {'university': 'UBC'},
            {'university': 'McGill'},
        ]
        builder.fit(data)
        result = builder.transform(data)
        assert result.shape == (3, 2)  # 3 categories - 1 = 2

    def test_transform_ordinal(self):
        """transform() produces ordinal encoding for ordinal features."""
        config = DesignMatrixConfig(
            feature_specs=[
                FeatureSpec(name='level', dtype='ordinal',
                           categories=['low', 'med', 'high']),
            ],
            include_intercept=False,
        )
        builder = DesignMatrixBuilder(config)
        data = [{'level': 'low'}, {'level': 'high'}, {'level': 'med'}]
        builder.fit(data)
        result = builder.transform(data)
        assert result.shape == (3, 1)
        np.testing.assert_allclose(result.ravel(), [0.0, 2.0, 1.0])

    def test_fit_transform(self):
        """fit_transform() fits and transforms in one call."""
        config = DesignMatrixConfig(
            feature_specs=[FeatureSpec(name='gpa', dtype='numeric')],
        )
        builder = DesignMatrixBuilder(config)
        data = [{'gpa': 3.5}, {'gpa': 3.0}, {'gpa': 3.9}, {'gpa': 2.8}, {'gpa': 3.5}]
        result = builder.fit_transform(data)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 5

    def test_get_feature_names(self):
        """get_feature_names() returns all output feature names in order."""
        config = DesignMatrixConfig(
            feature_specs=[FeatureSpec(name='gpa', dtype='numeric')],
            include_intercept=True,
        )
        builder = DesignMatrixBuilder(config)
        data = [{'gpa': 3.5}, {'gpa': 3.0}]
        builder.fit(data)
        result = builder.get_feature_names()
        assert isinstance(result, list)
        assert 'intercept' in result
        assert 'gpa' in result

    def test_get_feature_index(self):
        """get_feature_index() returns the correct column index."""
        config = DesignMatrixConfig(
            feature_specs=[FeatureSpec(name='gpa', dtype='numeric')],
            include_intercept=True,
        )
        builder = DesignMatrixBuilder(config)
        data = [{'gpa': 3.5}, {'gpa': 3.0}]
        builder.fit(data)
        assert builder.get_feature_index('intercept') == 0
        assert builder.get_feature_index('gpa') == 1

    def test_get_feature_index_not_found(self):
        """get_feature_index() raises ValueError for unknown feature."""
        config = DesignMatrixConfig(feature_specs=[])
        builder = DesignMatrixBuilder(config)
        builder.fit([{}])
        with pytest.raises(ValueError, match="Feature .* not found"):
            builder.get_feature_index('nonexistent')

    def test_extract_feature(self):
        """_extract_feature() extracts a single feature array from dicts."""
        config = DesignMatrixConfig()
        builder = DesignMatrixBuilder(config)
        data = [{'gpa': 3.7}, {'gpa': 3.2}, {'gpa': 3.9}]
        result = builder._extract_feature(data, 'gpa')
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        np.testing.assert_allclose(result, [3.7, 3.2, 3.9])

    def test_extract_feature_missing_key(self):
        """_extract_feature() returns NaN for missing keys."""
        config = DesignMatrixConfig()
        builder = DesignMatrixBuilder(config)
        data = [{'gpa': 3.7}, {'other': 'x'}, {'gpa': 3.9}]
        result = builder._extract_feature(data, 'gpa')
        assert len(result) == 3
        assert np.isnan(float(result[1]))

    def test_handle_missing_mean(self):
        """_handle_missing() with strategy='mean' replaces NaN with mean."""
        config = DesignMatrixConfig()
        builder = DesignMatrixBuilder(config)
        values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        spec = FeatureSpec(name='gpa', dtype='numeric', missing_strategy='mean')
        result = builder._handle_missing(values, spec)
        assert isinstance(result, np.ndarray)
        assert not np.any(np.isnan(result))
        # Mean of [1, 3, 5] = 3.0
        assert abs(result[1] - 3.0) < 1e-10
        assert abs(result[3] - 3.0) < 1e-10

    def test_handle_missing_median(self):
        """_handle_missing() with strategy='median' replaces NaN with median."""
        config = DesignMatrixConfig()
        builder = DesignMatrixBuilder(config)
        values = np.array([1.0, np.nan, 5.0, np.nan, 3.0])
        spec = FeatureSpec(name='gpa', dtype='numeric', missing_strategy='median')
        result = builder._handle_missing(values, spec)
        assert not np.any(np.isnan(result))
        assert abs(result[1] - 3.0) < 1e-10  # median of [1, 5, 3] = 3.0

    def test_handle_missing_zero(self):
        """_handle_missing() with strategy='zero' replaces NaN with 0."""
        config = DesignMatrixConfig()
        builder = DesignMatrixBuilder(config)
        values = np.array([1.0, np.nan, 3.0])
        spec = FeatureSpec(name='gpa', dtype='numeric', missing_strategy='zero')
        result = builder._handle_missing(values, spec)
        assert result[1] == 0.0

    def test_handle_missing_no_nans(self):
        """_handle_missing() returns values unchanged when no NaNs present."""
        config = DesignMatrixConfig()
        builder = DesignMatrixBuilder(config)
        values = np.array([1.0, 2.0, 3.0])
        spec = FeatureSpec(name='gpa', dtype='numeric', missing_strategy='mean')
        result = builder._handle_missing(values, spec)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_sparsity_ratio(self):
        """sparsity_ratio() returns the fraction of zeros."""
        config = DesignMatrixConfig()
        builder = DesignMatrixBuilder(config)
        X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = builder.sparsity_ratio(X)
        assert isinstance(result, float)
        # 4 zeros out of 6 elements
        assert abs(result - 4.0/6.0) < 1e-10

    def test_sparsity_ratio_empty(self):
        """sparsity_ratio() returns 0.0 for empty matrix."""
        config = DesignMatrixConfig()
        builder = DesignMatrixBuilder(config)
        X = np.array([]).reshape(0, 0)
        result = builder.sparsity_ratio(X)
        assert result == 0.0

    def test_to_sparse(self):
        """to_sparse() converts dense matrix to scipy sparse CSR format."""
        config = DesignMatrixConfig()
        builder = DesignMatrixBuilder(config)
        X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        try:
            result = builder.to_sparse(X)
            assert hasattr(result, 'toarray') or hasattr(result, 'todense')
            # Verify values are preserved
            np.testing.assert_array_equal(result.toarray(), X)
        except ImportError:
            pytest.skip("scipy not installed")


# =============================================================================
#                    DESIGNMATRIXBUILDER COVERAGE TESTS
# =============================================================================

class TestDesignMatrixBuilderCoverage:
    """Additional coverage for DesignMatrixBuilder methods."""

    def _make_config(self):
        """Helper to create a simple config with one numeric feature."""
        return DesignMatrixConfig(
            feature_specs=[
                FeatureSpec(name='gpa', dtype='numeric', encoding='standard'),
            ],
            interactions=[],
            include_intercept=True,
        )

    def _make_data(self):
        """Helper to create sample data."""
        return [
            {'gpa': 3.7},
            {'gpa': 3.2},
            {'gpa': 3.9},
            {'gpa': 2.8},
            {'gpa': 3.5},
        ]

    def test_transform(self):
        """DesignMatrixBuilder.transform() applies fitted transformations."""
        config = self._make_config()
        builder = DesignMatrixBuilder(config)
        data = self._make_data()
        builder.fit(data)
        result = builder.transform([{'gpa': 3.0}, {'gpa': 3.5}])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)  # intercept + gpa

    def test_fit_transform(self):
        """DesignMatrixBuilder.fit_transform() fits and transforms in one call."""
        config = self._make_config()
        builder = DesignMatrixBuilder(config)
        data = self._make_data()
        result = builder.fit_transform(data)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 5

    def test_get_feature_names(self):
        """DesignMatrixBuilder.get_feature_names() returns all output feature names."""
        config = self._make_config()
        builder = DesignMatrixBuilder(config)
        data = self._make_data()
        builder.fit(data)
        result = builder.get_feature_names()
        assert isinstance(result, list)
        assert 'intercept' in result
        assert 'gpa' in result

    def test_get_feature_index(self):
        """DesignMatrixBuilder.get_feature_index() returns column index for a name."""
        config = self._make_config()
        builder = DesignMatrixBuilder(config)
        data = self._make_data()
        builder.fit(data)
        result = builder.get_feature_index('gpa')
        assert isinstance(result, int)
        assert result == 1  # intercept=0, gpa=1

    def test_extract_feature(self):
        """DesignMatrixBuilder._extract_feature() extracts a single feature array."""
        config = self._make_config()
        builder = DesignMatrixBuilder(config)
        data = self._make_data()
        result = builder._extract_feature(data, 'gpa')
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)

    def test_handle_missing(self):
        """DesignMatrixBuilder._handle_missing() replaces missing values."""
        config = self._make_config()
        builder = DesignMatrixBuilder(config)
        values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        spec = FeatureSpec(name='gpa', dtype='numeric', missing_strategy='mean')
        result = builder._handle_missing(values, spec)
        assert isinstance(result, np.ndarray)
        assert not np.any(np.isnan(result))

    def test_sparsity_ratio(self):
        """DesignMatrixBuilder.sparsity_ratio() returns fraction of zeros."""
        config = self._make_config()
        builder = DesignMatrixBuilder(config)
        X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = builder.sparsity_ratio(X)
        assert isinstance(result, float)
        assert abs(result - 4.0/6.0) < 1e-10

    def test_to_sparse(self):
        """DesignMatrixBuilder.to_sparse() converts dense matrix to sparse format."""
        config = self._make_config()
        builder = DesignMatrixBuilder(config)
        X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        try:
            result = builder.to_sparse(X)
            # Result should be a sparse matrix
            assert hasattr(result, 'toarray') or hasattr(result, 'todense')
        except ImportError:
            pytest.skip("scipy not installed")


# =============================================================================
#                    BASEFEATURETRANSFORMER fit_transform COVERAGE
# =============================================================================

class TestBaseFeatureTransformerFitTransform:
    """Cover BaseFeatureTransformer.fit_transform() via a concrete subclass."""

    def test_fit_transform_via_numeric_scaler(self):
        """BaseFeatureTransformer.fit_transform() is called via unbound base method."""
        scaler = NumericScaler(method='standard', feature_name='gpa')
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Call the BASE class method directly
        result = BaseFeatureTransformer.fit_transform(scaler, data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 1)
        # Should be z-scored with mean ~0
        assert abs(np.mean(result)) < 1e-10


# =============================================================================
#                    NUMERICSCALER COVERAGE TESTS
# =============================================================================

class TestNumericScalerCoverage:
    """Additional coverage for NumericScaler methods."""

    def test_transform(self):
        """NumericScaler.transform() applies learned scaling parameters."""
        scaler = NumericScaler(method='standard', feature_name='gpa')
        X_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scaler.fit(X_train)
        result = scaler.transform(np.array([2.0, 4.0]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 1)

    def test_fit_transform(self):
        """NumericScaler.fit_transform() fits and transforms in one call."""
        scaler = NumericScaler(method='standard', feature_name='gpa')
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scaler.fit_transform(data)
        assert isinstance(result, np.ndarray)
        assert abs(np.mean(result)) < 1e-10

    def test_get_feature_names(self):
        """NumericScaler.get_feature_names() returns the feature name list."""
        scaler = NumericScaler(method='standard', feature_name='gpa')
        result = scaler.get_feature_names()
        assert isinstance(result, list)
        assert 'gpa' in result

    def test_inverse_transform(self):
        """NumericScaler.inverse_transform() converts scaled values back."""
        scaler = NumericScaler(method='standard', feature_name='gpa')
        X = np.array([1.0, 2.0, 3.0])
        scaler.fit(X)
        scaled = scaler.transform(X)
        recovered = scaler.inverse_transform(scaled)
        np.testing.assert_allclose(recovered, X, atol=1e-10)

    def test_transform_minmax(self):
        """NumericScaler.transform() works with minmax method."""
        scaler = NumericScaler(method='minmax', feature_name='score')
        X = np.array([10.0, 20.0, 30.0])
        scaler.fit(X)
        result = scaler.transform(np.array([15.0, 25.0]))
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result.ravel(), [0.25, 0.75])


# =============================================================================
#                    ONEHOTENCODER COVERAGE TESTS
# =============================================================================

class TestOneHotEncoderCoverage:
    """Additional coverage for OneHotEncoder methods."""

    def test_transform(self):
        """OneHotEncoder.transform() converts categories to one-hot vectors."""
        encoder = OneHotEncoder(feature_name='university')
        encoder.fit(np.array(['UofT', 'UBC', 'McGill']))
        result = encoder.transform(np.array(['UBC', 'UofT']))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)

    def test_fit_transform(self):
        """OneHotEncoder.fit_transform() fits and encodes in one call."""
        encoder = OneHotEncoder(feature_name='university')
        data = np.array(['UofT', 'UBC', 'McGill', 'UofT'])
        result = encoder.fit_transform(data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 3)

    def test_get_feature_names(self):
        """OneHotEncoder.get_feature_names() returns category-based names."""
        encoder = OneHotEncoder(feature_name='university')
        encoder.fit(np.array(['UBC', 'UofT']))
        result = encoder.get_feature_names()
        assert isinstance(result, list)
        assert len(result) == 2
        assert result == ['university_UBC', 'university_UofT']


# =============================================================================
#                    DUMMYENCODER COVERAGE TESTS
# =============================================================================

class TestDummyEncoderCoverage:
    """Additional coverage for DummyEncoder methods."""

    def test_fit(self):
        """DummyEncoder.fit() learns categories and determines reference."""
        encoder = DummyEncoder(feature_name='program')
        data = np.array(['CS', 'Math', 'Physics', 'CS'])
        result = encoder.fit(data)
        assert result is encoder
        assert encoder.dropped_category_ == 'CS'

    def test_transform(self):
        """DummyEncoder.transform() converts to k-1 dummy encoding."""
        encoder = DummyEncoder(feature_name='program')
        encoder.fit(np.array(['CS', 'Math', 'Physics']))
        result = encoder.transform(np.array(['Math', 'CS']))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        # Math should have a 1 in its column, CS all zeros
        np.testing.assert_array_equal(result[0], [1.0, 0.0])  # Math
        np.testing.assert_array_equal(result[1], [0.0, 0.0])  # CS (dropped)

    def test_fit_transform(self):
        """DummyEncoder.fit_transform() fits and transforms in one call."""
        encoder = DummyEncoder(feature_name='program')
        data = np.array(['CS', 'Math', 'Physics', 'CS'])
        result = encoder.fit_transform(data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 2)

    def test_get_feature_names(self):
        """DummyEncoder.get_feature_names() returns names excluding dropped category."""
        encoder = DummyEncoder(feature_name='program')
        encoder.fit(np.array(['CS', 'Math', 'Physics']))
        result = encoder.get_feature_names()
        assert isinstance(result, list)
        assert 'program_CS' not in result
        assert 'program_Math' in result
        assert 'program_Physics' in result


# =============================================================================
#                    VALIDATION FUNCTIONS TESTS
# =============================================================================

class TestValidationFunctions:
    """Tests for validate_design_matrix, check_column_rank, and related utilities."""

    def test_validate_design_matrix_valid(self):
        """validate_design_matrix returns is_valid=True for a well-formed matrix."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = validate_design_matrix(X, ['f1', 'f2'])
        assert isinstance(result, dict)
        assert result['is_valid'] is True
        assert result['has_nan'] is False
        assert result['has_inf'] is False

    def test_validate_design_matrix_with_nan(self):
        """validate_design_matrix detects NaN values."""
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        result = validate_design_matrix(X, ['f1', 'f2'])
        assert result['has_nan'] is True
        assert result['is_valid'] is False
        assert len(result['nan_locations']) > 0

    def test_validate_design_matrix_with_inf(self):
        """validate_design_matrix detects Inf values."""
        X = np.array([[1.0, np.inf], [3.0, 4.0]])
        result = validate_design_matrix(X, ['f1', 'f2'])
        assert result['has_inf'] is True
        assert result['is_valid'] is False

    def test_validate_design_matrix_constant_column(self):
        """validate_design_matrix detects constant columns."""
        X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        result = validate_design_matrix(X, ['f1', 'f2'])
        assert 'f2' in result['constant_columns']

    def test_validate_design_matrix_rank(self):
        """validate_design_matrix computes rank and condition number."""
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        result = validate_design_matrix(X, ['f1', 'f2'])
        assert result['rank'] == 2
        assert result['condition_number'] > 0

    def test_validate_design_matrix_extreme_values(self):
        """validate_design_matrix detects extreme values beyond 3 std devs."""
        # Need enough normal values so std > 0, with one outlier beyond 3 stds
        X = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],
                       [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],
                       [1.0, 1.0], [1.0, 100.0]])
        result = validate_design_matrix(X, ['f1', 'f2'])
        assert 'f2' in result['extreme_values']

    def test_check_column_rank_full_rank(self):
        """check_column_rank returns (rank, True) for full-rank matrix."""
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        rank, is_full_rank = check_column_rank(X)
        assert rank == 2
        assert is_full_rank is True

    def test_check_column_rank_deficient(self):
        """check_column_rank returns (rank, False) for rank-deficient matrix."""
        # Second column is 2x the first
        X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        rank, is_full_rank = check_column_rank(X)
        assert rank == 1
        assert is_full_rank is False

    def test_check_column_rank_identity(self):
        """check_column_rank on identity matrix returns (n, True)."""
        X = np.eye(5)
        rank, is_full_rank = check_column_rank(X)
        assert rank == 5
        assert is_full_rank is True


# =============================================================================
#                    CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctionStubs:
    """Tests for convenience functions."""

    def test_create_polynomial_features_degree_2(self):
        """create_polynomial_features with degree=2 produces correct expansion."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result, names = create_polynomial_features(X, degree=2)
        assert isinstance(result, np.ndarray)
        assert isinstance(names, list)
        # Should have: 1, x0, x1, x0^2, x1^2, x0*x1 = 6 columns
        assert result.shape == (2, 6)
        assert '1' in names
        assert 'x0' in names
        assert 'x1' in names
        assert 'x0^2' in names
        assert 'x1^2' in names
        assert 'x0*x1' in names

    def test_create_polynomial_features_values(self):
        """create_polynomial_features produces correct numeric values."""
        X = np.array([[2.0, 3.0]])
        result, names = create_polynomial_features(X, degree=2)
        # 1, x0=2, x1=3, x0^2=4, x1^2=9, x0*x1=6
        expected = [1.0, 2.0, 3.0, 4.0, 9.0, 6.0]
        np.testing.assert_allclose(result[0], expected)

    def test_create_polynomial_features_interaction_only(self):
        """create_polynomial_features with interaction_only=True skips pure powers."""
        X = np.array([[2.0, 3.0]])
        result, names = create_polynomial_features(X, degree=2, interaction_only=True)
        # Should have: 1, x0, x1, x0*x1 = 4 columns (no x0^2 or x1^2)
        assert result.shape == (1, 4)
        assert 'x0^2' not in names
        assert 'x0*x1' in names

    def test_create_polynomial_features_1d(self):
        """create_polynomial_features handles 1D input by reshaping."""
        X = np.array([1.0, 2.0, 3.0])
        result, names = create_polynomial_features(X, degree=2)
        assert result.shape[0] == 3
        # 1 feature: 1, x0, x0^2 = 3 columns (no interaction with only 1 feature)
        assert result.shape == (3, 3)

    def test_build_admission_matrix(self):
        """build_admission_matrix produces design matrix with intercept."""
        apps = [
            {'gpa': 3.5, 'university': 'UofT'},
            {'gpa': 3.0, 'university': 'UBC'},
            {'gpa': 3.8, 'university': 'UofT'},
        ]
        X, feature_names = build_admission_matrix(apps)
        assert isinstance(X, np.ndarray)
        assert isinstance(feature_names, list)
        assert X.shape[0] == 3
        assert len(feature_names) == X.shape[1]
        assert 'intercept' in feature_names

    def test_build_admission_matrix_gpa_only(self):
        """build_admission_matrix works with just GPA."""
        apps = [{'gpa': 3.5}, {'gpa': 3.0}]
        X, feature_names = build_admission_matrix(apps)
        assert X.shape[0] == 2
        assert 'gpa' in feature_names

    def test_identify_collinear_features(self):
        """identify_collinear_features finds highly correlated pairs."""
        # Create perfectly correlated columns
        X = np.array([[1.0, 2.0, 10.0],
                       [2.0, 4.0, 20.0],
                       [3.0, 6.0, 30.0]])
        result = identify_collinear_features(X, ['f1', 'f2', 'f3'], threshold=0.95)
        assert isinstance(result, list)
        # All three columns are perfectly correlated
        assert len(result) >= 1
        # Check structure of returned tuples
        for name1, name2, corr in result:
            assert isinstance(name1, str)
            assert isinstance(name2, str)
            assert isinstance(corr, float)
            assert abs(corr) > 0.95

    def test_identify_collinear_features_uncorrelated(self):
        """identify_collinear_features returns empty for orthogonal columns."""
        X = np.eye(3)
        result = identify_collinear_features(X, ['f1', 'f2', 'f3'], threshold=0.95)
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
