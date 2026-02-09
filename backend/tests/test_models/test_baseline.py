"""
Unit Tests for Baseline Model Module
=====================================

Tests for src/models/baseline.py — BetaPrior, ProgramPosterior,
BaselineModel, and compute_shrinkage_factor.
"""

import pytest
import numpy as np

from src.models.baseline import (
    BetaPrior,
    ProgramPosterior,
    BaselineModel,
    compute_shrinkage_factor,
)


# =============================================================================
#                    BETAPRIOR TESTS
# =============================================================================

class TestBetaPrior:
    """Tests for BetaPrior dataclass."""

    def test_default_values(self):
        """Default prior is Beta(1, 1) — uniform."""
        prior = BetaPrior()
        assert prior.alpha == 1.0
        assert prior.beta == 1.0

    def test_mean(self):
        """Mean = alpha / (alpha + beta)."""
        prior = BetaPrior(alpha=3.0, beta=7.0)
        assert np.isclose(prior.mean, 0.3)

    def test_variance(self):
        """Variance = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))."""
        prior = BetaPrior(alpha=2.0, beta=2.0)
        expected = (2 * 2) / ((4) ** 2 * 5)  # = 4/80 = 0.05
        assert np.isclose(prior.variance, expected)

    def test_strength(self):
        """Strength = alpha + beta."""
        prior = BetaPrior(alpha=5.0, beta=10.0)
        assert np.isclose(prior.strength, 15.0)

    def test_from_mean_strength(self):
        """Factory method creates prior with correct mean and strength."""
        prior = BetaPrior.from_mean_strength(mean=0.4, strength=10.0)
        assert np.isclose(prior.alpha, 4.0)
        assert np.isclose(prior.beta, 6.0)
        assert np.isclose(prior.mean, 0.4)
        assert np.isclose(prior.strength, 10.0)

    def test_uniform_prior(self):
        """Beta(1,1) has mean 0.5."""
        prior = BetaPrior(alpha=1.0, beta=1.0)
        assert np.isclose(prior.mean, 0.5)


# =============================================================================
#                    PROGRAMPOSTERIOR TESTS
# =============================================================================

class TestProgramPosterior:
    """Tests for ProgramPosterior dataclass."""

    @pytest.fixture
    def sample_posterior(self):
        """A sample posterior for testing."""
        return ProgramPosterior(
            program_id="uoft_cs",
            university="University of Toronto",
            program_name="Computer Science",
            alpha_post=15.0,
            beta_post=35.0,
            n_observations=40,
            n_admits=10,
        )

    def test_posterior_mean(self, sample_posterior):
        """Posterior mean = alpha_post / (alpha_post + beta_post)."""
        expected = 15.0 / (15.0 + 35.0)
        assert np.isclose(sample_posterior.posterior_mean, expected)

    def test_posterior_mode(self, sample_posterior):
        """Posterior mode = (alpha-1) / (alpha+beta-2) for alpha,beta > 1."""
        expected = (15.0 - 1) / (15.0 + 35.0 - 2)
        assert np.isclose(sample_posterior.posterior_mode, expected)

    def test_posterior_variance(self, sample_posterior):
        """Posterior variance follows Beta formula."""
        a, b = 15.0, 35.0
        expected = (a * b) / ((a + b) ** 2 * (a + b + 1))
        assert np.isclose(sample_posterior.posterior_variance, expected)

    def test_credible_interval_contains_mean(self, sample_posterior):
        """80% credible interval should contain the posterior mean."""
        lower, upper = sample_posterior.credible_interval(0.80)
        mean = sample_posterior.posterior_mean
        assert lower <= mean <= upper

    def test_credible_interval_wider_with_higher_level(self, sample_posterior):
        """95% CI should be wider than 80% CI."""
        lo_80, hi_80 = sample_posterior.credible_interval(0.80)
        lo_95, hi_95 = sample_posterior.credible_interval(0.95)
        assert (hi_95 - lo_95) >= (hi_80 - lo_80)

    def test_credible_interval_in_01(self, sample_posterior):
        """Credible interval bounds should be in [0, 1]."""
        lower, upper = sample_posterior.credible_interval(0.80)
        assert 0.0 <= lower <= upper <= 1.0


# =============================================================================
#                    BASELINEMODEL TESTS
# =============================================================================

class TestBaselineModel:
    """Tests for BaselineModel (Beta-Binomial baseline)."""

    @pytest.fixture
    def simple_data(self):
        """Simple data with 3 programs."""
        n = 30
        X = np.random.default_rng(42).standard_normal((n, 2))
        y = np.array([1, 0, 1, 0, 0, 1, 1, 1, 0, 0,  # prog A: 5/10
                       1, 1, 1, 1, 1, 1, 1, 0, 0, 0,  # prog B: 7/10
                       0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).astype(float)  # prog C: 1/10
        programs = (["prog_A"] * 10 + ["prog_B"] * 10 + ["prog_C"] * 10)
        program_ids = np.array(programs)
        return X, y, program_ids

    def test_fit_sets_fitted(self, simple_data):
        """Model is fitted after fit()."""
        X, y, prog_ids = simple_data
        model = BaselineModel(prior_strength=10.0)
        model.fit(X, y, prog_ids)
        assert model.is_fitted

    def test_posteriors_created(self, simple_data):
        """Posteriors are created for each program."""
        X, y, prog_ids = simple_data
        model = BaselineModel(prior_strength=10.0)
        model.fit(X, y, prog_ids)
        assert "prog_A" in model.posteriors
        assert "prog_B" in model.posteriors
        assert "prog_C" in model.posteriors

    def test_predictions_in_01(self, simple_data):
        """Predictions are in [0, 1]."""
        X, y, prog_ids = simple_data
        model = BaselineModel(prior_strength=10.0)
        model.fit(X, y, prog_ids)
        probs = model.predict_proba(X, prog_ids)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_prediction_length(self, simple_data):
        """Predictions have same length as input."""
        X, y, prog_ids = simple_data
        model = BaselineModel(prior_strength=10.0)
        model.fit(X, y, prog_ids)
        probs = model.predict_proba(X, prog_ids)
        assert len(probs) == len(y)

    def test_shrinkage_toward_prior(self, simple_data):
        """Programs with few observations shrink more toward prior."""
        X, y, prog_ids = simple_data
        model = BaselineModel(prior_strength=10.0)
        model.fit(X, y, prog_ids)
        # prog_C has 1/10 admits = 10% raw rate
        # Global rate = 13/30 ≈ 43%
        # Posterior should be between 10% and 43% (shrunk toward global)
        prog_c_post = model.posteriors["prog_C"]
        assert prog_c_post.posterior_mean > 0.10  # shrunk away from raw rate

    def test_model_name(self):
        """Model has a name."""
        model = BaselineModel()
        assert isinstance(model.name, str)
        assert len(model.name) > 0

    def test_n_params(self, simple_data):
        """n_params = 2 * number of programs."""
        X, y, prog_ids = simple_data
        model = BaselineModel(prior_strength=10.0)
        model.fit(X, y, prog_ids)
        assert model.n_params == 2 * 3  # 3 programs, 2 params each


# =============================================================================
#                    COMPUTE SHRINKAGE FACTOR TESTS
# =============================================================================

class TestComputeShrinkageFactor:
    """Tests for compute_shrinkage_factor."""

    def test_no_data_full_shrinkage(self):
        """With 0 observations, shrinkage = 1 (100% prior)."""
        assert np.isclose(compute_shrinkage_factor(0, 10.0), 1.0)

    def test_lots_of_data_low_shrinkage(self):
        """With lots of data, shrinkage approaches 0."""
        s = compute_shrinkage_factor(1000, 10.0)
        assert s < 0.02

    def test_equal_data_and_prior(self):
        """When n = prior_strength, shrinkage = 0.5."""
        assert np.isclose(compute_shrinkage_factor(10, 10.0), 0.5)

    def test_monotone_decreasing(self):
        """Shrinkage decreases as n increases."""
        vals = [compute_shrinkage_factor(n, 10.0) for n in [0, 5, 10, 50, 100]]
        for i in range(len(vals) - 1):
            assert vals[i] > vals[i + 1]


# =============================================================================
#                    STUB COVERAGE TESTS
# =============================================================================

class TestBaselineAdditionalMethods:
    """Tests for BaselineModel methods: predict_interval, summary, params."""

    @pytest.fixture
    def fitted_model(self):
        """A fitted model with simple data."""
        n = 30
        X = np.random.default_rng(42).standard_normal((n, 2))
        y = np.array([1, 0, 1, 0, 0, 1, 1, 1, 0, 0,
                       1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).astype(float)
        programs = np.array(["prog_A"] * 10 + ["prog_B"] * 10 + ["prog_C"] * 10)
        model = BaselineModel(prior_strength=10.0)
        model.fit(X, y, programs)
        return model, X, programs

    def test_predict_interval_unfitted_raises(self):
        """predict_interval raises RuntimeError if not fitted."""
        model = BaselineModel(prior_strength=10.0)
        with pytest.raises(RuntimeError):
            model.predict_interval(np.zeros((5, 2)), np.array(["a"] * 5), level=0.80)

    def test_predict_interval_returns_bounds(self, fitted_model):
        """predict_interval returns (lower, upper) arrays in [0, 1]."""
        model, X, programs = fitted_model
        lower, upper = model.predict_interval(X, programs, level=0.80)
        assert lower.shape == (len(programs),)
        assert upper.shape == (len(programs),)
        assert np.all(lower >= 0.0)
        assert np.all(upper <= 1.0)
        assert np.all(lower <= upper)

    def test_get_program_summary_unfitted_raises(self):
        """get_program_summary raises KeyError for unknown program."""
        model = BaselineModel(prior_strength=10.0)
        with pytest.raises(KeyError):
            model.get_program_summary("test_prog")

    def test_get_program_summary_valid(self, fitted_model):
        """get_program_summary returns expected keys for known program."""
        model, _, _ = fitted_model
        summary = model.get_program_summary("prog_A")
        assert isinstance(summary, dict)
        assert 'program_id' in summary
        assert 'posterior_mean' in summary
        assert 'observed_rate' in summary
        assert 'shrinkage' in summary
        assert 0.0 <= summary['posterior_mean'] <= 1.0

    def test_rank_programs_by_difficulty(self, fitted_model):
        """rank_programs_by_difficulty returns sorted list."""
        model, _, _ = fitted_model
        rankings = model.rank_programs_by_difficulty()
        assert isinstance(rankings, list)
        assert len(rankings) == 3
        # Sorted ascending by posterior_mean
        rates = [r[1] for r in rankings]
        for i in range(len(rates) - 1):
            assert rates[i] <= rates[i + 1]

    def test_get_params_returns_dict(self, fitted_model):
        """get_params returns a dict with expected keys."""
        model, _, _ = fitted_model
        params = model.get_params()
        assert isinstance(params, dict)
        assert 'prior_strength' in params
        assert 'posteriors' in params
        assert 'is_fitted' in params
        assert params['is_fitted'] is True

    def test_set_params_restores_state(self):
        """set_params restores model state from params dict."""
        # Get params from a fitted model
        model1 = BaselineModel(prior_strength=10.0)
        n = 10
        X = np.random.default_rng(0).standard_normal((n, 2))
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]).astype(float)
        programs = np.array(["p1"] * 5 + ["p2"] * 5)
        model1.fit(X, y, programs)
        params = model1.get_params()

        # Restore into a new model
        model2 = BaselineModel()
        model2.set_params(params)
        assert model2._is_fitted is True
        assert model2.prior_strength == 10.0
        assert "p1" in model2.posteriors
        assert "p2" in model2.posteriors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
