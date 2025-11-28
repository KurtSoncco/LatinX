"""Tests for standalone Bayesian Last Layer."""

import jax.numpy as jnp
import numpy as np
import pytest

from latinx.models.standalone_bayesian_last_layer import StandaloneBayesianLastLayer


@pytest.fixture
def simple_data():
    """Generate simple synthetic data for testing."""
    np.random.seed(42)
    n_samples = 100
    feature_dim = 5

    features = np.random.randn(n_samples, feature_dim).astype(np.float32)
    # True weights
    true_weights = np.array([1.0, -0.5, 0.3, -0.2, 0.8])
    # y = X @ w + noise
    y = features @ true_weights + 0.1 * np.random.randn(n_samples)

    return features, y, feature_dim


def test_initialization():
    """Test basic initialization."""
    bll = StandaloneBayesianLastLayer(sigma=0.1, alpha=0.01, feature_dim=10)

    assert bll.sigma == 0.1
    assert bll.alpha == 0.01
    assert bll.feature_dim == 10
    assert bll.beta == 1.0 / (0.1**2)
    assert not bll._is_fitted


def test_fit_with_numpy(simple_data):
    """Test fitting with NumPy arrays."""
    features, y, feature_dim = simple_data

    bll = StandaloneBayesianLastLayer(sigma=0.1, alpha=0.01, feature_dim=feature_dim)
    bll.fit(features, y)

    assert bll._is_fitted
    assert bll.posterior_mean is not None
    assert bll.posterior_precision is not None
    assert bll.posterior_covariance is not None
    assert bll.posterior_mean.shape == (feature_dim,)


def test_fit_with_jax(simple_data):
    """Test fitting with JAX arrays."""
    features, y, feature_dim = simple_data

    # Convert to JAX arrays
    features_jax = jnp.array(features)
    y_jax = jnp.array(y)

    bll = StandaloneBayesianLastLayer(sigma=0.1, alpha=0.01, feature_dim=feature_dim)
    bll.fit(features_jax, y_jax)

    assert bll._is_fitted
    assert bll.posterior_mean.shape == (feature_dim,)


def test_fit_infers_feature_dim(simple_data):
    """Test that feature_dim is inferred if not provided."""
    features, y, feature_dim = simple_data

    bll = StandaloneBayesianLastLayer(sigma=0.1, alpha=0.01)
    assert bll.feature_dim is None

    bll.fit(features, y)
    assert bll.feature_dim == feature_dim


def test_predict_before_fit_raises_error(simple_data):
    """Test that predict raises error if called before fit."""
    features, _, feature_dim = simple_data

    bll = StandaloneBayesianLastLayer(sigma=0.1, alpha=0.01, feature_dim=feature_dim)

    with pytest.raises(RuntimeError, match="must be fitted"):
        bll.predict(features)


def test_predict_returns_correct_shape(simple_data):
    """Test that predict returns correct shape."""
    features, y, feature_dim = simple_data

    bll = StandaloneBayesianLastLayer(sigma=0.1, alpha=0.01, feature_dim=feature_dim)
    bll.fit(features, y)

    # Predict without std
    y_pred = bll.predict(features)
    assert y_pred.shape == (len(features),)

    # Predict with std
    y_pred, y_std = bll.predict(features, return_std=True)
    assert y_pred.shape == (len(features),)
    assert y_std.shape == (len(features),)


def test_predict_uncertainty_is_positive(simple_data):
    """Test that uncertainty estimates are positive."""
    features, y, feature_dim = simple_data

    bll = StandaloneBayesianLastLayer(sigma=0.1, alpha=0.01, feature_dim=feature_dim)
    bll.fit(features, y)

    _, y_std = bll.predict(features, return_std=True)
    assert jnp.all(y_std > 0)


def test_weight_statistics(simple_data):
    """Test get_weight_statistics method."""
    features, y, feature_dim = simple_data

    bll = StandaloneBayesianLastLayer(sigma=0.1, alpha=0.01, feature_dim=feature_dim)
    bll.fit(features, y)

    w_mean, w_std = bll.get_weight_statistics()

    assert w_mean.shape == (feature_dim,)
    assert w_std.shape == (feature_dim,)
    assert jnp.all(w_std > 0)


def test_total_uncertainty(simple_data):
    """Test get_total_uncertainty method."""
    features, y, feature_dim = simple_data

    bll = StandaloneBayesianLastLayer(sigma=0.1, alpha=0.01, feature_dim=feature_dim)
    bll.fit(features, y)

    total_unc = bll.get_total_uncertainty()

    assert isinstance(total_unc, float)
    assert total_unc > 0


def test_higher_alpha_increases_regularization(simple_data):
    """Test that higher alpha (prior precision) increases regularization."""
    features, y, feature_dim = simple_data

    bll_low_alpha = StandaloneBayesianLastLayer(sigma=0.1, alpha=0.001, feature_dim=feature_dim)
    bll_high_alpha = StandaloneBayesianLastLayer(sigma=0.1, alpha=10.0, feature_dim=feature_dim)

    bll_low_alpha.fit(features, y)
    bll_high_alpha.fit(features, y)

    # Higher alpha should shrink weights toward zero
    w_mean_low, _ = bll_low_alpha.get_weight_statistics()
    w_mean_high, _ = bll_high_alpha.get_weight_statistics()

    assert float(jnp.mean(jnp.abs(w_mean_low))) > float(jnp.mean(jnp.abs(w_mean_high)))


def test_lower_sigma_increases_confidence(simple_data):
    """Test that lower sigma (observation noise) increases confidence."""
    features, y, feature_dim = simple_data

    bll_high_sigma = StandaloneBayesianLastLayer(sigma=1.0, alpha=0.01, feature_dim=feature_dim)
    bll_low_sigma = StandaloneBayesianLastLayer(sigma=0.01, alpha=0.01, feature_dim=feature_dim)

    bll_high_sigma.fit(features, y)
    bll_low_sigma.fit(features, y)

    _, std_high = bll_high_sigma.predict(features, return_std=True)
    _, std_low = bll_low_sigma.predict(features, return_std=True)

    # Lower sigma should give lower uncertainty
    assert float(jnp.mean(std_low)) < float(jnp.mean(std_high))


def test_update_hyperparameters(simple_data):
    """Test hyperparameter update."""
    features, y, feature_dim = simple_data

    bll = StandaloneBayesianLastLayer(sigma=0.1, alpha=0.01, feature_dim=feature_dim)
    bll.fit(features, y)

    assert bll._is_fitted

    # Update hyperparameters
    bll.update_hyperparameters(sigma=0.2, alpha=0.05)

    assert bll.sigma == 0.2
    assert bll.alpha == 0.05
    assert bll.beta == 1.0 / (0.2**2)
    assert not bll._is_fitted  # Should reset fitted state


def test_dimension_mismatch_raises_error(simple_data):
    """Test that dimension mismatch raises error."""
    features, y, _ = simple_data

    # Fit with correct dimensions
    bll = StandaloneBayesianLastLayer(sigma=0.1, alpha=0.01, feature_dim=5)
    bll.fit(features, y)

    # Try to predict with wrong dimensions
    features_wrong = np.random.randn(10, 3)

    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        bll.predict(features_wrong)


def test_y_shape_handling(simple_data):
    """Test that y can be 1D or 2D."""
    features, y, feature_dim = simple_data

    # Test with 1D y
    bll_1d = StandaloneBayesianLastLayer(sigma=0.1, alpha=0.01, feature_dim=feature_dim)
    bll_1d.fit(features, y)

    # Test with 2D y
    y_2d = y.reshape(-1, 1)
    bll_2d = StandaloneBayesianLastLayer(sigma=0.1, alpha=0.01, feature_dim=feature_dim)
    bll_2d.fit(features, y_2d)

    # Both should give same results
    pred_1d = bll_1d.predict(features)
    pred_2d = bll_2d.predict(features)

    assert jnp.allclose(pred_1d, pred_2d, rtol=1e-5)


def test_repr(simple_data):
    """Test string representation."""
    features, y, feature_dim = simple_data

    bll = StandaloneBayesianLastLayer(sigma=0.1, alpha=0.01, feature_dim=feature_dim)
    repr_before = repr(bll)
    assert "not fitted" in repr_before

    bll.fit(features, y)
    repr_after = repr(bll)
    assert "fitted" in repr_after
    assert "sigma=0.1000" in repr_after
    assert "alpha=0.0100" in repr_after


def test_method_chaining(simple_data):
    """Test that fit returns self for method chaining."""
    features, y, feature_dim = simple_data

    bll = StandaloneBayesianLastLayer(sigma=0.1, alpha=0.01, feature_dim=feature_dim)
    result = bll.fit(features, y)

    assert result is bll
    assert bll._is_fitted
