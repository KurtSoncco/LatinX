"""Tests for Bayesian Last Layer model."""

import jax.numpy as jnp
import numpy as np
import pytest

from latinx.models.bayesian_last_layer import BayesianLastLayer


@pytest.fixture
def simple_data():
    """Generate simple 1D synthetic data for testing."""
    np.random.seed(42)
    n_samples = 50
    x = np.linspace(0, 2 * np.pi, n_samples)[:, None]
    y = np.sin(x).squeeze() + np.random.normal(0, 0.1, n_samples)
    return jnp.array(x), jnp.array(y)


def test_initialization():
    """Test BayesianLastLayer initialization with default parameters."""
    bll = BayesianLastLayer()

    assert bll.hidden_dims == (10, 10)
    assert bll.sigma == 0.3
    assert bll.alpha == 0.05
    assert bll.learning_rate == 1e-3
    assert bll.n_steps == 3000
    assert bll.seed == 31415

    # Should be None before fitting
    assert bll.params_full is None
    assert bll.params_hidden is None
    assert bll.posterior_mean is None
    assert bll.posterior_precision is None
    assert bll.beta is None


def test_initialization_custom_params():
    """Test BayesianLastLayer initialization with custom parameters."""
    bll = BayesianLastLayer(
        hidden_dims=(20, 30),
        sigma=0.5,
        alpha=0.1,
        learning_rate=1e-2,
        n_steps=1000,
        seed=123,
    )

    assert bll.hidden_dims == (20, 30)
    assert bll.sigma == 0.5
    assert bll.alpha == 0.1
    assert bll.learning_rate == 1e-2
    assert bll.n_steps == 1000
    assert bll.seed == 123


def test_fit(simple_data):
    """Test fitting the BayesianLastLayer model."""
    x, y = simple_data

    bll = BayesianLastLayer(
        hidden_dims=(10, 10),
        sigma=0.1,
        alpha=0.05,
        learning_rate=1e-3,
        n_steps=1000,
        seed=42,
    )

    loss_history = bll.fit(x, y)

    # Check loss history shape
    assert loss_history.shape == (1000,)

    # Loss should generally decrease
    assert loss_history[-1] < loss_history[0]

    # Check that parameters are initialized after fitting
    assert bll.params_full is not None
    assert bll.params_hidden is not None
    assert bll.posterior_mean is not None
    assert bll.posterior_precision is not None
    assert bll.beta is not None

    # Check posterior parameters have correct shapes
    # For hidden_dims=(10,10), last hidden layer output is 10
    assert bll.posterior_mean.shape == (10,)
    assert bll.posterior_precision.shape == (10, 10)

    # Check beta matches sigma
    expected_beta = 1 / (0.1**2)
    assert jnp.isclose(bll.beta, expected_beta)


def test_fit_with_1d_target(simple_data):
    """Test fitting works with 1D target array."""
    x, y = simple_data

    bll = BayesianLastLayer(n_steps=100, seed=42)

    # y is already 1D
    assert y.ndim == 1

    # Should work without error
    loss_history = bll.fit(x, y)
    assert loss_history is not None


def test_fit_with_2d_target(simple_data):
    """Test fitting works with 2D target array."""
    x, y = simple_data

    bll = BayesianLastLayer(n_steps=100, seed=42)

    # Reshape y to 2D
    y_2d = y[:, None]
    assert y_2d.ndim == 2

    # Should work without error
    loss_history = bll.fit(x, y_2d)
    assert loss_history is not None


def test_predict_before_fit():
    """Test that prediction raises error before fitting."""
    bll = BayesianLastLayer()
    x_test = jnp.array([[1.0], [2.0]])

    with pytest.raises(RuntimeError, match="Model must be fitted before making predictions"):
        bll.predict(x_test)


def test_predict(simple_data):
    """Test prediction after fitting."""
    x_train, y_train = simple_data

    bll = BayesianLastLayer(n_steps=500, seed=42)
    bll.fit(x_train, y_train)

    # Test prediction
    x_test = jnp.linspace(0, 2 * jnp.pi, 20)[:, None]
    y_pred = bll.predict(x_test)

    # Check shape
    assert y_pred.shape == (20,)

    # Predictions should be finite
    assert jnp.all(jnp.isfinite(y_pred))


def test_predict_with_uncertainty(simple_data):
    """Test prediction with uncertainty estimation."""
    x_train, y_train = simple_data

    bll = BayesianLastLayer(n_steps=500, seed=42)
    bll.fit(x_train, y_train)

    x_test = jnp.linspace(0, 2 * jnp.pi, 20)[:, None]
    y_pred, y_std = bll.predict(x_test, return_std=True)

    # Check shapes
    assert y_pred.shape == (20,)
    assert y_std.shape == (20,)

    # Standard deviations should be positive
    assert jnp.all(y_std > 0)

    # Standard deviations should be finite
    assert jnp.all(jnp.isfinite(y_std))


def test_predict_full_nn_before_fit():
    """Test that full NN prediction raises error before fitting."""
    bll = BayesianLastLayer()
    x_test = jnp.array([[1.0], [2.0]])

    with pytest.raises(RuntimeError, match="Model must be fitted before making predictions"):
        bll.predict_full_nn(x_test)


def test_predict_full_nn(simple_data):
    """Test prediction using the full original neural network."""
    x_train, y_train = simple_data

    bll = BayesianLastLayer(n_steps=500, seed=42)
    bll.fit(x_train, y_train)

    x_test = jnp.linspace(0, 2 * jnp.pi, 20)[:, None]
    y_pred_nn = bll.predict_full_nn(x_test)

    # Check shape
    assert y_pred_nn.shape == (20,)

    # Predictions should be finite
    assert jnp.all(jnp.isfinite(y_pred_nn))


def test_bayesian_vs_nn_predictions(simple_data):
    """Test that Bayesian predictions differ from original NN predictions."""
    x_train, y_train = simple_data

    bll = BayesianLastLayer(n_steps=1000, seed=42, alpha=0.05)
    bll.fit(x_train, y_train)

    x_test = jnp.linspace(0, 2 * jnp.pi, 20)[:, None]

    y_pred_bayesian = bll.predict(x_test, return_std=False)
    y_pred_nn = bll.predict_full_nn(x_test)

    # They should be different (Bayesian last layer modifies predictions)
    # Allow some to be similar but not all
    differences = jnp.abs(y_pred_bayesian - y_pred_nn)
    assert jnp.mean(differences) > 0.01  # Should have some meaningful difference


def test_uncertainty_increases_with_extrapolation(simple_data):
    """Test that uncertainty increases when extrapolating beyond training data."""
    x_train, y_train = simple_data

    bll = BayesianLastLayer(n_steps=1000, seed=42)
    bll.fit(x_train, y_train)

    # Training range: [0, 2*pi]
    # Test in-range and out-of-range
    x_in_range = jnp.linspace(0.5, 2 * jnp.pi - 0.5, 10)[:, None]
    x_out_range = jnp.linspace(-2, -0.5, 10)[:, None]  # Before training data

    _, std_in = bll.predict(x_in_range, return_std=True)
    _, std_out = bll.predict(x_out_range, return_std=True)

    # Uncertainty should generally be higher outside training range
    assert jnp.mean(std_out) > jnp.mean(std_in)


def test_different_hidden_dims():
    """Test that different hidden dimensions work correctly."""
    np.random.seed(42)
    x = jnp.array(np.random.randn(30, 2))  # 2D input
    y = jnp.array(np.random.randn(30))

    # Test with different architectures
    for hidden_dims in [(5,), (10, 5), (20, 15, 10)]:
        bll = BayesianLastLayer(hidden_dims=hidden_dims, n_steps=100, seed=42)
        loss_history = bll.fit(x, y)

        # Should fit without error
        assert loss_history is not None

        # Posterior mean should match last hidden dimension
        assert bll.posterior_mean.shape == (hidden_dims[-1],)


def test_reproducibility():
    """Test that results are reproducible with the same seed."""
    np.random.seed(123)
    x = jnp.array(np.random.randn(30, 1))
    y = jnp.array(np.random.randn(30))

    bll1 = BayesianLastLayer(n_steps=500, seed=999)
    loss1 = bll1.fit(x, y)

    bll2 = BayesianLastLayer(n_steps=500, seed=999)
    loss2 = bll2.fit(x, y)

    # Losses should be identical
    assert jnp.allclose(loss1, loss2)

    # Predictions should be identical
    x_test = jnp.array([[0.5], [1.0], [1.5]])
    pred1 = bll1.predict(x_test)
    pred2 = bll2.predict(x_test)

    assert jnp.allclose(pred1, pred2)
