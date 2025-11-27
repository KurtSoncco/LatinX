"""Tests for Bayesian Last Layer metrics."""

import jax.numpy as jnp
import numpy as np

from latinx.metrics.metrics_bayesian_last_layer import (
    compute_all_metrics,
    mae,
    observation_noise_std,
    observation_precision,
    posterior_condition_number,
    posterior_uncertainty,
    prediction_uncertainty,
    rmse,
    weight_norm,
)
from latinx.models.bayesian_last_layer import BayesianLastLayer


def test_bayesian_last_layer_metrics():
    """Test all Bayesian Last Layer metrics on a simple dataset."""
    # Create simple synthetic data
    np.random.seed(42)
    n_samples = 50
    x_train = np.linspace(0, 2 * np.pi, n_samples)[:, None]
    y_train = np.sin(x_train).squeeze() + np.random.normal(0, 0.1, n_samples)

    x_train = jnp.array(x_train)
    y_train = jnp.array(y_train)

    # Fit model
    bll = BayesianLastLayer(
        hidden_dims=(10, 10),
        sigma=0.1,
        alpha=0.05,
        learning_rate=1e-3,
        n_steps=1000,
        seed=42,
    )
    bll.fit(x_train, y_train)

    # Test individual metrics
    post_unc = posterior_uncertainty(bll)
    assert post_unc > 0, "Posterior uncertainty should be positive"

    pred_unc = prediction_uncertainty(bll, x_train)
    assert pred_unc > 0, "Prediction uncertainty should be positive"

    w_norm = weight_norm(bll)
    assert w_norm >= 0, "Weight norm should be non-negative"

    cond_num = posterior_condition_number(bll)
    assert cond_num > 0, "Condition number should be positive"
    assert cond_num < 1e10, "Condition number should not be too large (numerical issues)"

    obs_prec = observation_precision(bll)
    assert obs_prec > 0, "Observation precision should be positive"
    assert jnp.isclose(obs_prec, 1 / 0.1**2, atol=1e-6), "Should match 1/sigma^2"

    obs_std = observation_noise_std(bll)
    assert jnp.isclose(obs_std, 0.1, atol=1e-6), "Should match sigma parameter"

    # Test prediction metrics
    y_pred = bll.predict(x_train, return_std=False)
    rmse_val = rmse(y_train, y_pred)
    assert rmse_val >= 0, "RMSE should be non-negative"

    mae_val = mae(y_train, y_pred)
    assert mae_val >= 0, "MAE should be non-negative"

    # Test compute_all_metrics
    all_metrics = compute_all_metrics(bll, x_train, y_train, y_pred)
    assert isinstance(all_metrics, dict), "Should return a dictionary"
    assert len(all_metrics) == 11, "Should return 11 metrics"
    assert "posterior_uncertainty" in all_metrics
    assert "prediction_uncertainty" in all_metrics
    assert "rmse" in all_metrics
    assert "mae" in all_metrics
    assert "log_likelihood" in all_metrics

    # Check all values are finite
    for key, value in all_metrics.items():
        assert jnp.isfinite(value), f"Metric {key} should be finite, got {value}"


def test_unfitted_model_metrics():
    """Test that metrics handle unfitted models gracefully."""
    bll = BayesianLastLayer()

    # These should return NaN for unfitted model
    assert jnp.isnan(posterior_uncertainty(bll))
    assert jnp.isnan(weight_norm(bll))
    assert jnp.isnan(posterior_condition_number(bll))

    # These should still work (constructor parameters)
    assert observation_noise_std(bll) == 0.3  # Default sigma
