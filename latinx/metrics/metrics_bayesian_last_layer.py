"""
Evaluation metrics for Bayesian Last Layer implementations.

This module provides standardized metrics for evaluating Bayesian Last Layer performance,
including posterior uncertainty, prediction uncertainty, weight norms, and other diagnostic metrics.
"""

import jax.numpy as jnp
from jax import Array

from latinx.models.bayesian_last_layer import BayesianLastLayer


def posterior_uncertainty(bll: BayesianLastLayer) -> float:
    """
    Compute the trace of the posterior covariance matrix.

    The trace represents the total uncertainty/variance in the posterior parameter estimates.
    Lower values indicate more confidence in the learned parameters.

    Args:
        bll: BayesianLastLayer instance (must be fitted)

    Returns:
        Trace of the posterior covariance matrix as a float
    """
    if bll.posterior_precision is None:
        return float(jnp.nan)

    posterior_cov = jnp.linalg.inv(bll.posterior_precision)
    return float(jnp.trace(posterior_cov))


def prediction_uncertainty(bll: BayesianLastLayer, x: jnp.ndarray) -> float:
    """
    Compute the average prediction uncertainty (standard deviation) over input data.

    This represents the average epistemic uncertainty in predictions,
    which should be higher in regions with less training data.

    Args:
        bll: BayesianLastLayer instance (must be fitted)
        x: Input features of shape (n_samples, n_features)

    Returns:
        Mean prediction standard deviation as a float
    """
    if bll.posterior_mean is None:
        return float(jnp.nan)

    _, std = bll.predict(x, return_std=True)
    return float(jnp.mean(std))


def max_prediction_uncertainty(bll: BayesianLastLayer, x: jnp.ndarray) -> float:
    """
    Compute the maximum prediction uncertainty over input data.

    Useful for identifying regions of highest uncertainty.

    Args:
        bll: BayesianLastLayer instance (must be fitted)
        x: Input features of shape (n_samples, n_features)

    Returns:
        Maximum prediction standard deviation as a float
    """
    if bll.posterior_mean is None:
        return float(jnp.nan)

    _, std = bll.predict(x, return_std=True)
    return float(jnp.max(std))


def min_prediction_uncertainty(bll: BayesianLastLayer, x: jnp.ndarray) -> float:
    """
    Compute the minimum prediction uncertainty over input data.

    Useful for identifying regions of lowest uncertainty (typically near training data).

    Args:
        bll: BayesianLastLayer instance (must be fitted)
        x: Input features of shape (n_samples, n_features)

    Returns:
        Minimum prediction standard deviation as a float
    """
    if bll.posterior_mean is None:
        return float(jnp.nan)

    _, std = bll.predict(x, return_std=True)
    return float(jnp.min(std))


def weight_norm(bll: BayesianLastLayer) -> float:
    """
    Compute the L2 norm of the posterior mean weight vector.

    This metric tracks the magnitude of the learned parameters,
    useful for monitoring regularization and model complexity.

    Args:
        bll: BayesianLastLayer instance (must be fitted)

    Returns:
        L2 norm of the posterior mean as a float
    """
    if bll.posterior_mean is None:
        return float(jnp.nan)

    return float(jnp.linalg.norm(bll.posterior_mean))


def posterior_condition_number(bll: BayesianLastLayer) -> float:
    """
    Compute the condition number of the posterior precision matrix.

    High condition numbers indicate numerical instability or ill-conditioning.
    Values > 1e6 may indicate problems.

    Args:
        bll: BayesianLastLayer instance (must be fitted)

    Returns:
        Condition number as a float
    """
    if bll.posterior_precision is None:
        return float(jnp.nan)

    return float(jnp.linalg.cond(bll.posterior_precision))


def observation_precision(bll: BayesianLastLayer) -> float:
    """
    Get the observation precision (beta = 1/sigma^2).

    Higher values indicate lower assumed observation noise.

    Args:
        bll: BayesianLastLayer instance

    Returns:
        Observation precision (beta) as a float
    """
    if bll.beta is None:
        return float(jnp.nan)

    return float(bll.beta)


def observation_noise_std(bll: BayesianLastLayer) -> float:
    """
    Get the observation noise standard deviation (sigma = 1/sqrt(beta)).

    This is the assumed noise level in the observations.

    Args:
        bll: BayesianLastLayer instance

    Returns:
        Observation noise std as a float
    """
    return bll.sigma


def rmse(y_true: float | Array | list[float], y_pred: float | Array | list[float]) -> float:
    """
    Compute Root Mean Squared Error (RMSE).

    Args:
        y_true: True values (scalar, array, or list)
        y_pred: Predicted values (scalar, array, or list)

    Returns:
        RMSE as a float. For single scalars, returns absolute difference.
    """
    # Handle single float case
    if isinstance(y_true, (int, float)) and isinstance(y_pred, (int, float)):
        return float(abs(y_true - y_pred))
    y_true_arr = jnp.array(y_true)
    y_pred_arr = jnp.array(y_pred)
    mse = jnp.mean((y_true_arr - y_pred_arr) ** 2)
    return float(jnp.sqrt(mse))


def mae(y_true: float | Array | list[float], y_pred: float | Array | list[float]) -> float:
    """
    Compute Mean Absolute Error (MAE).

    Args:
        y_true: True value(s) (scalar, array, or list)
        y_pred: Predicted value(s) (scalar, array, or list)

    Returns:
        MAE as a float.
    """
    if isinstance(y_true, (int, float)) and isinstance(y_pred, (int, float)):
        return float(abs(y_true - y_pred))
    y_true_arr = jnp.array(y_true)
    y_pred_arr = jnp.array(y_pred)
    return float(jnp.mean(jnp.abs(y_true_arr - y_pred_arr)))


def log_likelihood(
    bll: BayesianLastLayer,
    x: jnp.ndarray,
    y_true: jnp.ndarray
) -> float:
    """
    Compute the average log-likelihood of the data under the predictive distribution.

    Higher values indicate better fit. This metric accounts for both accuracy
    and uncertainty calibration.

    Args:
        bll: BayesianLastLayer instance (must be fitted)
        x: Input features of shape (n_samples, n_features)
        y_true: True values of shape (n_samples,)

    Returns:
        Average log-likelihood as a float
    """
    if bll.posterior_mean is None:
        return float(jnp.nan)

    y_pred, y_std = bll.predict(x, return_std=True)

    # Gaussian log-likelihood: -0.5 * log(2*pi*var) - 0.5 * (y - mu)^2 / var
    variance = y_std ** 2
    log_lik = -0.5 * jnp.log(2 * jnp.pi * variance) - 0.5 * (y_true - y_pred) ** 2 / variance

    return float(jnp.mean(log_lik))


def compute_all_metrics(
    bll: BayesianLastLayer,
    x: jnp.ndarray,
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray | None = None
) -> dict[str, float]:
    """
    Compute all available metrics for a Bayesian Last Layer model.

    This is a convenience function that computes all metrics in one call.

    Args:
        bll: BayesianLastLayer instance (must be fitted)
        x: Input features of shape (n_samples, n_features)
        y_true: True values of shape (n_samples,)
        y_pred: Optional predicted values. If None, will be computed.

    Returns:
        Dictionary mapping metric names to their values
    """
    if y_pred is None:
        y_pred = bll.predict(x, return_std=False)

    metrics = {
        "posterior_uncertainty": posterior_uncertainty(bll),
        "prediction_uncertainty": prediction_uncertainty(bll, x),
        "max_prediction_uncertainty": max_prediction_uncertainty(bll, x),
        "min_prediction_uncertainty": min_prediction_uncertainty(bll, x),
        "weight_norm": weight_norm(bll),
        "posterior_condition_number": posterior_condition_number(bll),
        "observation_precision": observation_precision(bll),
        "observation_noise_std": observation_noise_std(bll),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "log_likelihood": log_likelihood(bll, x, y_true),
    }

    return metrics
