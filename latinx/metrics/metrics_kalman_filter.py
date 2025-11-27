"""
Evaluation metrics for Kalman Filter implementations.

This module provides standardized metrics for evaluating Kalman filter performance,
including trace of covariance matrix, NIS (Normalized Innovation Squared), RMSE,
and other diagnostic metrics.
"""

import jax.numpy as jnp
from jax import Array

from latinx.models.kalman_filter import KalmanFilterHead


def trace_covariance(kf: KalmanFilterHead) -> float:
    """
    Compute the trace of the covariance matrix P.

    The trace represents the total uncertainty/variance in the filter.
    Lower values indicate more confidence in the parameter estimates.

    Args:
        kf: KalmanFilterHead instance

    Returns:
        Trace of the covariance matrix P as a float
    """
    return float(jnp.trace(kf.P))


def normalized_innovation_squared(kf: KalmanFilterHead) -> float:
    """
    Compute the Normalized Innovation Squared (NIS).

    NIS = innovation^2 / innovation_variance

    For a well-calibrated filter, NIS should be close to 1.0 on average.
    Values significantly > 1 indicate the filter is underestimating uncertainty.
    Values significantly < 1 indicate the filter is overestimating uncertainty.

    Args:
        kf: KalmanFilterHead instance (must have called update() recently)

    Returns:
        NIS value as a float. Returns NaN if innovation or innovation_variance is None.
    """
    if kf.innovation is None or kf.innovation_variance is None:
        return float(jnp.nan)

    innovation_squared = kf.innovation**2
    nis = innovation_squared / kf.innovation_variance
    return float(nis.item())


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


def absolute_error(
    y_true: float | Array | list[float], y_pred: float | Array | list[float]
) -> float:
    """
    Compute absolute error between true and predicted values.

    Args:
        y_true: True value(s) (scalar, array, or list)
        y_pred: Predicted value(s) (scalar, array, or list)

    Returns:
        Absolute error as a float. For sequences, returns mean absolute error.
    """
    if isinstance(y_true, (int, float)) and isinstance(y_pred, (int, float)):
        return float(abs(y_true - y_pred))
    y_true_arr = jnp.array(y_true)
    y_pred_arr = jnp.array(y_pred)
    return float(jnp.abs(y_true_arr - y_pred_arr).item())


def uncertainty_3sigma(kf: KalmanFilterHead) -> float:
    """
    Compute 3-sigma uncertainty bound.

    This represents the 99.7% confidence interval (3 standard deviations)
    for the prediction uncertainty.

    Args:
        kf: KalmanFilterHead instance

    Returns:
        3-sigma uncertainty as a float. Uses innovation_variance if available,
        otherwise computes from current P and H.
    """
    if kf.innovation_variance is not None:
        sigma = jnp.sqrt(kf.innovation_variance)
    elif kf.H is not None and kf.P_minus is not None:
        # Compute from predicted covariance
        S = kf.H @ kf.P_minus @ kf.H.T + kf.R
        sigma = jnp.sqrt(S)
    else:
        return float(jnp.nan)

    return float((3.0 * sigma).item())


def weight_norm(kf: KalmanFilterHead) -> float:
    """
    Compute the L2 norm of the weight vector (mu).

    This metric tracks the magnitude of the learned parameters,
    useful for monitoring adaptation effort and regularization.

    Args:
        kf: KalmanFilterHead instance

    Returns:
        L2 norm of the weight vector as a float
    """
    return float(jnp.linalg.norm(kf.mu))


def kalman_gain_norm(kf: KalmanFilterHead) -> float:
    """
    Compute the L2 norm of the Kalman gain vector.

    The Kalman gain determines how much the filter trusts new measurements
    versus prior estimates. Larger values indicate more trust in measurements.

    Args:
        kf: KalmanFilterHead instance (must have called update() recently)

    Returns:
        L2 norm of Kalman gain as a float. Returns NaN if K is None.
    """
    if kf.K is None:
        return float(jnp.nan)
    return float(jnp.linalg.norm(kf.K))


def innovation(kf: KalmanFilterHead) -> float:
    """
    Get the innovation (prediction error).

    Innovation = y_true - y_predicted

    Args:
        kf: KalmanFilterHead instance (must have called update() recently)

    Returns:
        Innovation value as a float. Returns NaN if innovation is None.
    """
    if kf.innovation is None:
        return float(jnp.nan)
    return float(kf.innovation)


def innovation_variance(kf: KalmanFilterHead) -> float:
    """
    Get the innovation variance (S).

    This is the predicted uncertainty in the innovation,
    computed as H @ P_minus @ H.T + R.

    Args:
        kf: KalmanFilterHead instance (must have called update() recently)

    Returns:
        Innovation variance as a float. Returns NaN if innovation_variance is None.
    """
    if kf.innovation_variance is None:
        return float(jnp.nan)
    return float(kf.innovation_variance.item())


def compute_all_metrics(
    kf: KalmanFilterHead, y_true: float | Array | list[float], y_pred: float | Array | list[float]
) -> dict[str, float]:
    """
    Compute all available metrics for a Kalman filter at a given time step.

    This is a convenience function that computes all metrics in one call.

    Args:
        kf: KalmanFilterHead instance
        y_true: True value(s)
        y_pred: Predicted value(s)

    Returns:
        Dictionary mapping metric names to their values
    """
    metrics = {
        "trace_covariance": trace_covariance(kf),
        "nis": normalized_innovation_squared(kf),
        "rmse": rmse(y_true, y_pred),
        "absolute_error": absolute_error(y_true, y_pred),
        "uncertainty_3sigma": uncertainty_3sigma(kf),
        "weight_norm": weight_norm(kf),
        "kalman_gain_norm": kalman_gain_norm(kf),
        "innovation": innovation(kf),
        "innovation_variance": innovation_variance(kf),
    }
    return metrics


def compute_metrics_over_time(
    kf: KalmanFilterHead,
    y_true_history: list[float] | Array,
    y_pred_history: list[float] | Array,
) -> dict[str, list[float]]:
    """
    Compute metrics over a time series of predictions.

    This function tracks metrics at each time step, useful for plotting
    and analyzing filter performance over time.

    Args:
        kf: KalmanFilterHead instance (will be used to access current state)
        y_true_history: List or array of true values over time
        y_pred_history: List or array of predicted values over time

    Returns:
        Dictionary mapping metric names to lists of values over time.
        Note: Some metrics (like NIS, innovation) require the filter state
        at each time step, so this function computes what it can from the
        current filter state and the provided histories.
    """
    # Metrics that can be computed from histories
    absolute_errors = [
        absolute_error(yt, yp) for yt, yp in zip(y_true_history, y_pred_history, strict=True)
    ]
    rmse_value = rmse(y_true_history, y_pred_history)

    # Metrics from current filter state
    metrics = {
        "absolute_error": absolute_errors,
        "rmse": [rmse_value] * len(y_true_history),  # Constant over time
        "trace_covariance": [trace_covariance(kf)],  # Current state only
        "uncertainty_3sigma": [uncertainty_3sigma(kf)],  # Current state only
        "weight_norm": [weight_norm(kf)],  # Current state only
        "kalman_gain_norm": [kalman_gain_norm(kf)],  # Current state only
        "innovation": [innovation(kf)],  # Current state only
        "innovation_variance": [innovation_variance(kf)],  # Current state only
        "nis": [normalized_innovation_squared(kf)],  # Current state only
    }

    return metrics
