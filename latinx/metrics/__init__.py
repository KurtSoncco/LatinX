"""Metrics module for evaluating Kalman filters and other models."""

# Kalman Filter metrics
from latinx.metrics.metrics_kalman_filter import (
    absolute_error,
    compute_all_metrics as compute_all_kalman_metrics,
    compute_metrics_over_time,
    innovation,
    innovation_variance,
    kalman_gain_norm,
    normalized_innovation_squared,
    rmse as kalman_rmse,
    trace_covariance,
    uncertainty_3sigma,
    weight_norm as kalman_weight_norm,
)

# Bayesian Last Layer metrics
from latinx.metrics.metrics_bayesian_last_layer import (
    compute_all_metrics as compute_all_bll_metrics,
    log_likelihood,
    mae,
    max_prediction_uncertainty,
    min_prediction_uncertainty,
    observation_noise_std,
    observation_precision,
    posterior_condition_number,
    posterior_uncertainty,
    prediction_uncertainty,
    rmse as bll_rmse,
    weight_norm as bll_weight_norm,
)

__all__ = [
    # Kalman Filter metrics
    "absolute_error",
    "compute_all_kalman_metrics",
    "compute_metrics_over_time",
    "innovation",
    "innovation_variance",
    "kalman_gain_norm",
    "kalman_rmse",
    "kalman_weight_norm",
    "normalized_innovation_squared",
    "trace_covariance",
    "uncertainty_3sigma",
    # Bayesian Last Layer metrics
    "bll_rmse",
    "bll_weight_norm",
    "compute_all_bll_metrics",
    "log_likelihood",
    "mae",
    "max_prediction_uncertainty",
    "min_prediction_uncertainty",
    "observation_noise_std",
    "observation_precision",
    "posterior_condition_number",
    "posterior_uncertainty",
    "prediction_uncertainty",
]
