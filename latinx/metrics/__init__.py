"""Metrics module for evaluating Kalman filters and other models."""

# Kalman Filter metrics
# Bayesian Last Layer metrics
from latinx.metrics.metrics_bayesian_last_layer import (
    compute_all_metrics as compute_all_bll_metrics,
)
from latinx.metrics.metrics_bayesian_last_layer import (
    log_likelihood,
    mae,
    max_prediction_uncertainty,
    min_prediction_uncertainty,
    observation_noise_std,
    observation_precision,
    posterior_condition_number,
    posterior_uncertainty,
    prediction_uncertainty,
)
from latinx.metrics.metrics_bayesian_last_layer import rmse as bll_rmse
from latinx.metrics.metrics_bayesian_last_layer import weight_norm as bll_weight_norm
from latinx.metrics.metrics_kalman_filter import absolute_error
from latinx.metrics.metrics_kalman_filter import compute_all_metrics as compute_all_kalman_metrics
from latinx.metrics.metrics_kalman_filter import (
    compute_metrics_over_time,
    innovation,
    innovation_variance,
    kalman_gain_norm,
    normalized_innovation_squared,
)
from latinx.metrics.metrics_kalman_filter import rmse as kalman_rmse
from latinx.metrics.metrics_kalman_filter import trace_covariance, uncertainty_3sigma
from latinx.metrics.metrics_kalman_filter import weight_norm as kalman_weight_norm

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
