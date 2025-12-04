"""
Standalone Bayesian Last Layer implementation.

This module provides a model-agnostic Bayesian linear regression layer that works
with pre-computed features from any JAX or PyTorch model.
"""

import jax.numpy as jnp
import jax
import numpy as np

# Enable float64 precision for numerical stability
jax.config.update("jax_enable_x64", True)

class StandaloneBayesianLastLayer:
    """
    Model-agnostic Bayesian last layer using Bayesian linear regression.

    This class accepts pre-computed features from a JAX model 
    and performs Bayesian linear regression to compute posterior distributions
    over weights. Supports both single-output and multi-output regression.

    Mathematical formulation:
        Prior: w ~ N(0, (1/alpha) * I)
        Likelihood: y | w, X ~ N(Xw, (1/beta) * I)
        Posterior: w | X, y ~ N(m_N, S_N)
        where:
            beta = 1 / sigma^2 (observation precision)
            S_N = (alpha * I + beta * X^T X)^{-1} (posterior covariance)
            m_N = beta * S_N * X^T y (posterior mean)

    Args:
        sigma: Observation noise standard deviation (controls beta = 1/sigma^2).
        alpha: Prior precision parameter (regularization strength).
        feature_dim: Dimensionality of input features (optional, inferred from data if not provided).

    Example:
        >>> # With JAX model
        >>> features = model_hidden.apply(params, x)  # Extract features
        >>> bll = StandaloneBayesianLastLayer(sigma=0.1, alpha=0.01, feature_dim=features.shape[1])
        >>> bll.fit(features, y)
        >>> y_pred, y_std = bll.predict(features, return_std=True)
    """

    def __init__(
        self,
        sigma: float = 0.3,
        alpha: float = 0.05,
        feature_dim: int | None = None,
    ):
        """
        Initialize Bayesian last layer.

        Args:
            sigma: Observation noise standard deviation.
            alpha: Prior precision parameter.
            feature_dim: Dimensionality of input features. If None, will be inferred from fit().
        """
        self.sigma = sigma
        self.alpha = alpha
        self.feature_dim = feature_dim

        # Derived parameters
        self.beta = 1.0 / (sigma**2)  # observation precision

        # Posterior parameters (fitted)
        self.posterior_mean = None
        self.posterior_precision = None
        self.posterior_covariance = None
        self._is_fitted = False

    def fit(self, features: jnp.ndarray | np.ndarray, y: jnp.ndarray | np.ndarray)->None:
        """
        Fit Bayesian linear regression on pre-computed features.

        Args:
            features: Pre-computed features of shape (n_samples, feature_dim).
                      Can be JAX or NumPy array.
            y: Target values of shape (n_samples,), (n_samples, 1), or (n_samples, output_dim).
               Can be JAX or NumPy array.
               - For single-output: (n_samples,) or (n_samples, 1)
               - For multi-output: (n_samples, output_dim) where output_dim > 1

        Returns:
            self (for method chaining).
        """
        # Convert to JAX arrays
        phi_x = jnp.asarray(features)
        y_arr = jnp.asarray(y)

        # Handle y shape: convert (n_samples, 1) to (n_samples,) for single-output
        if y_arr.ndim == 2 and y_arr.shape[1] == 1:
            y_arr = y_arr.ravel()
        # For multi-output (n_samples, output_dim), keep as 2D

        # Infer feature dimension if not provided
        if self.feature_dim is None:
            self.feature_dim = phi_x.shape[1]

        # Validate shapes
        n_samples, M = phi_x.shape
        if M != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch: expected {self.feature_dim}, got {M}")
        if len(y_arr) != n_samples:
            raise ValueError(
                f"Number of samples mismatch: features has {n_samples}, y has {len(y_arr)}"
            )

        # Compute posterior parameters
        IM = jnp.eye(M)
        self.posterior_precision = self.alpha * IM + self.beta * phi_x.T @ phi_x
        self.posterior_mean = self.beta * jnp.linalg.solve(
            self.posterior_precision, phi_x.T @ y_arr
        )

        # For single-output: ensure posterior_mean is 1D (feature_dim,)
        # For multi-output: keep posterior_mean as 2D (feature_dim, output_dim)
        if y_arr.ndim == 1:
            self.posterior_mean = self.posterior_mean.ravel()

        self.posterior_covariance = jnp.linalg.inv(self.posterior_precision)
        self._is_fitted = True

    def predict(
        self,
        features: jnp.ndarray | np.ndarray,
        return_std: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make predictions using the fitted Bayesian last layer.

        Args:
            features: Pre-computed features of shape (n_samples, feature_dim).
                      Can be JAX or NumPy array.
            return_std: If True, return both mean and standard deviation.

        Returns:
            If return_std=False: predictions of shape (n_samples,).
            If return_std=True: tuple of (predictions, std) both of shape (n_samples,).

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before making predictions. Call fit() first.")

        # Convert to JAX array
        phi_x = jnp.asarray(features)

        # Validate feature dimension
        if phi_x.shape[1] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.feature_dim}, got {phi_x.shape[1]}"
            )

        # Compute predictive mean using einsum
        if self.posterior_mean.ndim == 1:
            # Single-output
            pred_mean = jnp.einsum("d,td->t", self.posterior_mean, phi_x)
        else:
            # Multi-output
            pred_mean = jnp.einsum("md,tm->td", self.posterior_mean, phi_x)

        if return_std:
            # Compute predictive variance
            # Var[y*] = 1/beta + phi(x*)^T @ S_N @ phi(x*)
            pred_var = 1.0 / self.beta + jnp.einsum(
                "si,ij,sj->s", phi_x, self.posterior_covariance, phi_x
            )
            pred_std = jnp.sqrt(pred_var)
            return pred_mean, pred_std

        return pred_mean

    def get_weight_statistics(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get posterior statistics of the weights.

        Returns:
            Tuple of (posterior_mean, posterior_std) where:
                - posterior_mean: Mean weights
                  Single-output: shape (feature_dim,)
                  Multi-output: shape (feature_dim, output_dim)
                - posterior_std: Standard deviation of weights (same shape as posterior_mean)

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before accessing weight statistics.")

        # Diagonal of covariance gives variance for each weight
        # Note: For multi-output, each output dimension shares the same covariance
        posterior_std = jnp.sqrt(jnp.diag(self.posterior_covariance))

        # Match the shape of posterior_mean
        if self.posterior_mean.ndim == 2:
            # Multi-output: broadcast std to match (feature_dim, output_dim)
            posterior_std = posterior_std[:, None]

        return self.posterior_mean, posterior_std

    def get_total_uncertainty(self) -> float:
        """
        Get total uncertainty in weight estimates (trace of posterior covariance).

        Returns:
            Trace of posterior covariance matrix.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before accessing uncertainty.")

        return float(jnp.trace(self.posterior_covariance))

    def update_hyperparameters(self, sigma: float | None = None, alpha: float | None = None):
        """
        Update hyperparameters and refit if model was already fitted.

        Args:
            sigma: New observation noise standard deviation (optional).
            alpha: New prior precision parameter (optional).

        Note:
            This requires re-fitting with the original features and targets,
            which are not stored. Consider calling fit() again with new hyperparameters.
        """
        if sigma is not None:
            self.sigma = sigma
            self.beta = 1.0 / (sigma**2)

        if alpha is not None:
            self.alpha = alpha

        # Reset fitted state since hyperparameters changed
        if self._is_fitted:
            self._is_fitted = False
            print(
                "Warning: Hyperparameters updated. Model is no longer fitted. "
                "Call fit() again with your features and targets."
            )

    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"StandaloneBayesianLastLayer("
            f"sigma={self.sigma:.4f}, "
            f"alpha={self.alpha:.4f}, "
            f"feature_dim={self.feature_dim}, "
            f"status={status})"
        )
