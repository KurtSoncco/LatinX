import jax
import jax.numpy as jnp
from jax import Array

# Enable float64 precision for numerical stability
jax.config.update("jax_enable_x64", True)


class KalmanFilterHead:
    """
    Kalman Filter-based adaptive linear regression head.

    Implements Bayesian linear regression with a forgetting factor to enable
    adaptation to non-stationary environments. Uses the Kalman filter update
    equations for online learning.

    Args:
        feature_dim: Dimension of input features
        rho: Forgetting factor (0 < rho <= 1). Values < 1 allow adaptation
        Q_std: Standard deviation of process noise
        R_std: Standard deviation of measurement noise
        initial_uncertainty: Initial covariance scale for parameter uncertainty
    """

    def __init__(
        self,
        feature_dim: int,
        rho: float = 0.99,
        Q_std: float = 0.01,
        R_std: float = 0.1,
        initial_uncertainty: float = 1.0,
    ):
        if not 0 < rho <= 1:
            raise ValueError(f"rho must be in (0, 1], got {rho}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if Q_std < 0 or R_std <= 0:
            raise ValueError("Q_std and R_std must be non-negative and positive respectively")

        self.M = feature_dim
        self.rho = rho

        # Initial State: Weights (mu) and Uncertainty (P)
        self.mu = jnp.zeros((self.M, 1), dtype=jnp.float64)
        self.P = jnp.eye(self.M, dtype=jnp.float64) * initial_uncertainty

        # Noise Covariances
        self.Q = jnp.eye(self.M, dtype=jnp.float64) * (Q_std**2)  # Process noise
        self.R = jnp.array([[R_std**2]], dtype=jnp.float64)  # Measurement noise
        self.A = jnp.eye(self.M, dtype=jnp.float64) * rho  # Forgetting factor

        # For storing intermediate values
        self.mu_minus = None
        self.P_minus = None
        self.H = None
        self.K = None  # Kalman gain
        self.innovation = None  # Innovation (prediction error)
        self.innovation_variance = None  # Innovation covariance (S)

    def predict(self, phi_x: Array) -> float:
        """
        Predict the output given input features using current parameter estimates.

        Args:
            phi_x: Feature vector of shape (M, 1) or (M,)

        Returns:
            Predicted scalar value
        """
        # Ensure phi_x is 2D column vector
        if phi_x.ndim == 1:
            phi_x = phi_x.reshape(-1, 1)

        if phi_x.shape[0] != self.M:
            raise ValueError(f"Feature dimension mismatch: expected {self.M}, got {phi_x.shape[0]}")

        # --- 1. Time Update (Predict) ---
        self.mu_minus = self.A @ self.mu
        self.P_minus = self.A @ self.P @ self.A.T + self.Q

        # --- 2. Measurement Prediction ---
        self.H = phi_x.T
        y_pred = self.H @ self.mu_minus
        return float(y_pred.item())

    def update(self, y_true: float, y_pred_val: float) -> tuple[float, float]:
        """
        Update parameter estimates given observed true value.

        Args:
            y_true: True observed value
            y_pred_val: Previously predicted value from predict()

        Returns:
            Tuple of (innovation_covariance, prediction_error)
        """
        if self.mu_minus is None or self.P_minus is None or self.H is None:
            raise RuntimeError("Must call predict() before update()")

        # --- 3. Measurement Update (Correct) ---
        error = y_true - y_pred_val

        # Innovation Covariance (S)
        S = self.H @ self.P_minus @ self.H.T + self.R

        # Kalman Gain (K) - use solve for better numerical stability
        K = self.P_minus @ self.H.T / S  # More stable than matrix inverse for 1D case

        # Store metrics-related values
        self.innovation = error
        self.innovation_variance = S
        self.K = K

        # Update State (Weights)
        self.mu = self.mu_minus + K * error

        # Update Uncertainty (Covariance) - Joseph form for numerical stability
        I_KH = jnp.eye(self.M, dtype=jnp.float64) - K @ self.H
        self.P = I_KH @ self.P_minus @ I_KH.T + K @ self.R @ K.T

        return float(S.item()), float(error)

    def get_weight_statistics(self) -> tuple[Array, Array]:
        """
        Get current weight estimates and their uncertainties.

        Returns:
            Tuple of (weights, weight_std)
            - weights: Current weight estimates, shape (M,)
            - weight_std: Standard deviation of weights, shape (M,)

        Example:
            >>> weights, std = kf.get_weight_statistics()
            >>> print(f"Weight: {weights[0]:.3f} ± {std[0]:.3f}")
        """
        # Diagonal of covariance gives variance for each weight
        weight_std = jnp.sqrt(jnp.diag(self.P))
        return self.mu.ravel(), weight_std  # Use ravel() to ensure 1D array


class MultiOutputKalmanFilterHead:
    """
    Multi-output Kalman Filter for vector-valued predictions.

    Implements Bayesian linear regression for multiple output dimensions.
    Each output dimension has independent parameters but shares the same
    input features and forgetting factor.

    Args:
        feature_dim: Dimension of input features
        output_dim: Number of output dimensions (default: 3 for x, y, z)
        rho: Forgetting factor (0 < rho <= 1). Values < 1 allow adaptation
        Q_std: Standard deviation of process noise
        R_std: Standard deviation of measurement noise (assumed same for all outputs)
        initial_uncertainty: Initial covariance scale for parameter uncertainty
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int = 3,
        rho: float = 0.99,
        Q_std: float = 0.01,
        R_std: float = 0.1,
        initial_uncertainty: float = 1.0,
    ):
        if not 0 < rho <= 1:
            raise ValueError(f"rho must be in (0, 1], got {rho}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if Q_std < 0 or R_std <= 0:
            raise ValueError("Q_std and R_std must be non-negative and positive respectively")

        self.M = feature_dim
        self.D = output_dim
        self.rho = rho

        # Weights: (M, D) - one column per output dimension
        # Each column is an independent weight vector
        self.mu = jnp.zeros((self.M, self.D), dtype=jnp.float64)

        # Covariance: (M, M) - shared across all output dimensions
        # This assumes outputs are conditionally independent given features
        self.P = jnp.eye(self.M, dtype=jnp.float64) * initial_uncertainty

        # Noise Covariances
        self.Q = jnp.eye(self.M, dtype=jnp.float64) * (Q_std**2)  # Process noise (shared)
        self.R = jnp.eye(self.D, dtype=jnp.float64) * (R_std**2)  # Measurement noise (diagonal)
        self.A = jnp.eye(self.M, dtype=jnp.float64) * rho

        # Storage for intermediate values
        self.mu_minus = None
        self.P_minus = None
        self.H = None
        self.K = None
        self.innovation = None  # (D,) vector
        self.innovation_variance = None  # (D, D) matrix

    def predict(self, phi_x: Array) -> Array:
        """
        Predict all output dimensions given input features.

        Args:
            phi_x: Feature vector of shape (M, 1) or (M,)

        Returns:
            Predicted output vector of shape (D,)
        """
        # Ensure phi_x is 2D column vector
        if phi_x.ndim == 1:
            phi_x = phi_x.reshape(-1, 1)

        if phi_x.shape[0] != self.M:
            raise ValueError(f"Feature dimension mismatch: expected {self.M}, got {phi_x.shape[0]}")

        # Time Update (Predict) - shared across all outputs
        self.mu_minus = self.A @ self.mu  # (M, D)
        self.P_minus = self.A @ self.P @ self.A.T + self.Q  # (M, M)

        # Measurement Prediction for all outputs
        self.H = phi_x.T  # (1, M)
        y_pred = self.H @ self.mu_minus  # (1, D)

        return y_pred.squeeze()  # (D,)

    def update(self, y_true: Array, y_pred: Array) -> tuple[Array, Array]:
        """
        Update parameter estimates given observed true values.

        Args:
            y_true: True observed values, shape (D,) or (D, 1)
            y_pred: Previously predicted values from predict(), shape (D,)

        Returns:
            Tuple of (innovation_covariance_diagonal, prediction_errors)
            Both are arrays of shape (D,)
        """
        if self.mu_minus is None or self.P_minus is None or self.H is None:
            raise RuntimeError("Must call predict() before update()")

        # Ensure inputs are proper shapes
        y_true = jnp.asarray(y_true).reshape(-1)  # (D,)
        y_pred = jnp.asarray(y_pred).reshape(-1)  # (D,)

        if y_true.shape[0] != self.D:
            raise ValueError(f"Output dimension mismatch: expected {self.D}, got {y_true.shape[0]}")

        # Measurement Update - process each output dimension
        # Innovation (error)
        error = y_true - y_pred  # (D,)

        # For each output dimension, compute Kalman gain and update
        # Since outputs are independent given features, we can process them separately
        updated_mu = jnp.zeros((self.M, self.D), dtype=jnp.float64)
        S_diag = jnp.zeros(self.D, dtype=jnp.float64)

        for d in range(self.D):
            # Innovation covariance for this output: S = H @ P_minus @ H^T + R[d,d]
            S_d = self.H @ self.P_minus @ self.H.T + self.R[d, d]  # (1, 1)
            S_diag = S_diag.at[d].set(S_d.squeeze())

            # Kalman gain for this output: K = P_minus @ H^T / S
            K_d = self.P_minus @ self.H.T / S_d  # (M, 1)

            # Update weights for this output: mu_d = mu_minus_d + K_d * error_d
            updated_mu = updated_mu.at[:, d].set(self.mu_minus[:, d] + K_d.squeeze() * error[d])

        self.mu = updated_mu

        # Update covariance (shared across outputs, use average Kalman gain)
        # For simplicity, use the update from the first dimension
        # (In principle, could average across dimensions)
        S_0 = self.H @ self.P_minus @ self.H.T + self.R[0, 0]
        K_0 = self.P_minus @ self.H.T / S_0
        I_KH = jnp.eye(self.M, dtype=jnp.float64) - K_0 @ self.H
        # Use outer product for covariance update: K @ R @ K^T = K @ K^T * R (when R is scalar)
        self.P = I_KH @ self.P_minus @ I_KH.T + K_0 @ K_0.T * self.R[0, 0]

        # Store for metrics
        self.innovation = error
        self.innovation_variance = jnp.diag(S_diag)

        return S_diag, error
