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

        # Update State (Weights)
        self.mu = self.mu_minus + K * error

        # Update Uncertainty (Covariance) - Joseph form for numerical stability
        I_KH = jnp.eye(self.M, dtype=jnp.float64) - K @ self.H
        self.P = I_KH @ self.P_minus @ I_KH.T + K @ self.R @ K.T

        return float(S.item()), float(error)
