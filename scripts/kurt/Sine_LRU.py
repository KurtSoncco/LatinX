from __future__ import annotations

from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from latinx.data.sine_cosine import SineCosineTranslator

# Task configuration: cleaner and more maintainable
TASK_CONFIGS: dict[int, dict[str, float]] = {
    0: {"amplitude": 1.0, "angle_multiplier": 2 * np.pi},
    1: {"amplitude": 0.5, "angle_multiplier": 4 * np.pi},
    2: {"amplitude": 1.5, "angle_multiplier": 1 * np.pi},
}


def _build_task_translators(num_samples: int) -> dict[int, dict[str, np.ndarray]]:
    """
    Build per-task translators and return precomputed arrays for indexing.

    Returns a dict mapping task_id -> {"t": np.ndarray, "sine": np.ndarray, "cosine": np.ndarray}.
    """
    cache: dict[int, dict[str, np.ndarray]] = {}
    for tid, cfg in TASK_CONFIGS.items():
        translator = SineCosineTranslator(
            amplitude=cfg["amplitude"],
            angle_multiplier=cfg["angle_multiplier"],
            num_samples=num_samples,
        )
        df = translator.generate()
        cache[tid] = {
            "t": np.asarray(df["t"].values, dtype=np.float64),
            "sine": np.asarray(df["sine"].values, dtype=np.float64),
            "cosine": np.asarray(df["cosine"].values, dtype=np.float64),
        }
    return cache


# ==========================================
# 2. RNN MODEL (The Backbone)
# ==========================================
class SimpleRNN(nn.Module):
    """
    Simple RNN model with feature extraction and optional prediction head.

    Args:
        input_size: Dimension of input features
        hidden_size: Dimension of hidden state
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 32):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # A standard RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity="tanh")
        # A temporary linear head used only for pre-training
        self.temp_head = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the RNN.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            h: Optional initial hidden state of shape (1, batch, hidden_size)

        Returns:
            Tuple of (pred, last_step_features, h_n) where:
                pred: Predictions of shape (batch, 1)
                last_step_features: Features from last timestep (batch, hidden_size)
                h_n: Final hidden state (1, batch, hidden_size)
        """
        # x shape: (batch, seq_len, input_size)
        out, h_n = self.rnn(x, h)

        # We extract features from the last time step of the sequence
        last_step_features = out[:, -1, :]

        # Prediction (used during pre-training only)
        pred = self.temp_head(last_step_features)

        return pred, last_step_features, h_n


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
        self.mu = np.zeros((self.M, 1), dtype=np.float64)
        self.P = np.eye(self.M, dtype=np.float64) * initial_uncertainty

        # Noise Covariances
        self.Q = np.eye(self.M, dtype=np.float64) * (Q_std**2)  # Process noise
        self.R = np.array([[R_std**2]], dtype=np.float64)  # Measurement noise
        self.A = np.eye(self.M, dtype=np.float64) * rho  # Forgetting factor

        # For storing intermediate values
        self.mu_minus = None
        self.P_minus = None
        self.H = None

    def predict(self, phi_x: np.ndarray) -> float:
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
        I_KH = np.eye(self.M, dtype=np.float64) - K @ self.H
        self.P = I_KH @ self.P_minus @ I_KH.T + K @ self.R @ K.T

        return float(S.item()), float(error)


if __name__ == "__main__":
    # ==========================================
    # 4. PHASE 1: PRE-TRAINING THE RNN
    # ==========================================
    print("Phase 1: Pre-training RNN on Task 0 (Sine -> Cosine)...")

    # Configuration
    SEQ_LEN = 10
    HIDDEN_SIZE = 32
    rnn_model = SimpleRNN(input_size=1, hidden_size=HIDDEN_SIZE)
    optimizer = optim.Adam(rnn_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Train Loop (Offline)
    for epoch in range(100):
        # Generate batch data
        t_vals = np.arange(0, 200, 0.1)
        inputs = []
        targets = []

        # Create sequences for the RNN
        for i in range(len(t_vals) - SEQ_LEN):
            # Input: Sequence of sine values
            seq_in = np.sin(0.1 * np.arange(i, i + SEQ_LEN))
            # Target: The cosine value at the end of that sequence
            target_out = np.cos(0.1 * (i + SEQ_LEN - 1))

            inputs.append(seq_in.reshape(SEQ_LEN, 1))
            targets.append(target_out)

        inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
        targets = torch.tensor(np.array(targets), dtype=torch.float32).view(-1, 1)

        # Standard PyTorch training step
        optimizer.zero_grad()
        preds, _, _ = rnn_model(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    print("Pre-training complete. Freezing RNN weights.")
    # Freeze the backbone!
    for param in rnn_model.parameters():
        param.requires_grad = False

    # ==========================================
    # 5. PHASE 2: ONLINE ADAPTATION
    # ==========================================
    print("Phase 2: Online Bayesian Adaptation across Multiple Functions...")

    SIM_STEPS = 600
    SWITCH_POINTS = [200, 400]  # Points where the task changes

    # Precompute sequences for each task using SineCosineTranslator
    translators_cache = _build_task_translators(num_samples=SIM_STEPS)

    # Initialize the Bayesian Head
    kf = KalmanFilterHead(feature_dim=HIDDEN_SIZE, rho=0.99, Q_std=0.05, R_std=0.1)

    # Buffer for the sliding window
    input_buffer = deque(np.zeros(SEQ_LEN), maxlen=SEQ_LEN)

    # Storage for plotting
    history = {"y_true": [], "y_pred": [], "sigma": [], "weights_norm": []}

    curr_task = 0

    for t in range(SIM_STEPS):
        # --- Determine Current Task (Drift) ---
        if t >= SWITCH_POINTS[0] and t < SWITCH_POINTS[1]:
            curr_task = 1  # Amplitude Change (2x)
        elif t >= SWITCH_POINTS[1]:
            curr_task = 2  # Inversion (-1x)
        else:
            curr_task = 0  # Original Cosine

        # --- Generate Data from translator ---
        seq = translators_cache[curr_task]
        # Guard against indexing beyond available samples
        idx = min(t, len(seq["sine"]) - 1)
        x_t = float(seq["sine"][idx])
        y_t = float(seq["cosine"][idx])
        input_buffer.append(x_t)

        # --- Forward Pass ---
        # 1. Backbone: Get Features from Frozen RNN
        input_tensor = torch.tensor(np.array(input_buffer), dtype=torch.float32).view(1, SEQ_LEN, 1)
        _, features_tensor, _ = rnn_model(input_tensor)

        # Convert to numpy for the Kalman Filter
        phi = features_tensor.detach().numpy().T  # Shape (M, 1)

        # 2. Head: Bayesian Prediction & Update
        pred_val = kf.predict(phi)
        uncertainty_S, _ = kf.update(y_t, pred_val)

        # --- Store Results ---
        history["y_true"].append(y_t)
        history["y_pred"].append(pred_val)
        history["sigma"].append(np.sqrt(uncertainty_S))
        history["weights_norm"].append(np.linalg.norm(kf.mu))

    # ==========================================
    # 6. PLOTTING
    # ==========================================
    save_path = "results/figures/Figure_A.png"
    time_axis = np.arange(SIM_STEPS)
    y_true = np.array(history["y_true"])
    y_pred = np.array(history["y_pred"])
    sigma = np.array(history["sigma"])

    plt.figure(figsize=(15, 12))

    # Plot 1: Trajectory and Adaptation
    plt.subplot(3, 1, 1)
    plt.title("RNN Backbone + Bayesian Head: Adapting to Task Shifts")
    plt.plot(time_axis, y_true, "k-", label="Ground Truth", alpha=0.6)
    plt.plot(time_axis, y_pred, "r--", label="KF Prediction", linewidth=1.5)

    # Mark context switches
    for pt in SWITCH_POINTS:
        plt.axvline(pt, color="blue", linestyle=":", linewidth=2)

    plt.text(50, 1.5, "Task 1: Cosine\n(Known Task)", fontsize=10, color="blue")
    plt.text(250, 2.5, "Task 2: 2x Cosine\n(Amplitude Drift)", fontsize=10, color="blue")
    plt.text(450, 1.5, "Task 3: -Cosine\n(Inversion Drift)", fontsize=10, color="blue")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    # Plot 2: Uncertainty
    plt.subplot(3, 1, 2)
    plt.title("Uncertainty Estimation (Sigma)")
    error = y_true - y_pred
    plt.plot(time_axis, np.abs(error), "grey", alpha=0.5, label="Absolute Error")
    # Plot 3-sigma confidence interval
    plt.fill_between(
        time_axis, 0, 3 * sigma, color="red", alpha=0.2, label=r"Uncertainty (3$\sigma$)"
    )
    for pt in SWITCH_POINTS:
        plt.axvline(pt, color="blue", linestyle=":")
    plt.legend()
    plt.ylabel("Magnitude")

    # Plot 3: Weight Norms
    plt.subplot(3, 1, 3)
    plt.title("Norm of Weight Vector (Adaptation Effort)")
    plt.plot(time_axis, history["weights_norm"], "purple", label="||w||")
    for pt in SWITCH_POINTS:
        plt.axvline(pt, color="blue", linestyle=":")
    plt.xlabel("Time Step")
    plt.ylabel("L2 Norm")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
