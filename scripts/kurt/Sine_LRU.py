from __future__ import annotations

from collections import deque

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from latinx.data.sine_cosine import SineCosineTranslator
from latinx.metrics.metrics_bayesian_last_layer import (
    mae as bll_mae,
)
from latinx.metrics.metrics_kalman_filter import (
    absolute_error,
    innovation,
    innovation_variance,
    kalman_gain_norm,
    normalized_innovation_squared,
    trace_covariance,
    uncertainty_3sigma,
    weight_norm,
)
from latinx.models.kalman_filter import KalmanFilterHead
from latinx.models.standalone_bayesian_last_layer import StandaloneBayesianLastLayer

# Task configuration: cleaner and more maintainable
TASK_CONFIGS: dict[int, dict[str, float]] = {
    0: {"amplitude": 1.0, "angle_multiplier": 10},
    1: {"amplitude": 0.5, "angle_multiplier": 10},
    2: {"amplitude": -1.5, "angle_multiplier": 10},
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
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity="tanh")

    def forward(
        self, x: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning backbone features only (no prediction head).

        Args:
            x: (batch, seq_len, input_size)
            h: Optional initial hidden state (1, batch, hidden_size)

        Returns:
            Tuple (last_step_features, h_n)
        """
        out, h_n = self.rnn(x, h)
        last_step_features = out[:, -1, :]
        return last_step_features, h_n


if __name__ == "__main__":
    # ==========================================
    # 4. PHASE 1: PRE-TRAINING THE RNN
    # ==========================================
    print("Phase 1: Pre-training RNN on Task 0 (Sine -> Cosine)...")

    # Configuration
    SEQ_LEN = 10
    HIDDEN_SIZE = 32
    rnn_model = SimpleRNN(input_size=1, hidden_size=HIDDEN_SIZE)
    pretrain_head = nn.Linear(HIDDEN_SIZE, 1)  # external head only used here
    params = list(rnn_model.parameters()) + list(pretrain_head.parameters())
    optimizer = optim.Adam(params, lr=0.01)
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
        features, _ = rnn_model(inputs)
        preds = pretrain_head(features)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    print("Pre-training complete. Freezing RNN weights.")
    # Freeze the backbone!
    for param in rnn_model.parameters():
        param.requires_grad = False
    # Discard the pretrain head explicitly
    del pretrain_head

    # ==========================================
    # 5. PHASE 2: ONLINE ADAPTATION
    # ==========================================
    print("Phase 2: Online Bayesian Adaptation across Multiple Functions...")

    SIM_STEPS = 600
    SWITCH_POINTS = [200, 400]  # Points where the task changes

    # Precompute sequences for each task using SineCosineTranslator
    translators_cache = _build_task_translators(num_samples=SIM_STEPS)

    # Initialize the Bayesian Heads
    kf = KalmanFilterHead(feature_dim=HIDDEN_SIZE, rho=0.99, Q_std=0.05, R_std=0.1)
    bll = StandaloneBayesianLastLayer(sigma=0.1, alpha=0.05, feature_dim=HIDDEN_SIZE)

    # Buffer for the sliding window
    input_buffer = deque(np.zeros(SEQ_LEN), maxlen=SEQ_LEN)

    # Storage for accumulated features and targets (for BLL incremental refitting)
    accumulated_features = []
    accumulated_targets = []

    # Storage for plotting
    history = {
        "y_true": [],
        "y_pred": [],  # Kalman Filter predictions
        "y_pred_bll": [],  # Bayesian Last Layer predictions
        "sigma": [],  # Kalman Filter uncertainty
        "sigma_bll": [],  # Bayesian Last Layer uncertainty
        "weights_norm": [],  # KF weight norm
        "weights_norm_bll": [],  # BLL weight norm
        "trace_covariance": [],  # KF trace covariance
        "trace_covariance_bll": [],  # BLL posterior uncertainty
        "nis": [],
        "absolute_error": [],  # KF absolute error
        "absolute_error_bll": [],  # BLL absolute error
        "uncertainty_3sigma": [],
        "uncertainty_3sigma_bll": [],
        "kalman_gain_norm": [],
        "innovation": [],
        "innovation_variance": [],
    }

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
        features_tensor, _ = rnn_model(input_tensor)

        # Convert to JAX array for the Kalman Filter
        phi = jnp.array(features_tensor.detach().numpy().T)  # Shape (M, 1)

        # Convert features for BLL: shape (1, feature_dim)
        phi_bll = jnp.array(features_tensor.detach().numpy())  # Shape (1, M)

        # 2. Kalman Filter: Bayesian Prediction & (conditional) Update
        # Train (update) ONLY during Task 0; Tasks 1 & 2 are prediction-only.
        pred_val = kf.predict(phi)
        if curr_task == 0:
            # Update parameters (training phase)
            uncertainty_S, error = kf.update(y_t, pred_val)
        else:
            # Prediction only (no parameter update). Use predicted covariance (P_minus) for uncertainty.
            # Ensure predict() populated H and P_minus.
            if kf.H is not None and kf.P_minus is not None:
                S_mat = kf.H @ kf.P_minus @ kf.H.T + kf.R  # shape (1,1)
                uncertainty_S = float(S_mat.item())
            else:
                # Fallback if internal state not set (should not happen if predict() succeeded)
                uncertainty_S = float(jnp.nan)
            error = y_t - pred_val

        # 3. Bayesian Last Layer: Prediction & Incremental Refitting
        # Incremental refitting for BLL (only during Task 0)
        if curr_task == 0:
            # Accumulate features and targets
            accumulated_features.append(phi_bll[0])  # Store as 1D array (M,)
            accumulated_targets.append(y_t)

            # Refit BLL on accumulated data
            if len(accumulated_features) > 0:
                features_array = jnp.array(accumulated_features)  # (n_samples, M)
                targets_array = jnp.array(accumulated_targets)  # (n_samples,)
                bll.fit(features_array, targets_array)

        # Make prediction with BLL (after refitting if Task 0, or with existing model if Tasks 1-2)
        if bll._is_fitted:
            pred_val_bll, sigma_bll = bll.predict(phi_bll, return_std=True)
            pred_val_bll = float(pred_val_bll.item())
            sigma_bll = float(sigma_bll.item())
        else:
            # Not fitted yet (shouldn't happen after first Task 0 step)
            pred_val_bll = 0.0
            sigma_bll = float(jnp.nan)

        # --- Store Results ---
        history["y_true"].append(y_t)
        history["y_pred"].append(pred_val)
        history["y_pred_bll"].append(pred_val_bll)
        history["sigma"].append(np.sqrt(uncertainty_S))
        history["sigma_bll"].append(sigma_bll)
        history["weights_norm"].append(weight_norm(kf))
        # Compute BLL weight norm directly (compatible interface)
        if bll._is_fitted and bll.posterior_mean is not None:
            history["weights_norm_bll"].append(float(jnp.linalg.norm(bll.posterior_mean)))
        else:
            history["weights_norm_bll"].append(float(jnp.nan))
        history["trace_covariance"].append(trace_covariance(kf))
        # Compute BLL posterior uncertainty directly (trace of covariance)
        if bll._is_fitted and bll.posterior_covariance is not None:
            history["trace_covariance_bll"].append(float(jnp.trace(bll.posterior_covariance)))
        else:
            history["trace_covariance_bll"].append(float(jnp.nan))
        history["nis"].append(normalized_innovation_squared(kf))
        history["absolute_error"].append(absolute_error(y_t, pred_val))
        history["absolute_error_bll"].append(bll_mae([y_t], [pred_val_bll]))
        history["uncertainty_3sigma"].append(uncertainty_3sigma(kf))
        history["uncertainty_3sigma_bll"].append(
            3.0 * sigma_bll if not jnp.isnan(sigma_bll) else float(jnp.nan)
        )
        history["kalman_gain_norm"].append(kalman_gain_norm(kf))
        history["innovation"].append(innovation(kf))
        history["innovation_variance"].append(innovation_variance(kf))

    # ==========================================
    # 6. PLOTTING - COMPARISON FIGURES
    # ==========================================
    save_path = "results/figures/Figure_A.png"
    time_axis = np.arange(SIM_STEPS)
    y_true = np.array(history["y_true"])
    y_pred = np.array(history["y_pred"])
    y_pred_bll = np.array(history["y_pred_bll"])
    sigma = np.array(history["sigma"])
    sigma_bll = np.array(history["sigma_bll"])

    # Filter out NaN values for BLL (before it's fitted)
    valid_bll_mask = ~np.isnan(y_pred_bll)

    plt.figure(figsize=(15, 12))

    # Plot 1: Trajectory and Adaptation - Comparison
    plt.subplot(3, 1, 1)
    plt.title("RNN Backbone + Bayesian Heads: Kalman Filter vs Bayesian Last Layer")
    plt.plot(time_axis, y_true, "k-", label="Ground Truth", alpha=0.6, linewidth=2)
    plt.plot(time_axis, y_pred, "r--", label="Kalman Filter", linewidth=1.5, alpha=0.8)
    plt.plot(
        time_axis[valid_bll_mask],
        y_pred_bll[valid_bll_mask],
        "b:",
        label="Bayesian Last Layer",
        linewidth=1.5,
        alpha=0.8,
    )

    # Mark context switches
    for pt in SWITCH_POINTS:
        plt.axvline(pt, color="blue", linestyle=":", linewidth=2)

    plt.text(50, 1.5, "Task 0: Cosine\n(amplitude=1.0, Training)", fontsize=10, color="blue")
    plt.text(
        250, 2.5, "Task 1: 0.5x Cosine\n(amplitude=0.5, Prediction Only)", fontsize=10, color="blue"
    )
    plt.text(
        450,
        1.5,
        "Task 2: -1.5x Cosine\n(amplitude=-1.5, Prediction Only)",
        fontsize=10,
        color="blue",
    )
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.ylabel("Value")

    # Plot 2: Uncertainty Comparison
    plt.subplot(3, 1, 2)
    plt.title("Uncertainty Estimation Comparison")
    error_kf = y_true - y_pred
    error_bll = y_true - y_pred_bll
    plt.plot(time_axis, np.abs(error_kf), "grey", alpha=0.5, label="KF Absolute Error", linewidth=1)
    plt.plot(
        time_axis[valid_bll_mask],
        np.abs(error_bll[valid_bll_mask]),
        "lightblue",
        alpha=0.5,
        label="BLL Absolute Error",
        linewidth=1,
    )
    # Plot 3-sigma confidence intervals
    plt.fill_between(
        time_axis, 0, 3 * sigma, color="red", alpha=0.15, label=r"KF Uncertainty (3$\sigma$)"
    )
    plt.fill_between(
        time_axis[valid_bll_mask],
        0,
        3 * sigma_bll[valid_bll_mask],
        color="blue",
        alpha=0.15,
        label=r"BLL Uncertainty (3$\sigma$)",
    )
    for pt in SWITCH_POINTS:
        plt.axvline(pt, color="blue", linestyle=":")
    plt.legend()
    plt.ylabel("Magnitude")

    # Plot 3: Weight Norms Comparison
    plt.subplot(3, 1, 3)
    plt.title("Norm of Weight Vector (Adaptation Effort)")
    plt.plot(time_axis, history["weights_norm"], "purple", label="KF ||w||", linewidth=1.5)
    weights_norm_bll = np.array(history["weights_norm_bll"])
    valid_weights_mask = ~np.isnan(weights_norm_bll)
    plt.plot(
        time_axis[valid_weights_mask],
        weights_norm_bll[valid_weights_mask],
        "orange",
        label="BLL ||w||",
        linewidth=1.5,
    )
    for pt in SWITCH_POINTS:
        plt.axvline(pt, color="blue", linestyle=":")
    plt.xlabel("Time Step")
    plt.ylabel("L2 Norm")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    # ==========================================
    # 7. METRICS PLOTTING - COMPARISON
    # ==========================================
    print("Generating metrics comparison figure...")
    metrics_save_path = "results/figures/Figure_Metrics.png"
    time_axis = np.arange(SIM_STEPS)

    # Create a comprehensive metrics figure with comparisons
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle(
        "Kalman Filter vs Bayesian Last Layer: Metrics Comparison", fontsize=16, fontweight="bold"
    )

    # Prepare BLL data with NaN filtering
    trace_cov_bll = np.array(history["trace_covariance_bll"])
    abs_error_bll = np.array(history["absolute_error_bll"])
    uncertainty_3sigma_bll = np.array(history["uncertainty_3sigma_bll"])
    weights_norm_bll = np.array(history["weights_norm_bll"])
    valid_bll_mask = ~np.isnan(trace_cov_bll)

    # Plot 1: Trace of Covariance Matrix - Comparison
    axes[0, 0].plot(
        time_axis, history["trace_covariance"], "b-", linewidth=1.5, label="KF Trace(P)", alpha=0.8
    )
    axes[0, 0].plot(
        time_axis[valid_bll_mask],
        trace_cov_bll[valid_bll_mask],
        "r--",
        linewidth=1.5,
        label="BLL Trace(Cov)",
        alpha=0.8,
    )
    axes[0, 0].set_title("Trace of Covariance Matrix (Uncertainty)")
    axes[0, 0].set_ylabel("Trace")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    for pt in SWITCH_POINTS:
        axes[0, 0].axvline(pt, color="red", linestyle=":", linewidth=1, alpha=0.7)

    # Plot 2: Normalized Innovation Squared (NIS) - KF only
    axes[0, 1].plot(time_axis, history["nis"], "g-", linewidth=1.5, label="NIS")
    axes[0, 1].axhline(1.0, color="r", linestyle="--", linewidth=1, label="Target (NIS=1)")
    axes[0, 1].set_title("Normalized Innovation Squared (NIS) - KF Only")
    axes[0, 1].set_ylabel("NIS")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    for pt in SWITCH_POINTS:
        axes[0, 1].axvline(pt, color="red", linestyle=":", linewidth=1, alpha=0.7)

    # Plot 3: Absolute Error - Comparison
    axes[1, 0].plot(
        time_axis, history["absolute_error"], "orange", linewidth=1.5, label="KF", alpha=0.8
    )
    axes[1, 0].plot(
        time_axis[valid_bll_mask],
        abs_error_bll[valid_bll_mask],
        "brown",
        linewidth=1.5,
        label="BLL",
        alpha=0.8,
        linestyle="--",
    )
    axes[1, 0].set_title("Absolute Error Comparison")
    axes[1, 0].set_ylabel("|y_true - y_pred|")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    for pt in SWITCH_POINTS:
        axes[1, 0].axvline(pt, color="red", linestyle=":", linewidth=1, alpha=0.7)

    # Plot 4: Uncertainty (3 Sigma) - Comparison
    axes[1, 1].plot(
        time_axis, history["uncertainty_3sigma"], "purple", linewidth=1.5, label="KF", alpha=0.8
    )
    valid_uncertainty_mask = ~np.isnan(uncertainty_3sigma_bll)
    axes[1, 1].plot(
        time_axis[valid_uncertainty_mask],
        uncertainty_3sigma_bll[valid_uncertainty_mask],
        "cyan",
        linewidth=1.5,
        label="BLL",
        alpha=0.8,
        linestyle="--",
    )
    axes[1, 1].set_title("Uncertainty (3-Sigma Confidence Interval)")
    axes[1, 1].set_ylabel("3σ")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    for pt in SWITCH_POINTS:
        axes[1, 1].axvline(pt, color="red", linestyle=":", linewidth=1, alpha=0.7)

    # Plot 5: Weight Norm - Comparison
    axes[2, 0].plot(
        time_axis, history["weights_norm"], "brown", linewidth=1.5, label="KF ||μ||", alpha=0.8
    )
    valid_weights_mask = ~np.isnan(weights_norm_bll)
    axes[2, 0].plot(
        time_axis[valid_weights_mask],
        weights_norm_bll[valid_weights_mask],
        "orange",
        linewidth=1.5,
        label="BLL ||w||",
        alpha=0.8,
        linestyle="--",
    )
    axes[2, 0].set_title("Norm of Weight Vector")
    axes[2, 0].set_ylabel("L2 Norm")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    for pt in SWITCH_POINTS:
        axes[2, 0].axvline(pt, color="red", linestyle=":", linewidth=1, alpha=0.7)

    # Plot 6: Kalman Gain Norm - KF only
    axes[2, 1].plot(time_axis, history["kalman_gain_norm"], "teal", linewidth=1.5)
    axes[2, 1].set_title("Norm of Kalman Gain (KF Only)")
    axes[2, 1].set_ylabel("||K||")
    axes[2, 1].grid(True, alpha=0.3)
    for pt in SWITCH_POINTS:
        axes[2, 1].axvline(pt, color="red", linestyle=":", linewidth=1, alpha=0.7)

    # Plot 7: Innovation - KF only
    axes[3, 0].plot(time_axis, history["innovation"], "magenta", linewidth=1.5)
    axes[3, 0].axhline(0.0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[3, 0].set_title("Innovation (Prediction Error) - KF Only")
    axes[3, 0].set_xlabel("Time Step")
    axes[3, 0].set_ylabel("Innovation")
    axes[3, 0].grid(True, alpha=0.3)
    for pt in SWITCH_POINTS:
        axes[3, 0].axvline(pt, color="red", linestyle=":", linewidth=1, alpha=0.7)

    # Plot 8: Innovation Variance - KF only
    axes[3, 1].plot(time_axis, history["innovation_variance"], "darkgreen", linewidth=1.5)
    axes[3, 1].set_title("Innovation Variance (S) - KF Only")
    axes[3, 1].set_xlabel("Time Step")
    axes[3, 1].set_ylabel("Variance")
    axes[3, 1].grid(True, alpha=0.3)
    for pt in SWITCH_POINTS:
        axes[3, 1].axvline(pt, color="red", linestyle=":", linewidth=1, alpha=0.7)

    plt.tight_layout()
    plt.savefig(metrics_save_path, dpi=300)
    plt.show()
    print(f"Metrics comparison figure saved to {metrics_save_path}")
