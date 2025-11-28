"""
Kalman Filter Experiment with Bessel Ripple Dataset (Radial Sequences)

Demonstrates:
1. Training an RNN backbone on radial wave sequences (Bessel ripple)
2. Freezing the backbone and using it as a feature extractor
3. Applying Kalman Filter head for online Bayesian adaptation
4. Testing generalization to different amplitude (concept drift)
5. Comprehensive visualization of KF behavior and uncertainty

Architecture:
- RNN Backbone: sequence of z(r) values → features (frozen after pre-training)
- KF Head: features → z(r_next) (Bayesian linear regression)

Key Insight:
- Bessel ripple is radially symmetric: z(x,y) = f(r) where r = sqrt(x² + y²)
- We treat radius r as a "pseudo-time" sequential dimension
- RNN learns the oscillatory pattern along the radial direction
- Similar to Sine_LRU but for radial waves instead of temporal waves

Data Processing:
- Sort all points by radial distance from origin
- Create sliding window sequences of z values: [z(r), z(r+Δr), ..., z(r+9Δr)]
- RNN predicts z at the next radius step
- This captures the oscillatory Bessel function along r
"""

from __future__ import annotations

from collections import deque

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from latinx.data.bessel_ripple import BesselRippleTranslator
from latinx.metrics.metrics_kalman_filter import innovation, trace_covariance
from latinx.models.kalman_filter import KalmanFilterHead
from latinx.models.rnn import SimpleRNN
from latinx.models.standalone_bayesian_last_layer import StandaloneBayesianLastLayer
from latinx.models.bll_utils import (
    run_bll_training,
    run_bll_prediction_only,
    combine_bll_results,
    plot_bll_results,
)

# ==========================================
# HELPER FUNCTIONS FOR KF PROCESSING
# ==========================================


def run_kf_with_updates(
    kf_model: KalmanFilterHead,
    rnn_model: nn.Module,
    r_values: np.ndarray,
    z_clean: np.ndarray,
    z_noisy: np.ndarray,
    start_idx: int,
    end_idx: int,
    seq_len: int,
) -> dict[str, list]:
    """
    Run KF with weight updates (training/adaptation mode).

    Args:
        kf_model: KalmanFilterHead instance
        rnn_model: Trained and frozen RNN model
        r_values: Radial distance values
        z_clean: Clean z values (ground truth)
        z_noisy: Noisy z values (observations)
        start_idx: Starting index for processing
        end_idx: Ending index for processing
        seq_len: Sequence length for RNN input

    Returns:
        Dictionary with all metrics for plotting
    """
    results = {
        "r": [],
        "z_true": [],
        "z_noisy": [],
        "z_pred_kf": [],
        "sigma_kf": [],
        "innovation": [],
        "innovation_sq": [],
        "trace_P": [],
    }

    # Initialize buffer
    input_buffer = deque(maxlen=seq_len)
    # Pre-fill buffer with first seq_len points
    for i in range(seq_len):
        idx = start_idx + i
        if idx < end_idx:
            input_buffer.append(z_noisy[idx])

    # Process data with KF updates
    for t in range(start_idx + seq_len, end_idx):
        # Get current data
        r_t = r_values[t]
        z_true = z_clean[t]
        z_noisy_t = z_noisy[t]

        # Extract features using frozen RNN
        input_tensor = torch.tensor(np.array(input_buffer), dtype=torch.float32).view(1, seq_len, 1)

        with torch.no_grad():
            features_tensor, _ = rnn_model(input_tensor)

        # Convert to JAX for Kalman Filter
        phi = jnp.array(features_tensor.detach().numpy().T)  # Shape (HIDDEN_SIZE, 1)

        # KF prediction
        z_pred_kf = kf_model.predict(phi)

        # KF update
        uncertainty_S, error = kf_model.update(z_noisy_t, z_pred_kf)

        # Store results
        results["r"].append(r_t)
        results["z_true"].append(z_true)
        results["z_noisy"].append(z_noisy_t)
        results["z_pred_kf"].append(z_pred_kf)
        results["sigma_kf"].append(np.sqrt(uncertainty_S))
        results["innovation"].append(innovation(kf_model))
        results["innovation_sq"].append(innovation(kf_model) ** 2)
        results["trace_P"].append(trace_covariance(kf_model))

        # Update buffer for next iteration
        input_buffer.append(z_noisy_t)

    return results


def run_kf_prediction_only(
    kf_model: KalmanFilterHead,
    rnn_model: nn.Module,
    r_values: np.ndarray,
    z_clean: np.ndarray,
    z_noisy: np.ndarray,
    start_idx: int,
    end_idx: int,
    seq_len: int,
) -> dict[str, list]:
    """
    Run KF prediction only (extrapolation mode, no weight updates).

    Args:
        kf_model: KalmanFilterHead instance (already trained)
        rnn_model: Trained and frozen RNN model
        r_values: Radial distance values
        z_clean: Clean z values (ground truth)
        z_noisy: Noisy z values (observations)
        start_idx: Starting index for processing
        end_idx: Ending index for processing
        seq_len: Sequence length for RNN input

    Returns:
        Dictionary with all metrics for plotting
    """
    results = {
        "r": [],
        "z_true": [],
        "z_noisy": [],
        "z_pred_kf": [],
        "sigma_kf": [],
        "innovation": [],
        "innovation_sq": [],
        "trace_P": [],
    }

    # Initialize buffer
    input_buffer = deque(maxlen=seq_len)
    # Pre-fill buffer with first seq_len points
    for i in range(seq_len):
        idx = start_idx + i
        if idx < end_idx:
            input_buffer.append(z_noisy[idx])

    # Process data with KF prediction only (no update step)
    for t in range(start_idx + seq_len, end_idx):
        # Get current data
        r_t = r_values[t]
        z_true = z_clean[t]
        z_noisy_t = z_noisy[t]

        # Extract features using frozen RNN
        input_tensor = torch.tensor(np.array(input_buffer), dtype=torch.float32).view(1, seq_len, 1)

        with torch.no_grad():
            features_tensor, _ = rnn_model(input_tensor)

        # Convert to JAX for Kalman Filter
        phi = jnp.array(features_tensor.detach().numpy().T)  # Shape (HIDDEN_SIZE, 1)

        # KF prediction (no update step - extrapolation mode)
        z_pred_kf = kf_model.predict(phi)

        # Compute uncertainty from KF state
        # S = H @ P_minus @ H.T + R
        if kf_model.H is not None and kf_model.P_minus is not None:
            S_mat = kf_model.H @ kf_model.P_minus @ kf_model.H.T + kf_model.R
            uncertainty_S = float(S_mat.item())
        else:
            uncertainty_S = float(jnp.nan)

        # Compute prediction error (innovation)
        error = z_true - z_pred_kf

        # Store results
        results["r"].append(r_t)
        results["z_true"].append(z_true)
        results["z_noisy"].append(z_noisy_t)
        results["z_pred_kf"].append(z_pred_kf)
        results["sigma_kf"].append(np.sqrt(uncertainty_S))
        results["innovation"].append(error)
        results["innovation_sq"].append(error**2)
        results["trace_P"].append(float(jnp.trace(kf_model.P)))

        # Update buffer for next iteration
        input_buffer.append(z_noisy_t)

    return results


def combine_results(*results_list) -> dict[str, np.ndarray]:
    """
    Combine multiple result dictionaries into one for plotting.

    Args:
        *results_list: Variable number of result dictionaries

    Returns:
        Combined dictionary with numpy arrays
    """
    combined = {
        "r": [],
        "z_true": [],
        "z_noisy": [],
        "z_pred_kf": [],
        "sigma_kf": [],
        "innovation": [],
        "innovation_sq": [],
        "trace_P": [],
    }

    for results in results_list:
        for key in combined.keys():
            combined[key].extend(results[key])

    # Convert to numpy arrays
    return {key: np.array(val) for key, val in combined.items()}


# ==========================================
# TASK CONFIGURATION
# ==========================================
TASK_CONFIGS: dict[int, dict[str, float]] = {
    0: {
        "k": 6.0,
        "amplitude": 5.0,
        "damping": 0.05,
        "noise_std": 0.03,
    },
    1: {
        "k": 20.0,
        "amplitude": 15.0,
        "damping": 0.19,
        "noise_std": 0.03,
    },
}


def _build_task_data_radial(
    n_radial_points: int = 200,
    r_max: float = 8.0,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Build truly radial Bessel ripple datasets with ONE sample per radius.

    Uses BesselRippleTranslator.generate_radial() to properly handle
    noise injection using the class's noise parameters.

    Args:
        n_radial_points: Number of radial points to sample
        r_max: Maximum radius

    Returns:
        Dict mapping task_id -> {"r", "z", "z_noisy"}
    """
    cache: dict[int, dict[str, np.ndarray]] = {}

    for tid, cfg in TASK_CONFIGS.items():
        translator = BesselRippleTranslator(
            k=cfg["k"],
            amplitude=cfg["amplitude"],
            damping=cfg["damping"],
            x_range=(-r_max, r_max),
            y_range=(-r_max, r_max),
            grid_size=n_radial_points,
            noise_std=cfg["noise_std"],
            use_bessel=True,
            seed=42,
        )

        # Generate 1D radial data (one sample per radius)
        df = translator.generate_radial(n_points=n_radial_points, r_max=r_max)

        cache[tid] = {
            "r": np.asarray(df["r"].values, dtype=np.float32),
            "z": np.asarray(df["z"].values, dtype=np.float32),
            "z_noisy": np.asarray(df["z_noisy"].values, dtype=np.float32),
        }

    return cache


# ==========================================
# MAIN EXPERIMENT
# ==========================================
if __name__ == "__main__":
    # ==========================================
    # CONFIGURATION
    # ==========================================
    BACKWARD_EXTRAPOLATION = True  # Train on last half, test on first half

    print("Bessel Ripple RNN+KF Experiment (Radial Sequences)")

    # ==========================================
    # PHASE 1: PRE-TRAIN RNN BACKBONE
    # ==========================================
    print("Phase 1: Pre-training RNN on Task 0 (Bessel Ripple, amplitude=5.0)...")

    N_RADIAL_POINTS = 400  # Number of radial samples (one per radius)
    SEQ_LEN = 10  # Sequence length for RNN
    HIDDEN_SIZE = 32
    PRETRAIN_EPOCHS = 150

    # Build radial datasets (one sample per radius)
    task_data = _build_task_data_radial(n_radial_points=N_RADIAL_POINTS, r_max=8)

    # Create RNN model (reusing from Sine_LRU)
    rnn_model = SimpleRNN(input_size=1, hidden_size=HIDDEN_SIZE)
    pretrain_head = nn.Linear(HIDDEN_SIZE, 1)
    params = list(rnn_model.parameters()) + list(pretrain_head.parameters())
    optimizer = optim.Adam(params, lr=0.005)
    criterion = nn.MSELoss()

    # Prepare Task 0 training sequences
    task0 = task_data[0]
    z_values_full = task0["z_noisy"]

    # Select training portion based on mode
    split_idx = len(z_values_full) // 2
    if BACKWARD_EXTRAPOLATION:
        # Train on second half (outer radii)
        z_values = z_values_full[split_idx:]
        print(f"Training on SECOND HALF: indices [{split_idx}:{len(z_values_full)}]")
        print(f"  (outer radii: r ≈ {task0['r'][split_idx]:.2f} to {task0['r'][-1]:.2f})")
    else:
        # Train on first half (inner radii)
        z_values = z_values_full[:split_idx]
        print(f"Training on FIRST HALF: indices [0:{split_idx}]")
        print(f"  (inner radii: r ≈ {task0['r'][0]:.2f} to {task0['r'][split_idx - 1]:.2f})")

    print(f"Creating sequences from {len(z_values)} radial samples (1 per radius)...")
    print(f"Sequence length: {SEQ_LEN} (learning pattern over {SEQ_LEN} radius steps)")

    # Create training sequences
    inputs = []
    targets = []
    for i in range(len(z_values) - SEQ_LEN):
        # Input: sequence of z values at consecutive radii
        seq_in = z_values[i : i + SEQ_LEN]
        # Target: z value at the next radius
        target_out = z_values[i + SEQ_LEN]

        inputs.append(seq_in.reshape(SEQ_LEN, 1))
        targets.append(target_out)

    inputs_torch = torch.tensor(np.array(inputs), dtype=torch.float32)
    targets_torch = torch.tensor(np.array(targets), dtype=torch.float32).view(-1, 1)

    # Training loop
    print("\nTraining RNN to predict z(r_next) from sequence [z(r), ..., z(r+9)]...")
    for epoch in range(PRETRAIN_EPOCHS):
        optimizer.zero_grad()
        features, _ = rnn_model(inputs_torch)
        preds = pretrain_head(features)
        loss = criterion(preds, targets_torch)
        loss.backward()
        optimizer.step()

        if epoch % 30 == 0 or epoch == PRETRAIN_EPOCHS - 1:
            print(f"Epoch {epoch:3d}: Loss {loss.item():.6f}")

    print("\nPre-training complete. Freezing RNN weights.")
    for param in rnn_model.parameters():
        param.requires_grad = False
    rnn_model.eval()  # Set to evaluation mode
    del pretrain_head

    # ==========================================
    # PHASE 2: ONLINE KALMAN FILTER ADAPTATION
    # ==========================================
    SWITCH_POINT = len(z_values_full) // 2

    # Kalman Filter hyperparameters
    KF_RHO = 1.0
    KF_Q_STD = 0.0002
    KF_R_STD = 0.01

    # Initialize Kalman Filter Head
    kf = KalmanFilterHead(feature_dim=HIDDEN_SIZE, rho=KF_RHO, Q_std=KF_Q_STD, R_std=KF_R_STD)

    # Prepare data
    r_values = task0["r"]
    z_clean = task0["z"]
    z_noisy = task0["z_noisy"]

    if BACKWARD_EXTRAPOLATION:
        print("\nPhase 2: Backward Extrapolation + Task Switching Mode")
        print("Step 1: Training KF on SECOND HALF of Task 0 (outer radii, amplitude=5.0)")
        print(f"  Indices: [{SWITCH_POINT}, {len(r_values)})")
        print(f"  Radius range: r ∈ [{r_values[SWITCH_POINT]:.2f}, {r_values[-1]:.2f}]")

        # Train KF on second half of Task 0
        results_train = run_kf_with_updates(
            kf, rnn_model, r_values, z_clean, z_noisy, SWITCH_POINT, len(r_values), SEQ_LEN
        )

        print("\nStep 2: Testing KF on FIRST HALF of Task 1 (inner radii - BACKWARD EXTRAPOLATION + NEW TASK)")
        print(f"  Indices: [0, {SWITCH_POINT})")
        print(f"  Radius range: r ∈ [{r_values[0]:.2f}, {r_values[SWITCH_POINT - 1]:.2f}]")
        print(f"  Challenge: Unseen inner radii + different amplitude (Task 1: amplitude=15.0)")

        # Get Task 1 data for backward extrapolation test
        task1 = task_data[1]
        z_clean_task1 = task1["z"]
        z_noisy_task1 = task1["z_noisy"]

        # Test KF on first half of Task 1 (backward extrapolation + task shift)
        results_test = run_kf_prediction_only(
            kf, rnn_model, r_values, z_clean_task1, z_noisy_task1, 0, SWITCH_POINT, SEQ_LEN
        )

        # Combine: test first (Task 1 backward), then train (Task 0 forward)
        results = combine_results(results_test, results_train)
        task_ids = np.concatenate(
            [
                np.ones(len(results_test["r"])),  # 1 = Task 1 (backward extrapolation)
                np.zeros(len(results_train["r"])),  # 0 = Task 0 (training)
            ]
        )

    else:
        print("\nPhase 2: Normal Mode (Task Switching)")
        print("Step 1: Training KF on FIRST HALF (Task 0, amplitude=5.0)")
        print(f"  Indices: [0, {SWITCH_POINT})")
        print(f"  Radius range: r ∈ [{r_values[0]:.2f}, {r_values[SWITCH_POINT - 1]:.2f}]")

        # Train KF on first half (Task 0)
        results_train = run_kf_with_updates(
            kf, rnn_model, r_values, z_clean, z_noisy, 0, SWITCH_POINT, SEQ_LEN
        )

        print("\nStep 2: Testing KF on SECOND HALF (Task 1, amplitude=2.0)")
        print(f"  Indices: [{SWITCH_POINT}, {len(r_values)})")
        print(f"  Radius range: r ∈ [{r_values[SWITCH_POINT]:.2f}, {r_values[-1]:.2f}]")

        # Get Task 1 data
        task1 = task_data[1]
        z_clean_task1 = task1["z"]
        z_noisy_task1 = task1["z_noisy"]

        # Test KF on second half (Task 1)
        results_test = run_kf_prediction_only(
            kf,
            rnn_model,
            r_values,
            z_clean_task1,
            z_noisy_task1,
            SWITCH_POINT,
            len(r_values),
            SEQ_LEN,
        )

        # Combine: training first, then testing
        results = combine_results(results_train, results_test)
        task_ids = np.concatenate(
            [
                np.zeros(len(results_train["r"])),  # 0 = Task 0
                np.ones(len(results_test["r"])),  # 1 = Task 1
            ]
        )

    print("\nOnline adaptation complete!")

    # ==========================================
    # VISUALIZATION
    # ==========================================
    print("\nGenerating visualizations...")

    # Extract results (already numpy arrays from combine_results)
    r_axis = results["r"]
    z_true = results["z_true"]
    z_noisy = results["z_noisy"]
    z_pred_kf = results["z_pred_kf"]
    sigma_kf = results["sigma_kf"]
    innovation_sq = results["innovation_sq"]
    trace_P = results["trace_P"]

    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 14))

    # Find switch point in plotted data
    if BACKWARD_EXTRAPOLATION:
        # In backward mode: task 1 first, then task 0. Find transition from 1→0
        switch_idx = np.where(task_ids == 0)[0][0] if 0 in task_ids else len(r_axis)
    else:
        # In normal mode: task 0 first, then task 1. Find transition from 0→1
        switch_idx = np.where(task_ids == 1)[0][0] if 1 in task_ids else len(r_axis)
    switch_r = r_axis[switch_idx] if switch_idx < len(r_axis) else r_axis[-1]

    # ==========================================
    # PLOT 1: Radial Profile - All Predictions
    # ==========================================
    ax1 = plt.subplot(3, 2, 1)
    ax1.set_title("Radial Wave Profile: Predictions vs Truth", fontsize=12, fontweight="bold")
    ax1.plot(r_axis, z_true, "k-", label="Ground Truth (Clean)", alpha=0.7, linewidth=2)
    ax1.plot(r_axis, z_noisy, "gray", label="Noisy Observations", alpha=0.3, linewidth=1)
    ax1.plot(r_axis, z_pred_kf, "r-", label="KF Prediction", linewidth=2)

    if BACKWARD_EXTRAPOLATION:
        ax1.axvline(
            switch_r, color="orange", linestyle=":", linewidth=2, label="Train/Test Boundary"
        )
    else:
        ax1.axvline(switch_r, color="green", linestyle=":", linewidth=2, label="Task Switch")

    ax1.set_xlabel("Radius (r)")
    ax1.set_ylabel("z Value (Wave Height)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    if BACKWARD_EXTRAPOLATION:
        # Backward extrapolation + task switching labels
        ax1.text(
            r_axis[len(r_axis) // 4],
            ax1.get_ylim()[1] * 0.85,
            "Task 1 BACKWARD\n(inner radii)\nA=15.0\nNot Trained",
            ha="center",
            fontsize=9,
            color="darkred",
            bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.3},
        )
        if switch_idx < len(r_axis):
            ax1.text(
                r_axis[switch_idx + (len(r_axis) - switch_idx) // 2],
                ax1.get_ylim()[1] * 0.85,
                "Task 0 TRAINED\n(outer radii)\nA=1.0\nKF Updates",
                ha="center",
                fontsize=9,
                color="darkgreen",
                bbox={"boxstyle": "round", "facecolor": "lightgreen", "alpha": 0.3},
            )
    else:
        # Normal mode labels
        ax1.text(
            r_axis[len(r_axis) // 4],
            ax1.get_ylim()[1] * 0.85,
            "Task 0\n(A=1.0)\nKF Training",
            ha="center",
            fontsize=9,
            color="blue",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.3},
        )
        if switch_idx < len(r_axis):
            ax1.text(
                r_axis[switch_idx + (len(r_axis) - switch_idx) // 2],
                ax1.get_ylim()[1] * 0.85,
                "Task 1\n(A=2.0)\nPrediction Only",
                ha="center",
                fontsize=9,
                color="blue",
                bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.3},
            )

    # ==========================================
    # PLOT 2: KF Uncertainty Along Radius
    # ==========================================
    ax2 = plt.subplot(3, 2, 2)
    ax2.set_title("KF Uncertainty (σ) vs Radius", fontsize=12, fontweight="bold")
    ax2.plot(r_axis, sigma_kf, "purple", linewidth=2, label="KF Uncertainty (σ)")
    ax2.fill_between(
        r_axis,
        0,
        3 * sigma_kf,
        color="purple",
        alpha=0.2,
        label="3σ Confidence",
    )
    ax2.axvline(switch_r, color="green", linestyle=":", linewidth=2)
    ax2.set_xlabel("Radius (r)")
    ax2.set_ylabel("Uncertainty (σ)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ==========================================
    # PLOT 3: Innovation Squared Along Radius
    # ==========================================
    ax3 = plt.subplot(3, 2, 3)
    ax3.set_title("Innovation² vs Radius (Prediction Surprise)", fontsize=12, fontweight="bold")
    ax3.plot(r_axis, innovation_sq, "orange", linewidth=1.5, label="Innovation²")
    ax3.axvline(switch_r, color="green", linestyle=":", linewidth=2)
    ax3.set_xlabel("Radius (r)")
    ax3.set_ylabel("Innovation² (Surprise)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ==========================================
    # PLOT 4: Trace of Covariance P vs Radius
    # ==========================================
    ax4 = plt.subplot(3, 2, 4)
    ax4.set_title("Trace(P) vs Radius (Weight Uncertainty)", fontsize=12, fontweight="bold")
    ax4.plot(r_axis, trace_P, "teal", linewidth=2, label="Trace(P)")
    ax4.axvline(switch_r, color="green", linestyle=":", linewidth=2)
    ax4.set_xlabel("Radius (r)")
    ax4.set_ylabel("Trace(P)")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ==========================================
    # PLOT 5: Absolute Prediction Error (KF)
    # ==========================================
    ax5 = plt.subplot(3, 2, 5)
    ax5.set_title("KF Prediction Error vs Radius", fontsize=12, fontweight="bold")

    # Calculate absolute error
    abs_error_kf = np.abs(z_true - z_pred_kf)

    ax5.plot(
        r_axis,
        abs_error_kf,
        "r-",
        label="KF Absolute Error",
        linewidth=1.5,
        alpha=0.7,
    )
    if BACKWARD_EXTRAPOLATION:
        ax5.axvline(switch_r, color="orange", linestyle=":", linewidth=2)
    else:
        ax5.axvline(switch_r, color="green", linestyle=":", linewidth=2)
    ax5.set_xlabel("Radius (r)")
    ax5.set_ylabel("Absolute Error")
    ax5.legend(loc="upper right", fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ==========================================
    # PLOT 6: Summary Statistics
    # ==========================================
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis("off")

    # Calculate summary statistics
    region0_mask = task_ids == 0
    region1_mask = task_ids == 1

    abs_error_kf = np.abs(z_true - z_pred_kf)
    kf_rmse_region0 = np.sqrt(np.mean(abs_error_kf[region0_mask] ** 2))
    kf_rmse_region1 = np.sqrt(np.mean(abs_error_kf[region1_mask] ** 2)) if region1_mask.any() else 0

    if BACKWARD_EXTRAPOLATION:
        region0_name = "TASK 0 TRAINING (outer radii, A=5.0)"
        region1_name = "TASK 1 BACKWARD (inner radii, A=15.0)"
    else:
        region0_name = "TASK 0 (amplitude=5.0)"
        region1_name = "TASK 1 (amplitude=15.0)"

    summary_text = f"""
    SUMMARY STATISTICS
    {"=" * 40}

    {region0_name}:
    ────────────────────────────────────────
    KF RMSE:        {kf_rmse_region0:.6f}
    Mean σ (KF):    {np.mean(sigma_kf[region0_mask]):.6f}
    Mean Trace(P):  {np.mean(trace_P[region0_mask]):.6f}

    {region1_name}:
    ────────────────────────────────────────
    KF RMSE:        {kf_rmse_region1:.6f}
    Mean σ (KF):    {(np.mean(sigma_kf[region1_mask]) if region1_mask.any() else 0):.6f}
    Mean Trace(P):  {(np.mean(trace_P[region1_mask]) if region1_mask.any() else 0):.6f}    CONFIGURATION:
    ────────────────────────────────────────
    Total Points:   {len(r_axis)}
    Switch Radius:  {switch_r:.2f}
    RNN Hidden:     {HIDDEN_SIZE}
    Sequence Len:   {SEQ_LEN}
    """

    print(summary_text)

    plt.tight_layout()

    # Create descriptive filename with KF parameters
    mode = "backward" if BACKWARD_EXTRAPOLATION else "forward"
    save_path = (
        f"results/figures/Bessel_Ripple_KF_{mode}_rho{KF_RHO}_Q{KF_Q_STD:.2e}_R{KF_R_STD:.3f}.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to {save_path}")
    plt.show()

    print("\n" + "=" * 70)
    print("KF Experiment Complete!")
    print("=" * 70)
    print("\nKey Insight: RNN learned the radial oscillation pattern!")
    print("Radius treated as sequential dimension (like time)")
    print("KF adapts to amplitude changes in the radial wave")

    # ==========================================
    # PHASE 3: BAYESIAN LAST LAYER EXPERIMENT
    # ==========================================
    print("\n" + "=" * 70)
    print("PHASE 3: BAYESIAN LAST LAYER (BLL) EXPERIMENT")
    print("=" * 70)

    # BLL hyperparameters (should match actual data noise level)
    BLL_SIGMA = 0.03  # Observation noise std (matches noise_std in task configs)
    BLL_ALPHA = 0.01  # Prior precision (reduced for less regularization)

    # Initialize Bayesian Last Layer
    bll = StandaloneBayesianLastLayer(sigma=BLL_SIGMA, alpha=BLL_ALPHA, feature_dim=HIDDEN_SIZE)

    # Fetch fresh Task 0 data for BLL (avoid using modified variables from KF experiment)
    task0_bll = task_data[0]
    r_values_task0_bll = task0_bll["r"]
    z_clean_task0_bll = task0_bll["z"]
    z_noisy_task0_bll = task0_bll["z_noisy"]

    if BACKWARD_EXTRAPOLATION:
        print("\nStep 1: Training BLL on SECOND HALF of Task 0 (outer radii, amplitude=5.0)")
        print(f"  Indices: [{SWITCH_POINT}, {len(r_values_task0_bll)})")

        # Train BLL on second half of Task 0
        bll_results_train = run_bll_training(
            bll, rnn_model, r_values_task0_bll, z_clean_task0_bll, z_noisy_task0_bll, SWITCH_POINT, len(r_values_task0_bll), SEQ_LEN
        )

        print("\nStep 2: Testing BLL on FIRST HALF of Task 1 (inner radii - BACKWARD + NEW TASK)")
        print(f"  Indices: [0, {SWITCH_POINT})")
        print(f"  Challenge: Unseen inner radii + different amplitude (Task 1: amplitude=15.0)")

        # Get Task 1 data for backward extrapolation test (fetch fresh to avoid any issues)
        task1_bll = task_data[1]
        r_values_task1_bll = task1_bll["r"]
        z_clean_task1_bll = task1_bll["z"]
        z_noisy_task1_bll = task1_bll["z_noisy"]

        # Test BLL on first half of Task 1 (backward extrapolation + task shift)
        bll_results_test = run_bll_prediction_only(
            bll, rnn_model, r_values_task1_bll, z_clean_task1_bll, z_noisy_task1_bll, 0, SWITCH_POINT, SEQ_LEN
        )

        # Combine results: test first (Task 1 backward), then train (Task 0 forward)
        bll_results = combine_bll_results(bll_results_test, bll_results_train)
        bll_task_ids = np.concatenate(
            [
                np.ones(len(bll_results_test["r"])),  # 1 = Task 1 (backward extrapolation)
                np.zeros(len(bll_results_train["r"])),  # 0 = Task 0 (training)
            ]
        )

    else:
        print("\nStep 1: Training BLL on FIRST HALF (Task 0)")
        print(f"  Indices: [0, {SWITCH_POINT})")

        # Train BLL on first half of Task 0
        bll_results_train = run_bll_training(
            bll, rnn_model, r_values_task0_bll, z_clean_task0_bll, z_noisy_task0_bll, 0, SWITCH_POINT, SEQ_LEN
        )

        print("\nStep 2: Testing BLL on SECOND HALF (Task 1)")
        print(f"  Indices: [{SWITCH_POINT}, {len(r_values)})")

        # Get Task 1 data for testing (fetch fresh to avoid any issues)
        task1_bll = task_data[1]
        r_values_task1_bll = task1_bll["r"]
        z_clean_task1_bll = task1_bll["z"]
        z_noisy_task1_bll = task1_bll["z_noisy"]

        # Test BLL on second half (Task 1 data)
        bll_results_test = run_bll_prediction_only(
            bll, rnn_model, r_values_task1_bll, z_clean_task1_bll, z_noisy_task1_bll, SWITCH_POINT, len(r_values_task1_bll), SEQ_LEN
        )

        # Combine results
        bll_results = combine_bll_results(bll_results_train, bll_results_test)
        bll_task_ids = np.concatenate(
            [
                np.zeros(len(bll_results_train["r"])),
                np.ones(len(bll_results_test["r"])),
            ]
        )

    print("\nBLL training and prediction complete!")

    # ==========================================
    # BLL VISUALIZATION (SEPARATE WINDOW)
    # ==========================================
    print("\n" + "=" * 70)
    print("Generating BLL visualizations...")
    print("=" * 70)

    # Use shared utility function for BLL plotting
    mode = "backward" if BACKWARD_EXTRAPOLATION else "forward"
    save_path_bll = (
        f"results/figures/Bessel_Ripple_BLL_{mode}_sigma{BLL_SIGMA:.3f}_alpha{BLL_ALPHA:.3f}.png"
    )
    plot_bll_results(
        bll_results=bll_results,
        task_ids=bll_task_ids,
        task_configs=TASK_CONFIGS,
        bll_sigma=BLL_SIGMA,
        bll_alpha=BLL_ALPHA,
        hidden_size=HIDDEN_SIZE,
        seq_len=SEQ_LEN,
        pretrain_epochs=PRETRAIN_EPOCHS,
        backward_mode=BACKWARD_EXTRAPOLATION,
        save_path=save_path_bll,
    )

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print("\nKey Insights:")
    print("- KF: Online Bayesian learning with sequential updates")
    print("- BLL: Batch Bayesian learning with constant posterior uncertainty")
    print("- Both use the same frozen RNN backbone for feature extraction")
    print("=" * 70)
