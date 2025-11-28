"""
Rössler System RNN+KF Experiment (Temporal Sequences)

This experiment demonstrates Bayesian online learning on chaotic time series data:

Architecture:
- RNN Backbone: sequence of state values → features (frozen after pre-training)
- Kalman Filter Head: features → prediction (online adaptation)

Tasks:
- Task 0: Rössler with standard chaotic parameters (a=0.2, b=0.2, c=5.7)
- Task 1: Rössler with different parameters (different dynamics)

The experiment shows how the Kalman Filter adapts to distributional shifts
in chaotic dynamical systems.
"""

import os
from collections import deque

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Bessel_Ripple_KF import (
    SimpleRNN,
)
from latinx.data.rossler import RosslerTranslator
from latinx.models.kalman_filter import MultiOutputKalmanFilterHead


# ==========================================
# TASK CONFIGURATION
# ==========================================
TASK_CONFIGS: dict[int, dict] = {
    0: {
        "a": 0.2,
        "b": 0.2,
        "c": 5.7,
        "noise_pct": 0.02,  # 2% noise relative to state magnitude
        "label": "Standard Chaotic",
    },
    1: {
        "a": 0.3,
        "b": 0.2,
        "c": 3.0,  # Different c parameter (changes attractor shape)
        "noise_pct": 0.02,
        "label": "Modified Dynamics",
    },
}


def _build_task_data(
    n_steps: int = 5000,
    dt: float = 0.01,
    n_transient: int = 1000,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Build Rössler system trajectories for each task.

    Args:
        n_steps: Number of time steps
        dt: Time step size
        n_transient: Number of transient steps to remove

    Returns:
        Dict mapping task_id -> {"t", "x", "y", "z", "x_noisy", "y_noisy", "z_noisy"}
    """
    cache: dict[int, dict[str, np.ndarray]] = {}

    for tid, cfg in TASK_CONFIGS.items():
        translator = RosslerTranslator(
            n_steps=n_steps,
            dt=dt,
            a=cfg["a"],
            b=cfg["b"],
            c=cfg["c"],
            initial_state=[1.0, 0.0, 0.0],
            noise_pct=cfg["noise_pct"],
            seed=42 + tid,  # Different seed for each task
        )

        # Generate trajectory with transient removal
        df = translator.generate_with_transient_removal(n_transient=n_transient)

        cache[tid] = {
            "t": np.asarray(df["t"].values, dtype=np.float32),
            "x": np.asarray(df["x"].values, dtype=np.float32),
            "y": np.asarray(df["y"].values, dtype=np.float32),
            "z": np.asarray(df["z"].values, dtype=np.float32),
            "x_noisy": np.asarray(df["x_noisy"].values, dtype=np.float32),
            "y_noisy": np.asarray(df["y_noisy"].values, dtype=np.float32),
            "z_noisy": np.asarray(df["z_noisy"].values, dtype=np.float32),
        }

    return cache


# ==========================================
# MULTI-OUTPUT KF HELPER FUNCTIONS
# ==========================================


def run_multiout_kf_with_updates(
    kf_model: MultiOutputKalmanFilterHead,
    rnn_model: nn.Module,
    t_values: np.ndarray,
    xyz_clean: np.ndarray,  # Shape (N, 3)
    xyz_noisy: np.ndarray,  # Shape (N, 3)
    start_idx: int,
    end_idx: int,
    seq_len: int,
) -> dict[str, list]:
    """
    Run multi-output KF with weight updates (training/adaptation mode).

    Args:
        kf_model: MultiOutputKalmanFilterHead instance
        rnn_model: Trained and frozen RNN model
        t_values: Time values
        xyz_clean: Clean [x, y, z] values (ground truth), shape (N, 3)
        xyz_noisy: Noisy [x, y, z] values (observations), shape (N, 3)
        start_idx: Starting index for processing
        end_idx: Ending index for processing
        seq_len: Sequence length for RNN input

    Returns:
        Dictionary with all metrics for plotting
    """
    results = {
        "t": [],
        "xyz_true": [],  # Ground truth (N, 3)
        "xyz_noisy": [],  # Noisy observations (N, 3)
        "xyz_pred_kf": [],  # KF predictions (N, 3)
        "sigma_kf": [],  # Uncertainty per dimension (N, 3)
        "innovation": [],  # Innovation per dimension (N, 3)
        "trace_P": [],
    }

    # Initialize buffer with X dimension (could use any dimension for temporal pattern)
    input_buffer = deque(maxlen=seq_len)
    for i in range(seq_len):
        idx = start_idx + i
        if idx < end_idx:
            input_buffer.append(xyz_noisy[idx, 0])  # Use x dimension for sequence

    # Process data with KF updates
    for t in range(start_idx + seq_len, end_idx):
        # Get current data
        t_val = t_values[t]
        xyz_true = xyz_clean[t]  # (3,)
        xyz_obs = xyz_noisy[t]  # (3,)

        # Extract features using frozen RNN
        input_tensor = torch.tensor(np.array(input_buffer), dtype=torch.float32).view(1, seq_len, 1)

        with torch.no_grad():
            features_tensor, _ = rnn_model(input_tensor)

        # Convert to JAX for Kalman Filter
        phi = jnp.array(features_tensor.numpy().T)  # Shape (HIDDEN_SIZE, 1)

        # KF prediction (returns 3D vector)
        xyz_pred = kf_model.predict(phi)  # (3,)

        # KF update
        S_diag, error = kf_model.update(xyz_obs, xyz_pred)  # Both (3,)

        # Store results
        results["t"].append(t_val)
        results["xyz_true"].append(xyz_true)
        results["xyz_noisy"].append(xyz_obs)
        results["xyz_pred_kf"].append(xyz_pred)
        results["sigma_kf"].append(np.sqrt(S_diag))  # Convert to std dev
        results["innovation"].append(error)
        results["trace_P"].append(np.trace(kf_model.P))

        # Update buffer for next iteration (use x dimension)
        input_buffer.append(xyz_obs[0])

    return results


def run_multiout_kf_prediction_only(
    kf_model: MultiOutputKalmanFilterHead,
    rnn_model: nn.Module,
    t_values: np.ndarray,
    xyz_clean: np.ndarray,
    xyz_noisy: np.ndarray,
    start_idx: int,
    end_idx: int,
    seq_len: int,
) -> dict[str, list]:
    """
    Run multi-output KF prediction only (extrapolation mode, no weight updates).

    Same as run_multiout_kf_with_updates but without calling update().
    """
    results = {
        "t": [],
        "xyz_true": [],
        "xyz_noisy": [],
        "xyz_pred_kf": [],
        "sigma_kf": [],
        "innovation": [],
        "trace_P": [],
    }

    input_buffer = deque(maxlen=seq_len)
    for i in range(seq_len):
        idx = start_idx + i
        if idx < end_idx:
            input_buffer.append(xyz_noisy[idx, 0])

    for t in range(start_idx + seq_len, end_idx):
        t_val = t_values[t]
        xyz_true = xyz_clean[t]
        xyz_obs = xyz_noisy[t]

        input_tensor = torch.tensor(np.array(input_buffer), dtype=torch.float32).view(1, seq_len, 1)

        with torch.no_grad():
            features_tensor, _ = rnn_model(input_tensor)

        phi = jnp.array(features_tensor.numpy().T)

        # KF prediction only (no update)
        xyz_pred = kf_model.predict(phi)

        # Compute predictive uncertainty from covariance
        # sigma_d = sqrt(H @ P_minus @ H^T + R[d,d]) for each dimension
        sigma_pred = np.zeros(3)
        for d in range(3):
            S_d = kf_model.H @ kf_model.P_minus @ kf_model.H.T + kf_model.R[d, d]
            sigma_pred[d] = np.sqrt(S_d.item())

        # Compute innovation manually (no update)
        error = xyz_true - xyz_pred

        # Store results
        results["t"].append(t_val)
        results["xyz_true"].append(xyz_true)
        results["xyz_noisy"].append(xyz_obs)
        results["xyz_pred_kf"].append(xyz_pred)
        results["sigma_kf"].append(sigma_pred)  # Actual predictive uncertainty
        results["innovation"].append(error)
        results["trace_P"].append(np.trace(kf_model.P))

        # Update buffer (use x dimension)
        input_buffer.append(xyz_obs[0])

        # Propagate covariance forward (time update only, no measurement update)
        # This causes uncertainty to grow over time in prediction-only mode
        kf_model.P = kf_model.A @ kf_model.P @ kf_model.A.T + kf_model.Q

    return results


def combine_multiout_results(*results_list) -> dict[str, np.ndarray]:
    """
    Combine multiple multi-output result dictionaries.

    Creates a monotonic time axis to avoid fold-back when switching tasks.

    Returns:
        Combined dictionary with numpy arrays
    """
    combined = {
        "t": [],
        "xyz_true": [],
        "xyz_noisy": [],
        "xyz_pred_kf": [],
        "sigma_kf": [],
        "innovation": [],
        "trace_P": [],
    }

    # Track cumulative time offset to ensure monotonic axis
    time_offset = 0.0

    for i, results in enumerate(results_list):
        if i > 0:
            # Offset subsequent tasks so time continues monotonically
            t_array = np.array(results["t"])
            if len(t_array) > 1:
                dt = t_array[1] - t_array[0]
                time_offset += dt  # Add one time step gap
            results["t"] = [t + time_offset for t in results["t"]]
            # Update offset for next task
            if len(results["t"]) > 0:
                time_offset = results["t"][-1]

        for key in combined.keys():
            combined[key].extend(results[key])

    # Convert to numpy arrays
    return {
        "t": np.array(combined["t"]),
        "xyz_true": np.array(combined["xyz_true"]),  # (N, 3)
        "xyz_noisy": np.array(combined["xyz_noisy"]),  # (N, 3)
        "xyz_pred_kf": np.array(combined["xyz_pred_kf"]),  # (N, 3)
        "sigma_kf": np.array(combined["sigma_kf"]),  # (N, 3)
        "innovation": np.array(combined["innovation"]),  # (N, 3)
        "trace_P": np.array(combined["trace_P"]),  # (N,)
    }


# ==========================================
# MAIN EXPERIMENT
# ==========================================
if __name__ == "__main__":
    # ==========================================
    # CONFIGURATION
    # ==========================================
    print("Rössler System RNN+KF Experiment (Temporal Sequences)")

    # ==========================================
    # PHASE 1: PRE-TRAIN RNN BACKBONE
    # ==========================================
    print("\nPhase 1: Pre-training RNN on Task 0 (Rössler 3D trajectories)...")

    N_STEPS = 500  # Number of time steps
    DT = 0.01  # Time step size
    SEQ_LEN = 20  # Sequence length for RNN
    HIDDEN_SIZE = 32
    PRETRAIN_EPOCHS = 100

    # Build temporal datasets (reduced transient for faster execution)
    task_data = _build_task_data(n_steps=N_STEPS, dt=DT, n_transient=200)

    # Create RNN model (input is still 1D - we use x for temporal features)
    rnn_model = SimpleRNN(input_size=1, hidden_size=HIDDEN_SIZE)
    # Multi-output head: predict all 3 dimensions
    pretrain_head = nn.Linear(HIDDEN_SIZE, 3)
    params = list(rnn_model.parameters()) + list(pretrain_head.parameters())
    optimizer = optim.Adam(params, lr=0.005)
    criterion = nn.MSELoss()

    # Prepare training data (Task 0)
    task0 = task_data[0]
    time_values = task0["t"]

    # Use x dimension for input sequences (temporal pattern)
    input_values = task0["x_noisy"]

    # Target is all 3 dimensions (x, y, z)
    target_xyz = np.stack(
        [task0["x_noisy"], task0["y_noisy"], task0["z_noisy"]], axis=1
    )  # Shape: (N, 3)

    # Create training sequences
    inputs = []
    targets = []
    for i in range(len(input_values) - SEQ_LEN):
        # Input: sequence of x values (temporal pattern)
        seq_in = input_values[i : i + SEQ_LEN]
        # Target: [x, y, z] at next time step
        target_out = target_xyz[i + SEQ_LEN]

        inputs.append(seq_in.reshape(SEQ_LEN, 1))
        targets.append(target_out)

    inputs_torch = torch.tensor(np.array(inputs), dtype=torch.float32)
    targets_torch = torch.tensor(np.array(targets), dtype=torch.float32)  # Shape: (N, 3)

    print(f"Created {len(inputs)} training sequences")
    print(f"Input shape: {inputs_torch.shape}, Target shape: {targets_torch.shape}")

    # Training loop
    print("\nTraining RNN to predict [x, y, z](t+1) from x-sequence [x(t-{SEQ_LEN}), ..., x(t)]...")
    for epoch in range(PRETRAIN_EPOCHS):
        optimizer.zero_grad()
        features, _ = rnn_model(inputs_torch)
        preds = pretrain_head(features)  # Shape: (N, 3)
        loss = criterion(preds, targets_torch)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == PRETRAIN_EPOCHS - 1:
            print(f"Epoch {epoch:3d}: Loss {loss.item():.6f}")

    print("\nPre-training complete. Freezing RNN weights.")
    for param in rnn_model.parameters():
        param.requires_grad = False
    del pretrain_head

    # ==========================================
    # PHASE 2: ONLINE KALMAN FILTER ADAPTATION
    # ==========================================
    SWITCH_POINT = len(input_values) // 2

    # Kalman Filter hyperparameters
    KF_RHO = 1.0
    KF_Q_STD = 0.2
    KF_R_STD = 0.05

    # Initialize Multi-Output Kalman Filter Head (3D output)
    kf = MultiOutputKalmanFilterHead(
        feature_dim=HIDDEN_SIZE,
        output_dim=3,  # Predict x, y, z
        rho=KF_RHO,
        Q_std=KF_Q_STD,
        R_std=KF_R_STD,
    )

    # Prepare data as (N, 3) arrays
    t_values = task0["t"]
    xyz_clean_task0 = np.stack([task0["x"], task0["y"], task0["z"]], axis=1)  # (N, 3)
    xyz_noisy_task0 = np.stack(
        [task0["x_noisy"], task0["y_noisy"], task0["z_noisy"]], axis=1
    )  # (N, 3)

    # Task 1 data
    task1 = task_data[1]
    xyz_clean_task1 = np.stack([task1["x"], task1["y"], task1["z"]], axis=1)  # (N, 3)
    xyz_noisy_task1 = np.stack(
        [task1["x_noisy"], task1["y_noisy"], task1["z_noisy"]], axis=1
    )  # (N, 3)

    print("\n" + "=" * 60)
    print("PHASE 2: Online Kalman Filter Adaptation (3D Multi-Output)")
    print("=" * 60)

    print(f"\nStep 1: Training KF on Task 0 (first half, {SWITCH_POINT} points)")
    print(f"Task 0: {TASK_CONFIGS[0]['label']}")
    print(
        f"Parameters: a={TASK_CONFIGS[0]['a']}, b={TASK_CONFIGS[0]['b']}, c={TASK_CONFIGS[0]['c']}"
    )

    # Train KF on first half of Task 0
    results_train = run_multiout_kf_with_updates(
        kf, rnn_model, t_values, xyz_clean_task0, xyz_noisy_task0, 0, SWITCH_POINT, SEQ_LEN
    )

    print(f"\nStep 2: Testing KF on Task 1 (different dynamics)")
    print(f"Task 1: {TASK_CONFIGS[1]['label']}")
    print(
        f"Parameters: a={TASK_CONFIGS[1]['a']}, b={TASK_CONFIGS[1]['b']}, c={TASK_CONFIGS[1]['c']}"
    )

    # Test KF on Task 1 (prediction only)
    results_test = run_multiout_kf_prediction_only(
        kf,
        rnn_model,
        task1["t"],
        xyz_clean_task1,
        xyz_noisy_task1,
        0,
        len(xyz_clean_task1),
        SEQ_LEN,
    )

    # Combine results for plotting
    results = combine_multiout_results(results_train, results_test)

    # Diagnostic: Check shapes of all result arrays
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Result Array Shapes")
    print("=" * 60)
    for key in ["t", "xyz_true", "xyz_noisy", "xyz_pred_kf", "sigma_kf", "innovation", "trace_P"]:
        arr = np.array(results[key])
        print(f"{key:15s} shape: {str(arr.shape):20s} dtype: {arr.dtype}")
    print("=" * 60)

    # Extract data and create monotonic time axis
    print("\nPreparing data for visualization...")

    # 3D arrays: (N, 3) for [x, y, z]
    xyz_true = results["xyz_true"]  # (N, 3)
    xyz_noisy = results["xyz_noisy"]  # (N, 3)
    xyz_pred_kf = results["xyz_pred_kf"]  # (N, 3)
    sigma_kf_xyz = results["sigma_kf"]  # (N, 3) - uncertainty per dimension
    innovation_xyz = results["innovation"]  # (N, 3) - innovation per dimension
    trace_P = results["trace_P"]  # (N,)

    # Create monotonic time axis (index-based to avoid fold-back)
    # This is the robust solution: simple index * dt
    t_axis = np.arange(len(xyz_true)) * DT

    # Split points for task switch
    split_point = len(results_train["t"])

    print(f"Data shapes:")
    print(f"  t_axis:        {t_axis.shape}")
    print(f"  xyz_true:      {xyz_true.shape}")
    print(f"  xyz_pred_kf:   {xyz_pred_kf.shape}")
    print(f"  sigma_kf_xyz:  {sigma_kf_xyz.shape}")
    print(f"  Split point:   {split_point}")
    print("=" * 60)

    print(f"\nProcessing complete!")
    print(f"Total points processed: {len(t_axis)}")

    # ==========================================
    # ==========================================
    # PHASE 3: VISUALIZATION (3D Multi-Output)
    # ==========================================
    print("\n" + "=" * 60)
    print("PHASE 3: Visualization (3D Trajectories)")
    print("=" * 60)

    from mpl_toolkits.mplot3d import Axes3D

    # Create main figure with 3D plots
    fig = plt.figure(figsize=(20, 12))

    # =================================================================
    # ROW 1: 3D TRAJECTORY PLOTS
    # =================================================================

    # Plot 1: Ground Truth 3D Trajectory (Task 0 + Task 1)
    ax1 = fig.add_subplot(2, 4, 1, projection="3d")
    ax1.plot(
        xyz_true[:split_point, 0],
        xyz_true[:split_point, 1],
        xyz_true[:split_point, 2],
        linewidth=0.8,
        alpha=0.8,
        label="Task 0",
        color="blue",
    )
    ax1.plot(
        xyz_true[split_point:, 0],
        xyz_true[split_point:, 1],
        xyz_true[split_point:, 2],
        linewidth=0.8,
        alpha=0.8,
        label="Task 1",
        color="orange",
    )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("Ground Truth Trajectories")
    ax1.legend(fontsize=8)

    # Plot 2: KF Predicted 3D Trajectory
    ax2 = fig.add_subplot(2, 4, 2, projection="3d")
    ax2.plot(
        xyz_pred_kf[:split_point, 0],
        xyz_pred_kf[:split_point, 1],
        xyz_pred_kf[:split_point, 2],
        linewidth=0.8,
        alpha=0.8,
        label="Task 0",
        color="blue",
    )
    ax2.plot(
        xyz_pred_kf[split_point:, 0],
        xyz_pred_kf[split_point:, 1],
        xyz_pred_kf[split_point:, 2],
        linewidth=0.8,
        alpha=0.8,
        label="Task 1",
        color="orange",
    )
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("KF Predicted Trajectories")
    ax2.legend(fontsize=8)

    # Plot 3: Overlay - Ground Truth vs KF Prediction
    ax3 = fig.add_subplot(2, 4, 3, projection="3d")
    ax3.plot(
        xyz_true[:, 0],
        xyz_true[:, 1],
        xyz_true[:, 2],
        linewidth=0.6,
        alpha=0.5,
        label="Ground Truth",
        color="blue",
    )
    ax3.plot(
        xyz_pred_kf[:, 0],
        xyz_pred_kf[:, 1],
        xyz_pred_kf[:, 2],
        linewidth=0.6,
        alpha=0.7,
        label="KF Prediction",
        color="red",
        linestyle="--",
    )
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_title("Truth vs KF Prediction (Overlay)")
    ax3.legend(fontsize=8)

    # Plot 4: 3D Prediction Error Vectors (sampled)
    ax4 = fig.add_subplot(2, 4, 4, projection="3d")
    # Sample every 10th point to avoid clutter
    sample_idx = np.arange(0, len(xyz_true), 10)
    errors_3d = xyz_true - xyz_pred_kf
    ax4.scatter(
        xyz_true[sample_idx, 0],
        xyz_true[sample_idx, 1],
        xyz_true[sample_idx, 2],
        c="blue",
        s=5,
        alpha=0.3,
        label="True Position",
    )
    # Draw error vectors
    for idx in sample_idx[::5]:  # Further subsample for vectors
        ax4.plot(
            [xyz_true[idx, 0], xyz_pred_kf[idx, 0]],
            [xyz_true[idx, 1], xyz_pred_kf[idx, 1]],
            [xyz_true[idx, 2], xyz_pred_kf[idx, 2]],
            "r-",
            alpha=0.3,
            linewidth=0.5,
        )
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_zlabel("Z")
    ax4.set_title("Prediction Errors (Red Lines)")

    # =================================================================
    # ROW 2: PER-DIMENSION TIME SERIES
    # =================================================================

    # Plot 5: X Dimension - Time Series
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.plot(t_axis, xyz_true[:, 0], "b-", linewidth=1, alpha=0.7, label="Ground Truth")
    ax5.plot(t_axis, xyz_pred_kf[:, 0], "r--", linewidth=1, alpha=0.8, label="KF Prediction")
    ax5.axvline(
        x=t_axis[split_point], color="green", linestyle=":", linewidth=2, label="Task Switch"
    )
    ax5.set_xlabel("Time")
    ax5.set_ylabel("X")
    ax5.set_title("X Coordinate: Truth vs KF")
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Y Dimension - Time Series
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.plot(t_axis, xyz_true[:, 1], "b-", linewidth=1, alpha=0.7, label="Ground Truth")
    ax6.plot(t_axis, xyz_pred_kf[:, 1], "r--", linewidth=1, alpha=0.8, label="KF Prediction")
    ax6.axvline(
        x=t_axis[split_point], color="green", linestyle=":", linewidth=2, label="Task Switch"
    )
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Y")
    ax6.set_title("Y Coordinate: Truth vs KF")
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.3)

    # Plot 7: Z Dimension - Time Series
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.plot(t_axis, xyz_true[:, 2], "b-", linewidth=1, alpha=0.7, label="Ground Truth")
    ax7.plot(t_axis, xyz_pred_kf[:, 2], "r--", linewidth=1, alpha=0.8, label="KF Prediction")
    ax7.axvline(
        x=t_axis[split_point], color="green", linestyle=":", linewidth=2, label="Task Switch"
    )
    ax7.set_xlabel("Time")
    ax7.set_ylabel("Z")
    ax7.set_title("Z Coordinate: Truth vs KF")
    ax7.legend(fontsize=7)
    ax7.grid(True, alpha=0.3)

    # Plot 8: Summary Statistics
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis("off")

    # Compute per-dimension errors
    errors_3d = xyz_true - xyz_pred_kf
    error_x = errors_3d[:, 0]
    error_y = errors_3d[:, 1]
    error_z = errors_3d[:, 2]
    error_norm = np.linalg.norm(errors_3d, axis=1)

    # Task-specific statistics
    task0_error_norm = np.mean(error_norm[:split_point])
    task1_error_norm = np.mean(error_norm[split_point:])
    task0_rmse_x = np.sqrt(np.mean(error_x[:split_point] ** 2))
    task1_rmse_x = np.sqrt(np.mean(error_x[split_point:] ** 2))
    task0_rmse_y = np.sqrt(np.mean(error_y[:split_point] ** 2))
    task1_rmse_y = np.sqrt(np.mean(error_y[split_point:] ** 2))
    task0_rmse_z = np.sqrt(np.mean(error_z[:split_point] ** 2))
    task1_rmse_z = np.sqrt(np.mean(error_z[split_point:] ** 2))

    summary_text = f"""
3D MULTI-OUTPUT KF PERFORMANCE

Task 0 (Training):
  Mean ||Error||: {task0_error_norm:.4f}
  RMSE X: {task0_rmse_x:.4f}
  RMSE Y: {task0_rmse_y:.4f}
  RMSE Z: {task0_rmse_z:.4f}

Task 1 (Testing):
  Mean ||Error||: {task1_error_norm:.4f}
  RMSE X: {task1_rmse_x:.4f}
  RMSE Y: {task1_rmse_y:.4f}
  RMSE Z: {task1_rmse_z:.4f}

Error Increase: {(task1_error_norm / task0_error_norm - 1) * 100:.1f}%

Trace(P) Range: [{trace_P.min():.2e}, {trace_P.max():.2e}]
    """

    ax8.text(
        0.1,
        0.5,
        summary_text,
        transform=ax8.transAxes,
        fontsize=9,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        family="monospace",
    )

    plt.suptitle(
        f"Rössler 3D Multi-Output KF: Ground Truth vs Predictions", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    # Save figure
    os.makedirs("results/figures", exist_ok=True)
    save_path = f"results/figures/Rossler_3D_KF_rho{KF_RHO}_Q{KF_Q_STD:.2e}_R{KF_R_STD:.3f}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure 1 saved to: {save_path}")

    # =================================================================
    # FIGURE 2: UNCERTAINTY EVOLUTION (σ and 3σ over time)
    # =================================================================
    fig2 = plt.figure(figsize=(18, 10))

    dim_names = ["X", "Y", "Z"]
    dim_colors = ["red", "green", "blue"]

    for dim_idx in range(3):
        ax = fig2.add_subplot(3, 1, dim_idx + 1)

        # Extract uncertainty for this dimension
        sigma = sigma_kf_xyz[:, dim_idx]
        sigma_3 = 3 * sigma

        # Plot 1-sigma (uncertainty)
        ax.plot(
            t_axis,
            sigma,
            color=dim_colors[dim_idx],
            linewidth=2,
            alpha=0.8,
            label="σ (1-sigma)",
            zorder=2,
        )

        # Plot 3-sigma bounds
        ax.plot(
            t_axis,
            sigma_3,
            color=dim_colors[dim_idx],
            linewidth=1.5,
            alpha=0.6,
            label="3σ (99.7% confidence)",
            linestyle="--",
            zorder=1,
        )

        # Fill between sigma and 3*sigma
        ax.fill_between(t_axis, sigma, sigma_3, color=dim_colors[dim_idx], alpha=0.15, zorder=0)

        # Task switch line
        ax.axvline(
            x=t_axis[split_point],
            color="orange",
            linestyle=":",
            linewidth=2.5,
            label="Task Switch",
            alpha=0.8,
            zorder=3,
        )

        # Labels and formatting
        ax.set_xlabel("Time", fontsize=11)
        ax.set_ylabel("Uncertainty", fontsize=11)
        ax.set_title(
            f"{dim_names[dim_idx]} Dimension: Uncertainty Evolution (σ and 3σ)",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Compute statistics
        mean_sigma_task0 = np.mean(sigma[:split_point])
        mean_sigma_task1 = np.mean(sigma[split_point:])
        max_sigma_task0 = np.max(sigma[:split_point])
        max_sigma_task1 = np.max(sigma[split_point:])

        # Add text box with statistics
        stats_text = f"Task 0: mean σ={mean_sigma_task0:.4f}, max σ={max_sigma_task0:.4f}\n"
        stats_text += f"Task 1: mean σ={mean_sigma_task1:.4f}, max σ={max_sigma_task1:.4f}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.suptitle(
        "Predictive Uncertainty Evolution: σ and 3σ Bounds Over Time",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save uncertainty figure
    save_path2 = (
        f"results/figures/Rossler_3D_KF_Uncertainty_rho{KF_RHO}_Q{KF_Q_STD:.2e}_R{KF_R_STD:.3f}.png"
    )
    plt.savefig(save_path2, dpi=150, bbox_inches="tight")
    print(f"Figure 2 saved to: {save_path2}")

    plt.show()

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)
