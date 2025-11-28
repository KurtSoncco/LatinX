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
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Bessel_Ripple_KF import (
    SimpleRNN,
    run_kf_with_updates,
    run_kf_prediction_only,
    combine_results,
)
from latinx.data.rossler import RosslerTranslator
from latinx.models.kalman_filter import KalmanFilterHead


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
    print("\nPhase 1: Pre-training RNN on Task 0 (Rössler, standard chaotic)...")

    N_STEPS = 500  # Number of time steps
    DT = 0.01  # Time step size
    SEQ_LEN = 20  # Sequence length for RNN
    HIDDEN_SIZE = 32
    PRETRAIN_EPOCHS = 100
    TARGET_DIM = "x"  # Predict x dimension

    # Build temporal datasets (reduced transient for faster execution)
    task_data = _build_task_data(n_steps=N_STEPS, dt=DT, n_transient=200)

    # Create RNN model
    rnn_model = SimpleRNN(input_size=1, hidden_size=HIDDEN_SIZE)
    pretrain_head = nn.Linear(HIDDEN_SIZE, 1)
    params = list(rnn_model.parameters()) + list(pretrain_head.parameters())
    optimizer = optim.Adam(params, lr=0.005)
    criterion = nn.MSELoss()

    # Prepare training data (Task 0)
    task0 = task_data[0]
    time_values = task0["t"]
    target_values = task0[f"{TARGET_DIM}_noisy"]  # Use noisy x for training

    # Create training sequences
    inputs = []
    targets = []
    for i in range(len(target_values) - SEQ_LEN):
        # Input: sequence of target values at consecutive time steps
        seq_in = target_values[i : i + SEQ_LEN]
        # Target: target value at the next time step
        target_out = target_values[i + SEQ_LEN]

        inputs.append(seq_in.reshape(SEQ_LEN, 1))
        targets.append(target_out)

    inputs_torch = torch.tensor(np.array(inputs), dtype=torch.float32)
    targets_torch = torch.tensor(np.array(targets), dtype=torch.float32).view(-1, 1)

    print(f"Created {len(inputs)} training sequences")

    # Training loop
    print(f"\nTraining RNN to predict {TARGET_DIM}(t+1) from sequence [{TARGET_DIM}(t-{SEQ_LEN}), ..., {TARGET_DIM}(t)]...")
    for epoch in range(PRETRAIN_EPOCHS):
        optimizer.zero_grad()
        features, _ = rnn_model(inputs_torch)
        preds = pretrain_head(features)
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
    SWITCH_POINT = len(target_values) // 2

    # Kalman Filter hyperparameters
    KF_RHO = 1.0
    KF_Q_STD = 0.01
    KF_R_STD = 0.05

    # Initialize Kalman Filter Head
    kf = KalmanFilterHead(feature_dim=HIDDEN_SIZE, rho=KF_RHO, Q_std=KF_Q_STD, R_std=KF_R_STD)

    # Prepare data
    t_values = task0["t"]
    target_clean = task0[TARGET_DIM]
    target_noisy = task0[f"{TARGET_DIM}_noisy"]

    # Task 1 data
    task1 = task_data[1]
    target_clean_task1 = task1[TARGET_DIM]
    target_noisy_task1 = task1[f"{TARGET_DIM}_noisy"]

    print("\n" + "=" * 60)
    print("PHASE 2: Online Kalman Filter Adaptation")
    print("=" * 60)

    print(f"\nStep 1: Training KF on Task 0 (first half, {SWITCH_POINT} points)")
    print(f"Task 0: {TASK_CONFIGS[0]['label']}")
    print(f"Parameters: a={TASK_CONFIGS[0]['a']}, b={TASK_CONFIGS[0]['b']}, c={TASK_CONFIGS[0]['c']}")

    # Train KF on first half of Task 0
    results_train = run_kf_with_updates(
        kf, rnn_model, t_values, target_clean, target_noisy,
        0, SWITCH_POINT, SEQ_LEN
    )

    print(f"\nStep 2: Testing KF on Task 1 (different dynamics)")
    print(f"Task 1: {TASK_CONFIGS[1]['label']}")
    print(f"Parameters: a={TASK_CONFIGS[1]['a']}, b={TASK_CONFIGS[1]['b']}, c={TASK_CONFIGS[1]['c']}")

    # Test KF on Task 1 (prediction only)
    results_test = run_kf_prediction_only(
        kf, rnn_model, task1["t"], target_clean_task1, target_noisy_task1,
        0, len(target_clean_task1), SEQ_LEN
    )

    # Combine results for plotting
    results = combine_results(results_train, results_test)

    print(f"\nProcessing complete!")
    print(f"Total points processed: {len(results['r'])}")  # Note: 'r' used for time dimension

    # ==========================================
    # PHASE 3: VISUALIZATION
    # ==========================================
    print("\n" + "=" * 60)
    print("PHASE 3: Visualization")
    print("=" * 60)

    # =================================================================
    # FIGURE 1: SYSTEM DYNAMICS & TRAJECTORIES
    # =================================================================
    fig1 = plt.figure(figsize=(18, 10))

    # Task 0: 3D trajectory
    ax1 = fig1.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(task0["x"][:SWITCH_POINT], task0["y"][:SWITCH_POINT], task0["z"][:SWITCH_POINT],
             linewidth=0.5, alpha=0.8, label="Task 0")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title(f"Task 0: {TASK_CONFIGS[0]['label']}\n3D Attractor")
    ax1.legend()

    # Task 1: 3D trajectory
    ax2 = fig1.add_subplot(2, 3, 2, projection='3d')
    ax2.plot(task1["x"], task1["y"], task1["z"],
             linewidth=0.5, alpha=0.8, color='orange', label="Task 1")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title(f"Task 1: {TASK_CONFIGS[1]['label']}\n3D Attractor")
    ax2.legend()

    # X-Y Projection (Task 0 vs Task 1)
    ax3 = fig1.add_subplot(2, 3, 3)
    ax3.plot(task0["x"][:SWITCH_POINT], task0["y"][:SWITCH_POINT],
             linewidth=0.5, alpha=0.6, label="Task 0")
    ax3.plot(task1["x"], task1["y"],
             linewidth=0.5, alpha=0.6, color='orange', label="Task 1")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_title("X-Y Projection")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Task 0 vs Task 1: X time series
    ax4 = fig1.add_subplot(2, 3, 4)
    ax4.plot(task0["t"][:SWITCH_POINT], task0["x"][:SWITCH_POINT],
              linewidth=1, alpha=0.8, label="Task 0")
    ax4.plot(task1["t"], task1["x"],
              linewidth=1, alpha=0.8, color='orange', label="Task 1")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("X")
    ax4.set_title("X Coordinate: Task Comparison")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Y-Z Projection
    ax5 = fig1.add_subplot(2, 3, 5)
    ax5.plot(task0["y"][:SWITCH_POINT], task0["z"][:SWITCH_POINT],
              linewidth=0.5, alpha=0.6, label="Task 0")
    ax5.plot(task1["y"], task1["z"],
              linewidth=0.5, alpha=0.6, color='orange', label="Task 1")
    ax5.set_xlabel("Y")
    ax5.set_ylabel("Z")
    ax5.set_title("Y-Z Projection")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Task 0 vs Task 1: Y time series
    ax6 = fig1.add_subplot(2, 3, 6)
    ax6.plot(task0["t"][:SWITCH_POINT], task0["y"][:SWITCH_POINT],
              linewidth=1, alpha=0.8, label="Task 0")
    ax6.plot(task1["t"], task1["y"],
              linewidth=1, alpha=0.8, color='orange', label="Task 1")
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Y")
    ax6.set_title("Y Coordinate: Task Comparison")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.suptitle(f"Figure 1: Rössler System Dynamics - Task Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()

    # =================================================================
    # FIGURE 2: KALMAN FILTER PERFORMANCE
    # =================================================================
    fig2 = plt.figure(figsize=(18, 12))

    # Prediction vs Ground Truth (Full view)
    ax7 = fig2.add_subplot(3, 3, 1)
    ax7.plot(results["r"], results["z_true"], label="Ground Truth", linewidth=1, alpha=0.7, color='blue')
    ax7.plot(results["r"], results["z_pred_kf"], label="KF Prediction", linewidth=1, alpha=0.7, color='red')
    ax7.axvline(x=results["r"][len(results_train["r"])], color='green', linestyle='--',
                linewidth=2, label='Task Switch', alpha=0.7)
    ax7.set_xlabel("Time")
    ax7.set_ylabel(f"{TARGET_DIM.upper()} Value")
    ax7.set_title(f"{TARGET_DIM.upper()}: Full Timeline (Blue=Truth, Red=KF)")
    ax7.legend(fontsize=8, loc='upper right')
    ax7.grid(True, alpha=0.3)

    # Prediction Error
    ax8 = fig2.add_subplot(3, 3, 2)
    error = np.array(results["z_true"]) - np.array(results["z_pred_kf"])
    ax8.plot(results["r"], error, linewidth=0.8, alpha=0.8)
    ax8.axvline(x=results["r"][len(results_train["r"])], color='red', linestyle='--',
                linewidth=2, label='Task Switch', alpha=0.7)
    ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax8.set_xlabel("Time")
    ax8.set_ylabel("Prediction Error")
    ax8.set_title("Prediction Error Over Time")
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)

    # KF Uncertainty (σ)
    ax9 = fig2.add_subplot(3, 3, 3)
    ax9.plot(results["r"], results["sigma_kf"], linewidth=1.2, alpha=0.8, color='purple')
    ax9.axvline(x=results["r"][len(results_train["r"])], color='red', linestyle='--',
                linewidth=2, label='Task Switch', alpha=0.7)
    ax9.set_xlabel("Time")
    ax9.set_ylabel("Uncertainty (σ)")
    ax9.set_title("KF Predictive Uncertainty")
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)

    # Trace of P
    ax10 = fig2.add_subplot(3, 3, 4)
    ax10.plot(results["r"], results["trace_P"], linewidth=1.2, alpha=0.8, color='green')
    ax10.axvline(x=results["r"][len(results_train["r"])], color='red', linestyle='--',
                linewidth=2, label='Task Switch', alpha=0.7)
    ax10.set_xlabel("Time")
    ax10.set_ylabel("Trace(P)")
    ax10.set_title("KF Covariance Trace (Total Uncertainty)")
    ax10.legend(fontsize=8)
    ax10.grid(True, alpha=0.3)

    # Innovation (prediction error before update)
    ax11 = fig2.add_subplot(3, 3, 5)
    ax11.plot(results["r"], results["innovation"], linewidth=0.8, alpha=0.8)
    ax11.axvline(x=results["r"][len(results_train["r"])], color='red', linestyle='--',
                linewidth=2, label='Task Switch', alpha=0.7)
    ax11.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax11.set_xlabel("Time")
    ax11.set_ylabel("Innovation")
    ax11.set_title("KF Innovation (Prediction Error)")
    ax11.legend(fontsize=8)
    ax11.grid(True, alpha=0.3)

    # Innovation Squared
    ax12 = fig2.add_subplot(3, 3, 6)
    ax12.plot(results["r"], results["innovation_sq"], linewidth=0.8, alpha=0.8, color='orange')
    ax12.axvline(x=results["r"][len(results_train["r"])], color='red', linestyle='--',
                linewidth=2, label='Task Switch', alpha=0.7)
    ax12.set_xlabel("Time")
    ax12.set_ylabel("Innovation²")
    ax12.set_title("Squared Innovation (Detection Metric)")
    ax12.legend(fontsize=8)
    ax12.grid(True, alpha=0.3)

    # Zoomed view: Task switch region
    ax13 = fig2.add_subplot(3, 3, 7)
    switch_idx = len(results_train["r"])
    zoom_start = max(0, switch_idx - 50)
    zoom_end = min(len(results["r"]), switch_idx + 50)
    ax13.plot(results["r"][zoom_start:zoom_end], results["z_true"][zoom_start:zoom_end],
              label="Ground Truth", linewidth=1.5, alpha=0.8, color='blue')
    ax13.plot(results["r"][zoom_start:zoom_end], results["z_pred_kf"][zoom_start:zoom_end],
              label="KF Prediction", linewidth=1.5, alpha=0.8, color='red')
    ax13.axvline(x=results["r"][switch_idx], color='green', linestyle='--',
                linewidth=2, label='Task Switch')
    ax13.set_xlabel("Time")
    ax13.set_ylabel(f"{TARGET_DIM.upper()} Value")
    ax13.set_title("Zoomed View: Task Switch Region")
    ax13.legend(fontsize=8)
    ax13.grid(True, alpha=0.3)

    # Cumulative error
    ax14 = fig2.add_subplot(3, 3, 8)
    cumulative_error = np.cumsum(np.abs(error))
    ax14.plot(results["r"], cumulative_error, linewidth=1.2, alpha=0.8, color='darkred')
    ax14.axvline(x=results["r"][len(results_train["r"])], color='green', linestyle='--',
                linewidth=2, label='Task Switch', alpha=0.7)
    ax14.set_xlabel("Time")
    ax14.set_ylabel("Cumulative |Error|")
    ax14.set_title("Cumulative Absolute Error")
    ax14.legend(fontsize=8)
    ax14.grid(True, alpha=0.3)

    # Summary statistics
    ax15 = fig2.add_subplot(3, 3, 9)
    ax15.axis('off')
    split_point = len(results_train["r"])
    task0_innovation_mean = np.mean(np.abs(results["innovation"][:split_point]))
    task1_innovation_mean = np.mean(np.abs(results["innovation"][split_point:]))
    task0_uncertainty_mean = np.mean(results["sigma_kf"][:split_point])
    task1_uncertainty_mean = np.mean(results["sigma_kf"][split_point:])
    task0_error_mean = np.mean(np.abs(error[:split_point]))
    task1_error_mean = np.mean(np.abs(error[split_point:]))

    summary_text = f"""
KF PERFORMANCE METRICS

Task 0 (Training):
  Mean |Error|: {task0_error_mean:.4f}
  Mean |Innovation|: {task0_innovation_mean:.4f}
  Mean Uncertainty: {task0_uncertainty_mean:.4f}

Task 1 (Testing):
  Mean |Error|: {task1_error_mean:.4f}
  Mean |Innovation|: {task1_innovation_mean:.4f}
  Mean Uncertainty: {task1_uncertainty_mean:.4f}

Error Increase: {(task1_error_mean/task0_error_mean - 1)*100:.1f}%
    """
    ax15.text(0.1, 0.5, summary_text, transform=ax15.transAxes,
              fontsize=10, verticalalignment='center',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
              family='monospace')

    plt.suptitle(f"Figure 2: Kalman Filter Performance Metrics", fontsize=14, fontweight='bold')
    plt.tight_layout()

    # =================================================================
    # FIGURE 3: STATISTICAL ANALYSIS
    # =================================================================
    fig3 = plt.figure(figsize=(16, 8))

    # Histogram of innovations (Task 0 vs Task 1)
    ax16 = fig3.add_subplot(2, 3, 1)
    ax16.hist(results["innovation"][:split_point], bins=50, alpha=0.6,
              label="Task 0 (Training)", density=True)
    ax16.hist(results["innovation"][split_point:], bins=50, alpha=0.6,
              label="Task 1 (Testing)", density=True, color='orange')
    ax16.set_xlabel("Innovation")
    ax16.set_ylabel("Density")
    ax16.set_title("Innovation Distribution")
    ax16.legend(fontsize=8)
    ax16.grid(True, alpha=0.3)

    # Uncertainty comparison
    ax17 = fig3.add_subplot(2, 3, 2)
    ax17.hist(results["sigma_kf"][:split_point], bins=50, alpha=0.6,
              label="Task 0 (Training)", density=True, color='purple')
    ax17.hist(results["sigma_kf"][split_point:], bins=50, alpha=0.6,
              label="Task 1 (Testing)", density=True, color='orange')
    ax17.set_xlabel("Uncertainty (σ)")
    ax17.set_ylabel("Density")
    ax17.set_title("Uncertainty Distribution")
    ax17.legend(fontsize=8)
    ax17.grid(True, alpha=0.3)

    # Error distribution
    ax18 = fig3.add_subplot(2, 3, 3)
    ax18.hist(error[:split_point], bins=50, alpha=0.6,
              label="Task 0 (Training)", density=True, color='blue')
    ax18.hist(error[split_point:], bins=50, alpha=0.6,
              label="Task 1 (Testing)", density=True, color='orange')
    ax18.set_xlabel("Prediction Error")
    ax18.set_ylabel("Density")
    ax18.set_title("Error Distribution")
    ax18.legend(fontsize=8)
    ax18.grid(True, alpha=0.3)

    # X-Z Projection
    ax19 = fig3.add_subplot(2, 3, 4)
    ax19.plot(task0["x"][:SWITCH_POINT], task0["z"][:SWITCH_POINT],
              linewidth=0.5, alpha=0.6, label="Task 0")
    ax19.plot(task1["x"], task1["z"],
              linewidth=0.5, alpha=0.6, color='orange', label="Task 1")
    ax19.set_xlabel("X")
    ax19.set_ylabel("Z")
    ax19.set_title("X-Z Projection")
    ax19.legend(fontsize=8)
    ax19.grid(True, alpha=0.3)

    # Innovation vs Uncertainty scatter
    ax20 = fig3.add_subplot(2, 3, 5)
    ax20.scatter(results["sigma_kf"][:split_point], np.abs(results["innovation"][:split_point]),
                 alpha=0.3, s=10, label="Task 0")
    ax20.scatter(results["sigma_kf"][split_point:], np.abs(results["innovation"][split_point:]),
                 alpha=0.3, s=10, color='orange', label="Task 1")
    ax20.set_xlabel("Uncertainty (σ)")
    ax20.set_ylabel("|Innovation|")
    ax20.set_title("Uncertainty vs Innovation")
    ax20.legend(fontsize=8)
    ax20.grid(True, alpha=0.3)

    # Experiment summary
    ax21 = fig3.add_subplot(2, 3, 6)
    ax21.axis('off')

    # Compute statistics
    task0_innovation_mean = np.mean(np.abs(results["innovation"][:split_point]))
    task1_innovation_mean = np.mean(np.abs(results["innovation"][split_point:]))
    task0_uncertainty_mean = np.mean(results["sigma_kf"][:split_point])
    task1_uncertainty_mean = np.mean(results["sigma_kf"][split_point:])

    summary_text = f"""
EXPERIMENT SUMMARY

Kalman Filter Parameters:
  ρ (forgetting): {KF_RHO}
  Q (process noise): {KF_Q_STD:.6f}
  R (measurement noise): {KF_R_STD:.3f}

Task 0 ({TASK_CONFIGS[0]['label']}):
  Parameters: a={TASK_CONFIGS[0]['a']}, b={TASK_CONFIGS[0]['b']}, c={TASK_CONFIGS[0]['c']}
  Mean |Innovation|: {task0_innovation_mean:.4f}
  Mean Uncertainty: {task0_uncertainty_mean:.4f}

Task 1 ({TASK_CONFIGS[1]['label']}):
  Parameters: a={TASK_CONFIGS[1]['a']}, b={TASK_CONFIGS[1]['b']}, c={TASK_CONFIGS[1]['c']}
  Mean |Innovation|: {task1_innovation_mean:.4f}
  Mean Uncertainty: {task1_uncertainty_mean:.4f}

Target: {TARGET_DIM.upper()} coordinate
Sequence Length: {SEQ_LEN}
Hidden Size: {HIDDEN_SIZE}
    """

    ax16.text(0.1, 0.5, summary_text, transform=ax16.transAxes,
              fontsize=9, verticalalignment='center',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
              family='monospace')

    plt.suptitle(f"Rössler System: RNN+KF Experiment (Target={TARGET_DIM.upper()})",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure
    os.makedirs("results/figures", exist_ok=True)
    save_path = f"results/figures/Rossler_KF_rho{KF_RHO}_Q{KF_Q_STD:.2e}_R{KF_R_STD:.3f}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {save_path}")

    plt.show()

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)
