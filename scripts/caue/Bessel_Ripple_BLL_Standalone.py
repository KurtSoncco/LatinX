"""
Standalone RNN + Bayesian Last Layer (BLL) Experiment on Bessel-Ripple Dataset.

This script demonstrates:
1. Training RNN on outer radii (r > 4) of Task 0
2. Freezing RNN and using it as feature extractor
3. Training BLL (batch) on outer radii of Task 0
4. Testing BLL on inner radii (r < 4) of Task 1 (backward extrapolation + task switch)

The experiment tests spatial generalization (outer → inner radii) combined with
task switching (different amplitude and wavenumber).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import jax.numpy as jnp

from latinx.data.bessel_ripple import BesselRippleTranslator
from latinx.models.standalone_bayesian_last_layer import StandaloneBayesianLastLayer
from latinx.models.rnn import SimpleRNN


# ==========================================
# TASK CONFIGURATION
# ==========================================
TASK_CONFIGS = {
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


# ==========================================
# DATA GENERATION
# ==========================================
def build_task_data_radial(n_radial_points: int = 400, r_max: float = 8.0):
    """
    Build radial Bessel ripple datasets for two tasks.

    Args:
        n_radial_points: Number of radial points to sample
        r_max: Maximum radius

    Returns:
        Dict mapping task_id -> {"r", "z", "z_noisy"}
    """
    cache = {}

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

        # Generate 1D radial data
        df = translator.generate_radial(n_points=n_radial_points, r_max=r_max)

        cache[tid] = {
            "r": np.asarray(df["r"].values, dtype=np.float32),
            "z": np.asarray(df["z"].values, dtype=np.float32),
            "z_noisy": np.asarray(df["z_noisy"].values, dtype=np.float32),
        }

    return cache


# ==========================================
# FEATURE EXTRACTION
# ==========================================
def extract_features_and_targets(
    rnn_model: nn.Module,
    r_values: np.ndarray,
    z_clean: np.ndarray,
    z_noisy: np.ndarray,
    start_idx: int,
    end_idx: int,
    seq_len: int,
):
    """
    Extract features from frozen RNN for all data points in range.

    Returns:
        Tuple of (features, r_vals, z_true_vals, z_noisy_vals)
    """
    features_list = []
    r_list = []
    z_true_list = []
    z_noisy_list = []

    # Ensure we don't exceed array bounds
    actual_end_idx = min(end_idx, len(r_values), len(z_clean), len(z_noisy))

    # Initialize buffer with noisy observations
    input_buffer = deque(maxlen=seq_len)
    for i in range(seq_len):
        idx = start_idx + i
        if idx < actual_end_idx:
            input_buffer.append(z_noisy[idx])

    # Process all data points
    for t in range(start_idx + seq_len, actual_end_idx):
        r_t = r_values[t]
        z_true = z_clean[t]
        z_noisy_t = z_noisy[t]

        # Extract features using frozen RNN
        input_tensor = torch.tensor(np.array(input_buffer), dtype=torch.float32).view(1, seq_len, 1)

        with torch.no_grad():
            features_tensor, _ = rnn_model(input_tensor)

        # Convert to numpy
        features = features_tensor.detach().numpy().squeeze()
        features_list.append(features)
        r_list.append(r_t)
        z_true_list.append(z_true)
        z_noisy_list.append(z_noisy_t)

        # Update buffer
        input_buffer.append(z_noisy_t)

    return (
        np.array(features_list),
        np.array(r_list),
        np.array(z_true_list),
        np.array(z_noisy_list),
    )


# ==========================================
# MAIN EXPERIMENT
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("RNN + BLL EXPERIMENT: Bessel-Ripple Backward Extrapolation")
    print("=" * 70)

    # Configuration
    N_RADIAL_POINTS = 400
    SEQ_LEN = 10
    HIDDEN_SIZE = 32
    PRETRAIN_EPOCHS = 150
    R_MAX = 8.0

    # BLL hyperparameters (match actual noise level)
    BLL_SIGMA = 0.03
    BLL_ALPHA = 0.01

    # ==========================================
    # STEP 1: GENERATE DATA
    # ==========================================
    print("\nStep 1: Generating Bessel-Ripple data...")
    task_data = build_task_data_radial(n_radial_points=N_RADIAL_POINTS, r_max=R_MAX)

    task0 = task_data[0]
    task1 = task_data[1]

    r_values_task0 = task0["r"]
    z_clean_task0 = task0["z"]
    z_noisy_task0 = task0["z_noisy"]

    r_values_task1 = task1["r"]
    z_clean_task1 = task1["z"]
    z_noisy_task1 = task1["z_noisy"]

    SWITCH_POINT = len(r_values_task0) // 2

    print(f"  Task 0: amplitude={TASK_CONFIGS[0]['amplitude']}, k={TASK_CONFIGS[0]['k']}")
    print(f"  Task 1: amplitude={TASK_CONFIGS[1]['amplitude']}, k={TASK_CONFIGS[1]['k']}")
    print(f"  Total radial points: {N_RADIAL_POINTS}")
    print(f"  Switch point: {SWITCH_POINT} (r ≈ {r_values_task0[SWITCH_POINT]:.2f})")
    print(f"  Outer radii (training): r ∈ [{r_values_task0[SWITCH_POINT]:.2f}, {r_values_task0[-1]:.2f}]")
    print(f"  Inner radii (testing): r ∈ [{r_values_task1[0]:.2f}, {r_values_task1[SWITCH_POINT-1]:.2f}]")

    # ==========================================
    # STEP 2: TRAIN RNN ON OUTER RADII (TASK 0)
    # ==========================================
    print(f"\nStep 2: Training RNN on OUTER radii of Task 0...")
    print(f"  Training region: [{SWITCH_POINT}, {N_RADIAL_POINTS}] (outer radii)")

    # Create RNN model
    rnn_model = SimpleRNN(input_size=1, hidden_size=HIDDEN_SIZE)
    pretrain_head = nn.Linear(HIDDEN_SIZE, 1)
    params = list(rnn_model.parameters()) + list(pretrain_head.parameters())
    optimizer = optim.Adam(params, lr=0.005)
    criterion = nn.MSELoss()

    # Prepare training data (outer radii only)
    z_values_train = z_noisy_task0[SWITCH_POINT:]

    # Create training sequences
    inputs = []
    targets = []
    for i in range(len(z_values_train) - SEQ_LEN):
        seq_in = z_values_train[i : i + SEQ_LEN]
        target_out = z_values_train[i + SEQ_LEN]
        inputs.append(seq_in.reshape(SEQ_LEN, 1))
        targets.append(target_out)

    inputs_torch = torch.tensor(np.array(inputs), dtype=torch.float32)
    targets_torch = torch.tensor(np.array(targets), dtype=torch.float32).view(-1, 1)

    print(f"  Created {len(inputs)} training sequences")

    # Training loop
    for epoch in range(PRETRAIN_EPOCHS):
        optimizer.zero_grad()
        features, _ = rnn_model(inputs_torch)
        preds = pretrain_head(features)
        loss = criterion(preds, targets_torch)
        loss.backward()
        optimizer.step()

        if epoch % 30 == 0 or epoch == PRETRAIN_EPOCHS - 1:
            print(f"  Epoch {epoch:3d}: Loss {loss.item():.6f}")

    print("  RNN training complete! Freezing RNN weights...")

    # Freeze RNN
    for param in rnn_model.parameters():
        param.requires_grad = False
    rnn_model.eval()

    # ==========================================
    # STEP 3: TRAIN BLL ON OUTER RADII (TASK 0)
    # ==========================================
    print(f"\nStep 3: Training BLL on OUTER radii of Task 0...")

    # Initialize BLL
    bll = StandaloneBayesianLastLayer(sigma=BLL_SIGMA, alpha=BLL_ALPHA, feature_dim=HIDDEN_SIZE)

    # Extract features for training (outer radii, Task 0)
    print(f"  Extracting features from outer radii [{SWITCH_POINT}, {N_RADIAL_POINTS}]...")
    features_train, r_train, z_true_train, z_noisy_train = extract_features_and_targets(
        rnn_model, r_values_task0, z_clean_task0, z_noisy_task0, SWITCH_POINT, N_RADIAL_POINTS, SEQ_LEN
    )

    print(f"  Extracted {len(features_train)} feature vectors")
    print(f"  Feature dimension: {features_train.shape[1]}")

    # Fit BLL (batch fitting)
    print(f"  Fitting BLL on {len(features_train)} samples (batch mode)...")
    bll.fit(features_train, z_noisy_train)

    # Make predictions on training data
    z_pred_train, sigma_pred_train = bll.predict(features_train, return_std=True)
    z_pred_train = np.array(z_pred_train)
    sigma_pred_train = np.array(sigma_pred_train)

    train_rmse = np.sqrt(np.mean((z_true_train - z_pred_train) ** 2))
    print(f"  Training RMSE: {train_rmse:.4f}")

    # ==========================================
    # STEP 4: TEST BLL ON INNER RADII (TASK 1)
    # ==========================================
    print(f"\nStep 4: Testing BLL on INNER radii of Task 1 (backward extrapolation + task switch)...")
    print(f"  Test region: [0, {SWITCH_POINT}] (inner radii)")
    print(f"  Challenge: Different amplitude ({TASK_CONFIGS[1]['amplitude']}) + different k ({TASK_CONFIGS[1]['k']})")

    # Extract features for testing (inner radii, Task 1)
    features_test, r_test, z_true_test, z_noisy_test = extract_features_and_targets(
        rnn_model, r_values_task1, z_clean_task1, z_noisy_task1, 0, SWITCH_POINT, SEQ_LEN
    )

    print(f"  Extracted {len(features_test)} feature vectors for testing")

    # Make predictions with frozen BLL
    z_pred_test, sigma_pred_test = bll.predict(features_test, return_std=True)
    z_pred_test = np.array(z_pred_test)
    sigma_pred_test = np.array(sigma_pred_test)

    test_rmse = np.sqrt(np.mean((z_true_test - z_pred_test) ** 2))
    print(f"  Testing RMSE: {test_rmse:.4f}")

    # ==========================================
    # STEP 5: COMBINE RESULTS FOR VISUALIZATION
    # ==========================================
    print(f"\nStep 5: Combining results for visualization...")

    # Combine in order: test (inner radii, Task 1) then train (outer radii, Task 0)
    r_combined = np.concatenate([r_test, r_train])
    z_true_combined = np.concatenate([z_true_test, z_true_train])
    z_noisy_combined = np.concatenate([z_noisy_test, z_noisy_train])
    z_pred_combined = np.concatenate([z_pred_test, z_pred_train])
    sigma_combined = np.concatenate([sigma_pred_test, sigma_pred_train])
    task_ids = np.concatenate([np.ones(len(r_test)), np.zeros(len(r_train))])

    # Find switch point in combined data
    switch_idx = np.where(task_ids == 0)[0][0] if 0 in task_ids else len(r_combined)
    switch_r = r_combined[switch_idx] if switch_idx < len(r_combined) else r_combined[-1]

    print(f"  Combined {len(r_combined)} data points")
    print(f"  Switch at index {switch_idx}, r = {switch_r:.2f}")

    # ==========================================
    # STEP 6: VISUALIZATION
    # ==========================================
    print(f"\nStep 6: Creating visualizations...")

    fig = plt.figure(figsize=(18, 12))

    # Plot 1: Radial Wave Profile
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title("Radial Wave Profile: BLL Predictions vs Truth", fontsize=12, fontweight="bold")
    ax1.plot(r_combined, z_true_combined, "k-", label="Ground Truth (Clean)", alpha=0.7, linewidth=2)
    ax1.plot(r_combined, z_noisy_combined, "gray", label="Noisy Observations", alpha=0.3, linewidth=1)
    ax1.plot(r_combined, z_pred_combined, "b-", label="BLL Prediction", linewidth=2)
    ax1.axvline(switch_r, color="orange", linestyle=":", linewidth=2, label="Train/Test Boundary")
    ax1.set_xlabel("Radius (r)")
    ax1.set_ylabel("z Value (Wave Height)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Add region labels
    ax1.text(
        r_combined[len(r_combined) // 4],
        ax1.get_ylim()[1] * 0.85,
        f"Task 1 BACKWARD\n(inner radii)\nA={TASK_CONFIGS[1]['amplitude']}\nNot Trained",
        ha="center",
        fontsize=9,
        color="darkred",
        bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.3},
    )
    if switch_idx < len(r_combined):
        ax1.text(
            r_combined[switch_idx + (len(r_combined) - switch_idx) // 2],
            ax1.get_ylim()[1] * 0.85,
            f"Task 0 TRAINED\n(outer radii)\nA={TASK_CONFIGS[0]['amplitude']}\nBLL Trained",
            ha="center",
            fontsize=9,
            color="darkgreen",
            bbox={"boxstyle": "round", "facecolor": "lightgreen", "alpha": 0.3},
        )

    # Plot 2: Predictive Uncertainty
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title("BLL Predictive Uncertainty (σ) vs Radius", fontsize=12, fontweight="bold")
    ax2.plot(r_combined, sigma_combined, "purple", linewidth=2, label="BLL Uncertainty (σ)")
    ax2.fill_between(r_combined, 0, 3 * sigma_combined, color="purple", alpha=0.2, label="3σ Confidence")
    ax2.axvline(switch_r, color="orange", linestyle=":", linewidth=2)
    ax2.set_xlabel("Radius (r)")
    ax2.set_ylabel("Uncertainty (σ)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Absolute Error
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title("BLL Prediction Error vs Radius", fontsize=12, fontweight="bold")
    abs_error = np.abs(z_true_combined - z_pred_combined)
    ax3.plot(r_combined, abs_error, "b-", label="BLL Absolute Error", linewidth=1.5, alpha=0.7)
    ax3.axvline(switch_r, color="orange", linestyle=":", linewidth=2)
    ax3.set_xlabel("Radius (r)")
    ax3.set_ylabel("Absolute Error")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary Statistics
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis("off")

    region0_mask = task_ids == 0
    region1_mask = task_ids == 1

    rmse_region0 = np.sqrt(np.mean(abs_error[region0_mask] ** 2))
    rmse_region1 = np.sqrt(np.mean(abs_error[region1_mask] ** 2)) if region1_mask.any() else 0

    summary_text = f"""
    SUMMARY STATISTICS
    {"=" * 40}

    Training Region (Task 0, Outer Radii):
      RMSE: {rmse_region0:.4f}
      Mean |Error|: {np.mean(abs_error[region0_mask]):.4f}
      Max |Error|: {np.max(abs_error[region0_mask]):.4f}
      Mean σ: {np.mean(sigma_combined[region0_mask]):.4f}
      Samples: {np.sum(region0_mask)}

    Testing Region (Task 1, Inner Radii):
      RMSE: {rmse_region1:.4f}
      Mean |Error|: {np.mean(abs_error[region1_mask]):.4f}
      Max |Error|: {np.max(abs_error[region1_mask]):.4f}
      Mean σ: {np.mean(sigma_combined[region1_mask]):.4f}
      Samples: {np.sum(region1_mask)}

    BLL Hyperparameters:
      σ (noise std): {BLL_SIGMA}
      α (prior precision): {BLL_ALPHA}

    Model Architecture:
      RNN Hidden Size: {HIDDEN_SIZE}
      Sequence Length: {SEQ_LEN}
      RNN Training Epochs: {PRETRAIN_EPOCHS}
    """

    ax4.text(
        0.1,
        0.95,
        summary_text,
        transform=ax4.transAxes,
        fontsize=9,
        verticalalignment="top",
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.suptitle(
        "RNN + BLL: Bessel-Ripple Backward Extrapolation with Task Switching",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("Experiment complete!")
    print("=" * 70)
