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

from latinx.data.bessel_ripple import BesselRippleTranslator
from latinx.models.standalone_bayesian_last_layer import StandaloneBayesianLastLayer
from latinx.models.rnn import SimpleRNN
from latinx.models.bll_utils import (
    run_bll_training,
    run_bll_prediction_only,
    combine_bll_results,
    plot_bll_results,
)


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
    print(f"  Training region: [{SWITCH_POINT}, {N_RADIAL_POINTS}] (outer radii)")

    # Initialize BLL
    bll = StandaloneBayesianLastLayer(sigma=BLL_SIGMA, alpha=BLL_ALPHA, feature_dim=HIDDEN_SIZE)

    # Train BLL using shared utility function
    bll_results_train = run_bll_training(
        bll, rnn_model, r_values_task0, z_clean_task0, z_noisy_task0,
        SWITCH_POINT, N_RADIAL_POINTS, SEQ_LEN, verbose=True
    )

    # ==========================================
    # STEP 4: TEST BLL ON INNER RADII (TASK 1)
    # ==========================================
    print(f"\nStep 4: Testing BLL on INNER radii of Task 1 (backward extrapolation + task switch)...")
    print(f"  Test region: [0, {SWITCH_POINT}] (inner radii)")
    print(f"  Challenge: Different amplitude ({TASK_CONFIGS[1]['amplitude']}) + different k ({TASK_CONFIGS[1]['k']})")

    # Test BLL using shared utility function
    bll_results_test = run_bll_prediction_only(
        bll, rnn_model, r_values_task1, z_clean_task1, z_noisy_task1,
        0, SWITCH_POINT, SEQ_LEN, verbose=True
    )

    # ==========================================
    # STEP 5: COMBINE RESULTS FOR VISUALIZATION
    # ==========================================
    print(f"\nStep 5: Combining results for visualization...")

    # Combine in order: test (inner radii, Task 1) then train (outer radii, Task 0)
    bll_results = combine_bll_results(bll_results_test, bll_results_train)
    task_ids = np.concatenate([
        np.ones(len(bll_results_test["r"])),
        np.zeros(len(bll_results_train["r"]))
    ])

    print(f"  Combined {len(bll_results['r'])} data points")

    # ==========================================
    # STEP 6: VISUALIZATION
    # ==========================================
    print(f"\nStep 6: Creating visualizations...")

    # Use shared utility function for plotting
    plot_bll_results(
        bll_results=bll_results,
        task_ids=task_ids,
        task_configs=TASK_CONFIGS,
        bll_sigma=BLL_SIGMA,
        bll_alpha=BLL_ALPHA,
        hidden_size=HIDDEN_SIZE,
        seq_len=SEQ_LEN,
        pretrain_epochs=PRETRAIN_EPOCHS,
        backward_mode=True,
        save_path=None,
    )

    print("\n" + "=" * 70)
    print("Experiment complete!")
    print("=" * 70)
