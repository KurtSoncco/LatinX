"""
Utility functions for Bayesian Last Layer (BLL) experiments.

This module provides shared functions for:
- Feature extraction from frozen RNN
- BLL training (batch mode)
- BLL prediction/testing
- BLL results visualization
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
import matplotlib.pyplot as plt


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

    Args:
        rnn_model: Frozen RNN model for feature extraction
        r_values: Radial coordinates
        z_clean: Clean ground truth values
        z_noisy: Noisy observations
        start_idx: Starting index
        end_idx: Ending index (exclusive)
        seq_len: Sequence length for RNN input

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


def run_bll_training(
    bll_model,
    rnn_model: nn.Module,
    r_values: np.ndarray,
    z_clean: np.ndarray,
    z_noisy: np.ndarray,
    start_idx: int,
    end_idx: int,
    seq_len: int,
    verbose: bool = True,
):
    """
    Train BLL with batch fitting (not incremental).

    Args:
        bll_model: StandaloneBayesianLastLayer instance
        rnn_model: Frozen RNN model for feature extraction
        r_values: Radial coordinates
        z_clean: Clean ground truth values
        z_noisy: Noisy observations
        start_idx: Starting index
        end_idx: Ending index (exclusive)
        seq_len: Sequence length for RNN input
        verbose: If True, print debug information

    Returns:
        Dictionary with keys: r, z_true, z_noisy, z_pred_bll, sigma_bll, innovation, innovation_sq, trace_P
    """
    # Step 1: Extract ALL features and targets first
    features, r_vals, z_true_vals, z_noisy_vals = extract_features_and_targets(
        rnn_model, r_values, z_clean, z_noisy, start_idx, end_idx, seq_len
    )

    if verbose:
        print(f"  Extracted {len(features)} feature vectors")
        print(f"  Feature dimension: {features.shape[1]}")

    # Step 2: Fit BLL ONCE on all training data
    if verbose:
        print(f"  Fitting BLL on {len(features)} samples (batch mode)...")
    bll_model.fit(features, z_noisy_vals)

    # Step 3: Make predictions with the fitted model
    z_pred_bll, sigma_bll = bll_model.predict(features, return_std=True)
    z_pred_bll = np.array(z_pred_bll)
    sigma_bll = np.array(sigma_bll)

    # Step 4: Compute residuals
    innovation = z_true_vals - z_pred_bll
    innovation_sq = innovation**2
    trace_P = np.full(len(r_vals), bll_model.get_total_uncertainty())

    # Step 5: Compute weight norm (constant in batch mode)
    import jax.numpy as jnp
    weight_norm_value = float(jnp.linalg.norm(bll_model.posterior_mean))
    weight_norm = np.full(len(r_vals), weight_norm_value)

    if verbose:
        rmse = np.sqrt(np.mean(innovation**2))
        print(f"  Training RMSE: {rmse:.4f}")

    return {
        "r": r_vals,
        "z_true": z_true_vals,
        "z_noisy": z_noisy_vals,
        "z_pred_bll": z_pred_bll,
        "sigma_bll": sigma_bll,
        "innovation": innovation,
        "innovation_sq": innovation_sq,
        "trace_P": trace_P,
        "weight_norm": weight_norm,
    }


def run_bll_training_incremental(
    bll_model,
    rnn_model: nn.Module,
    r_values: np.ndarray,
    z_clean: np.ndarray,
    z_noisy: np.ndarray,
    start_idx: int,
    end_idx: int,
    seq_len: int,
    verbose: bool = True,
):
    """
    Train BLL incrementally (refitting at each timestep).

    At each timestep:
    1. Extract features for current point
    2. Accumulate features and targets
    3. Refit BLL on accumulated data
    4. Make prediction with updated model

    This mimics online learning similar to Kalman Filter, but refits the entire
    posterior at each step instead of doing incremental Bayesian updates.

    Args:
        bll_model: StandaloneBayesianLastLayer instance
        rnn_model: Frozen RNN model for feature extraction
        r_values: Radial coordinates
        z_clean: Clean ground truth values
        z_noisy: Noisy observations
        start_idx: Starting index
        end_idx: Ending index (exclusive)
        seq_len: Sequence length for RNN input
        verbose: If True, print debug information

    Returns:
        Dictionary with keys: r, z_true, z_noisy, z_pred_bll, sigma_bll, innovation, innovation_sq, trace_P
    """
    results = {
        "r": [],
        "z_true": [],
        "z_noisy": [],
        "z_pred_bll": [],
        "sigma_bll": [],
        "innovation": [],
        "innovation_sq": [],
        "trace_P": [],
        "weight_norm": [],
    }

    # Accumulation buffers
    accumulated_features = []
    accumulated_targets = []

    # Ensure we don't exceed array bounds
    actual_end_idx = min(end_idx, len(r_values), len(z_clean), len(z_noisy))

    # Initialize input buffer with noisy observations
    input_buffer = deque(maxlen=seq_len)
    for i in range(seq_len):
        idx = start_idx + i
        if idx < actual_end_idx:
            input_buffer.append(z_noisy[idx])

    if verbose:
        print(f"  Starting incremental BLL training...")
        print(f"  Processing {actual_end_idx - start_idx - seq_len} timesteps")

    # Process each timestep incrementally
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

        # Accumulate features and targets
        accumulated_features.append(features)
        accumulated_targets.append(z_noisy_t)

        # Refit BLL on accumulated data
        features_array = np.array(accumulated_features)
        targets_array = np.array(accumulated_targets)
        bll_model.fit(features_array, targets_array)

        # Predict with updated model
        z_pred_bll, sigma_bll = bll_model.predict(features.reshape(1, -1), return_std=True)
        z_pred_bll = float(z_pred_bll[0])
        sigma_bll = float(sigma_bll[0])

        # Compute residuals
        innovation = z_true - z_pred_bll
        innovation_sq = innovation**2
        trace_P = bll_model.get_total_uncertainty()

        # Compute weight norm
        import jax.numpy as jnp
        weight_norm = float(jnp.linalg.norm(bll_model.posterior_mean))

        # Store results
        results["r"].append(r_t)
        results["z_true"].append(z_true)
        results["z_noisy"].append(z_noisy_t)
        results["z_pred_bll"].append(z_pred_bll)
        results["sigma_bll"].append(sigma_bll)
        results["innovation"].append(innovation)
        results["innovation_sq"].append(innovation_sq)
        results["trace_P"].append(trace_P)
        results["weight_norm"].append(weight_norm)

        # Update buffer
        input_buffer.append(z_noisy_t)

    # Convert to numpy arrays
    for key in results.keys():
        results[key] = np.array(results[key])

    if verbose:
        rmse = np.sqrt(np.mean(results["innovation"]**2))
        print(f"  Incremental training complete: {len(results['r'])} timesteps")
        print(f"  Final RMSE: {rmse:.4f}")

    return results


def run_bll_prediction_only(
    bll_model,
    rnn_model: nn.Module,
    r_values: np.ndarray,
    z_clean: np.ndarray,
    z_noisy: np.ndarray,
    start_idx: int,
    end_idx: int,
    seq_len: int,
    verbose: bool = True,
):
    """
    Make predictions using already-fitted BLL (no retraining).

    Args:
        bll_model: Already-fitted StandaloneBayesianLastLayer instance
        rnn_model: Frozen RNN model for feature extraction
        r_values: Radial coordinates
        z_clean: Clean ground truth values
        z_noisy: Noisy observations
        start_idx: Starting index
        end_idx: Ending index (exclusive)
        seq_len: Sequence length for RNN input
        verbose: If True, print debug information

    Returns:
        Dictionary with keys: r, z_true, z_noisy, z_pred_bll, sigma_bll, innovation, innovation_sq, trace_P
    """
    # Extract features
    features, r_vals, z_true_vals, z_noisy_vals = extract_features_and_targets(
        rnn_model, r_values, z_clean, z_noisy, start_idx, end_idx, seq_len
    )

    if verbose:
        print(f"  Extracted {len(features)} feature vectors for testing")

    # Predict with frozen BLL
    z_pred_bll, sigma_bll = bll_model.predict(features, return_std=True)
    z_pred_bll = np.array(z_pred_bll)
    sigma_bll = np.array(sigma_bll)

    # Compute residuals
    innovation = z_true_vals - z_pred_bll
    innovation_sq = innovation**2
    trace_P = np.full(len(r_vals), bll_model.get_total_uncertainty())

    # Compute weight norm (constant in prediction mode)
    import jax.numpy as jnp
    weight_norm_value = float(jnp.linalg.norm(bll_model.posterior_mean))
    weight_norm = np.full(len(r_vals), weight_norm_value)

    if verbose:
        rmse = np.sqrt(np.mean(innovation**2))
        print(f"  Testing RMSE: {rmse:.4f}")

    return {
        "r": r_vals,
        "z_true": z_true_vals,
        "z_noisy": z_noisy_vals,
        "z_pred_bll": z_pred_bll,
        "sigma_bll": sigma_bll,
        "innovation": innovation,
        "innovation_sq": innovation_sq,
        "trace_P": trace_P,
        "weight_norm": weight_norm,
    }


def combine_bll_results(*results_list):
    """
    Combine multiple BLL result dictionaries.

    Args:
        *results_list: Variable number of result dictionaries

    Returns:
        Combined dictionary with concatenated arrays
    """
    combined = {
        "r": [],
        "z_true": [],
        "z_noisy": [],
        "z_pred_bll": [],
        "sigma_bll": [],
        "innovation": [],
        "innovation_sq": [],
        "trace_P": [],
        "weight_norm": [],
    }

    for results in results_list:
        for key in combined.keys():
            combined[key].extend(results[key])

    return {key: np.array(val) for key, val in combined.items()}


def plot_bll_results(
    bll_results: dict,
    task_ids: np.ndarray,
    task_configs: dict,
    bll_sigma: float,
    bll_alpha: float,
    hidden_size: int,
    seq_len: int,
    pretrain_epochs: int,
    backward_mode: bool = True,
    save_path: str | None = None,
):
    """
    Create comprehensive BLL visualization.

    Args:
        bll_results: Dictionary with BLL results
        task_ids: Array indicating which task each sample belongs to (0 or 1)
        task_configs: Dictionary with task configurations
        bll_sigma: BLL sigma hyperparameter
        bll_alpha: BLL alpha hyperparameter
        hidden_size: RNN hidden size
        seq_len: Sequence length
        pretrain_epochs: Number of RNN training epochs
        backward_mode: If True, using backward extrapolation mode
        save_path: Optional path to save figure
    """
    # Extract results
    r_axis = bll_results["r"]
    z_true = bll_results["z_true"]
    z_noisy = bll_results["z_noisy"]
    z_pred_bll = bll_results["z_pred_bll"]
    sigma_bll = bll_results["sigma_bll"]
    innovation_sq = bll_results["innovation_sq"]
    trace_P = bll_results["trace_P"]

    # Find switch point
    if backward_mode:
        switch_idx = np.where(task_ids == 0)[0][0] if 0 in task_ids else len(r_axis)
    else:
        switch_idx = np.where(task_ids == 1)[0][0] if 1 in task_ids else len(r_axis)
    switch_r = r_axis[switch_idx] if switch_idx < len(r_axis) else r_axis[-1]

    # Create figure
    fig = plt.figure(figsize=(18, 12))

    # Plot 1: Radial Wave Profile
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title("Radial Wave Profile: BLL Predictions vs Truth", fontsize=12, fontweight="bold")
    ax1.plot(r_axis, z_true, "k-", label="Ground Truth (Clean)", alpha=0.7, linewidth=2)
    ax1.plot(r_axis, z_noisy, "gray", label="Noisy Observations", alpha=0.3, linewidth=1)
    ax1.plot(r_axis, z_pred_bll, "b-", label="BLL Prediction", linewidth=2)
    ax1.axvline(switch_r, color="orange", linestyle=":", linewidth=2, label="Train/Test Boundary")
    ax1.set_xlabel("Radius (r)")
    ax1.set_ylabel("z Value (Wave Height)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Add region labels
    if backward_mode:
        ax1.text(
            r_axis[len(r_axis) // 4],
            ax1.get_ylim()[1] * 0.85,
            f"Task 1 BACKWARD\n(inner radii)\nA={task_configs[1]['amplitude']}\nNot Trained",
            ha="center",
            fontsize=9,
            color="darkred",
            bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.3},
        )
        if switch_idx < len(r_axis):
            ax1.text(
                r_axis[switch_idx + (len(r_axis) - switch_idx) // 2],
                ax1.get_ylim()[1] * 0.85,
                f"Task 0 TRAINED\n(outer radii)\nA={task_configs[0]['amplitude']}\nBLL Trained",
                ha="center",
                fontsize=9,
                color="darkgreen",
                bbox={"boxstyle": "round", "facecolor": "lightgreen", "alpha": 0.3},
            )

    # Plot 2: Predictive Uncertainty
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title("BLL Predictive Uncertainty (σ) vs Radius", fontsize=12, fontweight="bold")
    ax2.plot(r_axis, sigma_bll, "purple", linewidth=2, label="BLL Uncertainty (σ)")
    ax2.fill_between(r_axis, 0, 3 * sigma_bll, color="purple", alpha=0.2, label="3σ Confidence")
    ax2.axvline(switch_r, color="orange", linestyle=":", linewidth=2)
    ax2.set_xlabel("Radius (r)")
    ax2.set_ylabel("Uncertainty (σ)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Absolute Error
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title("BLL Prediction Error vs Radius", fontsize=12, fontweight="bold")
    abs_error = np.abs(z_true - z_pred_bll)
    ax3.plot(r_axis, abs_error, "b-", label="BLL Absolute Error", linewidth=1.5, alpha=0.7)
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

    if backward_mode:
        region0_name = f"TASK 0 TRAINING (outer radii, A={task_configs[0]['amplitude']})"
        region1_name = f"TASK 1 BACKWARD (inner radii, A={task_configs[1]['amplitude']})"
    else:
        region0_name = f"TASK 0 (amplitude={task_configs[0]['amplitude']})"
        region1_name = f"TASK 1 (amplitude={task_configs[1]['amplitude']})"

    summary_text = f"""
    BLL SUMMARY STATISTICS
    {"=" * 40}

    {region0_name}:
    {"─" * 40}
    BLL RMSE:       {rmse_region0:.6f}
    Mean σ (BLL):   {np.mean(sigma_bll[region0_mask]):.6f}
    Mean Trace(P):  {np.mean(trace_P[region0_mask]):.6f}

    {region1_name}:
    {"─" * 40}
    BLL RMSE:       {rmse_region1:.6f}
    Mean σ (BLL):   {np.mean(sigma_bll[region1_mask]):.6f}
    Mean Trace(P):  {np.mean(trace_P[region1_mask]):.6f}

    CONFIGURATION:
    {"─" * 40}
    Total Points:   {len(r_axis)}
    Switch Radius:  {switch_r:.2f}
    RNN Hidden:     {hidden_size}
    Sequence Len:   {seq_len}
    BLL Sigma:      {bll_sigma:.3f}
    BLL Alpha:      {bll_alpha:.3f}
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

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nBLL Figure saved to {save_path}")

    plt.show()
