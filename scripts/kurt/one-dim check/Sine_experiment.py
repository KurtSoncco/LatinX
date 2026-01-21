from __future__ import annotations

from collections import deque
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from latinx.data.sine_cosine import SineCosineTranslator
from latinx.models.kalman_filter import KalmanFilterHead
from latinx.models.rnn_jax import (
    SimpleRNN,
    create_prediction_head,
    create_rnn,
    create_train_step,
)
from latinx.models.standalone_bayesian_last_layer import StandaloneBayesianLastLayer

# Task configuration: sinx -> cosx, sin0.5x -> cos0.5x, sin2x -> cos2x
TASK_CONFIGS: dict[int, dict[str, float]] = {
    0: {"amplitude": 1.0, "angle_multiplier": 1.0},  # sinx -> cosx
    1: {"amplitude": 1.0, "angle_multiplier": 1.0},  # sin0.5x -> cos0.5x
    2: {"amplitude": 1.0, "angle_multiplier": 2.0},  # sin2x -> cos2x
}


def _build_task_translators(
    num_samples: int, t_start: float = 0.0, dt: float = 0.05
) -> dict[int, dict[str, np.ndarray]]:
    """
    Build per-task translators with continuous time axis.

    Args:
        num_samples: Number of samples to generate for each task.
        t_start: Starting time value (default 0.0). Use this to continue time from a previous call.
        dt: Time step between samples (default 0.05).

    Returns:
        dict mapping task_id -> {"t": np.ndarray, "sine": np.ndarray, "cosine": np.ndarray}.
        All tasks share the same continuous time axis, only amplitude/angle_multiplier differ.
    """
    cache: dict[int, dict[str, np.ndarray]] = {}
    for tid, cfg in TASK_CONFIGS.items():
        translator = SineCosineTranslator(
            amplitude=cfg["amplitude"],
            angle_multiplier=cfg["angle_multiplier"],
            num_samples=num_samples,
            t_start=t_start,  # Pass continuous time start
        )
        df = translator.generate(dt=dt)
        cache[tid] = {
            "t": np.asarray(df["t"].values, dtype=np.float64),
            "sine": np.asarray(df["sine"].values, dtype=np.float64),
            "cosine": np.asarray(df["cosine"].values, dtype=np.float64),
        }
    return cache


def train_jax_rnn_on_task0(
    data_cache: dict[int, dict[str, np.ndarray]],
    seq_len: int = 10,
    hidden_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[SimpleRNN, dict]:
    """
    Train JAX RNN on Task 0 data (sine -> cosine mapping).
    """
    if verbose:
        print("Training JAX RNN on Task 0 (sinx -> cosx)...")

    # Extract Task 0 data
    task0_data = data_cache[0]
    sine_data = task0_data["sine"]
    cosine_data = task0_data["cosine"]

    if verbose:
        print(f"Task 0 data: {len(sine_data)} samples")

    # Create JAX RNN and Prediction Head
    rnn_model, rnn_params = create_rnn(input_size=1, hidden_size=hidden_size, seed=seed)
    pred_head, head_params = create_prediction_head(
        hidden_size=hidden_size, output_dim=1, seed=seed
    )

    if verbose:
        print(f"Created RNN (hidden_size={hidden_size}) and prediction head")

    # Setup Optimizers
    rnn_optimizer = optax.adam(learning_rate)
    head_optimizer = optax.adam(learning_rate)
    rnn_opt_state = rnn_optimizer.init(rnn_params)
    head_opt_state = head_optimizer.init(head_params)

    # Create Training Step
    train_step = create_train_step(rnn_model, pred_head, rnn_optimizer, head_optimizer)

    # Prepare Training Data (Sliding Windows)
    train_sequences = []
    train_targets = []

    for i in range(len(sine_data) - seq_len):
        seq_in = sine_data[i : i + seq_len]
        target_out = cosine_data[i + seq_len - 1]
        train_sequences.append(seq_in)
        train_targets.append(target_out)

    train_sequences = np.array(train_sequences)
    train_targets = np.array(train_targets)

    num_samples = len(train_sequences)
    if verbose:
        print(f"Created {num_samples} training sequences (sliding window)")

    # Training Loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        indices = np.random.permutation(num_samples)

        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_indices = indices[batch_start:batch_end]

            batch_sequences = train_sequences[batch_indices]
            batch_targets = train_targets[batch_indices]

            x_batch = jnp.array(batch_sequences[:, :, None])
            y_batch = jnp.array(batch_targets)

            rnn_params, head_params, rnn_opt_state, head_opt_state, loss = train_step(
                rnn_params, head_params, rnn_opt_state, head_opt_state, x_batch, y_batch
            )

            epoch_loss += float(loss)
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch}/{num_epochs}: Loss = {avg_loss:.6f}")

    if verbose:
        print("JAX RNN training complete!")

    return rnn_model, rnn_params


def pretrain_kf_on_task0(
    rnn_model,
    rnn_params: dict,
    kf: KalmanFilterHead,
    task0_data: dict[str, np.ndarray],
    seq_len: int,
    verbose: bool = True,
) -> tuple[dict, list]:
    """
    Pre-train Kalman Filter on Task 0 data using the frozen RNN.

    Returns:
        Tuple of (history, final_buffer) where:
            - history: Dictionary with training history
            - final_buffer: List of last seq_len values for initializing next task
    """
    num_samples = len(task0_data["sine"])

    if verbose:
        print(f"Pre-training KF on {num_samples} samples from Task 0...")

    kf_buffer = deque(np.zeros(seq_len), maxlen=seq_len)

    # Track metrics
    history = {
        "predictions": [],
        "ground_truth": [],
        "uncertainties": [],
        "covariance_norms": [],
        "innovation": [],
        "innovation_covariance": [],
        "kalman_gain_norms": [],
        "phi_norms": [],  # Track feature vector norms
    }

    for t in range(num_samples):
        x_t = float(task0_data["sine"][t])
        y_t = float(task0_data["cosine"][t])
        kf_buffer.append(x_t)

        input_array = jnp.array(np.array(kf_buffer)).reshape(1, seq_len, 1)
        features_array, _ = rnn_model.apply(rnn_params, input_array)
        phi = features_array.T

        y_pred = kf.predict(phi)
        _, pred_std = kf.get_prediction_uncertainty()
        kf.update(y_t, y_pred)

        # Track metrics
        history["predictions"].append(y_pred)
        history["ground_truth"].append(y_t)
        history["uncertainties"].append(pred_std)
        history["covariance_norms"].append(float(jnp.linalg.norm(kf.P)))
        history["innovation"].append(float(kf.innovation) if kf.innovation is not None else 0.0)
        # innovation_variance is a (1,1) array, extract scalar
        if kf.innovation_variance is not None:
            innovation_cov_val = float(kf.innovation_variance[0, 0])
        else:
            innovation_cov_val = 0.0
        history["innovation_covariance"].append(innovation_cov_val)
        history["kalman_gain_norms"].append(
            float(jnp.linalg.norm(kf.K)) if kf.K is not None else 0.0
        )
        # Track phi norm (features vector norm)
        history["phi_norms"].append(float(jnp.linalg.norm(phi)))

    if verbose:
        print(f"KF pre-training complete on {num_samples} samples.")

    # Return history and final buffer state for continuity to next task
    final_buffer = list(kf_buffer)
    return history, final_buffer


def pretrain_bll_on_task0(
    rnn_model,
    rnn_params: dict,
    bll: StandaloneBayesianLastLayer,
    task0_data: dict[str, np.ndarray],
    seq_len: int,
    verbose: bool = True,
) -> dict:
    """
    Pre-train Bayesian Last Layer on Task 0 data using the frozen RNN.

    Returns:
        Dictionary with training history including predictions, uncertainties,
        and covariance norms (after fitting).
    """
    num_samples = len(task0_data["sine"])

    if verbose:
        print(f"Pre-training BLL on {num_samples} samples from Task 0...")

    features_list = []
    targets_list = []
    bll_buffer = deque(np.zeros(seq_len), maxlen=seq_len)

    # Track predictions during training (incremental)
    history = {
        "predictions": [],
        "ground_truth": [],
        "uncertainties": [],
        "covariance_norms": [],
    }

    # Fit frequency: fit every N steps to track progress without being too slow
    FIT_FREQUENCY = 10  # Fit every 10 steps instead of every step

    for t in range(num_samples):
        x_t = float(task0_data["sine"][t])
        y_t = float(task0_data["cosine"][t])
        bll_buffer.append(x_t)

        input_array = jnp.array(np.array(bll_buffer)).reshape(1, seq_len, 1)
        features_array, _ = rnn_model.apply(rnn_params, input_array)

        features_list.append(features_array[0])  # Store as 1D array (feature_dim,)
        targets_list.append(y_t)

        # Incremental fit and predict (only every FIT_FREQUENCY steps to save time)
        should_fit = (t + 1) % FIT_FREQUENCY == 0 or t == num_samples - 1

        if should_fit and len(features_list) > 0:
            features_array_temp = jnp.array(features_list)
            targets_array_temp = jnp.array(targets_list)
            bll.fit(features_array_temp, targets_array_temp)

        # Predict with current model (if fitted) - features_array is already (1, feature_dim) from rnn_model.apply
        if bll._is_fitted:
            pred, pred_std = bll.predict(features_array, return_std=True)
            history["predictions"].append(float(pred.item()))
            history["ground_truth"].append(y_t)
            history["uncertainties"].append(float(pred_std.item()))

            # Get covariance norm if available
            if hasattr(bll, "posterior_covariance") and bll.posterior_covariance is not None:
                cov_norm = float(jnp.linalg.norm(bll.posterior_covariance))
            else:
                cov_norm = 0.0
            history["covariance_norms"].append(cov_norm)
        else:
            # Not fitted yet, use placeholder values
            history["predictions"].append(0.0)
            history["ground_truth"].append(y_t)
            history["uncertainties"].append(float("inf"))
            history["covariance_norms"].append(0.0)

    # Final fit on all data to ensure model is fully trained
    features_array = jnp.array(features_list)
    targets_array = jnp.array(targets_list)
    bll.fit(features_array, targets_array)

    if verbose:
        print(f"BLL pre-training complete on {len(features_list)} samples.")

    return history


def evaluate_on_tasks(
    rnn_model,
    rnn_params: dict,
    kf: KalmanFilterHead,
    bll: StandaloneBayesianLastLayer,
    eval_cache: dict[int, dict[str, np.ndarray]],
    tasks: list[int],
    seq_len: int,
    initial_buffer: list | None = None,
    verbose: bool = True,
) -> dict[int, dict]:
    """
    Evaluate KF and BLL on specified tasks (prediction only, no updates).

    Args:
        initial_buffer: Optional list of seq_len values to initialize first task's buffer
                       (for continuity from previous task/training)

    Returns:
        Dictionary mapping task_id -> {
            "kf_predictions": list,
            "bll_predictions": list,
            "kf_uncertainties": list,
            "bll_uncertainties": list,
            "ground_truth": list,
            "kf_covariance_norms": list,
            "bll_covariance_norms": list,
            "kf_innovation": list,
            "kf_innovation_covariance": list,
            "kf_kalman_gain_norms": list,
        }
    """
    eval_results = {
        task_id: {
            "kf_predictions": [],
            "bll_predictions": [],
            "kf_uncertainties": [],
            "bll_uncertainties": [],
            "ground_truth": [],
            "kf_covariance_norms": [],
            "bll_covariance_norms": [],
            "kf_innovation": [],
            "kf_innovation_covariance": [],
            "kf_kalman_gain_norms": [],
            "phi_norms": [],  # Track feature vector norms
        }
        for task_id in tasks
    }

    # Keep track of buffer between tasks for continuity
    task_buffer = None

    for task_idx, task in enumerate(tasks):
        if verbose:
            print(f"\nEvaluating on Task {task}...")

        data = eval_cache[task]
        num_eval_samples = len(data["sine"])

        # Initialize buffer based on task position
        if task_idx == 0 and initial_buffer is not None:
            # First task: use provided initial buffer from training
            eval_buffer = deque(initial_buffer, maxlen=seq_len)
        elif task_idx > 0 and task_buffer is not None:
            # Subsequent tasks: use buffer from previous task
            eval_buffer = deque(task_buffer, maxlen=seq_len)
        else:
            # Fallback: start with zeros
            eval_buffer = deque(np.zeros(seq_len), maxlen=seq_len)

        for t in range(num_eval_samples):
            x_t = float(data["sine"][t])
            y_t = float(data["cosine"][t])
            eval_buffer.append(x_t)

            input_array = jnp.array(np.array(eval_buffer)).reshape(1, seq_len, 1)
            features_array, _ = rnn_model.apply(rnn_params, input_array)

            phi_kf = features_array.T
            phi_bll = features_array

            # KF prediction (no update)
            kf_pred = kf.predict(phi_kf)
            _, kf_uncertainty = kf.get_prediction_uncertainty()

            # Compute innovation (prediction error) without updating
            kf_innovation = y_t - kf_pred
            # Innovation covariance from predicted state
            if kf.H is not None and kf.P_minus is not None:
                kf_innovation_cov = float((kf.H @ kf.P_minus @ kf.H.T + kf.R)[0, 0])
            else:
                kf_innovation_cov = 0.0

            # BLL prediction
            bll_pred, bll_uncertainty = bll.predict(phi_bll, return_std=True)
            bll_pred = float(bll_pred.item())
            bll_uncertainty = float(bll_uncertainty.item())

            # Get BLL covariance norm if available
            if hasattr(bll, "posterior_covariance") and bll.posterior_covariance is not None:
                bll_cov_norm = float(jnp.linalg.norm(bll.posterior_covariance))
            else:
                bll_cov_norm = 0.0

            eval_results[task]["kf_predictions"].append(kf_pred)
            eval_results[task]["bll_predictions"].append(bll_pred)
            eval_results[task]["kf_uncertainties"].append(kf_uncertainty)
            eval_results[task]["bll_uncertainties"].append(bll_uncertainty)
            eval_results[task]["ground_truth"].append(y_t)
            eval_results[task]["kf_covariance_norms"].append(float(jnp.linalg.norm(kf.P)))
            eval_results[task]["bll_covariance_norms"].append(bll_cov_norm)
            eval_results[task]["kf_innovation"].append(kf_innovation)
            eval_results[task]["kf_innovation_covariance"].append(kf_innovation_cov)
            eval_results[task]["kf_kalman_gain_norms"].append(
                float(jnp.linalg.norm(kf.K)) if kf.K is not None else 0.0
            )
            # Track phi norm
            eval_results[task]["phi_norms"].append(float(jnp.linalg.norm(phi_kf)))

        if verbose:
            kf_mean = np.mean(eval_results[task]["kf_predictions"])
            bll_mean = np.mean(eval_results[task]["bll_predictions"])
            kf_std = np.std(eval_results[task]["kf_predictions"])
            bll_std = np.std(eval_results[task]["bll_predictions"])
            print(f"  Task {task} - KF: mean={kf_mean:.6f}, std={kf_std:.6f}")
            print(f"  Task {task} - BLL: mean={bll_mean:.6f}, std={bll_std:.6f}")

        # Save last buffer of the current task for next one
        task_buffer = list(eval_buffer)

    return eval_results


def compute_comparison_metrics(eval_results: dict[int, dict], tasks: list[int]) -> dict:
    """
    Compute comparison metrics for mean prediction and standard deviation.

    Returns:
        Dictionary with metrics for each task and method
    """
    comparison = {}
    for task in tasks:
        kf_preds = np.array(eval_results[task]["kf_predictions"])
        bll_preds = np.array(eval_results[task]["bll_predictions"])
        ground_truth = np.array(eval_results[task]["ground_truth"])

        comparison[task] = {
            "kf": {
                "mean_prediction": np.mean(kf_preds),
                "std_prediction": np.std(kf_preds),
                "mean_uncertainty": np.mean(eval_results[task]["kf_uncertainties"]),
                "mean_error": np.mean(np.abs(ground_truth - kf_preds)),
            },
            "bll": {
                "mean_prediction": np.mean(bll_preds),
                "std_prediction": np.std(bll_preds),
                "mean_uncertainty": np.mean(eval_results[task]["bll_uncertainties"]),
                "mean_error": np.mean(np.abs(ground_truth - bll_preds)),
            },
        }
    return comparison


def plot_comparison_results(
    comparison_results: dict[float, dict],
    q_values: list[float],
    save_path: str = "results/figures/BLL_KF_Q_Comparison.png",
) -> None:
    """
    Plot comparison of BLL and KF results across different Q values.
    """
    # Create save directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "BLL vs KF Comparison: Mean Prediction and Std Dev Across Q Values",
        fontsize=14,
        fontweight="bold",
    )

    task_labels = ["Task 1 (sin0.5x->cos0.5x)", "Task 2 (sin2x->cos2x)"]

    # Extract data for plotting
    kf_means_task1 = [comparison_results[q][1]["kf"]["mean_prediction"] for q in q_values]
    bll_means_task1 = [comparison_results[q][1]["bll"]["mean_prediction"] for q in q_values]
    kf_stds_task1 = [comparison_results[q][1]["kf"]["std_prediction"] for q in q_values]
    bll_stds_task1 = [comparison_results[q][1]["bll"]["std_prediction"] for q in q_values]

    kf_means_task2 = [comparison_results[q][2]["kf"]["mean_prediction"] for q in q_values]
    bll_means_task2 = [comparison_results[q][2]["bll"]["mean_prediction"] for q in q_values]
    kf_stds_task2 = [comparison_results[q][2]["kf"]["std_prediction"] for q in q_values]
    bll_stds_task2 = [comparison_results[q][2]["bll"]["std_prediction"] for q in q_values]

    # Plot 1: Mean Prediction - Task 1
    ax1 = axes[0, 0]
    ax1.plot(q_values, kf_means_task1, "r-o", label="KF", linewidth=2, markersize=6)
    ax1.plot(q_values, bll_means_task1, "b-s", label="BLL", linewidth=2, markersize=6)
    ax1.set_xlabel("Q (Process Noise)")
    ax1.set_ylabel("Mean Prediction")
    ax1.set_title(f"{task_labels[0]}: Mean Prediction")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Std Dev - Task 1
    ax2 = axes[0, 1]
    ax2.plot(q_values, kf_stds_task1, "r-o", label="KF", linewidth=2, markersize=6)
    ax2.plot(q_values, bll_stds_task1, "b-s", label="BLL", linewidth=2, markersize=6)
    ax2.set_xlabel("Q (Process Noise)")
    ax2.set_ylabel("Std Dev of Predictions")
    ax2.set_title(f"{task_labels[0]}: Std Dev")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Mean Prediction - Task 2
    ax3 = axes[1, 0]
    ax3.plot(q_values, kf_means_task2, "r-o", label="KF", linewidth=2, markersize=6)
    ax3.plot(q_values, bll_means_task2, "b-s", label="BLL", linewidth=2, markersize=6)
    ax3.set_xlabel("Q (Process Noise)")
    ax3.set_ylabel("Mean Prediction")
    ax3.set_title(f"{task_labels[1]}: Mean Prediction")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Std Dev - Task 2
    ax4 = axes[1, 1]
    ax4.plot(q_values, kf_stds_task2, "r-o", label="KF", linewidth=2, markersize=6)
    ax4.plot(q_values, bll_stds_task2, "b-s", label="BLL", linewidth=2, markersize=6)
    ax4.set_xlabel("Q (Process Noise)")
    ax4.set_ylabel("Std Dev of Predictions")
    ax4.set_title(f"{task_labels[1]}: Std Dev")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Comparison plot saved to: {save_path}")


def plot_detailed_metrics_all_q(
    all_q_data: dict[float, dict],
    q_values: list[float],
    save_dir: str = "results/figures",
) -> None:
    """
    Plot detailed metrics comparing all Q values:
    - Predictions with +/- 1 std dev across all steps (training + evaluation)
    - Norm of covariance matrices
    - Single prediction plot and single std plot for each case
    - Innovation and innovation covariance history (KF only)
    - Kalman gain history (KF only)
    """
    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Get reference dimensions from first Q value
    first_q = q_values[0]
    ref_data = all_q_data[first_q]
    train_steps = len(ref_data["kf_train_history"]["predictions"])
    eval_steps_task1 = len(ref_data["kf_eval_results"][1]["kf_predictions"])
    eval_steps_task2 = len(ref_data["kf_eval_results"][2]["kf_predictions"])
    total_steps = train_steps + eval_steps_task1 + eval_steps_task2

    time_axis = np.arange(total_steps)
    task_boundaries = [train_steps, train_steps + eval_steps_task1]

    # Color map for different Q values
    import matplotlib.cm as cm

    colors = cm.get_cmap("viridis")(np.linspace(0, 1, len(q_values)))

    # ==========================================
    # Plot 1: KF Predictions with ±1σ (all Q values) + BLL for comparison
    # ==========================================
    fig1, ax1 = plt.subplots(1, 1, figsize=(16, 6))
    fig1.suptitle(
        "KF Predictions (All Q Values) vs BLL with ±1σ Uncertainty", fontsize=14, fontweight="bold"
    )

    # Plot ground truth once
    ref_gt = (
        ref_data["kf_train_history"]["ground_truth"]
        + ref_data["kf_eval_results"][1]["ground_truth"]
        + ref_data["kf_eval_results"][2]["ground_truth"]
    )
    ax1.plot(time_axis, ref_gt, "k-", label="Ground Truth", alpha=0.7, linewidth=2)

    # Plot BLL once (same for all Q values)
    bll_preds = (
        ref_data["bll_train_history"]["predictions"]
        + ref_data["kf_eval_results"][1]["bll_predictions"]
        + ref_data["kf_eval_results"][2]["bll_predictions"]
    )
    bll_unc = (
        ref_data["bll_train_history"]["uncertainties"]
        + ref_data["kf_eval_results"][1]["bll_uncertainties"]
        + ref_data["kf_eval_results"][2]["bll_uncertainties"]
    )
    ax1.plot(time_axis, bll_preds, "b:", label="BLL", alpha=0.8, linewidth=2)
    bll_upper = np.array(bll_preds) + np.array(bll_unc)
    bll_lower = np.array(bll_preds) - np.array(bll_unc)
    ax1.fill_between(time_axis, bll_lower, bll_upper, color="blue", alpha=0.15, label="BLL ±1σ")

    # Plot predictions for each Q value
    for i, q_val in enumerate(q_values):
        data = all_q_data[q_val]
        kf_preds = (
            data["kf_train_history"]["predictions"]
            + data["kf_eval_results"][1]["kf_predictions"]
            + data["kf_eval_results"][2]["kf_predictions"]
        )
        kf_unc = (
            data["kf_train_history"]["uncertainties"]
            + data["kf_eval_results"][1]["kf_uncertainties"]
            + data["kf_eval_results"][2]["kf_uncertainties"]
        )
        ax1.plot(
            time_axis,
            kf_preds,
            "--",
            label=f"KF Q={q_val}",
            color=colors[i],
            alpha=0.7,
            linewidth=1,
        )
        # Show uncertainty bounds for a few Q values to avoid clutter
        if i % 2 == 0:  # Every other Q value
            upper = np.array(kf_preds) + np.array(kf_unc)
            lower = np.array(kf_preds) - np.array(kf_unc)
            ax1.fill_between(time_axis, lower, upper, color=colors[i], alpha=0.1)

    for boundary in task_boundaries:
        ax1.axvline(boundary, color="blue", linestyle=":", alpha=0.5, linewidth=2)

    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Prediction Value")
    ax1.legend(loc="upper right", ncol=2, fontsize=8)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/KF_Predictions_All_Q.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ==========================================
    # Plot 2: BLL Predictions with ±1σ (single, same for all Q)
    # ==========================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(16, 6))
    fig2.suptitle(
        "BLL Predictions with ±1σ Uncertainty (Same for All Q Values)",
        fontsize=14,
        fontweight="bold",
    )

    ax2.plot(time_axis, ref_gt, "k-", label="Ground Truth", alpha=0.7, linewidth=2)

    # BLL is the same for all Q values, plot once
    bll_preds = (
        ref_data["bll_train_history"]["predictions"]
        + ref_data["kf_eval_results"][1]["bll_predictions"]
        + ref_data["kf_eval_results"][2]["bll_predictions"]
    )
    bll_unc = (
        ref_data["bll_train_history"]["uncertainties"]
        + ref_data["kf_eval_results"][1]["bll_uncertainties"]
        + ref_data["kf_eval_results"][2]["bll_uncertainties"]
    )
    ax2.plot(time_axis, bll_preds, "b:", label="BLL", alpha=0.8, linewidth=2)
    upper = np.array(bll_preds) + np.array(bll_unc)
    lower = np.array(bll_preds) - np.array(bll_unc)
    ax2.fill_between(time_axis, lower, upper, color="blue", alpha=0.2, label="BLL ±1σ")

    for boundary in task_boundaries:
        ax2.axvline(boundary, color="blue", linestyle=":", alpha=0.5, linewidth=2)

    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Prediction Value")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/BLL_Predictions_All_Q.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ==========================================
    # Plot 3: Covariance Matrix Norms (all Q values)
    # ==========================================
    fig3, ax3 = plt.subplots(1, 1, figsize=(16, 6))
    fig3.suptitle("Covariance Matrix Norms (All Q Values)", fontsize=14, fontweight="bold")

    # Plot BLL once (same for all Q)
    bll_cov_norms = (
        ref_data["bll_train_history"]["covariance_norms"]
        + ref_data["kf_eval_results"][1]["bll_covariance_norms"]
        + ref_data["kf_eval_results"][2]["bll_covariance_norms"]
    )
    ax3.plot(time_axis, bll_cov_norms, "b--", label="BLL", alpha=0.8, linewidth=2)

    # Plot KF for each Q value
    for i, q_val in enumerate(q_values):
        data = all_q_data[q_val]
        kf_cov_norms = (
            data["kf_train_history"]["covariance_norms"]
            + data["kf_eval_results"][1]["kf_covariance_norms"]
            + data["kf_eval_results"][2]["kf_covariance_norms"]
        )
        ax3.plot(
            time_axis,
            kf_cov_norms,
            "-",
            label=f"KF Q={q_val}",
            color=colors[i],
            alpha=0.7,
            linewidth=1,
        )

    for boundary in task_boundaries:
        ax3.axvline(boundary, color="blue", linestyle=":", alpha=0.5, linewidth=2)

    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Covariance Matrix Norm")
    ax3.legend(loc="upper right", ncol=2, fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale("log")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/Covariance_Norms_All_Q.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ==========================================
    # Plot 4: KF Uncertainty (Std Dev) - all Q values
    # ==========================================
    fig4, ax4 = plt.subplots(1, 1, figsize=(16, 6))
    fig4.suptitle("KF Uncertainty (Std Dev) - All Q Values", fontsize=14, fontweight="bold")

    for i, q_val in enumerate(q_values):
        data = all_q_data[q_val]
        kf_unc = (
            data["kf_train_history"]["uncertainties"]
            + data["kf_eval_results"][1]["kf_uncertainties"]
            + data["kf_eval_results"][2]["kf_uncertainties"]
        )
        ax4.plot(
            time_axis, kf_unc, "-", label=f"KF Q={q_val}", color=colors[i], alpha=0.7, linewidth=1
        )

    for boundary in task_boundaries:
        ax4.axvline(boundary, color="blue", linestyle=":", alpha=0.5, linewidth=2)

    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("Standard Deviation")
    ax4.legend(loc="upper right", ncol=2, fontsize=8)
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/KF_Uncertainty_All_Q.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ==========================================
    # Plot 5: BLL Uncertainty (Std Dev) - single (same for all Q)
    # ==========================================
    fig5, ax5 = plt.subplots(1, 1, figsize=(16, 6))
    fig5.suptitle(
        "BLL Uncertainty (Std Dev) - Same for All Q Values", fontsize=14, fontweight="bold"
    )

    # BLL is the same for all Q values, plot once
    bll_unc = (
        ref_data["bll_train_history"]["uncertainties"]
        + ref_data["kf_eval_results"][1]["bll_uncertainties"]
        + ref_data["kf_eval_results"][2]["bll_uncertainties"]
    )
    ax5.plot(time_axis, bll_unc, "b-", label="BLL", alpha=0.8, linewidth=2)

    for boundary in task_boundaries:
        ax5.axvline(boundary, color="blue", linestyle=":", alpha=0.5, linewidth=2)

    ax5.set_xlabel("Time Step")
    ax5.set_ylabel("Standard Deviation")
    ax5.legend(loc="upper right")
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/BLL_Uncertainty_All_Q.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ==========================================
    # Plot 6: KF Innovation (all Q values)
    # ==========================================
    fig6, ax6 = plt.subplots(1, 1, figsize=(16, 6))
    fig6.suptitle("KF Innovation (All Q Values)", fontsize=14, fontweight="bold")

    for i, q_val in enumerate(q_values):
        data = all_q_data[q_val]
        kf_innovation = (
            data["kf_train_history"]["innovation"]
            + data["kf_eval_results"][1]["kf_innovation"]
            + data["kf_eval_results"][2]["kf_innovation"]
        )
        ax6.plot(
            time_axis,
            kf_innovation,
            "-",
            label=f"KF Q={q_val}",
            color=colors[i],
            alpha=0.7,
            linewidth=1,
        )

    ax6.axhline(0, color="k", linestyle="--", alpha=0.3)
    for boundary in task_boundaries:
        ax6.axvline(boundary, color="blue", linestyle=":", alpha=0.5, linewidth=2)

    ax6.set_xlabel("Time Step")
    ax6.set_ylabel("Innovation (y_true - y_pred)")
    ax6.legend(loc="upper right", ncol=2, fontsize=8)
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/KF_Innovation_All_Q.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ==========================================
    # Plot 7: KF Innovation Covariance (all Q values)
    # ==========================================
    fig7, ax7 = plt.subplots(1, 1, figsize=(16, 6))
    fig7.suptitle("KF Innovation Covariance (All Q Values)", fontsize=14, fontweight="bold")

    for i, q_val in enumerate(q_values):
        data = all_q_data[q_val]
        kf_innovation_cov = (
            data["kf_train_history"]["innovation_covariance"]
            + data["kf_eval_results"][1]["kf_innovation_covariance"]
            + data["kf_eval_results"][2]["kf_innovation_covariance"]
        )
        ax7.plot(
            time_axis,
            kf_innovation_cov,
            "-",
            label=f"KF Q={q_val}",
            color=colors[i],
            alpha=0.7,
            linewidth=1,
        )

    for boundary in task_boundaries:
        ax7.axvline(boundary, color="blue", linestyle=":", alpha=0.5, linewidth=2)

    ax7.set_xlabel("Time Step")
    ax7.set_ylabel("Innovation Covariance")
    ax7.legend(loc="upper right", ncol=2, fontsize=8)
    ax7.grid(True, alpha=0.3)
    ax7.set_yscale("log")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/KF_Innovation_Cov_All_Q.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ==========================================
    # Plot 8: KF Kalman Gain Norm (all Q values)
    # ==========================================
    fig8, ax8 = plt.subplots(1, 1, figsize=(16, 6))
    fig8.suptitle("KF Kalman Gain Norm (All Q Values)", fontsize=14, fontweight="bold")

    for i, q_val in enumerate(q_values):
        data = all_q_data[q_val]
        kf_kalman_gain = (
            data["kf_train_history"]["kalman_gain_norms"]
            + data["kf_eval_results"][1]["kf_kalman_gain_norms"]
            + data["kf_eval_results"][2]["kf_kalman_gain_norms"]
        )
        ax8.plot(
            time_axis,
            kf_kalman_gain,
            "-",
            label=f"KF Q={q_val}",
            color=colors[i],
            alpha=0.7,
            linewidth=1,
        )

    for boundary in task_boundaries:
        ax8.axvline(boundary, color="blue", linestyle=":", alpha=0.5, linewidth=2)

    ax8.set_xlabel("Time Step")
    ax8.set_ylabel("Kalman Gain Norm")
    ax8.legend(loc="upper right", ncol=2, fontsize=8)
    ax8.grid(True, alpha=0.3)
    ax8.set_yscale("log")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/KF_Kalman_Gain_All_Q.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nAll summary plots saved to: {save_dir}/")


def plot_uncertainty_vs_phi_norm(
    all_q_data: dict[float, dict],
    q_values: list[float],
    save_dir: str = "results/figures",
) -> None:
    """
    Plot KF uncertainty (±1σ band) and phi_x norm over time for all Q values.

    Creates a 2-panel plot:
    - Top: Uncertainty bands showing predictions ± 1σ for each Q value
    - Bottom: Phi (feature vector) norm over time for each Q value

    Args:
        all_q_data: Dictionary containing all Q value data
        q_values: List of Q values tested
        save_dir: Directory to save plots
    """
    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Get reference dimensions
    first_q = q_values[0]
    ref_data = all_q_data[first_q]
    train_steps = len(ref_data["kf_train_history"]["predictions"])
    eval_steps_task1 = len(ref_data["kf_eval_results"][1]["kf_predictions"])
    eval_steps_task2 = len(ref_data["kf_eval_results"][2]["kf_predictions"])
    total_steps = train_steps + eval_steps_task1 + eval_steps_task2

    time_axis = np.arange(total_steps)
    task_boundaries = [train_steps, train_steps + eval_steps_task1]

    # Color map for different Q values
    import matplotlib.cm as cm
    colors = cm.get_cmap("viridis")(np.linspace(0, 1, len(q_values)))

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(
        "KF Uncertainty and Feature Norm vs Time (All Q Values)",
        fontsize=14,
        fontweight="bold",
    )

    # Get ground truth (same for all Q)
    ref_gt = (
        ref_data["kf_train_history"]["ground_truth"]
        + ref_data["kf_eval_results"][1]["ground_truth"]
        + ref_data["kf_eval_results"][2]["ground_truth"]
    )

    # ==========================================
    # Panel 1: Predictions with ±1σ uncertainty bands
    # ==========================================
    ax1.plot(time_axis, ref_gt, "k-", label="Ground Truth", alpha=0.7, linewidth=2)

    for i, q_val in enumerate(q_values):
        data = all_q_data[q_val]

        # Concatenate training and evaluation data
        kf_preds = (
            data["kf_train_history"]["predictions"]
            + data["kf_eval_results"][1]["kf_predictions"]
            + data["kf_eval_results"][2]["kf_predictions"]
        )
        kf_unc = (
            data["kf_train_history"]["uncertainties"]
            + data["kf_eval_results"][1]["kf_uncertainties"]
            + data["kf_eval_results"][2]["kf_uncertainties"]
        )

        kf_preds = np.array(kf_preds)
        kf_unc = np.array(kf_unc)

        # Plot prediction line
        ax1.plot(
            time_axis,
            kf_preds,
            "--",
            label=f"KF Q={q_val}",
            color=colors[i],
            alpha=0.7,
            linewidth=1.5,
        )

        # Plot ±1σ uncertainty band
        upper = kf_preds + kf_unc
        lower = kf_preds - kf_unc
        ax1.fill_between(time_axis, lower, upper, color=colors[i], alpha=0.15)

    # Add task boundaries
    for boundary in task_boundaries:
        ax1.axvline(boundary, color="blue", linestyle=":", alpha=0.5, linewidth=2)

    # Add task labels
    ax1.text(
        train_steps / 2,
        ax1.get_ylim()[1] * 0.95,
        "Task 0\n(Training)",
        ha="center",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.7},
    )
    ax1.text(
        train_steps + eval_steps_task1 / 2,
        ax1.get_ylim()[1] * 0.95,
        "Task 1\n(Eval)",
        ha="center",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "lightgreen", "alpha": 0.7},
    )
    ax1.text(
        train_steps + eval_steps_task1 + eval_steps_task2 / 2,
        ax1.get_ylim()[1] * 0.95,
        "Task 2\n(Eval)",
        ha="center",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.7},
    )

    ax1.set_ylabel("Prediction Value")
    ax1.legend(loc="upper right", ncol=3, fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("KF Predictions with ±1σ Uncertainty Bands")

    # ==========================================
    # Panel 2: Phi (feature vector) norm over time
    # ==========================================
    for i, q_val in enumerate(q_values):
        data = all_q_data[q_val]

        # Concatenate phi norms
        phi_norms = (
            data["kf_train_history"]["phi_norms"]
            + data["kf_eval_results"][1]["phi_norms"]
            + data["kf_eval_results"][2]["phi_norms"]
        )

        ax2.plot(
            time_axis,
            phi_norms,
            "-",
            label=f"Q={q_val}",
            color=colors[i],
            alpha=0.7,
            linewidth=1.5,
        )

    # Add task boundaries
    for boundary in task_boundaries:
        ax2.axvline(boundary, color="blue", linestyle=":", alpha=0.5, linewidth=2)

    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("||φ(x)||₂ (Feature Norm)")
    ax2.legend(loc="upper right", ncol=3, fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Feature Vector Norm (φ) Over Time")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/KF_Uncertainty_vs_Phi_Norm_All_Q.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Uncertainty vs Phi norm plot saved to: {save_dir}/KF_Uncertainty_vs_Phi_Norm_All_Q.png")


def plot_detailed_metrics(
    kf_train_history: dict,
    bll_train_history: dict,
    kf_eval_results: dict[int, dict],
    bll_eval_results: dict[int, dict],
    q_val: float,
    save_dir: str = "results/figures",
) -> None:
    """
    Plot detailed metrics including:
    - Predictions with +/- 1 std dev across all steps (training + evaluation)
    - Norm of covariance matrices
    - Single prediction plot and single std plot for each case
    - Innovation and innovation covariance history (KF only)
    - Kalman gain history (KF only)
    """
    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Combine training and evaluation data
    train_steps = len(kf_train_history["predictions"])
    eval_steps_task1 = len(kf_eval_results[1]["kf_predictions"])
    eval_steps_task2 = len(kf_eval_results[2]["kf_predictions"])

    total_steps = train_steps + eval_steps_task1 + eval_steps_task2

    # Create time axis
    time_axis = np.arange(total_steps)
    task_boundaries = [train_steps, train_steps + eval_steps_task1]

    # Combine all data
    kf_all_preds = (
        kf_train_history["predictions"]
        + kf_eval_results[1]["kf_predictions"]
        + kf_eval_results[2]["kf_predictions"]
    )
    kf_all_uncertainties = (
        kf_train_history["uncertainties"]
        + kf_eval_results[1]["kf_uncertainties"]
        + kf_eval_results[2]["kf_uncertainties"]
    )
    kf_all_gt = (
        kf_train_history["ground_truth"]
        + kf_eval_results[1]["ground_truth"]
        + kf_eval_results[2]["ground_truth"]
    )
    kf_all_cov_norms = (
        kf_train_history["covariance_norms"]
        + kf_eval_results[1]["kf_covariance_norms"]
        + kf_eval_results[2]["kf_covariance_norms"]
    )
    kf_all_innovation = (
        kf_train_history["innovation"]
        + kf_eval_results[1]["kf_innovation"]
        + kf_eval_results[2]["kf_innovation"]
    )
    kf_all_innovation_cov = (
        kf_train_history["innovation_covariance"]
        + kf_eval_results[1]["kf_innovation_covariance"]
        + kf_eval_results[2]["kf_innovation_covariance"]
    )
    kf_all_kalman_gain = (
        kf_train_history["kalman_gain_norms"]
        + kf_eval_results[1]["kf_kalman_gain_norms"]
        + kf_eval_results[2]["kf_kalman_gain_norms"]
    )

    bll_all_preds = (
        bll_train_history["predictions"]
        + kf_eval_results[1]["bll_predictions"]
        + kf_eval_results[2]["bll_predictions"]
    )
    bll_all_uncertainties = (
        bll_train_history["uncertainties"]
        + kf_eval_results[1]["bll_uncertainties"]
        + kf_eval_results[2]["bll_uncertainties"]
    )
    bll_all_gt = (
        bll_train_history["ground_truth"]
        + kf_eval_results[1]["ground_truth"]
        + kf_eval_results[2]["ground_truth"]
    )
    bll_all_cov_norms = (
        bll_train_history["covariance_norms"]
        + kf_eval_results[1]["bll_covariance_norms"]
        + kf_eval_results[2]["bll_covariance_norms"]
    )

    # Convert to numpy arrays
    kf_all_preds = np.array(kf_all_preds)
    kf_all_uncertainties = np.array(kf_all_uncertainties)
    kf_all_gt = np.array(kf_all_gt)
    kf_all_cov_norms = np.array(kf_all_cov_norms)
    kf_all_innovation = np.array(kf_all_innovation)
    kf_all_innovation_cov = np.array(kf_all_innovation_cov)
    kf_all_kalman_gain = np.array(kf_all_kalman_gain)

    bll_all_preds = np.array(bll_all_preds)
    bll_all_uncertainties = np.array(bll_all_uncertainties)
    bll_all_gt = np.array(bll_all_gt)
    bll_all_cov_norms = np.array(bll_all_cov_norms)

    # ==========================================
    # Plot 1: Predictions with +/- 1 std dev (KF)
    # ==========================================
    fig1, ax1 = plt.subplots(1, 1, figsize=(16, 6))
    fig1.suptitle(
        f"KF Predictions with ±1σ Uncertainty (Q={q_val})", fontsize=14, fontweight="bold"
    )

    ax1.plot(time_axis, kf_all_gt, "k-", label="Ground Truth", alpha=0.7, linewidth=2)
    ax1.plot(time_axis, kf_all_preds, "r--", label="KF Prediction", alpha=0.8, linewidth=1.5)

    # ±1 std dev bounds
    upper_bound = kf_all_preds + kf_all_uncertainties
    lower_bound = kf_all_preds - kf_all_uncertainties
    ax1.fill_between(
        time_axis, lower_bound, upper_bound, color="red", alpha=0.2, label="±1σ bounds"
    )

    # Task boundaries
    for boundary in task_boundaries:
        ax1.axvline(boundary, color="blue", linestyle=":", alpha=0.5, linewidth=2)

    ax1.text(
        train_steps / 2,
        ax1.get_ylim()[1] * 0.9,
        "Task 0\n(Training)",
        ha="center",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.7},
    )
    ax1.text(
        train_steps + eval_steps_task1 / 2,
        ax1.get_ylim()[1] * 0.9,
        "Task 1\n(Prediction)",
        ha="center",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "lightgreen", "alpha": 0.7},
    )
    ax1.text(
        train_steps + eval_steps_task1 + eval_steps_task2 / 2,
        ax1.get_ylim()[1] * 0.9,
        "Task 2\n(Prediction)",
        ha="center",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.7},
    )

    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Prediction Value")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/KF_Predictions_Q{q_val:.3f}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"KF predictions plot saved to: {save_dir}/KF_Predictions_Q{q_val:.3f}.png")

    # ==========================================
    # Plot 2: Predictions with +/- 1 std dev (BLL)
    # ==========================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(16, 6))
    fig2.suptitle(
        f"BLL Predictions with ±1σ Uncertainty (Q={q_val})", fontsize=14, fontweight="bold"
    )

    ax2.plot(time_axis, bll_all_gt, "k-", label="Ground Truth", alpha=0.7, linewidth=2)
    ax2.plot(time_axis, bll_all_preds, "b--", label="BLL Prediction", alpha=0.8, linewidth=1.5)

    # ±1 std dev bounds
    upper_bound_bll = bll_all_preds + bll_all_uncertainties
    lower_bound_bll = bll_all_preds - bll_all_uncertainties
    ax2.fill_between(
        time_axis, lower_bound_bll, upper_bound_bll, color="blue", alpha=0.2, label="±1σ bounds"
    )

    # Task boundaries
    for boundary in task_boundaries:
        ax2.axvline(boundary, color="blue", linestyle=":", alpha=0.5, linewidth=2)

    ax2.text(
        train_steps / 2,
        ax2.get_ylim()[1] * 0.9,
        "Task 0\n(Training)",
        ha="center",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.7},
    )
    ax2.text(
        train_steps + eval_steps_task1 / 2,
        ax2.get_ylim()[1] * 0.9,
        "Task 1\n(Prediction)",
        ha="center",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "lightgreen", "alpha": 0.7},
    )
    ax2.text(
        train_steps + eval_steps_task1 + eval_steps_task2 / 2,
        ax2.get_ylim()[1] * 0.9,
        "Task 2\n(Prediction)",
        ha="center",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.7},
    )

    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Prediction Value")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/BLL_Predictions_Q{q_val:.3f}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"BLL predictions plot saved to: {save_dir}/BLL_Predictions_Q{q_val:.3f}.png")

    # ==========================================
    # Plot 3: Covariance Matrix Norms
    # ==========================================
    fig3, ax3 = plt.subplots(1, 1, figsize=(16, 6))
    fig3.suptitle(f"Covariance Matrix Norms (Q={q_val})", fontsize=14, fontweight="bold")

    ax3.plot(time_axis, kf_all_cov_norms, "r-", label="KF ||P||", linewidth=1.5, alpha=0.8)
    ax3.plot(time_axis, bll_all_cov_norms, "b-", label="BLL ||Cov||", linewidth=1.5, alpha=0.8)

    # Task boundaries
    for boundary in task_boundaries:
        ax3.axvline(boundary, color="blue", linestyle=":", alpha=0.5, linewidth=2)

    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Covariance Matrix Norm")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale("log")  # Log scale for better visualization

    plt.tight_layout()
    plt.savefig(f"{save_dir}/Covariance_Norms_Q{q_val:.3f}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Covariance norms plot saved to: {save_dir}/Covariance_Norms_Q{q_val:.3f}.png")

    # ==========================================
    # Plot 4: Single Prediction Plot (KF)
    # ==========================================
    fig4, ax4 = plt.subplots(1, 1, figsize=(16, 6))
    fig4.suptitle(f"KF Predictions Only (Q={q_val})", fontsize=14, fontweight="bold")

    ax4.plot(time_axis, kf_all_gt, "k-", label="Ground Truth", alpha=0.7, linewidth=2)
    ax4.plot(time_axis, kf_all_preds, "r--", label="KF Prediction", alpha=0.8, linewidth=1.5)

    for boundary in task_boundaries:
        ax4.axvline(boundary, color="blue", linestyle=":", alpha=0.5, linewidth=2)

    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("Prediction Value")
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/KF_Predictions_Only_Q{q_val:.3f}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ==========================================
    # Plot 5: Single Prediction Plot (BLL)
    # ==========================================
    fig5, ax5 = plt.subplots(1, 1, figsize=(16, 6))
    fig5.suptitle(f"BLL Predictions Only (Q={q_val})", fontsize=14, fontweight="bold")

    ax5.plot(time_axis, bll_all_gt, "k-", label="Ground Truth", alpha=0.7, linewidth=2)
    ax5.plot(time_axis, bll_all_preds, "b--", label="BLL Prediction", alpha=0.8, linewidth=1.5)

    for boundary in task_boundaries:
        ax5.axvline(boundary, color="blue", linestyle=":", alpha=0.5, linewidth=2)

    ax5.set_xlabel("Time Step")
    ax5.set_ylabel("Prediction Value")
    ax5.legend(loc="upper right")
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/BLL_Predictions_Only_Q{q_val:.3f}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ==========================================
    # Plot 6: Single Std Plot (KF)
    # ==========================================
    fig6, ax6 = plt.subplots(1, 1, figsize=(16, 6))
    fig6.suptitle(f"KF Uncertainty (Std Dev) (Q={q_val})", fontsize=14, fontweight="bold")

    ax6.plot(time_axis, kf_all_uncertainties, "r-", label="KF σ", linewidth=1.5, alpha=0.8)

    for boundary in task_boundaries:
        ax6.axvline(boundary, color="blue", linestyle=":", alpha=0.5, linewidth=2)

    ax6.set_xlabel("Time Step")
    ax6.set_ylabel("Standard Deviation")
    ax6.legend(loc="upper right")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/KF_Uncertainty_Q{q_val:.3f}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ==========================================
    # Plot 7: Single Std Plot (BLL)
    # ==========================================
    fig7, ax7 = plt.subplots(1, 1, figsize=(16, 6))
    fig7.suptitle(f"BLL Uncertainty (Std Dev) (Q={q_val})", fontsize=14, fontweight="bold")

    ax7.plot(time_axis, bll_all_uncertainties, "b-", label="BLL σ", linewidth=1.5, alpha=0.8)

    for boundary in task_boundaries:
        ax7.axvline(boundary, color="blue", linestyle=":", alpha=0.5, linewidth=2)

    ax7.set_xlabel("Time Step")
    ax7.set_ylabel("Standard Deviation")
    ax7.legend(loc="upper right")
    ax7.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/BLL_Uncertainty_Q{q_val:.3f}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ==========================================
    # Plot 8: Innovation and Innovation Covariance (KF only)
    # ==========================================
    fig8, (ax8a, ax8b) = plt.subplots(2, 1, figsize=(16, 10))
    fig8.suptitle(f"KF Innovation Metrics (Q={q_val})", fontsize=14, fontweight="bold")

    # Innovation
    ax8a.plot(time_axis, kf_all_innovation, "g-", label="Innovation", linewidth=1.5, alpha=0.8)
    ax8a.axhline(0, color="k", linestyle="--", alpha=0.3)
    for boundary in task_boundaries:
        ax8a.axvline(boundary, color="blue", linestyle=":", alpha=0.5, linewidth=2)
    ax8a.set_xlabel("Time Step")
    ax8a.set_ylabel("Innovation (y_true - y_pred)")
    ax8a.legend(loc="upper right")
    ax8a.grid(True, alpha=0.3)

    # Innovation Covariance
    ax8b.plot(
        time_axis,
        kf_all_innovation_cov,
        "purple",
        label="Innovation Covariance",
        linewidth=1.5,
        alpha=0.8,
    )
    for boundary in task_boundaries:
        ax8b.axvline(boundary, color="blue", linestyle=":", alpha=0.5, linewidth=2)
    ax8b.set_xlabel("Time Step")
    ax8b.set_ylabel("Innovation Covariance")
    ax8b.legend(loc="upper right")
    ax8b.grid(True, alpha=0.3)
    ax8b.set_yscale("log")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/KF_Innovation_Q{q_val:.3f}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"KF innovation plot saved to: {save_dir}/KF_Innovation_Q{q_val:.3f}.png")

    # ==========================================
    # Plot 9: Kalman Gain Norm (KF only)
    # ==========================================
    fig9, ax9 = plt.subplots(1, 1, figsize=(16, 6))
    fig9.suptitle(f"KF Kalman Gain Norm (Q={q_val})", fontsize=14, fontweight="bold")

    ax9.plot(time_axis, kf_all_kalman_gain, "orange", label="||K||", linewidth=1.5, alpha=0.8)

    for boundary in task_boundaries:
        ax9.axvline(boundary, color="blue", linestyle=":", alpha=0.5, linewidth=2)

    ax9.set_xlabel("Time Step")
    ax9.set_ylabel("Kalman Gain Norm")
    ax9.legend(loc="upper right")
    ax9.grid(True, alpha=0.3)
    ax9.set_yscale("log")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/KF_Kalman_Gain_Q{q_val:.3f}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"KF Kalman gain plot saved to: {save_dir}/KF_Kalman_Gain_Q{q_val:.3f}.png")


if __name__ == "__main__":
    # Configuration
    SEQ_LEN = 10
    HIDDEN_SIZE = 32
    SAMPLES_PER_TASK = 600  # Same number of samples for all tasks
    DT = 0.05  # Time step for continuous time axis
    RHO = 1.0
    ALPHA = 0.05
    MEASUREMENT_STD = 0.05

    # Q values to test: start with 0, then increase
    Q_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

    print("=" * 70)
    print("BLL vs KF Comparison Experiment")
    print("=" * 70)
    print("Task 0: sinx -> cosx (training)")
    print("Task 1: sin0.5x -> cos0.5x (prediction)")
    print("Task 2: sin2x -> cos2x (prediction)")
    print(f"Q values to test: {Q_VALUES}")
    print("=" * 70)

    # ==========================================
    # Phase 1: Train RNN on Task 0
    # ==========================================
    print("\nPhase 1: Training RNN on Task 0...")
    train_cache = _build_task_translators(num_samples=SAMPLES_PER_TASK, dt=DT)
    rnn_model, rnn_params = train_jax_rnn_on_task0(
        data_cache=train_cache,
        seq_len=SEQ_LEN,
        hidden_size=HIDDEN_SIZE,
        num_epochs=100,
        learning_rate=0.01,
        batch_size=32,
        seed=42,
        verbose=True,
    )
    print("RNN training complete. Parameters frozen.\n")

    # ==========================================
    # Phase 2: Pre-train BLL once (doesn't depend on Q)
    # ==========================================
    print("\n" + "=" * 70)
    print("Pre-training BLL once (independent of Q)...")
    print("=" * 70)
    bll = StandaloneBayesianLastLayer(sigma=MEASUREMENT_STD, alpha=ALPHA, feature_dim=HIDDEN_SIZE)
    task0_train_data = train_cache[0]
    bll_train_history = pretrain_bll_on_task0(
        rnn_model, rnn_params, bll, task0_train_data, SEQ_LEN, verbose=True
    )
    print("BLL pre-training complete. Will be reused for all Q values.\n")

    # ==========================================
    # Phase 3: Evaluate for each Q value (only KF changes)
    # ==========================================
    comparison_results = {}
    all_q_data = {}  # Store all training and evaluation data for plotting
    # Use same number of samples for all tasks
    # Continue time axis from where training ended to avoid phase discontinuity
    T_TRAIN_END = SAMPLES_PER_TASK * DT
    eval_cache = _build_task_translators(num_samples=SAMPLES_PER_TASK, t_start=T_TRAIN_END, dt=DT)

    for q_val in Q_VALUES:
        print(f"\n{'=' * 70}")
        print(f"Testing Q = {q_val}")
        print(f"{'=' * 70}")

        # Initialize KF (BLL already trained and reused)
        kf = KalmanFilterHead(
            feature_dim=HIDDEN_SIZE,
            rho=RHO,
            Q_std=q_val,
            R_std=MEASUREMENT_STD,
            initial_uncertainty=1 / ALPHA,
        )

        # Pre-train KF on Task 0 (returns history and final buffer)
        kf_train_history, task0_final_buffer = pretrain_kf_on_task0(
            rnn_model, rnn_params, kf, task0_train_data, SEQ_LEN, verbose=True
        )

        # Evaluate on Tasks 1 and 2 (prediction only), using buffer from Task 0
        eval_results = evaluate_on_tasks(
            rnn_model,
            rnn_params,
            kf,
            bll,
            eval_cache,
            tasks=[1, 2],
            seq_len=SEQ_LEN,
            initial_buffer=task0_final_buffer,
            verbose=True,
        )

        # Store all data for later plotting
        all_q_data[q_val] = {
            "kf_train_history": kf_train_history,
            "bll_train_history": bll_train_history,
            "kf_eval_results": eval_results,
        }

        # Compute comparison metrics
        comparison = compute_comparison_metrics(eval_results, tasks=[1, 2])
        comparison_results[q_val] = comparison

        # Print summary
        print(f"\nQ = {q_val} Summary:")
        for task in [1, 2]:
            kf_mean = comparison[task]["kf"]["mean_prediction"]
            kf_std = comparison[task]["kf"]["std_prediction"]
            kf_mae = comparison[task]["kf"]["mean_error"]
            bll_mean = comparison[task]["bll"]["mean_prediction"]
            bll_std = comparison[task]["bll"]["std_prediction"]
            bll_mae = comparison[task]["bll"]["mean_error"]
            print(f"  Task {task}:")
            print(f"    KF  - Mean: {kf_mean:.6f}, Std: {kf_std:.6f}, MAE: {kf_mae:.6f}")
            print(f"    BLL - Mean: {bll_mean:.6f}, Std: {bll_std:.6f}, MAE: {bll_mae:.6f}")

    # ==========================================
    # Phase 3: Generate Summary Plots for All Q Values
    # ==========================================
    print("\n" + "=" * 70)
    print("Generating summary plots comparing all Q values...")
    print("=" * 70)
    plot_detailed_metrics_all_q(all_q_data, Q_VALUES, save_dir="results/figures")

    # ==========================================
    # Phase 4: Plot Uncertainty vs Phi Norm
    # ==========================================
    print("\n" + "=" * 70)
    print("Generating uncertainty vs phi norm plot...")
    print("=" * 70)
    plot_uncertainty_vs_phi_norm(all_q_data, Q_VALUES, save_dir="results/figures")

    # ==========================================
    # Phase 5: Plot Results
    # ==========================================
    print("\n" + "=" * 70)
    print("Generating comparison plots...")
    print("=" * 70)
    plot_comparison_results(comparison_results, Q_VALUES)

    # Print final summary table
    print("\n" + "=" * 70)
    print("Final Summary Table")
    print("=" * 70)
    print(f"{'Q':<8} {'Task':<6} {'Method':<6} {'Mean Pred':<12} {'Std Dev':<12} {'MAE':<12}")
    print("-" * 70)
    for q_val in Q_VALUES:
        for task in [1, 2]:
            for method in ["kf", "bll"]:
                comp = comparison_results[q_val][task][method]
                print(
                    f"{q_val:<8.3f} {task:<6} {method.upper():<6} "
                    f"{comp['mean_prediction']:<12.6f} {comp['std_prediction']:<12.6f} "
                    f"{comp['mean_error']:<12.6f}"
                )
    print("=" * 70)
