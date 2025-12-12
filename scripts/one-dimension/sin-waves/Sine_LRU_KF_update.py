from __future__ import annotations

from collections import deque

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from latinx.data.sine_cosine import SineCosineTranslator
from latinx.metrics.metrics_bayesian_last_layer import mae as bll_mae
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
from latinx.models.rnn_jax import (
    SimpleRNN,
    create_prediction_head,
    create_rnn,
    create_train_step,
)
from latinx.models.standalone_bayesian_last_layer import StandaloneBayesianLastLayer

# Task configuration: cleaner and more maintainable
TASK_CONFIGS: dict[int, dict[str, float]] = {
    0: {"amplitude": 1.0, "angle_multiplier": 2},
    1: {"amplitude": 0.5, "angle_multiplier": 2},
    2: {"amplitude": 0.3, "angle_multiplier": 2},
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


def pretrain_kf_on_task0(
    rnn_model,
    rnn_params: dict,
    kf: KalmanFilterHead,
    task0_data: dict[str, np.ndarray],
    seq_len: int,
    verbose: bool = True,
) -> dict:
    """
    Pre-train Kalman Filter on Task 0 data using the frozen RNN.

    This function trains the KF by:
    1. Extracting features from the frozen RNN for each Task 0 sample
    2. Making predictions and updating KF parameters online
    3. Learning the Task 0 sine->cosine mapping
    4. Collecting training metrics for analysis

    Args:
        rnn_model: Trained JAX RNN model for feature extraction
        rnn_params: JAX RNN parameters
        kf: KalmanFilterHead instance to train
        task0_data: Task 0 data dictionary with "sine" and "cosine" keys
        seq_len: Sequence length for RNN input
        verbose: Whether to print progress

    Returns:
        Dictionary containing training history:
            - predictions: Predicted values at each step
            - ground_truth: True values at each step
            - errors: Prediction errors (absolute)
            - uncertainties: Prediction uncertainties (std)
            - weight_norms: L2 norm of weight vector at each step
            - weight_means: Mean weight values at each step
            - covariance_traces: Trace of covariance matrix at each step

    Example:
        >>> history = pretrain_kf_on_task0(rnn_model, rnn_params, kf, task0_data, seq_len=10)
        >>> plot_kf_training_history(history)
    """
    num_samples = len(task0_data["sine"]) - seq_len

    if verbose:
        print(f"Pre-training KF on {num_samples} samples from Task 0...")

    kf_buffer = deque(np.zeros(seq_len), maxlen=seq_len)

    # Initialize history tracking
    history = {
        "predictions": [],
        "ground_truth": [],
        "errors": [],
        "uncertainties": [],
        "weight_norms": [],
        "weight_means": [],
        "covariance_traces": [],
    }

    for t in range(num_samples):
        # Get data
        x_t = float(task0_data["sine"][t])
        y_t = float(task0_data["cosine"][t])
        kf_buffer.append(x_t)

        # Extract features from JAX RNN
        input_array = jnp.array(np.array(kf_buffer)).reshape(1, seq_len, 1)
        features_array, _ = rnn_model.apply(rnn_params, input_array)

        # Features for KF: shape (M, 1)
        phi = features_array.T

        # Train KF: predict then update
        y_pred = kf.predict(phi)
        _, pred_std = kf.get_prediction_uncertainty()
        kf.update(y_t, y_pred)

        # Collect metrics
        error = abs(y_t - y_pred)
        weights, _ = kf.get_weight_statistics()
        weight_norm = float(jnp.linalg.norm(weights))
        weight_mean = float(jnp.mean(weights))
        cov_trace = float(jnp.trace(kf.P))

        history["predictions"].append(y_pred)
        history["ground_truth"].append(y_t)
        history["errors"].append(error)
        history["uncertainties"].append(pred_std)
        history["weight_norms"].append(weight_norm)
        history["weight_means"].append(weight_mean)
        history["covariance_traces"].append(cov_trace)

    if verbose:
        print(f"KF pre-training complete on {num_samples} samples.")

    # Convert lists to numpy arrays
    for key in history:
        history[key] = np.array(history[key])

    return history


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

    This function:
    1. Extracts Task 0 data from the cache
    2. Creates sliding window sequences for training
    3. Trains the RNN + prediction head using JAX/Optax
    4. Returns the trained RNN model and parameters (head is discarded)

    Args:
        data_cache: Dictionary from _build_task_translators containing task data
        seq_len: Length of input sequences (default: 10)
        hidden_size: RNN hidden dimension (default: 32)
        num_epochs: Number of training epochs (default: 100)
        learning_rate: Learning rate for Adam optimizer (default: 0.01)
        batch_size: Batch size for training (default: 32)
        seed: Random seed for reproducibility (default: 42)
        verbose: Whether to print training progress (default: True)

    Returns:
        Tuple of (rnn_model, rnn_params) where:
            - rnn_model: The trained RNN model (JAX/Flax)
            - rnn_params: Trained RNN parameters (dict)

    Example:
        >>> data_cache = _build_task_translators(num_samples=2000)
        >>> rnn_model, rnn_params = train_jax_rnn_on_task0(
        ...     data_cache, seq_len=10, hidden_size=32
        ... )
        >>> # Use rnn_params with Bayesian heads for online adaptation
    """
    if verbose:
        print("Training JAX RNN on Task 0 (Sine -> Cosine)...")

    # Extract Task 0 data
    task0_data = data_cache[0]
    sine_data = task0_data["sine"]
    cosine_data = task0_data["cosine"]

    if verbose:
        print(f"Task 0 data: {len(sine_data)} samples")

    # ==========================================
    # 1. Create JAX RNN and Prediction Head
    # ==========================================
    rnn_model, rnn_params = create_rnn(
        input_size=1, hidden_size=hidden_size, seed=seed
    )

    # Create prediction head (simple linear layer)
    pred_head, head_params = create_prediction_head(
        hidden_size=hidden_size, output_dim=1, seed=seed
    )

    if verbose:
        print(f"Created RNN (hidden_size={hidden_size}) and prediction head")

    # ==========================================
    # 2. Setup Optimizers
    # ==========================================
    rnn_optimizer = optax.adam(learning_rate)
    head_optimizer = optax.adam(learning_rate)
    rnn_opt_state = rnn_optimizer.init(rnn_params)
    head_opt_state = head_optimizer.init(head_params)

    # ==========================================
    # 3. Create Training Step
    # ==========================================
    train_step = create_train_step(rnn_model, pred_head, rnn_optimizer, head_optimizer)

    # ==========================================
    # 4. Prepare Training Data (Sliding Windows)
    # ==========================================
    # Create sliding window sequences
    train_sequences = []
    train_targets = []

    for i in range(len(sine_data) - seq_len):
        # Input: sequence of sine values
        seq_in = sine_data[i : i + seq_len]
        # Target: cosine value at the end of sequence (with noise)
        target_out = cosine_data[i + seq_len - 1]

        train_sequences.append(seq_in)
        train_targets.append(target_out)

    train_sequences = np.array(train_sequences)  # (num_samples, seq_len)
    train_targets = np.array(train_targets)  # (num_samples,)

    num_samples = len(train_sequences)
    if verbose:
        print(f"Created {num_samples} training sequences (sliding window)")

    # ==========================================
    # 5. Training Loop
    # ==========================================
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Shuffle data each epoch
        indices = np.random.permutation(num_samples)

        # Mini-batch training
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_indices = indices[batch_start:batch_end]

            # Get batch data
            batch_sequences = train_sequences[batch_indices]  # (batch, seq_len)
            batch_targets = train_targets[batch_indices]  # (batch,)

            # Reshape for RNN: (batch, seq_len, input_size=1)
            x_batch = jnp.array(batch_sequences[:, :, None])
            y_batch = jnp.array(batch_targets)

            # Training step
            rnn_params, head_params, rnn_opt_state, head_opt_state, loss = train_step(
                rnn_params, head_params, rnn_opt_state, head_opt_state, x_batch, y_batch
            )

            epoch_loss += float(loss)
            num_batches += 1

        # Print progress
        avg_loss = epoch_loss / num_batches
        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch}/{num_epochs}: Loss = {avg_loss:.6f}")

    if verbose:
        print("JAX RNN training complete!")
        print("Returning trained RNN model and parameters (prediction head discarded)")

    # Return only RNN model and params (head is discarded)
    return rnn_model, rnn_params


def plot_frozen_evaluation_results(
    eval_results: dict,
    tasks: list[int],
    task_configs: dict,
) -> None:
    """
    Plot frozen model evaluation results for KF and BLL across tasks.

    Creates plots organized by metric type (predictions, errors, uncertainties)
    showing all tasks with task boundaries marked.

    Args:
        eval_results: Dictionary containing evaluation results for each task
        tasks: List of task IDs to plot
        task_configs: Dictionary mapping task IDs to configuration dicts

    Example:
        >>> plot_frozen_evaluation_results(eval_results, [0, 1, 2], TASK_CONFIGS)
    """
    print("\nGenerating plots...")

    # Concatenate all task data
    all_ground_truth = []
    all_kf_preds = []
    all_bll_preds = []
    all_kf_errors = []
    all_bll_errors = []
    all_kf_errors_normalized = []
    all_bll_errors_normalized = []
    all_kf_uncertainties = []
    all_bll_uncertainties = []
    all_kf_uncertainties_normalized = []
    all_bll_uncertainties_normalized = []

    task_boundaries = [0]  # Start of first task

    for task in tasks:
        task_config = task_configs[task]
        amplitude = abs(task_config['amplitude'])  # Use absolute value for normalization

        # Get task data
        ground_truth = eval_results[task]["ground_truth"]
        kf_errors = eval_results[task]["kf_errors"]
        bll_errors = eval_results[task]["bll_errors"]
        kf_uncertainties = eval_results[task]["kf_uncertainties"]
        bll_uncertainties = eval_results[task]["bll_uncertainties"]

        # Normalize errors by amplitude
        kf_errors_norm = [err / amplitude for err in kf_errors]
        bll_errors_norm = [err / amplitude for err in bll_errors]

        # Normalize uncertainties by amplitude
        kf_uncertainties_norm = [unc / amplitude for unc in kf_uncertainties]
        bll_uncertainties_norm = [unc / amplitude for unc in bll_uncertainties]

        all_ground_truth.extend(ground_truth)
        all_kf_preds.extend(eval_results[task]["kf_predictions"])
        all_bll_preds.extend(eval_results[task]["bll_predictions"])
        all_kf_errors.extend(kf_errors)
        all_bll_errors.extend(bll_errors)
        all_kf_errors_normalized.extend(kf_errors_norm)
        all_bll_errors_normalized.extend(bll_errors_norm)
        all_kf_uncertainties.extend(kf_uncertainties)
        all_bll_uncertainties.extend(bll_uncertainties)
        all_kf_uncertainties_normalized.extend(kf_uncertainties_norm)
        all_bll_uncertainties_normalized.extend(bll_uncertainties_norm)

        # Add boundary at end of this task (start of next task)
        task_boundaries.append(len(all_ground_truth))

    # Convert to numpy arrays
    all_ground_truth = np.array(all_ground_truth)
    all_kf_preds = np.array(all_kf_preds)
    all_bll_preds = np.array(all_bll_preds)
    all_kf_errors = np.array(all_kf_errors)
    all_bll_errors = np.array(all_bll_errors)
    all_kf_errors_normalized = np.array(all_kf_errors_normalized)
    all_bll_errors_normalized = np.array(all_bll_errors_normalized)
    all_kf_uncertainties = np.array(all_kf_uncertainties)
    all_bll_uncertainties = np.array(all_bll_uncertainties)
    all_kf_uncertainties_normalized = np.array(all_kf_uncertainties_normalized)
    all_bll_uncertainties_normalized = np.array(all_bll_uncertainties_normalized)

    time_steps = np.arange(len(all_ground_truth))

    # ==========================================
    # Plot 1: Predictions vs Ground Truth
    # ==========================================
    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 5))
    fig1.suptitle("Frozen Model Evaluation: Predictions Across All Tasks", fontsize=14, fontweight="bold")

    ax1.plot(time_steps, all_ground_truth, 'k-', label='Ground Truth', alpha=0.7, linewidth=2)
    ax1.plot(time_steps, all_kf_preds, 'r--', label='Kalman Filter', alpha=0.8, linewidth=1.5)
    ax1.plot(time_steps, all_bll_preds, 'b:', label='Bayesian Last Layer', alpha=0.8, linewidth=1.5)

    # Add task boundaries and labels
    for i, task in enumerate(tasks):
        boundary = task_boundaries[i]
        ax1.axvline(boundary, color='blue', linestyle=':', alpha=0.5, linewidth=2)

        # Add task label in the middle of each task region
        task_config = task_configs[task]
        if i < len(tasks):
            mid_point = (task_boundaries[i] + task_boundaries[i + 1]) / 2
            task_label = f"Task {task}: Cosine\n(amplitude={task_config['amplitude']}, Training)" if task == 0 else \
                        f"Task {task}: {task_config['amplitude']}x Cosine\n(amplitude={task_config['amplitude']}, Prediction Only)"
            ax1.text(mid_point, ax1.get_ylim()[1] * 0.9, task_label,
                    ha='center', va='top', fontsize=9, color='blue',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='blue'))

    ax1.set_title('Predictions vs Ground Truth')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Output Value')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file1 = "results/figures/Frozen_Eval_Predictions.png"
    plt.savefig(output_file1, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Predictions plot saved to: {output_file1}")

    # ==========================================
    # Plot 2: Absolute Errors
    # ==========================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 5))
    fig2.suptitle("Frozen Model Evaluation: Errors Across All Tasks", fontsize=14, fontweight="bold")

    ax2.plot(time_steps, all_kf_errors, 'r-', label='KF', alpha=0.7, linewidth=1.5)
    ax2.plot(time_steps, all_bll_errors, 'b-', label='BLL', alpha=0.7, linewidth=1.5)

    # Add task boundaries
    for i, task in enumerate(tasks):
        boundary = task_boundaries[i]
        ax2.axvline(boundary, color='blue', linestyle=':', alpha=0.5, linewidth=2)

        # Add task label
        task_config = task_configs[task]
        if i < len(tasks):
            mid_point = (task_boundaries[i] + task_boundaries[i + 1]) / 2
            task_label = f"Task {task}"
            ax2.text(mid_point, ax2.get_ylim()[1] * 0.9, task_label,
                    ha='center', va='top', fontsize=9, color='blue',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='blue'))

    ax2.set_title('Absolute Errors')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('|Error|')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Add overall MAE text
    kf_mae = np.mean(all_kf_errors)
    bll_mae = np.mean(all_bll_errors)
    ax2.text(0.02, 0.98, f'Overall KF MAE: {kf_mae:.4f}\nOverall BLL MAE: {bll_mae:.4f}',
            transform=ax2.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_file2 = "results/figures/Frozen_Eval_Errors.png"
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Errors plot saved to: {output_file2}")

    # ==========================================
    # Plot 3: Normalized Errors (by amplitude)
    # ==========================================
    fig3, ax3 = plt.subplots(1, 1, figsize=(14, 5))
    fig3.suptitle("Frozen Model Evaluation: Normalized Errors Across All Tasks", fontsize=14, fontweight="bold")

    ax3.plot(time_steps, all_kf_errors_normalized, 'r-', label='KF', alpha=0.7, linewidth=1.5)
    ax3.plot(time_steps, all_bll_errors_normalized, 'b-', label='BLL', alpha=0.7, linewidth=1.5)

    # Add task boundaries
    for i, task in enumerate(tasks):
        boundary = task_boundaries[i]
        ax3.axvline(boundary, color='blue', linestyle=':', alpha=0.5, linewidth=2)

        # Add task label
        task_config = task_configs[task]
        if i < len(tasks):
            mid_point = (task_boundaries[i] + task_boundaries[i + 1]) / 2
            task_label = f"Task {task}\n(A={task_config['amplitude']})"
            ax3.text(mid_point, ax3.get_ylim()[1] * 0.9, task_label,
                    ha='center', va='top', fontsize=9, color='blue',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='blue'))

    ax3.set_title('Normalized Absolute Errors (÷ amplitude)')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Normalized |Error|')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Add overall normalized MAE text
    kf_nmae = np.mean(all_kf_errors_normalized)
    bll_nmae = np.mean(all_bll_errors_normalized)
    ax3.text(0.02, 0.98, f'Overall KF NMAE: {kf_nmae:.4f}\nOverall BLL NMAE: {bll_nmae:.4f}',
            transform=ax3.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_file3 = "results/figures/Frozen_Eval_Errors_Normalized.png"
    plt.savefig(output_file3, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Normalized errors plot saved to: {output_file3}")

    # ==========================================
    # Plot 4: Prediction Uncertainties
    # ==========================================
    fig4, ax4 = plt.subplots(1, 1, figsize=(14, 5))
    fig4.suptitle("Frozen Model Evaluation: Uncertainties Across All Tasks", fontsize=14, fontweight="bold")

    ax4.plot(time_steps, all_kf_uncertainties, 'r-', label='KF σ', alpha=0.7, linewidth=1.5)
    ax4.plot(time_steps, all_bll_uncertainties, 'b-', label='BLL σ', alpha=0.7, linewidth=1.5)

    # Add task boundaries
    for i, task in enumerate(tasks):
        boundary = task_boundaries[i]
        ax4.axvline(boundary, color='blue', linestyle=':', alpha=0.5, linewidth=2)

        # Add task label
        if i < len(tasks):
            mid_point = (task_boundaries[i] + task_boundaries[i + 1]) / 2
            task_label = f"Task {task}"
            ax4.text(mid_point, ax4.get_ylim()[1] * 0.9, task_label,
                    ha='center', va='top', fontsize=9, color='blue',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='blue'))

    ax4.set_title('Prediction Uncertainty')
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Standard Deviation')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    # Add overall mean uncertainty text
    kf_mean_unc = np.mean(all_kf_uncertainties)
    bll_mean_unc = np.mean(all_bll_uncertainties)
    ax4.text(0.02, 0.98, f'Overall KF σ̄: {kf_mean_unc:.4f}\nOverall BLL σ̄: {bll_mean_unc:.4f}',
            transform=ax4.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_file4 = "results/figures/Frozen_Eval_Uncertainties.png"
    plt.savefig(output_file4, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Uncertainties plot saved to: {output_file4}")

    # ==========================================
    # Plot 5: Normalized Uncertainties (by amplitude)
    # ==========================================
    fig6, ax6 = plt.subplots(1, 1, figsize=(14, 5))
    fig6.suptitle("Frozen Model Evaluation: Normalized Uncertainties Across All Tasks", fontsize=14, fontweight="bold")

    ax6.plot(time_steps, all_kf_uncertainties_normalized, 'r-', label='KF σ', alpha=0.7, linewidth=1.5)
    ax6.plot(time_steps, all_bll_uncertainties_normalized, 'b-', label='BLL σ', alpha=0.7, linewidth=1.5)

    # Add task boundaries
    for i, task in enumerate(tasks):
        boundary = task_boundaries[i]
        ax6.axvline(boundary, color='blue', linestyle=':', alpha=0.5, linewidth=2)

        # Add task label
        task_config = task_configs[task]
        if i < len(tasks):
            mid_point = (task_boundaries[i] + task_boundaries[i + 1]) / 2
            task_label = f"Task {task}\n(A={task_config['amplitude']})"
            ax6.text(mid_point, ax6.get_ylim()[1] * 0.9, task_label,
                    ha='center', va='top', fontsize=9, color='blue',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='blue'))

    ax6.set_title('Normalized Prediction Uncertainty (÷ amplitude)')
    ax6.set_xlabel('Sample Index')
    ax6.set_ylabel('Normalized σ')
    ax6.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)

    # Add overall normalized mean uncertainty text
    kf_mean_unc_norm = np.mean(all_kf_uncertainties_normalized)
    bll_mean_unc_norm = np.mean(all_bll_uncertainties_normalized)
    ax6.text(0.02, 0.98, f'Overall KF σ̄: {kf_mean_unc_norm:.4f}\nOverall BLL σ̄: {bll_mean_unc_norm:.4f}',
            transform=ax6.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_file6 = "results/figures/Frozen_Eval_Uncertainties_Normalized.png"
    plt.savefig(output_file6, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Normalized uncertainties plot saved to: {output_file6}")

    # ==========================================
    # Summary Bar Chart
    # ==========================================
    fig5, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig5.suptitle("Frozen Model Evaluation Summary", fontsize=14, fontweight="bold")

    tasks_labels = [f"Task {t}" for t in tasks]

    # Compute per-task metrics
    kf_maes = []
    bll_maes = []
    kf_nmaes = []
    bll_nmaes = []
    kf_uncs = []
    bll_uncs = []

    for task in tasks:
        task_config = task_configs[task]
        amplitude = abs(task_config['amplitude'])

        kf_errors = eval_results[task]["kf_errors"]
        bll_errors = eval_results[task]["bll_errors"]

        kf_maes.append(np.mean(kf_errors))
        bll_maes.append(np.mean(bll_errors))
        kf_nmaes.append(np.mean([e / amplitude for e in kf_errors]))
        bll_nmaes.append(np.mean([e / amplitude for e in bll_errors]))
        kf_uncs.append(np.mean(eval_results[task]["kf_uncertainties"]))
        bll_uncs.append(np.mean(eval_results[task]["bll_uncertainties"]))

    x = np.arange(len(tasks))
    width = 0.35

    # Absolute MAE comparison
    ax1.bar(x - width/2, kf_maes, width, label='KF', color='red', alpha=0.7)
    ax1.bar(x + width/2, bll_maes, width, label='BLL', color='blue', alpha=0.7)
    ax1.set_xlabel('Task')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title('Absolute MAE by Task')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (kf_val, bll_val) in enumerate(zip(kf_maes, bll_maes)):
        ax1.text(i - width/2, kf_val, f'{kf_val:.4f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width/2, bll_val, f'{bll_val:.4f}', ha='center', va='bottom', fontsize=8)

    # Normalized MAE comparison
    ax2.bar(x - width/2, kf_nmaes, width, label='KF', color='red', alpha=0.7)
    ax2.bar(x + width/2, bll_nmaes, width, label='BLL', color='blue', alpha=0.7)
    ax2.set_xlabel('Task')
    ax2.set_ylabel('Normalized MAE')
    ax2.set_title('Normalized MAE by Task (÷ amplitude)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (kf_val, bll_val) in enumerate(zip(kf_nmaes, bll_nmaes)):
        ax2.text(i - width/2, kf_val, f'{kf_val:.4f}', ha='center', va='bottom', fontsize=8)
        ax2.text(i + width/2, bll_val, f'{bll_val:.4f}', ha='center', va='bottom', fontsize=8)

    # Uncertainty comparison
    ax3.bar(x - width/2, kf_uncs, width, label='KF', color='red', alpha=0.7)
    ax3.bar(x + width/2, bll_uncs, width, label='BLL', color='blue', alpha=0.7)
    ax3.set_xlabel('Task')
    ax3.set_ylabel('Mean Uncertainty (σ)')
    ax3.set_title('Average Prediction Uncertainty by Task')
    ax3.set_xticks(x)
    ax3.set_xticklabels(tasks_labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (kf_val, bll_val) in enumerate(zip(kf_uncs, bll_uncs)):
        ax3.text(i - width/2, kf_val, f'{kf_val:.4f}', ha='center', va='bottom', fontsize=8)
        ax3.text(i + width/2, bll_val, f'{bll_val:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig("results/figures/Frozen_Eval_Summary.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Summary plot saved to: results/figures/Frozen_Eval_Summary.png")

    print("\nAll plots generated successfully!")


def plot_kf_training_history(
    history: dict,
    save_path: str = "results/figures/KF_Training_History.png",
) -> None:
    """
    Plot Kalman Filter training history metrics.

    Creates a 2x3 grid showing:
    - Predictions vs ground truth with 3-sigma bounds
    - Prediction errors over time
    - Prediction uncertainty over time
    - Weight vector L2 norm over time
    - Mean weight value over time
    - Covariance trace over time

    Args:
        history: Dictionary containing training metrics from pretrain_kf_on_task0
        save_path: Path to save the figure

    Example:
        >>> history = pretrain_kf_on_task0(...)
        >>> plot_kf_training_history(history)
    """
    print(f"\nPlotting KF training history...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Kalman Filter Training History (Task 0)", fontsize=16, fontweight="bold")

    time_steps = np.arange(len(history["predictions"]))
    predictions = history["predictions"]
    ground_truth = history["ground_truth"]
    errors = history["errors"]
    uncertainties = history["uncertainties"]
    weight_norms = history["weight_norms"]
    weight_means = history["weight_means"]
    cov_traces = history["covariance_traces"]

    # ==========================================
    # Row 1, Col 1: Predictions vs Ground Truth with 3-sigma bounds
    # ==========================================
    ax1 = axes[0, 0]
    ax1.plot(time_steps, ground_truth, 'k-', label='Ground Truth', alpha=0.7, linewidth=1.5)
    ax1.plot(time_steps, predictions, 'r--', label='KF Prediction', alpha=0.8, linewidth=1.5)

    # 3-sigma bounds
    upper_bound = predictions + 3 * uncertainties
    lower_bound = predictions - 3 * uncertainties
    ax1.fill_between(time_steps, lower_bound, upper_bound,
                     color='red', alpha=0.2, label='3σ bounds')

    ax1.set_title('Predictions vs Ground Truth')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Output Value')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ==========================================
    # Row 1, Col 2: Prediction Errors
    # ==========================================
    ax2 = axes[0, 1]
    ax2.plot(time_steps, errors, 'b-', alpha=0.7, linewidth=1.5)
    ax2.axhline(np.mean(errors), color='b', linestyle='--',
                alpha=0.5, linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
    ax2.set_title('Prediction Errors')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Absolute Error')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # ==========================================
    # Row 1, Col 3: Prediction Uncertainty (σ)
    # ==========================================
    ax3 = axes[0, 2]
    ax3.plot(time_steps, uncertainties, 'g-', alpha=0.7, linewidth=1.5)
    ax3.axhline(np.mean(uncertainties), color='g', linestyle='--',
                alpha=0.5, linewidth=2, label=f'Mean: {np.mean(uncertainties):.4f}')
    ax3.set_title('Prediction Uncertainty (σ)')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Standard Deviation')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # ==========================================
    # Row 2, Col 1: Weight Vector L2 Norm
    # ==========================================
    ax4 = axes[1, 0]
    ax4.plot(time_steps, weight_norms, 'purple', alpha=0.7, linewidth=1.5)
    ax4.axhline(weight_norms[-1], color='purple', linestyle='--',
                alpha=0.5, linewidth=2, label=f'Final: {weight_norms[-1]:.4f}')
    ax4.set_title('Weight Vector L2 Norm')
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('||w||₂')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    # ==========================================
    # Row 2, Col 2: Mean Weight Value
    # ==========================================
    ax5 = axes[1, 1]
    ax5.plot(time_steps, weight_means, 'orange', alpha=0.7, linewidth=1.5)
    ax5.axhline(0, color='k', linestyle='-', alpha=0.3, linewidth=1)
    ax5.axhline(weight_means[-1], color='orange', linestyle='--',
                alpha=0.5, linewidth=2, label=f'Final: {weight_means[-1]:.4f}')
    ax5.set_title('Mean Weight Value')
    ax5.set_xlabel('Sample Index')
    ax5.set_ylabel('mean(w)')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)

    # ==========================================
    # Row 2, Col 3: Covariance Trace
    # ==========================================
    ax6 = axes[1, 2]
    ax6.plot(time_steps, cov_traces, 'brown', alpha=0.7, linewidth=1.5)
    ax6.axhline(cov_traces[-1], color='brown', linestyle='--',
                alpha=0.5, linewidth=2, label=f'Final: {cov_traces[-1]:.4f}')
    ax6.set_title('Covariance Matrix Trace')
    ax6.set_xlabel('Sample Index')
    ax6.set_ylabel('tr(P)')
    ax6.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"KF training history plot saved to: {save_path}")


def plot_trajectory_comparison(
    history: dict,
    sim_steps: int,
    switch_points: list[int],
    save_path: str = "results/figures/Figure_A.png",
) -> None:
    """
    Plot trajectory, uncertainty, and weight norms comparison between KF and BLL.

    Args:
        history: Dictionary containing simulation history
        sim_steps: Total number of simulation steps
        switch_points: List of time points where tasks switch
        save_path: Path to save the figure
    """
    time_axis = np.arange(sim_steps)
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
    for pt in switch_points:
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
    for pt in switch_points:
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
    for pt in switch_points:
        plt.axvline(pt, color="blue", linestyle=":")
    plt.xlabel("Time Step")
    plt.ylabel("L2 Norm")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Trajectory comparison figure saved to {save_path}")

def plot_metrics_comparison(
    history: dict,
    sim_steps: int,
    switch_points: list[int],
    save_path: str = "results/figures/Figure_Metrics.png",
) -> None:
    """
    Plot detailed metrics comparison between KF and BLL.

    Args:
        history: Dictionary containing simulation history
        sim_steps: Total number of simulation steps
        switch_points: List of time points where tasks switch
        save_path: Path to save the figure
    """
    print("Generating metrics comparison figure...")
    time_axis = np.arange(sim_steps)

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
    for pt in switch_points:
        axes[0, 0].axvline(pt, color="red", linestyle=":", linewidth=1, alpha=0.7)

    # Plot 2: Normalized Innovation Squared (NIS) - KF only
    axes[0, 1].plot(time_axis, history["nis"], "g-", linewidth=1.5, label="NIS")
    axes[0, 1].axhline(1.0, color="r", linestyle="--", linewidth=1, label="Target (NIS=1)")
    axes[0, 1].set_title("Normalized Innovation Squared (NIS) - KF Only")
    axes[0, 1].set_ylabel("NIS")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    for pt in switch_points:
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
    for pt in switch_points:
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
    for pt in switch_points:
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
    for pt in switch_points:
        axes[2, 0].axvline(pt, color="red", linestyle=":", linewidth=1, alpha=0.7)

    # Plot 6: Kalman Gain Norm - KF only
    axes[2, 1].plot(time_axis, history["kalman_gain_norm"], "teal", linewidth=1.5)
    axes[2, 1].set_title("Norm of Kalman Gain (KF Only)")
    axes[2, 1].set_ylabel("||K||")
    axes[2, 1].grid(True, alpha=0.3)
    for pt in switch_points:
        axes[2, 1].axvline(pt, color="red", linestyle=":", linewidth=1, alpha=0.7)

    # Plot 7: Innovation - KF only
    axes[3, 0].plot(time_axis, history["innovation"], "magenta", linewidth=1.5)
    axes[3, 0].axhline(0.0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[3, 0].set_title("Innovation (Prediction Error) - KF Only")
    axes[3, 0].set_xlabel("Time Step")
    axes[3, 0].set_ylabel("Innovation")
    axes[3, 0].grid(True, alpha=0.3)
    for pt in switch_points:
        axes[3, 0].axvline(pt, color="red", linestyle=":", linewidth=1, alpha=0.7)

    # Plot 8: Innovation Variance - KF only
    axes[3, 1].plot(time_axis, history["innovation_variance"], "darkgreen", linewidth=1.5)
    axes[3, 1].set_title("Innovation Variance (S) - KF Only")
    axes[3, 1].set_xlabel("Time Step")
    axes[3, 1].set_ylabel("Variance")
    axes[3, 1].grid(True, alpha=0.3)
    for pt in switch_points:
        axes[3, 1].axvline(pt, color="red", linestyle=":", linewidth=1, alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Metrics comparison figure saved to {save_path}")


if __name__ == "__main__":
    # ==========================================
    # 4. PHASE 1: PRE-TRAINING THE JAX RNN
    # ==========================================
    print("Phase 1: Pre-training JAX RNN on Task 0 (Sine -> Cosine)...")

    # Configuration
    SEQ_LEN = 10
    HIDDEN_SIZE = 32
    TRAIN_STEPS = 600
    EVAL_STEPS_PER_TASK = 200  

    # Generate Task 0 data for pre-training
    print("Generating Task 0 training data...")
    train_cache = _build_task_translators(num_samples=TRAIN_STEPS)

    # Train JAX RNN using the helper function
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

    print("JAX RNN pre-training complete. Parameters frozen (immutable in JAX).")

    # ==========================================
    # 5. PHASE 2: ONLINE ADAPTATION
    # ==========================================
    print("Phase 2: Online Bayesian Adaptation across Multiple Functions...")

    Q = 0.3
    RHO = 1.0
    ALPHA = 0.05
    MEASUREMENT_STD = 0.05 #observation/measurement standard deviation
    """
    Set These params to verify BLL and KF are equivalent when params are right (Sanity Check)
    Q = 0.0
    RHO = 1.0
    """

    # Initialize the Bayesian Heads
    # KF and BLL configured for equivalence: Q=0, rho=1.0, matching prior
    kf = KalmanFilterHead(
        feature_dim=HIDDEN_SIZE,
        rho=RHO,  # No forgetting (equivalent to batch learning)
        Q_std=Q, 
        R_std=MEASUREMENT_STD,  # Observation noise std
        initial_uncertainty=1/ALPHA,  # 1/alpha = 1/0.05 (match BLL prior)
    )
    bll = StandaloneBayesianLastLayer(sigma=MEASUREMENT_STD, alpha=ALPHA, feature_dim=HIDDEN_SIZE)

    # ==========================================
    # PRE-TRAIN KF ON TASK 0 DATA (ONLINE)
    # ==========================================
    task0_train_data = train_cache[0]
    kf_training_history = pretrain_kf_on_task0(
        rnn_model=rnn_model,
        rnn_params=rnn_params,
        kf=kf,
        task0_data=task0_train_data,
        seq_len=SEQ_LEN,
        verbose=True,
    )

    # ==========================================
    # PRE-TRAIN BLL ON TASK 0 DATA (BATCH)
    # ==========================================
    # Use the same Task 0 data that was used for RNN pre-training
    num_bll_samples = len(task0_train_data["sine"]) - SEQ_LEN  # Max samples we can use

    print(f"Pre-training BLL on {num_bll_samples} samples from Task 0 (RNN pre-training data)...")

    features_list = []
    targets_list = []
    bll_buffer = deque(np.zeros(SEQ_LEN), maxlen=SEQ_LEN)

    for t in range(num_bll_samples):
        # Get data from RNN pre-training cache
        x_t = float(task0_train_data["sine"][t])
        y_t = float(task0_train_data["cosine"][t])
        bll_buffer.append(x_t)

        # Extract features from JAX RNN
        input_array = jnp.array(np.array(bll_buffer)).reshape(1, SEQ_LEN, 1)
        features_array, _ = rnn_model.apply(rnn_params, input_array)

        # Store features and targets
        features_list.append(features_array[0])  # Store as 1D array (M,)
        targets_list.append(y_t)

    # Fit BLL on all accumulated data
    features_array = jnp.array(features_list)  # (n_samples, M)
    targets_array = jnp.array(targets_list)  # (n_samples,)
    bll.fit(features_array, targets_array)

    print(f"BLL pre-training complete on {len(features_list)} samples.")

    # ==========================================
    # 6. ONLINE ADAPTATION ON ALL TASKS
    # ==========================================
    # KF: Online learning (predict + update each step)
    # BLL: Frozen (prediction only, no updates)
    # RNN: Frozen (feature extraction only)
    print(f"\nGenerating evaluation data ({EVAL_STEPS_PER_TASK} samples per task)...")
    eval_cache = _build_task_translators(num_samples=EVAL_STEPS_PER_TASK)
    TASKS = [0, 1, 2]

    # Storage for evaluation results
    eval_results = {
        task_id: {
            "kf_predictions": [],
            "kf_errors": [],
            "kf_uncertainties": [],
            "bll_predictions": [],
            "bll_errors": [],
            "bll_uncertainties": [],
            "ground_truth": [],
        }
        for task_id in TASKS
    }

    print("\nRunning online adaptation (KF updates, BLL frozen)...")

    for task in TASKS:
        print(f"\n{'='*60}")
        print(f"Task {task}: {TASK_CONFIGS[task]}")
        print(f"{'='*60}")

        data = eval_cache[task]
        num_eval_samples = len(data["sine"]) - SEQ_LEN
        eval_buffer = deque(np.zeros(SEQ_LEN), maxlen=SEQ_LEN)

        for t in range(num_eval_samples):
            # Get data
            x_t = float(data["sine"][t])
            y_t = float(data["cosine"][t])
            eval_buffer.append(x_t)

            # Extract features from frozen RNN
            input_array = jnp.array(np.array(eval_buffer)).reshape(1, SEQ_LEN, 1)
            features_array, _ = rnn_model.apply(rnn_params, input_array)

            # Features for KF: shape (M, 1)
            phi_kf = features_array.T

            # Features for BLL: shape (1, M)
            phi_bll = features_array

            # ==========================================
            # KF Evaluation (ONLINE: predict then update)
            # ==========================================
            kf_pred = kf.predict(phi_kf)
            _, kf_uncertainty = kf.get_prediction_uncertainty()
            kf.update(y_t, kf_pred)  # Update KF with ground truth

            # ==========================================
            # BLL Evaluation (FROZEN: prediction only, no update)
            # ==========================================
            bll_pred, bll_uncertainty = bll.predict(phi_bll, return_std=True)
            bll_pred = float(bll_pred.item())
            bll_uncertainty = float(bll_uncertainty.item())

            # Store results
            eval_results[task]["kf_predictions"].append(kf_pred)
            eval_results[task]["kf_errors"].append(abs(y_t - kf_pred))
            eval_results[task]["kf_uncertainties"].append(kf_uncertainty)
            eval_results[task]["bll_predictions"].append(bll_pred)
            eval_results[task]["bll_errors"].append(abs(y_t - bll_pred))
            eval_results[task]["bll_uncertainties"].append(bll_uncertainty)
            eval_results[task]["ground_truth"].append(y_t)

        # Compute and print statistics for this task
        kf_mae = np.mean(eval_results[task]["kf_errors"])
        bll_mae = np.mean(eval_results[task]["bll_errors"])
        kf_std = np.std(eval_results[task]["kf_errors"])
        bll_std = np.std(eval_results[task]["bll_errors"])

        print(f"\nTask {task} Results ({num_eval_samples} samples):")
        print(f"  KF  - MAE: {kf_mae:.6f} ± {kf_std:.6f}")
        print(f"  BLL - MAE: {bll_mae:.6f} ± {bll_std:.6f}")

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for task in TASKS:
        kf_mae = np.mean(eval_results[task]["kf_errors"])
        bll_mae = np.mean(eval_results[task]["bll_errors"])
        print(f"Task {task}: KF={kf_mae:.6f}, BLL={bll_mae:.6f}")

    print("\nEvaluation complete! Results stored in eval_results dictionary.")

    # ==========================================
    # 7. PLOTTING RESULTS
    # ==========================================
    # Plot KF training history
    plot_kf_training_history(kf_training_history)

    # Plot frozen evaluation results
    plot_frozen_evaluation_results(eval_results, TASKS, TASK_CONFIGS)
