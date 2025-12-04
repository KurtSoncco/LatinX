"""
Validation script comparing Kalman Filter and Bayesian Linear Regression.

This script validates that KalmanFilterHead (online) and StandaloneBayesianLastLayer (batch)
give equivalent results when Q=0 (no process noise).

Uses the same data generation as last-layer-bnn.ipynb:
- Feature transformation: phi(x) = x^3
- True relationship: y = c * x^3 + noise
- No bias term (regression through origin)
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from latinx.models.kalman_filter import KalmanFilterHead
from latinx.models.standalone_bayesian_last_layer import StandaloneBayesianLastLayer

def phi(x):
    """Feature transformation: cubic."""
    return x**3


def generate_data(key, n_obs_all=100, sigma=0.3, c=0.01):
    """
    Generate data matching the notebook's approach.

    Args:
        key: JAX random key
        n_obs_all: Total number of samples to generate
        sigma: Observation noise std
        c: True coefficient (y = c * x^3 + noise)

    Returns:
        x_train, y_train: Training data (filtered to |x| in [2, 5])
        x_all, y_all: All generated data (for plotting)
    """
    key_x, key_y = jax.random.split(key)

    # Generate data
    x_all = jax.random.uniform(key_x, (n_obs_all,), minval=-7.5, maxval=7.5)
    y_all = c * phi(x_all) + jax.random.normal(key_y, shape=(n_obs_all,)) * sigma

    # Filter to region |x| in [2, 5] (same as notebook)
    mask_vals = jnp.logical_and(jnp.abs(x_all) > 2, jnp.abs(x_all) < 5)
    x_train = x_all[mask_vals]
    y_train = y_all[mask_vals]

    return x_train, y_train, x_all, y_all


def predict_with_kf(kf:KalmanFilterHead, x_eval):
    """
    Make predictions using KalmanFilterHead.

    Args:
        kf: Fitted KalmanFilterHead
        x_eval: Evaluation inputs, shape (n_eval,)

    Returns:
        y_pred, y_std: Predictions and uncertainties, shape (n_eval,)
    """
    n_eval = len(x_eval)
    y_pred = jnp.zeros(n_eval)
    y_std = jnp.zeros(n_eval)

    for i in range(n_eval):
        # Compute features
        phi_x = jnp.array([[phi(x_eval[i])]])  # Shape: (1, 1)

        # Predict (don't update - just inference)
        pred = kf.predict(phi_x)

        # Get prediction uncertainty using the new method
        S, std = kf.get_prediction_uncertainty()

        y_pred = y_pred.at[i].set(pred)
        y_std = y_std.at[i].set(std)

        # Clean up intermediate state
        kf.mu_minus = None
        kf.P_minus = None
        kf.H = None

    return y_pred, y_std


def main():
    """Main validation script."""
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    seed = 3141
    sigma = 0.3  # Observation noise std
    alpha = 1.0  # Prior precision
    c = 0.01  # True coefficient

    print("=" * 60)
    print("KF vs BLR Validation Script")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  sigma (obs noise): {sigma}")
    print(f"  alpha (prior precision): {alpha}")
    print(f"  c (true coefficient): {c}")
    print(f"  Feature transform: phi(x) = x^3")
    print()

    # ==========================================
    # 2. GENERATE DATA
    # ==========================================
    key = jax.random.PRNGKey(seed)
    x_train, y_train, x_all, y_all = generate_data(key, sigma=sigma, c=c)

    n_train = len(x_train)
    print(f"Generated {n_train} training samples (filtered from 100)")
    print(f"  x range: [{x_train.min():.2f}, {x_train.max():.2f}]")
    print(f"  y range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    print()

    # ==========================================
    # 3. FIT MODELS
    # ==========================================
    print("Fitting models...")

    # KF (online)
    print("  [1/2] Kalman Filter (online)...", end=" ")
    # Inline fit_kalman_filter_online
    kf = KalmanFilterHead(
        feature_dim=1,  # phi(x) gives 1D features
        rho=1.0,  # No forgetting
        Q_std=0.0,  # Q=0 for equivalence with batch BLR
        R_std=sigma,  # Match observation noise
        initial_uncertainty=1 / alpha,  # Match prior
    )

    # Online updates
    for i in range(len(x_train)):
        # Compute features
        phi_x = jnp.array([[phi(x_train[i])]])  # Shape: (1, 1)

        # Predict and update
        y_pred = kf.predict(phi_x)
        kf.update(y_train[i], y_pred)

    # BLL (batch)
    print("  [2/2] Bayesian Last Layer (batch)...", end=" ")
    # Inline fit_bayesian_last_layer_batch
    # Compute features for all training data
    features_train = phi(x_train)[:, None]  # Shape: (n_samples, 1)

    # Fit batch BLR
    bll = StandaloneBayesianLastLayer(sigma=sigma, alpha=alpha, feature_dim=1)
    bll.fit(features_train, y_train)

    # ==========================================
    # 4. COMPARE POSTERIORS
    # ==========================================
    print("Comparing posterior parameters:")

    # Get posterior statistics
    mu_kf, std_kf = kf.get_weight_statistics()
    mu_bll, std_bll = bll.get_weight_statistics()

    print(f"  KF  posterior mean: {mu_kf[0]:.6f}  {std_kf[0]:.6f}")
    print(f"  BLL posterior mean: {mu_bll[0]:.6f}  {std_bll[0]:.6f}")
    print(f"  Difference (mean): {abs(mu_kf[0] - mu_bll[0]):.2e}")
    print(f"  Difference (std):  {abs(std_kf[0] - std_bll[0]):.2e}")
    print()

    # Check if they match
    mean_match = jnp.allclose(mu_kf, mu_bll, rtol=1e-5, atol=1e-8)
    std_match = jnp.allclose(std_kf, std_bll, rtol=1e-5, atol=1e-8)

    if mean_match and std_match:
        print("✓ Posteriors match! KF (Q=0) ≡ BLR")
    else:
        print("⚠ Warning: Posteriors don't match exactly")
        if not mean_match:
            print(f"  Mean mismatch: {jnp.max(jnp.abs(mu_kf - mu_bll)):.2e}")
        if not std_match:
            print(f"  Std mismatch: {jnp.max(jnp.abs(std_kf - std_bll)):.2e}")
    print()

    # ==========================================
    # 5. GENERATE PREDICTIONS
    # ==========================================
    print("Generating predictions for plotting...")

    # Evaluation points
    n_eval = 1000
    x_eval = jnp.linspace(-7, 7, n_eval)

    # KF predictions
    y_pred_kf, y_std_kf = predict_with_kf(kf, x_eval)

    # BLL predictions
    features_eval = phi(x_eval)[:, None]  # Shape: (n_eval, 1)
    y_pred_bll, y_std_bll = bll.predict(features_eval, return_std=True)

    # True function (no noise)
    y_true = c * phi(x_eval)

    print("Predictions generated")
    print()

    # ==========================================
    # 6. PLOTTING
    # ==========================================
    print("Creating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: KF Predictions
    ax1 = axes[0, 0]
    ax1.plot(x_eval, y_true, "k--", label=r"True: $y = 0.01x^3$", linewidth=2, alpha=0.7)
    ax1.plot(x_eval, y_pred_kf, "r-", label="KF Mean", linewidth=2)
    ax1.fill_between(
        x_eval,
        y_pred_kf - 2 * y_std_kf,
        y_pred_kf + 2 * y_std_kf,
        color="red",
        alpha=0.2,
        label=r"KF $\pm 2\sigma$",
    )
    ax1.scatter(x_train, y_train, c="black", s=10, label="Training Data", zorder=5)
    ax1.set_title("Kalman Filter (Online)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: BLL Predictions
    ax2 = axes[0, 1]
    ax2.plot(x_eval, y_true, "k--", label=r"True: $y = 0.01x^3$", linewidth=2, alpha=0.7)
    ax2.plot(x_eval, y_pred_bll, "b-", label="BLL Mean", linewidth=2)
    ax2.fill_between(
        x_eval,
        y_pred_bll - 2 * y_std_bll,
        y_pred_bll + 2 * y_std_bll,
        color="blue",
        alpha=0.2,
        label=r"BLL $\pm 2\sigma$",
    )
    ax2.scatter(x_train, y_train, c="black", s=10, label="Training Data", zorder=5)
    ax2.set_title("Bayesian Last Layer (Batch)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Overlay Comparison
    ax3 = axes[1, 0]
    ax3.plot(x_eval, y_true, "k--", label=r"True: $y = 0.01x^3$", linewidth=2, alpha=0.7)
    ax3.plot(x_eval, y_pred_kf, "r-", label="KF Mean", linewidth=2, alpha=0.7)
    ax3.plot(x_eval, y_pred_bll, "b--", label="BLL Mean", linewidth=2, alpha=0.7)
    ax3.fill_between(
        x_eval,
        y_pred_kf - 2 * y_std_kf,
        y_pred_kf + 2 * y_std_kf,
        color="red",
        alpha=0.15,
        label=r"KF $\pm 2\sigma$",
    )
    ax3.fill_between(
        x_eval,
        y_pred_bll - 2 * y_std_bll,
        y_pred_bll + 2 * y_std_bll,
        color="blue",
        alpha=0.15,
        label=r"BLL $\pm 2\sigma$",
    )
    ax3.scatter(x_train, y_train, c="black", s=10, label="Training Data", zorder=5)
    ax3.set_title("Overlay: KF vs BLL", fontsize=12, fontweight="bold")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Difference Analysis
    ax4 = axes[1, 1]
    mean_diff = jnp.abs(y_pred_kf - y_pred_bll)
    std_diff = jnp.abs(y_std_kf - y_std_bll)

    ax4.semilogy(x_eval, mean_diff, "g-", label="Mean Prediction Diff", linewidth=2)
    ax4.semilogy(x_eval, std_diff, "m-", label="Std Prediction Diff", linewidth=2)
    ax4.axhline(1e-10, color="gray", linestyle=":", label="Threshold (1e-10)")
    ax4.set_title("Prediction Differences (Log Scale)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("x")
    ax4.set_ylabel("Absolute Difference")
    ax4.legend()
    ax4.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    # Save figure
    output_path = "results/figures/BLR_KF_Validation.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    plt.show()

    # ==========================================
    # 7. SUMMARY
    # ==========================================
    print()
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Posterior means match: {mean_match}")
    print(f"Posterior stds match: {std_match}")
    print(
        f"Max prediction difference: {jnp.max(mean_diff):.2e} "
        f"(should be ~0 if Q=0 and equivalent)"
    )
    print()

    if mean_match and std_match and jnp.max(mean_diff) < 1e-8:
        print(" ✓ SUCCESS: KF with Q=0 is equivalent to batch BLR!")
    else:
        print("⚠ WARNING: Models show some differences - check parameters")

    print("=" * 60)


if __name__ == "__main__":
    main()
