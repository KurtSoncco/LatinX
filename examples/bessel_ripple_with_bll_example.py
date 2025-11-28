"""
Example: Using Bessel Ripple dataset for ML tasks with visualization.

Demonstrates:
1. Generating Bessel ripple data (water droplet pattern)
2. Comparing Bessel vs simple sin(kr)/r implementations
3. Using it with Bayesian Last Layer model
4. Visualizing results with plotting functions
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from latinx.data.bessel_ripple import BesselRippleTranslator
from latinx.models.bayesian_last_layer import BayesianLastLayer
from latinx.plotting import (
    plot_bessel_ripple,
    plot_bessel_ripple_3d,
    plot_bessel_ripple_contour,
    plot_bessel_ripple_with_uncertainty,
    plot_bll_vs_full_nn,
)


def main():
    print("=" * 60)
    print("Bessel Ripple Dataset Example with Visualization")
    print("=" * 60)

    # 1. Generate Bessel ripple data
    print("\n1. Generating Bessel ripple data...")
    translator = BesselRippleTranslator(
        k=6.0,  # Higher k = more ripples
        amplitude=1.0,
        damping=0.05,  # Slight decay with distance
        x_range=(-8, 8),
        y_range=(-8, 8),
        grid_size=60,  # 60x60 = 3600 points
        noise_std=0.01,
        use_bessel=True,  # Use accurate spherical Bessel function
        seed=42,
    )

    data = translator.generate()
    print(f"Generated {len(data)} data points")
    print(f"Grid size: {translator.grid_size}x{translator.grid_size}")

    # 2. Visualize clean Bessel ripple
    print("\n2. Creating comprehensive visualization of clean ripple...")
    plot_bessel_ripple(
        data,
        column="z",
        title="Bessel Ripple Wave - Clean Data (k=6.0, damping=0.05)",
        cmap="plasma",
    )

    # 3. Compare Bessel vs simple implementation
    print("\n3. Comparing Bessel vs simple sin(kr)/r implementation...")
    translator_simple = BesselRippleTranslator(
        k=6.0,
        amplitude=1.0,
        damping=0.05,
        x_range=(-8, 8),
        y_range=(-8, 8),
        grid_size=60,
        noise_std=0.0,  # No noise for fair comparison
        use_bessel=False,  # Use sin(kr)/r approximation
        seed=42,
    )

    data_simple = translator_simple.generate()

    # Show numerical comparison
    sample_indices = [100, 500, 1000, 2000]
    print("\nComparison at sample points:")
    print(f"{'Index':<10} {'Bessel':<15} {'Simple':<15} {'Diff':<15}")
    print("-" * 55)
    for idx in sample_indices:
        z_bessel = data.loc[idx, "z"]
        z_simple = data_simple.loc[idx, "z"]
        diff = abs(z_bessel - z_simple)
        print(f"{idx:<10} {z_bessel:< 15.6f} {z_simple:< 15.6f} {diff:< 15.6f}")

    # 4. Visualize noisy data
    print("\n4. Creating visualization of noisy ripple...")
    plot_bessel_ripple(
        data,
        column="z_noisy",
        title="Bessel Ripple Wave - Noisy Data (σ=0.05)",
        cmap="viridis",
    )

    # 5. Create standalone 3D plot
    print("\n5. Creating 3D surface plot...")
    plot_bessel_ripple_3d(
        data,
        column="z",
        title="Bessel Ripple - 3D View",
        cmap="plasma",
        elev=35,
        azim=50,
    )

    # 6. Create contour plot
    print("\n6. Creating contour plot...")
    plot_bessel_ripple_contour(
        data,
        column="z",
        title="Bessel Ripple - Contour Map",
        cmap="plasma",
        levels=40,
        show_lines=True,
    )

    # 7. Train Bayesian Last Layer
    print("\n" + "=" * 60)
    print("7. Training Bayesian Last Layer...")
    print("=" * 60)

    # Use x, y as inputs (2D) and z_noisy as target
    X = jnp.array(data[["x", "y"]].values)
    y_target = jnp.array(data["z_noisy"].values)

    bll = BayesianLastLayer(
        hidden_dims=(30, 30, 20),  # Deeper network for complex ripples
        sigma=0.01,
        alpha=0.01,  # Less regularization for flexible fit
        learning_rate=1e-3,
        n_steps=3000,
        seed=42,
    )

    loss_history = bll.fit(X, y_target)
    print(f"Initial loss: {loss_history[0]:.6f}")
    print(f"Final loss: {loss_history[-1]:.6f}")
    print(f"Loss reduction: {(1 - loss_history[-1] / loss_history[0]) * 100:.2f}%")

    # 8. Make predictions
    print("\n" + "=" * 60)
    print("8. Making predictions with uncertainty...")
    print("=" * 60)

    y_pred, y_std = bll.predict(X, return_std=True)

    rmse = np.sqrt(np.mean((y_target - y_pred) ** 2))
    print(f"RMSE on training data: {rmse:.6f}")
    print(f"Mean prediction uncertainty: {np.mean(y_std):.6f}")
    print(f"Max prediction uncertainty: {np.max(y_std):.6f}")

    # 9. Analyze uncertainty vs distance
    print("\n" + "=" * 60)
    print("9. Uncertainty vs Radial Distance...")
    print("=" * 60)

    r_bins = [(0, 2), (2, 4), (4, 6), (6, 8)]
    for r_min, r_max in r_bins:
        mask = (data["r"] >= r_min) & (data["r"] < r_max)
        # Convert pandas mask to numpy array for JAX indexing
        mask_np = mask.values
        unc_in_bin = float(y_std[mask_np].mean())
        count = int(mask.sum())
        print(f"r ∈ [{r_min}, {r_max}): uncertainty = {unc_in_bin:.6f} (n={count})")

    # 10. Visualize predictions and uncertainty
    print("\n10. Creating prediction and uncertainty visualization...")
    plot_bessel_ripple_with_uncertainty(
        data,
        predictions=y_pred,
        uncertainty=y_std,
        title="Bessel Ripple: BLL Predictions and Uncertainty",
        cmap="plasma",
    )

    print("\n11. Creating BLL vs Full NN comparison plot...")
    y_pred_nn = bll.predict_full_nn(X)
    rmse_full_nn = np.sqrt(np.mean(y_target - y_pred_nn) ** 2)

    plot_bll_vs_full_nn(
        data=data,
        bll_predictions=np.array(y_pred),
        nn_predictions=np.array(y_pred_nn),
        figsize=(18, 5),
        title="Bessel Ripple: Ground Truth vs BLL vs Full NN",
        cmap="plasma",
    )

    print(f"RMSE BLL: {rmse}, RMSE Normal NN {rmse_full_nn}")

    print("\n" + "=" * 60)
    print("All visualizations created!")
    print("=" * 60)
    print("\nDisplaying plots...")
    print("Close plot windows to continue or exit the script.")

    # Show all plots
    plt.show()


if __name__ == "__main__":
    main()
