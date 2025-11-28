"""
Example: Using ricker wavelet dataset for ML tasks with visualization.

Demonstrates:
1. Generating ricker wavelet data
2. Visualizing the 2D function with plotting utilities
3. Using it with Bayesian Last Layer model
4. Comparing predictions vs ground truth
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from latinx.data.ricker_wavelet import RickerWaveletTranslator
from latinx.models.bayesian_last_layer import BayesianLastLayer
from latinx.plotting import (
    plot_ricker_wavelet,
    plot_ricker_wavelet_3d,
    plot_ricker_wavelet_contour,
    plot_ricker_wavelet_comparison,
    plot_bll_vs_full_nn,
)


def main():
    print("=" * 60)
    print("ricker wavelet Dataset Example with Visualization")
    print("=" * 60)

    # 1. Generate ricker wavelet data
    print("\n1. Generating ricker wavelet data...")
    translator = RickerWaveletTranslator(
        sigma=1.5,
        amplitude=2.0,
        x_range=(-5, 5),
        y_range=(-5, 5),
        grid_size=50,  # 50x50 = 2500 points
        noise_std=0.1,
        seed=42,
    )

    data = translator.generate()
    print(f"Generated {len(data)} data points")
    print(f"Grid size: {translator.grid_size}x{translator.grid_size}")

    # 2. Visualize clean ricker wavelet
    print("\n2. Creating comprehensive visualization of clean function...")
    fig1 = plot_ricker_wavelet(
        data,
        column="z",
        title="ricker wavelet Function - Clean Data (σ=1.5, A=2.0)",
        cmap="viridis",
    )

    # 3. Visualize noisy data
    print("\n3. Creating visualization of noisy data...")
    fig2 = plot_ricker_wavelet(
        data,
        column="z_noisy",
        title="ricker wavelet Function - Noisy Data (noise σ=0.1)",
        cmap="plasma",
    )

    # 4. Create standalone 3D plot
    print("\n4. Creating 3D surface plot...")
    fig3 = plot_ricker_wavelet_3d(
        data,
        column="z",
        title="ricker wavelet - 3D Surface View",
        cmap="coolwarm",
        elev=25,
        azim=45,
    )

    # 5. Create contour plot
    print("\n5. Creating contour plot...")
    fig4 = plot_ricker_wavelet_contour(
        data,
        column="z",
        title="ricker wavelet - Contour Map",
        cmap="viridis",
        levels=25,
        show_lines=True,
    )

    # 6. Compare clean vs noisy
    print("\n6. Creating clean vs noisy comparison...")
    fig5 = plot_ricker_wavelet_comparison(
        data,
        columns=("z", "z_noisy"),
        labels=("Clean Function", "With Noise (σ=0.1)"),
        title="ricker wavelet: Clean vs Noisy Data Comparison",
        cmap="viridis",
    )

    # 7. Train Bayesian Last Layer on noisy data
    print("\n" + "=" * 60)
    print("7. Training Bayesian Last Layer...")
    print("=" * 60)

    # Use x, y as inputs (2D) and z_noisy as target
    X = jnp.array(data[["x", "y"]].values)
    y_target = jnp.array(data["z_noisy"].values)

    bll = BayesianLastLayer(
        hidden_dims=(20, 20),
        sigma=0.00001,
        alpha=0.00000001,
        learning_rate=1e-3,
        n_steps=2000,
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

    # 9. Add predictions to data and visualize
    print("\n9. Visualizing BLL predictions...")
    data["z_predicted"] = np.array(y_pred)

    fig6 = plot_ricker_wavelet_comparison(
        data,
        columns=("z", "z_predicted"),
        labels=("Ground Truth", "BLL Predictions"),
        title="ricker wavelet: Ground Truth vs BLL Predictions",
        cmap="plasma",
    )

    # 10. Test on larger grid (extrapolation)
    print("\n" + "=" * 60)
    print("10. Testing extrapolation...")
    print("=" * 60)

    translator_test = RickerWaveletTranslator(
        sigma=1.5,
        amplitude=2.0,
        x_range=(-7, 7),  # Larger range for extrapolation
        y_range=(-7, 7),
        grid_size=40,
        noise_std=0.0,  # No noise for clean test
        seed=123,
    )

    data_test = translator_test.generate()
    X_test = jnp.array(data_test[["x", "y"]].values)

    y_pred_test, y_std_test = bll.predict(X_test, return_std=True)

    # Analyze uncertainty in vs out of training range
    in_range = (np.abs(data_test["x"]) <= 5) & (np.abs(data_test["y"]) <= 5)
    # Convert pandas mask to numpy array for JAX indexing
    in_range_np = in_range.values
    unc_in = float(y_std_test[in_range_np].mean())
    unc_out = float(y_std_test[~in_range_np].mean())

    print(f"Mean uncertainty IN training range:  {unc_in:.6f}")
    print(f"Mean uncertainty OUT training range: {unc_out:.6f}")
    print(f"Uncertainty increase ratio: {unc_out / unc_in:.2f}x")

    # 11. Visualize extrapolation with uncertainty
    print("\n11. Visualizing extrapolation results...")
    data_test["z_predicted"] = np.array(y_pred_test)
    data_test["uncertainty"] = np.array(y_std_test)

    # Create comparison on test grid
    fig7 = plot_ricker_wavelet_comparison(
        data_test,
        columns=("z", "z_predicted"),
        labels=("True Function (Extended)", "BLL Extrapolation"),
        title="ricker wavelet: Extrapolation to Larger Domain",
        cmap="viridis",
    )

    # Visualize uncertainty on extended grid
    fig8 = plot_ricker_wavelet(
        data_test,
        column="uncertainty",
        title="Prediction Uncertainty on Extended Domain",
        cmap="hot",
    )

    print("\n11. Creating BLL vs Full NN comparison plot...")
    y_pred_nn = bll.predict_full_nn(X)
    rmse_full_nn = np.sqrt((np.mean(y_target - y_pred_nn) ** 2))

    fig_comparison_bessel = plot_bll_vs_full_nn(
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

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nKey findings:")
    print(f"- Generated {len(data)} training data points")
    print(f"- Trained BLL model achieved RMSE: {rmse:.6f}")
    print(f"- Average prediction uncertainty: {np.mean(y_std):.6f}")
    print(f"- Uncertainty increases {unc_out / unc_in:.2f}x outside training range")
    print("\nTo save plots, add save_path parameter:")
    print("""
    plot_ricker_wavelet(
        data,
        column='z',
        save_path='ricker_wavelet_plot.png'
    )
    """)


if __name__ == "__main__":
    main()
