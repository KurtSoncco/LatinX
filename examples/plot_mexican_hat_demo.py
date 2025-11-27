"""
Demonstration of Mexican Hat plotting functions.

This script shows how to use the various plotting utilities
to visualize Mexican Hat dataset data.
"""

from latinx.data.mexican_hat import MexicanHatTranslator
from latinx.plotting import (
    plot_mexican_hat,
    plot_mexican_hat_3d,
    plot_mexican_hat_contour,
    plot_mexican_hat_comparison,
)
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("Mexican Hat Plotting Demo")
    print("=" * 60)

    # Generate Mexican Hat data
    translator = MexicanHatTranslator(
        sigma=1.5,
        amplitude=2.0,
        x_range=(-5, 5),
        y_range=(-5, 5),
        grid_size=80,
        noise_std=0.15,
        seed=42,
    )

    data = translator.generate()
    print(f"\nGenerated {len(data)} data points")
    print(f"Grid size: {translator.grid_size}x{translator.grid_size}")

    # 1. Comprehensive plot (3 subplots)
    print("\n1. Creating comprehensive visualization...")
    fig1 = plot_mexican_hat(
        data,
        column="z",
        title="Mexican Hat Function - Clean Data",
        cmap="viridis",
    )

    # 2. Standalone 3D surface plot
    print("2. Creating 3D surface plot...")
    fig2 = plot_mexican_hat_3d(
        data,
        column="z",
        title="Mexican Hat - 3D View",
        cmap="plasma",
        elev=30,
        azim=60,
    )

    # 3. Standalone contour plot
    print("3. Creating contour plot...")
    fig3 = plot_mexican_hat_contour(
        data,
        column="z",
        title="Mexican Hat - Contour Levels",
        cmap="coolwarm",
        levels=25,
        show_lines=True,
    )

    # 4. Comparison plot (clean vs noisy)
    print("4. Creating comparison plot (clean vs noisy)...")
    fig4 = plot_mexican_hat_comparison(
        data,
        columns=("z", "z_noisy"),
        labels=("Clean Function", "With Noise (σ=0.15)"),
        title="Mexican Hat: Clean vs Noisy Data",
        cmap="viridis",
    )

    # 5. Plot noisy data with comprehensive view
    print("5. Creating comprehensive plot for noisy data...")
    fig5 = plot_mexican_hat(
        data,
        column="z_noisy",
        title="Mexican Hat Function - Noisy Data (σ=0.15)",
        cmap="inferno",
    )

    print("\n" + "=" * 60)
    print("All plots created successfully!")
    print("=" * 60)
    print("\nDisplaying plots...")
    print("Close plot windows to continue or exit the script.")

    # Show all plots
    plt.show()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nTo save plots, add save_path parameter:")
    print("""
    plot_mexican_hat(
        data,
        column='z',
        save_path='mexican_hat_plot.png'
    )
    """)


if __name__ == "__main__":
    main()
