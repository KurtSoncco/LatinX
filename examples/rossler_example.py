"""
Example usage of the Rössler system dataset generator.

This script demonstrates how to generate and visualize Rössler system trajectories.
"""

import matplotlib.pyplot as plt

from latinx.data.rossler import RosslerTranslator


def main():
    # Create Rössler system generator with standard chaotic parameters
    print("Generating Rössler system trajectory...")
    rossler = RosslerTranslator(
        n_steps=5000,
        dt=0.01,
        a=0.2,
        b=0.2,
        c=5.7,
        initial_state=[1.0, 0.0, 0.0],
        noise_pct=0.02,  # 2% noise relative to state magnitude
        seed=42,
    )

    print(rossler)

    # Generate trajectory with transient removal
    df = rossler.generate_with_transient_removal(n_transient=1000)

    print(f"\nGenerated {len(df)} points")
    print(f"Time range: [{df['t'].min():.2f}, {df['t'].max():.2f}]")
    print(f"X range: [{df['x'].min():.2f}, {df['x'].max():.2f}]")
    print(f"Y range: [{df['y'].min():.2f}, {df['y'].max():.2f}]")
    print(f"Z range: [{df['z'].min():.2f}, {df['z'].max():.2f}]")

    # Create visualization
    fig = plt.figure(figsize=(16, 10))

    # 3D trajectory (clean)
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    ax1.plot(df["x"], df["y"], df["z"], linewidth=0.5, alpha=0.8)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("Rössler Attractor (Clean)")

    # 3D trajectory (noisy)
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    ax2.plot(df["x_noisy"], df["y_noisy"], df["z_noisy"], linewidth=0.5, alpha=0.8)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("Rössler Attractor (Noisy, 2%)")

    # X-Y projection
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(df["x"], df["y"], linewidth=0.5, alpha=0.8)
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_title("X-Y Projection")
    ax3.grid(True, alpha=0.3)

    # Time series: X
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(df["t"], df["x"], label="Clean", linewidth=1, alpha=0.8)
    ax4.plot(df["t"], df["x_noisy"], label="Noisy", linewidth=0.5, alpha=0.6)
    ax4.set_xlabel("Time")
    ax4.set_ylabel("X")
    ax4.set_title("X Time Series")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Time series: Y
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(df["t"], df["y"], label="Clean", linewidth=1, alpha=0.8)
    ax5.plot(df["t"], df["y_noisy"], label="Noisy", linewidth=0.5, alpha=0.6)
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Y")
    ax5.set_title("Y Time Series")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Time series: Z
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(df["t"], df["z"], label="Clean", linewidth=1, alpha=0.8)
    ax6.plot(df["t"], df["z_noisy"], label="Noisy", linewidth=0.5, alpha=0.6)
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Z")
    ax6.set_title("Z Time Series")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/figures/rossler_example.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved to: results/figures/rossler_example.png")
    plt.show()


if __name__ == "__main__":
    main()
