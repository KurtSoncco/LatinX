import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_bll_vs_full_nn(
    data: pd.DataFrame,
    bll_predictions: np.ndarray,
    nn_predictions: np.ndarray,
    figsize: tuple[int, int] = (18, 5),
    title: str | None = None,
    cmap: str = "plasma",
    save_path: str | None = None,
):
    """
    Visualize ground truth, Bayesian Last Layer predictions, and Full NN predictions in 3D.

    Creates three side by side 3D surface plots:
    1. Ground truth (z)
    2. Bayesian Last Layer predictions
    3. Full neural network predictions

    Args:
        data: DataFrame with columns 'x', 'y', 'z'.
        bll_predictions: Predictions from BayesianLastLayer.predict(...), shape (N,).
        nn_predictions: Predictions from BayesianLastLayer.predict_full_nn(...), shape (N,).
        figsize: Figure size as (width, height) tuple.
        title: Overall figure title.
        cmap: Colormap for the surfaces.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure object.
    """
    if title is None:
        title = "Ground Truth vs Bayesian Last Layer vs Full NN"

    # Infer grid size and reshape
    n_unique_x = data["x"].nunique()
    n_unique_y = data["y"].nunique()

    X = data["x"].values.reshape(n_unique_y, n_unique_x)
    Y = data["y"].values.reshape(n_unique_y, n_unique_x)
    Z_true = data["z"].values.reshape(n_unique_y, n_unique_x)
    Z_bll = bll_predictions.reshape(n_unique_y, n_unique_x)
    Z_nn = nn_predictions.reshape(n_unique_y, n_unique_x)

    # Shared color scale for fair visual comparison
    z_min = min(Z_true.min(), Z_bll.min(), Z_nn.min())
    z_max = max(Z_true.max(), Z_bll.max(), Z_nn.max())

    fig = plt.figure(figsize=figsize)

    # 1. Ground truth
    ax1 = fig.add_subplot(131, projection="3d")
    surf1 = ax1.plot_surface(
        X, Y, Z_true, cmap=cmap, linewidth=0, antialiased=True, vmin=z_min, vmax=z_max
    )
    ax1.set_xlabel("X", fontsize=9)
    ax1.set_ylabel("Y", fontsize=9)
    ax1.set_zlabel("Z", fontsize=9)
    ax1.set_title("Ground Truth", fontsize=11, fontweight="bold")
    ax1.view_init(elev=25, azim=45)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # 2. Bayesian Last Layer
    ax2 = fig.add_subplot(132, projection="3d")
    surf2 = ax2.plot_surface(
        X, Y, Z_bll, cmap=cmap, linewidth=0, antialiased=True, vmin=z_min, vmax=z_max
    )
    ax2.set_xlabel("X", fontsize=9)
    ax2.set_ylabel("Y", fontsize=9)
    ax2.set_zlabel("Z", fontsize=9)
    ax2.set_title("Bayesian Last Layer", fontsize=11, fontweight="bold")
    ax2.view_init(elev=25, azim=45)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    # 3. Full NN
    ax3 = fig.add_subplot(133, projection="3d")
    surf3 = ax3.plot_surface(
        X, Y, Z_nn, cmap=cmap, linewidth=0, antialiased=True, vmin=z_min, vmax=z_max
    )
    ax3.set_xlabel("X", fontsize=9)
    ax3.set_ylabel("Y", fontsize=9)
    ax3.set_zlabel("Z", fontsize=9)
    ax3.set_title("Full Neural Network", fontsize=11, fontweight="bold")
    ax3.view_init(elev=25, azim=45)
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig
