"""
Plotting functions for Bessel Ripple (water droplet wave) datasets.

This module provides various visualization utilities for Bessel ripple data,
including 3D surface plots, contour plots, and comparison visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional


def plot_bessel_ripple(
    data: pd.DataFrame,
    column: str = "z",
    figsize: Tuple[int, int] = (16, 5),
    title: Optional[str] = None,
    cmap: str = "plasma",
    save_path: Optional[str] = None,
):
    """
    Create a comprehensive visualization of Bessel ripple data with 3D and 2D views.

    Displays three subplots:
    1. 3D surface plot
    2. 2D contour plot
    3. Radial profile (z vs r) showing oscillations

    Args:
        data: DataFrame with columns 'x', 'y', and the specified z column.
        column: Name of the z-value column to plot (default: 'z').
        figsize: Figure size as (width, height) tuple.
        title: Overall figure title. If None, uses "Bessel Ripple Wave".
        cmap: Colormap for surface and contour plots.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure object.
    """
    if title is None:
        title = f"Bessel Ripple Wave ({column})"

    # Infer grid size
    n_unique_x = data["x"].nunique()
    n_unique_y = data["y"].nunique()

    # Reshape data to grid
    X = data["x"].values.reshape(n_unique_y, n_unique_x)
    Y = data["y"].values.reshape(n_unique_y, n_unique_x)
    Z = data[column].values.reshape(n_unique_y, n_unique_x)

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)

    # 1. 3D Surface plot
    ax1 = fig.add_subplot(131, projection="3d")
    surf = ax1.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.9)
    ax1.set_xlabel("X", fontsize=10)
    ax1.set_ylabel("Y", fontsize=10)
    ax1.set_zlabel("Z", fontsize=10)
    ax1.set_title("3D Surface (Ripple Pattern)", fontsize=11)
    ax1.view_init(elev=30, azim=45)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

    # 2. Contour plot
    ax2 = fig.add_subplot(132)
    contour = ax2.contourf(X, Y, Z, levels=30, cmap=cmap)
    ax2.contour(X, Y, Z, levels=30, colors="black", alpha=0.2, linewidths=0.5)
    ax2.set_xlabel("X", fontsize=10)
    ax2.set_ylabel("Y", fontsize=10)
    ax2.set_title("Contour Plot (Top View)", fontsize=11)
    ax2.set_aspect("equal")
    fig.colorbar(contour, ax=ax2)

    # 3. Radial profile showing oscillations
    ax3 = fig.add_subplot(133)
    r_values = data["r"].values
    z_values = data[column].values

    # Sort by radius for clean line plot
    sort_idx = np.argsort(r_values)
    r_sorted = r_values[sort_idx]
    z_sorted = z_values[sort_idx]

    # Subsample for cleaner plot (every nth point)
    n_points = len(r_sorted)
    step = max(1, n_points // 500)  # Keep ~500 points

    ax3.plot(r_sorted[::step], z_sorted[::step], "-", linewidth=1, alpha=0.7, label=column)
    ax3.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax3.set_xlabel("Radial Distance (r)", fontsize=10)
    ax3.set_ylabel("Z", fontsize=10)
    ax3.set_title("Radial Profile (Oscillations)", fontsize=11)
    ax3.grid(alpha=0.3)
    ax3.legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


def plot_bessel_ripple_3d(
    data: pd.DataFrame,
    column: str = "z",
    figsize: Tuple[int, int] = (9, 7),
    title: Optional[str] = None,
    cmap: str = "plasma",
    elev: int = 30,
    azim: int = 45,
    save_path: Optional[str] = None,
):
    """
    Create a standalone 3D surface plot of Bessel ripple data.

    Args:
        data: DataFrame with columns 'x', 'y', and the specified z column.
        column: Name of the z-value column to plot (default: 'z').
        figsize: Figure size as (width, height) tuple.
        title: Plot title. If None, uses "Bessel Ripple - 3D Surface".
        cmap: Colormap for the surface.
        elev: Elevation angle for 3D view.
        azim: Azimuth angle for 3D view.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure object.
    """
    if title is None:
        title = f"Bessel Ripple - 3D Surface ({column})"

    # Infer grid size and reshape
    n_unique_x = data["x"].nunique()
    n_unique_y = data["y"].nunique()

    X = data["x"].values.reshape(n_unique_y, n_unique_x)
    Y = data["y"].values.reshape(n_unique_y, n_unique_x)
    Z = data[column].values.reshape(n_unique_y, n_unique_x)

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.9)

    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    ax.set_zlabel("Z", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.view_init(elev=elev, azim=azim)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


def plot_bessel_ripple_contour(
    data: pd.DataFrame,
    column: str = "z",
    figsize: Tuple[int, int] = (8, 7),
    title: Optional[str] = None,
    cmap: str = "plasma",
    levels: int = 30,
    show_lines: bool = True,
    save_path: Optional[str] = None,
):
    """
    Create a standalone contour plot of Bessel ripple data.

    Args:
        data: DataFrame with columns 'x', 'y', and the specified z column.
        column: Name of the z-value column to plot (default: 'z').
        figsize: Figure size as (width, height) tuple.
        title: Plot title. If None, uses "Bessel Ripple - Contour".
        cmap: Colormap for filled contours.
        levels: Number of contour levels.
        show_lines: If True, overlay black contour lines.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure object.
    """
    if title is None:
        title = f"Bessel Ripple - Contour ({column})"

    # Infer grid size and reshape
    n_unique_x = data["x"].nunique()
    n_unique_y = data["y"].nunique()

    X = data["x"].values.reshape(n_unique_y, n_unique_x)
    Y = data["y"].values.reshape(n_unique_y, n_unique_x)
    Z = data[column].values.reshape(n_unique_y, n_unique_x)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Filled contours
    contourf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)

    # Optional contour lines
    if show_lines:
        ax.contour(X, Y, Z, levels=levels, colors="black", alpha=0.2, linewidths=0.5)

    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_aspect("equal")

    plt.colorbar(contourf, ax=ax, label="Z")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


def plot_bessel_ripple_with_uncertainty(
    data: pd.DataFrame,
    predictions: np.ndarray,
    uncertainty: np.ndarray,
    figsize: Tuple[int, int] = (18, 5),
    title: Optional[str] = None,
    cmap: str = "plasma",
    save_path: Optional[str] = None,
):
    """
    Visualize Bessel ripple with model predictions and uncertainty.

    Creates three side-by-side plots:
    1. Ground truth (z)
    2. Model predictions
    3. Prediction uncertainty

    Args:
        data: DataFrame with columns 'x', 'y', 'z'.
        predictions: Model predictions (same shape as data).
        uncertainty: Prediction uncertainty/std (same shape as data).
        figsize: Figure size as (width, height) tuple.
        title: Overall figure title.
        cmap: Colormap for the plots.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure object.
    """
    if title is None:
        title = "Bessel Ripple: Ground Truth, Predictions, and Uncertainty"

    # Infer grid size and reshape
    n_unique_x = data["x"].nunique()
    n_unique_y = data["y"].nunique()

    X = data["x"].values.reshape(n_unique_y, n_unique_x)
    Y = data["y"].values.reshape(n_unique_y, n_unique_x)
    Z_true = data["z"].values.reshape(n_unique_y, n_unique_x)
    Z_pred = predictions.reshape(n_unique_y, n_unique_x)
    Z_unc = uncertainty.reshape(n_unique_y, n_unique_x)

    # Create figure with three 3D subplots
    fig = plt.figure(figsize=figsize)

    # 1. Ground truth
    ax1 = fig.add_subplot(131, projection="3d")
    surf1 = ax1.plot_surface(X, Y, Z_true, cmap=cmap, linewidth=0, antialiased=True)
    ax1.set_xlabel("X", fontsize=9)
    ax1.set_ylabel("Y", fontsize=9)
    ax1.set_zlabel("Z", fontsize=9)
    ax1.set_title("Ground Truth", fontsize=11, fontweight="bold")
    ax1.view_init(elev=25, azim=45)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # 2. Predictions
    ax2 = fig.add_subplot(132, projection="3d")
    surf2 = ax2.plot_surface(X, Y, Z_pred, cmap=cmap, linewidth=0, antialiased=True)
    ax2.set_xlabel("X", fontsize=9)
    ax2.set_ylabel("Y", fontsize=9)
    ax2.set_zlabel("Z", fontsize=9)
    ax2.set_title("Model Predictions", fontsize=11, fontweight="bold")
    ax2.view_init(elev=25, azim=45)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    # 3. Uncertainty
    ax3 = fig.add_subplot(133, projection="3d")
    surf3 = ax3.plot_surface(X, Y, Z_unc, cmap="hot", linewidth=0, antialiased=True)
    ax3.set_xlabel("X", fontsize=9)
    ax3.set_ylabel("Y", fontsize=9)
    ax3.set_zlabel("Std", fontsize=9)
    ax3.set_title("Prediction Uncertainty", fontsize=11, fontweight="bold")
    ax3.view_init(elev=25, azim=45)
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig
