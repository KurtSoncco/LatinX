"""
Plotting functions for ricker wavelet (Laplacian of Gaussian) datasets.

This module provides various visualization utilities for ricker wavelet data,
including 3D surface plots, contour plots, and comparison visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional


def plot_ricker_wavelet(
    data: pd.DataFrame,
    column: str = "z",
    figsize: Tuple[int, int] = (14, 5),
    title: Optional[str] = None,
    cmap: str = "viridis",
    save_path: Optional[str] = None,
):
    """
    Create a comprehensive visualization of ricker wavelet data with 3D and 2D views.

    Displays three subplots:
    1. 3D surface plot
    2. 2D contour plot
    3. Radial profile (z vs r)

    Args:
        data: DataFrame with columns 'x', 'y', and the specified z column.
        column: Name of the z-value column to plot (default: 'z').
        figsize: Figure size as (width, height) tuple.
        title: Overall figure title. If None, uses "ricker wavelet Function".
        cmap: Colormap for surface and contour plots.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure object.
    """
    if title is None:
        title = f"ricker wavelet Function ({column})"

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
    ax1.set_title("3D Surface", fontsize=11)
    ax1.view_init(elev=25, azim=45)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

    # 2. Contour plot
    ax2 = fig.add_subplot(132)
    contour = ax2.contourf(X, Y, Z, levels=20, cmap=cmap)
    ax2.contour(X, Y, Z, levels=20, colors="black", alpha=0.3, linewidths=0.5)
    ax2.set_xlabel("X", fontsize=10)
    ax2.set_ylabel("Y", fontsize=10)
    ax2.set_title("Contour Plot", fontsize=11)
    ax2.set_aspect("equal")
    fig.colorbar(contour, ax=ax2)

    # 3. Radial profile
    ax3 = fig.add_subplot(133)
    # Sample points along a radial line
    r_values = data["r"].values
    z_values = data[column].values

    # Bin by radius for cleaner plot
    r_bins = np.linspace(0, r_values.max(), 50)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    z_binned = []
    for i in range(len(r_bins) - 1):
        mask = (r_values >= r_bins[i]) & (r_values < r_bins[i + 1])
        if mask.any():
            z_binned.append(z_values[mask].mean())
        else:
            z_binned.append(np.nan)

    ax3.plot(r_centers, z_binned, "o-", markersize=3, linewidth=1.5, label=column)
    ax3.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax3.set_xlabel("Radial Distance (r)", fontsize=10)
    ax3.set_ylabel("Z", fontsize=10)
    ax3.set_title("Radial Profile", fontsize=11)
    ax3.grid(alpha=0.3)
    ax3.legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


def plot_ricker_wavelet_3d(
    data: pd.DataFrame,
    column: str = "z",
    figsize: Tuple[int, int] = (8, 6),
    title: Optional[str] = None,
    cmap: str = "viridis",
    elev: int = 25,
    azim: int = 45,
    save_path: Optional[str] = None,
):
    """
    Create a standalone 3D surface plot of ricker wavelet data.

    Args:
        data: DataFrame with columns 'x', 'y', and the specified z column.
        column: Name of the z-value column to plot (default: 'z').
        figsize: Figure size as (width, height) tuple.
        title: Plot title. If None, uses "ricker wavelet - 3D Surface".
        cmap: Colormap for the surface.
        elev: Elevation angle for 3D view.
        azim: Azimuth angle for 3D view.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure object.
    """
    if title is None:
        title = f"ricker wavelet - 3D Surface ({column})"

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


def plot_ricker_wavelet_contour(
    data: pd.DataFrame,
    column: str = "z",
    figsize: Tuple[int, int] = (7, 6),
    title: Optional[str] = None,
    cmap: str = "viridis",
    levels: int = 20,
    show_lines: bool = True,
    save_path: Optional[str] = None,
):
    """
    Create a standalone contour plot of ricker wavelet data.

    Args:
        data: DataFrame with columns 'x', 'y', and the specified z column.
        column: Name of the z-value column to plot (default: 'z').
        figsize: Figure size as (width, height) tuple.
        title: Plot title. If None, uses "ricker wavelet - Contour".
        cmap: Colormap for filled contours.
        levels: Number of contour levels.
        show_lines: If True, overlay black contour lines.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure object.
    """
    if title is None:
        title = f"ricker wavelet - Contour ({column})"

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
        ax.contour(X, Y, Z, levels=levels, colors="black", alpha=0.3, linewidths=0.5)

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


def plot_ricker_wavelet_comparison(
    data: pd.DataFrame,
    columns: Tuple[str, str] = ("z", "z_noisy"),
    labels: Optional[Tuple[str, str]] = None,
    figsize: Tuple[int, int] = (14, 5),
    title: Optional[str] = None,
    cmap: str = "viridis",
    save_path: Optional[str] = None,
):
    """
    Compare two ricker wavelet datasets side-by-side (e.g., clean vs noisy).

    Creates a figure with two 3D surface plots for comparison.

    Args:
        data: DataFrame with columns 'x', 'y', and both z columns.
        columns: Tuple of two column names to compare (default: ('z', 'z_noisy')).
        labels: Custom labels for the two plots. If None, uses column names.
        figsize: Figure size as (width, height) tuple.
        title: Overall figure title.
        cmap: Colormap for both surfaces.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure object.
    """
    if labels is None:
        labels = columns

    if title is None:
        title = f"ricker wavelet Comparison: {labels[0]} vs {labels[1]}"

    # Infer grid size and reshape
    n_unique_x = data["x"].nunique()
    n_unique_y = data["y"].nunique()

    X = data["x"].values.reshape(n_unique_y, n_unique_x)
    Y = data["y"].values.reshape(n_unique_y, n_unique_x)
    Z1 = data[columns[0]].values.reshape(n_unique_y, n_unique_x)
    Z2 = data[columns[1]].values.reshape(n_unique_y, n_unique_x)

    # Create figure with two 3D subplots
    fig = plt.figure(figsize=figsize)

    # First plot
    ax1 = fig.add_subplot(121, projection="3d")
    surf1 = ax1.plot_surface(X, Y, Z1, cmap=cmap, linewidth=0, antialiased=True, alpha=0.9)
    ax1.set_xlabel("X", fontsize=10)
    ax1.set_ylabel("Y", fontsize=10)
    ax1.set_zlabel("Z", fontsize=10)
    ax1.set_title(labels[0], fontsize=12, fontweight="bold")
    ax1.view_init(elev=25, azim=45)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # Second plot
    ax2 = fig.add_subplot(122, projection="3d")
    surf2 = ax2.plot_surface(X, Y, Z2, cmap=cmap, linewidth=0, antialiased=True, alpha=0.9)
    ax2.set_xlabel("X", fontsize=10)
    ax2.set_ylabel("Y", fontsize=10)
    ax2.set_zlabel("Z", fontsize=10)
    ax2.set_title(labels[1], fontsize=12, fontweight="bold")
    ax2.view_init(elev=25, azim=45)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig
