"""Tests for Mexican Hat plotting functions."""

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing

from latinx.data.mexican_hat import MexicanHatTranslator
from latinx.plotting import (
    plot_mexican_hat,
    plot_mexican_hat_3d,
    plot_mexican_hat_contour,
    plot_mexican_hat_comparison,
)


@pytest.fixture
def mexican_hat_data():
    """Generate sample Mexican Hat data for testing."""
    translator = MexicanHatTranslator(
        sigma=1.5, amplitude=2.0, grid_size=20, noise_std=0.1, seed=42
    )
    return translator.generate()


def test_plot_mexican_hat(mexican_hat_data):
    """Test comprehensive Mexican Hat plot."""
    fig = plot_mexican_hat(mexican_hat_data, column="z", cmap="viridis")

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 5  # 3 subplots + 2 colorbars

    plt.close(fig)


def test_plot_mexican_hat_with_noisy_column(mexican_hat_data):
    """Test plot with noisy data column."""
    fig = plot_mexican_hat(mexican_hat_data, column="z_noisy", cmap="plasma")

    assert fig is not None
    assert isinstance(fig, plt.Figure)

    plt.close(fig)


def test_plot_mexican_hat_3d(mexican_hat_data):
    """Test standalone 3D surface plot."""
    fig = plot_mexican_hat_3d(
        mexican_hat_data, column="z", cmap="viridis", elev=30, azim=45
    )

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1

    plt.close(fig)


def test_plot_mexican_hat_contour(mexican_hat_data):
    """Test standalone contour plot."""
    fig = plot_mexican_hat_contour(
        mexican_hat_data, column="z", cmap="coolwarm", levels=15, show_lines=True
    )

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1

    plt.close(fig)


def test_plot_mexican_hat_contour_no_lines(mexican_hat_data):
    """Test contour plot without lines."""
    fig = plot_mexican_hat_contour(
        mexican_hat_data, column="z", show_lines=False
    )

    assert fig is not None
    plt.close(fig)


def test_plot_mexican_hat_comparison(mexican_hat_data):
    """Test comparison plot of clean vs noisy."""
    fig = plot_mexican_hat_comparison(
        mexican_hat_data,
        columns=("z", "z_noisy"),
        labels=("Clean", "Noisy"),
        cmap="viridis",
    )

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 2  # Two 3D plots

    plt.close(fig)


def test_plot_mexican_hat_custom_title(mexican_hat_data):
    """Test plot with custom title."""
    custom_title = "My Custom Mexican Hat Plot"
    fig = plot_mexican_hat(mexican_hat_data, title=custom_title)

    assert fig is not None
    assert fig._suptitle.get_text() == custom_title

    plt.close(fig)


def test_plot_mexican_hat_different_cmaps(mexican_hat_data):
    """Test plot with different colormaps."""
    cmaps = ["viridis", "plasma", "inferno", "coolwarm", "RdYlBu"]

    for cmap in cmaps:
        fig = plot_mexican_hat_3d(mexican_hat_data, cmap=cmap)
        assert fig is not None
        plt.close(fig)


def test_plot_functions_close_properly(mexican_hat_data):
    """Test that plots can be created and closed without errors."""
    # Create multiple plots
    fig1 = plot_mexican_hat(mexican_hat_data)
    fig2 = plot_mexican_hat_3d(mexican_hat_data)
    fig3 = plot_mexican_hat_contour(mexican_hat_data)
    fig4 = plot_mexican_hat_comparison(mexican_hat_data)

    # Close all
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)

    # Close any remaining figures
    plt.close("all")

    # Verify no open figures
    assert len(plt.get_fignums()) == 0
