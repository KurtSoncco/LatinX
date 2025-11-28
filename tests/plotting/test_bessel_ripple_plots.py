"""Tests for Bessel Ripple plotting functions."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing

from latinx.data.bessel_ripple import BesselRippleTranslator
from latinx.plotting import (
    plot_bessel_ripple,
    plot_bessel_ripple_3d,
    plot_bessel_ripple_contour,
    plot_bessel_ripple_with_uncertainty,
)


@pytest.fixture
def bessel_ripple_data():
    """Generate sample Bessel ripple data for testing."""
    translator = BesselRippleTranslator(
        k=6.0,
        amplitude=1.0,
        damping=0.05,
        grid_size=20,
        noise_std=0.05,
        use_bessel=True,
        seed=42,
    )
    return translator.generate()


@pytest.fixture
def sample_predictions(bessel_ripple_data):
    """Generate sample predictions for testing uncertainty plots."""
    n = len(bessel_ripple_data)
    predictions = np.random.randn(n) * 0.1 + bessel_ripple_data["z"].values
    uncertainty = np.abs(np.random.randn(n) * 0.05)
    return predictions, uncertainty


def test_plot_bessel_ripple(bessel_ripple_data):
    """Test comprehensive Bessel ripple plot."""
    fig = plot_bessel_ripple(bessel_ripple_data, column="z", cmap="plasma")

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 5  # 3 subplots + 2 colorbars

    plt.close(fig)


def test_plot_bessel_ripple_with_noisy_column(bessel_ripple_data):
    """Test plot with noisy data column."""
    fig = plot_bessel_ripple(bessel_ripple_data, column="z_noisy", cmap="viridis")

    assert fig is not None
    assert isinstance(fig, plt.Figure)

    plt.close(fig)


def test_plot_bessel_ripple_3d(bessel_ripple_data):
    """Test standalone 3D surface plot."""
    fig = plot_bessel_ripple_3d(bessel_ripple_data, column="z", cmap="plasma", elev=35, azim=50)

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1

    plt.close(fig)


def test_plot_bessel_ripple_contour(bessel_ripple_data):
    """Test standalone contour plot."""
    fig = plot_bessel_ripple_contour(
        bessel_ripple_data, column="z", cmap="plasma", levels=40, show_lines=True
    )

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1

    plt.close(fig)


def test_plot_bessel_ripple_contour_no_lines(bessel_ripple_data):
    """Test contour plot without lines."""
    fig = plot_bessel_ripple_contour(bessel_ripple_data, column="z", show_lines=False)

    assert fig is not None
    plt.close(fig)


def test_plot_bessel_ripple_with_uncertainty(bessel_ripple_data, sample_predictions):
    """Test uncertainty visualization plot."""
    predictions, uncertainty = sample_predictions

    fig = plot_bessel_ripple_with_uncertainty(
        bessel_ripple_data,
        predictions=predictions,
        uncertainty=uncertainty,
        cmap="plasma",
    )

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 3  # Three 3D plots

    plt.close(fig)


def test_plot_bessel_ripple_custom_title(bessel_ripple_data):
    """Test plot with custom title."""
    custom_title = "My Custom Bessel Ripple Plot"
    fig = plot_bessel_ripple(bessel_ripple_data, title=custom_title)

    assert fig is not None
    assert fig._suptitle.get_text() == custom_title

    plt.close(fig)


def test_plot_bessel_ripple_different_cmaps(bessel_ripple_data):
    """Test plot with different colormaps."""
    cmaps = ["viridis", "plasma", "inferno", "coolwarm", "RdYlBu"]

    for cmap in cmaps:
        fig = plot_bessel_ripple_3d(bessel_ripple_data, cmap=cmap)
        assert fig is not None
        plt.close(fig)


def test_plot_bessel_ripple_simple_vs_bessel():
    """Test plotting both Bessel and simple approximation."""
    # Generate Bessel data
    translator_bessel = BesselRippleTranslator(
        k=6.0,
        amplitude=1.0,
        grid_size=15,
        use_bessel=True,
        seed=42,
    )
    data_bessel = translator_bessel.generate()

    # Generate simple approximation data
    translator_simple = BesselRippleTranslator(
        k=6.0,
        amplitude=1.0,
        grid_size=15,
        use_bessel=False,
        seed=42,
    )
    data_simple = translator_simple.generate()

    # Plot both
    fig1 = plot_bessel_ripple_3d(data_bessel, title="Bessel j0")
    fig2 = plot_bessel_ripple_3d(data_simple, title="sin(kr)/r")

    assert fig1 is not None
    assert fig2 is not None

    plt.close(fig1)
    plt.close(fig2)


def test_plot_functions_close_properly(bessel_ripple_data, sample_predictions):
    """Test that plots can be created and closed without errors."""
    predictions, uncertainty = sample_predictions

    # Create multiple plots
    fig1 = plot_bessel_ripple(bessel_ripple_data)
    fig2 = plot_bessel_ripple_3d(bessel_ripple_data)
    fig3 = plot_bessel_ripple_contour(bessel_ripple_data)
    fig4 = plot_bessel_ripple_with_uncertainty(bessel_ripple_data, predictions, uncertainty)

    # Close all
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)

    # Close any remaining figures
    plt.close("all")

    # Verify no open figures
    assert len(plt.get_fignums()) == 0


def test_plot_bessel_ripple_radial_profile(bessel_ripple_data):
    """Test that radial profile subplot is created correctly."""
    fig = plot_bessel_ripple(bessel_ripple_data, column="z")

    # Find the radial profile axis (has "Radial" in title)
    radial_ax = None
    for ax in fig.axes:
        if ax.get_title() and "Radial" in ax.get_title():
            radial_ax = ax
            break

    assert radial_ax is not None, "Radial profile subplot not found"

    # Check that it has data
    assert len(radial_ax.lines) > 0

    # Check axis labels
    assert "Radial Distance" in radial_ax.get_xlabel() or "r" in radial_ax.get_xlabel()

    plt.close(fig)


def test_plot_bessel_ripple_with_high_k(bessel_ripple_data):
    """Test plotting with high k value (many ripples)."""
    translator = BesselRippleTranslator(
        k=12.0,  # High k = many ripples
        amplitude=1.0,
        grid_size=30,
        use_bessel=True,
        seed=42,
    )
    data = translator.generate()

    fig = plot_bessel_ripple(data, column="z", cmap="plasma")

    assert fig is not None
    plt.close(fig)


def test_plot_bessel_ripple_with_damping():
    """Test plotting with different damping values."""
    dampings = [0.0, 0.05, 0.2]

    for damping in dampings:
        translator = BesselRippleTranslator(
            k=6.0,
            amplitude=1.0,
            damping=damping,
            grid_size=20,
            use_bessel=True,
            seed=42,
        )
        data = translator.generate()
        fig = plot_bessel_ripple_3d(data, title=f"Damping={damping}")

        assert fig is not None
        plt.close(fig)
