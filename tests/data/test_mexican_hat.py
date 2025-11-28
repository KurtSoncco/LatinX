"""Tests for ricker wavelet dataset generator."""

import numpy as np
import pytest

from latinx.data.ricker_wavelet import RickerWaveletTranslator


def test_ricker_wavelet_initialization():
    """Test RickerWaveletTranslator initialization with default parameters."""
    translator = RickerWaveletTranslator()

    assert translator.sigma == 1.0
    assert translator.amplitude == 1.0
    assert translator.x_range == (-5.0, 5.0)
    assert translator.y_range == (-5.0, 5.0)
    assert translator.grid_size == 100
    assert translator.noise_std == 0.0
    assert translator.seed is None


def test_ricker_wavelet_custom_params():
    """Test RickerWaveletTranslator with custom parameters."""
    translator = RickerWaveletTranslator(
        sigma=2.0,
        amplitude=3.0,
        x_range=(-10, 10),
        y_range=(-8, 8),
        grid_size=50,
        noise_std=0.1,
        seed=42,
    )

    assert translator.sigma == 2.0
    assert translator.amplitude == 3.0
    assert translator.x_range == (-10, 10)
    assert translator.y_range == (-8, 8)
    assert translator.grid_size == 50
    assert translator.noise_std == 0.1
    assert translator.seed == 42


def test_ricker_wavelet_invalid_params():
    """Test that invalid parameters raise errors."""
    with pytest.raises(ValueError, match="sigma must be positive"):
        RickerWaveletTranslator(sigma=-1.0)

    with pytest.raises(ValueError, match="grid_size must be positive"):
        RickerWaveletTranslator(grid_size=0)

    with pytest.raises(ValueError, match="noise_std must be non-negative"):
        RickerWaveletTranslator(noise_std=-0.1)


def test_ricker_wavelet_generate():
    """Test data generation."""
    translator = RickerWaveletTranslator(grid_size=20, seed=42)
    data = translator.generate()

    # Check shape
    assert len(data) == 20 * 20  # grid_size^2

    # Check columns
    assert list(data.columns) == ["x", "y", "z", "z_noisy", "r"]

    # Check all values are finite
    assert data.notna().all().all()
    assert np.isfinite(data.values).all()


def test_ricker_wavelet_function_at_origin():
    """Test that ricker wavelet has maximum at origin."""
    translator = RickerWaveletTranslator(sigma=1.0, amplitude=1.0)

    # At origin (r=0), function should be 1.0
    z_origin = translator.ricker_wavelet_function(0.0, 0.0)
    assert np.isclose(z_origin, 1.0)


def test_ricker_wavelet_radial_symmetry():
    """Test that ricker wavelet is radially symmetric."""
    translator = RickerWaveletTranslator(sigma=1.5, amplitude=2.0)

    # Test at same radius, different angles
    r = 2.0
    z1 = translator.ricker_wavelet_function(r, 0.0)
    z2 = translator.ricker_wavelet_function(0.0, r)
    z3 = translator.ricker_wavelet_function(r / np.sqrt(2), r / np.sqrt(2))

    assert np.isclose(z1, z2)
    assert np.isclose(z1, z3)


def test_ricker_wavelet_noise():
    """Test that noise is added correctly."""
    translator = RickerWaveletTranslator(grid_size=50, noise_std=0.5, seed=42)
    data = translator.generate()

    # z and z_noisy should be different
    assert not np.allclose(data["z"], data["z_noisy"])

    # Noise magnitude should be reasonable
    noise = data["z_noisy"] - data["z"]
    assert np.std(noise) > 0.3  # Should be around noise_std=0.5
    assert np.std(noise) < 0.7


def test_ricker_wavelet_no_noise():
    """Test that z and z_noisy are identical when noise_std=0."""
    translator = RickerWaveletTranslator(grid_size=30, noise_std=0.0, seed=42)
    data = translator.generate()

    # Should be identical
    assert np.allclose(data["z"], data["z_noisy"])


def test_ricker_wavelet_generate_grid():
    """Test meshgrid generation for 3D plotting."""
    translator = RickerWaveletTranslator(grid_size=25, seed=42)
    X, Y, Z = translator.generate_grid()

    # Check shapes
    assert X.shape == (25, 25)
    assert Y.shape == (25, 25)
    assert Z.shape == (25, 25)

    # Check all values are finite
    assert np.isfinite(X).all()
    assert np.isfinite(Y).all()
    assert np.isfinite(Z).all()


def test_ricker_wavelet_reproducibility():
    """Test that same seed produces same results."""
    translator1 = RickerWaveletTranslator(grid_size=30, noise_std=0.1, seed=123)
    data1 = translator1.generate()

    translator2 = RickerWaveletTranslator(grid_size=30, noise_std=0.1, seed=123)
    data2 = translator2.generate()

    # Should be identical
    assert np.allclose(data1["z"], data2["z"])
    assert np.allclose(data1["z_noisy"], data2["z_noisy"])


def test_ricker_wavelet_amplitude_scaling():
    """Test that amplitude parameter scales the output correctly."""
    translator1 = RickerWaveletTranslator(amplitude=1.0, grid_size=20)
    data1 = translator1.generate()

    translator2 = RickerWaveletTranslator(amplitude=3.0, grid_size=20)
    data2 = translator2.generate()

    # z values should be scaled by amplitude
    assert np.allclose(data2["z"], 3.0 * data1["z"])
