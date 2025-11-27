"""Tests for Bessel Ripple dataset generator."""

import numpy as np
import pytest

from latinx.data.bessel_ripple import BesselRippleTranslator


def test_bessel_ripple_initialization():
    """Test BesselRippleTranslator initialization with default parameters."""
    translator = BesselRippleTranslator()

    assert translator.k == 5.0
    assert translator.amplitude == 1.0
    assert translator.damping == 0.0
    assert translator.x_range == (-10.0, 10.0)
    assert translator.y_range == (-10.0, 10.0)
    assert translator.grid_size == 100
    assert translator.noise_std == 0.0
    assert translator.use_bessel is True
    assert translator.epsilon == 1e-6
    assert translator.seed is None


def test_bessel_ripple_custom_params():
    """Test BesselRippleTranslator with custom parameters."""
    translator = BesselRippleTranslator(
        k=8.0,
        amplitude=2.0,
        damping=0.1,
        x_range=(-20, 20),
        y_range=(-15, 15),
        grid_size=50,
        noise_std=0.05,
        use_bessel=False,
        epsilon=1e-8,
        seed=42,
    )

    assert translator.k == 8.0
    assert translator.amplitude == 2.0
    assert translator.damping == 0.1
    assert translator.x_range == (-20, 20)
    assert translator.y_range == (-15, 15)
    assert translator.grid_size == 50
    assert translator.noise_std == 0.05
    assert translator.use_bessel is False
    assert translator.epsilon == 1e-8
    assert translator.seed == 42


def test_bessel_ripple_invalid_params():
    """Test that invalid parameters raise errors."""
    with pytest.raises(ValueError, match="k .* must be positive"):
        BesselRippleTranslator(k=-1.0)

    with pytest.raises(ValueError, match="grid_size must be positive"):
        BesselRippleTranslator(grid_size=0)

    with pytest.raises(ValueError, match="noise_std must be non-negative"):
        BesselRippleTranslator(noise_std=-0.1)

    with pytest.raises(ValueError, match="damping must be non-negative"):
        BesselRippleTranslator(damping=-0.5)

    with pytest.raises(ValueError, match="epsilon must be positive"):
        BesselRippleTranslator(epsilon=-1e-6)


def test_bessel_ripple_generate():
    """Test data generation."""
    translator = BesselRippleTranslator(grid_size=20, seed=42)
    data = translator.generate()

    # Check shape
    assert len(data) == 20 * 20  # grid_size^2

    # Check columns
    assert list(data.columns) == ["x", "y", "z", "z_noisy", "r"]

    # Check all values are finite
    assert data.notna().all().all()
    assert np.isfinite(data.values).all()


def test_bessel_ripple_at_origin():
    """Test that Bessel function has correct value at origin."""
    translator = BesselRippleTranslator(k=5.0, amplitude=1.0, use_bessel=True)

    # At origin, j_0(0) = 1.0
    z_origin = translator.ripple_function_bessel(0.0, 0.0)
    assert np.isclose(z_origin, 1.0)


def test_bessel_ripple_radial_symmetry():
    """Test that ripple is radially symmetric."""
    translator = BesselRippleTranslator(k=6.0, amplitude=2.0)

    # Test at same radius, different angles
    r = 3.0
    z1 = translator.ripple_function(r, 0.0)
    z2 = translator.ripple_function(0.0, r)
    z3 = translator.ripple_function(r / np.sqrt(2), r / np.sqrt(2))

    assert np.isclose(z1, z2)
    assert np.isclose(z1, z3)


def test_bessel_vs_simple():
    """Test that both Bessel and simple implementations work correctly."""
    k = 5.0

    translator_bessel = BesselRippleTranslator(k=k, use_bessel=True)
    translator_simple = BesselRippleTranslator(k=k, use_bessel=False)

    # Test at a single point
    x, y = 2.0, 0.0
    z_bessel = translator_bessel.ripple_function(x, y)
    z_simple = translator_simple.ripple_function(x, y)

    # Both should produce finite values
    assert np.isfinite(z_bessel)
    assert np.isfinite(z_simple)

    # Both should produce decaying oscillations
    # Test on a grid
    x_arr = np.array([1.0, 2.0, 3.0, 4.0])
    y_arr = np.zeros_like(x_arr)

    z_bessel_arr = translator_bessel.ripple_function(x_arr, y_arr)
    z_simple_arr = translator_simple.ripple_function(x_arr, y_arr)

    # All should be finite
    assert np.isfinite(z_bessel_arr).all()
    assert np.isfinite(z_simple_arr).all()


def test_bessel_ripple_damping():
    """Test that damping reduces amplitude at large distances."""
    translator_no_damping = BesselRippleTranslator(k=5.0, damping=0.0, grid_size=30)
    translator_with_damping = BesselRippleTranslator(k=5.0, damping=0.2, grid_size=30)

    data_no_damping = translator_no_damping.generate()
    data_with_damping = translator_with_damping.generate()

    # At large r, damped version should have smaller amplitude
    large_r_mask = data_no_damping["r"] > 5.0

    amplitude_no_damping = np.abs(data_no_damping.loc[large_r_mask, "z"]).mean()
    amplitude_with_damping = np.abs(data_with_damping.loc[large_r_mask, "z"]).mean()

    assert amplitude_with_damping < amplitude_no_damping


def test_bessel_ripple_noise():
    """Test that noise is added correctly."""
    translator = BesselRippleTranslator(grid_size=50, noise_std=0.1, seed=42)
    data = translator.generate()

    # z and z_noisy should be different
    assert not np.allclose(data["z"], data["z_noisy"])

    # Noise magnitude should be reasonable
    noise = data["z_noisy"] - data["z"]
    assert np.std(noise) > 0.05  # Should be around noise_std=0.1
    assert np.std(noise) < 0.15


def test_bessel_ripple_no_noise():
    """Test that z and z_noisy are identical when noise_std=0."""
    translator = BesselRippleTranslator(grid_size=30, noise_std=0.0, seed=42)
    data = translator.generate()

    # Should be identical
    assert np.allclose(data["z"], data["z_noisy"])


def test_bessel_ripple_generate_grid():
    """Test meshgrid generation for 3D plotting."""
    translator = BesselRippleTranslator(grid_size=25, seed=42)
    X, Y, Z = translator.generate_grid()

    # Check shapes
    assert X.shape == (25, 25)
    assert Y.shape == (25, 25)
    assert Z.shape == (25, 25)

    # Check all values are finite
    assert np.isfinite(X).all()
    assert np.isfinite(Y).all()
    assert np.isfinite(Z).all()


def test_bessel_ripple_reproducibility():
    """Test that same seed produces same results."""
    translator1 = BesselRippleTranslator(grid_size=30, noise_std=0.1, seed=123)
    data1 = translator1.generate()

    translator2 = BesselRippleTranslator(grid_size=30, noise_std=0.1, seed=123)
    data2 = translator2.generate()

    # Should be identical
    assert np.allclose(data1["z"], data2["z"])
    assert np.allclose(data1["z_noisy"], data2["z_noisy"])


def test_bessel_ripple_amplitude_scaling():
    """Test that amplitude parameter scales the output correctly."""
    translator1 = BesselRippleTranslator(amplitude=1.0, grid_size=20)
    data1 = translator1.generate()

    translator2 = BesselRippleTranslator(amplitude=4.0, grid_size=20)
    data2 = translator2.generate()

    # z values should be scaled by amplitude
    assert np.allclose(data2["z"], 4.0 * data1["z"])


def test_bessel_ripple_oscillations():
    """Test that ripple oscillates as expected."""
    translator = BesselRippleTranslator(k=2 * np.pi, grid_size=50, use_bessel=False)
    data = translator.generate()

    # Should have oscillations (both positive and negative values)
    assert data["z"].max() > 0.1
    assert data["z"].min() < -0.1

    # Should oscillate with period related to k
    # Number of zero crossings should be roughly 2*k*max_r/π
