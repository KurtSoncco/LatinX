"""Tests for Rössler system dataset generator."""

import numpy as np
import pytest

from latinx.data.rossler import RosslerTranslator


def test_initialization():
    """Test basic initialization."""
    rossler = RosslerTranslator(
        n_steps=1000,
        dt=0.01,
        a=0.2,
        b=0.2,
        c=5.7,
        noise_std=0.1,
    )

    assert rossler.n_steps == 1000
    assert rossler.dt == 0.01
    assert rossler.a == 0.2
    assert rossler.b == 0.2
    assert rossler.c == 5.7
    assert rossler.noise_std == 0.1
    assert rossler.noise_pct is None


def test_initialization_with_noise_pct():
    """Test initialization with percentage-based noise."""
    rossler = RosslerTranslator(n_steps=100, dt=0.01, noise_pct=0.05)

    assert rossler.noise_pct == 0.05
    assert rossler.noise_std == 0.0


def test_invalid_parameters():
    """Test that invalid parameters raise errors."""
    with pytest.raises(ValueError, match="n_steps must be positive"):
        RosslerTranslator(n_steps=0)

    with pytest.raises(ValueError, match="dt must be positive"):
        RosslerTranslator(dt=-0.01)

    with pytest.raises(ValueError, match="noise_std must be non-negative"):
        RosslerTranslator(noise_std=-0.1)

    with pytest.raises(ValueError, match="noise_pct must be in"):
        RosslerTranslator(noise_pct=1.5)


def test_initial_state_shape():
    """Test that invalid initial state raises error."""
    with pytest.raises(ValueError, match="initial_state must have shape"):
        RosslerTranslator(initial_state=[1.0, 2.0])


def test_generate_output_shape():
    """Test that generate() returns correct shape."""
    rossler = RosslerTranslator(n_steps=100, dt=0.01, seed=42)
    df = rossler.generate()

    assert len(df) == 100
    assert list(df.columns) == ["t", "x", "y", "z", "x_noisy", "y_noisy", "z_noisy"]


def test_generate_deterministic():
    """Test that using same seed produces same results."""
    rossler1 = RosslerTranslator(n_steps=100, dt=0.01, noise_std=0.1, seed=42)
    rossler2 = RosslerTranslator(n_steps=100, dt=0.01, noise_std=0.1, seed=42)

    df1 = rossler1.generate()
    df2 = rossler2.generate()

    np.testing.assert_array_almost_equal(df1["x"].values, df2["x"].values)
    np.testing.assert_array_almost_equal(df1["y"].values, df2["y"].values)
    np.testing.assert_array_almost_equal(df1["z"].values, df2["z"].values)
    np.testing.assert_array_almost_equal(df1["x_noisy"].values, df2["x_noisy"].values)


def test_time_array():
    """Test that time array is correct."""
    n_steps = 100
    dt = 0.01
    rossler = RosslerTranslator(n_steps=n_steps, dt=dt)
    df = rossler.generate()

    # Check time starts at 0
    assert df["t"].iloc[0] == 0.0

    # Check time increments correctly
    expected_times = np.arange(n_steps) * dt
    np.testing.assert_array_almost_equal(df["t"].values, expected_times)


def test_euler_integration():
    """Test that Euler integration produces expected behavior."""
    # Use simple parameters and check trajectory evolves
    rossler = RosslerTranslator(
        n_steps=10,
        dt=0.01,
        initial_state=[1.0, 0.0, 0.0],
        noise_std=0.0,
    )
    df = rossler.generate()

    # First point should be initial state
    assert df["x"].iloc[0] == 1.0
    assert df["y"].iloc[0] == 0.0
    assert df["z"].iloc[0] == 0.0

    # State should evolve (not stay constant)
    assert not np.allclose(df["x"].iloc[-1], df["x"].iloc[0])


def test_noise_injection():
    """Test that noise is added correctly."""
    rossler = RosslerTranslator(n_steps=100, dt=0.01, noise_std=0.1, seed=42)
    df = rossler.generate()

    # Clean and noisy should be different
    assert not np.allclose(df["x"].values, df["x_noisy"].values)
    assert not np.allclose(df["y"].values, df["y_noisy"].values)
    assert not np.allclose(df["z"].values, df["z_noisy"].values)

    # Mean difference should be small (noise has zero mean)
    assert np.abs(np.mean(df["x_noisy"] - df["x"])) < 0.1


def test_noise_pct():
    """Test percentage-based noise."""
    rossler = RosslerTranslator(n_steps=100, dt=0.01, noise_pct=0.1, seed=42)
    df = rossler.generate()

    # Noise should scale with state magnitude
    # Check that noise is not uniform across the trajectory
    noise_x = df["x_noisy"] - df["x"]
    # Standard deviation of noise should vary along trajectory
    # (since state magnitude varies)
    assert np.std(noise_x[:50]) > 0  # Has some variation


def test_generate_with_transient_removal():
    """Test transient removal."""
    rossler = RosslerTranslator(n_steps=100, dt=0.01, seed=42)

    # Generate with transient removal
    df_with_removal = rossler.generate_with_transient_removal(n_transient=50)

    # Should still have n_steps points
    assert len(df_with_removal) == 100

    # Initial state should be different (transient was run)
    df_no_removal = rossler.generate()
    assert not np.isclose(df_with_removal["x"].iloc[0], df_no_removal["x"].iloc[0])


def test_repr():
    """Test string representation."""
    rossler = RosslerTranslator(n_steps=100, dt=0.01, a=0.2, b=0.2, c=5.7)
    repr_str = repr(rossler)

    assert "RosslerTranslator" in repr_str
    assert "n_steps=100" in repr_str
    assert "a=0.2000" in repr_str
