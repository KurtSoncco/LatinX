"""
Bessel-based ripple wave function dataset generator.

This generator creates radial wave patterns that resemble water droplet ripples.
It uses the spherical Bessel function j_0(kr) ≈ sin(kr)/(kr) to create smooth,
oscillating, decaying waves.

Mathematical forms:
    Simple: z(r) = A * sin(k*r) / r
    Bessel: z(r) = A * j_0(k*r) where j_0 is spherical Bessel function of first kind
"""

import numpy as np
import pandas as pd
from scipy.special import spherical_jn


class BesselRippleTranslator:
    """
    Generation of Bessel-based ripple wave function data.

    This generator creates 2D radial wave patterns mimicking water droplet ripples,
    using either the simple sin(kr)/r approximation or the spherical Bessel function.

    Args:
        k: Wave number controlling oscillation frequency (default: 5.0).
        amplitude: Amplitude multiplier for the output (default: 1.0).
        damping: Exponential damping coefficient (0 = no damping) (default: 0.0).
        x_range: Tuple of (min, max) for x coordinates (default: (-10, 10)).
        y_range: Tuple of (min, max) for y coordinates (default: (-10, 10)).
        grid_size: Number of points along each axis (default: 100).
        noise_std: Standard deviation of Gaussian noise to add (default: 0.0).
        use_bessel: If True, use spherical Bessel j_0; if False, use sin(kr)/r (default: True).
        epsilon: Small value to avoid division by zero at r=0 (default: 1e-6).
        seed: Random seed for reproducibility (default: None).

    Returns:
        pd.DataFrame: DataFrame containing 'x', 'y', 'z', 'z_noisy', and 'r' columns.
    """

    def __init__(
        self,
        k: float = 5.0,
        amplitude: float = 1.0,
        damping: float = 0.0,
        x_range: tuple[float, float] = (-10.0, 10.0),
        y_range: tuple[float, float] = (-10.0, 10.0),
        grid_size: int = 100,
        noise_std: float = 0.0,
        use_bessel: bool = True,
        epsilon: float = 1e-6,
        seed: int | None = None,
    ):
        if k <= 0:
            raise ValueError(f"k (wave number) must be positive, got {k}")
        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")
        if noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}")
        if damping < 0:
            raise ValueError(f"damping must be non-negative, got {damping}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.k = k
        self.amplitude = amplitude
        self.damping = damping
        self.x_range = x_range
        self.y_range = y_range
        self.grid_size = grid_size
        self.noise_std = noise_std
        self.use_bessel = use_bessel
        self.epsilon = epsilon
        self.seed = seed

    def ripple_function_simple(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute ripple using sin(kr)/r approximation.

        Args:
            x: X coordinates (can be array or meshgrid)
            y: Y coordinates (can be array or meshgrid)

        Returns:
            Ripple function values
        """
        r = np.sqrt(x**2 + y**2) + self.epsilon
        z = np.sin(self.k * r) / r

        # Apply damping if specified
        if self.damping > 0:
            z = z * np.exp(-self.damping * r)

        return self.amplitude * z

    def ripple_function_bessel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute ripple using spherical Bessel function j_0.

        Args:
            x: X coordinates (can be array or meshgrid)
            y: Y coordinates (can be array or meshgrid)

        Returns:
            Ripple function values
        """
        r = np.sqrt(x**2 + y**2)

        # Handle r=0 case (Bessel j_0(0) = 1)
        with np.errstate(invalid='ignore'):
            z = np.where(r < self.epsilon, 1.0, spherical_jn(0, self.k * r))

        # Apply damping if specified
        if self.damping > 0:
            z = z * np.exp(-self.damping * r)

        return self.amplitude * z

    def ripple_function(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute ripple function (dispatches to Bessel or simple version).

        Args:
            x: X coordinates (can be array or meshgrid)
            y: Y coordinates (can be array or meshgrid)

        Returns:
            Ripple function values
        """
        if self.use_bessel:
            return self.ripple_function_bessel(x, y)
        else:
            return self.ripple_function_simple(x, y)

    @property
    def function(self):
        """Return a callable ripple function."""
        return lambda x, y: self.ripple_function(x, y)

    def generate(self) -> pd.DataFrame:
        """
        Generate Bessel ripple wave data on a 2D grid.

        Returns:
            DataFrame with columns:
                - x: X coordinate
                - y: Y coordinate
                - z: Clean ripple function value
                - z_noisy: Function value with added Gaussian noise
                - r: Radial distance from origin (sqrt(x² + y²))
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        # Create 2D grid
        x = np.linspace(self.x_range[0], self.x_range[1], self.grid_size)
        y = np.linspace(self.y_range[0], self.y_range[1], self.grid_size)
        X, Y = np.meshgrid(x, y)

        # Flatten for DataFrame
        x_flat = X.flatten()
        y_flat = Y.flatten()

        # Compute ripple values
        z_clean = self.ripple_function(x_flat, y_flat)

        # Add noise
        noise = np.random.normal(0, self.noise_std, size=z_clean.shape)
        z_noisy = z_clean + noise

        # Compute radial distance
        r = np.sqrt(x_flat**2 + y_flat**2)

        # Create DataFrame
        data = pd.DataFrame({
            "x": x_flat,
            "y": y_flat,
            "z": z_clean,
            "z_noisy": z_noisy,
            "r": r,
        })

        return data

    def generate_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate Bessel ripple data in meshgrid format (useful for 3D plotting).

        Returns:
            Tuple of (X, Y, Z) where X and Y are coordinate meshgrids
            and Z is the ripple function values.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        x = np.linspace(self.x_range[0], self.x_range[1], self.grid_size)
        y = np.linspace(self.y_range[0], self.y_range[1], self.grid_size)
        X, Y = np.meshgrid(x, y)

        Z = self.ripple_function(X, Y)

        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=Z.shape)
            Z = Z + noise

        return X, Y, Z
