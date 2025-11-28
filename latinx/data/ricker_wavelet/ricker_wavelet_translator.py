"""
Ricker Wavelet (Laplacian of Gaussian) function dataset generator.

The Ricker Wavelet function is a radially symmetric wavelet that resembles
a sombrero. It's the Laplacian of a 2D Gaussian and is useful for feature
detection and signal processing applications.

Mathematical form:
    z(r) = (1 - r²/σ²) * exp(-r²/(2σ²))
    where r = sqrt(x² + y²)
"""

import numpy as np
import pandas as pd


class RickerWaveletTranslator:
    """
    Generation of Ricker Wavelet (Laplacian of Gaussian) function data.

    This generator creates a 2D radially symmetric function with a central peak
    and surrounding trough, useful for testing ML models on smooth analytical functions.

    Args:
        sigma: Scale parameter controlling the width of the wavelet (default: 1.0).
        amplitude: Amplitude multiplier for the output (default: 1.0).
        x_range: Tuple of (min, max) for x coordinates (default: (-5, 5)).
        y_range: Tuple of (min, max) for y coordinates (default: (-5, 5)).
        grid_size: Number of points along each axis (default: 100).
        noise_std: Standard deviation of Gaussian noise to add (default: 0.0).
        seed: Random seed for reproducibility (default: None).

    Returns:
        pd.DataFrame: DataFrame containing 'x', 'y', 'z', 'z_noisy', and 'r' columns.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        amplitude: float = 1.0,
        x_range: tuple[float, float] = (-5.0, 5.0),
        y_range: tuple[float, float] = (-5.0, 5.0),
        grid_size: int = 100,
        noise_std: float = 0.0,
        seed: int | None = None,
    ):
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")
        if noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}")

        self.sigma = sigma
        self.amplitude = amplitude
        self.x_range = x_range
        self.y_range = y_range
        self.grid_size = grid_size
        self.noise_std = noise_std
        self.seed = seed

    def ricker_wavelet_function(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the Ricker Wavelet function at given coordinates.

        Args:
            x: X coordinates (can be array or meshgrid)
            y: Y coordinates (can be array or meshgrid)

        Returns:
            Ricker Wavelet function values
        """
        r_squared = x**2 + y**2
        sigma_squared = self.sigma**2

        z = (1 - r_squared / sigma_squared) * np.exp(-r_squared / (2 * sigma_squared))
        return self.amplitude * z

    @property
    def function(self):
        """Return a callable Ricker Wavelet function."""
        return lambda x, y: self.ricker_wavelet_function(x, y)

    def generate(self) -> pd.DataFrame:
        """
        Generate Ricker Wavelet function data on a 2D grid.

        Returns:
            DataFrame with columns:
                - x: X coordinate
                - y: Y coordinate
                - z: Clean Ricker Wavelet function value
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

        # Compute Ricker Wavelet values
        z_clean = self.ricker_wavelet_function(x_flat, y_flat)

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
        Generate Ricker Wavelet data in meshgrid format (useful for 3D plotting).

        Returns:
            Tuple of (X, Y, Z) where X and Y are coordinate meshgrids
            and Z is the Ricker Wavelet function values.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        x = np.linspace(self.x_range[0], self.x_range[1], self.grid_size)
        y = np.linspace(self.y_range[0], self.y_range[1], self.grid_size)
        X, Y = np.meshgrid(x, y)

        Z = self.ricker_wavelet_function(X, Y)

        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=Z.shape)
            Z = Z + noise

        return X, Y, Z
