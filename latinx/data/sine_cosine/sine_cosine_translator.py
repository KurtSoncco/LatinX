import numpy as np
import pandas as pd


class SineCosineTranslator:
    """
    Generation of sine and cosine function data based on specified parameters.
    The output present some noise.

    Args:
        amplitude (float): Amplitude of the sine and cosine functions.
        angle_multiplier (float): Multiplier for the angle in the sine and cosine functions.
        seed (int, optional): Random seed for reproducibility.
        num_samples (int): Number of samples to generate.
        t_start (float): Starting time value for continuous time axis (default 0.0).
        noise_mean (int): Mean of the noise added to cosine values.
        noise_std (float): Standard deviation of the noise added to cosine values.

    Returns:
        pd.DataFrame: DataFrame containing 't', 'sine', and 'cosine' columns.
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        angle_multiplier: float = 1.0,
        seed: int | None = None,
        num_samples: int = 1000,
        t_start: float = 0.0,
        noise_mean: int = 0,
        noise_std: float = 0.1,
    ):
        self.amplitude = amplitude
        self.angle_multiplier = angle_multiplier
        self.seed = seed
        self.num_samples = num_samples
        self.t_start = t_start
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    @property
    def sine_function(self):
        return lambda t: self.amplitude * np.sin(self.angle_multiplier * t)

    @property
    def cosine_function(self):
        return lambda t: self.amplitude * np.cos(self.angle_multiplier * t)

    def generate(self, dt: float = 0.05) -> pd.DataFrame:
        """
        Generate sine and cosine data with specified time step.

        Args:
            dt: Time step between samples (default 0.05). Controls the temporal
                resolution of the generated data.

        Returns:
            pd.DataFrame with columns 't', 'sine', 'cosine', 'x1', 'y1'
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        # Calculate time values based on start, num_samples, and dt
        t_end = self.t_start + self.num_samples * dt
        t_values = np.arange(self.t_start, t_end, dt)[:self.num_samples]
        sine_values = self.sine_function(t_values)
        cosine_values = self.cosine_function(t_values) + np.random.normal(
            self.noise_mean, self.noise_std, self.num_samples
        )

        data = pd.DataFrame(
            {
                "t": t_values,
                "sine": sine_values,
                "cosine": cosine_values,
                "x1": sine_values,
                "y1": cosine_values,
            }
        )

        return data
