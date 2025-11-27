import numpy as np
import pandas as pd


class SineCosineTranslator:
    """
    Generation of sine and cosine function data based on specified parameters.

    Args:
        amplitude (float): Amplitude of the sine and cosine functions.
        angle_multiplier (float): Multiplier for the angle in the sine and cosine functions.
        seed (int, optional): Random seed for reproducibility.
        num_samples (int): Number of samples to generate.

    Returns:
        pd.DataFrame: DataFrame containing 't', 'sine', and 'cosine' columns.
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        angle_multiplier: float = 1.0,
        seed: int | None = None,
        num_samples: int = 1000,
    ):
        self.amplitude = amplitude
        self.angle_multiplier = angle_multiplier
        self.seed = seed
        self.num_samples = num_samples

    @property
    def sine_function(self):
        return lambda t: self.amplitude * np.sin(self.angle_multiplier * t)

    @property
    def cosine_function(self):
        return lambda t: self.amplitude * np.cos(self.angle_multiplier * t)

    def generate(self) -> pd.DataFrame:
        if self.seed is not None:
            np.random.seed(self.seed)

        t_values = np.linspace(0, 2 * np.pi, self.num_samples)
        sine_values = self.sine_function(t_values)
        cosine_values = self.cosine_function(t_values)

        data = pd.DataFrame({"t": t_values, "sine": sine_values, "cosine": cosine_values})

        return data
