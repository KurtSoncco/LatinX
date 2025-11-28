"""
Rössler system trajectory dataset generator.

The Rössler system is a 3D system of ordinary differential equations that exhibits
chaotic behavior. It is characterized by oscillatory dynamics and strange attractors.

Mathematical form:
    dx/dt = -y - z
    dy/dt = x + a*y
    dz/dt = b + z*(x - c)

where (a, b, c) are parameters that control the system's behavior.
Standard chaotic regime: a=0.2, b=0.2, c=5.7
"""

import numpy as np
import pandas as pd


class RosslerTranslator:
    """
    Generation of Rössler system trajectory data.

    This generator simulates the Rössler chaotic system using Euler integration
    and produces time-series data with optional noise injection.

    Args:
        n_steps: Number of integration steps (default: 1000).
        dt: Time step size for Euler integration (default: 0.01).
        a: Parameter 'a' in Rössler equations (default: 0.2).
        b: Parameter 'b' in Rössler equations (default: 0.2).
        c: Parameter 'c' in Rössler equations (default: 5.7).
        initial_state: Initial conditions [x0, y0, z0] (default: [1.0, 0.0, 0.0]).
        noise_std: Standard deviation of Gaussian noise to add (default: 0.0).
                   If noise_pct is provided, this will be overridden.
        noise_pct: Noise as percentage of local state magnitude (0-1 scale). If provided,
                   noise at each point will be: noise_pct * |state(t)|, making noise
                   proportional to the local signal strength (default: None).
        seed: Random seed for reproducibility (default: None).

    Returns:
        pd.DataFrame: DataFrame containing 't', 'x', 'y', 'z', 'x_noisy', 'y_noisy', 'z_noisy'.
    """

    def __init__(
        self,
        n_steps: int = 1000,
        dt: float = 0.01,
        a: float = 0.2,
        b: float = 0.2,
        c: float = 5.7,
        initial_state: list[float] | np.ndarray | None = None,
        noise_std: float = 0.0,
        noise_pct: float | None = None,
        seed: int | None = None,
    ):
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        self.n_steps = n_steps
        self.dt = dt
        self.a = a
        self.b = b
        self.c = c

        # Set initial state
        if initial_state is None:
            self.initial_state = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            self.initial_state = np.asarray(initial_state, dtype=np.float64)
            if self.initial_state.shape != (3,):
                raise ValueError(f"initial_state must have shape (3,), got {self.initial_state.shape}")

        # Handle noise: use percentage if provided, otherwise use fixed std
        if noise_pct is not None:
            if not 0 <= noise_pct <= 1:
                raise ValueError(f"noise_pct must be in [0, 1], got {noise_pct}")
            self.noise_pct = noise_pct
            self.noise_std = 0.0  # Will be computed per-point based on local magnitude
        else:
            if noise_std < 0:
                raise ValueError(f"noise_std must be non-negative, got {noise_std}")
            self.noise_pct = None
            self.noise_std = noise_std

        self.seed = seed

    def f(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Compute derivatives for the Rössler system.

        Args:
            state: Current state [x, y, z]
            t: Current time (not used, but kept for compatibility)

        Returns:
            Derivatives [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state
        dx = -y - z
        dy = x + self.a * y
        dz = self.b + z * (x - self.c)
        return np.array([dx, dy, dz], dtype=np.float64)

    def generate(self) -> pd.DataFrame:
        """
        Generate Rössler system trajectory data.

        Returns:
            DataFrame with columns:
                - t: Time
                - x: X coordinate (clean)
                - y: Y coordinate (clean)
                - z: Z coordinate (clean)
                - x_noisy: X with added Gaussian noise
                - y_noisy: Y with added Gaussian noise
                - z_noisy: Z with added Gaussian noise
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        # Initialize arrays
        trajectory = np.zeros((self.n_steps, 3), dtype=np.float64)
        time = np.zeros(self.n_steps, dtype=np.float64)

        # Initial conditions
        state = self.initial_state.copy()
        t = 0.0

        # Euler integration
        for n in range(self.n_steps):
            trajectory[n] = state.copy()
            time[n] = t

            # Euler step: state_{n+1} = state_n + dt * f(state_n, t_n)
            state = state + self.dt * self.f(state, t)
            t = t + self.dt

        # Add noise (proportional to local magnitude if noise_pct is set)
        if self.noise_pct is not None:
            # Noise scales with local state magnitude (heteroscedastic)
            # Compute magnitude for each point
            magnitude = np.linalg.norm(trajectory, axis=1, keepdims=True)
            noise_std_local = self.noise_pct * magnitude
            noise = np.random.normal(0, 1, size=trajectory.shape) * noise_std_local
        else:
            # Fixed noise across all points (homoscedastic)
            noise = np.random.normal(0, self.noise_std, size=trajectory.shape)

        trajectory_noisy = trajectory + noise

        # Create DataFrame
        data = pd.DataFrame({
            "t": time,
            "x": trajectory[:, 0],
            "y": trajectory[:, 1],
            "z": trajectory[:, 2],
            "x_noisy": trajectory_noisy[:, 0],
            "y_noisy": trajectory_noisy[:, 1],
            "z_noisy": trajectory_noisy[:, 2],
        })

        return data

    def generate_with_transient_removal(
        self, n_transient: int = 1000
    ) -> pd.DataFrame:
        """
        Generate trajectory with initial transient removed.

        This is useful for chaotic systems where you want to ensure the trajectory
        is on the attractor before collecting data.

        Args:
            n_transient: Number of initial steps to discard (default: 1000).

        Returns:
            DataFrame with transient removed.
        """
        # Run transient
        state = self.initial_state.copy()
        t = 0.0
        for _ in range(n_transient):
            state = state + self.dt * self.f(state, t)
            t = t + self.dt

        # Now generate data from this settled state
        original_initial_state = self.initial_state.copy()
        self.initial_state = state
        data = self.generate()
        self.initial_state = original_initial_state  # Restore

        return data

    def __repr__(self) -> str:
        """String representation of the translator."""
        return (
            f"RosslerTranslator("
            f"n_steps={self.n_steps}, "
            f"dt={self.dt:.4f}, "
            f"a={self.a:.4f}, "
            f"b={self.b:.4f}, "
            f"c={self.c:.4f}, "
            f"noise_std={self.noise_std:.4f}, "
            f"noise_pct={self.noise_pct})"
        )
