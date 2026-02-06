"""
Bayesian FFT Adapter for time series forecasting.

This module implements a Bayesian version of the ELF-Forecaster that combines
FFT-based feature extraction with Bayesian linear regression using Kalman/RLS
online updates. Instead of a point-estimate W matrix, it maintains a full
posterior over weights, enabling principled uncertainty quantification.

Key ideas:
1. FFT preprocessing: same as ELF (context FFT → crop → predict target spectrum)
2. Complex-to-Real conversion: stack real/imaginary parts to use standard BLR
3. Online updates: Kalman filter equations for sequential posterior refinement
4. Uncertainty: sample from posterior + observation noise, propagate through IRFFT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


def _complex_dtype_for(real_dtype: np.dtype) -> np.dtype:
    if np.dtype(real_dtype) == np.float32:
        return np.complex64
    return np.complex128


def _context_keep_indices(L: int, alpha: float) -> np.ndarray:
    """
    Cropping rule: keep k/L <= alpha/2 OR k/L >= 1 - alpha/2
    Returns indices into FFT bins of length L.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")

    lo = alpha / 2.0
    hi = 1.0 - alpha / 2.0

    ks = np.arange(L)
    frac = ks / float(L)
    keep = (frac <= lo) | (frac >= hi)
    return ks[keep].astype(np.int64)


def _target_keep_size(H: int, alpha: float) -> int:
    Hr = H // 2 + 1
    d = int(np.ceil(alpha * Hr))
    return max(1, min(Hr, d))


@dataclass
class BayesianFFTAdapterConfig:
    """Configuration for the Bayesian FFT Adapter.

    Attributes:
        L: Context length (input window size)
        H: Horizon length (prediction window size)
        alpha_freq: Frequency cropping parameter (same as ELF's alpha)
        prior_precision: Prior precision on weights (higher = stronger regularization).
            Default 1.0 gives P_0 = I. Use smaller values (e.g., 0.01) for weaker prior.
        sigma: Observation noise standard deviation in spectrum space
        baseline: Baseline type for detrending ("last" or "mean")
        init_seasonal: If True, initialize posterior mean with seasonal-ish pattern
        jitter: Small constant for numerical stability
    """
    L: int
    H: int
    alpha_freq: float = 0.9
    prior_precision: float = 1.0
    sigma: float = 0.1
    baseline: str = "last"
    init_seasonal: bool = True
    jitter: float = 1e-6


class BayesianFFTAdapter:
    """
    Bayesian FFT-based adapter for time series forecasting.

    This class implements Bayesian linear regression in the frequency domain
    with online Kalman filter updates. The posterior over weights enables
    principled uncertainty quantification through Monte Carlo sampling.

    Approach: Shared covariance across output frequencies (Approach 3A).

    Mathematical formulation:
        Prior: Theta ~ N(0, (1/prior_precision) * I)
        Likelihood: Y_big | Theta, A_big ~ N(A_big @ Theta, sigma^2 * I)
        Posterior: Theta | data ~ N(M, P)

    The complex-to-real conversion stacks real and imaginary parts:
        - Design matrix per window: A = [[a, -b], [b, a]] for x = a + ib
        - Parameters: Theta = [U; V] for W = U + iV
        - This ensures A @ Theta reproduces complex multiplication x @ W
    """

    def __init__(self, cfg: BayesianFFTAdapterConfig, real_dtype: np.dtype = np.float32):
        self.cfg = cfg
        self.real_dtype = np.dtype(real_dtype)
        self.cdtype = _complex_dtype_for(self.real_dtype)

        L, H, alpha = cfg.L, cfg.H, cfg.alpha_freq

        # Frequency bin indices to keep (same as ELF)
        self.keep_idx_x = _context_keep_indices(L, alpha)
        self.d_x = int(self.keep_idx_x.shape[0])

        self.Hr = H // 2 + 1
        self.d_y = _target_keep_size(H, alpha)

        # Real-valued dimensions after complex-to-real conversion
        # For input: stack [real; imag] -> 2 * d_x
        # For output: stack [real; imag] -> 2 * d_y per window row
        self.dim_theta = 2 * self.d_x  # Parameter dimension

        # Initialize posterior: M = 0 (or seasonal), P = (1/prior_precision) * I
        self.M = np.zeros((self.dim_theta, self.d_y), dtype=self.real_dtype)
        self.P = (1.0 / cfg.prior_precision) * np.eye(self.dim_theta, dtype=self.real_dtype)

        if cfg.init_seasonal:
            self._init_weights_seasonal()

        # Observation variance
        self.sigma_sq = cfg.sigma ** 2

        self.num_windows_seen = 0
        self._update_count = 0  # Track updates for occasional checks

    def _init_weights_seasonal(self) -> None:
        """
        Initialize posterior mean with seasonal-ish pattern.

        Sets W ≈ I on overlapping input/output frequency bins.
        This gives a stable starting point: predict same frequencies as input.

        In real parameterization: U = I (on overlap), V = 0
        """
        k = min(self.d_x, self.d_y)
        # U is in M[:d_x, :], set diagonal to 1
        for i in range(k):
            self.M[i, i] = 1.0
        # V (in M[d_x:, :]) stays zero

    def _baseline_value(self, x: np.ndarray) -> np.ndarray:
        """Compute baseline value for detrending (same as ELF)."""
        x = np.asarray(x, dtype=self.real_dtype).reshape(-1)
        if self.cfg.baseline == "mean":
            return np.asarray(x.mean(), dtype=self.real_dtype)
        # default: last
        return np.asarray(x[-1], dtype=self.real_dtype)

    def _x_to_features(self, x: np.ndarray, base: np.ndarray) -> np.ndarray:
        """
        Transform context to cropped FFT features (same as ELF).

        Args:
            x: Context window of shape [L]
            base: Scalar baseline value

        Returns:
            Complex cropped FFT bins of shape [d_x]
        """
        x = np.asarray(x, dtype=self.real_dtype).reshape(-1)
        if x.shape[0] != self.cfg.L:
            raise ValueError(f"Expected context length {self.cfg.L}, got {x.shape[0]}.")

        x0 = x - base
        Xf = np.fft.fft(x0, n=self.cfg.L, norm="ortho").astype(self.cdtype)
        return Xf[self.keep_idx_x]  # [d_x]

    def _y_to_targets(self, y: np.ndarray, base: np.ndarray) -> np.ndarray:
        """
        Transform target to cropped RFFT bins (same as ELF).

        Args:
            y: Target window of shape [H]
            base: SAME baseline used for x in this window

        Returns:
            Complex cropped RFFT bins of shape [d_y]
        """
        y = np.asarray(y, dtype=self.real_dtype).reshape(-1)
        if y.shape[0] != self.cfg.H:
            raise ValueError(f"Expected target length {self.cfg.H}, got {y.shape[0]}.")

        y0 = y - base
        Yf = np.fft.rfft(y0, n=self.cfg.H, norm="ortho").astype(self.cdtype)
        return Yf[: self.d_y]  # [d_y]

    def _build_design_matrix(self, X_spec: np.ndarray) -> np.ndarray:
        """
        Build real-valued design matrix from complex features.

        For complex multiplication y = x @ W where x = a + ib and W = U + iV:
            Re(y) = a*U - b*V
            Im(y) = b*U + a*V

        This can be written as: [Re(y); Im(y)] = A @ [U; V]
        where A = [[a, -b], [b, a]] per window.

        Args:
            X_spec: Complex features of shape [M, d_x]

        Returns:
            Real design matrix of shape [2*M, 2*d_x]
        """
        M = X_spec.shape[0]
        a = X_spec.real.astype(self.real_dtype)  # [M, d_x]
        b = X_spec.imag.astype(self.real_dtype)  # [M, d_x]

        # Build block matrix: for each window row i, we have
        # Row 2i:   [a_i, -b_i]
        # Row 2i+1: [b_i,  a_i]
        A_big = np.zeros((2 * M, 2 * self.d_x), dtype=self.real_dtype)

        A_big[0::2, :self.d_x] = a       # Upper-left: a
        A_big[0::2, self.d_x:] = -b      # Upper-right: -b
        A_big[1::2, :self.d_x] = b       # Lower-left: b
        A_big[1::2, self.d_x:] = a       # Lower-right: a

        return A_big

    def _build_target_matrix(self, Y_spec: np.ndarray) -> np.ndarray:
        """
        Build real-valued target matrix from complex targets.

        Args:
            Y_spec: Complex targets of shape [M, d_y]

        Returns:
            Real target matrix of shape [2*M, d_y]
            Row 2i contains Re(Y[i]), row 2i+1 contains Im(Y[i])
        """
        M = Y_spec.shape[0]
        c = Y_spec.real.astype(self.real_dtype)  # [M, d_y]
        d = Y_spec.imag.astype(self.real_dtype)  # [M, d_y]

        Y_big = np.zeros((2 * M, self.d_y), dtype=self.real_dtype)
        Y_big[0::2, :] = c  # Real parts
        Y_big[1::2, :] = d  # Imaginary parts

        return Y_big

    def _safe_cholesky(self, P: np.ndarray) -> np.ndarray:
        """
        Compute Cholesky decomposition with eigenvalue clipping fallback.

        If P is not positive definite, clips negative eigenvalues and reconstructs.
        """
        try:
            return np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            # Eigenvalue clipping: ensure all eigenvalues are >= min_eig
            min_eig = 1e-6
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals_clipped = np.maximum(eigvals, min_eig)
            P_fixed = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
            P_fixed = 0.5 * (P_fixed + P_fixed.T)  # Ensure symmetry
            return np.linalg.cholesky(P_fixed)

    def _reconstruct_W(self, Theta: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reconstruct complex W matrix from real parameterization.

        Theta has shape [2*d_x, d_y] = [[U], [V]]
        W = U + iV has shape [d_x, d_y]

        Args:
            Theta: Real parameter matrix. If None, uses posterior mean M.
        """
        if Theta is None:
            Theta = self.M
        U = Theta[:self.d_x, :]  # [d_x, d_y]
        V = Theta[self.d_x:, :]  # [d_x, d_y]
        return (U + 1j * V).astype(self.cdtype)

    def update_with_batch(self, X_ctx: np.ndarray, Y_true: np.ndarray) -> None:
        """
        Update posterior using Kalman filter equations.

        Kalman update for multi-observation batch:
            S = A_big @ P @ A_big^T + sigma^2 * I    # Innovation covariance
            K = P @ A_big^T @ S^{-1}                  # Kalman gain
            M <- M + K @ (Y_big - A_big @ M)          # Mean update
            P <- P - K @ A_big @ P                    # Covariance update

        Args:
            X_ctx: Context windows of shape [M, L]
            Y_true: Target windows of shape [M, H]
        """
        X_ctx = np.asarray(X_ctx, dtype=self.real_dtype)
        Y_true = np.asarray(Y_true, dtype=self.real_dtype)

        if X_ctx.ndim != 2 or X_ctx.shape[1] != self.cfg.L:
            raise ValueError(f"X_ctx must be [M, {self.cfg.L}]. Got {X_ctx.shape}.")
        if Y_true.ndim != 2 or Y_true.shape[1] != self.cfg.H:
            raise ValueError(f"Y_true must be [M, {self.cfg.H}]. Got {Y_true.shape}.")

        M_windows = int(X_ctx.shape[0])
        if M_windows == 0:
            return

        # Transform to frequency domain
        Xf = np.empty((M_windows, self.d_x), dtype=self.cdtype)
        Yf = np.empty((M_windows, self.d_y), dtype=self.cdtype)

        for i in range(M_windows):
            base = self._baseline_value(X_ctx[i])
            Xf[i, :] = self._x_to_features(X_ctx[i], base=base)
            Yf[i, :] = self._y_to_targets(Y_true[i], base=base)

        # Build real-valued matrices
        A_big = self._build_design_matrix(Xf)  # [2*M, 2*d_x]
        Y_big = self._build_target_matrix(Yf)  # [2*M, d_y]

        n_obs = 2 * M_windows  # Number of real observations

        # Kalman update equations
        # S = A @ P @ A^T + sigma^2 * I
        AP = A_big @ self.P  # [2M, 2*d_x]
        S = AP @ A_big.T + self.sigma_sq * np.eye(n_obs, dtype=self.real_dtype)

        # Add jitter for numerical stability
        if self.cfg.jitter > 0:
            S += self.cfg.jitter * np.eye(n_obs, dtype=self.real_dtype)

        # K = P @ A^T @ S^{-1}
        # Solve S @ K^T = A @ P for efficiency
        K = np.linalg.solve(S, AP).T  # [2*d_x, 2M]

        # Innovation: Y_big - A_big @ M
        innovation = Y_big - A_big @ self.M  # [2M, d_y]

        # Mean update: M <- M + K @ innovation
        self.M = self.M + K @ innovation

        # Covariance update using Joseph form (more numerically stable):
        # P <- (I - K @ A) @ P @ (I - K @ A)^T + K @ R @ K^T
        # where R = sigma^2 * I
        I_KA = np.eye(self.dim_theta, dtype=self.real_dtype) - K @ A_big
        self.P = I_KA @ self.P @ I_KA.T + self.sigma_sq * (K @ K.T)

        # Symmetrize for numerical stability
        self.P = 0.5 * (self.P + self.P.T)

        # Add small diagonal jitter unconditionally
        self.P += self.cfg.jitter * np.eye(self.dim_theta, dtype=self.real_dtype)

        self.num_windows_seen += M_windows
        self._update_count += 1

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make point prediction using posterior mean.

        Args:
            x: Context window of shape [L]

        Returns:
            Forecast of shape [H]
        """
        x = np.asarray(x, dtype=self.real_dtype).reshape(-1)
        base = self._baseline_value(x)

        x_feat = self._x_to_features(x, base=base)  # [d_x] complex
        W = self._reconstruct_W()  # [d_x, d_y] complex

        y_spec_crop = x_feat @ W  # [d_y] complex

        # Clamp DC bin imaginary part to zero (DC should be real for RFFT)
        y_spec_crop[0] = y_spec_crop[0].real + 0j

        # Zero-pad and inverse transform
        y_spec_full = np.zeros((self.Hr,), dtype=self.cdtype)
        y_spec_full[: self.d_y] = y_spec_crop

        yhat0 = np.fft.irfft(y_spec_full, n=self.cfg.H, norm="ortho").astype(self.real_dtype)
        return yhat0 + base

    def sample_forecasts(
        self,
        x: np.ndarray,
        num_samples: int = 100,
        rng: Optional[np.random.Generator] = None,
        include_obs_noise: bool = True,
    ) -> np.ndarray:
        """
        Sample forecasts from the predictive distribution.

        Samples Theta ~ N(M, P), reconstructs W, adds observation noise in
        spectrum space, and propagates through IRFFT.

        The predictive distribution includes both:
        1. Parameter uncertainty (from posterior P)
        2. Observation noise (sigma in spectrum space)

        Args:
            x: Context window of shape [L]
            num_samples: Number of Monte Carlo samples
            rng: Random number generator (optional)
            include_obs_noise: If True, add observation noise to spectrum

        Returns:
            Forecast samples of shape [num_samples, H]
        """
        if rng is None:
            rng = np.random.default_rng()

        x = np.asarray(x, dtype=self.real_dtype).reshape(-1)
        base = self._baseline_value(x)
        x_feat = self._x_to_features(x, base=base)  # [d_x] complex

        # Cholesky of posterior covariance for sampling
        # Use eigenvalue clipping for robustness
        L_chol = self._safe_cholesky(self.P)

        forecasts = np.empty((num_samples, self.cfg.H), dtype=self.real_dtype)
        sigma = self.cfg.sigma

        for s in range(num_samples):
            # Sample: Theta = M + L @ z where z ~ N(0, I)
            z = rng.standard_normal((self.dim_theta, self.d_y)).astype(self.real_dtype)
            Theta_sample = self.M + L_chol @ z  # [2*d_x, d_y]

            # Reconstruct complex W from sampled Theta
            W_sample = self._reconstruct_W(Theta_sample)  # [d_x, d_y]

            # Forward pass: y_spec = x_feat @ W
            y_spec_crop = x_feat @ W_sample  # [d_y] complex

            # Add observation noise in spectrum space
            # For complex noise with variance sigma^2: real and imag each have variance sigma^2/2
            if include_obs_noise and sigma > 0:
                noise_real = rng.standard_normal(self.d_y).astype(self.real_dtype)
                noise_imag = rng.standard_normal(self.d_y).astype(self.real_dtype)
                # Complex noise: (sigma/sqrt(2)) * (n_r + i*n_i) has |noise|^2 ~ sigma^2
                y_spec_crop = y_spec_crop + (sigma / np.sqrt(2)) * (noise_real + 1j * noise_imag)

            # Clamp DC bin imaginary part to zero
            y_spec_crop[0] = y_spec_crop[0].real + 0j

            # Zero-pad and inverse transform
            y_spec_full = np.zeros((self.Hr,), dtype=self.cdtype)
            y_spec_full[: self.d_y] = y_spec_crop

            yhat0 = np.fft.irfft(y_spec_full, n=self.cfg.H, norm="ortho").astype(self.real_dtype)
            forecasts[s, :] = yhat0 + base

        return forecasts

    def get_quantiles(
        self,
        x: np.ndarray,
        quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
        num_samples: int = 100,
        rng: Optional[np.random.Generator] = None,
        include_obs_noise: bool = True,
    ) -> dict:
        """
        Compute forecast quantiles via Monte Carlo sampling.

        Args:
            x: Context window of shape [L]
            quantiles: Tuple of quantile levels (e.g., (0.1, 0.5, 0.9))
            num_samples: Number of Monte Carlo samples
            rng: Random number generator (optional)
            include_obs_noise: If True, include observation noise in samples

        Returns:
            Dictionary with keys like 'q10', 'q50', 'q90' containing
            numpy arrays of shape [H]
        """
        samples = self.sample_forecasts(
            x, num_samples=num_samples, rng=rng, include_obs_noise=include_obs_noise
        )

        result = {}
        for q in quantiles:
            key = f"q{int(q * 100)}"
            result[key] = np.quantile(samples, q, axis=0)

        return result

    def get_posterior_stats(self) -> dict:
        """
        Get summary statistics of the posterior distribution.

        Returns:
            Dictionary with posterior statistics
        """
        W = self._reconstruct_W()

        # Only compute expensive eigvalsh occasionally or on demand
        try:
            min_eig = float(np.linalg.eigvalsh(self.P).min())
        except np.linalg.LinAlgError:
            min_eig = float('nan')

        return {
            "posterior_mean_norm": float(np.linalg.norm(self.M)),
            "posterior_cov_trace": float(np.trace(self.P)),
            "posterior_cov_min_eig": min_eig,
            "W_mean_abs": float(np.abs(W).mean()),
            "num_windows_seen": self.num_windows_seen,
        }
