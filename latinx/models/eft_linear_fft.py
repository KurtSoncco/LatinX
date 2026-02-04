"""
ELF-Forecaster (lightweight model) from:
"Lightweight Online Adaption for Time Series Foundation Model Forecasts"

This implements ONLY the ELF-Forecaster (no ELF-Weighter yet).

Core ideas implemented:
1) Forecasting in Fourier domain:
   - x (context length L) -> DFT(x) -> crop frequencies -> multiply by complex W
   - output is a cropped RFFT spectrum for horizon H, zero-pad, then irfft -> yhat

2) Online fitting in closed form (ridge-regularized OLS) with Woodbury updates:
   - W = (X*X + lam I)^-1 (X*Y)
   - Maintain:
       A_inv = (X*X + lam I)^-1   (complex, d_x by d_x)
       XTY   = X*Y               (complex, d_x by d_y)
   - Update every batch of M new windows using Woodbury:
       A_inv <- A_inv - A_inv X* (I + X A_inv X*)^-1 X A_inv
       XTY   <- XTY + X* Y
       W     <- A_inv @ XTY

Stability patches:
- Orthonormal FFT scaling: norm="ortho" for fft/rfft/irfft
- Use a consistent per-window baseline for BOTH x and y (prevents offset drift)
- Add small jitter to Woodbury "middle" matrix before solve

You can feed this from your PyTorch windows by doing .cpu().numpy() on batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

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
class ELFForecasterConfig:
    L: int
    H: int
    alpha: float = 0.9
    lam: float = 1e-2
    init_seasonal: bool = True
    jitter: float = 1e-6
    baseline: str = "last"   # "last" or "mean"


class ELFForecaster:
    def __init__(self, cfg: ELFForecasterConfig, real_dtype: np.dtype = np.float32):
        self.cfg = cfg
        self.real_dtype = np.dtype(real_dtype)
        self.cdtype = _complex_dtype_for(self.real_dtype)

        L, H, alpha = cfg.L, cfg.H, cfg.alpha

        self.keep_idx_x = _context_keep_indices(L, alpha)
        self.d_x = int(self.keep_idx_x.shape[0])

        self.Hr = H // 2 + 1
        self.d_y = _target_keep_size(H, alpha)

        if cfg.lam <= 0:
            raise ValueError("lam must be > 0")

        # A_inv = (X*X + lam I)^-1
        self.A_inv = (1.0 / cfg.lam) * np.eye(self.d_x, dtype=self.cdtype)
        self.XTY = np.zeros((self.d_x, self.d_y), dtype=self.cdtype)
        self.W = np.zeros((self.d_x, self.d_y), dtype=self.cdtype)

        if cfg.init_seasonal:
            self._init_weights_seasonalish()

        self.num_windows_seen = 0

    def _init_weights_seasonalish(self) -> None:
        k = min(self.d_x, self.d_y)
        self.W[:k, :k] = np.eye(k, dtype=self.cdtype)

    def _baseline_value(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=self.real_dtype).reshape(-1)
        if self.cfg.baseline == "mean":
            return np.asarray(x.mean(), dtype=self.real_dtype)
        # default: last
        return np.asarray(x[-1], dtype=self.real_dtype)

    def _x_to_features(self, x: np.ndarray, base: np.ndarray) -> np.ndarray:
        """
        x: [L] real
        base: scalar baseline used for this window
        returns: [d_x] complex cropped FFT bins (ortho)
        """
        x = np.asarray(x, dtype=self.real_dtype).reshape(-1)
        if x.shape[0] != self.cfg.L:
            raise ValueError(f"Expected context length {self.cfg.L}, got {x.shape[0]}.")

        x0 = x - base
        Xf = np.fft.fft(x0, n=self.cfg.L, norm="ortho").astype(self.cdtype)  # [L]
        return Xf[self.keep_idx_x]  # [d_x]

    def _y_to_targets(self, y: np.ndarray, base: np.ndarray) -> np.ndarray:
        """
        y: [H] real
        base: SAME baseline used for x in this window
        returns: [d_y] complex cropped RFFT bins (ortho)
        """
        y = np.asarray(y, dtype=self.real_dtype).reshape(-1)
        if y.shape[0] != self.cfg.H:
            raise ValueError(f"Expected target length {self.cfg.H}, got {y.shape[0]}.")

        y0 = y - base
        Yf = np.fft.rfft(y0, n=self.cfg.H, norm="ortho").astype(self.cdtype)  # [Hr]
        return Yf[: self.d_y]  # [d_y]

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=self.real_dtype).reshape(-1)
        base = self._baseline_value(x)

        x_feat = self._x_to_features(x, base=base)   # [d_x]
        y_spec_crop = x_feat @ self.W                # [d_y]

        y_spec_full = np.zeros((self.Hr,), dtype=self.cdtype)
        y_spec_full[: self.d_y] = y_spec_crop

        yhat0 = np.fft.irfft(y_spec_full, n=self.cfg.H, norm="ortho").astype(self.real_dtype)
        return yhat0 + base

    def update_with_batch(self, X_ctx: np.ndarray, Y_true: np.ndarray) -> None:
        X_ctx = np.asarray(X_ctx, dtype=self.real_dtype)
        Y_true = np.asarray(Y_true, dtype=self.real_dtype)

        if X_ctx.ndim != 2 or X_ctx.shape[1] != self.cfg.L:
            raise ValueError(f"X_ctx must be [M, {self.cfg.L}]. Got {X_ctx.shape}.")
        if Y_true.ndim != 2 or Y_true.shape[1] != self.cfg.H:
            raise ValueError(f"Y_true must be [M, {self.cfg.H}]. Got {Y_true.shape}.")

        M = int(X_ctx.shape[0])
        if M == 0:
            return

        Xf = np.empty((M, self.d_x), dtype=self.cdtype)
        Yf = np.empty((M, self.d_y), dtype=self.cdtype)

        for i in range(M):
            base = self._baseline_value(X_ctx[i])        # IMPORTANT: same base for x and y
            Xf[i, :] = self._x_to_features(X_ctx[i], base=base)
            Yf[i, :] = self._y_to_targets(Y_true[i], base=base)

        self.XTY += Xf.conj().T @ Yf

        Ainv = self.A_inv
        XA = Xf @ Ainv

        middle = np.eye(M, dtype=self.cdtype) + XA @ Xf.conj().T
        if self.cfg.jitter > 0:
            middle += (self.cfg.jitter * np.eye(M, dtype=self.cdtype))

        T = np.linalg.solve(middle, XA)
        B = Ainv @ (Xf.conj().T @ T)

        self.A_inv = Ainv - B
        self.W = self.A_inv @ self.XTY
        self.num_windows_seen += M