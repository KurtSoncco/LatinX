"""
OLS Linear Forecaster – time-domain ridge regression baseline.

Based on the ICML 2024 paper "Are Linear Models Still Useful in Time Series
Forecasting?" (sir-lab/linear-forecasting) which proves that DLinear, NLinear,
RLinear, and FITS all reduce to standard linear regression.

This implements the simplest version: Ridge regression (SVD solver) with
optional instance normalization (per-window mean subtraction + std feature).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OLSLinearForecasterConfig:
    L: int                          # context length
    H: int                          # forecast horizon
    alpha: float = 1e-6             # ridge regularisation (near-zero ≈ OLS)
    instance_norm: bool = True      # per-window mean subtraction + std feature
    epsilon: float = 1e-5           # numerical stability for std


class OLSLinearForecaster:
    """Time-domain ridge-regression forecaster."""

    def __init__(self, cfg: OLSLinearForecasterConfig):
        self.cfg = cfg
        self._fitted = False
        self.W: np.ndarray | None = None   # [d_in, H]

    @property
    def _d_in(self) -> int:
        """Input dimension (L, or L+1 when instance_norm appends std)."""
        return self.cfg.L + 1 if self.cfg.instance_norm else self.cfg.L

    def _build_features(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build feature matrix and per-window means.

        Parameters
        ----------
        X : ndarray, shape [N, L]

        Returns
        -------
        X_aug : ndarray, shape [N, d_in]
        means : ndarray, shape [N]
        """
        means = X.mean(axis=1)                      # [N]
        X_centered = X - means[:, None]              # [N, L]

        if self.cfg.instance_norm:
            stds = X_centered.std(axis=1)            # [N]
            X_aug = np.column_stack([X_centered, stds])  # [N, L+1]
        else:
            X_aug = X_centered

        return X_aug, means

    def _build_features_single(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Build feature vector for a single window.

        Parameters
        ----------
        x : ndarray, shape [L]

        Returns
        -------
        x_aug : ndarray, shape [d_in]
        mean  : float
        """
        mean = float(x.mean())
        x_centered = x - mean

        if self.cfg.instance_norm:
            std = float(x_centered.std())
            x_aug = np.append(x_centered, std)
        else:
            x_aug = x_centered

        return x_aug, mean

    def fit(self, X_ctx: np.ndarray, Y_true: np.ndarray) -> OLSLinearForecaster:
        """Fit via closed-form ridge regression.

        Parameters
        ----------
        X_ctx  : ndarray, shape [N, L]
        Y_true : ndarray, shape [N, H]

        Returns
        -------
        self (for method chaining)
        """
        X_ctx = np.asarray(X_ctx, dtype=np.float64)
        Y_true = np.asarray(Y_true, dtype=np.float64)

        if X_ctx.ndim != 2 or X_ctx.shape[1] != self.cfg.L:
            raise ValueError(
                f"X_ctx must be [N, {self.cfg.L}], got {X_ctx.shape}"
            )
        if Y_true.ndim != 2 or Y_true.shape[1] != self.cfg.H:
            raise ValueError(
                f"Y_true must be [N, {self.cfg.H}], got {Y_true.shape}"
            )

        X_aug, means = self._build_features(X_ctx)        # [N, d_in], [N]
        Y_centered = Y_true - means[:, None]               # [N, H]

        # Solve (X^T X + alpha*I) W = X^T Y
        XTX = X_aug.T @ X_aug                              # [d_in, d_in]
        XTY = X_aug.T @ Y_centered                         # [d_in, H]
        reg = self.cfg.alpha * np.eye(self._d_in, dtype=np.float64)

        self.W = np.linalg.solve(XTX + reg, XTY)           # [d_in, H]
        self._fitted = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict horizon for a single context window.

        Parameters
        ----------
        x : ndarray, shape [L]

        Returns
        -------
        y_hat : ndarray, shape [H]
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit before calling predict.")

        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.cfg.L:
            raise ValueError(
                f"Expected context length {self.cfg.L}, got {x.shape[0]}"
            )

        x_aug, mean = self._build_features_single(x)
        y_hat = x_aug @ self.W                              # [H]
        y_hat += mean                                        # undo centering
        return y_hat

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict horizon for multiple context windows.

        Parameters
        ----------
        X : ndarray, shape [N, L]

        Returns
        -------
        Y_hat : ndarray, shape [N, H]
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit before calling predict_batch.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.cfg.L:
            raise ValueError(
                f"X must be [N, {self.cfg.L}], got {X.shape}"
            )

        X_aug, means = self._build_features(X)
        Y_hat = X_aug @ self.W                              # [N, H]
        Y_hat += means[:, None]                              # undo centering
        return Y_hat
