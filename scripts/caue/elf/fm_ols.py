"""
Chronos vs ELF vs OLS Linear comparison on ETTm1.

Same data setup as fm.py with the addition of the OLS Linear Forecaster
baseline from sir-lab/linear-forecasting (ICML 2024).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from latinx.data.ett import ETTLoader
from latinx.models.eft_linear_fft import ELFForecaster, ELFForecasterConfig
from latinx.models.ols_linear_forecaster import OLSLinearForecaster, OLSLinearForecasterConfig

import matplotlib.pyplot as plt

# Output directory for plots
PLOT_DIR = Path(__file__).parent / "plots"

# -------------------------
# Reusable window dataset
# -------------------------


@dataclass(frozen=True)
class WindowSpec:
    context_len: int
    horizon: int
    stride: int = 1


class TimeSeriesWindowDataset(Dataset):
    """Sliding windows over one or many 1D time series."""

    def __init__(
        self,
        series_list: List[torch.Tensor],
        spec: WindowSpec,
        dtype: torch.dtype = torch.float32,
    ):
        if not series_list:
            raise ValueError("series_list must be non-empty")

        self.series_list = [s.to(dtype=dtype).flatten() for s in series_list]
        self.spec = spec

        self._index: List[Tuple[int, int]] = []
        self._build_index()

        if len(self._index) == 0:
            raise ValueError("No windows created. Check series length vs context_len+horizon.")

    def _build_index(self) -> None:
        C, H, st = self.spec.context_len, self.spec.horizon, self.spec.stride
        for sid, s in enumerate(self.series_list):
            T = int(s.shape[0])
            max_t0 = T - (C + H)
            if max_t0 < 0:
                continue
            n = (max_t0 // st) + 1
            for k in range(n):
                self._index.append((sid, k * st))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid, t0 = self._index[idx]
        s = self.series_list[sid]
        C, H = self.spec.context_len, self.spec.horizon
        context = s[t0 : t0 + C]
        target = s[t0 + C : t0 + C + H]
        return {"series_id": sid, "t0": t0, "context": context, "target": target}


def collate_windows(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    series_id = torch.tensor([b["series_id"] for b in batch], dtype=torch.int64)
    t0 = torch.tensor([b["t0"] for b in batch], dtype=torch.int64)
    context = torch.stack([b["context"] for b in batch], dim=0)
    target = torch.stack([b["target"] for b in batch], dim=0)
    return {"series_id": series_id, "t0": t0, "context": context, "target": target}


# -------------------------
# Chronos runner
# -------------------------


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def chronos_batch_forecast(
    pipeline,
    context_batch: torch.Tensor,
    prediction_length: int,
    num_samples: int = 128,
) -> torch.Tensor:
    ctx = context_batch.detach().to("cpu")
    forecast = pipeline.predict(
        ctx,
        prediction_length=prediction_length,
        num_samples=num_samples,
    )
    return forecast


def run_chronos_over_loader(
    model_id: str,
    loader: DataLoader,
    torch_dtype: torch.dtype = torch.float32,
    num_samples: int = 128,
) -> Dict[str, torch.Tensor]:
    from chronos import ChronosPipeline

    device_map = _pick_device()
    pipeline = ChronosPipeline.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )

    all_samples: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    all_series_id: List[torch.Tensor] = []
    all_t0: List[torch.Tensor] = []

    for batch in loader:
        context = batch["context"]
        target = batch["target"]
        series_id = batch["series_id"]
        t0 = batch["t0"]

        samples = chronos_batch_forecast(
            pipeline=pipeline,
            context_batch=context,
            prediction_length=int(target.shape[1]),
            num_samples=num_samples,
        )

        all_samples.append(samples)
        all_targets.append(target)
        all_series_id.append(series_id)
        all_t0.append(t0)

    samples_cat = torch.cat(all_samples, dim=0)
    target_cat = torch.cat(all_targets, dim=0)
    series_id_cat = torch.cat(all_series_id, dim=0)
    t0_cat = torch.cat(all_t0, dim=0)
    median = samples_cat.quantile(0.5, dim=1)

    return {
        "samples": samples_cat,
        "median": median,
        "target": target_cat,
        "series_id": series_id_cat,
        "t0": t0_cat,
    }


# -------------------------
# Plotting
# -------------------------


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("fm_ols")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(h)
    return logger


def _plot_window(
    context: torch.Tensor,
    target: torch.Tensor,
    chronos_samples: torch.Tensor,
    elf_pred: np.ndarray,
    ols_pred: np.ndarray,
    title: str = "",
    save_path: Optional[Path] = None,
) -> None:
    ctx = context.detach().cpu().numpy()
    tgt = target.detach().cpu().numpy()
    s = chronos_samples.detach().cpu().numpy()

    q10 = np.quantile(s, 0.10, axis=0)
    q50 = np.quantile(s, 0.50, axis=0)
    q90 = np.quantile(s, 0.90, axis=0)

    C = ctx.shape[0]
    H = tgt.shape[0]
    x_ctx = np.arange(C)
    x_fut = np.arange(C, C + H)

    plt.figure(figsize=(14, 7))
    plt.title(title)
    plt.plot(x_ctx, ctx, label="context", color="gray")
    plt.plot(x_fut, tgt, label="target", color="black", linewidth=2)

    # Chronos
    plt.plot(x_fut, q50, label="chronos median", color="blue")
    plt.fill_between(x_fut, q10, q90, alpha=0.2, color="blue", label="chronos 10-90")

    # ELF
    plt.plot(x_fut, elf_pred, label="elf fft", color="green", linestyle="--")

    # OLS Linear
    plt.plot(x_fut, ols_pred, label="ols linear", color="red", linestyle="-.")

    plt.legend(loc="upper left")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")

    plt.show()


# -------------------------
# Main
# -------------------------


def main() -> None:
    logger = _setup_logger()
    torch.manual_seed(0)

    # Load ETTm1 dataset (Oil Temperature column)
    logger.info("Loading ETTm1 dataset...")
    ett_loader = ETTLoader(dataset="ETTm1", target_column="OT")
    ett_loader.load()
    info = ett_loader.info()
    logger.info(
        f"ETTm1: total_samples={info['total_samples']} "
        f"train={info['train_samples']} val={info['val_samples']} test={info['test_samples']}"
    )

    max_timesteps = 500

    s1 = ett_loader.get_series("OT", split="train", as_tensor=True)[:max_timesteps]
    s2 = ett_loader.get_series("HUFL", split="train", as_tensor=True)[:max_timesteps]

    logger.info(f"Series shapes (limited to {max_timesteps}): OT={tuple(s1.shape)} HUFL={tuple(s2.shape)}")

    spec = WindowSpec(context_len=256, horizon=55, stride=8)
    ds = TimeSeriesWindowDataset([s1, s2], spec)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_windows)

    logger.info(f"Dataset: windows={len(ds)} context_len={spec.context_len} horizon={spec.horizon} stride={spec.stride}")

    # -------------------------
    # Run Chronos FM
    # -------------------------
    model_id = "amazon/chronos-t5-tiny"
    logger.info(f"Chronos: running model_id={model_id}")

    out = run_chronos_over_loader(
        model_id=model_id,
        loader=loader,
        torch_dtype=torch.float32,
        num_samples=164,
    )

    logger.info(
        f"Chronos: samples={tuple(out['samples'].shape)}  "
        f"median={tuple(out['median'].shape)}  target={tuple(out['target'].shape)}"
    )                                                       

    # -------------------------
    # Clean temporal split (no leakage)
    # -------------------------
    # Split point: training data lives in [0, T_split), test target in [T_split, T_split+H).
    # The test context is [T_split - L, T_split) — it overlaps with training time range
    # (that's fine, we're testing the model's ability to forecast ahead).
    # Training windows must satisfy: t0 + L + H <= T_split (entire window within train region).
    L, H = spec.context_len, spec.horizon
    T_split = max_timesteps - H  # test target = last H timesteps of the series

    s0_indices = [i for i in range(len(ds)) if ds[i]["series_id"] == 0]
    logger.info(f"Series 0 (OT) has {len(s0_indices)} windows, T_split={T_split}")

    # Test: the window whose target starts exactly at T_split
    test_t0 = T_split - L
    test_idx = None
    for i in s0_indices:
        if ds[i]["t0"] == test_t0:
            test_idx = i
            break
    # If exact t0 not in the strided index, pick the last window that starts at or before test_t0
    if test_idx is None:
        candidates = [i for i in s0_indices if ds[i]["t0"] <= test_t0]
        test_idx = candidates[-1]
        test_t0 = ds[test_idx]["t0"]
        logger.info(f"Exact test t0 not in stride grid, using closest: t0={test_t0}")

    # Train: only windows whose target ends at or before T_split (no target leakage)
    train_indices = [
        i for i in s0_indices
        if ds[i]["t0"] + L + H <= T_split and i != test_idx
    ]

    context0 = ds[test_idx]["context"]
    target0 = ds[test_idx]["target"]
    chronos_samples0 = out["samples"][test_idx]

    logger.info(
        f"Test: window {test_idx} (t0={test_t0}, target=[{test_t0+L}:{test_t0+L+H}]), "
        f"Train: {len(train_indices)} windows (targets end <= {T_split})"
    )

    # Build training arrays
    X_train = np.stack([ds[i]["context"].numpy() for i in train_indices], axis=0)
    Y_train = np.stack([ds[i]["target"].numpy() for i in train_indices], axis=0)

    # -------------------------
    # Run ELF FFT
    # -------------------------
    elf_cfg = ELFForecasterConfig(
        L=spec.context_len,
        H=spec.horizon,
        alpha=0.9,
        init_seasonal=True,
        baseline="last",
        lam=1e-1,
    )
    elf = ELFForecaster(elf_cfg, real_dtype=np.float32)

    logger.info(f"ELF: alpha={elf_cfg.alpha} lam={elf_cfg.lam} d_x={elf.d_x} d_y={elf.d_y} train_windows={X_train.shape[0]}")

    t_elf = time.time()
    elf.update_with_batch(X_train, Y_train)
    dt_elf = time.time() - t_elf
    logger.info(f"ELF: update took {dt_elf:.3f}s  seen={elf.num_windows_seen}")

    pred_elf = elf.predict(context0.numpy())

    # -------------------------
    # Run OLS Linear
    # -------------------------
    ols_cfg = OLSLinearForecasterConfig(
        L=spec.context_len,
        H=spec.horizon,
        alpha=1e-6,
        instance_norm=True,
    )
    ols = OLSLinearForecaster(ols_cfg)

    logger.info(f"OLS: alpha={ols_cfg.alpha} instance_norm={ols_cfg.instance_norm} d_in={ols._d_in}")

    t_ols = time.time()
    ols.fit(X_train, Y_train)
    dt_ols = time.time() - t_ols
    logger.info(f"OLS: fit took {dt_ols:.3f}s  W_shape={ols.W.shape}")

    pred_ols = ols.predict(context0.numpy())

    # -------------------------
    # Compare MSE
    # -------------------------
    tgt_np = target0.numpy()
    chronos_median = out["median"][test_idx].numpy()

    mse_chronos = float(np.mean((chronos_median - tgt_np) ** 2))
    mse_elf = float(np.mean((pred_elf - tgt_np) ** 2))
    mse_ols = float(np.mean((pred_ols - tgt_np) ** 2))

    logger.info(f"MSE: Chronos={mse_chronos:.4f}  ELF={mse_elf:.4f}  OLS={mse_ols:.4f}")

    # -------------------------
    # Plot Chronos vs ELF vs OLS
    # -------------------------
    plot_filename = f"chronos_vs_elf_vs_ols_t{max_timesteps}_h{spec.horizon}.png"
    _plot_window(
        context=context0,
        target=target0,
        chronos_samples=chronos_samples0,
        elf_pred=pred_elf,
        ols_pred=pred_ols,
        title=f"Chronos vs ELF vs OLS Linear (ETTm1 OT, T={max_timesteps}, H={spec.horizon})",
        save_path=PLOT_DIR / plot_filename,
    )


if __name__ == "__main__":
    main()
