from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import time
from latinx.models.eft_linear_fft import ELFForecaster, ELFForecasterConfig
from latinx.data.ett import ETTLoader

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
    """
    Sliding windows over one or many 1D time series.

    series_list: list of 1D tensors, each [T]
    item:
      - context: [C]
      - target:  [H]
      - series_id: int
      - t0: start index
    """
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
    context = torch.stack([b["context"] for b in batch], dim=0)  # [B,C]
    target = torch.stack([b["target"] for b in batch], dim=0)    # [B,H]
    return {"series_id": series_id, "t0": t0, "context": context, "target": target}


# -------------------------
# Chronos runner
# -------------------------

def _pick_device() -> str:
    # ChronosPipeline uses HuggingFace style device_map strings.
    # On M2, "mps" works if supported by your torch build, otherwise use "cpu".
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def chronos_batch_forecast(
    pipeline,
    context_batch: torch.Tensor,   # [B,C]
    prediction_length: int,
    num_samples: int = 128,
) -> torch.Tensor:
    """
    Returns forecast samples with shape [B, S, H].

    ChronosPipeline.predict accepts:
      - a list of 1D tensors, OR
      - a left-padded 2D tensor [B, C]
    We pass the [B,C] tensor directly.

    The pipeline returns [num_series, num_samples, prediction_length].
    """
    # Make sure it is on CPU for the pipeline if needed; pipeline handles device internally,
    # but passing a CPU tensor is usually safest across setups.
    ctx = context_batch.detach().to("cpu")

    forecast = pipeline.predict(
        ctx,
        prediction_length=prediction_length,
        num_samples=num_samples,
    )
    # forecast: [B,S,H]
    return forecast


def run_chronos_over_loader(
    model_id: str,
    loader: DataLoader,
    torch_dtype: torch.dtype = torch.float32,
    num_samples: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Runs Chronos on each batch and concatenates outputs.
    Returns:
      - samples: [N, S, H]
      - median:  [N, H]
      - target:  [N, H]
      - series_id: [N]
      - t0: [N]
    """
    from chronos import ChronosPipeline  # pip install chronos-forecasting

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
        context = batch["context"]  # [B,C] on CPU
        target = batch["target"]    # [B,H]
        series_id = batch["series_id"]
        t0 = batch["t0"]

        samples = chronos_batch_forecast(
            pipeline=pipeline,
            context_batch=context,
            prediction_length=int(target.shape[1]),
            num_samples=num_samples,
        )  # [B,S,H]

        all_samples.append(samples)
        all_targets.append(target)
        all_series_id.append(series_id)
        all_t0.append(t0)

    samples_cat = torch.cat(all_samples, dim=0)     # [N,S,H]
    target_cat = torch.cat(all_targets, dim=0)      # [N,H]
    series_id_cat = torch.cat(all_series_id, dim=0) # [N]
    t0_cat = torch.cat(all_t0, dim=0)               # [N]

    median = samples_cat.quantile(0.5, dim=1)       # [N,H]

    return {
        "samples": samples_cat,
        "median": median,
        "target": target_cat,
        "series_id": series_id_cat,
        "t0": t0_cat,
    }




def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(h)
    return logger


def _plot_window(
    context: torch.Tensor,        # [C]
    target: torch.Tensor,         # [H]
    chronos_samples: torch.Tensor,  # [S,H]
    elf_pred: np.ndarray,         # [H]
    title: str,
    save_path: Optional[Path] = None,
) -> None:
    ctx = context.detach().cpu().numpy()
    tgt = target.detach().cpu().numpy()
    s = chronos_samples.detach().cpu().numpy()  # [S,H]

    q10 = np.quantile(s, 0.10, axis=0)
    q50 = np.quantile(s, 0.50, axis=0)
    q90 = np.quantile(s, 0.90, axis=0)

    C = ctx.shape[0]
    H = tgt.shape[0]
    x_ctx = np.arange(C)
    x_fut = np.arange(C, C + H)

    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.plot(x_ctx, ctx, label="context")
    plt.plot(x_fut, tgt, label="target")
    plt.plot(x_fut, q50, label="chronos median")
    plt.fill_between(x_fut, q10, q90, alpha=0.2, label="chronos 10-90")
    plt.plot(x_fut, elf_pred, label="elf fft")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")

    plt.show()


def main() -> None:
    logger = _setup_logger()
    torch.manual_seed(0)

    # Load ETTm1 dataset (Oil Temperature column)
    logger.info("Loading ETTm1 dataset...")
    ett_loader = ETTLoader(dataset="ETTm1", target_column="OT")
    ett_loader.load()
    info = ett_loader.info()
    logger.info(f"ETTm1: total_samples={info['total_samples']} train={info['train_samples']} val={info['val_samples']} test={info['test_samples']}")

    # Get OT (Oil Temperature) series from train split as primary series
    # Also get HUFL for variety (multiple series)
    # Limit to first N timesteps for faster iteration
    max_timesteps = 500  # ~207 windows per series with stride=8, horizon=96

    s1 = ett_loader.get_series("OT", split="train", as_tensor=True)[:max_timesteps]
    s2 = ett_loader.get_series("HUFL", split="train", as_tensor=True)[:max_timesteps]

    logger.info(f"Series shapes (limited to {max_timesteps}): OT={tuple(s1.shape)} HUFL={tuple(s2.shape)}")

    spec = WindowSpec(context_len=256, horizon=55, stride=8)  # horizon=96 = 24 hours at 15min intervals
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
        num_samples=128,
    )

    logger.info(f"Chronos: samples={tuple(out['samples'].shape)}  median={tuple(out['median'].shape)}  target={tuple(out['target'].shape)}")

    # Grab a single window to visualize (dataset order, shuffle=False)
    batch0 = next(iter(loader))
    context0 = batch0["context"][0]   # [C]
    target0 = batch0["target"][0]     # [H]
    chronos_samples0 = out["samples"][0]  # [S,H]

    # -------------------------
    # Run ELF FFT lightweight (no weighting)
    # -------------------------

    elf_cfg = ELFForecasterConfig(
        L=spec.context_len,
        H=spec.horizon,
        alpha=0.9,
        init_seasonal=True,
        baseline="last", lam=1e-1
    )
    elf = ELFForecaster(elf_cfg, real_dtype=np.float32)

    # Fit ELF on the first K windows (simple warmup)
    K = 64
    X = np.stack([ds[i]["context"].numpy() for i in range(min(K, len(ds)))], axis=0)
    Y = np.stack([ds[i]["target"].numpy() for i in range(min(K, len(ds)))], axis=0)

    logger.info(f"ELF: init alpha={elf_cfg.alpha} lam={elf_cfg.lam} d_x={elf.d_x} d_y={elf.d_y} fit_windows={X.shape[0]}")

    pred_before = elf.predict(context0.numpy())
    logger.info(f"ELF: pred(before) mean={float(pred_before.mean()):.4f} std={float(pred_before.std()):.4f}")

    t_update = time.time()
    elf.update_with_batch(X, Y)
    dt = time.time() - t_update
    logger.info(f"ELF: update took {dt:.3f}s  avg_per_window={(dt/max(1,X.shape[0]))*1000:.2f}ms  seen={elf.num_windows_seen}")

    pred_after = elf.predict(context0.numpy())
    logger.info(f"ELF: pred(after) mean={float(pred_after.mean()):.4f} std={float(pred_after.std()):.4f}")

    # -------------------------
    # Plot Chronos + ELF on same window
    # -------------------------
    plot_filename = f"chronos_vs_elf_t{max_timesteps}_h{spec.horizon}.png"
    _plot_window(
        context=context0,
        target=target0,
        chronos_samples=chronos_samples0,
        elf_pred=pred_after,
        title=f"Chronos vs ELF FFT (ETTm1 OT, T={max_timesteps}, H={spec.horizon})",
        save_path=PLOT_DIR / plot_filename,
    )


if __name__ == "__main__":
    main()