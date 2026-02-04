from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


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


# -------------------------
# Example usage
# -------------------------

def main() -> None:
    # Dummy data: two series
    T = 600
    t = torch.arange(T, dtype=torch.float32)
    s1 = torch.sin(0.02 * t) + 0.05 * torch.randn(T)
    s2 = torch.cos(0.015 * t + 1.0) + 0.05 * torch.randn(T)

    spec = WindowSpec(context_len=256, horizon=24, stride=8)
    ds = TimeSeriesWindowDataset([s1, s2], spec)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_windows)

    # Chronos model id (tiny is easiest on CPU)
    model_id = "amazon/chronos-t5-tiny"

    out = run_chronos_over_loader(
        model_id=model_id,
        loader=loader,
        torch_dtype=torch.float32,
        num_samples=128,
    )

    print("samples:", tuple(out["samples"].shape))  # [N,S,H]
    print("median :", tuple(out["median"].shape))   # [N,H]
    print("target :", tuple(out["target"].shape))   # [N,H]


if __name__ == "__main__":
    main()
