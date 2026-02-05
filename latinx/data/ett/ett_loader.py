"""ETT (Electricity Transformer Temperature) dataset loader.

This module provides utilities to load the ETT datasets commonly used in
time series forecasting research. The datasets contain transformer temperature
readings recorded at regular intervals.

Datasets:
- ETTm1, ETTm2: Recorded every 15 minutes (m = minute-level)
- ETTh1, ETTh2: Recorded every hour (h = hour-level)

Each dataset contains 7 features:
- HUFL: High UseFul Load
- HULL: High UseLess Load
- MUFL: Middle UseFul Load
- MULL: Middle UseLess Load
- LUFL: Low UseFul Load
- LULL: Low UseLess Load
- OT: Oil Temperature (target variable in most benchmarks)

Reference:
    Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence
    Time-Series Forecasting", AAAI 2021.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


ETT_DATASET_URLS = {
    "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
}

ETT_FEATURE_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


@dataclass
class ETTSplitIndices:
    """Standard train/val/test split indices for ETT datasets."""

    train_end: int
    val_end: int
    total: int

    @property
    def train_slice(self) -> slice:
        return slice(0, self.train_end)

    @property
    def val_slice(self) -> slice:
        return slice(self.train_end, self.val_end)

    @property
    def test_slice(self) -> slice:
        return slice(self.val_end, self.total)


# Standard splits used in Informer and subsequent papers
ETT_SPLITS = {
    # ETTm1/ETTm2: 12 months train, 4 months val, 4 months test
    # 15-min intervals: 4*24*30 = 2880 per month, ~69120 total for 24 months
    "ETTm1": ETTSplitIndices(train_end=12*30*24*4, val_end=16*30*24*4, total=69680),
    "ETTm2": ETTSplitIndices(train_end=12*30*24*4, val_end=16*30*24*4, total=69680),
    # ETTh1/ETTh2: 12 months train, 4 months val, 4 months test
    # Hourly intervals: 24*30 = 720 per month
    "ETTh1": ETTSplitIndices(train_end=12*30*24, val_end=16*30*24, total=17420),
    "ETTh2": ETTSplitIndices(train_end=12*30*24, val_end=16*30*24, total=17420),
}


class ETTLoader:
    """
    Loader for ETT (Electricity Transformer Temperature) datasets.

    Downloads and caches the dataset, provides methods to access the data
    as pandas DataFrames or torch Tensors.

    Args:
        dataset: Which ETT dataset to load. One of 'ETTm1', 'ETTm2', 'ETTh1', 'ETTh2'.
        cache_dir: Directory to cache downloaded data. Defaults to ~/.cache/ett_data.
        target_column: The column to use as the primary target. Defaults to 'OT'.

    Example:
        >>> loader = ETTLoader(dataset="ETTm1")
        >>> df = loader.load()
        >>> train_df, val_df, test_df = loader.load_splits()
    """

    def __init__(
        self,
        dataset: Literal["ETTm1", "ETTm2", "ETTh1", "ETTh2"] = "ETTm1",
        cache_dir: Optional[str] = None,
        target_column: str = "OT",
    ):
        if dataset not in ETT_DATASET_URLS:
            raise ValueError(f"Unknown dataset: {dataset}. Must be one of {list(ETT_DATASET_URLS.keys())}")

        self.dataset = dataset
        self.target_column = target_column
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "ett_data"
        self._df: Optional[pd.DataFrame] = None

    @property
    def url(self) -> str:
        return ETT_DATASET_URLS[self.dataset]

    @property
    def cache_path(self) -> Path:
        return self.cache_dir / f"{self.dataset}.csv"

    @property
    def splits(self) -> ETTSplitIndices:
        return ETT_SPLITS[self.dataset]

    @property
    def feature_columns(self) -> List[str]:
        return ETT_FEATURE_COLUMNS.copy()

    def _download(self) -> None:
        """Download the dataset if not already cached."""
        import urllib.request

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.cache_path.exists():
            print(f"Downloading {self.dataset} from {self.url}...")
            urllib.request.urlretrieve(self.url, self.cache_path)
            print(f"Saved to {self.cache_path}")

    def load(self, force_download: bool = False) -> pd.DataFrame:
        """
        Load the full dataset as a pandas DataFrame.

        Args:
            force_download: If True, re-download even if cached.

        Returns:
            DataFrame with columns: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
        """
        if force_download and self.cache_path.exists():
            os.remove(self.cache_path)

        self._download()

        df = pd.read_csv(self.cache_path)
        df["date"] = pd.to_datetime(df["date"])
        self._df = df
        return df

    def load_splits(
        self,
        force_download: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load the dataset split into train/val/test sets.

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        df = self.load(force_download=force_download)

        train = df.iloc[self.splits.train_slice].reset_index(drop=True)
        val = df.iloc[self.splits.val_slice].reset_index(drop=True)
        test = df.iloc[self.splits.test_slice].reset_index(drop=True)

        return train, val, test

    def get_series(
        self,
        column: Optional[str] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
        as_tensor: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> np.ndarray | torch.Tensor:
        """
        Get a single column as a 1D array or tensor.

        Args:
            column: Column name. Defaults to target_column.
            split: If specified, return only that split's data.
            as_tensor: If True, return a torch.Tensor instead of numpy array.
            dtype: Torch dtype if as_tensor=True.

        Returns:
            1D numpy array or torch Tensor of the specified column.
        """
        if self._df is None:
            self.load()

        column = column or self.target_column

        if split == "train":
            data = self._df.iloc[self.splits.train_slice][column].values
        elif split == "val":
            data = self._df.iloc[self.splits.val_slice][column].values
        elif split == "test":
            data = self._df.iloc[self.splits.test_slice][column].values
        else:
            data = self._df[column].values

        if as_tensor:
            return torch.tensor(data, dtype=dtype)
        return data.astype(np.float32)

    def get_all_series(
        self,
        split: Optional[Literal["train", "val", "test"]] = None,
        as_tensor: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> List[np.ndarray] | List[torch.Tensor]:
        """
        Get all feature columns as a list of 1D arrays/tensors.

        Args:
            split: If specified, return only that split's data.
            as_tensor: If True, return torch.Tensors instead of numpy arrays.
            dtype: Torch dtype if as_tensor=True.

        Returns:
            List of 1D arrays/tensors, one per feature column.
        """
        return [
            self.get_series(col, split=split, as_tensor=as_tensor, dtype=dtype)
            for col in self.feature_columns
        ]

    def get_multivariate(
        self,
        columns: Optional[List[str]] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
        as_tensor: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> np.ndarray | torch.Tensor:
        """
        Get multiple columns as a 2D array [T, D].

        Args:
            columns: List of column names. Defaults to all feature columns.
            split: If specified, return only that split's data.
            as_tensor: If True, return a torch.Tensor instead of numpy array.
            dtype: Torch dtype if as_tensor=True.

        Returns:
            2D array/tensor of shape [T, D] where D is number of columns.
        """
        if self._df is None:
            self.load()

        columns = columns or self.feature_columns

        if split == "train":
            data = self._df.iloc[self.splits.train_slice][columns].values
        elif split == "val":
            data = self._df.iloc[self.splits.val_slice][columns].values
        elif split == "test":
            data = self._df.iloc[self.splits.test_slice][columns].values
        else:
            data = self._df[columns].values

        if as_tensor:
            return torch.tensor(data, dtype=dtype)
        return data.astype(np.float32)

    def info(self) -> dict:
        """Return dataset information."""
        if self._df is None:
            self.load()

        return {
            "dataset": self.dataset,
            "total_samples": len(self._df),
            "train_samples": self.splits.train_end,
            "val_samples": self.splits.val_end - self.splits.train_end,
            "test_samples": self.splits.total - self.splits.val_end,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "date_range": (self._df["date"].min(), self._df["date"].max()),
            "frequency": "15min" if "m" in self.dataset else "1H",
        }
