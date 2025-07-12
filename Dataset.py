import torch
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


class CMapssDataset(Dataset):
    """
    PyTorch Dataset for NASA C-MAPSS turbofan engine degradation dataset.
    - Handles both train and test subsets.
    - Drops irrelevant sensors to reduce noise.
    - Applies MinMax scaling per feature.
    - Generates sliding windows of fixed length.
    - Computes RUL (Remaining Useful Life), clipped at a maximum (e.g., 125).
    - Applies exponential smoothing to input features if desired.
    Args:
            data_file - path to the C-MAPSS data file.
            window_size - length of sliding window.
            scaler - feature scaling technique.
            max_rul - maximum RUL value to clip at.
            is_train - indicates whether this is a training set.
            truth_file - path to the RUL truth file for test data.
            smoothing_alpha - exponential smoothing factor.
            include_units - list of unit IDs to include (for train/val split)
    """

    # List of sensors to remove (constant or non-informative)
    IRRELEVANT_SENSORS = ["op1", "op2", "op3", "s1", "s5", "s6", "s10", "s16", "s18", "s19"]

    def __init__(
            self,
            data_file: str,
            window_size: int,
            scaler: Optional[MinMaxScaler] = None,
            max_rul: int = 125,
            is_train: bool = None,
            truth_file: Optional[str] = None,
            smoothing_alpha: Optional[float] = None,
            include_units: Optional[List[int]] = None
    ):
        super().__init__()
        self.window_size = window_size
        self.max_rul = max_rul
        self.is_train = is_train
        self.smoothing_alpha = smoothing_alpha

        # Column names including 3 operational settings and 21 sensors
        col_names = [
            "unit", "cycle",
            "op1", "op2", "op3",
            *[f"s{i}" for i in range(1, 22)]
        ]
        raw_data = pd.read_csv(
            data_file,
            sep="\s+",
            header=None,
            names=col_names,
            dtype={"unit": int, "cycle": int}
        )

        # Drop irrelevant sensors early to reduce noise
        raw_data.drop(columns=self.IRRELEVANT_SENSORS, inplace=True)

        # Filter by selected units (for train/val split)
        if include_units is not None:
            raw_data = raw_data[raw_data["unit"].isin(include_units)].copy()

        # Compute RUL for train or load for test
        if self.is_train:
            rul_df = raw_data.groupby("unit")["cycle"].max().reset_index().rename(columns={"cycle": "max_cycle"})
            raw_data = raw_data.merge(rul_df, on="unit", how="left")
            raw_data["RUL"] = (raw_data["max_cycle"] - raw_data["cycle"]).clip(upper=self.max_rul)
            raw_data.drop(columns=["max_cycle"], inplace=True)
        else:
            if truth_file is None:
                raise ValueError("truth_file must be provided for test data.")
            true_rul = pd.read_csv(truth_file, sep="\s+", header=None, usecols=[0], names=["RUL"])
            true_rul["unit"] = true_rul.index + 1
            last_cycle = raw_data.groupby("unit")["cycle"].max().reset_index().rename(columns={"cycle": "last_cycle"})
            true_rul = true_rul.merge(last_cycle, on="unit", how="left")
            raw_data = raw_data.merge(true_rul, on="unit", how="left")
            raw_data["RUL"] = (raw_data["RUL"] + raw_data["last_cycle"] - raw_data["cycle"]).clip(upper=self.max_rul)
            raw_data.drop(columns=["last_cycle"], inplace=True)

        # Determine feature columns after dropping irrelevant sensors
        feature_cols = [col for col in raw_data.columns if col not in ["unit", "cycle", "RUL"]]

        # Optional exponential smoothing
        if self.smoothing_alpha is not None:
            raw_data.sort_values(["unit", "cycle"], inplace=True)
            for col in feature_cols:
                raw_data[col] = raw_data.groupby("unit")[col].transform(
                    lambda x: x.ewm(alpha=self.smoothing_alpha, adjust=False).mean()
                )

        # Scaling
        features = raw_data[feature_cols].values.astype(np.float32)
        if scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            self.scaler.fit(features)
        else:
            self.scaler = scaler
        raw_data[feature_cols] = self.scaler.transform(features)

        self.data = raw_data
        self.samples: List[Tuple[np.ndarray, float]] = []
        self._build_sliding_windows(feature_cols)

    def _build_sliding_windows(self, feature_cols: List[str]):
        for unit_id, group in self.data.groupby("unit"):
            group = group.sort_values("cycle").reset_index(drop=True)
            feats = group[feature_cols].values
            rul_vals = group["RUL"].values
            cycles = feats.shape[0]
            if cycles < self.window_size:
                continue
            if self.is_train:
                for end in range(self.window_size - 1, cycles):
                    start = end - self.window_size + 1
                    self.samples.append((feats[start:end + 1], float(rul_vals[end])))
            else:
                end = cycles - 1
                start = end - self.window_size + 1
                self.samples.append((feats[start:end + 1], float(rul_vals[end])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feats, rul = self.samples[idx]
        return torch.from_numpy(feats), torch.tensor(rul, dtype=torch.float32)


def get_dataloaders(
        train_file: str,
        test_file: str,
        truth_file: str,
        window_size: int,
        train_batch: int,
        train_workers: int,
        eval_batch: int = None,
        eval_workers: int = None,
        val_fraction: float = 0.1,
        smoothing_alpha: Optional[float] = 0.3,
):
    # Determine train/val split
    all_units = pd.read_csv(train_file, sep="\s+", header=None, usecols=[0], names=["unit"])['unit'].unique().tolist()
    all_units.sort()
    n_val = max(1, int(len(all_units) * val_fraction))
    val_units = all_units[-n_val:]
    train_units = [u for u in all_units if u not in val_units]

    # Create datasets
    train_ds = CMapssDataset(train_file, window_size, scaler=None, is_train=True,
                             smoothing_alpha=smoothing_alpha, include_units=train_units)
    scaler = train_ds.scaler

    test_ds = CMapssDataset(test_file, window_size, scaler, is_train=False,
                            truth_file=truth_file, smoothing_alpha=smoothing_alpha)
    val_ds = CMapssDataset(train_file, window_size, scaler, is_train=True,
                           smoothing_alpha=smoothing_alpha, include_units=val_units)

    train_loader = DataLoader(train_ds, batch_size=train_batch, shuffle=True,
                              num_workers=train_workers, pin_memory=True,
                              persistent_workers=train_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=eval_batch or train_batch, shuffle=False,
                            num_workers=eval_workers or train_workers, pin_memory=False,
                            persistent_workers=(eval_workers or train_workers) > 0)
    test_loader = DataLoader(test_ds, batch_size=eval_batch or train_batch, shuffle=False,
                             num_workers=eval_workers or train_workers, pin_memory=False,
                             persistent_workers=(eval_workers or train_workers) > 0)

    return train_loader, test_loader, val_loader

    # print(f"Train units: {len(train_units)} | Val units: {len(val_units)}")
    # print(f"Val units IDs: {val_units}")
