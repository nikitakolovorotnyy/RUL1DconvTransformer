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
    """

    def __init__(
            self,
            data_file: str,
            window_size: int,
            scaler: Optional[MinMaxScaler] = None,
            max_rul: int = 125,
            is_train: bool = None,
            truth_file: Optional[str] = None,
            smoothing_alpha: Optional[float] = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.max_rul = max_rul
        self.is_train = is_train
        self.smoothing_alpha = smoothing_alpha

        col_names = [
            "unit", "cycle",
            # 3 operational settings
            "op1", "op2", "op3",
            # 21 sensor measurements
            *[f"s{i}" for i in range(1, 22)]
        ]
        raw_data = pd.read_csv(
            data_file,
            sep="\s+",
            header=None,
            names=col_names,
            dtype={"unit": int, "cycle": int}
        )

        # Compute RUL for each unit (train) or read true RUL (test)
        if self.is_train:
            # For each unit, compute its last cycle, subtract current cycle to get RUL, then clip
            rul_df = (raw_data.groupby("unit")["cycle"].max().reset_index().rename(columns={"cycle": "max_cycle"}))
            raw_data = raw_data.merge(rul_df, on="unit", how="left")
            raw_data["RUL"] = raw_data["max_cycle"] - raw_data["cycle"]
            raw_data["RUL"] = raw_data["RUL"].clip(upper=self.max_rul)
            raw_data.drop(columns=["max_cycle"], inplace=True)
        else:
            if truth_file is None:
                raise ValueError("truth_file must be provided for test data.")
            true_rul = pd.read_csv(truth_file, sep="\s+", header=None, usecols=[0], names=["RUL"])
            true_rul["unit"] = true_rul.index + 1
            # Get last cycle per unit in test data
            last_cycle = raw_data.groupby("unit")["cycle"].max().reset_index().rename(columns={"cycle": "last_cycle"})
            true_rul = true_rul.merge(last_cycle, on="unit", how="left")

            raw_data = raw_data.merge(true_rul, on="unit", how="left")
            raw_data["RUL"] = raw_data["RUL"] + (raw_data["last_cycle"] - raw_data["cycle"])
            raw_data["RUL"] = raw_data["RUL"].clip(upper=self.max_rul)
            raw_data.drop(columns=["last_cycle"], inplace=True)

        feature_cols = [col for col in raw_data.columns if col not in ["unit", "cycle", "RUL"]]

        if self.smoothing_alpha is not None:
            raw_data.sort_values(["unit", "cycle"], inplace=True)
            for col in feature_cols:
                raw_data[col] = (raw_data.groupby("unit")[col].transform(lambda x: x.ewm(alpha=self.smoothing_alpha,
                                                                                         adjust=False).mean()))

        features = raw_data[feature_cols].values.astype(np.float32)

        if scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            self.scaler.fit(features)
        else:
            self.scaler = scaler

        scaled_features = self.scaler.transform(features)
        raw_data[feature_cols] = scaled_features
        self.data = raw_data
        self.samples: List[Tuple[np.ndarray, float]] = []
        self._build_sliding_windows(feature_cols)

    def _build_sliding_windows(self, feature_cols: List[str]):
        """
        For each unit, create overlapping windows of length self.window_size.
        Each window is a 2D array [window_size, num_features], and the target RUL is at the window's end.
        """
        for unit_id, group in self.data.groupby("unit"):
            group = group.sort_values("cycle").reset_index(drop=True)
            feature_array = group[feature_cols].values  # (num_cycles, num_features)
            rul_array = group["RUL"].values  # (num_cycles,)

            num_cycles = feature_array.shape[0]
            if self.is_train:
                if num_cycles < self.window_size:
                    continue
                for end_idx in range(self.window_size - 1, num_cycles):
                    start_idx = end_idx - self.window_size + 1
                    window_feats = feature_array[start_idx:end_idx + 1]
                    window_rul = float(rul_array[end_idx])
                    self.samples.append((window_feats, window_rul))

            else:
                if num_cycles < self.window_size:
                    continue
                end_idx = num_cycles - 1
                start_idx = end_idx - self.window_size + 1
                window_feats = feature_array[start_idx:end_idx + 1]
                window_rul = float(rul_array[end_idx])
                self.samples.append((window_feats, window_rul))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window_feats, window_rul = self.samples[idx]
        feats_tensor = torch.from_numpy(window_feats)  # torch.float32
        rul_tensor = torch.tensor(window_rul, dtype=torch.float32)
        return feats_tensor, rul_tensor


def get_dataloaders(
        train_file: str,
        test_file: str,
        truth_file: str,
        window_size: int,
        train_batch: int,
        train_workers: int,
        test_batch: int = None,
        test_workers: int = None,
        smoothing_alpha: Optional[float] = 0.3,
):
    if test_batch is None:
        test_batch = train_batch
    if test_workers is None:
        test_workers = train_workers

    train_dataset = CMapssDataset(data_file=train_file, window_size=window_size, scaler=None, is_train=True,
                                  truth_file=None, smoothing_alpha=smoothing_alpha)
    fitted_scaler = train_dataset.scaler

    test_dataset = CMapssDataset(data_file=test_file, window_size=window_size, scaler=fitted_scaler, is_train=False,
                                 truth_file=truth_file, smoothing_alpha=smoothing_alpha)

    train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, num_workers=train_workers,
                              drop_last=False, pin_memory=True, persistent_workers=(train_workers > 0))

    test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=False, num_workers=test_workers,
                             drop_last=False, pin_memory=False, persistent_workers=(test_workers > 0))

    return train_loader, test_loader
