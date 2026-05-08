import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional

from config import Config


class RankingDataset(Dataset):
    """
    One dataset item = one trial (3 candidates).

    Returns:
        features : (3, D) float32 — scaled feature vectors
        ranks    : (3,)   int64   — ai_rank (1=best, 3=worst)
        sku_pos  : (3,)   int64   — position shown to AI (1/2/3); needed for position-bias analysis
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        trial_keys: List[str],
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
        filter_ambiguous: bool = True,
    ):
        df = df.copy()

        if filter_ambiguous and "is_ambiguous" in df.columns:
            df = df[~df["is_ambiguous"].astype(bool)]

        # Impute NaN with column-level median (computed from this df slice;
        # callers ensure only training data is used when fit_scaler=True)
        medians = df[feature_cols].median()
        df[feature_cols] = df[feature_cols].fillna(medians)

        if fit_scaler:
            self.scaler = StandardScaler()
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            self.scaler = scaler
            if scaler is not None:
                df[feature_cols] = scaler.transform(df[feature_cols])

        self.trials: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for _, group in df.groupby(trial_keys, sort=False):
            if len(group) != 3:
                continue  # skip malformed / filtered-down trials
            group = group.sort_values("sku_pos")
            feats   = torch.tensor(group[feature_cols].values, dtype=torch.float32)
            ranks   = torch.tensor(group["ai_rank"].values,  dtype=torch.long)
            sku_pos = torch.tensor(group["sku_pos"].values,  dtype=torch.long)
            self.trials.append((feats, ranks, sku_pos))

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx: int):
        return self.trials[idx]


def build_loaders(
    cfg: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Loads train / val / test splits for the chosen protocol and returns
    DataLoaders plus the fitted StandardScaler (needed for inference).
    """
    def _load(split: str) -> pd.DataFrame:
        df = pd.read_csv(f"{cfg.data_dir}/{cfg.protocol}_{split}_features.csv")
        if cfg.engine_filter is not None:
            df = df[df["engine"] == cfg.engine_filter]
        return df

    train_df = _load("train")
    val_df   = _load("val")
    test_df  = _load("test")

    train_ds = RankingDataset(train_df, cfg.feature_cols, cfg.trial_keys, fit_scaler=True)
    val_ds   = RankingDataset(val_df,   cfg.feature_cols, cfg.trial_keys, scaler=train_ds.scaler)
    test_ds  = RankingDataset(test_df,  cfg.feature_cols, cfg.trial_keys, scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, train_ds.scaler
