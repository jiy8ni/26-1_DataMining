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
        log_transform_cols: Optional[List[str]] = None,
        use_position_feature: bool = False,
        pl_df: Optional[pd.DataFrame] = None,  # columns: resolved_url, pl_theta
    ):
        df = df.copy()

        if filter_ambiguous and "is_ambiguous" in df.columns:
            df = df[~df["is_ambiguous"].astype(bool)]

        # log1p-transform skewed count/price features before imputation so that
        # medians and StandardScaler operate on the already-compressed scale
        if log_transform_cols:
            cols_to_transform = [c for c in log_transform_cols if c in feature_cols]
            df[cols_to_transform] = np.log1p(df[cols_to_transform].clip(lower=0))

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

        # Merge PL-fitted theta (if provided); default 0.0 for missing items
        if pl_df is not None:
            df = df.merge(
                pl_df[["resolved_url", "pl_theta"]],
                on="resolved_url",
                how="left",
            )
            df["pl_theta"] = df["pl_theta"].fillna(0.0)
        else:
            df["pl_theta"] = 0.0

        self.trials: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for _, group in df.groupby(trial_keys, sort=False):
            if len(group) != 3:
                continue  # skip malformed / filtered-down trials
            group = group.sort_values("sku_pos")
            feats    = torch.tensor(group[feature_cols].values,  dtype=torch.float32)
            ranks    = torch.tensor(group["ai_rank"].values,     dtype=torch.long)
            sku_pos  = torch.tensor(group["sku_pos"].values,     dtype=torch.long)
            pl_theta = torch.tensor(group["pl_theta"].values,    dtype=torch.float32)
            if use_position_feature:
                pos_feat = (sku_pos.float() - 1) / 2  # normalize 1/2/3 → 0/0.5/1
                feats = torch.cat([feats, pos_feat.unsqueeze(1)], dim=1)
            self.trials.append((feats, ranks, sku_pos, pl_theta))

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx: int):
        return self.trials[idx]


def _load_pl_df(cfg: Config) -> Optional[pd.DataFrame]:
    path = getattr(cfg, "pl_labels_path", None)
    if path is None:
        return None
    try:
        return pd.read_csv(path)[["resolved_url", "pl_theta"]]
    except FileNotFoundError:
        return None


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

    log_cols = getattr(cfg, "log_transform_cols", None)
    use_pos  = getattr(cfg, "use_position_feature", False)
    pl_df    = _load_pl_df(cfg)
    train_ds = RankingDataset(train_df, cfg.feature_cols, cfg.trial_keys, fit_scaler=True,  log_transform_cols=log_cols, use_position_feature=use_pos, pl_df=pl_df)
    val_ds   = RankingDataset(val_df,   cfg.feature_cols, cfg.trial_keys, scaler=train_ds.scaler, log_transform_cols=log_cols, use_position_feature=use_pos, pl_df=pl_df)
    test_ds  = RankingDataset(test_df,  cfg.feature_cols, cfg.trial_keys, scaler=train_ds.scaler, log_transform_cols=log_cols, use_position_feature=use_pos, pl_df=pl_df)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, train_ds.scaler


def build_arrays(
    cfg: Config,
) -> Tuple:
    """
    Same preprocessing pipeline as build_loaders but returns numpy arrays
    for LightGBM LambdaRank.

    Returns for each split: (X, relevance, ranks, groups)
        X         : (N_items, D)   scaled feature matrix  (N_items = N_trials * 3)
        relevance : (N_items,)     4 - ai_rank  (higher = better; LightGBM convention)
        ranks     : (N_trials, 3)  original ai_rank values for metrics / calibration
        groups    : (N_trials,)    all-3 array (items per trial)
    Also returns the fitted StandardScaler.
    """
    def _load(split: str) -> pd.DataFrame:
        df = pd.read_csv(f"{cfg.data_dir}/{cfg.protocol}_{split}_features.csv")
        if cfg.engine_filter is not None:
            df = df[df["engine"] == cfg.engine_filter]
        return df

    train_df = _load("train")
    val_df   = _load("val")
    test_df  = _load("test")

    log_cols = getattr(cfg, "log_transform_cols", None)
    use_pos  = getattr(cfg, "use_position_feature", False)

    train_ds = RankingDataset(train_df, cfg.feature_cols, cfg.trial_keys, fit_scaler=True,          log_transform_cols=log_cols, use_position_feature=use_pos)
    val_ds   = RankingDataset(val_df,   cfg.feature_cols, cfg.trial_keys, scaler=train_ds.scaler,   log_transform_cols=log_cols, use_position_feature=use_pos)
    test_ds  = RankingDataset(test_df,  cfg.feature_cols, cfg.trial_keys, scaler=train_ds.scaler,   log_transform_cols=log_cols, use_position_feature=use_pos)

    def _extract(ds: RankingDataset):
        X_list, rel_list, rank_list = [], [], []
        for feats, ranks, _ in ds:
            X_list.append(feats.numpy())
            rank_list.append(ranks.numpy())
            rel_list.append((4 - ranks).numpy())          # rank1→3, rank2→2, rank3→1
        X         = np.concatenate(X_list, axis=0)        # (N*3, D)
        relevance = np.concatenate(rel_list, axis=0).astype(np.float32)
        ranks_2d  = np.stack(rank_list, axis=0)           # (N, 3)
        groups    = np.full(len(ds), 3, dtype=np.int32)
        return X, relevance, ranks_2d, groups

    return (
        _extract(train_ds),
        _extract(val_ds),
        _extract(test_ds),
        train_ds.scaler,
    )
