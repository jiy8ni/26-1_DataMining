import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional

from config import Config


def load_embeddings(cfg: Config) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load raw text/image embedding tables (keyed by resolved_url) if semantic
    features are enabled and the files exist. Returns (text_df, image_df); either
    may be None when missing."""
    if not getattr(cfg, "use_semantic_features", False):
        return None, None

    def _try(path: Optional[str]) -> Optional[pd.DataFrame]:
        if not path:
            return None
        try:
            return pd.read_parquet(path)
        except FileNotFoundError:
            return None

    text_df = _try(getattr(cfg, "text_emb_path", None))
    image_df = _try(getattr(cfg, "image_emb_path", None))
    return text_df, image_df


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
        text_emb_df: Optional[pd.DataFrame] = None,   # resolved_url + txt_*
        image_emb_df: Optional[pd.DataFrame] = None,  # resolved_url + img_* (+img_missing)
        text_pca_dim: int = 0,
        image_pca_dim: int = 0,
        pca: Optional[Tuple[Optional[PCA], Optional[PCA]]] = None,  # (text_pca, image_pca) for val/test
        medians: Optional[pd.Series] = None,  # train-fold medians for val/test imputation
    ):
        df = df.copy()

        if filter_ambiguous and "is_ambiguous" in df.columns:
            df = df[~df["is_ambiguous"].astype(bool)]

        # --- Semantic embeddings: merge raw vectors, PCA-reduce (train-fit) ---
        # Produces txt_pca_* / img_pca_* columns appended to feature_cols. PCA is
        # fit on the training fold only (when fit_scaler=True) to avoid leakage,
        # mirroring the StandardScaler pattern below. self.text_pca / self.image_pca
        # are stored so callers can pass them to val/test datasets.
        feature_cols = list(feature_cols)
        self.text_pca: Optional[PCA] = None
        self.image_pca: Optional[PCA] = None
        in_text_pca, in_image_pca = (pca if pca is not None else (None, None))

        df, feature_cols = self._add_pca_block(
            df, feature_cols, text_emb_df, "txt_", "txt_pca_",
            text_pca_dim, fit_scaler, in_text_pca, which="text",
        )
        df, feature_cols = self._add_pca_block(
            df, feature_cols, image_emb_df, "img_", "img_pca_",
            image_pca_dim, fit_scaler, in_image_pca, which="image",
        )
        self.feature_cols = feature_cols

        # log1p-transform skewed count/price features before imputation so that
        # medians and StandardScaler operate on the already-compressed scale
        if log_transform_cols:
            cols_to_transform = [c for c in log_transform_cols if c in feature_cols]
            df[cols_to_transform] = np.log1p(df[cols_to_transform].clip(lower=0))

        # Impute NaN with column-level median. Medians are fit on the training
        # fold only (fit_scaler=True) and reused for val/test/pool — mirroring the
        # StandardScaler pattern — so the same missing feature is filled with the
        # same value across splits (no train/serving skew, no test-distribution
        # leakage). Falls back to this df's medians only when none were passed in.
        if fit_scaler:
            self.medians = df[feature_cols].median()
        else:
            self.medians = medians if medians is not None else df[feature_cols].median()
        df[feature_cols] = df[feature_cols].fillna(self.medians)

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

    def _add_pca_block(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        emb_df: Optional[pd.DataFrame],
        raw_prefix: str,     # e.g. "txt_" / "img_"
        out_prefix: str,     # e.g. "txt_pca_" / "img_pca_"
        n_dim: int,
        fit: bool,
        in_pca: Optional[PCA],
        which: str,          # "text" | "image" (which attribute to store on self)
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Merge raw per-URL embeddings by resolved_url, PCA-reduce to n_dim, and
        append the reduced columns (out_prefix0..n) to feature_cols.

        PCA is fit on the training fold's UNIQUE items only (when fit=True) to
        avoid leakage and frequency-weighting bias; val/test reuse in_pca. URLs
        with no embedding row (failed crawl) get a zero raw vector, which maps to
        the origin in PCA space and is then median/scaler-handled downstream.
        """
        if emb_df is None or n_dim <= 0:
            return df, feature_cols

        # raw embedding dims only — exclude flag columns like 'img_missing'
        # (prefix-match alone would wrongly treat the flag as a dimension)
        raw_cols = [c for c in emb_df.columns
                    if c.startswith(raw_prefix) and c[len(raw_prefix):].isdigit()]
        if not raw_cols:
            return df, feature_cols

        # cap dims at the available embedding rank
        n_comp = min(n_dim, len(raw_cols))

        merged = df.merge(emb_df[["resolved_url"] + raw_cols],
                          on="resolved_url", how="left")
        raw = merged[raw_cols].to_numpy(dtype=np.float32)
        missing_mask = np.isnan(raw).any(axis=1)
        raw = np.nan_to_num(raw, nan=0.0)  # failed-crawl URLs -> zero vector

        if fit:
            # fit on unique resolved_urls present in this (train) split, excluding
            # missing-embedding rows, so PCA isn't dominated by row frequency
            uniq = merged.loc[~missing_mask, ["resolved_url"] + raw_cols] \
                         .drop_duplicates("resolved_url")
            fit_mat = uniq[raw_cols].to_numpy(dtype=np.float32)
            n_comp = min(n_comp, fit_mat.shape[0], fit_mat.shape[1])
            pca = PCA(n_components=n_comp, random_state=0)
            pca.fit(fit_mat)
            if which == "text":
                self.text_pca = pca
            else:
                self.image_pca = pca
        else:
            pca = in_pca
            if pca is None:
                return df, feature_cols
            if which == "text":
                self.text_pca = pca
            else:
                self.image_pca = pca

        reduced = pca.transform(raw)                      # (N_rows, n_comp)
        out_cols = [f"{out_prefix}{i}" for i in range(reduced.shape[1])]
        for j, col in enumerate(out_cols):
            df[col] = reduced[:, j]
        return df, feature_cols + out_cols

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


def _semantic_kwargs(cfg: Config, text_df, image_df) -> dict:
    """RankingDataset kwargs for semantic embeddings (empty if disabled)."""
    if not getattr(cfg, "use_semantic_features", False):
        return {}
    return dict(
        text_emb_df=text_df,
        image_emb_df=image_df,
        text_pca_dim=getattr(cfg, "text_pca_dim", 0),
        image_pca_dim=getattr(cfg, "image_pca_dim", 0),
    )


def _train_pca(train_ds: "RankingDataset") -> Tuple[Optional[PCA], Optional[PCA]]:
    return (train_ds.text_pca, train_ds.image_pca)


def semantic_added_dims(cfg: Config) -> int:
    """Number of feature columns the semantic block appends (txt_pca_* + img_pca_*).
    Returns 0 when semantic features are disabled or embedding files are absent."""
    if not getattr(cfg, "use_semantic_features", False):
        return 0
    text_df, image_df = load_embeddings(cfg)
    added = 0
    if text_df is not None and getattr(cfg, "text_pca_dim", 0) > 0:
        n_raw = sum(c.startswith("txt_") and c[4:].isdigit() for c in text_df.columns)
        added += min(cfg.text_pca_dim, n_raw)
    if image_df is not None and getattr(cfg, "image_pca_dim", 0) > 0:
        n_raw = sum(c.startswith("img_") and c[4:].isdigit() for c in image_df.columns)
        added += min(cfg.image_pca_dim, n_raw)
    return added


def effective_feature_dim(cfg: Config) -> int:
    """Total model input width: structural features + semantic PCA dims + position."""
    return (len(cfg.feature_cols)
            + semantic_added_dims(cfg)
            + (1 if cfg.use_position_feature else 0))


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
    text_df, image_df = load_embeddings(cfg)
    sem = _semantic_kwargs(cfg, text_df, image_df)
    train_ds = RankingDataset(train_df, cfg.feature_cols, cfg.trial_keys, fit_scaler=True,  log_transform_cols=log_cols, use_position_feature=use_pos, pl_df=pl_df, **sem)
    pca = _train_pca(train_ds)
    val_ds   = RankingDataset(val_df,   cfg.feature_cols, cfg.trial_keys, scaler=train_ds.scaler, log_transform_cols=log_cols, use_position_feature=use_pos, pl_df=pl_df, pca=pca, medians=train_ds.medians, **sem)
    test_ds  = RankingDataset(test_df,  cfg.feature_cols, cfg.trial_keys, scaler=train_ds.scaler, log_transform_cols=log_cols, use_position_feature=use_pos, pl_df=pl_df, pca=pca, medians=train_ds.medians, **sem)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, train_ds.scaler


def _ds_to_arrays(ds: RankingDataset) -> Tuple:
    """Convert a RankingDataset to numpy arrays for tree-based models.

    Returns: (X, relevance, ranks_2d, groups)
        X         : (N*3, D)  scaled feature matrix
        relevance : (N*3,)    4 - ai_rank  (LightGBM convention: higher = better)
        ranks_2d  : (N, 3)    original ai_rank values
        groups    : (N,)      all-3 array
    """
    X_list, rel_list, rank_list = [], [], []
    for feats, ranks, _, _ in ds:
        X_list.append(feats.numpy())
        rank_list.append(ranks.numpy())
        rel_list.append((4 - ranks).numpy())
    X         = np.concatenate(X_list, axis=0)
    relevance = np.concatenate(rel_list, axis=0).astype(np.float32)
    ranks_2d  = np.stack(rank_list, axis=0)
    groups    = np.full(len(ds), 3, dtype=np.int32)
    return X, relevance, ranks_2d, groups


def _brand_kfold_splits(
    df: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> list:
    """Brand-level k-fold: assign brands round-robin to folds, then mark any
    trial that contains a held-out brand as the val fold.

    A trial goes to fold k's val set if ANY of its brands belongs to fold k.
    A trial goes to fold k's train set if NONE of its brands belongs to fold k.
    (Trials whose brands span multiple folds appear in multiple val sets — this
    is correct for ensemble training.)

    Returns list of n_folds (train_df, val_df) tuples.
    """
    brands   = sorted(df["brand_ko"].dropna().unique())
    rng      = np.random.default_rng(seed)
    shuffled = rng.permutation(brands)
    brand_fold = {b: int(i % n_folds) for i, b in enumerate(shuffled)}

    trial_brands = df.groupby("set_id")["brand_ko"].apply(set)

    folds = []
    for k in range(n_folds):
        val_brand_set = {b for b, f in brand_fold.items() if f == k}
        val_ids   = frozenset(sid for sid, bs in trial_brands.items() if bs & val_brand_set)
        train_ids = frozenset(sid for sid in trial_brands.index if sid not in val_ids)
        folds.append((
            df[df["set_id"].isin(train_ids)].reset_index(drop=True),
            df[df["set_id"].isin(val_ids)].reset_index(drop=True),
        ))
    return folds


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
    text_df, image_df = load_embeddings(cfg)
    sem = _semantic_kwargs(cfg, text_df, image_df)

    train_ds = RankingDataset(train_df, cfg.feature_cols, cfg.trial_keys, fit_scaler=True,          log_transform_cols=log_cols, use_position_feature=use_pos, **sem)
    pca = _train_pca(train_ds)
    val_ds   = RankingDataset(val_df,   cfg.feature_cols, cfg.trial_keys, scaler=train_ds.scaler,   log_transform_cols=log_cols, use_position_feature=use_pos, pca=pca, medians=train_ds.medians, **sem)
    test_ds  = RankingDataset(test_df,  cfg.feature_cols, cfg.trial_keys, scaler=train_ds.scaler,   log_transform_cols=log_cols, use_position_feature=use_pos, pca=pca, medians=train_ds.medians, **sem)

    return (
        _ds_to_arrays(train_ds),
        _ds_to_arrays(val_ds),
        _ds_to_arrays(test_ds),
        train_ds.scaler,
    )


def build_kfold_arrays(cfg: Config) -> Tuple:
    """Brand-level k-fold for LightGBM / XGBoost.

    Combines protocol train+val, splits into cfg.n_folds brand-level folds.

    Returns:
        folds      : list of n_folds (train_arrays, val_arrays) where each
                     arrays tuple is (X, relevance, ranks_2d, groups)
        test_folds : list of n_folds test_arrays, each scaled with the
                     corresponding fold's scaler (needed for ensemble scoring)
        scalers    : list of n_folds StandardScalers
        pcas       : list of n_folds (text_pca, image_pca) tuples (None when
                     semantic features are disabled) — for inference-time reuse
    """
    def _load(split: str) -> pd.DataFrame:
        df = pd.read_csv(f"{cfg.data_dir}/{cfg.protocol}_{split}_features.csv")
        if cfg.engine_filter is not None:
            df = df[df["engine"] == cfg.engine_filter]
        return df

    combined = pd.concat([_load("train"), _load("val")], ignore_index=True)
    test_df  = _load("test")

    log_cols = getattr(cfg, "log_transform_cols", None)
    use_pos  = getattr(cfg, "use_position_feature", False)
    text_df, image_df = load_embeddings(cfg)
    sem = _semantic_kwargs(cfg, text_df, image_df)

    folds, test_folds, scalers, pcas = [], [], [], []
    for train_df, val_df in _brand_kfold_splits(combined, cfg.n_folds, cfg.seed):
        train_ds = RankingDataset(train_df, cfg.feature_cols, cfg.trial_keys,
                                  fit_scaler=True, log_transform_cols=log_cols,
                                  use_position_feature=use_pos, **sem)
        pca = _train_pca(train_ds)
        val_ds   = RankingDataset(val_df, cfg.feature_cols, cfg.trial_keys,
                                  scaler=train_ds.scaler, log_transform_cols=log_cols,
                                  use_position_feature=use_pos, pca=pca,
                                  medians=train_ds.medians, **sem)
        test_ds  = RankingDataset(test_df, cfg.feature_cols, cfg.trial_keys,
                                  scaler=train_ds.scaler, log_transform_cols=log_cols,
                                  use_position_feature=use_pos, pca=pca,
                                  medians=train_ds.medians, **sem)
        folds.append((_ds_to_arrays(train_ds), _ds_to_arrays(val_ds)))
        test_folds.append(_ds_to_arrays(test_ds))
        scalers.append(train_ds.scaler)
        pcas.append(pca)

    return folds, test_folds, scalers, pcas


def build_kfold_loaders(cfg: Config) -> Tuple:
    """Brand-level k-fold for the MLP trainer.

    Returns:
        folds        : list of n_folds (train_loader, val_loader) tuples
        test_loaders : list of n_folds test DataLoaders (fold-specific scaler)
        scalers      : list of n_folds StandardScalers
    """
    def _load(split: str) -> pd.DataFrame:
        df = pd.read_csv(f"{cfg.data_dir}/{cfg.protocol}_{split}_features.csv")
        if cfg.engine_filter is not None:
            df = df[df["engine"] == cfg.engine_filter]
        return df

    combined = pd.concat([_load("train"), _load("val")], ignore_index=True)
    test_df  = _load("test")

    log_cols = getattr(cfg, "log_transform_cols", None)
    use_pos  = getattr(cfg, "use_position_feature", False)
    text_df, image_df = load_embeddings(cfg)
    sem = _semantic_kwargs(cfg, text_df, image_df)

    folds, test_loaders, scalers = [], [], []
    for train_df, val_df in _brand_kfold_splits(combined, cfg.n_folds, cfg.seed):
        train_ds = RankingDataset(train_df, cfg.feature_cols, cfg.trial_keys,
                                  fit_scaler=True, log_transform_cols=log_cols,
                                  use_position_feature=use_pos, **sem)
        pca = _train_pca(train_ds)
        val_ds   = RankingDataset(val_df, cfg.feature_cols, cfg.trial_keys,
                                  scaler=train_ds.scaler, log_transform_cols=log_cols,
                                  use_position_feature=use_pos, pca=pca,
                                  medians=train_ds.medians, **sem)
        test_ds  = RankingDataset(test_df, cfg.feature_cols, cfg.trial_keys,
                                  scaler=train_ds.scaler, log_transform_cols=log_cols,
                                  use_position_feature=use_pos, pca=pca,
                                  medians=train_ds.medians, **sem)
        folds.append((
            DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  drop_last=False),
            DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, drop_last=False),
        ))
        test_loaders.append(
            DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
        )
        scalers.append(train_ds.scaler)

    return folds, test_loaders, scalers
