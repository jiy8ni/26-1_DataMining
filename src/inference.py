import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Callable, List, Optional, Tuple, Union

import lightgbm as lgb
import xgboost as xgb

from mlp.model import RecommendationScoreModel
from calibration import TemperatureCalibration


def _reduce_with_pca(
    df:        pd.DataFrame,
    id_col:    str,
    emb_df:    Optional[pd.DataFrame],
    pca:       Optional[PCA],
    raw_prefix: str,
    out_prefix: str,
) -> pd.DataFrame:
    """Return a DataFrame of PCA-reduced embedding columns (out_prefix*) aligned
    to df's rows. Missing embeddings are zero-filled. Empty/disabled -> empty."""
    if emb_df is None or pca is None:
        return pd.DataFrame(index=df.index)
    raw_cols = [c for c in emb_df.columns
                if c.startswith(raw_prefix) and c[len(raw_prefix):].isdigit()]
    if not raw_cols:
        return pd.DataFrame(index=df.index)
    merged = df[[id_col]].merge(
        emb_df[["resolved_url"] + raw_cols],
        left_on=id_col, right_on="resolved_url", how="left",
    )
    raw = merged[raw_cols].fillna(0.0).values
    reduced = pca.transform(raw)
    out_cols = [f"{out_prefix}{i}" for i in range(reduced.shape[1])]
    return pd.DataFrame(reduced, columns=out_cols, index=df.index)


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

class _BasePoolScorer:
    """
    Shared logic for pool scoring across model types.

    Subclasses supply `_predict_fns`: a list of callables
        fn(X: np.ndarray) -> np.ndarray  shape (N,)
    one per fold.  Raw scores are averaged across folds, then divided by
    the mean temperature for calibration.
    """

    def __init__(
        self,
        predict_fns:        List[Callable],
        calibrations:       List[TemperatureCalibration],
        pool_df:            pd.DataFrame,
        feature_cols:       list,
        scalers:            List[StandardScaler],
        id_col:             str = "url",
        log_transform_cols: list = None,
        text_emb_df:        Optional[pd.DataFrame] = None,
        image_emb_df:       Optional[pd.DataFrame] = None,
        pcas:               Optional[List[Tuple[Optional[PCA], Optional[PCA]]]] = None,
    ):
        self._predict_fns        = predict_fns
        self._log_transform_cols = log_transform_cols or []
        self.temperature         = float(np.mean([c.temperature for c in calibrations]))
        self.item_ids            = pool_df[id_col].tolist()
        self._id_col             = id_col

        # one (text_pca, image_pca) per fold; default no semantic reduction
        self._text_emb_df  = text_emb_df
        self._image_emb_df = image_emb_df
        self._pcas         = pcas if pcas is not None else [(None, None)] * len(scalers)

        self._X_list: List[np.ndarray] = self._build_fold_matrices(
            pool_df, feature_cols, scalers, id_col
        )

    def _build_fold_matrices(
        self,
        df:           pd.DataFrame,
        feature_cols: list,
        scalers:      List[StandardScaler],
        id_col:       str,
    ) -> List[np.ndarray]:
        """Per fold: structural (log1p+median) + fold-specific PCA-reduced
        embedding columns, in the same column order used at training time, then
        scaler.transform. Returns one matrix per fold."""
        base = df[feature_cols].copy()
        if self._log_transform_cols:
            cols = [c for c in self._log_transform_cols if c in feature_cols]
            base[cols] = np.log1p(base[cols].clip(lower=0))
        base = base.fillna(base.median())

        matrices = []
        for sc, (text_pca, image_pca) in zip(scalers, self._pcas):
            txt = _reduce_with_pca(df, id_col, self._text_emb_df, text_pca, "txt_", "txt_pca_")
            img = _reduce_with_pca(df, id_col, self._image_emb_df, image_pca, "img_", "img_pca_")
            fold_X = pd.concat([base.reset_index(drop=True),
                                txt.reset_index(drop=True),
                                img.reset_index(drop=True)], axis=1)
            matrices.append(sc.transform(fold_X))
        return matrices

    def score_pool(self) -> np.ndarray:
        """Temperature-calibrated scores for every pool item."""
        raw = np.mean([fn(X) for fn, X in zip(self._predict_fns, self._X_list)], axis=0)
        return raw / self.temperature

    def _score_new_item_raw(
        self,
        item_df:      pd.DataFrame,
        feature_cols: list,
        scalers:      List[StandardScaler],
        id_col:       str = "url",
    ) -> float:
        matrices = self._build_fold_matrices(item_df, feature_cols, scalers, id_col)
        per_fold = [fn(X) for fn, X in zip(self._predict_fns, matrices)]
        return float(np.mean(per_fold) / self.temperature)

    def score_new_item(
        self,
        item_df:      pd.DataFrame,
        feature_cols: list,
        scaler:       Union[StandardScaler, List[StandardScaler]],
    ) -> dict:
        """
        Score a single new product against the fixed pool distribution.

        Returns: score, pool_rank (1 = best), top_pct (lower = better).
        """
        scalers   = scaler if isinstance(scaler, list) else [scaler] * len(self._predict_fns)
        new_score = self._score_new_item_raw(item_df, feature_cols, scalers, self._id_col)

        pool_scores = self.score_pool()
        rank    = int((pool_scores > new_score).sum()) + 1
        top_pct = rank / len(pool_scores) * 100
        return {"score": new_score, "pool_rank": rank, "top_pct": top_pct}

    def get_report(self, reference_item_id: str = None) -> pd.DataFrame:
        """
        DataFrame sorted by pool_rank with columns:
            item_id, score, pool_rank, top_pct, rec_prob, odds_uplift
        """
        scores = self.score_pool()
        N      = len(scores)

        pool_rank = N - scores.argsort().argsort()
        top_pct   = pool_rank / N * 100

        exp_s    = np.exp(scores - scores.max())
        rec_prob = exp_s / exp_s.sum()

        mean_prob   = rec_prob.mean()
        mean_odds   = mean_prob / (1 - mean_prob + 1e-12)
        item_odds   = rec_prob / (1 - rec_prob + 1e-12)
        odds_uplift = item_odds / mean_odds

        report = pd.DataFrame({
            "item_id":     self.item_ids,
            "score":       scores,
            "pool_rank":   pool_rank,
            "top_pct":     top_pct,
            "rec_prob":    rec_prob,
            "odds_uplift": odds_uplift,
        }).sort_values("pool_rank").reset_index(drop=True)

        if reference_item_id is not None:
            row = report[report["item_id"] == reference_item_id]
            if not row.empty:
                r = row.iloc[0]
                print(
                    f"[{reference_item_id}]  rank {int(r.pool_rank)}/{N}  "
                    f"top {r.top_pct:.1f}%  "
                    f"rec_prob {r.rec_prob:.4f}  "
                    f"odds_uplift {r.odds_uplift:.2f}x"
                )

        return report


# ──────────────────────────────────────────────────────────────────────────────
# MLP (PyTorch)
# ──────────────────────────────────────────────────────────────────────────────

class PoolScorer(_BasePoolScorer):
    """
    Pool scorer for the MLP ranker.

    Accepts single model or a list of k-fold models.

    Usage (single):
        scorer = PoolScorer(model, calib, pool_df, feature_cols, scaler)
    Usage (k-fold ensemble):
        scorer = PoolScorer(models, calibs, pool_df, feature_cols, scalers)
    """

    def __init__(
        self,
        model:              Union[RecommendationScoreModel, List[RecommendationScoreModel]],
        calibration:        Union[TemperatureCalibration,   List[TemperatureCalibration]],
        pool_df:            pd.DataFrame,
        feature_cols:       list,
        scaler:             Union[StandardScaler, List[StandardScaler]],
        id_col:             str = "url",
        device:             str = "cpu",
        log_transform_cols: list = None,
        text_emb_df:        Optional[pd.DataFrame] = None,
        image_emb_df:       Optional[pd.DataFrame] = None,
        pcas:               Optional[List[Tuple[Optional[PCA], Optional[PCA]]]] = None,
    ):
        models       = model       if isinstance(model,       list) else [model]
        calibrations = calibration if isinstance(calibration, list) else [calibration]
        scalers      = scaler      if isinstance(scaler,      list) else [scaler]

        fitted_models = [m.to(device).eval() for m in models]

        def _make_predict_fn(m):
            @torch.no_grad()
            def fn(X: np.ndarray) -> np.ndarray:
                t = torch.tensor(X, dtype=torch.float32).to(device)
                return m(t).cpu().numpy()
            return fn

        predict_fns = [_make_predict_fn(m) for m in fitted_models]
        super().__init__(predict_fns, calibrations, pool_df, feature_cols, scalers, id_col,
                         log_transform_cols, text_emb_df, image_emb_df, pcas)


# ──────────────────────────────────────────────────────────────────────────────
# LightGBM
# ──────────────────────────────────────────────────────────────────────────────

class LGBMPoolScorer(_BasePoolScorer):
    """
    Pool scorer for the LightGBM LambdaRank ranker.

    Usage (single):
        scorer = LGBMPoolScorer(booster, calib, pool_df, feature_cols, scaler)
    Usage (k-fold ensemble):
        scorer = LGBMPoolScorer(boosters, calibs, pool_df, feature_cols, scalers)
    """

    def __init__(
        self,
        model:              Union[lgb.Booster, List[lgb.Booster]],
        calibration:        Union[TemperatureCalibration, List[TemperatureCalibration]],
        pool_df:            pd.DataFrame,
        feature_cols:       list,
        scaler:             Union[StandardScaler, List[StandardScaler]],
        id_col:             str = "url",
        log_transform_cols: list = None,
        text_emb_df:        Optional[pd.DataFrame] = None,
        image_emb_df:       Optional[pd.DataFrame] = None,
        pcas:               Optional[List[Tuple[Optional[PCA], Optional[PCA]]]] = None,
    ):
        models       = model       if isinstance(model,       list) else [model]
        calibrations = calibration if isinstance(calibration, list) else [calibration]
        scalers      = scaler      if isinstance(scaler,      list) else [scaler]

        predict_fns = [
            lambda X, m=m: m.predict(X, raw_score=True)
            for m in models
        ]
        super().__init__(predict_fns, calibrations, pool_df, feature_cols, scalers, id_col,
                         log_transform_cols, text_emb_df, image_emb_df, pcas)


# ──────────────────────────────────────────────────────────────────────────────
# XGBoost
# ──────────────────────────────────────────────────────────────────────────────

class XGBPoolScorer(_BasePoolScorer):
    """
    Pool scorer for the XGBoost LambdaRank ranker.

    Usage (single):
        scorer = XGBPoolScorer(booster, calib, pool_df, feature_cols, scaler)
    Usage (k-fold ensemble):
        scorer = XGBPoolScorer(boosters, calibs, pool_df, feature_cols, scalers)
    """

    def __init__(
        self,
        model:              Union[xgb.Booster, List[xgb.Booster]],
        calibration:        Union[TemperatureCalibration, List[TemperatureCalibration]],
        pool_df:            pd.DataFrame,
        feature_cols:       list,
        scaler:             Union[StandardScaler, List[StandardScaler]],
        id_col:             str = "url",
        log_transform_cols: list = None,
        text_emb_df:        Optional[pd.DataFrame] = None,
        image_emb_df:       Optional[pd.DataFrame] = None,
        pcas:               Optional[List[Tuple[Optional[PCA], Optional[PCA]]]] = None,
    ):
        models       = model       if isinstance(model,       list) else [model]
        calibrations = calibration if isinstance(calibration, list) else [calibration]
        scalers      = scaler      if isinstance(scaler,      list) else [scaler]

        predict_fns = [
            lambda X, m=m: m.predict(xgb.DMatrix(X))
            for m in models
        ]
        super().__init__(predict_fns, calibrations, pool_df, feature_cols, scalers, id_col,
                         log_transform_cols, text_emb_df, image_emb_df, pcas)
