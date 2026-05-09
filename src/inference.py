import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from typing import Callable, List, Union

import lightgbm as lgb
import xgboost as xgb

from model import RecommendationScoreModel
from calibration import TemperatureCalibration


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
    ):
        self._predict_fns       = predict_fns
        self._log_transform_cols = log_transform_cols or []
        self.temperature        = float(np.mean([c.temperature for c in calibrations]))
        self.item_ids           = pool_df[id_col].tolist()

        base_X = pool_df[feature_cols].copy()
        if self._log_transform_cols:
            cols = [c for c in self._log_transform_cols if c in feature_cols]
            base_X[cols] = np.log1p(base_X[cols].clip(lower=0))
        base_X = base_X.fillna(base_X.median())
        self._X_list: List[np.ndarray] = [sc.transform(base_X) for sc in scalers]

    def score_pool(self) -> np.ndarray:
        """Temperature-calibrated scores for every pool item."""
        raw = np.mean([fn(X) for fn, X in zip(self._predict_fns, self._X_list)], axis=0)
        return raw / self.temperature

    def _score_new_item_raw(
        self,
        item_df:      pd.DataFrame,
        feature_cols: list,
        scalers:      List[StandardScaler],
    ) -> float:
        base_X = item_df[feature_cols].copy()
        if self._log_transform_cols:
            cols = [c for c in self._log_transform_cols if c in feature_cols]
            base_X[cols] = np.log1p(base_X[cols].clip(lower=0))
        base_X = base_X.fillna(base_X.median())
        per_fold = [fn(sc.transform(base_X)) for fn, sc in zip(self._predict_fns, scalers)]
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
        new_score = self._score_new_item_raw(item_df, feature_cols, scalers)

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
        super().__init__(predict_fns, calibrations, pool_df, feature_cols, scalers, id_col, log_transform_cols)


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
    ):
        models       = model       if isinstance(model,       list) else [model]
        calibrations = calibration if isinstance(calibration, list) else [calibration]
        scalers      = scaler      if isinstance(scaler,      list) else [scaler]

        predict_fns = [
            lambda X, m=m: m.predict(X, raw_score=True)
            for m in models
        ]
        super().__init__(predict_fns, calibrations, pool_df, feature_cols, scalers, id_col, log_transform_cols)


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
    ):
        models       = model       if isinstance(model,       list) else [model]
        calibrations = calibration if isinstance(calibration, list) else [calibration]
        scalers      = scaler      if isinstance(scaler,      list) else [scaler]

        predict_fns = [
            lambda X, m=m: m.predict(xgb.DMatrix(X))
            for m in models
        ]
        super().__init__(predict_fns, calibrations, pool_df, feature_cols, scalers, id_col, log_transform_cols)
