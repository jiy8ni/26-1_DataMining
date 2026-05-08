import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from model import RecommendationScoreModel
from calibration import TemperatureCalibration


class PoolScorer:
    """
    Scores every item in the 300-product pool and computes the report metrics
    described in the model plan:

        score, pool_rank, percentile (top X%), rec_prob, odds_uplift

    Usage:
        scorer = PoolScorer(model, calib, pool_df, cfg.feature_cols, scaler)
        report = scorer.get_report()
    """

    def __init__(
        self,
        model: RecommendationScoreModel,
        calibration: TemperatureCalibration,
        pool_df: pd.DataFrame,        # one row per product; must contain feature_cols
        feature_cols: list,
        scaler: StandardScaler,
        id_col: str = "url",          # column used as item identifier in the report
        device: str = "cpu",
    ):
        self.model       = model.to(device).eval()
        self.calibration = calibration
        self.device      = device
        self.item_ids    = pool_df[id_col].tolist()

        X = pool_df[feature_cols].copy()
        X = X.fillna(X.median())
        X = scaler.transform(X)
        self._feats = torch.tensor(X, dtype=torch.float32).to(device)

    @torch.no_grad()
    def _raw_scores(self) -> torch.Tensor:
        return self.model(self._feats)                        # (N,)

    @torch.no_grad()
    def score_pool(self) -> np.ndarray:
        """Returns temperature-calibrated scores for every pool item."""
        return self.calibration.calibrate(self._raw_scores()).cpu().numpy()

    def get_report(self, reference_item_id: str = None) -> pd.DataFrame:
        """
        Returns a DataFrame sorted by pool_rank (ascending), with columns:

            item_id      : product identifier
            score        : calibrated recommendation score
            pool_rank    : rank within pool (1 = most likely to be recommended)
            top_pct      : "top X%" — fraction of pool this item outscores (lower = better)
            rec_prob     : P(this item chosen) via softmax over pool scores
            odds_uplift  : recommendation odds relative to pool average
        """
        scores = self.score_pool()
        N = len(scores)

        # pool_rank: 1 = highest score
        # argsort().argsort() gives 0-indexed position in ascending order;
        # N - that value converts to descending (rank 1 for highest).
        pool_rank = N - scores.argsort().argsort()           # 1-indexed, (N,)
        top_pct   = pool_rank / N * 100                      # "top X%" of pool

        # Softmax over the whole pool to get recommendation probabilities
        exp_s   = np.exp(scores - scores.max())              # numerically stable
        rec_prob = exp_s / exp_s.sum()

        # Odds uplift vs. pool average
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
