#!/usr/bin/env python3
"""
PL Score Regression Baseline

Pipeline:
  1. Fit Plackett-Luce scores on train trials (L-BFGS-B, L2 regularization)
  2. Build regression data: (train item features, pl_score)
  3. Preprocess: log1p -> median impute -> StandardScaler  (fit on train only)
  4. Train LinearRegression and Ridge; pick best on val RMSE
  5. Evaluate test trials with predicted scores (no test PL leakage)
  6. Pool-level inference: predicted_pl_score, pool_rank, percentile, pred_prob, odds_vs_mean
  7. Save artifacts to artifacts/pl_regression/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax as sp_softmax
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from config import Config
import metrics as M


# ---------------------------------------------------------------------------
# Plackett-Luce MLE via L-BFGS-B (L2-regularized)
# ---------------------------------------------------------------------------

def fit_plackett_luce(
    df: pd.DataFrame,
    item_col: str,
    rank_col: str,
    trial_keys: list,
    lambda_reg: float = 0.01,
) -> pd.DataFrame:
    """
    Fit Plackett-Luce log-scores for K=3 trials via negative log-likelihood
    minimization (L-BFGS-B) with L2 regularization on the scores.

    Args:
        df          : DataFrame with item_col, rank_col, trial_keys
        item_col    : column identifying items
        rank_col    : integer rank column (1=best)
        trial_keys  : columns identifying a unique trial
        lambda_reg  : L2 regularization on log-scores (prevents -inf for always-last items)

    Returns:
        DataFrame[item_col, pl_score, pl_prob, exposure_count, top1_count, avg_rank]
    """
    if "is_ambiguous" in df.columns:
        df = df[~df["is_ambiguous"].astype(bool)].copy()

    # Restrict to complete 3-item trials
    sz = df.groupby(trial_keys).size()
    valid = set(map(tuple, sz[sz == 3].reset_index()[trial_keys].values.tolist()))
    df = df[df[trial_keys].apply(tuple, axis=1).isin(valid)].copy()

    items = sorted(df[item_col].dropna().unique())
    item2idx = {it: i for i, it in enumerate(items)}
    n = len(items)

    # T[t, k] = index of item ranked (k+1)-th in trial t  - shape (N, 3)
    rows = []
    for _, g in df.groupby(trial_keys):
        g = g.sort_values(rank_col)
        rows.append([item2idx[u] for u in g[item_col]])
    T = np.array(rows, dtype=np.int32)

    def neg_ll_grad(s: np.ndarray):
        s_t = s[T]                                      # (N, 3)
        lse_all = logsumexp(s_t, axis=1)                # (N,)  log sum exp of all 3
        lse_23  = logsumexp(s_t[:, 1:], axis=1)         # (N,)  log sum exp of rank-2, rank-3

        # NLL = -[sum(s_rank1) + sum(s_rank2) - sum(lse_all) - sum(lse_23)]
        nll = -(s_t[:, 0].sum() + s_t[:, 1].sum() - lse_all.sum() - lse_23.sum())
        nll += lambda_reg * np.dot(s, s)

        sm_all = np.exp(s_t - lse_all[:, None])         # (N, 3) softmax over all 3 items
        sm_23  = np.exp(s_t[:, 1:] - lse_23[:, None])   # (N, 2) softmax over rank-2,3 items

        # Per-trial gradient of NLL w.r.t. each position's score:
        #   rank-1: sm_all[:,0] - 1
        #   rank-2: sm_all[:,1] + sm_23[:,0] - 1
        #   rank-3: sm_all[:,2] + sm_23[:,1]
        g_mat = sm_all.copy()
        g_mat[:, 0] -= 1.0
        g_mat[:, 1:] += sm_23
        g_mat[:, 1] -= 1.0

        grad = np.zeros(n)
        for k in range(3):
            grad += np.bincount(T[:, k], weights=g_mat[:, k], minlength=n)
        grad += 2.0 * lambda_reg * s

        return float(nll), grad

    res = minimize(
        neg_ll_grad, np.zeros(n), method="L-BFGS-B", jac=True,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    s_fit = res.x

    exp_cnt  = df.groupby(item_col).size()
    top1_cnt = df[df[rank_col] == 1].groupby(item_col).size()
    avg_rank = df.groupby(item_col)[rank_col].mean()

    out = pd.DataFrame({item_col: items, "pl_score": s_fit})
    out["pl_prob"]        = sp_softmax(s_fit)
    out["exposure_count"] = out[item_col].map(exp_cnt).fillna(0).astype(int)
    out["top1_count"]     = out[item_col].map(top1_cnt).fillna(0).astype(int)
    out["avg_rank"]       = out[item_col].map(avg_rank)
    return out


# ---------------------------------------------------------------------------
# Feature preprocessing helpers
# ---------------------------------------------------------------------------

def get_item_features(
    df: pd.DataFrame,
    feature_cols: list,
    log_cols: list,
    item_col: str = "resolved_url",
) -> pd.DataFrame:
    """Apply log1p, filter ambiguous rows, deduplicate by item (mean per item)."""
    if "is_ambiguous" in df.columns:
        df = df[~df["is_ambiguous"].astype(bool)].copy()
    df = df.dropna(subset=[item_col]).copy()

    for c in log_cols:
        if c in feature_cols:
            df[c] = np.log1p(df[c].clip(lower=0))

    return df.groupby(item_col)[feature_cols].mean().reset_index()


class FeaturePreprocessor:
    """Median imputation + StandardScaler; fit on train data only."""

    def __init__(self, feature_cols: list):
        self.feature_cols = feature_cols
        self.medians_: pd.Series = None
        self.scaler_: StandardScaler = None

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].copy()
        self.medians_ = X.median()
        X = X.fillna(self.medians_)
        self.scaler_ = StandardScaler()
        return self.scaler_.fit_transform(X)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].copy()
        X = X.fillna(self.medians_)
        return self.scaler_.transform(X)


# ---------------------------------------------------------------------------
# Trial-level score collection and temperature calibration
# ---------------------------------------------------------------------------

def collect_trial_scores(
    df: pd.DataFrame,
    score_lookup: dict,
    trial_keys: list,
    item_col: str = "resolved_url",
    rank_col: str = "ai_rank",
    fallback: float = 0.0,
) -> tuple:
    """
    For each complete, non-ambiguous trial, collect predicted scores and true ranks.

    Returns:
        scores     : (N_trials, 3) predicted scores ordered by sku_pos
        true_ranks : (N_trials, 3) ai_rank ordered by sku_pos
    """
    if "is_ambiguous" in df.columns:
        df = df[~df["is_ambiguous"].astype(bool)].copy()
    df = df.dropna(subset=[item_col]).copy()

    scores_list, ranks_list = [], []
    for _, grp in df.groupby(trial_keys):
        if len(grp) != 3:
            continue
        grp = grp.sort_values("sku_pos")
        pred_s = np.array([score_lookup.get(u, fallback) for u in grp[item_col]])
        true_r = grp[rank_col].values.astype(int)
        scores_list.append(pred_s)
        ranks_list.append(true_r)

    return np.array(scores_list), np.array(ranks_list)


def calibrate_temperature(
    scores: np.ndarray,
    true_ranks: np.ndarray,
    grid: list,
) -> float:
    """Grid search for temperature T minimizing PL-NLL on a held-out set."""
    best_T, best_nll = 1.0, float("inf")
    for T in grid:
        probs = sp_softmax(scores / T, axis=1)
        nll = M.nll_score(probs, true_ranks)
        if nll < best_nll:
            best_nll, best_T = nll, T
    return best_T


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    cfg = Config()

    out_dir = Path("artifacts/pl_regression")
    out_dir.mkdir(parents=True, exist_ok=True)

    ITEM_COL  = "resolved_url"
    RANK_COL  = "ai_rank"
    feat_cols = cfg.feature_cols
    log_cols  = cfg.log_transform_cols

    # ----------------------------------------------------------------
    # Load splits
    # ----------------------------------------------------------------
    def _load(split: str) -> pd.DataFrame:
        df = pd.read_csv(f"{cfg.data_dir}/{cfg.protocol}_{split}_features.csv")
        if cfg.engine_filter:
            df = df[df["engine"] == cfg.engine_filter]
        return df

    train_df = _load("train")
    val_df   = _load("val")
    test_df  = _load("test")

    print(f"Protocol: {cfg.protocol}  |  Engine: {cfg.engine_filter}")
    print(f"Rows  - train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # ----------------------------------------------------------------
    # Step 1: Fit Plackett-Luce on train trials only
    # ----------------------------------------------------------------
    print("\n[1] Fitting Plackett-Luce on train trials...")
    item_pl = fit_plackett_luce(
        train_df, item_col=ITEM_COL, rank_col=RANK_COL,
        trial_keys=cfg.trial_keys, lambda_reg=0.01,
    )
    print(f"    Items: {len(item_pl)}")
    print(f"    Score range: [{item_pl['pl_score'].min():.3f}, {item_pl['pl_score'].max():.3f}]")
    print(f"    Top-3 items by pl_score:")
    top3 = item_pl.nlargest(3, "pl_score")[[ITEM_COL, "pl_score", "top1_count", "avg_rank"]]
    print(top3.to_string(index=False))

    item_pl.to_csv(out_dir / "item_pl_scores.csv", index=False)
    print(f"    Saved -> {out_dir / 'item_pl_scores.csv'}")

    # ----------------------------------------------------------------
    # Step 2: Item-level features (log-transformed, deduped)
    # ----------------------------------------------------------------
    train_items = get_item_features(train_df, feat_cols, log_cols, ITEM_COL)
    val_items   = get_item_features(val_df,   feat_cols, log_cols, ITEM_COL)
    test_items  = get_item_features(test_df,  feat_cols, log_cols, ITEM_COL)

    print(f"\n[2] Unique items - train: {len(train_items)}, val: {len(val_items)}, test: {len(test_items)}")

    # ----------------------------------------------------------------
    # Step 3: Build regression training data
    # ----------------------------------------------------------------
    reg_train = train_items.merge(item_pl[[ITEM_COL, "pl_score"]], on=ITEM_COL, how="inner")
    print(f"\n[3] Regression train items (have PL scores): {len(reg_train)}")

    # Val items that also appeared in train get their true PL score for RMSE
    val_with_pl = val_items.merge(item_pl[[ITEM_COL, "pl_score"]], on=ITEM_COL, how="inner")
    print(f"    Val items with true PL score (for RMSE):  {len(val_with_pl)}")

    # ----------------------------------------------------------------
    # Step 4: Feature preprocessing - fit on regression train data
    # ----------------------------------------------------------------
    preprocessor = FeaturePreprocessor(feat_cols)
    X_train = preprocessor.fit_transform(reg_train)
    y_train = reg_train["pl_score"].values

    X_val_pl = preprocessor.transform(val_with_pl)
    y_val    = val_with_pl["pl_score"].values

    # ----------------------------------------------------------------
    # Step 5: Train and compare regression models
    # ----------------------------------------------------------------
    print("\n[4] Regression models:")
    model_specs = {
        "LinearRegression": Ridge(alpha=0.0, fit_intercept=True),
        "Ridge(alpha=1)":   Ridge(alpha=1.0, fit_intercept=True),
    }

    regression_results = {}
    best_model_name, best_val_rmse, best_model = None, float("inf"), None

    for name, mdl in model_specs.items():
        mdl.fit(X_train, y_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, mdl.predict(X_train)))
        val_rmse   = np.sqrt(mean_squared_error(y_val,   mdl.predict(X_val_pl)))
        print(f"    {name:<25} train_RMSE={train_rmse:.4f}  val_RMSE={val_rmse:.4f}")
        regression_results[name] = {"train_rmse": float(train_rmse), "val_rmse": float(val_rmse)}
        if val_rmse < best_val_rmse:
            best_val_rmse, best_model_name, best_model = val_rmse, name, mdl

    print(f"    -> Best: {best_model_name}")

    # ----------------------------------------------------------------
    # Step 6: Build score lookup for all unique items across splits
    # ----------------------------------------------------------------
    # Pool = union of all items; prefer train feature values where overlapping
    pool = pd.concat([train_items, val_items, test_items], ignore_index=True)
    pool = pool.drop_duplicates(subset=[ITEM_COL], keep="first").reset_index(drop=True)

    X_pool = preprocessor.transform(pool)
    pool["predicted_pl_score"] = best_model.predict(X_pool)

    score_lookup = dict(zip(pool[ITEM_COL], pool["predicted_pl_score"]))
    fallback     = float(pool["predicted_pl_score"].mean())

    print(f"\n[5] Pool size: {len(pool)} unique items")
    print(f"    Predicted score range: [{pool['predicted_pl_score'].min():.3f}, "
          f"{pool['predicted_pl_score'].max():.3f}]")

    # ----------------------------------------------------------------
    # Step 7: Collect trial-level scores for val and test
    # ----------------------------------------------------------------
    val_scores,  val_ranks  = collect_trial_scores(val_df,  score_lookup, cfg.trial_keys, fallback=fallback)
    test_scores, test_ranks = collect_trial_scores(test_df, score_lookup, cfg.trial_keys, fallback=fallback)

    print(f"    Trials - val: {len(val_scores)}, test: {len(test_scores)}")

    # ----------------------------------------------------------------
    # Step 8: Calibrate temperature on val ranking
    # ----------------------------------------------------------------
    best_T = calibrate_temperature(val_scores, val_ranks, cfg.temp_candidates)
    val_probs = sp_softmax(val_scores / best_T, axis=1)
    print(f"\n[6] Temperature calibrated on val: T={best_T}")

    # ----------------------------------------------------------------
    # Step 9: Val ranking metrics (with calibrated probs)
    # ----------------------------------------------------------------
    val_metrics = M.evaluate_all(val_scores, val_ranks, probs=val_probs)
    print("\nVal ranking metrics:")
    for k, v in val_metrics.items():
        print(f"    {k}: {v:.4f}")

    # ----------------------------------------------------------------
    # Step 10: Test ranking metrics - AI rankings used only for evaluation
    # ----------------------------------------------------------------
    test_probs  = sp_softmax(test_scores / best_T, axis=1)
    test_metrics = M.evaluate_all(test_scores, test_ranks, probs=test_probs)

    print("\nTest ranking metrics:")
    for k, v in test_metrics.items():
        print(f"    {k}: {v:.4f}")

    # ----------------------------------------------------------------
    # Step 11: Pool-level inference output
    # ----------------------------------------------------------------
    pred_scores = pool["predicted_pl_score"].values
    mean_score  = pred_scores.mean()

    pool["pool_rank"]    = pool["predicted_pl_score"].rank(ascending=False).astype(int)
    pool["percentile"]   = pool["predicted_pl_score"].rank(pct=True) * 100
    pool["pred_prob"]    = sp_softmax(pred_scores / best_T)
    pool["odds_vs_mean"] = np.exp(pred_scores - mean_score)

    # Attach true PL scores for train items (NaN for items without train exposure)
    pool = pool.merge(
        item_pl[[ITEM_COL, "pl_score", "pl_prob", "exposure_count", "top1_count", "avg_rank"]],
        on=ITEM_COL, how="left",
    )

    # Reorder columns for readability
    meta_cols = [ITEM_COL, "predicted_pl_score", "pool_rank", "percentile",
                 "pred_prob", "odds_vs_mean",
                 "pl_score", "pl_prob", "exposure_count", "top1_count", "avg_rank"]
    out_cols = meta_cols + feat_cols
    pool[out_cols].to_csv(out_dir / "regression_predictions.csv", index=False)
    print(f"\nSaved -> {out_dir / 'regression_predictions.csv'} ({len(pool)} items)")

    # ----------------------------------------------------------------
    # Step 12: Save metrics and model artifact
    # ----------------------------------------------------------------
    all_metrics = {
        "protocol": cfg.protocol,
        "engine":   cfg.engine_filter,
        "regression": regression_results,
        "best_model": best_model_name,
        "calibration": {"temperature": float(best_T)},
        "val_ranking":  {k: float(v) for k, v in val_metrics.items()},
        "test_ranking": {k: float(v) for k, v in test_metrics.items()},
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {out_dir / 'metrics.json'}")

    artifact = {
        "model":        best_model,
        "preprocessor": preprocessor,
        "best_T":       best_T,
        "feat_cols":    feat_cols,
        "log_cols":     log_cols,
        "item_col":     ITEM_COL,
    }
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved -> {out_dir / 'model.pkl'}")

    print("\n--- Done ---")
    return all_metrics


if __name__ == "__main__":
    main()
