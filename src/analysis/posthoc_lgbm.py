"""Post-hoc interpretability for the headline OpenAI ranker (LightGBM lambdarank
+ semantic embeddings, the model behind artifacts/tuning/TUNING_RESULTS.md).

It reproduces the exact training path of src/lightgbm/lgbm_train.py (build_arrays
-> 5 seed boosters on the train split, val for early-stopping/calibration, test
for reporting) so the analysed model matches the reported top1=0.7286. On top of
that ensemble it computes:

  A. Global importance        -- gain/split + native TreeSHAP (mean|phi|), with
                                 cross-seed std as a stability band.
  B. Direction / shape        -- per-feature signed-SHAP and dependence data on
                                 the original (un-scaled, un-log1p) axis.
  C. Block contribution       -- structural vs text-PCA vs image-PCA vs position
                                 share of total |SHAP|.
  D. Position bias            -- the position feature's own SHAP contribution.
  E. Embedding characterisation (assist) -- correlation of each important PCA dim
                                 with structural features (qualitative reading is
                                 done in the report) -- handled lightly here.
  F. Interactions             -- SHAP interaction values (top feature pairs).
  G. pl_theta cross-check     -- item-level Spearman of structural features vs the
                                 model-independent Plackett-Luce strength.
  H. Model agreement          -- XGBoost (rank:ndcg) importance vs LightGBM (rank corr).
  I. Error analysis           -- margin of correct vs wrong top-1 trials.

Outputs: artifacts/analysis/{importance.json, pl_theta_corr.csv, *.png}.

Run:  python src/analysis/posthoc_lgbm.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lightgbm as lgb
import xgboost as xgb
import torch
from scipy.stats import spearmanr

from config import Config
from data import build_arrays, load_embeddings
from calibration import TemperatureCalibration
from metrics import evaluate_all
from preds_io import load_scores
from tune.runtime import apply_saved_semantic_config, load_tuned_params
from analysis.feature_names import build_feature_layout

# Per-engine config: (engine_filter, tuning_dir, preds_dir, pl_labels, out_dir).
# Lets the same analysis run on either AI engine without editing src/config.py.
ENGINE_CONFIG = {
    "openai": dict(
        engine_filter="openai",
        tuning_dir="artifacts/tuning",
        preds_dir="artifacts/preds",
        pl_labels_path="data/processed/pl_labels_step2_openai.csv",
        out_dir="artifacts/analysis",
    ),
    "anthropic": dict(
        engine_filter="anthropic",
        tuning_dir="artifacts/tuning_anthropic",
        preds_dir="artifacts/preds_anthropic",
        pl_labels_path="data/processed/pl_labels_step2_anthropic.csv",
        out_dir="artifacts/analysis_anthropic",
    ),
}

N_TOP = 15            # features to show in bar/dependence plots
N_DEPENDENCE = 6      # top structural features to draw dependence panels for
N_INTERACT_SAMPLE = 400


# --------------------------------------------------------------------------- #
# Training (reproduce the headline lgbm seed-ensemble)
# --------------------------------------------------------------------------- #
def train_lgbm_ensemble(cfg, params, arrays):
    (X_tr, y_tr, _, g_tr), (X_val, y_val, ranks_val, g_val), (X_te, y_te, ranks_te, g_te), scaler = arrays
    train_data = lgb.Dataset(X_tr, label=y_tr, group=g_tr)
    val_data = lgb.Dataset(X_val, label=y_val, group=g_val, reference=train_data)

    boosters, val_s_list, test_s_list, temps = [], [], [], []
    for i in range(cfg.n_seeds):
        seed_params = {**params, "seed": cfg.seed + i,
                       "bagging_seed": cfg.seed + i, "feature_fraction_seed": cfg.seed + i}
        model = lgb.train(seed_params, train_data, num_boost_round=500,
                          valid_sets=[val_data],
                          callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
        val_s = model.predict(X_val, raw_score=True).reshape(-1, 3)
        test_s = model.predict(X_te, raw_score=True).reshape(-1, 3)
        calib = TemperatureCalibration(cfg.temp_candidates)
        calib.fit(torch.tensor(val_s, dtype=torch.float32),
                  torch.tensor(ranks_val, dtype=torch.long))
        boosters.append(model)
        val_s_list.append(val_s)
        test_s_list.append(test_s)
        temps.append(calib.temperature)

    test_scores = np.mean(test_s_list, axis=0)
    avg_temp = float(np.mean(temps))
    exp_cal = np.exp(test_scores / avg_temp)
    test_probs = exp_cal / exp_cal.sum(axis=1, keepdims=True)
    test_metrics = evaluate_all(test_scores, ranks_te, test_probs)
    return boosters, test_metrics, avg_temp


# --------------------------------------------------------------------------- #
# Importances
# --------------------------------------------------------------------------- #
def lgbm_native_importance(boosters):
    """gain / split importance, averaged across seeds (mean, std)."""
    gains = np.array([b.feature_importance(importance_type="gain") for b in boosters], dtype=float)
    splits = np.array([b.feature_importance(importance_type="split") for b in boosters], dtype=float)
    # normalise each seed to sum=1 so the average is scale-free
    gains = gains / gains.sum(axis=1, keepdims=True)
    splits = splits / splits.sum(axis=1, keepdims=True)
    return (gains.mean(0), gains.std(0)), (splits.mean(0), splits.std(0))


def lgbm_treeshap(boosters, X):
    """Native TreeSHAP per seed via predict(pred_contrib=True).

    Returns:
        shap_mean : (N, F)  per-row SHAP averaged across seeds (drops bias col)
        per_seed_absmean : (n_seeds, F)  mean|phi| per seed (for stability std)
    """
    F = X.shape[1]
    contribs = []
    for b in boosters:
        c = b.predict(X, pred_contrib=True)        # (N, F+1), last col = bias
        contribs.append(np.asarray(c)[:, :F])
    contribs = np.stack(contribs, axis=0)          # (S, N, F)
    shap_mean = contribs.mean(0)                   # (N, F)
    per_seed_absmean = np.abs(contribs).mean(axis=1)   # (S, F)
    return shap_mean, per_seed_absmean


def xgb_ensemble_importance(cfg, params, arrays, F):
    """Train an XGBoost rank:ndcg seed-ensemble on the same arrays and return
    mean|SHAP| per feature (for cross-model agreement)."""
    (X_tr, y_tr, _, g_tr), (X_val, y_val, ranks_val, g_val), (X_te, *_), _ = arrays
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtrain.set_group(g_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dval.set_group(g_val)
    dtest = xgb.DMatrix(X_te)

    absmeans = []
    for i in range(cfg.n_seeds):
        p = {**params, "seed": cfg.seed + i}
        bst = xgb.train(p, dtrain, num_boost_round=500,
                        evals=[(dval, "val")], early_stopping_rounds=20, verbose_eval=False)
        contrib = bst.predict(dtest, pred_contribs=True)   # (N, F+1)
        absmeans.append(np.abs(np.asarray(contrib)[:, :F]).mean(0))
    return np.mean(absmeans, axis=0)


# --------------------------------------------------------------------------- #
# Original-scale recovery for dependence plots
# --------------------------------------------------------------------------- #
def to_original_scale(X, scaler, layout, cfg):
    """Invert StandardScaler (+ expm1 for log1p cols) for the structural block so
    dependence-plot axes read in real units. Position col is left as-is."""
    n_scaled = scaler.n_features_in_                 # structural + pca (excludes position)
    inv = scaler.inverse_transform(X[:, :n_scaled])
    log_cols = set(getattr(cfg, "log_transform_cols", []) or [])
    names = layout.names
    for j in range(layout.structural[0], layout.structural[1]):
        if names[j] in log_cols:
            inv[:, j] = np.expm1(inv[:, j])
    return inv


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_global_importance(names, shap_absmean, shap_std, gain_mean, path):
    order = np.argsort(shap_absmean)[::-1][:N_TOP][::-1]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(y, shap_absmean[order], xerr=shap_std[order],
            color="#4C72B0", ecolor="#999", capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=9)
    ax.set_xlabel("mean |SHAP|  (contribution to ranking score)")
    ax.set_title(f"Top {N_TOP} feature importance (TreeSHAP, 5-seed mean +/- std)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_block_contribution(block_share, path):
    labels = list(block_share.keys())
    vals = [block_share[k] * 100 for k in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, vals, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    ax.set_ylabel("share of total |SHAP|  (%)")
    ax.set_title("Feature-block contribution to ranking score")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_dependence(names, shap_vals, X_orig, top_idx, path):
    n = len(top_idx)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, j in zip(axes, top_idx):
        x = X_orig[:, j]
        ax.scatter(x, shap_vals[:, j], s=6, alpha=0.3, color="#4C72B0")
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_title(names[j], fontsize=9)
        ax.set_xlabel("feature value (original scale)", fontsize=8)
        ax.set_ylabel("SHAP", fontsize=8)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Direction & shape of effect (SHAP dependence)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Analyses E / G / I
# --------------------------------------------------------------------------- #
def pl_theta_correlation(cfg, layout):
    """Item-level Spearman of each structural feature vs Plackett-Luce strength."""
    pl = pd.read_csv(cfg.pl_labels_path)[["resolved_url", "pl_theta"]]
    feat = pd.read_csv(f"{cfg.data_dir}/{cfg.protocol}_train_features.csv")
    if cfg.engine_filter is not None:
        feat = feat[feat["engine"] == cfg.engine_filter]
    struct = list(cfg.feature_cols)
    items = (feat.groupby("resolved_url")[struct].mean().reset_index()
             .merge(pl, on="resolved_url", how="inner"))
    rows = []
    for c in struct:
        rho, p = spearmanr(items[c], items["pl_theta"], nan_policy="omit")
        rows.append({"feature": c, "spearman_rho": float(rho), "p_value": float(p)})
    return pd.DataFrame(rows).sort_values("spearman_rho", key=np.abs, ascending=False), len(items)


def embedding_pc_characterisation(cfg, layout, shap_absmean, X_test, X_orig):
    """For the most important text/image PCA dim, correlate its per-item PC score
    with the original-scale structural features so the report can name what the
    opaque axis roughly captures (e.g. 'long-text / ingredient-heavy axis')."""
    s_lo, s_hi = layout.structural
    out = {}
    for block in ("text_pca", "image_pca"):
        lo, hi = getattr(layout, block)
        if hi <= lo:
            continue
        sub = shap_absmean[lo:hi]
        top_local = int(np.argmax(sub))
        top_idx = lo + top_local
        pc = X_test[:, top_idx]
        corrs = []
        for j in range(s_lo, s_hi):
            rho, _ = spearmanr(pc, X_orig[:, j], nan_policy="omit")
            corrs.append((layout.names[j], float(rho)))
        corrs.sort(key=lambda t: abs(t[1]), reverse=True)
        out[block] = {
            "top_dim": layout.names[top_idx],
            "abs_shap": float(sub[top_local]),
            "block_total_abs_shap": float(sub.sum()),
            "structural_correlates": [{"feature": f, "spearman_rho": r} for f, r in corrs[:6]],
        }
    return out


def error_analysis(cfg):
    """Margin of correct vs wrong top-1 picks from the saved headline predictions."""
    try:
        scores, ranks = load_scores(cfg.preds_dir, "lgbm", "test")
    except FileNotFoundError:
        return None
    pred_top1 = scores.argmax(axis=1)
    true_top1 = ranks.argmin(axis=1)
    correct = pred_top1 == true_top1
    srt = np.sort(scores, axis=1)
    margin = srt[:, -1] - srt[:, -2]      # top1 vs runner-up score gap
    return {
        "n_trials": int(len(scores)),
        "top1_accuracy": float(correct.mean()),
        "mean_margin_correct": float(margin[correct].mean()),
        "mean_margin_wrong": float(margin[~correct].mean()),
        "median_margin_correct": float(np.median(margin[correct])),
        "median_margin_wrong": float(np.median(margin[~correct])),
        "margins": margin.tolist(),
        "correct": correct.tolist(),
    }


def plot_error_margins(err, path):
    margin = np.array(err["margins"])
    correct = np.array(err["correct"])
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, np.percentile(margin, 99), 30)
    ax.hist(margin[correct], bins=bins, alpha=0.6, label="correct top-1", color="#55A868", density=True)
    ax.hist(margin[~correct], bins=bins, alpha=0.6, label="wrong top-1", color="#C44E52", density=True)
    ax.set_xlabel("top1 - runner-up raw score margin")
    ax.set_ylabel("density")
    ax.set_title("Decision margin: correct vs wrong trials")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main(engine: str = "openai"):
    ec = ENGINE_CONFIG[engine]
    out_dir = ec["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    cfg = Config()
    cfg.engine_filter = ec["engine_filter"]
    cfg.tuning_dir = ec["tuning_dir"]
    cfg.preds_dir = ec["preds_dir"]
    cfg.pl_labels_path = ec["pl_labels_path"]
    print(f"=== Post-hoc analysis: engine={engine} (tuning_dir={cfg.tuning_dir}) ===")
    apply_saved_semantic_config(cfg)
    lgbm_params = load_tuned_params(cfg, "lgbm_best_params.json", {}, "LightGBM")
    layout = build_feature_layout(cfg)
    names = layout.names
    F = len(names)

    print("Building arrays (build_arrays, OpenAI step2 + semantic)...")
    arrays = build_arrays(cfg)
    scaler = arrays[3]
    assert arrays[0][0].shape[1] == F, f"feature width {arrays[0][0].shape[1]} != names {F}"

    print(f"Training LightGBM {cfg.n_seeds}-seed ensemble (reproducing headline)...")
    boosters, test_metrics, avg_temp = train_lgbm_ensemble(cfg, lgbm_params, arrays)
    print("  Reproduced test metrics:")
    for k, v in test_metrics.items():
        print(f"    {k:<18} {v:.4f}")

    # ---- Importances (A) ----
    (gain_mean, gain_std), (split_mean, split_std) = lgbm_native_importance(boosters)
    X_test = arrays[2][0]
    shap_vals, per_seed_absmean = lgbm_treeshap(boosters, X_test)
    shap_absmean = np.abs(shap_vals).mean(0)               # (F,)
    shap_seed_std = per_seed_absmean.std(0)                # stability band
    signed_mean = shap_vals.mean(0)                        # direction proxy

    # ---- Block contribution (C) + position (D) ----
    total_abs = shap_absmean.sum()
    block_share = {}
    for block in ("structural", "text_pca", "image_pca", "position"):
        lo, hi = getattr(layout, block)
        block_share[block] = float(shap_absmean[lo:hi].sum() / total_abs) if hi > lo else 0.0
    pos_idx = layout.position[0] if layout.position[1] > layout.position[0] else None
    position_report = None
    if pos_idx is not None:
        position_report = {
            "abs_shap": float(shap_absmean[pos_idx]),
            "share": float(shap_absmean[pos_idx] / total_abs),
            "rank_among_all": int((shap_absmean > shap_absmean[pos_idx]).sum() + 1),
        }

    # ---- Dependence (B): top structural features ----
    struct_lo, struct_hi = layout.structural
    struct_order = struct_lo + np.argsort(shap_absmean[struct_lo:struct_hi])[::-1]
    top_struct = struct_order[:N_DEPENDENCE]
    X_orig = to_original_scale(X_test, scaler, layout, cfg)

    # ---- Interactions (F) ----
    interactions = None
    try:
        import shap
        expl = shap.TreeExplainer(boosters[0])
        sample = X_test[np.random.RandomState(0).choice(len(X_test),
                        size=min(N_INTERACT_SAMPLE, len(X_test)), replace=False)]
        inter = expl.shap_interaction_values(sample)       # (n, F, F)
        inter = np.asarray(inter)
        absinter = np.abs(inter).mean(0)
        np.fill_diagonal(absinter, 0.0)
        pairs = []
        triu = np.triu_indices(F, k=1)
        flat = absinter[triu]
        for rank_i in np.argsort(flat)[::-1][:8]:
            a, b = triu[0][rank_i], triu[1][rank_i]
            pairs.append({"a": names[a], "b": names[b], "abs_interaction": float(absinter[a, b])})
        interactions = pairs
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] interaction values skipped: {e}")

    # ---- pl_theta cross-check (G) ----
    pl_corr_df, n_items = pl_theta_correlation(cfg, layout)
    pl_corr_df.to_csv(os.path.join(out_dir, "pl_theta_corr.csv"), index=False)

    # ---- Embedding characterisation (E) ----
    emb_report = embedding_pc_characterisation(cfg, layout, shap_absmean, X_test, X_orig)

    # ---- Model agreement (H) ----
    xgb_params = load_tuned_params(cfg, "xgb_best_params.json", {}, "XGBoost")
    print("Training XGBoost ensemble for agreement check...")
    try:
        xgb_absmean = xgb_ensemble_importance(cfg, xgb_params, arrays, F)
        rho_agree, _ = spearmanr(shap_absmean, xgb_absmean)
        # structural-only agreement (drops opaque PCA dims)
        rho_struct, _ = spearmanr(shap_absmean[struct_lo:struct_hi],
                                  xgb_absmean[struct_lo:struct_hi])
        agreement = {"spearman_all": float(rho_agree),
                     "spearman_structural": float(rho_struct)}
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] xgb agreement skipped: {e}")
        xgb_absmean, agreement = None, None

    # ---- Error analysis (I) ----
    err = error_analysis(cfg)

    # ---- Plots ----
    plot_global_importance(names, shap_absmean, shap_seed_std, gain_mean,
                           os.path.join(out_dir, "global_importance.png"))
    plot_block_contribution(block_share, os.path.join(out_dir, "block_contribution.png"))
    plot_dependence(names, shap_vals, X_orig, list(top_struct),
                    os.path.join(out_dir, "dependence_top_structural.png"))
    if err is not None:
        plot_error_margins(err, os.path.join(out_dir, "error_margins.png"))
    try:
        import shap
        plt.figure()
        shap.summary_plot(shap_vals, X_test, feature_names=names, show=False, max_display=N_TOP)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "shap_summary.png"), dpi=130, bbox_inches="tight")
        plt.close()
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] shap summary plot skipped: {e}")

    # ---- Dump JSON ----
    def feat_table(values, std=None):
        out = []
        for i in np.argsort(values)[::-1]:
            row = {"feature": names[i], "value": float(values[i]), "block": layout.block_of(i)}
            if std is not None:
                row["std"] = float(std[i])
            out.append(row)
        return out

    report = {
        "model": "lgbm_lambdarank_openai_semantic",
        "n_seeds": cfg.n_seeds,
        "n_features": F,
        "reproduced_test_metrics": test_metrics,
        "avg_temperature": avg_temp,
        "global_importance_shap_absmean": feat_table(shap_absmean, shap_seed_std),
        "global_importance_gain": feat_table(gain_mean, gain_std),
        "global_importance_split": feat_table(split_mean, split_std),
        "signed_shap_mean": {names[i]: float(signed_mean[i]) for i in range(F)},
        "block_contribution": block_share,
        "position_bias": position_report,
        "top_structural_for_dependence": [names[i] for i in top_struct],
        "interactions_top": interactions,
        "embedding_blocks": emb_report,
        "pl_theta_n_items": int(n_items),
        "pl_theta_top_corr": pl_corr_df.head(12).to_dict(orient="records"),
        "model_agreement_lgbm_vs_xgb": agreement,
        "error_analysis": {k: v for k, v in (err or {}).items()
                           if k not in ("margins", "correct")} if err else None,
    }
    with open(os.path.join(out_dir, "importance.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nWrote artifacts to {out_dir}/")
    print("  global_importance.png, block_contribution.png, dependence_top_structural.png,")
    print("  shap_summary.png, error_margins.png, importance.json, pl_theta_corr.csv")
    print("\nTop-8 features by mean|SHAP|:")
    for r in report["global_importance_shap_absmean"][:8]:
        sign = signed_mean[names.index(r["feature"])]
        arrow = "up" if sign > 0 else "down"
        print(f"  {r['feature']:<26} {r['value']:.4f}  (mean dir: {arrow})  [{r['block']}]")
    print(f"\nBlock share: " + "  ".join(f"{k}={v*100:.1f}%" for k, v in block_share.items()))
    if agreement:
        print(f"lgbm-vs-xgb importance Spearman (structural): {agreement['spearman_structural']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Post-hoc interpretability for the headline lgbm ranker.")
    ap.add_argument("--engine", choices=list(ENGINE_CONFIG), default="openai",
                    help="which AI engine's tuned model to analyse")
    args = ap.parse_args()
    main(args.engine)
