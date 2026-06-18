"""
A/B verification: does adding PCA-reduced text+image semantic features improve
the LightGBM ranker? Runs the exact build_kfold_arrays pipeline twice — once
with use_semantic_features=False (structural baseline) and once True — using the
same LightGBM params as lgbm_kfold.py, and prints CV-mean val metrics + ensemble
test metrics side by side. No wandb.

    python src/verify_semantic.py
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb

from config import Config
from data import build_kfold_arrays, effective_feature_dim
from metrics import evaluate_all

PARAMS = {
    "objective":         "lambdarank",
    "metric":            "ndcg",
    "ndcg_eval_at":      [1, 3],
    "learning_rate":     0.05,
    "num_leaves":        31,
    "min_child_samples": 5,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "verbose":           -1,
}


def run(use_semantic: bool) -> dict:
    cfg = Config()
    cfg.use_semantic_features = use_semantic
    n_feat = effective_feature_dim(cfg)
    print(f"\n{'='*60}\nsemantic={use_semantic}  n_features={n_feat}\n{'='*60}")

    folds, test_folds, scalers, pcas = build_kfold_arrays(cfg)

    fold_models, fold_val = [], []
    for fi, ((X_tr, y_tr, _, g_tr), (X_val, y_val, ranks_val, g_val)) in enumerate(folds):
        train_data = lgb.Dataset(X_tr, label=y_tr, group=g_tr)
        val_data   = lgb.Dataset(X_val, label=y_val, group=g_val, reference=train_data)
        model = lgb.train(
            PARAMS, train_data, num_boost_round=500, valid_sets=[val_data],
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        val_scores = model.predict(X_val, raw_score=True).reshape(-1, 3)
        fold_val.append(evaluate_all(val_scores, ranks_val))
        fold_models.append(model)

    cv = {m: float(np.nanmean([r[m] for r in fold_val])) for m in fold_val[0]}

    # ensemble test
    test_scores, ranks_ref = [], None
    for model, (X_te, _, ranks_te, _) in zip(fold_models, test_folds):
        test_scores.append(model.predict(X_te, raw_score=True).reshape(-1, 3))
        ranks_ref = ranks_te
    ens = np.mean(test_scores, axis=0)
    test = evaluate_all(ens, ranks_ref)
    return {"n_features": n_feat, "cv_val": cv, "test": test}


def main():
    base = run(use_semantic=False)
    sem  = run(use_semantic=True)

    keys = ["top1_accuracy", "pairwise_accuracy", "ndcg@3", "kendall_tau"]
    print(f"\n{'='*60}\nRESULT  (baseline -> +semantic)   [features {base['n_features']} -> {sem['n_features']}]\n{'='*60}")
    for split in ("cv_val", "test"):
        print(f"\n[{split}]")
        for k in keys:
            b, s = base[split][k], sem[split][k]
            print(f"  {k:<20} {b:.4f} -> {s:.4f}  ({'+' if s>=b else ''}{s-b:+.4f})")


if __name__ == "__main__":
    main()
