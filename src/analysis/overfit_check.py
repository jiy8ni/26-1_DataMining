"""Overfitting diagnostics for the headline OpenAI LightGBM ranker.

Distinguishes 'the model relies heavily on the embedding block' (a contribution-
share fact) from 'the model overfit to embeddings' (a generalization-gap fact).

Two direct checks:
  1. Train vs unseen-brand-Test generalization gap (same fitted ensemble).
  2. Embedding ablation: structural-only vs +semantic, both scored on the SAME
     unseen-brand test. If semantics lifts held-out test, it is signal not memorisation.

Run:  python src/analysis/overfit_check.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data import build_arrays
from metrics import evaluate_all
from tune.runtime import apply_saved_semantic_config, load_tuned_params


def train_eval(cfg, params):
    """Train the same 5-seed lgbm ensemble; return (train_metrics, test_metrics)."""
    (X_tr, y_tr, ranks_tr, g_tr), (X_val, y_val, ranks_val, g_val), \
        (X_te, y_te, ranks_te, g_te), _ = build_arrays(cfg)
    train_data = lgb.Dataset(X_tr, label=y_tr, group=g_tr)
    val_data = lgb.Dataset(X_val, label=y_val, group=g_val, reference=train_data)

    tr_list, te_list = [], []
    for i in range(cfg.n_seeds):
        sp = {**params, "seed": cfg.seed + i,
              "bagging_seed": cfg.seed + i, "feature_fraction_seed": cfg.seed + i}
        m = lgb.train(sp, train_data, num_boost_round=500, valid_sets=[val_data],
                      callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
        tr_list.append(m.predict(X_tr, raw_score=True).reshape(-1, 3))
        te_list.append(m.predict(X_te, raw_score=True).reshape(-1, 3))
    tr_s = np.mean(tr_list, axis=0)
    te_s = np.mean(te_list, axis=0)
    return (evaluate_all(tr_s, ranks_tr), evaluate_all(te_s, ranks_te),
            len(g_tr), len(g_te))


def main():
    cfg = Config()
    apply_saved_semantic_config(cfg)
    params = load_tuned_params(cfg, "lgbm_best_params.json", {}, "LightGBM")

    print("=== [1] Generalization gap (semantic model) ===")
    tr_m, te_m, n_tr, n_te = train_eval(cfg, params)
    print(f"  train trials: {n_tr}   test(unseen-brand) trials: {n_te}")
    print(f"  {'metric':<18}{'train':>9}{'test':>9}{'gap':>9}")
    for k in ("top1_accuracy", "pairwise_accuracy", "ndcg@3", "kendall_tau"):
        print(f"  {k:<18}{tr_m[k]:>9.4f}{te_m[k]:>9.4f}{tr_m[k]-te_m[k]:>9.4f}")

    print("\n=== [2] Embedding ablation on the SAME unseen-brand test ===")
    cfg_struct = Config()
    cfg_struct.use_semantic_features = False
    # structural-only baseline uses the same tuned tree params
    _, te_struct, _, _ = train_eval(cfg_struct, params)
    print(f"  {'metric':<18}{'structural':>11}{'+semantic':>11}{'delta':>9}")
    for k in ("top1_accuracy", "pairwise_accuracy", "ndcg@3", "kendall_tau"):
        d = te_m[k] - te_struct[k]
        print(f"  {k:<18}{te_struct[k]:>11.4f}{te_m[k]:>11.4f}{d:>+9.4f}")

    print("\nInterpretation:")
    print("  - Small train-test gap  => not overfit (relies on embeddings AND generalizes).")
    print("  - Positive +semantic delta on UNSEEN brands => embeddings are signal, not memorisation.")


if __name__ == "__main__":
    main()
