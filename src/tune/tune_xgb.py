"""Brand-CV hyperparameter selection for the vanilla XGBoost ranker.

Shallower trees + stronger regularization for the small effective sample, with
row/column subsampling retained so the seed ensemble in xgb_train.py is diverse.
Writes artifacts/tuning/xgb_best_params.json.

Run:  python src/tune/tune_xgb.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import xgboost as xgb

from config import Config
from data import build_kfold_arrays
from tune.cv_common import grid_candidates, eval_fold, aggregate_candidate, select_and_save


def _make_qid(groups: np.ndarray) -> np.ndarray:
    """[3,3,...] -> [0,0,0,1,1,1,...] per-item query IDs."""
    return np.repeat(np.arange(len(groups)), groups)


FIXED = {
    "objective":        "rank:ndcg",
    "eval_metric":      "ndcg@1",
    "eta":              0.05,
    "subsample":        0.8,   # row subsample    -> seed diversity
    "colsample_bytree": 0.7,   # column subsample -> seed diversity
    "reg_alpha":        0.0,
    "verbosity":        0,
}

# 24 candidates.
GRID = {
    "max_depth":        [2, 3, 4],
    "min_child_weight": [5, 10],
    "gamma":            [0.0, 0.5],
    "reg_lambda":       [1.0, 2.0],
}


def main():
    cfg = Config()
    folds, _test_folds, _scalers, _pcas = build_kfold_arrays(cfg)

    candidates = [{**FIXED, "seed": cfg.seed, **g} for g in grid_candidates(GRID)]
    print(f"XGBoost brand-CV tuning — {len(candidates)} candidates x {cfg.n_folds} folds")

    candidate_means = []
    for ci, cand in enumerate(candidates):
        fold_results = []
        for (X_tr, y_tr, _, g_tr), (X_val, y_val, ranks_val, g_val) in folds:
            dtrain = xgb.DMatrix(X_tr,  label=y_tr,  qid=_make_qid(g_tr))
            dval   = xgb.DMatrix(X_val, label=y_val, qid=_make_qid(g_val))
            model = xgb.train(
                cand, dtrain, num_boost_round=500,
                evals=[(dval, "val")], verbose_eval=False,
                callbacks=[xgb.callback.EarlyStopping(rounds=20)],
            )
            val_scores = model.predict(dval).reshape(-1, 3)
            fold_results.append(eval_fold(val_scores, ranks_val, cfg.temp_candidates))
        means = aggregate_candidate(fold_results)
        candidate_means.append(means)
        print(f"  [{ci+1:2d}/{len(candidates)}] "
              f"depth={cand['max_depth']} mcw={cand['min_child_weight']:>2} "
              f"gamma={cand['gamma']:.1f} l2={cand['reg_lambda']:.1f} | "
              f"top1={means['top1_accuracy']:.4f} kτ={means['kendall_tau']:.4f} nll={means['nll']:.4f}")

    select_and_save("xgb", candidates, candidate_means, cfg.tuning_dir)


if __name__ == "__main__":
    main()
