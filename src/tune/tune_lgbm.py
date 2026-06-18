"""Brand-CV hyperparameter selection for the vanilla LightGBM ranker.

Grid is centred on smaller, more-regularized trees (only ~259 unique training
items) and includes column/row subsampling so the downstream seed ensemble in
lgbm_train.py actually produces diverse models. Writes the winning params to
artifacts/tuning/lgbm_best_params.json.

Run:  python src/tune/tune_lgbm.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import lightgbm as lgb

from config import Config
from data import build_kfold_arrays
from tune.cv_common import grid_candidates, eval_fold, aggregate_candidate, select_and_save


FIXED = {
    "objective":        "lambdarank",
    "metric":           "ndcg",
    "ndcg_eval_at":     [1, 3],
    "learning_rate":    0.05,
    "reg_alpha":        0.1,
    "feature_fraction": 0.7,   # column subsample -> seed diversity
    "bagging_fraction": 0.8,   # row subsample    -> seed diversity
    "bagging_freq":     1,
    "verbose":          -1,
}

# Capacity / regularization axes (12 candidates).
GRID = {
    "num_leaves":        [7, 15, 31],
    "min_child_samples": [20, 50],
    "reg_lambda":        [0.5, 1.0],
}


def main():
    cfg = Config()
    folds, _test_folds, _scalers, _pcas = build_kfold_arrays(cfg)

    candidates = [{**FIXED, **g} for g in grid_candidates(GRID)]
    print(f"LightGBM brand-CV tuning — {len(candidates)} candidates x {cfg.n_folds} folds")

    candidate_means = []
    for ci, cand in enumerate(candidates):
        fold_results = []
        for (X_tr, y_tr, _, g_tr), (X_val, y_val, ranks_val, g_val) in folds:
            train_data = lgb.Dataset(X_tr,  label=y_tr,  group=g_tr)
            val_data   = lgb.Dataset(X_val, label=y_val, group=g_val, reference=train_data)
            model = lgb.train(
                cand, train_data, num_boost_round=500, valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
            )
            val_scores = model.predict(X_val, raw_score=True).reshape(-1, 3)
            fold_results.append(eval_fold(val_scores, ranks_val, cfg.temp_candidates))
        means = aggregate_candidate(fold_results)
        candidate_means.append(means)
        print(f"  [{ci+1:2d}/{len(candidates)}] "
              f"leaves={cand['num_leaves']:>2} mcs={cand['min_child_samples']:>2} "
              f"l2={cand['reg_lambda']:.1f} | top1={means['top1_accuracy']:.4f} "
              f"kτ={means['kendall_tau']:.4f} nll={means['nll']:.4f}")

    select_and_save("lgbm", candidates, candidate_means, cfg.tuning_dir)


if __name__ == "__main__":
    main()
