"""Brand-CV hyperparameter selection for the LightGBM PL-objective ranker."""
import os as _os, sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import lightgbm as lgb

from config import Config
from data import build_kfold_arrays
from pl_objective import pl_grad_hess
from tune.cv_common import aggregate_candidate, eval_fold, grid_candidates, select_and_save
from tune.runtime import (
    apply_saved_semantic_config,
    apply_smoke_overrides,
    parse_tuner_args,
    smoke_candidates,
    smoke_early_stopping_rounds,
    smoke_num_boost_round,
)

FIXED = {
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3],
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "verbose": -1,
}

GRID = {
    "learning_rate": [0.03, 0.05],
    "num_leaves": [15, 31],
    "min_child_samples": [10, 30, 50],
    "reg_lambda": [0.1, 0.5],
}


def main():
    args = parse_tuner_args("Brand-CV hyperparameter selection for the LightGBM PL ranker.")
    cfg = Config()
    if args.smoke:
        apply_smoke_overrides(cfg)
    apply_saved_semantic_config(cfg, smoke=args.smoke)
    folds, _test_folds, _scalers, _pcas = build_kfold_arrays(cfg)

    candidates = [{**FIXED, **g} for g in grid_candidates(GRID)]
    if args.smoke:
        candidates = smoke_candidates(candidates, limit=3)
    print(
        f"LightGBM-PL brand-CV tuning {'[smoke] ' if args.smoke else ''}"
        f"x {len(candidates)} candidates x {cfg.n_folds} folds"
    )

    candidate_means = []
    for ci, cand in enumerate(candidates):
        fold_results = []
        for (X_tr, y_tr, _, g_tr), (X_val, y_val, ranks_val, g_val) in folds:
            train_data = lgb.Dataset(X_tr, label=y_tr, group=g_tr)
            val_data = lgb.Dataset(X_val, label=y_val, group=g_val, reference=train_data)

            def pl_obj_train(preds, train_dataset):
                labels = train_dataset.get_label()
                return pl_grad_hess(preds, labels, g_tr)

            model = lgb.train(
                {**cand, "objective": pl_obj_train},
                train_data,
                num_boost_round=smoke_num_boost_round(args.smoke),
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=smoke_early_stopping_rounds(args.smoke),
                        verbose=False,
                    )
                ],
            )
            val_scores = model.predict(X_val, raw_score=True).reshape(-1, 3)
            fold_results.append(eval_fold(val_scores, ranks_val, cfg.temp_candidates))
        means = aggregate_candidate(fold_results)
        candidate_means.append(means)
        print(
            f"  [{ci+1:2d}/{len(candidates)}] "
            f"lr={cand['learning_rate']:.2f} leaves={cand['num_leaves']:>2} "
            f"mcs={cand['min_child_samples']:>2} l2={cand['reg_lambda']:.1f} | "
            f"top1={means['top1_accuracy']:.4f} "
            f"k?={means['kendall_tau']:.4f} nll={means['nll']:.4f}"
        )

    select_and_save("lgbm_pl", candidates, candidate_means, cfg.tuning_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
