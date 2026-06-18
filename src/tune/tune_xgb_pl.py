"""Brand-CV hyperparameter selection for the XGBoost PL-objective ranker."""
import os as _os, sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import xgboost as xgb

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


def _make_qid(groups: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(groups)), groups)


FIXED = {
    "eval_metric": "ndcg@1",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "verbosity": 0,
}

GRID = {
    "eta": [0.03, 0.05],
    "max_depth": [3, 4, 5],
    "min_child_weight": [3, 5],
    "gamma": [0.0, 0.5],
    "reg_lambda": [0.5, 1.0],
}


def main():
    args = parse_tuner_args("Brand-CV hyperparameter selection for the XGBoost PL ranker.")
    cfg = Config()
    if args.smoke:
        apply_smoke_overrides(cfg)
    apply_saved_semantic_config(cfg, smoke=args.smoke)
    folds, _test_folds, _scalers, _pcas = build_kfold_arrays(cfg)

    candidates = [{**FIXED, "seed": cfg.seed, **g} for g in grid_candidates(GRID)]
    if args.smoke:
        candidates = smoke_candidates(candidates, limit=3)
    print(
        f"XGBoost-PL brand-CV tuning {'[smoke] ' if args.smoke else ''}"
        f"x {len(candidates)} candidates x {cfg.n_folds} folds"
    )

    candidate_means = []
    for ci, cand in enumerate(candidates):
        fold_results = []
        for (X_tr, y_tr, _, g_tr), (X_val, y_val, ranks_val, g_val) in folds:
            dtrain = xgb.DMatrix(X_tr, label=y_tr, qid=_make_qid(g_tr))
            dval = xgb.DMatrix(X_val, label=y_val, qid=_make_qid(g_val))

            def pl_obj_train(preds: np.ndarray, train_matrix: xgb.DMatrix):
                labels = train_matrix.get_label()
                return pl_grad_hess(preds, labels, g_tr)

            model = xgb.train(
                cand,
                dtrain,
                obj=pl_obj_train,
                num_boost_round=smoke_num_boost_round(args.smoke),
                evals=[(dval, "val")],
                verbose_eval=False,
                callbacks=[xgb.callback.EarlyStopping(rounds=smoke_early_stopping_rounds(args.smoke))],
            )
            val_scores = model.predict(dval).reshape(-1, 3)
            fold_results.append(eval_fold(val_scores, ranks_val, cfg.temp_candidates))
        means = aggregate_candidate(fold_results)
        candidate_means.append(means)
        print(
            f"  [{ci+1:2d}/{len(candidates)}] "
            f"eta={cand['eta']:.2f} depth={cand['max_depth']} "
            f"mcw={cand['min_child_weight']:>2} gamma={cand['gamma']:.1f} "
            f"l2={cand['reg_lambda']:.1f} | top1={means['top1_accuracy']:.4f} "
            f"k?={means['kendall_tau']:.4f} nll={means['nll']:.4f}"
        )

    select_and_save("xgb_pl", candidates, candidate_means, cfg.tuning_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
