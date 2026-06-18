"""Brand-CV selection for shared semantic feature settings."""
import copy
import os as _os, sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import lightgbm as lgb

from config import Config
from data import build_kfold_arrays
from tune.cv_common import aggregate_candidate, eval_fold, select_and_save_semantic
from tune.runtime import (
    apply_smoke_overrides,
    parse_tuner_args,
    smoke_candidates,
    smoke_early_stopping_rounds,
    smoke_num_boost_round,
)

PROXY_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 0.5,
    "verbose": -1,
}

SEMANTIC_CANDIDATES = [
    {"use_semantic_features": False, "text_pca_dim": 0, "image_pca_dim": 0},
    {"use_semantic_features": True, "text_pca_dim": 8, "image_pca_dim": 0},
    {"use_semantic_features": True, "text_pca_dim": 16, "image_pca_dim": 0},
    {"use_semantic_features": True, "text_pca_dim": 0, "image_pca_dim": 4},
    {"use_semantic_features": True, "text_pca_dim": 0, "image_pca_dim": 8},
    {"use_semantic_features": True, "text_pca_dim": 8, "image_pca_dim": 4},
    {"use_semantic_features": True, "text_pca_dim": 16, "image_pca_dim": 8},
    {"use_semantic_features": True, "text_pca_dim": 24, "image_pca_dim": 8},
]


def _candidate_cfg(base_cfg: Config, semantic_cfg: dict) -> Config:
    cfg = copy.deepcopy(base_cfg)
    cfg.use_semantic_features = bool(semantic_cfg["use_semantic_features"])
    cfg.text_pca_dim = int(semantic_cfg["text_pca_dim"])
    cfg.image_pca_dim = int(semantic_cfg["image_pca_dim"])
    return cfg


def main():
    args = parse_tuner_args("Brand-CV selection for shared semantic feature settings.")
    cfg = Config()
    if args.smoke:
        apply_smoke_overrides(cfg)

    candidates = list(SEMANTIC_CANDIDATES)
    if args.smoke:
        candidates = smoke_candidates(candidates, limit=3)
    print(
        f"Semantic brand-CV tuning {'[smoke] ' if args.smoke else ''}"
        f"x {len(candidates)} candidates x {cfg.n_folds} folds"
    )

    candidate_means = []
    for ci, semantic_cfg in enumerate(candidates):
        local_cfg = _candidate_cfg(cfg, semantic_cfg)
        folds, _test_folds, _scalers, _pcas = build_kfold_arrays(local_cfg)

        fold_results = []
        for (X_tr, y_tr, _, g_tr), (X_val, y_val, ranks_val, g_val) in folds:
            train_data = lgb.Dataset(X_tr, label=y_tr, group=g_tr)
            val_data = lgb.Dataset(X_val, label=y_val, group=g_val, reference=train_data)
            model = lgb.train(
                PROXY_PARAMS,
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
            fold_results.append(eval_fold(val_scores, ranks_val, local_cfg.temp_candidates))

        means = aggregate_candidate(fold_results)
        candidate_means.append(means)
        print(
            f"  [{ci+1:2d}/{len(candidates)}] "
            f"use_sem={semantic_cfg['use_semantic_features']} "
            f"txt={semantic_cfg['text_pca_dim']:>2} img={semantic_cfg['image_pca_dim']:>2} | "
            f"top1={means['top1_accuracy']:.4f} "
            f"k?={means['kendall_tau']:.4f} nll={means['nll']:.4f}"
        )

    select_and_save_semantic(candidates, candidate_means, cfg.tuning_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
