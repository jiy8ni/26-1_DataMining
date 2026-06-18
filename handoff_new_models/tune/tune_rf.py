"""Brand-CV hyperparameter selection for the pairwise Random Forest ranker.

PDF model 2: RandomForestClassifier on dX = X_i - X_j. P(i beats j) via
predict_proba, summed per trial. Bagging (vs the existing LightGBM boosting).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier

from config import Config
from data import build_kfold_arrays
from paths import configure_paths
from harness import run_pairwise_cv
from tune.cv_common import grid_candidates
from tune.runtime import (
    apply_saved_semantic_config,
    apply_smoke_overrides,
    parse_tuner_args,
    smoke_candidates,
)

FIXED = {
    "class_weight": "balanced",
    "n_jobs": -1,
}

GRID = {
    "n_estimators": [300, 500],
    "max_depth": [8, None],
    "min_samples_leaf": [5, 20],
    "max_features": ["sqrt", 0.5],
}


def make_model(cand, seed):
    return RandomForestClassifier(
        n_estimators=cand["n_estimators"],
        max_depth=cand["max_depth"],
        min_samples_leaf=cand["min_samples_leaf"],
        max_features=cand["max_features"],
        class_weight=cand["class_weight"],
        n_jobs=cand["n_jobs"],
        random_state=seed,
    )


def prob_fn(model):
    return lambda d: model.predict_proba(d)[:, 1]


def main():
    args = parse_tuner_args("Brand-CV selection for the pairwise Random Forest ranker.")
    cfg = configure_paths(Config())
    if args.smoke:
        apply_smoke_overrides(cfg)
    apply_saved_semantic_config(cfg, smoke=args.smoke)
    folds, _tf, _sc, _pca = build_kfold_arrays(cfg)

    candidates = [{**FIXED, **g} for g in grid_candidates(GRID)]
    if args.smoke:
        candidates = smoke_candidates(candidates, limit=3)
    print(f"Random Forest brand-CV {'[smoke] ' if args.smoke else ''}"
          f"x {len(candidates)} candidates x {cfg.n_folds} folds")

    run_pairwise_cv("rf", candidates, folds, cfg, make_model, prob_fn, smoke=args.smoke)


if __name__ == "__main__":
    main()
