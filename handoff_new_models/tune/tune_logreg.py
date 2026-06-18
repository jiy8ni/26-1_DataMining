"""Brand-CV hyperparameter selection for the pairwise Logistic Regression ranker.

PDF model 3: Bradley-Terry-style binary classification on dX = X_i - X_j with
fit_intercept=False. P(i beats j) via predict_proba, summed per trial.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression

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

# L2 is the LogisticRegression default; passing penalty="l2" explicitly is
# deprecated in sklearn 1.8, so we rely on the default and only vary C.
FIXED = {
    "fit_intercept": False,   # A-B vs B-A sign symmetry (PDF note)
    "class_weight": "balanced",
    "solver": "lbfgs",
    "max_iter": 5000,
}

GRID = {
    "C": [0.01, 0.1, 1.0, 10.0],
}


def make_model(cand, seed):
    return LogisticRegression(
        C=cand["C"],
        fit_intercept=cand["fit_intercept"],
        class_weight=cand["class_weight"],
        solver=cand["solver"],
        max_iter=cand["max_iter"],
        random_state=seed,
    )


def prob_fn(model):
    return lambda d: model.predict_proba(d)[:, 1]


def main():
    args = parse_tuner_args("Brand-CV selection for the pairwise Logistic Regression ranker.")
    cfg = configure_paths(Config())
    if args.smoke:
        apply_smoke_overrides(cfg)
    apply_saved_semantic_config(cfg, smoke=args.smoke)
    folds, _tf, _sc, _pca = build_kfold_arrays(cfg)

    candidates = [{**FIXED, **g} for g in grid_candidates(GRID)]
    if args.smoke:
        candidates = smoke_candidates(candidates, limit=3)
    print(f"Logistic Regression brand-CV {'[smoke] ' if args.smoke else ''}"
          f"x {len(candidates)} candidates x {cfg.n_folds} folds")

    run_pairwise_cv("logreg", candidates, folds, cfg, make_model, prob_fn, smoke=args.smoke)


if __name__ == "__main__":
    main()
