"""Brand-CV hyperparameter selection for the pairwise GAM/EBM ranker.

PDF model 4: ExplainableBoostingClassifier on dX = X_i - X_j. P(i beats j) via
predict_proba, summed per trial. Requires the `interpret` package.
Start at interactions=0 (feature-wise effects only); raise if performance lags.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

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
    "max_bins": 256,
    "outer_bags": 8,
}

GRID = {
    "interactions": [0, 5],
    "learning_rate": [0.01, 0.05],
    "min_samples_leaf": [5, 10],
}


def make_model(cand, seed):
    # Imported lazily so the other three models don't require `interpret`.
    from interpret.glassbox import ExplainableBoostingClassifier
    return ExplainableBoostingClassifier(
        max_bins=cand["max_bins"],
        interactions=cand["interactions"],
        learning_rate=cand["learning_rate"],
        min_samples_leaf=cand["min_samples_leaf"],
        outer_bags=cand["outer_bags"],
        random_state=seed,
    )


def prob_fn(model):
    return lambda d: model.predict_proba(d)[:, 1]


def main():
    args = parse_tuner_args("Brand-CV selection for the pairwise GAM/EBM ranker.")
    cfg = configure_paths(Config())
    if args.smoke:
        apply_smoke_overrides(cfg)
    apply_saved_semantic_config(cfg, smoke=args.smoke)
    folds, _tf, _sc, _pca = build_kfold_arrays(cfg)

    candidates = [{**FIXED, **g} for g in grid_candidates(GRID)]
    if args.smoke:
        candidates = smoke_candidates(candidates, limit=3)
    print(f"EBM brand-CV {'[smoke] ' if args.smoke else ''}"
          f"x {len(candidates)} candidates x {cfg.n_folds} folds")

    run_pairwise_cv("ebm", candidates, folds, cfg, make_model, prob_fn, smoke=args.smoke)


if __name__ == "__main__":
    main()
