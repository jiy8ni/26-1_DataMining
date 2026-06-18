"""Brand-CV hyperparameter selection for the pairwise RankSVM ranker.

PDF model 1: SVM adapted to ranking via pairwise margin loss on dX = X_i - X_j.
LinearSVC (linear kernel) or SVC (RBF). Scores via decision_function, summed
per trial. StandardScaler / imputation already handled upstream by build_arrays.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from sklearn.svm import LinearSVC, SVC

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
    "fit_intercept": False,   # sign symmetry of dX vs -dX (PDF note)
    "max_iter": 10000,
}

GRID = {
    "kernel": ["linear", "rbf"],
    "C": [0.01, 0.1, 1.0, 10.0],
}


def make_model(cand, seed):
    if cand["kernel"] == "rbf":
        return SVC(kernel="rbf", C=cand["C"], gamma="scale",
                   class_weight=cand["class_weight"], random_state=seed)
    return LinearSVC(C=cand["C"], class_weight=cand["class_weight"],
                     fit_intercept=cand["fit_intercept"], max_iter=cand["max_iter"],
                     random_state=seed)


def prob_fn(model):
    # RankSVM has no probabilities — use signed decision score (PDF: log loss not core).
    return model.decision_function


def main():
    args = parse_tuner_args("Brand-CV selection for the pairwise RankSVM ranker.")
    cfg = configure_paths(Config())
    if args.smoke:
        apply_smoke_overrides(cfg)
    apply_saved_semantic_config(cfg, smoke=args.smoke)
    folds, _tf, _sc, _pca = build_kfold_arrays(cfg)

    candidates = [{**FIXED, **g} for g in grid_candidates(GRID)]
    if args.smoke:
        candidates = smoke_candidates(candidates, limit=3)
    print(f"RankSVM brand-CV {'[smoke] ' if args.smoke else ''}"
          f"x {len(candidates)} candidates x {cfg.n_folds} folds")

    run_pairwise_cv("ranksvm", candidates, folds, cfg, make_model, prob_fn, smoke=args.smoke)


if __name__ == "__main__":
    main()
