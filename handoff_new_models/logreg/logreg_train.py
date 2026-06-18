"""Seed-ensemble trainer for the pairwise Logistic Regression ranker (PDF model 3).

Loads brand-CV-tuned params from artifacts/tuning/logreg_best_params.json (falls
back to defaults), fits cfg.n_seeds models, averages per-trial win-prob sums,
temperature-calibrates on val, evaluates on test, and dumps scores for blending.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from config import Config
from data import build_arrays, effective_feature_dim
from paths import configure_paths
from harness import train_seed_ensemble
from tune.runtime import apply_saved_semantic_config, load_tuned_params
from tune.tune_logreg import make_model, prob_fn

DEFAULT_PARAMS = {
    "C": 1.0,
    "fit_intercept": False,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "max_iter": 5000,
}


def main():
    cfg = configure_paths(Config())
    apply_saved_semantic_config(cfg)
    params = load_tuned_params(cfg, "logreg_best_params.json", DEFAULT_PARAMS, "LogisticRegression")
    print(f"Features: {effective_feature_dim(cfg)}  params: {params}")
    arrays = build_arrays(cfg)
    train_seed_ensemble("logreg", params, arrays, cfg, make_model, prob_fn)


if __name__ == "__main__":
    main()
