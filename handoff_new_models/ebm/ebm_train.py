"""Seed-ensemble trainer for the pairwise GAM/EBM ranker (PDF model 4).

Loads brand-CV-tuned params from artifacts/tuning/ebm_best_params.json (falls
back to defaults), fits cfg.n_seeds EBMs, averages per-trial win-prob sums,
temperature-calibrates on val, evaluates on test, and dumps scores for blending.

Extra deliverable: exports the global feature-effect explanation (the curves
f_k(dX_k)) to artifacts/ebm/ — the interpretability output the PDF highlights.
Note: because the input is the pairwise DIFFERENCE dX = X_i - X_j, each curve
describes the effect of a feature *difference* on P(i beats j).
"""
import json
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from config import Config
from data import build_arrays, effective_feature_dim
from paths import configure_paths, HANDOFF_ROOT
from harness import train_seed_ensemble
from tune.runtime import apply_saved_semantic_config, load_tuned_params
from tune.tune_ebm import make_model, prob_fn

DEFAULT_PARAMS = {
    "max_bins": 256,
    "interactions": 0,
    "learning_rate": 0.05,
    "min_samples_leaf": 5,
    "outer_bags": 8,
}


def _export_explanation(model, out_dir):
    """Dump the EBM global explanation (term names + importances) to JSON, and
    try to save the interactive HTML plot if plotly export is available."""
    _os.makedirs(out_dir, exist_ok=True)
    try:
        global_exp = model.explain_global()
        summary = {
            "term_names": list(getattr(model, "term_names_", [])),
            "term_importances": [float(v) for v in model.term_importances()],
        }
        with open(_os.path.join(out_dir, "ebm_global_importances.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  -> saved EBM global importances to {out_dir}")
        try:  # interactive HTML (needs plotly); non-fatal if missing
            global_exp.visualize().write_html(_os.path.join(out_dir, "ebm_global.html"))
            print(f"  -> saved EBM global plot to {out_dir}/ebm_global.html")
        except Exception as e:  # noqa: BLE001
            print(f"  (skipped HTML plot export: {e})")
    except Exception as e:  # noqa: BLE001
        print(f"  (skipped EBM explanation export: {e})")


def main():
    cfg = configure_paths(Config())
    apply_saved_semantic_config(cfg)
    params = load_tuned_params(cfg, "ebm_best_params.json", DEFAULT_PARAMS, "EBM")
    print(f"Features: {effective_feature_dim(cfg)}  params: {params}")
    arrays = build_arrays(cfg)
    *_, models = train_seed_ensemble("ebm", params, arrays, cfg, make_model, prob_fn)
    _export_explanation(models[0], _os.path.join(HANDOFF_ROOT, "artifacts", "ebm"))


if __name__ == "__main__":
    main()
