"""N-way blend of the new pairwise rankers (and, optionally, the original
mlp/lgbm/xgb if their .npz are copied into this folder's artifacts/preds/).

Generalizes the original src/blend.py 3-way simplex search to any number of
models. Per-model raw scores are standardized (val mean/std applied to both
splits) so their scales align, then a weight simplex is searched on val using
the same balanced selection objective as the tuners. The winning blend is
temperature-recalibrated on val and reported on test.

Prereq: run the trainers first so artifacts/preds/{model}_{val,test}.npz exist.
Edit MODELS below to control which models are included.

Run:  python blend_new.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import numpy as np
import torch

from config import Config
from paths import configure_paths
from metrics import evaluate_all, balanced_score
from preds_io import load_scores

# Include only models whose preds .npz exist in artifacts/preds/. To also blend
# the original rankers, copy their lgbm/xgb/mlp_{val,test}.npz here and add them.
MODELS = ["ranksvm", "rf", "logreg", "ebm"]
WEIGHT_STEP = 0.1   # simplex granularity


def _standardize(val: np.ndarray, test: np.ndarray):
    mu = float(val.mean())
    sd = float(val.std()) + 1e-8
    return (val - mu) / sd, (test - mu) / sd


def _probs(scores, ranks, temp_candidates):
    from calibration import TemperatureCalibration
    calib = TemperatureCalibration(temp_candidates)
    calib.fit(torch.tensor(scores, dtype=torch.float32),
              torch.tensor(ranks,  dtype=torch.long))
    exp_cal = np.exp(scores / calib.temperature)
    return exp_cal / exp_cal.sum(axis=1, keepdims=True), calib.temperature


def _simplex_weights(n_models: int, step: float):
    """All non-negative weight vectors of length n_models summing to 1 on a grid."""
    n = round(1.0 / step)
    out = []
    # compositions of n into n_models non-negative integer parts -> /n weights
    def _compose(remaining, slots):
        if slots == 1:
            yield (remaining,)
            return
        for first in range(remaining + 1):
            for rest in _compose(remaining - first, slots - 1):
                yield (first,) + rest
    for combo in _compose(n, n_models):
        out.append(tuple(c / n for c in combo))
    return out


def main():
    cfg = configure_paths(Config())

    available = [m for m in MODELS
                 if _os.path.exists(_os.path.join(cfg.preds_dir, f"{m}_val.npz"))]
    missing = [m for m in MODELS if m not in available]
    if missing:
        print(f"Skipping (no preds found): {missing}")
    if len(available) < 2:
        print(f"Need >=2 models with preds to blend; found {available}. Aborting.")
        return

    val_s, test_s, val_ranks, test_ranks = {}, {}, None, None
    for m in available:
        vs, vr = load_scores(cfg.preds_dir, m, "val")
        ts, tr = load_scores(cfg.preds_dir, m, "test")
        if val_ranks is None:
            val_ranks, test_ranks = vr, tr
        else:
            assert np.array_equal(vr, val_ranks),  f"{m} val ranks mismatch"
            assert np.array_equal(tr, test_ranks), f"{m} test ranks mismatch"
        val_s[m], test_s[m] = _standardize(vs, ts)

    print(f"Blending {available} - val trials: {len(val_ranks)}  test trials: {len(test_ranks)}")

    weights = _simplex_weights(len(available), WEIGHT_STEP)
    val_results, blend_temps = [], []
    for w in weights:
        blended = sum(w[i] * val_s[m] for i, m in enumerate(available))
        probs, temp = _probs(blended, val_ranks, cfg.temp_candidates)
        val_results.append(evaluate_all(blended, val_ranks, probs))
        blend_temps.append(temp)

    composites = balanced_score(val_results)
    best_i = int(np.argmax(composites))
    w_best = weights[best_i]
    best_temp = blend_temps[best_i]
    wstr = "  ".join(f"{m}={w_best[i]:.1f}" for i, m in enumerate(available))
    print(f"\nBest weights - {wstr}  (T*={best_temp})")
    print("  Val balanced metrics:")
    for k, v in val_results[best_i].items():
        print(f"    {k:<16} {v:.4f}")

    blended_test = sum(w_best[i] * test_s[m] for i, m in enumerate(available))
    exp_cal = np.exp(blended_test / best_temp)
    test_probs = exp_cal / exp_cal.sum(axis=1, keepdims=True)
    results = evaluate_all(blended_test, test_ranks, test_probs)

    print("\n=== Blended Test Results ===")
    for k, v in results.items():
        print(f"  {k:<22} {v:.4f}")

    print("\n=== Single-model test (for comparison) ===")
    for m in available:
        probs, temp = _probs(val_s[m], val_ranks, cfg.temp_candidates)
        exp_cal = np.exp(test_s[m] / temp)
        tp = exp_cal / exp_cal.sum(axis=1, keepdims=True)
        r = evaluate_all(test_s[m], test_ranks, tp)
        print(f"  {m:<8} top1={r['top1_accuracy']:.4f}  ndcg@3={r['ndcg@3']:.4f}  "
              f"ktau={r['kendall_tau']:.4f}  nll={r['nll']:.4f}")


if __name__ == "__main__":
    main()
