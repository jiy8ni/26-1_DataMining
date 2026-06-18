"""Blend the seed-ensembled raw scores of the three vanilla rankers (mlp, lgbm,
xgb). Per-model scores are standardized (val mean/std, applied to both splits)
so their scales align, then a simplex weight grid is searched on the validation
set using the same balanced selection objective as the tuners. The winning blend
is temperature-recalibrated on val and reported on test.

Prereq: run mlp/train.py, lightgbm/lgbm_train.py, xgboost/xgb_train.py first so
that artifacts/preds/{model}_{val,test}.npz exist.

Run:  python src/blend.py
"""
import itertools
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import numpy as np
import torch

from config import Config
from calibration import TemperatureCalibration
from metrics import evaluate_all, balanced_score
from preds_io import load_scores

MODELS = ["mlp", "lgbm", "xgb"]
WEIGHT_STEP = 0.1   # simplex granularity


def _standardize(val: np.ndarray, test: np.ndarray):
    """Z-score using val statistics (flattened), applied to both splits."""
    mu = float(val.mean())
    sd = float(val.std()) + 1e-8
    return (val - mu) / sd, (test - mu) / sd


def _probs(scores: np.ndarray, ranks: np.ndarray, temp_candidates):
    """Calibrate temperature on `scores`/`ranks`, return (probs, temperature)."""
    calib = TemperatureCalibration(temp_candidates)
    calib.fit(torch.tensor(scores, dtype=torch.float32),
              torch.tensor(ranks,  dtype=torch.long))
    exp_cal = np.exp(scores / calib.temperature)
    return exp_cal / exp_cal.sum(axis=1, keepdims=True), calib.temperature


def _simplex_weights(step: float):
    """All (w_mlp, w_lgbm, w_xgb) >= 0 summing to 1 on a `step` grid."""
    n = round(1.0 / step)
    out = []
    for a in range(n + 1):
        for b in range(n + 1 - a):
            c = n - a - b
            out.append((a / n, b / n, c / n))
    return out


def main():
    cfg = Config()

    val_s, test_s, val_ranks, test_ranks = {}, {}, None, None
    for m in MODELS:
        vs, vr = load_scores(cfg.preds_dir, m, "val")
        ts, tr = load_scores(cfg.preds_dir, m, "test")
        if val_ranks is None:
            val_ranks, test_ranks = vr, tr
        else:
            # All models score the same trials in the same order; verify.
            assert np.array_equal(vr, val_ranks),  f"{m} val ranks mismatch"
            assert np.array_equal(tr, test_ranks), f"{m} test ranks mismatch"
        val_s[m], test_s[m] = _standardize(vs, ts)

    print(f"Loaded {MODELS} — val trials: {len(val_ranks)}  test trials: {len(test_ranks)}")

    # ---- Search blend weights on val using the balanced objective ----
    weights = _simplex_weights(WEIGHT_STEP)
    val_results, blend_temps = [], []
    for (wm, wl, wx) in weights:
        blended = wm * val_s["mlp"] + wl * val_s["lgbm"] + wx * val_s["xgb"]
        probs, temp = _probs(blended, val_ranks, cfg.temp_candidates)
        val_results.append(evaluate_all(blended, val_ranks, probs))
        blend_temps.append(temp)

    composites = balanced_score(val_results)
    best_i = int(np.argmax(composites))
    wm, wl, wx = weights[best_i]
    best_temp  = blend_temps[best_i]
    print(f"\nBest weights — mlp={wm:.1f}  lgbm={wl:.1f}  xgb={wx:.1f}  (T*={best_temp})")
    print("  Val balanced metrics:")
    for k, v in val_results[best_i].items():
        print(f"    {k:<16} {v:.4f}")

    # ---- Apply to test (val-fitted weights + temperature) ----
    blended_test = wm * test_s["mlp"] + wl * test_s["lgbm"] + wx * test_s["xgb"]
    exp_cal    = np.exp(blended_test / best_temp)
    test_probs = exp_cal / exp_cal.sum(axis=1, keepdims=True)
    results = evaluate_all(blended_test, test_ranks, test_probs)

    print("\n=== Blended Test Results ===")
    for k, v in results.items():
        print(f"  {k:<22} {v:.4f}")

    # ---- For reference: each single model's standalone test metrics ----
    print("\n=== Single-model test (for comparison) ===")
    for m in MODELS:
        probs, temp = _probs(val_s[m], val_ranks, cfg.temp_candidates)  # temp from val
        exp_cal = np.exp(test_s[m] / temp)
        tp = exp_cal / exp_cal.sum(axis=1, keepdims=True)
        r = evaluate_all(test_s[m], test_ranks, tp)
        print(f"  {m:<5} top1={r['top1_accuracy']:.4f}  kτ={r['kendall_tau']:.4f}  nll={r['nll']:.4f}")


if __name__ == "__main__":
    main()
