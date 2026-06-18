"""Shared helpers for the brand-CV hyperparameter tuning scripts.

The tuners reuse the existing brand-level k-fold splitters in data.py
(build_kfold_arrays / build_kfold_loaders, both driven by _brand_kfold_splits)
to score each HP candidate as the mean over folds, then pick the candidate with
the highest balanced selection score (metrics.balanced_score). The chosen
params are written to artifacts/tuning/{model}_best_params.json, which the
vanilla single-split trainers load at runtime.
"""
import itertools
import json
import os
from typing import Dict, List

import numpy as np
import torch

from calibration import TemperatureCalibration
from metrics import evaluate_all, balanced_score
from tune.runtime import tuning_artifact_path


def grid_candidates(grid: Dict[str, list]) -> List[dict]:
    """Cartesian product of a {param: [values]} grid -> list of param dicts."""
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def eval_fold(val_scores_2d: np.ndarray, ranks_2d: np.ndarray, temp_candidates) -> dict:
    """Calibrate temperature on this fold's val scores, then evaluate_all (with
    probabilities so nll is included). Returns the metric dict + chosen temperature."""
    calib = TemperatureCalibration(temp_candidates)
    calib.fit(
        torch.tensor(val_scores_2d, dtype=torch.float32),
        torch.tensor(ranks_2d,      dtype=torch.long),
    )
    exp_cal = np.exp(val_scores_2d / calib.temperature)
    probs   = exp_cal / exp_cal.sum(axis=1, keepdims=True)
    res = evaluate_all(val_scores_2d, ranks_2d, probs)
    res["temperature"] = float(calib.temperature)
    return res


def aggregate_candidate(fold_results: List[dict]) -> dict:
    """Mean across folds for every metric (temperature included)."""
    return {k: float(np.nanmean([r[k] for r in fold_results])) for k in fold_results[0]}


def _select_best(candidate_means: List[dict]) -> tuple[list[float], int]:
    composites = balanced_score(candidate_means)
    best_idx = int(np.argmax(composites))
    return composites, best_idx


def select_and_save(
    model_name: str,
    candidates: List[dict],
    candidate_means: List[dict],
    tuning_dir: str,
    *,
    smoke: bool = False,
) -> int:
    """Pick the best candidate by balanced_score and dump the result JSON.

    Returns the index of the winning candidate. ``candidates`` are the FULL param
    dicts (fixed + grid) so the saved file is directly loadable by a trainer.
    """
    composites, best_idx = _select_best(candidate_means)

    os.makedirs(tuning_dir, exist_ok=True)
    path = tuning_artifact_path(tuning_dir, f"{model_name}_best_params.json", smoke=smoke)
    payload = {
        "model":       model_name,
        "params":      candidates[best_idx],
        "cv_balanced": float(composites[best_idx]),
        "per_metric":  candidate_means[best_idx],
        "all_candidates": [
            {"params": candidates[i], "balanced": float(composites[i]), "metrics": candidate_means[i]}
            for i in range(len(candidates))
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'='*60}\nBest {model_name} candidate (idx {best_idx}, balanced={composites[best_idx]:.4f}):")
    for k, v in candidates[best_idx].items():
        print(f"    {k}: {v}")
    print("  CV-mean metrics:")
    for k, v in candidate_means[best_idx].items():
        print(f"    {k:<16} {v:.4f}")
    print(f"  -> saved to {path}")
    return best_idx


def select_and_save_semantic(
    candidates: List[dict],
    candidate_means: List[dict],
    tuning_dir: str,
    *,
    smoke: bool = False,
) -> int:
    """Pick the best semantic configuration by balanced_score and dump JSON."""
    composites, best_idx = _select_best(candidate_means)

    os.makedirs(tuning_dir, exist_ok=True)
    path = tuning_artifact_path(tuning_dir, "semantic_best_config.json", smoke=smoke)
    payload = {
        "semantic_config": candidates[best_idx],
        "cv_balanced": float(composites[best_idx]),
        "per_metric": candidate_means[best_idx],
        "all_candidates": [
            {"semantic_config": candidates[i], "balanced": float(composites[i]), "metrics": candidate_means[i]}
            for i in range(len(candidates))
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(
        f"\n{'='*60}\nBest semantic config (idx {best_idx}, "
        f"balanced={composites[best_idx]:.4f}):"
    )
    for k, v in candidates[best_idx].items():
        print(f"    {k}: {v}")
    print("  CV-mean metrics:")
    for k, v in candidate_means[best_idx].items():
        print(f"    {k:<16} {v:.4f}")
    print(f"  -> saved to {path}")
    return best_idx
