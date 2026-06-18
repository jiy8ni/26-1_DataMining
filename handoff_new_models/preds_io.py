"""Tiny IO helper so the vanilla trainers can dump their seed-ensembled raw
scores and src/blend.py can read them back. One .npz per (model, split) holding
the (N_trials, 3) raw scores and matching AI ranks."""
import os
from typing import Tuple

import numpy as np


def save_scores(preds_dir: str, model_name: str, split: str,
                scores: np.ndarray, ranks: np.ndarray) -> str:
    os.makedirs(preds_dir, exist_ok=True)
    path = os.path.join(preds_dir, f"{model_name}_{split}.npz")
    np.savez(path, scores=np.asarray(scores), ranks=np.asarray(ranks))
    return path


def load_scores(preds_dir: str, model_name: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    path = os.path.join(preds_dir, f"{model_name}_{split}.npz")
    data = np.load(path)
    return data["scores"], data["ranks"]
