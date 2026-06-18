"""Runtime helpers shared by tuning / training scripts."""
from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, TypeVar

from config import Config

SMOKE_SUFFIX = "_smoke"
SEMANTIC_FILENAME = "semantic_best_config.json"

T = TypeVar("T")


def parse_tuner_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run a tiny tuning pass for orchestration checks without touching real artifacts",
    )
    return parser.parse_args()


def tuning_artifact_path(tuning_dir: str, filename: str, *, smoke: bool = False) -> str:
    stem, ext = os.path.splitext(filename)
    if smoke:
        filename = f"{stem}{SMOKE_SUFFIX}{ext or '.json'}"
    return os.path.join(tuning_dir, filename)


def apply_smoke_overrides(cfg: Config) -> None:
    cfg.n_folds = min(cfg.n_folds, 2)
    cfg.n_epochs = min(cfg.n_epochs, 8)
    cfg.patience = min(cfg.patience, 3)


def smoke_candidates(items: Iterable[T], limit: int) -> list[T]:
    return list(items)[:limit]


def smoke_num_boost_round(smoke: bool, default: int = 500) -> int:
    return 25 if smoke else default


def smoke_early_stopping_rounds(smoke: bool, default: int = 20) -> int:
    return 5 if smoke else default


def apply_saved_semantic_config(
    cfg: Config,
    *,
    smoke: bool = False,
    quiet: bool = False,
) -> bool:
    candidates = [tuning_artifact_path(cfg.tuning_dir, SEMANTIC_FILENAME, smoke=smoke)]
    if smoke:
        candidates.append(tuning_artifact_path(cfg.tuning_dir, SEMANTIC_FILENAME, smoke=False))

    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        if not quiet:
            print("No saved semantic config found; using Config defaults.")
        return False

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    semantic = payload["semantic_config"]
    cfg.use_semantic_features = bool(semantic["use_semantic_features"])
    cfg.text_pca_dim = int(semantic.get("text_pca_dim", 0))
    cfg.image_pca_dim = int(semantic.get("image_pca_dim", 0))
    if not quiet:
        print(
            f"Loaded semantic config from {path}: "
            f"use_semantic={cfg.use_semantic_features} "
            f"text_pca_dim={cfg.text_pca_dim} image_pca_dim={cfg.image_pca_dim}"
        )
    return True


def load_tuned_params(
    cfg: Config,
    filename: str,
    default_params: dict,
    label: str,
) -> dict:
    path = tuning_artifact_path(cfg.tuning_dir, filename, smoke=False)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            params = json.load(f)["params"]
        print(f"Loaded tuned {label} params from {path}")
        return params
    print(f"No tuned {label} params found; using fallback defaults.")
    return dict(default_params)
