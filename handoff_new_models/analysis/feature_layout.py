"""Exact feature-name layout for the pairwise rankers — a self-contained copy of
src/analysis/feature_names.py so the handoff folder stays independent of src/.

The model input matrix produced by data._ds_to_arrays lays columns out as:

    [ structural (cfg.feature_cols) ]
    [ txt_pca_0 .. txt_pca_{nt-1}   ]   (only if semantic enabled)
    [ img_pca_0 .. img_pca_{ni-1}   ]   (only if semantic enabled)
    [ position                       ]   (only if cfg.use_position_feature)

For the pairwise models the same layout applies to dX = X_i - X_j (each column is
the *difference* of that feature between the two items in a pair).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data import load_embeddings

POSITION_NAME = "position"


def _semantic_dims(cfg: Config) -> Tuple[int, int]:
    """(n_text_pca, n_image_pca) actually appended, matching data.semantic_added_dims."""
    if not getattr(cfg, "use_semantic_features", False):
        return 0, 0
    text_df, image_df = load_embeddings(cfg)
    nt = ni = 0
    if text_df is not None and getattr(cfg, "text_pca_dim", 0) > 0:
        n_raw = sum(c.startswith("txt_") and c[4:].isdigit() for c in text_df.columns)
        nt = min(cfg.text_pca_dim, n_raw)
    if image_df is not None and getattr(cfg, "image_pca_dim", 0) > 0:
        n_raw = sum(c.startswith("img_") and c[4:].isdigit() for c in image_df.columns)
        ni = min(cfg.image_pca_dim, n_raw)
    return nt, ni


@dataclass
class FeatureLayout:
    names: List[str]
    structural: Tuple[int, int]   # [start, end) index range
    text_pca: Tuple[int, int]
    image_pca: Tuple[int, int]
    position: Tuple[int, int]     # empty range (i, i) when disabled

    def block_of(self, idx: int) -> str:
        for name, (lo, hi) in (
            ("structural", self.structural),
            ("text_pca", self.text_pca),
            ("image_pca", self.image_pca),
            ("position", self.position),
        ):
            if lo <= idx < hi:
                return name
        return "unknown"


def build_feature_layout(cfg: Config) -> FeatureLayout:
    structural = list(cfg.feature_cols)
    nt, ni = _semantic_dims(cfg)
    text = [f"txt_pca_{i}" for i in range(nt)]
    image = [f"img_pca_{i}" for i in range(ni)]
    pos = [POSITION_NAME] if getattr(cfg, "use_position_feature", False) else []

    names = structural + text + image + pos
    s0 = 0
    s1 = s0 + len(structural)
    t1 = s1 + len(text)
    i1 = t1 + len(image)
    p1 = i1 + len(pos)
    return FeatureLayout(
        names=names,
        structural=(s0, s1),
        text_pca=(s1, t1),
        image_pca=(t1, i1),
        position=(i1, p1),
    )
