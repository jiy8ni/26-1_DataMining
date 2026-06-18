"""
JOIN PCA-reduced text/image embeddings onto the existing train/val/test feature
CSVs, keyed by resolved_url. This reuses the already-built embedding parquets —
it never recrawls or re-embeds.

A SINGLE global PCA is fit on the unique-resolved_url embeddings (all splits) and
applied to every row. This is intended for EDA on the split CSVs, where a shared
axis across splits is desirable. NOTE: do NOT feed these stored columns into model
training — the training pipeline (data.py) fits PCA per-fold on train only to avoid
leakage and reads the raw parquets directly, so it is unaffected by these columns.

For each data/processed/{step1,step2}_{train,val,test}_features.csv:
  - left-join txt_pca_0..(K-1)   (from item_text_emb.parquet,  1024-d -> K)
  - left-join img_pca_0..(M-1)   (from item_image_emb.parquet,  512-d -> M)
  - left-join img_missing        (1 if the page had no usable image)
Row count and order are preserved; URLs with no embedding row get NaN columns.

Dims default to Config.text_pca_dim / Config.image_pca_dim (16 / 8).

    python src/join_embeddings.py --inplace          # overwrite originals
    python src/join_embeddings.py                     # write *_features_emb.csv copies
    python src/join_embeddings.py --text-dim 32 --image-dim 16
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from config import Config

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
TEXT_EMB = PROC / "item_text_emb.parquet"
IMAGE_EMB = PROC / "item_image_emb.parquet"

SPLITS = [
    "step1_train", "step1_val", "step1_test",
    "step2_train", "step2_val", "step2_test",
]


def reduce_global(emb_df: pd.DataFrame, raw_prefix: str, out_prefix: str,
                  n_dim: int, drop_missing_for_fit: bool) -> pd.DataFrame:
    """Fit one global PCA on the unique-URL embedding rows and return a frame of
    [resolved_url, out_prefix0..]. When drop_missing_for_fit, rows flagged
    img_missing==1 are excluded from the fit (their zero vectors would distort
    the components) but still transformed."""
    raw_cols = [c for c in emb_df.columns
                if c.startswith(raw_prefix) and c[len(raw_prefix):].isdigit()]
    X = emb_df[raw_cols].to_numpy(dtype=np.float32)

    fit_mask = np.ones(len(emb_df), dtype=bool)
    if drop_missing_for_fit and "img_missing" in emb_df.columns:
        fit_mask = emb_df["img_missing"].to_numpy() == 0

    n_comp = min(n_dim, len(raw_cols), int(fit_mask.sum()))
    pca = PCA(n_components=n_comp, random_state=0)
    pca.fit(X[fit_mask])
    reduced = pca.transform(X)

    out = pd.DataFrame(reduced, columns=[f"{out_prefix}{i}" for i in range(n_comp)])
    out.insert(0, "resolved_url", emb_df["resolved_url"].values)
    evr = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA {raw_prefix}* {len(raw_cols)} -> {n_comp}  "
          f"(explained var {evr:.1%}, fit on {int(fit_mask.sum())} urls)")
    return out


def main() -> None:
    cfg = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--inplace", action="store_true",
                        help="Overwrite the original *_features.csv.")
    parser.add_argument("--text-dim", type=int, default=cfg.text_pca_dim)
    parser.add_argument("--image-dim", type=int, default=cfg.image_pca_dim)
    args = parser.parse_args()

    text_df = pd.read_parquet(TEXT_EMB)
    image_df = pd.read_parquet(IMAGE_EMB)
    print(f"emb urls: text={text_df.resolved_url.nunique()} "
          f"image={image_df.resolved_url.nunique()}")

    txt_red = reduce_global(text_df, "txt_", "txt_pca_", args.text_dim,
                            drop_missing_for_fit=False)
    img_red = reduce_global(image_df, "img_", "img_pca_", args.image_dim,
                            drop_missing_for_fit=True)
    img_red["img_missing"] = image_df["img_missing"].values  # carry the flag

    for name in SPLITS:
        src = PROC / f"{name}_features.csv"
        if not src.exists():
            print(f"  [skip] {src.name} not found")
            continue
        df = pd.read_csv(src)
        n0 = len(df)
        merged = (df.merge(txt_red, on="resolved_url", how="left")
                    .merge(img_red, on="resolved_url", how="left"))
        assert len(merged) == n0, f"row count changed for {name}: {n0} -> {len(merged)}"
        matched = merged["txt_pca_0"].notna().sum()
        out = src if args.inplace else PROC / f"{name}_features_emb.csv"
        merged.to_csv(out, index=False)
        print(f"  {name}: rows={n0}  cols {df.shape[1]} -> {merged.shape[1]}  "
              f"matched_emb={matched}/{n0}  -> {out.name}")


if __name__ == "__main__":
    main()
