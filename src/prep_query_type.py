"""
Build the query-type–aware dataset (protocol "step2qt").

Source: data/raw/anthropic_various_query_type.csv — the same product pool ranked
by claude-haiku-4-5 while query framing (query_type) and persona are varied.

This mirrors the step2 (brand-level holdout, unseen-item) preprocessing used for
the main dataset, but the raw file is already URL-resolved (sku{1,2,3}_url), so no
brand→URL resolution is needed. query_type and persona are one-hot encoded and
written as plain 0/1 feature columns; embeddings are NOT pre-joined (RankingDataset
merges them from the parquet at runtime by resolved_url).

Outputs:
    data/processed/step2qt_{train,val,test}_features.csv
    data/splits/step2qt_{train,val,test}.csv
    data/splits/split_meta_qt.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
os.chdir(_ROOT_DIR)  # make relative paths work regardless of cwd

RAW_PATH      = "data/raw/anthropic_various_query_type.csv"
PAGE_FEATURES = "data/raw/page_features.csv"
PROC_DIR      = "data/processed"
SPLITS_DIR    = "data/splits"

# 24 numeric structural features (FEATURE_COLS_V1 superset, so V1/V2/V3 all work)
FEATURE_COLS_V1 = [
    "text_length", "image_count", "table_count", "list_item_count",
    "paragraph_count", "section_count", "jsonld_field_count",
    "explicit_number_count", "ambiguous_term_count", "numeric_specificity_ratio",
    "price_krw", "skin_type_targets_count", "ph_value",
    "active_ingredient_count", "claim_keyword_count", "texture_keyword_count",
    "no_list_count", "cosmetic_cert_count", "volume_ml",
    "aggregate_rating_value", "aggregate_rating_count",
    "T7_eat_score", "Q4_social_proof_count", "Q9_external_authority_count",
]
# Columns where -1 is a "missing" sentinel (mirrors the EDA notebook cleanup)
SENTINEL_NEG1 = ["ph_value", "aggregate_rating_value", "aggregate_rating_count", "volume_ml"]

QUERY_TYPES = ["USE", "CAT", "SYM", "DEC", "PRC"]
PERSONAS    = ["PRIMARY", "SECONDARY", "TERTIARY1", "TERTIARY2"]

SEED          = 42
HOLDOUT_FRAC  = 0.10
VAL_FRAC      = 0.15   # of the non-holdout trials


def wide_to_long(raw: pd.DataFrame) -> pd.DataFrame:
    """Expand each wide trial row into 3 per-SKU rows with ai_rank derived from
    the rank_{1st,2nd,3rd} columns (each gives the sku position ranked N-th)."""
    rows = []
    for r in raw.itertuples(index=False):
        pos_to_rank = {getattr(r, "rank_1st"): 1,
                       getattr(r, "rank_2nd"): 2,
                       getattr(r, "rank_3rd"): 3}
        for p in (1, 2, 3):
            rows.append({
                "set_id":       r.set_id,
                "engine":       r.engine,
                "round":        r.round,
                "persona":      r.persona,
                "query_type":   r.query_type,
                "sku_pos":      p,
                "brand_ko":     getattr(r, f"sku{p}_brand"),
                "resolved_url": getattr(r, f"sku{p}_url"),
                "is_ambiguous": False,
                "ai_rank":      pos_to_rank[p],
            })
    return pd.DataFrame(rows)


def attach_features(long_df: pd.DataFrame) -> pd.DataFrame:
    pf = pd.read_csv(PAGE_FEATURES)
    pf[SENTINEL_NEG1] = pf[SENTINEL_NEG1].replace(-1, np.nan)
    feat = pf[["url"] + FEATURE_COLS_V1].rename(columns={"url": "resolved_url"})
    merged = long_df.merge(feat, on="resolved_url", how="left")
    missing = merged[FEATURE_COLS_V1[0]].isna().sum()
    if missing:
        print(f"[warn] {missing} item-rows have no page_features match")
    return merged


def add_onehot(df: pd.DataFrame) -> pd.DataFrame:
    for q in QUERY_TYPES:
        df[f"qt_{q}"] = (df["query_type"] == q).astype(int)
    for p in PERSONAS:
        df[f"persona_{p}"] = (df["persona"] == p).astype(int)
    return df


def brand_holdout_split(long_df: pd.DataFrame):
    """step2: hold out HOLDOUT_FRAC of brands → test (any trial touching a held-out
    brand); split remaining trials VAL_FRAC/(1-VAL_FRAC) → val/train."""
    brands = sorted(long_df["brand_ko"].dropna().unique())
    rng_b  = np.random.default_rng(SEED + 1)
    n_hold = max(1, int(len(brands) * HOLDOUT_FRAC))
    holdout = set(rng_b.choice(brands, size=n_hold, replace=False))

    trial_brands = long_df.groupby("set_id")["brand_ko"].apply(set)
    test_ids  = {sid for sid, bs in trial_brands.items() if bs & holdout}
    clean_ids = [sid for sid in trial_brands.index if sid not in test_ids]

    rng_t = np.random.default_rng(SEED)
    shuffled = rng_t.permutation(np.array(clean_ids, dtype=object))
    n_val = int(len(shuffled) * VAL_FRAC)
    val_ids   = set(shuffled[:n_val])
    train_ids = set(shuffled[n_val:])

    return train_ids, val_ids, test_ids, sorted(holdout)


def main():
    raw = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
    print(f"Raw rows: {len(raw)} | unique set_id: {raw['set_id'].nunique()}")

    long_df = wide_to_long(raw)
    long_df = attach_features(long_df)
    long_df = add_onehot(long_df)

    train_ids, val_ids, test_ids, holdout = brand_holdout_split(long_df)
    print(f"Holdout brands ({len(holdout)}): {holdout}")
    split_of = {**{s: "train" for s in train_ids},
                **{s: "val" for s in val_ids},
                **{s: "test" for s in test_ids}}
    long_df["split"] = long_df["set_id"].map(split_of)

    os.makedirs(PROC_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)

    counts = {}
    for split in ("train", "val", "test"):
        sub = long_df[long_df["split"] == split].drop(columns=["split"]).reset_index(drop=True)
        sub.to_csv(f"{PROC_DIR}/step2qt_{split}_features.csv", index=False, encoding="utf-8-sig")
        ids = sorted(sub["set_id"].unique())
        pd.DataFrame({"set_id": ids}).to_csv(
            f"{SPLITS_DIR}/step2qt_{split}.csv", index=False, encoding="utf-8-sig")
        counts[f"step2qt_{split}"] = len(ids)
        print(f"  {split}: {len(ids)} trials, {len(sub)} rows")

    meta = {
        "seed": SEED,
        "holdout_frac": HOLDOUT_FRAC,
        "val_frac": VAL_FRAC,
        "holdout_brands": holdout,
        "set_id_counts": counts,
        "onehot_cols": [f"qt_{q}" for q in QUERY_TYPES] + [f"persona_{p}" for p in PERSONAS],
        "description": (
            "Query-type-aware dataset from anthropic_various_query_type.csv. "
            "step2 brand-level holdout (10% brands -> test; remaining 85/15 -> train/val). "
            "query_type + persona one-hot encoded as 0/1 feature columns."
        ),
    }
    with open(f"{SPLITS_DIR}/split_meta_qt.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved split_meta_qt.json")


if __name__ == "__main__":
    main()
