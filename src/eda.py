from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts" / "eda"

TRIAL_KEYS = ["set_id", "engine", "round"]
SENTINEL_COLS = [
    "ph_value",
    "aggregate_rating_value",
    "aggregate_rating_count",
    "volume_ml",
]
RAW_OUTLIER_COLS = [
    "price_krw",
    "volume_ml",
    "text_length",
    "aggregate_rating_count",
    "image_count",
]
TRAIN_ID_COLS = {
    "sku_pos",
    "ai_rank",
    "is_ambiguous",
}
DISPLAY_LABELS = {
    "text_length": "텍스트 길이",
    "price_krw": "가격(원)",
    "volume_ml": "용량(ml)",
    "price_per_ml": "ml당 가격(원)",
    "aggregate_rating_count": "리뷰 수",
    "aggregate_rating_value": "평점",
    "brand_ko": "브랜드",
    "product_count": "상품 수",
    "cleansing_subtype": "클렌저 타입",
    "count": "개수",
    "appearances": "등장 횟수",
    "cooccur_count": "동시 등장 횟수",
    "price_bucket_combo": "가격대 조합",
    "degree": "비교 네트워크 연결 수",
    "top1_rate": "Top1 선택 비율",
    "avg_rank": "평균 순위",
    "sku_pos": "후보 제시 위치",
    "price_bucket": "가격대",
}
VALUE_LABELS = {
    "price_bucket": {
        "low": "저가",
        "mid": "중가",
        "high": "고가",
        "unknown": "미상",
    }
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_json(obj: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def display_label(name: str) -> str:
    if name in DISPLAY_LABELS:
        return f"{DISPLAY_LABELS[name]} ({name})"
    return name


def map_value_labels(values: pd.Series, key: str) -> pd.Series:
    mapping = VALUE_LABELS.get(key, {})
    return values.map(lambda x: mapping.get(x, x))


def replace_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in SENTINEL_COLS:
        if col in out.columns:
            out[col] = out[col].replace(-1, np.nan)
    return out


def load_page_features() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "page_features.csv")
    return df.drop(columns=[c for c in df.columns if df[c].isna().all()])


def load_response() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "response.csv")


def load_trial_items() -> pd.DataFrame:
    return pd.read_csv(PROC_DIR / "trial_items.csv")


def load_train_features(protocol: str) -> pd.DataFrame:
    return pd.read_csv(PROC_DIR / f"{protocol}_train_features.csv")


def numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in TRAIN_ID_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def build_item_id(df: pd.DataFrame) -> pd.Series:
    if "resolved_url" in df.columns:
        item_id = df["resolved_url"].astype("string")
    else:
        item_id = pd.Series(pd.NA, index=df.index, dtype="string")

    if "fp_url" in df.columns:
        item_id = item_id.fillna(df["fp_url"].astype("string"))

    brand_fallback = "BRAND::" + df["brand_ko"].astype("string")
    return item_id.fillna(brand_fallback)


def parse_skin_targets(series: pd.Series) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    for value in series.dropna():
        for token in str(value).split(","):
            token = token.strip()
            if token:
                counts[token] += 1
    rows = [{"skin_target": key, "count": value} for key, value in counts.most_common()]
    return pd.DataFrame(rows)


def iqr_outlier_summary(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for col in columns:
        if col not in df.columns:
            continue
        values = df[col].dropna()
        if values.empty:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        rows.append(
            {
                "column": col,
                "min": float(values.min()),
                "median": float(values.median()),
                "max": float(values.max()),
                "iqr_lower": float(lo),
                "iqr_upper": float(hi),
                "outlier_count": int(((values < lo) | (values > hi)).sum()),
            }
        )
    return pd.DataFrame(rows)


def add_price_per_ml(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"price_krw", "volume_ml"}.issubset(out.columns):
        valid = out["volume_ml"].gt(0) & out["price_krw"].notna()
        out["price_per_ml"] = np.where(valid, out["price_krw"] / out["volume_ml"], np.nan)
    return out


def assign_price_bucket(df: pd.DataFrame, item_col: str) -> pd.DataFrame:
    out = df.copy()
    unique_items = (
        out[[item_col, "price_krw"]]
        .dropna(subset=["price_krw"])
        .drop_duplicates(subset=[item_col])
        .copy()
    )
    if unique_items["price_krw"].empty:
        out["price_bucket"] = pd.NA
        return out

    q1 = float(unique_items["price_krw"].quantile(1 / 3))
    q2 = float(unique_items["price_krw"].quantile(2 / 3))

    def _bucket(value: float) -> str | pd._libs.missing.NAType:
        if pd.isna(value):
            return pd.NA
        if value <= q1:
            return "low"
        if value <= q2:
            return "mid"
        return "high"

    unique_items["price_bucket"] = unique_items["price_krw"].map(_bucket)
    return out.merge(unique_items[[item_col, "price_bucket"]], on=item_col, how="left")


def candidate_triplet_table(trial_items: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for set_id, group in trial_items.groupby("set_id", sort=False):
        item_ids = tuple(sorted(group["item_id"].tolist()))
        brand_ids = tuple(sorted(group["brand_ko"].astype(str).tolist()))
        rows.append(
            {
                "set_id": set_id,
                "item_triplet": " | ".join(item_ids),
                "brand_triplet": " | ".join(brand_ids),
                "n_items": int(len(group)),
                "n_unique_items": int(group["item_id"].nunique()),
                "n_unique_brands": int(group["brand_ko"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def pair_count_table(trial_items: pd.DataFrame) -> pd.DataFrame:
    pair_counter: Counter[tuple[str, str]] = Counter()
    for _, group in trial_items.groupby("set_id", sort=False):
        item_ids = sorted(group["item_id"].unique().tolist())
        for left, right in combinations(item_ids, 2):
            pair_counter[(left, right)] += 1
    rows = [
        {"item_a": pair[0], "item_b": pair[1], "cooccur_count": count}
        for pair, count in pair_counter.items()
    ]
    return pd.DataFrame(rows).sort_values("cooccur_count", ascending=False)


def graph_connectivity(trial_items: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    adjacency: defaultdict[str, set[str]] = defaultdict(set)
    for _, group in trial_items.groupby("set_id", sort=False):
        item_ids = group["item_id"].unique().tolist()
        for item_id in item_ids:
            adjacency[item_id]
        for left, right in combinations(item_ids, 2):
            adjacency[left].add(right)
            adjacency[right].add(left)

    degree_rows = [
        {"item_id": item_id, "degree": len(neighbors)}
        for item_id, neighbors in adjacency.items()
    ]
    degree_df = pd.DataFrame(degree_rows).sort_values("degree")

    visited: set[str] = set()
    component_sizes: list[int] = []
    for start in adjacency:
        if start in visited:
            continue
        stack = [start]
        size = 0
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            size += 1
            stack.extend(adjacency[node] - visited)
        component_sizes.append(size)

    summary = {
        "n_nodes": int(len(adjacency)),
        "n_components": int(len(component_sizes)),
        "largest_component_size": int(max(component_sizes) if component_sizes else 0),
        "min_degree": int(degree_df["degree"].min() if not degree_df.empty else 0),
        "median_degree": float(degree_df["degree"].median() if not degree_df.empty else 0),
        "max_degree": int(degree_df["degree"].max() if not degree_df.empty else 0),
    }
    return degree_df, summary


def run_full_data_qc(
    page_features: pd.DataFrame,
    response: pd.DataFrame,
    trial_items: pd.DataFrame,
    out_dir: Path,
) -> dict:
    ensure_dir(out_dir)
    pf = add_price_per_ml(replace_sentinels(page_features))
    ti = add_price_per_ml(trial_items.copy())
    ti["item_id"] = build_item_id(ti)

    quality_dir = out_dir / "00_feature_quality"
    market_dir = out_dir / "01_market_representativeness"
    candidate_dir = out_dir / "02_candidate_set_quality"
    ensure_dir(quality_dir)
    ensure_dir(market_dir)
    ensure_dir(candidate_dir)

    missing_df = (
        pf.isna()
        .mean()
        .mul(100)
        .rename("missing_pct")
        .rename_axis("column")
        .reset_index()
        .sort_values("missing_pct", ascending=False)
    )
    save_csv(missing_df, quality_dir / "missingness.csv")

    sentinel_rows: list[dict] = []
    for col in SENTINEL_COLS:
        if col in page_features.columns:
            sentinel_rows.append(
                {
                    "column": col,
                    "sentinel_count": int((page_features[col] == -1).sum()),
                    "sentinel_pct": float((page_features[col] == -1).mean() * 100),
                }
            )
    sentinel_df = pd.DataFrame(sentinel_rows)
    save_csv(sentinel_df, quality_dir / "sentinel_values.csv")

    outlier_df = iqr_outlier_summary(pf, RAW_OUTLIER_COLS + ["price_per_ml"])
    save_csv(outlier_df, quality_dir / "outlier_summary.csv")

    duplicate_summary = pd.DataFrame(
        [
            {"metric": "page_features_rows", "value": int(len(page_features))},
            {"metric": "unique_urls", "value": int(page_features["url"].nunique())},
            {"metric": "duplicate_urls", "value": int(page_features["url"].duplicated().sum())},
            {"metric": "unique_brands", "value": int(page_features["brand_ko"].nunique())},
            {
                "metric": "brands_with_multiple_products",
                "value": int((page_features["brand_ko"].value_counts() > 1).sum()),
            },
            {
                "metric": "image_missing_rate_pct",
                "value": float(page_features["image_count"].fillna(0).eq(0).mean() * 100),
            },
            {"metric": "response_rows", "value": int(len(response))},
            {"metric": "response_set_ids", "value": int(response["set_id"].nunique())},
            {"metric": "trial_item_rows", "value": int(len(trial_items))},
            {"metric": "trial_item_set_ids", "value": int(trial_items["set_id"].nunique())},
        ]
    )
    save_csv(duplicate_summary, quality_dir / "dataset_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_cols = ["text_length", "price_krw", "volume_ml", "price_per_ml"]
    for ax, col in zip(axes.ravel(), plot_cols):
        data = pf[col].dropna()
        ax.hist(data, bins=30, color="steelblue", edgecolor="white")
        ax.set_title(display_label(col))
        ax.set_xlabel(display_label(col))
        ax.set_ylabel("빈도")
    save_figure(fig, quality_dir / "core_numeric_distributions.png")

    brand_counts = (
        pf["brand_ko"]
        .value_counts()
        .rename_axis("brand_ko")
        .reset_index(name="product_count")
    )
    save_csv(brand_counts, market_dir / "brand_counts.csv")

    subtype_counts = (
        pf["cleansing_subtype"]
        .fillna("unknown")
        .value_counts()
        .rename_axis("cleansing_subtype")
        .reset_index(name="count")
    )
    save_csv(subtype_counts, market_dir / "subtype_counts.csv")

    skin_target_counts = parse_skin_targets(pf.get("skin_type_targets", pd.Series(dtype="object")))
    save_csv(skin_target_counts, market_dir / "skin_target_counts.csv")

    market_numeric_summary = (
        pf[["price_krw", "volume_ml", "price_per_ml", "aggregate_rating_count", "aggregate_rating_value"]]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .T
        .rename_axis("column")
        .reset_index()
    )
    save_csv(market_numeric_summary, market_dir / "numeric_summary.csv")

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.ravel()
    axes[0].bar(brand_counts.head(20)["brand_ko"], brand_counts.head(20)["product_count"], color="teal")
    axes[0].tick_params(axis="x", rotation=75)
    axes[0].set_title(f"상위 20개 브랜드별 상품 수: {display_label('product_count')}")
    axes[0].set_xlabel(display_label("brand_ko"))
    axes[0].set_ylabel(display_label("product_count"))

    axes[1].hist(pf["price_krw"].dropna(), bins=30, color="salmon", edgecolor="white")
    axes[1].set_title(f"{display_label('price_krw')} 분포")
    axes[1].set_xlabel(display_label("price_krw"))
    axes[1].set_ylabel("빈도")

    axes[2].bar(subtype_counts["cleansing_subtype"], subtype_counts["count"], color="slateblue")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].set_title(f"{display_label('cleansing_subtype')} 분포")
    axes[2].set_xlabel(display_label("cleansing_subtype"))
    axes[2].set_ylabel(display_label("count"))

    axes[3].hist(pf["volume_ml"].dropna(), bins=30, color="darkseagreen", edgecolor="white")
    axes[3].set_title(f"{display_label('volume_ml')} 분포")
    axes[3].set_xlabel(display_label("volume_ml"))
    axes[3].set_ylabel("빈도")

    axes[4].hist(pf["price_per_ml"].dropna(), bins=30, color="goldenrod", edgecolor="white")
    axes[4].set_title(f"{display_label('price_per_ml')} 분포")
    axes[4].set_xlabel(display_label("price_per_ml"))
    axes[4].set_ylabel("빈도")

    axes[5].hist(pf["aggregate_rating_count"].dropna(), bins=30, color="indianred", edgecolor="white")
    axes[5].set_title(f"{display_label('aggregate_rating_count')} 분포")
    axes[5].set_xlabel(display_label("aggregate_rating_count"))
    axes[5].set_ylabel("빈도")
    save_figure(fig, market_dir / "market_distribution_overview.png")

    triplet_df = candidate_triplet_table(ti)
    duplicate_triplets = (
        triplet_df["item_triplet"]
        .value_counts()
        .rename_axis("item_triplet")
        .reset_index(name="repeat_count")
        .query("repeat_count > 1")
        .sort_values("repeat_count", ascending=False)
    )
    save_csv(duplicate_triplets, candidate_dir / "duplicate_triplets.csv")

    appearance_df = (
        ti.groupby("item_id")
        .agg(
            brand_ko=("brand_ko", "first"),
            appearances=("set_id", "size"),
            unique_set_ids=("set_id", "nunique"),
            is_ambiguous=("is_ambiguous", "max"),
            price_krw=("price_krw", "first"),
        )
        .reset_index()
        .sort_values("appearances", ascending=False)
    )
    save_csv(appearance_df, candidate_dir / "item_appearance_counts.csv")

    pair_df = pair_count_table(ti)
    save_csv(pair_df, candidate_dir / "pair_cooccurrence_counts.csv")

    triplet_price = assign_price_bucket(ti, "item_id")
    price_combo_rows: list[dict] = []
    for set_id, group in triplet_price.groupby("set_id", sort=False):
        combo = " | ".join(sorted(group["price_bucket"].fillna("unknown").astype(str).tolist()))
        price_combo_rows.append({"set_id": set_id, "price_bucket_combo": combo})
    price_combo_df = pd.DataFrame(price_combo_rows)
    price_combo_counts = (
        price_combo_df["price_bucket_combo"]
        .value_counts()
        .rename_axis("price_bucket_combo")
        .reset_index(name="count")
    )
    save_csv(price_combo_counts, candidate_dir / "price_bucket_combo_counts.csv")

    degree_df, graph_summary = graph_connectivity(ti)
    save_csv(degree_df, candidate_dir / "network_degree_summary.csv")

    duplicate_brand_trials = (
        triplet_df.query("n_unique_brands < 3")
        .sort_values(["n_unique_brands", "set_id"])
        .reset_index(drop=True)
    )
    save_csv(duplicate_brand_trials, candidate_dir / "duplicate_brand_trials.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    axes[0].hist(appearance_df["appearances"], bins=30, color="steelblue", edgecolor="white")
    axes[0].set_title(f"{display_label('appearances')} 분포")
    axes[0].set_xlabel(display_label("appearances"))
    axes[0].set_ylabel("빈도")

    axes[1].hist(pair_df["cooccur_count"], bins=30, color="darkorange", edgecolor="white")
    axes[1].set_title(f"{display_label('cooccur_count')} 분포")
    axes[1].set_xlabel(display_label("cooccur_count"))
    axes[1].set_ylabel("빈도")

    axes[2].bar(
        price_combo_counts.head(10)["price_bucket_combo"].map(
            lambda x: " | ".join(map_value_labels(pd.Series(str(x).split(" | ")), "price_bucket").astype(str))
        ),
        price_combo_counts.head(10)["count"],
        color="seagreen",
    )
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].set_title(f"상위 가격대 조합: {display_label('price_bucket_combo')}")
    axes[2].set_xlabel(display_label("price_bucket_combo"))
    axes[2].set_ylabel(display_label("count"))

    axes[3].hist(degree_df["degree"], bins=30, color="mediumpurple", edgecolor="white")
    axes[3].set_title(f"{display_label('degree')} 분포")
    axes[3].set_xlabel(display_label("degree"))
    axes[3].set_ylabel("빈도")
    save_figure(fig, candidate_dir / "candidate_quality_overview.png")

    return {
        "page_features_rows": int(len(page_features)),
        "response_rows": int(len(response)),
        "trial_set_ids": int(trial_items["set_id"].nunique()),
        "duplicate_triplet_count": int(len(duplicate_triplets)),
        "duplicate_brand_trial_count": int(len(duplicate_brand_trials)),
        "graph_summary": graph_summary,
    }


def clean_train_for_modeling(train_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = train_df.copy()
    initial_rows = len(df)
    initial_trials = df.groupby(TRIAL_KEYS).ngroups

    ambiguous_rows = int(df["is_ambiguous"].astype(bool).sum()) if "is_ambiguous" in df.columns else 0
    df = df[~df["is_ambiguous"].astype(bool)].copy()

    trial_size = df.groupby(TRIAL_KEYS).size().rename("row_count").reset_index()
    valid_trial_keys = trial_size[trial_size["row_count"] == 3][TRIAL_KEYS]
    df = df.merge(valid_trial_keys, on=TRIAL_KEYS, how="inner")
    incomplete_trials = int(initial_trials - len(valid_trial_keys))

    brand_uniques = (
        df.groupby(TRIAL_KEYS)["brand_ko"]
        .nunique()
        .rename("brand_nunique")
        .reset_index()
    )
    duplicate_brand_keys = brand_uniques[brand_uniques["brand_nunique"] < 3][TRIAL_KEYS]
    duplicate_brand_trials = int(len(duplicate_brand_keys))
    if duplicate_brand_trials:
        dup_key_df = duplicate_brand_keys.assign(_drop_me=True)
        df = df.merge(dup_key_df, on=TRIAL_KEYS, how="left")
        df = df[df["_drop_me"] != True].drop(columns=["_drop_me"])

    df = df.copy()
    df["item_id"] = build_item_id(df)
    df = add_price_per_ml(df)

    summary = {
        "initial_rows": int(initial_rows),
        "initial_trials": int(initial_trials),
        "ambiguous_rows_removed": ambiguous_rows,
        "incomplete_trials_removed": incomplete_trials,
        "duplicate_brand_trials_removed": duplicate_brand_trials,
        "final_rows": int(len(df)),
        "final_trials": int(df.groupby(TRIAL_KEYS).ngroups),
    }
    return df, summary


def pairwise_win_table(train_df: pd.DataFrame) -> pd.DataFrame:
    wins: Counter[str] = Counter()
    total_pairs: Counter[str] = Counter()

    for _, group in train_df.groupby(TRIAL_KEYS, sort=False):
        rows = group[["item_id", "ai_rank"]].to_dict("records")
        if len(rows) != 3:
            continue
        for left, right in combinations(rows, 2):
            total_pairs[left["item_id"]] += 1
            total_pairs[right["item_id"]] += 1
            if left["ai_rank"] < right["ai_rank"]:
                wins[left["item_id"]] += 1
            elif right["ai_rank"] < left["ai_rank"]:
                wins[right["item_id"]] += 1

    rows = []
    for item_id in total_pairs:
        rows.append(
            {
                "item_id": item_id,
                "pair_wins": int(wins[item_id]),
                "pair_comparisons": int(total_pairs[item_id]),
                "win_rate": float(wins[item_id] / total_pairs[item_id]) if total_pairs[item_id] else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_train_only_eda(
    train_df: pd.DataFrame,
    page_features: pd.DataFrame,
    protocol: str,
    out_dir: Path,
) -> dict:
    ensure_dir(out_dir)
    clean_df, cleaning_summary = clean_train_for_modeling(train_df)
    feature_cols = numeric_feature_cols(clean_df)

    subtype_map = (
        page_features[["url", "cleansing_subtype"]]
        .dropna(subset=["url"])
        .drop_duplicates(subset=["url"])
        .rename(columns={"url": "resolved_url"})
    )
    clean_df = clean_df.merge(subtype_map, on="resolved_url", how="left")
    clean_df = assign_price_bucket(clean_df, "item_id")

    item_summary = (
        clean_df.groupby("item_id")
        .agg(
            brand_ko=("brand_ko", "first"),
            resolved_url=("resolved_url", "first"),
            appearances=("ai_rank", "size"),
            top1_count=("ai_rank", lambda s: int((s == 1).sum())),
            avg_rank=("ai_rank", "mean"),
            borda_score=("ai_rank", lambda s: float((4 - s).sum())),
            price_krw=("price_krw", "first"),
            price_bucket=("price_bucket", "first"),
            subtype=("cleansing_subtype", "first"),
        )
        .reset_index()
    )
    item_summary["top1_rate"] = item_summary["top1_count"] / item_summary["appearances"]

    win_df = pairwise_win_table(clean_df)
    item_summary = item_summary.merge(win_df, on="item_id", how="left")
    item_summary = item_summary.sort_values(["top1_rate", "avg_rank"], ascending=[False, True])
    save_csv(item_summary, out_dir / f"{protocol}_item_baseline_summary.csv")

    brand_summary = (
        clean_df.groupby("brand_ko")
        .agg(
            appearances=("ai_rank", "size"),
            avg_rank=("ai_rank", "mean"),
            top1_rate=("ai_rank", lambda s: float((s == 1).mean())),
        )
        .reset_index()
        .sort_values(["avg_rank", "appearances"], ascending=[True, False])
    )
    save_csv(brand_summary, out_dir / f"{protocol}_brand_rank_summary.csv")

    price_summary = (
        clean_df.groupby("price_bucket", dropna=False)
        .agg(
            appearances=("ai_rank", "size"),
            avg_rank=("ai_rank", "mean"),
            top1_rate=("ai_rank", lambda s: float((s == 1).mean())),
        )
        .reset_index()
        .sort_values("avg_rank")
    )
    save_csv(price_summary, out_dir / f"{protocol}_price_bucket_rank_summary.csv")

    subtype_summary = (
        clean_df.assign(cleansing_subtype=clean_df["cleansing_subtype"].fillna("unknown"))
        .groupby("cleansing_subtype")
        .agg(
            appearances=("ai_rank", "size"),
            avg_rank=("ai_rank", "mean"),
            top1_rate=("ai_rank", lambda s: float((s == 1).mean())),
        )
        .reset_index()
        .sort_values("avg_rank")
    )
    save_csv(subtype_summary, out_dir / f"{protocol}_subtype_rank_summary.csv")

    feature_rank_mean = (
        clean_df.groupby("ai_rank")[feature_cols]
        .mean()
        .T
        .rename_axis("feature")
        .reset_index()
    )
    feature_rank_mean.columns = ["feature"] + [f"rank_{col}_mean" for col in feature_rank_mean.columns[1:]]
    if {"rank_1_mean", "rank_3_mean"}.issubset(feature_rank_mean.columns):
        feature_rank_mean["rank1_minus_rank3"] = (
            feature_rank_mean["rank_1_mean"] - feature_rank_mean["rank_3_mean"]
        )
    save_csv(feature_rank_mean, out_dir / f"{protocol}_feature_mean_by_rank.csv")

    feature_rank_median = (
        clean_df.groupby("ai_rank")[feature_cols]
        .median()
        .T
        .rename_axis("feature")
        .reset_index()
    )
    feature_rank_median.columns = ["feature"] + [f"rank_{col}_median" for col in feature_rank_median.columns[1:]]
    save_csv(feature_rank_median, out_dir / f"{protocol}_feature_median_by_rank.csv")

    position_bias = (
        clean_df.groupby("sku_pos")
        .agg(
            appearances=("ai_rank", "size"),
            top1_rate=("ai_rank", lambda s: float((s == 1).mean())),
            avg_rank=("ai_rank", "mean"),
        )
        .reset_index()
        .sort_values("sku_pos")
    )
    save_csv(position_bias, out_dir / f"{protocol}_position_bias.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    top_items = item_summary.head(15).sort_values("top1_rate", ascending=True)
    axes[0].barh(top_items["brand_ko"], top_items["top1_rate"], color="royalblue")
    axes[0].set_title(f"상위 상품의 {display_label('top1_rate')}")
    axes[0].set_xlabel(display_label("top1_rate"))
    axes[0].set_ylabel(display_label("brand_ko"))

    axes[1].bar(position_bias["sku_pos"].astype(str), position_bias["top1_rate"], color="darkorange")
    axes[1].set_title(f"{display_label('sku_pos')}별 {display_label('top1_rate')}")
    axes[1].set_xlabel(display_label("sku_pos"))
    axes[1].set_ylabel(display_label("top1_rate"))

    axes[2].bar(brand_summary.head(15)["brand_ko"], brand_summary.head(15)["avg_rank"], color="seagreen")
    axes[2].tick_params(axis="x", rotation=75)
    axes[2].set_title(f"상위 브랜드의 {display_label('avg_rank')}")
    axes[2].set_xlabel(display_label("brand_ko"))
    axes[2].set_ylabel(display_label("avg_rank"))

    axes[3].bar(
        map_value_labels(price_summary["price_bucket"].fillna("unknown"), "price_bucket"),
        price_summary["avg_rank"],
        color="slateblue",
    )
    axes[3].set_title(f"{display_label('price_bucket')}별 {display_label('avg_rank')}")
    axes[3].set_xlabel(display_label("price_bucket"))
    axes[3].set_ylabel(display_label("avg_rank"))

    save_figure(fig, out_dir / f"{protocol}_train_only_overview.png")

    return {
        "protocol": protocol,
        "cleaning_summary": cleaning_summary,
        "modeled_items": int(item_summary["item_id"].nunique()),
        "modeled_brands": int(clean_df["brand_ko"].nunique()),
        "top_position_bias": {
            str(row["sku_pos"]): float(row["top1_rate"])
            for _, row in position_bias.iterrows()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA runner with full-data QC and train-only modeling analysis.")
    parser.add_argument("--protocol", choices=["step1", "step2"], default="step1")
    args = parser.parse_args()

    page_features = load_page_features()
    response = load_response()
    trial_items = load_trial_items()
    train_df = load_train_features(args.protocol)

    out_dir = ARTIFACT_DIR / args.protocol
    ensure_dir(out_dir)

    full_data_summary = run_full_data_qc(page_features, response, trial_items, out_dir / "full_data_qc")
    train_only_summary = run_train_only_eda(train_df, page_features, args.protocol, out_dir / "train_only")

    summary = {
        "protocol": args.protocol,
        "full_data_qc": full_data_summary,
        "train_only": train_only_summary,
    }
    save_json(summary, out_dir / "summary.json")

    print(f"[saved] {out_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
