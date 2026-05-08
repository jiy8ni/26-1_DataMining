from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

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
ARTIFACT_DIR = ROOT / "artifacts" / "eda" / "response_diagnostics"

RANK_COLS = {
    1: "Y2_top1_brand",
    2: "rank2_brand",
    3: "rank3_brand",
}
NUMERIC_RESPONSE_COLS = [
    "response_chars",
    "cost_usd",
    "Y3_numeric_count",
    "Y3_numeric_density",
    "Y4_safety_kw",
    "Y4_skin_type_kw",
    "Y4_total",
]
DISPLAY_LABELS = {
    "response_chars": "응답 길이(문자 수)",
    "cost_usd": "응답 비용(USD)",
    "Y3_numeric_count": "수치 언급 수",
    "Y3_numeric_density": "수치 밀도",
    "Y4_safety_kw": "안전 키워드 수",
    "Y4_skin_type_kw": "피부타입 키워드 수",
    "Y4_total": "전체 키워드 수",
    "top1_rate": "Top1 선택 비율",
    "avg_rank": "평균 순위",
    "shown_pos": "후보 제시 위치",
    "exposure_count": "노출 횟수",
    "top1_count": "Top1 횟수",
    "brand_ko": "브랜드",
    "item_id": "상품 ID",
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


def load_response() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "response.csv")


def load_trial_items() -> pd.DataFrame:
    return pd.read_csv(PROC_DIR / "trial_items.csv")


def build_item_id(df: pd.DataFrame) -> pd.Series:
    if "resolved_url" in df.columns:
        item_id = df["resolved_url"].astype("string")
    else:
        item_id = pd.Series(pd.NA, index=df.index, dtype="string")

    if "fp_url" in df.columns:
        item_id = item_id.fillna(df["fp_url"].astype("string"))

    brand_fallback = "BRAND::" + df["brand_ko"].astype("string")
    return item_id.fillna(brand_fallback)


def prepare_trial_lookup(trial_items: pd.DataFrame) -> pd.DataFrame:
    lookup = trial_items.copy()
    lookup["item_id"] = build_item_id(lookup)
    lookup["mapping_quality"] = np.select(
        [
            lookup["is_ambiguous"].astype(bool),
            lookup["resolved_url"].notna(),
            lookup["fp_url"].notna(),
        ],
        [
            "ambiguous_brand_level",
            "resolved_url",
            "fp_url_only",
        ],
        default="brand_fallback",
    )
    return lookup[
        [
            "set_id",
            "sku_pos",
            "brand_ko",
            "resolved_url",
            "fp_url",
            "item_id",
            "is_ambiguous",
            "mapping_quality",
        ]
    ].drop_duplicates(["set_id", "sku_pos"])


def has_duplicate_input_brands(row: pd.Series) -> bool:
    inputs = [row["input_sku1"], row["input_sku2"], row["input_sku3"]]
    return len(set(inputs)) < 3


def build_input_long(response: pd.DataFrame, trial_lookup: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in response.iterrows():
        duplicate_input_brand = has_duplicate_input_brands(row)
        for sku_pos in [1, 2, 3]:
            rows.append(
                {
                    "set_id": row["set_id"],
                    "round": row["round"],
                    "engine": row["engine"],
                    "sku_pos": sku_pos,
                    "input_brand": row[f"input_sku{sku_pos}"],
                    "duplicate_input_brand_trial": duplicate_input_brand,
                }
            )

    input_long = pd.DataFrame(rows)
    input_long = input_long.merge(
        trial_lookup.rename(columns={"brand_ko": "trial_brand"}),
        on=["set_id", "sku_pos"],
        how="left",
    )
    input_long["brand_match"] = input_long["input_brand"].fillna("") == input_long["trial_brand"].fillna("")
    return input_long


def build_rank_long(response: pd.DataFrame, trial_lookup: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in response.iterrows():
        inputs = {
            1: row["input_sku1"],
            2: row["input_sku2"],
            3: row["input_sku3"],
        }
        duplicate_input_brand = len(set(inputs.values())) < 3

        for rank, col in RANK_COLS.items():
            chosen_brand = row[col]
            if pd.isna(chosen_brand):
                mapping_status = "missing_rank"
                shown_pos = pd.NA
            else:
                matches = [pos for pos, brand in inputs.items() if brand == chosen_brand]
                if len(matches) == 1:
                    mapping_status = "unique"
                    shown_pos = matches[0]
                elif len(matches) > 1:
                    mapping_status = "ambiguous_duplicate_brand"
                    shown_pos = pd.NA
                else:
                    mapping_status = "unmatched_brand"
                    shown_pos = pd.NA

            rows.append(
                {
                    "set_id": row["set_id"],
                    "round": row["round"],
                    "engine": row["engine"],
                    "rank": rank,
                    "chosen_brand": chosen_brand,
                    "shown_pos": shown_pos,
                    "mapping_status": mapping_status,
                    "duplicate_input_brand_trial": duplicate_input_brand,
                }
            )

    rank_long = pd.DataFrame(rows)
    rank_long["shown_pos"] = rank_long["shown_pos"].astype("Int64")
    rank_long = rank_long.merge(
        trial_lookup.rename(columns={"brand_ko": "trial_brand", "sku_pos": "shown_pos"}),
        on=["set_id", "shown_pos"],
        how="left",
    )
    return rank_long


def response_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in NUMERIC_RESPONSE_COLS if col in df.columns]
    return (
        df[cols]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .T
        .rename_axis("column")
        .reset_index()
    )


def exposure_counts(input_long: pd.DataFrame, by_cols: list[str]) -> pd.DataFrame:
    return (
        input_long.groupby(by_cols)
        .size()
        .rename("exposure_count")
        .reset_index()
        .sort_values("exposure_count", ascending=False)
    )


def exposure_summary(count_df: pd.DataFrame, count_col: str = "exposure_count") -> dict:
    values = count_df[count_col]
    return {
        "n_candidates": int(len(values)),
        "min": int(values.min()) if len(values) else 0,
        "q1": float(values.quantile(0.25)) if len(values) else 0.0,
        "median": float(values.median()) if len(values) else 0.0,
        "q3": float(values.quantile(0.75)) if len(values) else 0.0,
        "max": int(values.max()) if len(values) else 0,
        "mean": float(values.mean()) if len(values) else 0.0,
        "std": float(values.std()) if len(values) else 0.0,
        "cv": float(values.std() / values.mean()) if len(values) and values.mean() else 0.0,
    }


def position_bias_table(rank_long: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    unique_rank = rank_long[rank_long["mapping_status"] == "unique"].copy()
    unique_rank = unique_rank[unique_rank["shown_pos"].notna()].copy()

    if unique_rank.empty:
        empty = pd.DataFrame(columns=["shown_pos", "mapped_appearances", "top1_count", "top1_rate", "avg_rank"])
        return empty, {
            "mapped_rows": 0,
            "unique_top1_rows": 0,
            "ambiguous_top1_rows": 0,
        }

    position = (
        unique_rank.groupby("shown_pos")
        .agg(
            mapped_appearances=("rank", "size"),
            top1_count=("rank", lambda s: int((s == 1).sum())),
            avg_rank=("rank", "mean"),
        )
        .reset_index()
        .sort_values("shown_pos")
    )
    position["top1_rate"] = position["top1_count"] / position["mapped_appearances"]

    summary = {
        "mapped_rows": int(len(unique_rank)),
        "unique_top1_rows": int(len(unique_rank[unique_rank["rank"] == 1])),
        "ambiguous_top1_rows": int(
            ((rank_long["rank"] == 1) & (rank_long["mapping_status"] != "unique")).sum()
        ),
    }
    return position, summary


def top1_item_counts(rank_long: pd.DataFrame) -> pd.DataFrame:
    top1 = rank_long[(rank_long["rank"] == 1) & (rank_long["mapping_status"] == "unique")].copy()
    if top1.empty:
        return pd.DataFrame(columns=["item_id", "trial_brand", "top1_count"])

    return (
        top1.groupby(["item_id", "trial_brand"])
        .size()
        .rename("top1_count")
        .reset_index()
        .sort_values("top1_count", ascending=False)
    )


def engine_agreement_tables(response: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    engines = sorted(response["engine"].dropna().unique().tolist())
    input_cols = ["set_id", "input_sku1", "input_sku2", "input_sku3"]
    input_map = response[input_cols].drop_duplicates("set_id").copy()
    input_map["duplicate_input_brand_trial"] = input_map.apply(has_duplicate_input_brands, axis=1)

    summary_rows: list[dict] = []
    case_rows: list[dict] = []

    for engine_a, engine_b in combinations(engines, 2):
        pair = response[response["engine"].isin([engine_a, engine_b])].copy()
        wide = pair.pivot(index="set_id", columns="engine", values=list(RANK_COLS.values()))
        wide.columns = [f"{col}__{engine}" for col, engine in wide.columns]
        wide = wide.join(input_map.set_index("set_id"), how="left")

        top1_a = wide[f"{RANK_COLS[1]}__{engine_a}"]
        top1_b = wide[f"{RANK_COLS[1]}__{engine_b}"]
        rank2_a = wide[f"{RANK_COLS[2]}__{engine_a}"]
        rank2_b = wide[f"{RANK_COLS[2]}__{engine_b}"]
        rank3_a = wide[f"{RANK_COLS[3]}__{engine_a}"]
        rank3_b = wide[f"{RANK_COLS[3]}__{engine_b}"]

        complete_pair = top1_a.notna() & top1_b.notna() & rank2_a.notna() & rank2_b.notna() & rank3_a.notna() & rank3_b.notna()
        top1_pair = top1_a.notna() & top1_b.notna()
        top1_match = top1_pair & (top1_a == top1_b)
        rank2_match = complete_pair & (rank2_a == rank2_b)
        rank3_match = complete_pair & (rank3_a == rank3_b)
        full_rank_match = complete_pair & top1_match & rank2_match & rank3_match

        pairwise_rates: list[float] = []
        for set_id, row in wide[complete_pair].iterrows():
            inputs = [row["input_sku1"], row["input_sku2"], row["input_sku3"]]
            if len(set(inputs)) < 3:
                continue

            ranks_a = {
                row[f"{RANK_COLS[1]}__{engine_a}"]: 1,
                row[f"{RANK_COLS[2]}__{engine_a}"]: 2,
                row[f"{RANK_COLS[3]}__{engine_a}"]: 3,
            }
            ranks_b = {
                row[f"{RANK_COLS[1]}__{engine_b}"]: 1,
                row[f"{RANK_COLS[2]}__{engine_b}"]: 2,
                row[f"{RANK_COLS[3]}__{engine_b}"]: 3,
            }
            pair_agree = 0
            total_pairs = 0
            for left, right in combinations(inputs, 2):
                total_pairs += 1
                order_a = ranks_a[left] < ranks_a[right]
                order_b = ranks_b[left] < ranks_b[right]
                pair_agree += int(order_a == order_b)
            if total_pairs:
                pairwise_rates.append(pair_agree / total_pairs)

        summary_rows.append(
            {
                "engine_a": engine_a,
                "engine_b": engine_b,
                "paired_set_ids": int(len(wide)),
                "paired_top1_available": int(top1_pair.sum()),
                "paired_complete_rankings": int(complete_pair.sum()),
                "top1_agreement_rate": float(top1_match[top1_pair].mean()) if top1_pair.any() else np.nan,
                "rank2_agreement_rate": float(rank2_match[complete_pair].mean()) if complete_pair.any() else np.nan,
                "rank3_agreement_rate": float(rank3_match[complete_pair].mean()) if complete_pair.any() else np.nan,
                "full_rank_agreement_rate": float(full_rank_match[complete_pair].mean()) if complete_pair.any() else np.nan,
                "pairwise_preference_agreement_rate": float(np.mean(pairwise_rates)) if pairwise_rates else np.nan,
            }
        )

        case_df = pd.DataFrame(
            {
                "engine_a": engine_a,
                "engine_b": engine_b,
                "set_id": wide.index,
                "input_sku1": wide["input_sku1"].values,
                "input_sku2": wide["input_sku2"].values,
                "input_sku3": wide["input_sku3"].values,
                "duplicate_input_brand_trial": wide["duplicate_input_brand_trial"].values,
                f"{engine_a}_top1": top1_a.values,
                f"{engine_a}_rank2": rank2_a.values,
                f"{engine_a}_rank3": rank3_a.values,
                f"{engine_b}_top1": top1_b.values,
                f"{engine_b}_rank2": rank2_b.values,
                f"{engine_b}_rank3": rank3_b.values,
                "top1_match": top1_match.values,
                "full_rank_match": full_rank_match.values,
                "complete_pair": complete_pair.values,
            }
        )
        case_df["agreement_signature"] = case_df.apply(
            lambda row: f"{row[f'{engine_a}_top1']} > {row[f'{engine_a}_rank2']} > {row[f'{engine_a}_rank3']} || "
            f"{row[f'{engine_b}_top1']} > {row[f'{engine_b}_rank2']} > {row[f'{engine_b}_rank3']}",
            axis=1,
        )
        case_rows.append(case_df)

    summary_df = pd.DataFrame(summary_rows)
    cases_df = pd.concat(case_rows, ignore_index=True) if case_rows else pd.DataFrame()
    return summary_df, cases_df


def plot_engine_agreement(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty:
        return

    metrics = ["top1_agreement_rate", "full_rank_agreement_rate", "pairwise_preference_agreement_rate"]
    plot_df = summary_df.melt(
        id_vars=["engine_a", "engine_b"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    plot_df["engine_pair"] = plot_df["engine_a"] + " vs " + plot_df["engine_b"]

    fig, ax = plt.subplots(figsize=(10, 5))
    pairs = plot_df["engine_pair"].unique().tolist()
    x = np.arange(len(pairs))
    width = 0.25

    for idx, metric in enumerate(metrics):
        subset = plot_df[plot_df["metric"] == metric].copy()
        subset = subset.set_index("engine_pair").reindex(pairs).reset_index()
        ax.bar(
            x + (idx - 1) * width,
            subset["value"],
            width=width,
            label=metric,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pairs)
    ax.set_ylim(0, 1)
    ax.set_ylabel("비율")
    ax.set_title("엔진 간 라벨 일치도")
    ax.legend()
    save_figure(fig, out_path)


def plot_position_bias(position_df: pd.DataFrame, title: str, out_path: Path) -> None:
    if position_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(position_df["shown_pos"].astype(str), position_df["top1_rate"], color="darkorange")
    axes[0].set_ylim(0, max(0.4, position_df["top1_rate"].max() * 1.15))
    axes[0].set_title(f"{title}: {display_label('top1_rate')}")
    axes[0].set_xlabel(display_label("shown_pos"))
    axes[0].set_ylabel(display_label("top1_rate"))

    axes[1].bar(position_df["shown_pos"].astype(str), position_df["avg_rank"], color="steelblue")
    axes[1].set_title(f"{title}: {display_label('avg_rank')}")
    axes[1].set_xlabel(display_label("shown_pos"))
    axes[1].set_ylabel(display_label("avg_rank"))

    save_figure(fig, out_path)


def plot_exposure_histogram(count_df: pd.DataFrame, title: str, out_path: Path) -> None:
    if count_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(count_df["exposure_count"], bins=30, color="mediumpurple", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(display_label("exposure_count"))
    ax.set_ylabel("빈도")
    save_figure(fig, out_path)


def plot_engine_overview(
    engine_name: str,
    response_df: pd.DataFrame,
    position_df: pd.DataFrame,
    top1_brand_df: pd.DataFrame,
    top1_item_df: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    axes[0].hist(response_df["response_chars"].dropna(), bins=30, color="teal", edgecolor="white")
    axes[0].set_title(f"{engine_name}: {display_label('response_chars')} 분포")
    axes[0].set_xlabel(display_label("response_chars"))
    axes[0].set_ylabel("빈도")

    if not position_df.empty:
        axes[1].bar(position_df["shown_pos"].astype(str), position_df["top1_rate"], color="darkorange")
        axes[1].set_title(f"{engine_name}: {display_label('shown_pos')}별 {display_label('top1_rate')}")
        axes[1].set_xlabel(display_label("shown_pos"))
        axes[1].set_ylabel(display_label("top1_rate"))

    brand_plot = top1_brand_df.head(15).sort_values("top1_count", ascending=True)
    axes[2].barh(brand_plot["brand_ko"], brand_plot["top1_count"], color="royalblue")
    axes[2].set_title(f"{engine_name}: Top1 브랜드 분포")
    axes[2].set_xlabel(display_label("top1_count"))
    axes[2].set_ylabel(display_label("brand_ko"))

    if not top1_item_df.empty:
        item_plot = top1_item_df.head(15).sort_values("top1_count", ascending=True)
        axes[3].barh(item_plot["trial_brand"], item_plot["top1_count"], color="seagreen")
        axes[3].set_title(f"{engine_name}: Top1 상품 분포")
        axes[3].set_xlabel(display_label("top1_count"))
        axes[3].set_ylabel("상품명 대신 브랜드")

    save_figure(fig, out_path)


def run_overall_diagnostics(
    response: pd.DataFrame,
    input_long: pd.DataFrame,
    rank_long: pd.DataFrame,
    out_dir: Path,
) -> dict:
    ensure_dir(out_dir)

    duplicate_input = (
        response[response.apply(has_duplicate_input_brands, axis=1)]
        .sort_values(["set_id", "engine"])
        .reset_index(drop=True)
    )
    save_csv(duplicate_input, out_dir / "duplicate_input_brand_trials.csv")

    input_brand_mismatch = input_long[~input_long["brand_match"]].copy()
    save_csv(input_brand_mismatch, out_dir / "input_trial_brand_mismatch.csv")

    agreement_summary, agreement_cases = engine_agreement_tables(response)
    save_csv(agreement_summary, out_dir / "engine_agreement_summary.csv")
    save_csv(agreement_cases, out_dir / "engine_agreement_cases.csv")
    disagreement_cases = agreement_cases[agreement_cases["top1_match"] == False].copy()
    save_csv(disagreement_cases, out_dir / "engine_top1_disagreement_cases.csv")
    plot_engine_agreement(agreement_summary, out_dir / "engine_agreement_overview.png")

    position_df, position_summary = position_bias_table(rank_long)
    save_csv(position_df, out_dir / "position_bias_overall.csv")
    plot_position_bias(position_df, "전체 응답", out_dir / "position_bias_overall.png")

    item_exposure = exposure_counts(
        input_long,
        ["item_id", "trial_brand", "resolved_url", "mapping_quality", "is_ambiguous"],
    )
    brand_exposure = exposure_counts(input_long, ["trial_brand"])
    save_csv(item_exposure, out_dir / "item_exposure_counts.csv")
    save_csv(brand_exposure, out_dir / "brand_exposure_counts.csv")
    plot_exposure_histogram(item_exposure, "전체 응답: 상품 노출 횟수 분포", out_dir / "item_exposure_histogram.png")

    top1_items = top1_item_counts(rank_long)
    save_csv(top1_items, out_dir / "top1_item_counts.csv")

    overall_summary = {
        "response_rows": int(len(response)),
        "set_ids": int(response["set_id"].nunique()),
        "engines": sorted(response["engine"].dropna().unique().tolist()),
        "duplicate_input_brand_rows": int(len(duplicate_input)),
        "duplicate_input_brand_set_ids": int(duplicate_input["set_id"].nunique()) if not duplicate_input.empty else 0,
        "input_trial_brand_mismatch_rows": int(len(input_brand_mismatch)),
        "position_bias": position_summary,
        "item_exposure_summary": exposure_summary(item_exposure),
        "brand_exposure_summary": exposure_summary(brand_exposure),
    }
    save_json(overall_summary, out_dir / "summary.json")
    return overall_summary


def run_engine_specific_eda(
    response: pd.DataFrame,
    input_long: pd.DataFrame,
    rank_long: pd.DataFrame,
    out_dir: Path,
) -> dict:
    ensure_dir(out_dir)
    summary: dict[str, dict] = {}

    for engine_name in sorted(response["engine"].dropna().unique().tolist()):
        engine_dir = out_dir / engine_name
        ensure_dir(engine_dir)

        engine_resp = response[response["engine"] == engine_name].copy()
        engine_input = input_long[input_long["engine"] == engine_name].copy()
        engine_rank = rank_long[rank_long["engine"] == engine_name].copy()

        numeric_summary = response_numeric_summary(engine_resp)
        save_csv(numeric_summary, engine_dir / "response_numeric_summary.csv")

        top1_brand_counts = (
            engine_resp["Y2_top1_brand"]
            .value_counts()
            .rename_axis("brand_ko")
            .reset_index(name="top1_count")
        )
        save_csv(top1_brand_counts, engine_dir / "top1_brand_counts.csv")

        position_df, position_summary = position_bias_table(engine_rank)
        save_csv(position_df, engine_dir / "position_bias.csv")
        plot_position_bias(position_df, f"{engine_name} 응답", engine_dir / "position_bias.png")

        item_exposure = exposure_counts(
            engine_input,
            ["item_id", "trial_brand", "resolved_url", "mapping_quality", "is_ambiguous"],
        )
        brand_exposure = exposure_counts(engine_input, ["trial_brand"])
        save_csv(item_exposure, engine_dir / "item_exposure_counts.csv")
        save_csv(brand_exposure, engine_dir / "brand_exposure_counts.csv")
        plot_exposure_histogram(
            item_exposure,
            f"{engine_name}: 상품 노출 횟수 분포",
            engine_dir / "item_exposure_histogram.png",
        )

        top1_items = top1_item_counts(engine_rank)
        save_csv(top1_items, engine_dir / "top1_item_counts.csv")

        plot_engine_overview(
            engine_name,
            engine_resp,
            position_df,
            top1_brand_counts,
            top1_items,
            engine_dir / "engine_overview.png",
        )

        summary[engine_name] = {
            "response_rows": int(len(engine_resp)),
            "set_ids": int(engine_resp["set_id"].nunique()),
            "missing_rank3_rows": int(engine_resp["rank3_brand"].isna().sum()),
            "position_bias": position_summary,
            "item_exposure_summary": exposure_summary(item_exposure),
            "brand_exposure_summary": exposure_summary(brand_exposure),
            "mean_response_chars": float(engine_resp["response_chars"].mean()),
            "mean_cost_usd": float(engine_resp["cost_usd"].mean()),
        }

    save_json(summary, out_dir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Response-level EDA and diagnostics.")
    parser.add_argument(
        "--outdir",
        default=str(ARTIFACT_DIR),
        help="Output directory for response diagnostics artifacts.",
    )
    args = parser.parse_args()

    response = load_response()
    trial_items = load_trial_items()
    trial_lookup = prepare_trial_lookup(trial_items)
    input_long = build_input_long(response, trial_lookup)
    rank_long = build_rank_long(response, trial_lookup)

    out_dir = Path(args.outdir)
    ensure_dir(out_dir)

    overall_summary = run_overall_diagnostics(
        response=response,
        input_long=input_long,
        rank_long=rank_long,
        out_dir=out_dir / "overall",
    )
    by_engine_summary = run_engine_specific_eda(
        response=response,
        input_long=input_long,
        rank_long=rank_long,
        out_dir=out_dir / "by_engine",
    )

    summary = {
        "overall": overall_summary,
        "by_engine": by_engine_summary,
    }
    save_json(summary, out_dir / "summary.json")

    print(f"[saved] {out_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
