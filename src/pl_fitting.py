"""
Plackett-Luce MLE fitting over the full dataset (train+val+test).

Outputs a CSV with one row per product:
    resolved_url, brand_ko, pl_theta, rec_prob, pool_rank, top_pct, split
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)
os.chdir(_ROOT_DIR)   # make relative paths in Config work

from config import Config


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_splits(data_dir: str, protocol: str, engine_filter: str | None) -> pd.DataFrame:
    dfs = []
    for split in ["train", "val", "test"]:
        df = pd.read_csv(f"{data_dir}/{protocol}_{split}_features.csv")
        df["split"] = split
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    if engine_filter is not None:
        df = df[df["engine"] == engine_filter]
    if "is_ambiguous" in df.columns:
        df = df[~df["is_ambiguous"].astype(bool)]
    df = df.dropna(subset=["resolved_url"])
    return df


def extract_trials(
    df: pd.DataFrame,
    trial_keys: list[str],
    id_col: str = "resolved_url",
) -> tuple[list[str], np.ndarray]:
    """
    Returns:
        item_ids      : list of unique item identifiers (length N)
        trial_indices : (T, 3) int array — [rank1_idx, rank2_idx, rank3_idx]
    """
    item_ids = df[id_col].unique().tolist()
    item_index = {item: i for i, item in enumerate(item_ids)}

    trial_indices = []
    for _, group in df.groupby(trial_keys, sort=False):
        if len(group) != 3:
            continue
        ordered = group.sort_values("ai_rank")
        urls = ordered[id_col].tolist()
        if all(u in item_index for u in urls):
            trial_indices.append([item_index[u] for u in urls])

    return item_ids, np.array(trial_indices, dtype=np.int32)


# ---------------------------------------------------------------------------
# Plackett-Luce MLE (vectorised L-BFGS-B)
# ---------------------------------------------------------------------------

def fit_pl(trial_indices: np.ndarray, n_items: int, l2: float = 1.0) -> np.ndarray:
    """
    Maximise the regularised Plackett-Luce log-likelihood for K=3 rankings.

    For a trial ordered [i0 (rank-1), i1 (rank-2), i2 (rank-3)]:
        log P = [theta_i0 - logsumexp(theta_i0,i1,i2)]
              + [theta_i1 - logsumexp(theta_i1,i2)]
    Regulariser: + 0.5 * l2 * ||theta||^2  (prevents score explosion)

    Returns theta (N,), zero-meaned.  rec_prob = softmax(theta).
    """
    i0, i1, i2 = trial_indices[:, 0], trial_indices[:, 1], trial_indices[:, 2]

    def neg_ll_grad(theta: np.ndarray):
        grad = np.zeros(n_items)

        # Stage 1: rank-1 chosen from all 3
        t3 = np.stack([theta[i0], theta[i1], theta[i2]], axis=1)   # (T, 3)
        lse3 = logsumexp(t3, axis=1)                                # (T,)
        loss = (lse3 - theta[i0]).sum()
        s3 = np.exp(t3 - lse3[:, None])                            # (T, 3) softmax
        np.add.at(grad, i0, s3[:, 0] - 1)
        np.add.at(grad, i1, s3[:, 1])
        np.add.at(grad, i2, s3[:, 2])

        # Stage 2: rank-2 chosen from {rank-2, rank-3}
        t2 = np.stack([theta[i1], theta[i2]], axis=1)              # (T, 2)
        lse2 = logsumexp(t2, axis=1)                                # (T,)
        loss += (lse2 - theta[i1]).sum()
        s2 = np.exp(t2 - lse2[:, None])                            # (T, 2) softmax
        np.add.at(grad, i1, s2[:, 0] - 1)
        np.add.at(grad, i2, s2[:, 1])

        # L2 regularisation
        loss += 0.5 * l2 * (theta ** 2).sum()
        grad += l2 * theta

        return float(loss), grad

    result = minimize(
        neg_ll_grad,
        x0=np.zeros(n_items),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-8},
    )
    if not result.success:
        print(f"[warn] optimiser: {result.message}")

    theta = result.x - result.x.mean()
    return theta


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cfg = Config()
    engine_tag = cfg.engine_filter or "all"
    print(f"Protocol: {cfg.protocol} | Engine: {engine_tag}")

    df = load_all_splits(cfg.data_dir, cfg.protocol, cfg.engine_filter)
    print(f"Rows after filter: {len(df)}")

    item_ids, trial_indices = extract_trials(df, cfg.trial_keys)
    print(f"Unique items: {len(item_ids)} | Valid trials: {len(trial_indices)}")

    theta = fit_pl(trial_indices, len(item_ids))
    rec_prob = softmax(theta)

    # Attach brand info (take first occurrence per url)
    meta = (
        df[["resolved_url", "brand_ko", "split"]]
        .drop_duplicates("resolved_url")
        .set_index("resolved_url")
    )

    result = (
        pd.DataFrame({"resolved_url": item_ids, "pl_theta": theta, "rec_prob": rec_prob})
        .join(meta, on="resolved_url")
        .sort_values("pl_theta", ascending=False)
        .reset_index(drop=True)
    )
    result["pool_rank"] = np.arange(1, len(result) + 1)
    result["top_pct"]   = result["pool_rank"] / len(result) * 100

    print("\n=== Top 10 ===")
    print(result.head(10)[["brand_ko", "pl_theta", "rec_prob", "pool_rank", "top_pct"]].to_string(index=False))
    print("\n=== Bottom 5 ===")
    print(result.tail(5)[["brand_ko", "pl_theta", "rec_prob", "pool_rank", "top_pct"]].to_string(index=False))

    out_path = f"{cfg.data_dir}/pl_labels_{cfg.protocol}_{engine_tag}.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
