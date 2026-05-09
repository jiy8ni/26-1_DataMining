"""
Option C: Pool-level KL Divergence training (leakage-free).

Loss = KL(pl_rec_prob_train || softmax(model scores over train pool))

Key design:
  - PL fitting uses TRAIN data only  → no test info leaks into labels
  - KL loss targets train items only → model must generalise to test via features
  - Evaluation: standard ranking metrics on held-out test triplets
"""
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)
os.chdir(_ROOT_DIR)

from calibration import TemperatureCalibration
from config import Config
from data import build_loaders
from metrics import evaluate_all
from model import RecommendationScoreModel
from pl_fitting import extract_trials, fit_pl
from scipy.special import softmax as scipy_softmax


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Pool data builder
# ---------------------------------------------------------------------------

def build_train_pool(
    cfg: Config,
    scaler: StandardScaler,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Fit PL on TRAIN data only, then build pool tensors for train items.

    No test/val information leaks into the labels or the training pool.

    Returns:
        X_pool   : (N_train, D) float32 — scaled feature matrix
        rec_prob : (N_train,)   float32 — PL rec_prob over train items (sums to 1)
        url_list : list of resolved_url (length N_train)
    """
    train_df = pd.read_csv(f"{cfg.data_dir}/{cfg.protocol}_train_features.csv")
    if cfg.engine_filter:
        train_df = train_df[train_df["engine"] == cfg.engine_filter]
    if "is_ambiguous" in train_df.columns:
        train_df = train_df[~train_df["is_ambiguous"].astype(bool)]
    train_df = train_df.dropna(subset=["resolved_url"])

    # Fit PL on train trials only
    item_ids, trial_indices = extract_trials(train_df, cfg.trial_keys)
    print(f"Train pool: {len(item_ids)} items | {len(trial_indices)} trials")
    theta    = fit_pl(trial_indices, len(item_ids))
    rec_prob = scipy_softmax(theta)

    # Apply same preprocessing as RankingDataset
    log_cols = getattr(cfg, "log_transform_cols", [])
    if log_cols:
        cols = [c for c in log_cols if c in cfg.feature_cols]
        train_df[cols] = np.log1p(train_df[cols].clip(lower=0))
    train_df[cfg.feature_cols] = train_df[cfg.feature_cols].fillna(
        train_df[cfg.feature_cols].median()
    )

    # One row per item, ordered to match PL item_ids
    item_df = (
        train_df.drop_duplicates("resolved_url")
        .set_index("resolved_url")
        .reindex(item_ids)
        .reset_index()
    )

    X = scaler.transform(item_df[cfg.feature_cols].values)

    X_pool_t   = torch.tensor(X, dtype=torch.float32)
    rec_prob_t = torch.tensor(rec_prob, dtype=torch.float32)

    return X_pool_t, rec_prob_t, item_ids


# ---------------------------------------------------------------------------
# KL loss
# ---------------------------------------------------------------------------

def pool_kl_loss(
    model: RecommendationScoreModel,
    X_pool: torch.Tensor,
    rec_prob_target: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """KL(pl_rec_prob || softmax(model scores)) over the full pool."""
    scores   = model(X_pool.to(device))                     # (N,)
    log_pred = F.log_softmax(scores, dim=0)                 # (N,)
    return F.kl_div(log_pred, rec_prob_target.to(device), reduction="sum")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def fit(
    model: RecommendationScoreModel,
    X_pool: torch.Tensor,
    rec_prob: torch.Tensor,
    val_loader,
    cfg: Config,
    device: torch.device,
    ckpt_path: str,
) -> None:
    optim = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = ReduceLROnPlateau(optim, mode="min", patience=5, factor=0.5)

    best_loss = float("inf")
    no_improve = 0

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        optim.zero_grad()
        loss = pool_kl_loss(model, X_pool, rec_prob, device)
        loss.backward()
        optim.step()

        # Val loss: KL on pool (same objective; no separate val split for pool)
        model.eval()
        with torch.no_grad():
            val_loss = pool_kl_loss(model, X_pool, rec_prob, device).item()

        sched.step(val_loss)
        lr = optim.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d} | kl={val_loss:.4f}  lr={lr:.2e}")
        wandb.log({"train/kl_loss": loss.item(), "val/kl_loss": val_loss, "lr": lr}, step=epoch)

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    wandb.summary["best_kl_loss"] = best_loss


# ---------------------------------------------------------------------------
# Collect scores for ranking metrics (reuses test_loader triplet structure)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_scores(model, loader, device):
    model.eval()
    all_scores, all_ranks = [], []
    for feats, ranks, _, _pl in loader:
        feats = feats.to(device)
        B, K, D = feats.shape
        scores = model(feats.view(B * K, D)).view(B, K)
        all_scores.append(scores.cpu())
        all_ranks.append(ranks)
    return torch.cat(all_scores).numpy(), torch.cat(all_ranks).numpy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    engine_tag = cfg.engine_filter or "all"
    run_name   = f"{cfg.protocol}_{cfg.version}_{engine_tag}_pool_kl"

    wandb.init(
        project="formcleaner-ranker",
        name=run_name,
        config={
            "model":         "pool_kl",
            "protocol":      cfg.protocol,
            "version":       cfg.version,
            "engine_filter": engine_tag,
            "n_features":    len(cfg.feature_cols),
            "hidden_dims":   cfg.hidden_dims,
            "dropout":       cfg.dropout,
            "lr":            cfg.lr,
            "weight_decay":  cfg.weight_decay,
            "n_epochs":      cfg.n_epochs,
            "patience":      cfg.patience,
            "seed":          cfg.seed,
        },
    )

    print(f"Device: {device} | Protocol: {cfg.protocol} | Engine: {engine_tag}")

    # Pool-level model has no position feature — disable before building loaders
    cfg.use_position_feature = False
    train_loader, val_loader, test_loader, scaler = build_loaders(cfg)

    # Fit PL on train data only and build train pool tensors
    X_pool, rec_prob, url_list = build_train_pool(cfg, scaler)
    print(f"Train pool: {len(url_list)} items | rec_prob sum: {rec_prob.sum():.4f}")

    model = RecommendationScoreModel(
        input_dim=len(cfg.feature_cols),   # no position feature for pool-level
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
        use_batch_norm=cfg.use_batch_norm,
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model = model.to(device)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.ckpt_dir, f"{cfg.protocol}_{cfg.version}_{engine_tag}_pool_kl_best.pt")

    fit(model, X_pool, rec_prob, val_loader, cfg, device, ckpt_path)

    # Temperature calibration on val triplets
    val_scores, val_ranks = collect_scores(model, val_loader, device)
    calib = TemperatureCalibration(cfg.temp_candidates)
    calib.fit(torch.tensor(val_scores), torch.tensor(val_ranks))
    print(f"Temperature calibration: T* = {calib.temperature}")
    wandb.summary["temperature"] = calib.temperature

    # Test evaluation with ranking metrics
    print("\n=== Test Set Results ===")
    test_scores, test_ranks = collect_scores(model, test_loader, device)

    exp_cal    = np.exp(test_scores / calib.temperature)
    test_probs = exp_cal / exp_cal.sum(axis=1, keepdims=True)

    results = evaluate_all(test_scores, test_ranks, test_probs)
    for metric, value in results.items():
        print(f"  {metric:<22} {value:.4f}")

    wandb.log({f"test/{k}": v for k, v in results.items()})
    wandb.finish()


if __name__ == "__main__":
    main()
