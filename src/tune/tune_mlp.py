"""Brand-CV hyperparameter selection for the vanilla MLP ranker.

Coarse grid over capacity / regularization (the original [128,64,32] over-fits
~259 unique items). Self-contained train loop (no wandb) mirroring mlp/train.py:
hybrid loss for training, val PL loss for early stopping. Writes
artifacts/tuning/mlp_best_params.json.

Run:  python src/tune/tune_mlp.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import copy
import random

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import Config
from data import build_kfold_loaders, effective_feature_dim
from mlp.model import RecommendationScoreModel
from loss import plackett_luce_loss, hybrid_loss
from tune.cv_common import grid_candidates, eval_fold, aggregate_candidate, select_and_save


# 18 candidates.
GRID = {
    "hidden_dims":  [[64, 32], [32, 16], [128, 64, 32]],
    "dropout":      [0.1, 0.3, 0.5],
    "weight_decay": [1e-4, 1e-3],
}
FIXED = {"lr": 1e-3, "lambda_mse": 0.5}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _collect(model, loader, device):
    model.eval()
    all_s, all_r = [], []
    for feats, ranks, _, _ in loader:
        feats = feats.to(device)
        B, K, D = feats.shape
        all_s.append(model(feats.view(B * K, D)).view(B, K).cpu())
        all_r.append(ranks)
    return torch.cat(all_s).numpy(), torch.cat(all_r).numpy()


def _train_fold(cfg, cand, train_loader, val_loader, device):
    _set_seed(cfg.seed)
    model = RecommendationScoreModel(
        input_dim=effective_feature_dim(cfg),
        hidden_dims=cand["hidden_dims"],
        dropout=cand["dropout"],
        use_batch_norm=cfg.use_batch_norm,
    ).to(device)
    optim = Adam(model.parameters(), lr=cand["lr"], weight_decay=cand["weight_decay"])
    sched = ReduceLROnPlateau(optim, mode="min", patience=5, factor=0.5)

    best_loss, best_state, no_improve = float("inf"), None, 0
    for _epoch in range(cfg.n_epochs):
        model.train()
        for feats, ranks, _, pl_theta in train_loader:
            feats    = feats.to(device)
            ranks    = ranks.to(device)
            pl_theta = pl_theta.to(device)
            B, K, D  = feats.shape
            scores   = model(feats.view(B * K, D)).view(B, K)
            loss = hybrid_loss(scores, ranks, pl_theta, cand["lambda_mse"])
            optim.zero_grad()
            loss.backward()
            optim.step()

        vs, vr = _collect(model, val_loader, device)
        vloss = plackett_luce_loss(torch.tensor(vs), torch.tensor(vr)).item()
        sched.step(vloss)
        if vloss < best_loss:
            best_loss, best_state, no_improve = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return _collect(model, val_loader, device)


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds, _test_loaders, _scalers = build_kfold_loaders(cfg)

    candidates = [{**FIXED, **g} for g in grid_candidates(GRID)]
    print(f"MLP brand-CV tuning — {len(candidates)} candidates x {cfg.n_folds} folds  (device={device})")

    candidate_means = []
    for ci, cand in enumerate(candidates):
        fold_results = []
        for train_loader, val_loader in folds:
            vs, vr = _train_fold(cfg, cand, train_loader, val_loader, device)
            fold_results.append(eval_fold(vs, vr, cfg.temp_candidates))
        means = aggregate_candidate(fold_results)
        candidate_means.append(means)
        print(f"  [{ci+1:2d}/{len(candidates)}] dims={cand['hidden_dims']} "
              f"do={cand['dropout']} wd={cand['weight_decay']:.0e} | "
              f"top1={means['top1_accuracy']:.4f} kτ={means['kendall_tau']:.4f} nll={means['nll']:.4f}")

    select_and_save("mlp", candidates, candidate_means, cfg.tuning_dir)


if __name__ == "__main__":
    main()
