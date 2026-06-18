"""Brand-CV hyperparameter selection for the vanilla MLP ranker."""
import copy
import os as _os, sys as _sys
import random

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import Config
from data import build_kfold_loaders, effective_feature_dim
from loss import hybrid_loss, plackett_luce_loss
from mlp.model import RecommendationScoreModel
from tune.cv_common import aggregate_candidate, eval_fold, grid_candidates, select_and_save
from tune.runtime import (
    apply_saved_semantic_config,
    apply_smoke_overrides,
    parse_tuner_args,
    smoke_candidates,
)

GRID = {
    "hidden_dims": [[64, 32], [96, 48, 24], [128, 64, 32]],
    "dropout": [0.1, 0.3, 0.5],
    "weight_decay": [1e-4, 1e-3],
    "lr": [5e-4, 1e-3],
}
FIXED = {"lambda_mse": 0.5}


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
        batch_size, group_size, width = feats.shape
        all_s.append(model(feats.view(batch_size * group_size, width)).view(batch_size, group_size).cpu())
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
            feats = feats.to(device)
            ranks = ranks.to(device)
            pl_theta = pl_theta.to(device)
            batch_size, group_size, width = feats.shape
            scores = model(feats.view(batch_size * group_size, width)).view(batch_size, group_size)
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
    args = parse_tuner_args("Brand-CV hyperparameter selection for the vanilla MLP ranker.")
    cfg = Config()
    if args.smoke:
        apply_smoke_overrides(cfg)
    apply_saved_semantic_config(cfg, smoke=args.smoke)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds, _test_loaders, _scalers = build_kfold_loaders(cfg)

    candidates = [{**FIXED, **g} for g in grid_candidates(GRID)]
    if args.smoke:
        candidates = smoke_candidates(candidates, limit=3)
    print(
        f"MLP brand-CV tuning {'[smoke] ' if args.smoke else ''}"
        f"x {len(candidates)} candidates x {cfg.n_folds} folds  (device={device})"
    )

    candidate_means = []
    for ci, cand in enumerate(candidates):
        fold_results = []
        for train_loader, val_loader in folds:
            vs, vr = _train_fold(cfg, cand, train_loader, val_loader, device)
            fold_results.append(eval_fold(vs, vr, cfg.temp_candidates))
        means = aggregate_candidate(fold_results)
        candidate_means.append(means)
        print(
            f"  [{ci+1:2d}/{len(candidates)}] "
            f"dims={cand['hidden_dims']} do={cand['dropout']} "
            f"wd={cand['weight_decay']:.0e} lr={cand['lr']:.0e} | "
            f"top1={means['top1_accuracy']:.4f} "
            f"k?={means['kendall_tau']:.4f} nll={means['nll']:.4f}"
        )

    select_and_save("mlp", candidates, candidate_means, cfg.tuning_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
