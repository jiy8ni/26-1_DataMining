import json
import os
import random

import numpy as np
import torch
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data import build_loaders, effective_feature_dim
from model import RecommendationScoreModel
from loss import plackett_luce_loss, hybrid_loss
from calibration import TemperatureCalibration
from metrics import evaluate_all
from preds_io import save_scores


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(
        self,
        model: RecommendationScoreModel,
        cfg: Config,
        device: torch.device,
    ):
        self.model   = model.to(device)
        self.cfg     = cfg
        self.device  = device
        self.optim   = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched   = ReduceLROnPlateau(self.optim, mode="min", patience=5, factor=0.5)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward(self, batch):
        feats, ranks, _, pl_theta = batch
        feats    = feats.to(self.device)      # (B, K, D)
        ranks    = ranks.to(self.device)      # (B, K)
        pl_theta = pl_theta.to(self.device)   # (B, K)
        B, K, D = feats.shape
        scores = self.model(feats.view(B * K, D)).view(B, K)
        return scores, ranks, pl_theta

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            scores, ranks, pl_theta = self._forward(batch)
            loss = hybrid_loss(scores, ranks, pl_theta, self.cfg.lambda_mse)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect_scores(self, loader) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scores, ranks) arrays for an entire DataLoader."""
        self.model.eval()
        all_scores, all_ranks = [], []
        for batch in loader:
            scores, ranks, _ = self._forward(batch)
            all_scores.append(scores.cpu())
            all_ranks.append(ranks.cpu())
        return torch.cat(all_scores).numpy(), torch.cat(all_ranks).numpy()

    def evaluate(self, loader) -> dict:
        scores, ranks = self.collect_scores(loader)
        return evaluate_all(scores, ranks)

    # ------------------------------------------------------------------
    # Fit loop
    # ------------------------------------------------------------------

    def fit(self, train_loader, val_loader, seed_idx: int = 0) -> float:
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        engine_tag = self.cfg.engine_filter or "all"
        ckpt_path  = os.path.join(
            self.cfg.ckpt_dir,
            f"{self.cfg.protocol}_{self.cfg.version}_{engine_tag}_seed{seed_idx}_best.pt",
        )
        step_offset = seed_idx * self.cfg.n_epochs

        best_val_loss = float("inf")
        no_improve    = 0

        for epoch in range(1, self.cfg.n_epochs + 1):
            train_loss = self.train_epoch(train_loader)

            val_scores, val_ranks = self.collect_scores(val_loader)
            val_loss = plackett_luce_loss(
                torch.tensor(val_scores), torch.tensor(val_ranks)
            ).item()   # early stopping tracks PL loss only (no pl_theta needed)

            self.sched.step(val_loss)
            lr = self.optim.param_groups[0]["lr"]

            print(f"[seed {seed_idx}] Epoch {epoch:3d} | train={train_loss:.4f}  val={val_loss:.4f}  lr={lr:.2e}")
            wandb.log({f"seed{seed_idx}/train_loss": train_loss,
                       f"seed{seed_idx}/val_loss": val_loss,
                       f"seed{seed_idx}/lr": lr}, step=step_offset + epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve    = 0
                torch.save(self.model.state_dict(), ckpt_path)
            else:
                no_improve += 1
                if no_improve >= self.cfg.patience:
                    print(f"[seed {seed_idx}] Early stopping at epoch {epoch}.")
                    break

        self.model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        print(f"[seed {seed_idx}] Best val loss: {best_val_loss:.4f}")
        return best_val_loss

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, val_loader) -> TemperatureCalibration:
        """Runs temperature grid search on the validation set."""
        val_scores, val_ranks = self.collect_scores(val_loader)
        calib = TemperatureCalibration(self.cfg.temp_candidates)
        calib.fit(torch.tensor(val_scores), torch.tensor(val_ranks))
        print(f"Temperature calibration: T* = {calib.temperature}")
        wandb.summary["temperature"] = calib.temperature
        return calib


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def apply_tuned_params(cfg: Config) -> None:
    """Override architecture/optim fields from artifacts/tuning/mlp_best_params.json
    if it exists (written by tune_mlp.py). No-op otherwise — cfg keeps its
    regularized defaults."""
    path = os.path.join(cfg.tuning_dir, "mlp_best_params.json")
    if not os.path.exists(path):
        print("No tuned params found — using Config defaults.")
        return
    with open(path) as f:
        p = json.load(f)["params"]
    for key in ("hidden_dims", "dropout", "weight_decay", "lr", "lambda_mse"):
        if key in p:
            setattr(cfg, key, p[key])
    print(f"Loaded tuned MLP params from {path}: "
          f"dims={cfg.hidden_dims} dropout={cfg.dropout} wd={cfg.weight_decay} "
          f"lr={cfg.lr} lambda_mse={cfg.lambda_mse}")


def main():
    cfg    = Config()
    apply_tuned_params(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    engine_tag = cfg.engine_filter or "all"
    run_name   = f"{cfg.protocol}_{cfg.version}_{engine_tag}"

    wandb.init(
        project="formcleaner-ranker",
        name=run_name,
        config={
            "protocol":      cfg.protocol,
            "version":       cfg.version,
            "engine_filter": engine_tag,
            "n_features":    effective_feature_dim(cfg),
            "hidden_dims":   cfg.hidden_dims,
            "dropout":       cfg.dropout,
            "lr":            cfg.lr,
            "weight_decay":  cfg.weight_decay,
            "batch_size":    cfg.batch_size,
            "n_epochs":      cfg.n_epochs,
            "patience":      cfg.patience,
            "seed":          cfg.seed,
            "n_seeds":       cfg.n_seeds,
            "lambda_mse":    cfg.lambda_mse,
        },
    )

    print(f"Device: {device}  |  Protocol: {cfg.protocol}  |  Version: {cfg.version}  |  Engine: {engine_tag}  |  Features: {effective_feature_dim(cfg)}  |  seeds: {cfg.n_seeds}")

    train_loader, val_loader, test_loader, _ = build_loaders(cfg)
    print(
        f"Trials — train: {len(train_loader.dataset)}  "
        f"val: {len(val_loader.dataset)}  "
        f"test: {len(test_loader.dataset)}"
    )

    # ---- Seed ensemble: independent inits/SGD noise give diverse models ----
    val_scores_per_seed, test_scores_per_seed, temps = [], [], []
    val_ranks_ref = test_ranks_ref = None
    for i in range(cfg.n_seeds):
        set_seed(cfg.seed + i)
        model = RecommendationScoreModel(
            input_dim=effective_feature_dim(cfg),
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
            use_batch_norm=cfg.use_batch_norm,
        )
        if i == 0:
            print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        trainer = Trainer(model, cfg, device)
        trainer.fit(train_loader, val_loader, seed_idx=i)

        calib = trainer.calibrate(val_loader)
        val_s,  val_ranks_ref  = trainer.collect_scores(val_loader)
        test_s, test_ranks_ref = trainer.collect_scores(test_loader)

        val_scores_per_seed.append(val_s)
        test_scores_per_seed.append(test_s)
        temps.append(calib.temperature)

    val_scores  = np.mean(val_scores_per_seed,  axis=0)
    test_scores = np.mean(test_scores_per_seed, axis=0)
    avg_temp    = float(np.mean(temps))
    wandb.summary["ensemble/temperature"] = avg_temp

    # ---- Test-set evaluation (ensembled scores, averaged temperature) ----
    print("\n=== Test Set Results ===")
    exp_cal    = np.exp(test_scores / avg_temp)
    test_probs = exp_cal / exp_cal.sum(axis=1, keepdims=True)

    results = evaluate_all(test_scores, test_ranks_ref, test_probs)
    for metric, value in results.items():
        print(f"  {metric:<22} {value:.4f}")

    # ---- Persist scores for blending ----
    save_scores(cfg.preds_dir, "mlp", "val",  val_scores,  val_ranks_ref)
    save_scores(cfg.preds_dir, "mlp", "test", test_scores, test_ranks_ref)

    wandb.log({f"test/{k}": v for k, v in results.items()})
    wandb.finish()


if __name__ == "__main__":
    main()
