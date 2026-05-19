import os
import pickle
import random

import numpy as np
import torch
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import Config
from data import build_kfold_loaders
from model import RecommendationScoreModel
from loss import plackett_luce_loss
from calibration import TemperatureCalibration
from metrics import evaluate_all


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

    def _forward(self, batch):
        feats, ranks, _, _ = batch
        feats = feats.to(self.device)
        ranks = ranks.to(self.device)
        B, K, D = feats.shape
        scores = self.model(feats.view(B * K, D)).view(B, K)
        return scores, ranks

    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            scores, ranks = self._forward(batch)
            loss = plackett_luce_loss(scores, ranks)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def collect_scores(self, loader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_scores, all_ranks = [], []
        for batch in loader:
            scores, ranks = self._forward(batch)
            all_scores.append(scores.cpu())
            all_ranks.append(ranks.cpu())
        return torch.cat(all_scores).numpy(), torch.cat(all_ranks).numpy()

    def fit(self, train_loader, val_loader, ckpt_path: str, fold_idx: int = 0) -> float:
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)

        best_val_loss = float("inf")
        no_improve    = 0
        step_offset   = fold_idx * self.cfg.n_epochs

        for epoch in range(1, self.cfg.n_epochs + 1):
            train_loss = self.train_epoch(train_loader)

            val_scores, val_ranks = self.collect_scores(val_loader)
            val_loss = plackett_luce_loss(
                torch.tensor(val_scores), torch.tensor(val_ranks)
            ).item()

            self.sched.step(val_loss)
            lr = self.optim.param_groups[0]["lr"]

            print(f"Epoch {epoch:3d} | train={train_loss:.4f}  val={val_loss:.4f}  lr={lr:.2e}")
            wandb.log({
                f"fold{fold_idx}/train_loss": train_loss,
                f"fold{fold_idx}/val_loss":   val_loss,
                f"fold{fold_idx}/lr":         lr,
            }, step=step_offset + epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve    = 0
                torch.save(self.model.state_dict(), ckpt_path)
            else:
                no_improve += 1
                if no_improve >= self.cfg.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        self.model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        print(f"Best val loss: {best_val_loss:.4f}")
        return best_val_loss

    def calibrate(self, val_loader) -> TemperatureCalibration:
        val_scores, val_ranks = self.collect_scores(val_loader)
        calib = TemperatureCalibration(self.cfg.temp_candidates)
        calib.fit(torch.tensor(val_scores), torch.tensor(val_ranks))
        print(f"Temperature calibration: T* = {calib.temperature}")
        return calib


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():
    cfg    = Config()
    cfg.version = "v4"
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    engine_tag = cfg.engine_filter or "all"
    run_name   = f"{cfg.protocol}_{cfg.version}_{engine_tag}_mlp"
    input_dim  = len(cfg.feature_cols) + (1 if cfg.use_position_feature else 0)

    wandb.init(
        project="formcleaner-ranker",
        name=run_name,
        config={
            "model":         "mlp",
            "protocol":      cfg.protocol,
            "version":       cfg.version,
            "engine_filter": engine_tag,
            "n_folds":       cfg.n_folds,
            "n_features":    input_dim,
            "hidden_dims":   cfg.hidden_dims,
            "dropout":       cfg.dropout,
            "lr":            cfg.lr,
            "weight_decay":  cfg.weight_decay,
            "batch_size":    cfg.batch_size,
            "n_epochs":      cfg.n_epochs,
            "patience":      cfg.patience,
            "seed":          cfg.seed,
        },
    )

    print(f"Device: {device}  |  Protocol: {cfg.protocol}  |  Version: {cfg.version}  |  Engine: {engine_tag}  |  Folds: {cfg.n_folds}")

    folds, test_loaders, scalers = build_kfold_loaders(cfg)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    fold_trainers: list[Trainer] = []
    fold_calibs:   list[TemperatureCalibration] = []
    fold_val_results: list[dict] = []

    for fold_idx, ((train_loader, val_loader), scaler) in enumerate(zip(folds, scalers)):
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx + 1}/{cfg.n_folds}  — train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}")

        set_seed(cfg.seed + fold_idx)
        model = RecommendationScoreModel(
            input_dim=input_dim,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
            use_batch_norm=cfg.use_batch_norm,
        )
        trainer     = Trainer(model, cfg, device)
        ckpt_path   = os.path.join(cfg.ckpt_dir, f"{cfg.protocol}_{cfg.version}_{engine_tag}_mlp_fold{fold_idx}.pt")
        scaler_path = os.path.join(cfg.ckpt_dir, f"{cfg.protocol}_{cfg.version}_{engine_tag}_mlp_fold{fold_idx}_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        best_val_loss = trainer.fit(train_loader, val_loader, ckpt_path, fold_idx)
        wandb.summary[f"fold{fold_idx}/best_val_loss"] = best_val_loss

        calib = trainer.calibrate(val_loader)
        wandb.summary[f"fold{fold_idx}/temperature"] = calib.temperature

        val_scores, val_ranks = trainer.collect_scores(val_loader)
        val_results = evaluate_all(val_scores, val_ranks)
        wandb.log({f"fold{fold_idx}/val/{k}": v for k, v in val_results.items()})
        print(f"  Val  — " + "  ".join(f"{k}: {v:.4f}" for k, v in val_results.items()))

        fold_trainers.append(trainer)
        fold_calibs.append(calib)
        fold_val_results.append(val_results)

    cv_log = {}
    for metric in fold_val_results[0]:
        vals = [r[metric] for r in fold_val_results]
        cv_log[f"cv/val_{metric}_mean"] = float(np.nanmean(vals))
        cv_log[f"cv/val_{metric}_std"]  = float(np.nanstd(vals))
    wandb.log(cv_log)

    # ---- Ensemble test evaluation ----
    print(f"\n{'='*50}")
    print("=== Ensemble Test Results ===")

    test_scores_per_fold, test_ranks_ref = [], None
    for trainer, test_loader in zip(fold_trainers, test_loaders):
        scores, test_ranks = trainer.collect_scores(test_loader)
        test_scores_per_fold.append(scores)
        test_ranks_ref = test_ranks

    ensemble_scores = np.mean(test_scores_per_fold, axis=0)
    avg_temp        = float(np.mean([c.temperature for c in fold_calibs]))
    wandb.summary["ensemble/temperature"] = avg_temp

    exp_cal    = np.exp(ensemble_scores / avg_temp)
    test_probs = exp_cal / exp_cal.sum(axis=1, keepdims=True)

    results = evaluate_all(ensemble_scores, test_ranks_ref, test_probs)
    for metric, value in results.items():
        print(f"  {metric:<22} {value:.4f}")

    wandb.log({f"test/{k}": v for k, v in results.items()})
    wandb.finish()


if __name__ == "__main__":
    main()
