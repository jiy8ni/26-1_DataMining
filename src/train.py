import os
import random

import numpy as np
import torch
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import Config
from data import build_loaders
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward(self, batch):
        feats, ranks, _ = batch
        feats = feats.to(self.device)   # (B, K, D)
        ranks = ranks.to(self.device)   # (B, K)
        B, K, D = feats.shape
        scores = self.model(feats.view(B * K, D)).view(B, K)
        return scores, ranks

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect_scores(self, loader) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scores, ranks) arrays for an entire DataLoader."""
        self.model.eval()
        all_scores, all_ranks = [], []
        for batch in loader:
            scores, ranks = self._forward(batch)
            all_scores.append(scores.cpu())
            all_ranks.append(ranks.cpu())
        return torch.cat(all_scores).numpy(), torch.cat(all_ranks).numpy()

    def evaluate(self, loader) -> dict:
        scores, ranks = self.collect_scores(loader)
        return evaluate_all(scores, ranks)

    # ------------------------------------------------------------------
    # Fit loop
    # ------------------------------------------------------------------

    def fit(self, train_loader, val_loader) -> None:
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(self.cfg.ckpt_dir, f"{self.cfg.protocol}_best.pt")

        best_val_loss = float("inf")
        no_improve    = 0

        for epoch in range(1, self.cfg.n_epochs + 1):
            train_loss = self.train_epoch(train_loader)

            val_scores, val_ranks = self.collect_scores(val_loader)
            val_loss = plackett_luce_loss(
                torch.tensor(val_scores), torch.tensor(val_ranks)
            ).item()

            self.sched.step(val_loss)
            lr = self.optim.param_groups[0]["lr"]

            print(f"Epoch {epoch:3d} | train={train_loss:.4f}  val={val_loss:.4f}  lr={lr:.2e}")
            wandb.log({"train/loss": train_loss, "val/loss": val_loss, "lr": lr}, step=epoch)

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
        wandb.summary["best_val_loss"] = best_val_loss
        print(f"Best val loss: {best_val_loss:.4f}")

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

def main():
    cfg    = Config()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="formcleaner-ranker",
        name=cfg.protocol,
        config={
            "protocol":    cfg.protocol,
            "hidden_dims": cfg.hidden_dims,
            "dropout":     cfg.dropout,
            "lr":          cfg.lr,
            "weight_decay": cfg.weight_decay,
            "batch_size":  cfg.batch_size,
            "n_epochs":    cfg.n_epochs,
            "patience":    cfg.patience,
            "seed":        cfg.seed,
        },
    )

    print(f"Device: {device}  |  Protocol: {cfg.protocol}")

    train_loader, val_loader, test_loader, scaler = build_loaders(cfg)
    print(
        f"Trials — train: {len(train_loader.dataset)}  "
        f"val: {len(val_loader.dataset)}  "
        f"test: {len(test_loader.dataset)}"
    )

    model = RecommendationScoreModel(
        input_dim=len(cfg.feature_cols),
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
        use_batch_norm=cfg.use_batch_norm,
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(model, cfg, device)
    trainer.fit(train_loader, val_loader)

    calib = trainer.calibrate(val_loader)

    # ---- Test-set evaluation ----
    print("\n=== Test Set Results ===")
    test_scores, test_ranks = trainer.collect_scores(test_loader)

    exp_cal = np.exp(test_scores / calib.temperature)
    test_probs = exp_cal / exp_cal.sum(axis=1, keepdims=True)

    results = evaluate_all(test_scores, test_ranks, test_probs)
    for metric, value in results.items():
        print(f"  {metric:<22} {value:.4f}")

    wandb.log({f"test/{k}": v for k, v in results.items()})
    wandb.finish()


if __name__ == "__main__":
    main()
