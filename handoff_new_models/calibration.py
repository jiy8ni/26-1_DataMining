import torch
from typing import List

from loss import plackett_luce_loss


class TemperatureCalibration:
    """
    Post-hoc temperature scaling for the pool-level recommendation probability.

        P(i | pool) = softmax(scores / T)[i]

    T* is found by grid search minimizing the Plackett-Luce NLL on the
    validation set.  The score model itself is never retrained.

    Usage:
        calib = TemperatureCalibration(cfg.temp_candidates)
        calib.fit(val_scores_tensor, val_ranks_tensor)
        calibrated_scores = calib.calibrate(raw_scores)
    """

    def __init__(self, temp_candidates: List[float]):
        self.temp_candidates = temp_candidates
        self.temperature: float = 1.0

    def fit(self, all_scores: torch.Tensor, all_ranks: torch.Tensor) -> None:
        """
        Args:
            all_scores : (N_trials, K) pre-computed raw scores on the validation set
            all_ranks  : (N_trials, K) corresponding AI ranks (1=best)
        """
        best_nll = float("inf")
        best_T   = 1.0
        for T in self.temp_candidates:
            nll = plackett_luce_loss(all_scores / T, all_ranks).item()
            if nll < best_nll:
                best_nll = nll
                best_T   = T
        self.temperature = best_T

    def calibrate(self, scores: torch.Tensor) -> torch.Tensor:
        return scores / self.temperature
