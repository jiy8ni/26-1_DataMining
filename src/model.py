import torch
import torch.nn as nn
from typing import List


class RecommendationScoreModel(nn.Module):
    """
    Item-level MLP: feature vector -> scalar recommendation score.

    s_i = f_theta(x_i)

    Called on (B*K, D) in training (where K=3 candidates per trial) and on
    (N, D) at inference time (the full pool).  Trial-level grouping lives in
    the training loop, not here.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, D)
        returns : (B,)  — unbounded real-valued scores
        """
        return self.net(x).squeeze(-1)
