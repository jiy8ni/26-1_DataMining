import torch
import torch.nn.functional as F


def plackett_luce_loss(scores: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
    """
    Plackett-Luce negative log-likelihood (ListMLE).

    For an observed ranking sigma (rank 1 = best), the probability is:

        P(sigma) = prod_{k=1}^{K-1}  exp(s_{sigma_k}) / sum_{j>=k} exp(s_{sigma_j})

    The last position is deterministic (only one item left), so the product
    runs to K-1.  NLL = -log P(sigma).

    Args:
        scores : (B, K)  model scores (unbounded reals)
        ranks  : (B, K)  AI-assigned ranks, integer, 1 = best / K = worst

    Returns:
        Scalar mean NLL over the batch.
    """
    # Reorder scores so index 0 = rank-1 item, index 1 = rank-2 item, ...
    sort_idx     = torch.argsort(ranks, dim=1)          # (B, K)
    sorted_scores = scores.gather(1, sort_idx)          # (B, K), best-first

    # Reverse-cumulative logsumexp:
    #   log_cumsum[:, k] = log( sum_{j >= k} exp(sorted_scores[:, j]) )
    flipped      = sorted_scores.flip(dims=[1])
    log_cumsum   = torch.logcumsumexp(flipped, dim=1).flip(dims=[1])  # (B, K)

    # NLL contribution at position k:  -(s_k - log_cumsum_k)
    # Exclude last position (k = K-1) — deterministic, contributes 0.
    nll = -(sorted_scores[:, :-1] - log_cumsum[:, :-1]).sum(dim=1)   # (B,)
    return nll.mean()


def hybrid_loss(
    scores: torch.Tensor,
    ranks: torch.Tensor,
    pl_theta: torch.Tensor,
    lambda_mse: float,
) -> torch.Tensor:
    """
    PL ranking loss + MSE regression toward PL-fitted theta.

    scores   : (B, K)  model scores
    ranks    : (B, K)  AI ranks, 1=best
    pl_theta : (B, K)  PL-fitted log-strength (ground truth)
    """
    rank_loss = plackett_luce_loss(scores, ranks)
    mse       = F.mse_loss(scores, pl_theta)
    return rank_loss + lambda_mse * mse
