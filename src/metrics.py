import numpy as np
import torch
from typing import Optional

from loss import plackett_luce_loss


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def top1_accuracy(scores: np.ndarray, true_ranks: np.ndarray) -> float:
    """Fraction of trials where the highest-scored item is AI's rank-1 pick."""
    pred_top1 = scores.argmax(axis=1)
    true_top1 = true_ranks.argmin(axis=1)   # rank 1 is the smallest integer
    return float((pred_top1 == true_top1).mean())


def pairwise_accuracy(scores: np.ndarray, true_ranks: np.ndarray) -> float:
    """
    Fraction of item pairs (i < j) where model and AI agree on preference direction.
    Vectorised over the batch; loops only over the C(K,2) pairs.
    """
    B, K = scores.shape
    correct = 0
    total   = 0
    for i in range(K):
        for j in range(i + 1, K):
            model_prefers_i = scores[:, i] > scores[:, j]
            ai_prefers_i    = true_ranks[:, i] < true_ranks[:, j]
            correct += int((model_prefers_i == ai_prefers_i).sum())
            total   += B
    return correct / total if total > 0 else 0.0


def ndcg_at_k(scores: np.ndarray, true_ranks: np.ndarray, k: int = 3) -> float:
    """
    NDCG@k.  Relevance = K + 1 - true_rank  (rank-1 item gets relevance K).
    """
    K = scores.shape[1]
    relevance = K + 1 - true_ranks                          # (B, K)

    pred_order  = scores.argsort(axis=1)[:, ::-1]          # descending score
    ideal_order = relevance.argsort(axis=1)[:, ::-1]       # descending relevance

    def dcg(rel, order, top_k):
        B_ = rel.shape[0]
        idx    = order[:, :top_k]                           # (B, k)
        gains  = rel[np.arange(B_)[:, None], idx]          # (B, k)
        discounts = 1.0 / np.log2(np.arange(1, top_k + 1) + 1)
        return (gains * discounts).sum(axis=1)              # (B,)

    dcg_scores  = dcg(relevance, pred_order,  k)
    idcg_scores = dcg(relevance, ideal_order, k)
    idcg_scores = np.where(idcg_scores == 0, 1.0, idcg_scores)  # avoid div-by-zero
    return float((dcg_scores / idcg_scores).mean())


def kendall_tau(scores: np.ndarray, true_ranks: np.ndarray) -> float:
    """Mean Kendall tau between predicted score ordering and AI rank ordering."""
    from scipy.stats import kendalltau as _kt
    taus = [_kt(-scores[b], true_ranks[b]).statistic for b in range(len(scores))]
    return float(np.mean(taus))


# ---------------------------------------------------------------------------
# Probability / calibration metrics
# ---------------------------------------------------------------------------

def nll_score(probs: np.ndarray, true_ranks: np.ndarray) -> float:
    """
    Plackett-Luce NLL using calibrated probabilities (log-space scores).
    probs : (B, K) softmax probabilities summing to 1 along axis=1.
    """
    log_scores = torch.tensor(np.log(probs + 1e-12), dtype=torch.float32)
    ranks      = torch.tensor(true_ranks, dtype=torch.long)
    return float(plackett_luce_loss(log_scores, ranks).item())


def brier_score(probs: np.ndarray, true_ranks: np.ndarray) -> float:
    """
    Brier score treating top-1 selection as a multinomial event.
    one_hot[i, j] = 1 if item j is rank-1 in trial i.
    """
    one_hot = (true_ranks == 1).astype(float)
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def evaluate_all(
    scores: np.ndarray,
    true_ranks: np.ndarray,
    probs: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute all ranking metrics (and optionally probability metrics).

    Args:
        scores     : (B, K) raw model scores
        true_ranks : (B, K) AI ranks, 1=best
        probs      : (B, K) calibrated softmax probabilities (optional)
    """
    result = {
        "top1_accuracy":    top1_accuracy(scores, true_ranks),
        "pairwise_accuracy": pairwise_accuracy(scores, true_ranks),
        "ndcg@3":           ndcg_at_k(scores, true_ranks, k=3),
        "kendall_tau":      kendall_tau(scores, true_ranks),
    }
    if probs is not None:
        result["nll"]         = nll_score(probs, true_ranks)
        result["brier_score"] = brier_score(probs, true_ranks)
    return result
