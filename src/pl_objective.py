"""
Vectorised Plackett-Luce gradient and hessian for tree-based models (K=3 fixed).

Derivation (K items, observed ranking π where π(k) = item ranked k):

  L = -Σ_{k=1}^{K-1} [s_{π(k)} - log Z_k]    (last position contributes 0)
  Z_k = Σ_{j : rank_j ≥ k} exp(s_j)

  ∂L/∂s_i   = -1_{rank_i < K}  +  e_i · Σ_{m=1}^{rank_i} 1/Z_m
  ∂²L/∂s_i² = e_i · (A_i - e_i · B_i)          (≥ 0, see note below)

  where  e_i    = exp(s_i - max_s)   (shifted for numerical stability)
         A_i    = Σ_{m=1}^{rank_i} 1/Z_m
         B_i    = Σ_{m=1}^{rank_i} 1/Z_m²
  and Z_m is also computed on the shifted scale (Z'_m = Σ exp(s_j - max_s)).

  The hessian is guaranteed ≥ 0 because Z_m ≥ e_i for all m ≤ rank_i
  (item i is included in every such Z_m).
"""

import numpy as np


def pl_grad_hess(
    preds: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    preds  : (N,)  raw model scores (flat, all groups concatenated)
    labels : (N,)  relevance = 4 - ai_rank  (3=best, 1=worst)
    groups : (G,)  group sizes (all expected to be 3)

    Returns
    -------
    grads    : (N,)
    hessians : (N,)  ≥ 0
    """
    all_g3 = np.all(groups == 3)

    if all_g3:
        return _pl_grad_hess_k3(preds, labels, len(groups))

    # Fallback: generic group-by-group loop (handles non-uniform sizes)
    grads    = np.empty_like(preds)
    hessians = np.empty_like(preds)
    offset = 0
    for g in groups:
        s = preds[offset : offset + g]
        r = np.round(4 - labels[offset : offset + g]).astype(int)
        _fill_group(s, r, g, grads, hessians, offset)
        offset += g
    return grads, hessians


# ---------------------------------------------------------------------------
# Vectorised K=3 path
# ---------------------------------------------------------------------------

def _pl_grad_hess_k3(
    preds: np.ndarray,
    labels: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray]:
    B = n_groups
    s = preds.reshape(B, 3)
    r = np.round(4 - labels.reshape(B, 3)).astype(int)   # (B, 3), values 1/2/3

    # Numerically stable exps
    s_max = s.max(axis=1, keepdims=True)
    e = np.exp(s - s_max)                                # (B, 3)

    # Sort each row best-first by rank
    order    = np.argsort(r, axis=1)                     # (B, 3)
    e_sorted = np.take_along_axis(e, order, axis=1)      # (B, 3)

    # Z[b, k] = Σ_{j ≥ k} e_sorted[b, j]  (reverse cumsum)
    Z = np.cumsum(e_sorted[:, ::-1], axis=1)[:, ::-1].copy()  # (B, 3)

    inv_Z      = 1.0 / Z
    inv_Z2     = inv_Z * inv_Z
    inv_Z_cum  = np.cumsum(inv_Z,  axis=1)               # (B, 3)
    inv_Z2_cum = np.cumsum(inv_Z2, axis=1)               # (B, 3)

    # For item with rank r[b,i], look up cumsum at index r[b,i]-1
    r_idx = r - 1                                        # 0-indexed, (B, 3)
    A = np.take_along_axis(inv_Z_cum,  r_idx, axis=1)   # (B, 3)
    C = np.take_along_axis(inv_Z2_cum, r_idx, axis=1)   # (B, 3)

    last_mask = (r == 3).astype(np.float64)
    grads    = -(1.0 - last_mask) + e * A               # (B, 3)
    hessians = np.maximum(e * (A - e * C), 1e-6)        # (B, 3), ≥ 0

    return grads.reshape(-1), hessians.reshape(-1)


# ---------------------------------------------------------------------------
# Generic per-group fallback
# ---------------------------------------------------------------------------

def _fill_group(
    s: np.ndarray,
    r: np.ndarray,
    g: int,
    grads: np.ndarray,
    hessians: np.ndarray,
    offset: int,
) -> None:
    e = np.exp(s - s.max())
    order    = np.argsort(r)
    e_sorted = e[order]

    Z         = np.cumsum(e_sorted[::-1])[::-1].copy()
    inv_Z_cum = np.cumsum(1.0 / Z)
    inv_Z2_cum = np.cumsum(1.0 / (Z * Z))

    for i in range(g):
        k  = r[i] - 1
        A  = inv_Z_cum[k]
        C_ = inv_Z2_cum[k]
        grads[offset + i]    = (0.0 if r[i] == g else -1.0) + e[i] * A
        hessians[offset + i] = max(e[i] * (A - e[i] * C_), 1e-6)
