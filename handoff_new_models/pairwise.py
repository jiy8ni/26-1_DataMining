"""Pairwise adapter shared by the four new ranking models.

All four models (RankSVM, Random Forest, Logistic Regression, EBM) follow the
same recipe from the PDF spec:

    input  : feature difference  dX = X_i - X_j
    target : y = 1 if item i ranks ABOVE item j (lower ai_rank), else 0
    predict: score(i) = sum_{j != i} P(i beats j),  then sort within the trial

The rest of the pipeline (temperature calibration, metrics, preds_io, blending)
is reused unchanged from the original project. This module only does the two
conversions that turn the item-level arrays from ``build_arrays`` /
``build_kfold_arrays`` into pairwise form and back into per-trial (B, 3) scores.

Key facts about the upstream arrays (see data.py):
    * ``X`` is (N_items, D); items come in consecutive groups of 3 = one trial,
      already sorted by sku_pos.
    * ``relevance`` is (N_items,) = ``4 - ai_rank`` (higher = better).
      => item-level ai_rank is recovered as ``4 - relevance``.
    * ``ranks_2d`` is (N_trials, 3), the original ai_rank values (1 = best).
"""
import numpy as np

K = 3  # candidates per trial (fixed in this dataset)

# All ordered pairs (a, b), a != b, within a 3-item trial -> 6 pairs per trial.
# Using ordered pairs (both directions) keeps the +/- sign symmetry the PDF
# asks for via fit_intercept=False: every dX appears with its negation -dX.
_ORDERED_PAIRS = [(a, b) for a in range(K) for b in range(K) if a != b]


def ranks_from_relevance(relevance: np.ndarray) -> np.ndarray:
    """Recover item-level ai_rank (1=best) from the LightGBM relevance array.

    ``relevance = 4 - ai_rank`` (data.py), so ``ai_rank = 4 - relevance``.
    Returns an (N_trials, 3) integer array.
    """
    ai_rank = (4 - np.asarray(relevance)).round().astype(int)
    return ai_rank.reshape(-1, K)


def make_pairwise_dataset(X_items: np.ndarray, item_ranks_2d: np.ndarray):
    """Build the pairwise training matrix from item-level features + ranks.

    Args:
        X_items       : (N_trials*3, D) scaled feature matrix (consecutive triples).
        item_ranks_2d : (N_trials, 3) ai_rank per item (1=best). Use
                        ``ranks_from_relevance(relevance)`` to get this for a
                        train fold where only relevance is available.

    Returns:
        dX : (N_trials*6, D) feature differences X[a] - X[b]
        y  : (N_trials*6,)  1 if item a ranks above item b (rank[a] < rank[b]) else 0
    """
    X = np.asarray(X_items, dtype=np.float32).reshape(-1, K, X_items.shape[1])  # (B,3,D)
    ranks = np.asarray(item_ranks_2d).reshape(-1, K)                            # (B,3)
    assert X.shape[0] == ranks.shape[0], "trial count mismatch between X and ranks"

    dX_blocks, y_blocks = [], []
    for a, b in _ORDERED_PAIRS:
        dX_blocks.append(X[:, a, :] - X[:, b, :])                # (B, D)
        y_blocks.append((ranks[:, a] < ranks[:, b]).astype(np.int64))  # (B,)
    dX = np.concatenate(dX_blocks, axis=0)
    y  = np.concatenate(y_blocks,  axis=0)
    return dX, y


def score_items_from_pairwise(prob_fn, X_items: np.ndarray) -> np.ndarray:
    """Aggregate pairwise win scores into per-trial item scores.

    Args:
        prob_fn  : callable mapping a (M, D) batch of feature differences to an
                   (M,) array of "i beats j" scores. For probability models pass
                   ``lambda d: clf.predict_proba(d)[:, 1]``; for RankSVM pass
                   ``clf.decision_function``.
        X_items  : (N_trials*3, D) scaled feature matrix.

    Returns:
        scores : (N_trials, 3) raw scores, score(a) = sum_{b != a} prob_fn(X[a]-X[b]).
                 Treated as raw scores downstream (temperature softmax applied later).
    """
    X = np.asarray(X_items, dtype=np.float32).reshape(-1, K, X_items.shape[1])  # (B,3,D)
    B = X.shape[0]
    scores = np.zeros((B, K), dtype=np.float64)
    for a, b in _ORDERED_PAIRS:
        scores[:, a] += np.asarray(prob_fn(X[:, a, :] - X[:, b, :]), dtype=np.float64)
    return scores
