"""Shared CV-tuning and seed-ensemble training loops for the pairwise models.

Each model script only supplies:
    * ``make_model(cand, seed)`` -> an unfitted sklearn-style classifier, and
    * ``prob_fn(model)``         -> callable (M, D) -> (M,) pairwise win scores.

Everything else (pairwise conversion, temperature calibration, metrics, saving)
is shared here and mirrors the original tune_*/_train.py flow.
"""
import numpy as np
import torch

from calibration import TemperatureCalibration
from metrics import evaluate_all
from preds_io import save_scores
from pairwise import make_pairwise_dataset, score_items_from_pairwise, ranks_from_relevance
from tune.cv_common import aggregate_candidate, eval_fold, select_and_save


def run_pairwise_cv(model_name, candidates, folds, cfg, make_model, prob_fn, *, smoke):
    """Brand-CV: score each candidate as the fold-mean, then select & save.

    Mirrors tune_lgbm.main()'s loop but with the pairwise fit/predict in place
    of lgb.train. ``folds`` is the list returned by build_kfold_arrays.
    """
    candidate_means = []
    for ci, cand in enumerate(candidates):
        fold_results = []
        for (X_tr, rel_tr, _, _), (X_val, _, ranks_val, _) in folds:
            ranks_tr = ranks_from_relevance(rel_tr)
            dX, y = make_pairwise_dataset(X_tr, ranks_tr)
            model = make_model(cand, cfg.seed)
            model.fit(dX, y)
            val_scores = score_items_from_pairwise(prob_fn(model), X_val)
            fold_results.append(eval_fold(val_scores, ranks_val, cfg.temp_candidates))
        means = aggregate_candidate(fold_results)
        candidate_means.append(means)
        print(
            f"  [{ci+1:2d}/{len(candidates)}] {cand} | "
            f"top1={means['top1_accuracy']:.4f} "
            f"ndcg@3={means['ndcg@3']:.4f} "
            f"ktau={means['kendall_tau']:.4f} nll={means['nll']:.4f}"
        )
    select_and_save(model_name, candidates, candidate_means, cfg.tuning_dir, smoke=smoke)


def train_seed_ensemble(model_name, params, arrays, cfg, make_model, prob_fn):
    """Fit cfg.n_seeds models on the train split, average raw scores, calibrate,
    evaluate on test, and dump val/test scores for blending.

    ``arrays`` is the (train, val, test, scaler) tuple from build_arrays.
    Returns (val_scores, test_scores, ranks_val, ranks_test, results, fitted_models).
    """
    (X_train, rel_train, ranks_train, _), \
    (X_val,   _,         ranks_val,   _), \
    (X_test,  _,         ranks_test,  _), _scaler = arrays

    dX, y = make_pairwise_dataset(X_train, ranks_train)
    print(f"Pairwise train rows: {dX.shape[0]}  features: {dX.shape[1]}  "
          f"trials val/test: {len(ranks_val)}/{len(ranks_test)}  seeds: {cfg.n_seeds}")

    val_per_seed, test_per_seed, temps, models = [], [], [], []
    for i in range(cfg.n_seeds):
        model = make_model(params, cfg.seed + i)
        model.fit(dX, y)
        val_s  = score_items_from_pairwise(prob_fn(model), X_val)
        test_s = score_items_from_pairwise(prob_fn(model), X_test)

        calib = TemperatureCalibration(cfg.temp_candidates)
        calib.fit(torch.tensor(val_s, dtype=torch.float32),
                  torch.tensor(ranks_val, dtype=torch.long))
        val_per_seed.append(val_s)
        test_per_seed.append(test_s)
        temps.append(calib.temperature)
        models.append(model)
        print(f"  seed {i}: T*={calib.temperature}")

    val_scores  = np.mean(val_per_seed,  axis=0)
    test_scores = np.mean(test_per_seed, axis=0)
    avg_temp = float(np.mean(temps))

    exp_cal = np.exp(test_scores / avg_temp)
    test_probs = exp_cal / exp_cal.sum(axis=1, keepdims=True)
    results = evaluate_all(test_scores, ranks_test, test_probs)

    print(f"\n=== {model_name} Test Results (avg T={avg_temp:.3f}) ===")
    for metric, value in results.items():
        print(f"  {metric:<22} {value:.4f}")

    save_scores(cfg.preds_dir, model_name, "val",  val_scores,  ranks_val)
    save_scores(cfg.preds_dir, model_name, "test", test_scores, ranks_test)
    print(f"  -> saved {model_name}_val.npz / {model_name}_test.npz to {cfg.preds_dir}")

    return val_scores, test_scores, ranks_val, ranks_test, results, models
