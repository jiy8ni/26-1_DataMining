import json
import os

import numpy as np
import torch
import wandb
import xgboost as xgb

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from config import Config
from data import build_arrays
from calibration import TemperatureCalibration
from metrics import evaluate_all
from preds_io import save_scores


# Regularized fallback params. Overridden by artifacts/tuning/xgb_best_params.json
# when present (written by tune_xgb.py).
DEFAULT_PARAMS = {
    "objective":        "rank:ndcg",
    "eval_metric":      "ndcg@1",
    "eta":              0.05,
    "max_depth":        3,
    "min_child_weight": 10,
    "gamma":            0.5,
    "subsample":        0.8,
    "colsample_bytree": 0.7,
    "reg_alpha":        0.0,
    "reg_lambda":       2.0,
    "verbosity":        0,
}


def _make_qid(groups: np.ndarray) -> np.ndarray:
    """Convert group-count array → per-item query ID (e.g. [3,3,...] → [0,0,0,1,1,1,...])."""
    return np.repeat(np.arange(len(groups)), groups)


def load_params(cfg: Config) -> dict:
    """Use brand-CV-selected params if available, else the regularized defaults."""
    path = os.path.join(cfg.tuning_dir, "xgb_best_params.json")
    if os.path.exists(path):
        with open(path) as f:
            params = json.load(f)["params"]
        print(f"Loaded tuned XGBoost params from {path}")
        return params
    print("No tuned params found — using regularized DEFAULT_PARAMS.")
    return dict(DEFAULT_PARAMS)


def main():
    cfg = Config()
    engine_tag = cfg.engine_filter or "all"
    run_name   = f"{cfg.protocol}_{cfg.version}_{engine_tag}_xgb"

    params = load_params(cfg)

    wandb.init(
        project="formcleaner-ranker",
        name=run_name,
        config={
            "model":                "xgboost_lambdarank",
            "protocol":             cfg.protocol,
            "version":              cfg.version,
            "engine_filter":        engine_tag,
            "n_seeds":              cfg.n_seeds,
            "n_features":           len(cfg.feature_cols) + (1 if cfg.use_position_feature else 0),
            "use_position_feature": cfg.use_position_feature,
            "log_transform":        bool(cfg.log_transform_cols),
            "params":               params,
        },
    )

    (X_train, y_train, ranks_train, g_train), \
    (X_val,   y_val,   ranks_val,   g_val),   \
    (X_test,  y_test,  ranks_test,  g_test),  \
    scaler = build_arrays(cfg)

    print(
        f"Trials — train: {len(g_train)}  val: {len(g_val)}  test: {len(g_test)}\n"
        f"Features: {X_train.shape[1]}  |  seeds: {cfg.n_seeds}"
    )

    dtrain = xgb.DMatrix(X_train, label=y_train, qid=_make_qid(g_train))
    dval   = xgb.DMatrix(X_val,   label=y_val,   qid=_make_qid(g_val))
    dtest  = xgb.DMatrix(X_test,  label=y_test,  qid=_make_qid(g_test))

    # ---- Seed ensemble: subsampling (subsample/colsample<1) makes seeds diverge ----
    val_scores_per_seed, test_scores_per_seed, temps = [], [], []
    for i in range(cfg.n_seeds):
        seed_params = {**params, "seed": cfg.seed + i}
        evals_result = {}
        model = xgb.train(
            seed_params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, "val")],
            evals_result=evals_result,
            verbose_eval=False,
            callbacks=[xgb.callback.EarlyStopping(rounds=20)],
        )
        val_s  = model.predict(dval).reshape(-1, 3)
        test_s = model.predict(dtest).reshape(-1, 3)

        calib = TemperatureCalibration(cfg.temp_candidates)
        calib.fit(torch.tensor(val_s, dtype=torch.float32),
                  torch.tensor(ranks_val, dtype=torch.long))

        val_scores_per_seed.append(val_s)
        test_scores_per_seed.append(test_s)
        temps.append(calib.temperature)
        print(f"  seed {i}: best_iter={model.best_iteration}  T*={calib.temperature}")

    val_scores  = np.mean(val_scores_per_seed,  axis=0)
    test_scores = np.mean(test_scores_per_seed, axis=0)
    avg_temp    = float(np.mean(temps))
    wandb.summary["ensemble/temperature"] = avg_temp

    # ---- Test set evaluation (ensembled scores, averaged temperature) ----
    print("\n=== Test Set Results ===")
    exp_cal    = np.exp(test_scores / avg_temp)
    test_probs = exp_cal / exp_cal.sum(axis=1, keepdims=True)

    results = evaluate_all(test_scores, ranks_test, test_probs)
    for metric, value in results.items():
        print(f"  {metric:<22} {value:.4f}")

    # ---- Persist scores for blending ----
    save_scores(cfg.preds_dir, "xgb", "val",  val_scores,  ranks_val)
    save_scores(cfg.preds_dir, "xgb", "test", test_scores, ranks_test)

    wandb.log({f"test/{k}": v for k, v in results.items()})
    wandb.finish()


if __name__ == "__main__":
    main()
