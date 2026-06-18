import os

import lightgbm as lgb
import numpy as np
import torch
import wandb

import os as _os, sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from calibration import TemperatureCalibration
from config import Config
from data import build_arrays, effective_feature_dim
from metrics import evaluate_all
from preds_io import save_scores
from tune.runtime import apply_saved_semantic_config, load_tuned_params

DEFAULT_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3],
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 20,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 0.5,
    "verbose": -1,
}


def load_params(cfg: Config) -> dict:
    return load_tuned_params(cfg, "lgbm_best_params.json", DEFAULT_PARAMS, "LightGBM")


def main():
    cfg = Config()
    apply_saved_semantic_config(cfg)
    engine_tag = cfg.engine_filter or "all"
    run_name = f"{cfg.protocol}_{cfg.version}_{engine_tag}_lgbm"

    params = load_params(cfg)

    wandb.init(
        project="formcleaner-ranker",
        name=run_name,
        config={
            "model": "lightgbm_lambdarank",
            "protocol": cfg.protocol,
            "version": cfg.version,
            "engine_filter": engine_tag,
            "n_seeds": cfg.n_seeds,
            "n_features": effective_feature_dim(cfg),
            "use_position_feature": cfg.use_position_feature,
            "log_transform": bool(cfg.log_transform_cols),
            "params": params,
        },
    )

    (X_train, y_train, ranks_train, g_train), (
        X_val,
        y_val,
        ranks_val,
        g_val,
    ), (
        X_test,
        y_test,
        ranks_test,
        g_test,
    ), scaler = build_arrays(cfg)

    print(
        f"Trials - train: {len(g_train)}  val: {len(g_val)}  test: {len(g_test)}\n"
        f"Features: {X_train.shape[1]}  |  seeds: {cfg.n_seeds}"
    )

    train_data = lgb.Dataset(X_train, label=y_train, group=g_train)
    val_data = lgb.Dataset(X_val, label=y_val, group=g_val, reference=train_data)

    val_scores_per_seed, test_scores_per_seed, temps = [], [], []
    for i in range(cfg.n_seeds):
        seed_params = {
            **params,
            "seed": cfg.seed + i,
            "bagging_seed": cfg.seed + i,
            "feature_fraction_seed": cfg.seed + i,
        }
        model = lgb.train(
            seed_params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )
        val_s = model.predict(X_val, raw_score=True).reshape(-1, 3)
        test_s = model.predict(X_test, raw_score=True).reshape(-1, 3)

        calib = TemperatureCalibration(cfg.temp_candidates)
        calib.fit(torch.tensor(val_s, dtype=torch.float32), torch.tensor(ranks_val, dtype=torch.long))

        val_scores_per_seed.append(val_s)
        test_scores_per_seed.append(test_s)
        temps.append(calib.temperature)
        print(f"  seed {i}: best_iter={model.best_iteration}  T*={calib.temperature}")

    val_scores = np.mean(val_scores_per_seed, axis=0)
    test_scores = np.mean(test_scores_per_seed, axis=0)
    avg_temp = float(np.mean(temps))
    wandb.summary["ensemble/temperature"] = avg_temp

    print("\n=== Test Set Results ===")
    exp_cal = np.exp(test_scores / avg_temp)
    test_probs = exp_cal / exp_cal.sum(axis=1, keepdims=True)

    results = evaluate_all(test_scores, ranks_test, test_probs)
    for metric, value in results.items():
        print(f"  {metric:<22} {value:.4f}")

    save_scores(cfg.preds_dir, "lgbm", "val", val_scores, ranks_val)
    save_scores(cfg.preds_dir, "lgbm", "test", test_scores, ranks_test)

    wandb.log({f"test/{k}": v for k, v in results.items()})
    wandb.finish()


if __name__ == "__main__":
    main()
