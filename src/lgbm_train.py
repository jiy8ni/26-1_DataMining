import numpy as np
import torch
import wandb
import lightgbm as lgb

from config import Config
from data import build_arrays
from calibration import TemperatureCalibration
from metrics import evaluate_all


def main():
    cfg = Config()
    engine_tag = cfg.engine_filter or "all"
    run_name   = f"{cfg.protocol}_{cfg.version}_{engine_tag}_lgbm"

    wandb.init(
        project="formcleaner-ranker",
        name=run_name,
        config={
            "model":                "lightgbm_lambdarank",
            "protocol":             cfg.protocol,
            "version":              cfg.version,
            "engine_filter":        engine_tag,
            "n_features":           len(cfg.feature_cols) + (1 if cfg.use_position_feature else 0),
            "use_position_feature": cfg.use_position_feature,
            "log_transform":        bool(cfg.log_transform_cols),
        },
    )

    (X_train, y_train, ranks_train, g_train), \
    (X_val,   y_val,   ranks_val,   g_val),   \
    (X_test,  y_test,  ranks_test,  g_test),  \
    scaler = build_arrays(cfg)

    print(
        f"Trials — train: {len(g_train)}  val: {len(g_val)}  test: {len(g_test)}\n"
        f"Features: {X_train.shape[1]}"
    )

    train_data = lgb.Dataset(X_train, label=y_train, group=g_train)
    val_data   = lgb.Dataset(X_val,   label=y_val,   group=g_val, reference=train_data)

    params = {
        "objective":         "lambdarank",
        "metric":            "ndcg",
        "ndcg_eval_at":      [1, 3],
        "learning_rate":     0.05,
        "num_leaves":        31,
        "min_child_samples": 5,
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
        "verbose":           -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=10),
        ],
    )

    wandb.summary["best_iteration"] = model.best_iteration

    # ---- Temperature calibration on val set ----
    val_scores = model.predict(X_val, raw_score=True).reshape(-1, 3)
    calib = TemperatureCalibration(cfg.temp_candidates)
    calib.fit(
        torch.tensor(val_scores, dtype=torch.float32),
        torch.tensor(ranks_val,  dtype=torch.long),
    )
    print(f"Temperature calibration: T* = {calib.temperature}")
    wandb.summary["temperature"] = calib.temperature

    # ---- Test set evaluation ----
    print("\n=== Test Set Results ===")
    test_scores = model.predict(X_test, raw_score=True).reshape(-1, 3)

    exp_cal    = np.exp(test_scores / calib.temperature)
    test_probs = exp_cal / exp_cal.sum(axis=1, keepdims=True)

    results = evaluate_all(test_scores, ranks_test, test_probs)
    for metric, value in results.items():
        print(f"  {metric:<22} {value:.4f}")

    wandb.log({f"test/{k}": v for k, v in results.items()})
    wandb.finish()


if __name__ == "__main__":
    main()
