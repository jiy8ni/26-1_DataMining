import numpy as np
import torch
import wandb
import xgboost as xgb

from config import Config
from data import build_arrays
from calibration import TemperatureCalibration
from metrics import evaluate_all


def _make_qid(groups: np.ndarray) -> np.ndarray:
    """Convert group-count array → per-item query ID (e.g. [3,3,...] → [0,0,0,1,1,1,...])."""
    return np.repeat(np.arange(len(groups)), groups)


def main():
    cfg = Config()
    engine_tag = cfg.engine_filter or "all"
    run_name   = f"{cfg.protocol}_{cfg.version}_{engine_tag}_xgb"

    wandb.init(
        project="formcleaner-ranker",
        name=run_name,
        config={
            "model":                "xgboost_lambdarank",
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

    dtrain = xgb.DMatrix(X_train, label=y_train, qid=_make_qid(g_train))
    dval   = xgb.DMatrix(X_val,   label=y_val,   qid=_make_qid(g_val))
    dtest  = xgb.DMatrix(X_test,  label=y_test,  qid=_make_qid(g_test))

    params = {
        "objective":        "rank:ndcg",
        "eval_metric":      "ndcg@1",
        "learning_rate":    0.05,
        "max_depth":        5,
        "min_child_weight": 5,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "seed":             cfg.seed,
        "verbosity":        0,
    }

    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        callbacks=[
            xgb.callback.EvaluationMonitor(period=10),
            xgb.callback.EarlyStopping(rounds=20),
        ],
    )

    for i, (tr, val) in enumerate(zip(
        evals_result.get("train", {}).get("ndcg@1", []),
        evals_result.get("val",   {}).get("ndcg@1", []),
    )):
        wandb.log({"iter/train_ndcg1": tr, "iter/val_ndcg1": val}, step=i)

    wandb.summary["best_iteration"] = model.best_iteration

    # ---- Temperature calibration on val set ----
    val_scores = model.predict(dval).reshape(-1, 3)
    calib = TemperatureCalibration(cfg.temp_candidates)
    calib.fit(
        torch.tensor(val_scores, dtype=torch.float32),
        torch.tensor(ranks_val,  dtype=torch.long),
    )
    print(f"Temperature calibration: T* = {calib.temperature}")
    wandb.summary["temperature"] = calib.temperature

    # ---- Test set evaluation ----
    print("\n=== Test Set Results ===")
    test_scores = model.predict(dtest).reshape(-1, 3)

    exp_cal    = np.exp(test_scores / calib.temperature)
    test_probs = exp_cal / exp_cal.sum(axis=1, keepdims=True)

    results = evaluate_all(test_scores, ranks_test, test_probs)
    for metric, value in results.items():
        print(f"  {metric:<22} {value:.4f}")

    wandb.log({f"test/{k}": v for k, v in results.items()})
    wandb.finish()


if __name__ == "__main__":
    main()
