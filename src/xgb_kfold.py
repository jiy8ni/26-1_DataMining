import os
import pickle

import numpy as np
import torch
import wandb
import xgboost as xgb

from config import Config
from data import build_kfold_arrays
from calibration import TemperatureCalibration
from metrics import evaluate_all


def _make_qid(groups: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(groups)), groups)


def main():
    cfg = Config()
    cfg.version = "v4"
    engine_tag  = cfg.engine_filter or "all"
    run_name    = f"{cfg.protocol}_{cfg.version}_{engine_tag}_xgb"

    wandb.init(
        project="formcleaner-ranker",
        name=run_name,
        config={
            "model":                "xgboost_lambdarank",
            "protocol":             cfg.protocol,
            "version":              cfg.version,
            "engine_filter":        engine_tag,
            "n_folds":              cfg.n_folds,
            "n_features":           len(cfg.feature_cols) + (1 if cfg.use_position_feature else 0),
            "use_position_feature": cfg.use_position_feature,
            "log_transform":        bool(cfg.log_transform_cols),
        },
    )

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

    folds, test_folds, scalers = build_kfold_arrays(cfg)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    fold_models = []
    fold_calibs = []

    for fold_idx, ((X_tr, y_tr, _, g_tr), (X_val, y_val, ranks_val, g_val)) in enumerate(folds):
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx + 1}/{cfg.n_folds}  — train trials: {len(g_tr)}  val trials: {len(g_val)}")

        dtrain = xgb.DMatrix(X_tr,  label=y_tr,  qid=_make_qid(g_tr))
        dval   = xgb.DMatrix(X_val, label=y_val, qid=_make_qid(g_val))

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, "val")],
            callbacks=[
                xgb.callback.EvaluationMonitor(period=50),
                xgb.callback.EarlyStopping(rounds=20),
            ],
        )

        wandb.summary[f"fold{fold_idx}/best_iteration"] = model.best_iteration

        ckpt_path   = os.path.join(cfg.ckpt_dir, f"{cfg.protocol}_{cfg.version}_{engine_tag}_xgb_fold{fold_idx}.ubj")
        scaler_path = os.path.join(cfg.ckpt_dir, f"{cfg.protocol}_{cfg.version}_{engine_tag}_xgb_fold{fold_idx}_scaler.pkl")
        model.save_model(ckpt_path)
        with open(scaler_path, "wb") as f:
            pickle.dump(scalers[fold_idx], f)

        val_scores = model.predict(dval).reshape(-1, 3)
        calib = TemperatureCalibration(cfg.temp_candidates)
        calib.fit(
            torch.tensor(val_scores, dtype=torch.float32),
            torch.tensor(ranks_val,  dtype=torch.long),
        )
        wandb.summary[f"fold{fold_idx}/temperature"] = calib.temperature

        val_results = evaluate_all(val_scores, ranks_val)
        wandb.log({f"fold{fold_idx}/val/{k}": v for k, v in val_results.items()})
        print(f"  Val  — " + "  ".join(f"{k}: {v:.4f}" for k, v in val_results.items()))
        print(f"  T*   = {calib.temperature}")

        fold_models.append(model)
        fold_calibs.append(calib)

    # ---- Ensemble test evaluation ----
    print(f"\n{'='*50}")
    print("=== Ensemble Test Results ===")

    test_scores_per_fold = []
    ranks_test_ref = None
    for model, (X_test, y_test, ranks_test, g_test) in zip(fold_models, test_folds):
        dtest = xgb.DMatrix(X_test, label=y_test, qid=_make_qid(g_test))
        test_scores_per_fold.append(model.predict(dtest).reshape(-1, 3))
        ranks_test_ref = ranks_test

    ensemble_scores = np.mean(test_scores_per_fold, axis=0)
    avg_temp        = float(np.mean([c.temperature for c in fold_calibs]))
    wandb.summary["ensemble/temperature"] = avg_temp

    exp_cal    = np.exp(ensemble_scores / avg_temp)
    test_probs = exp_cal / exp_cal.sum(axis=1, keepdims=True)

    results = evaluate_all(ensemble_scores, ranks_test_ref, test_probs)
    for metric, value in results.items():
        print(f"  {metric:<22} {value:.4f}")

    wandb.log({f"test/{k}": v for k, v in results.items()})
    wandb.finish()


if __name__ == "__main__":
    main()
