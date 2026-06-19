#!/usr/bin/env bash
# Re-run full tuning + final training pipeline, capturing all stdout.
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

export WANDB_MODE=disabled
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

run() {
  echo ""
  echo "########## $* ##########"
  python "$@"
  echo "########## DONE (exit=$?): $* ##########"
}

echo "===== TUNERS ====="
run src/tune/tune_semantic.py
run src/tune/tune_mlp.py
run src/tune/tune_lgbm.py
run src/tune/tune_xgb.py
run src/tune/tune_lgbm_pl.py
run src/tune/tune_xgb_pl.py

echo "===== TRAINERS ====="
run src/mlp/train.py
run src/lightgbm/lgbm_train.py
run src/xgboost/xgb_train.py
run src/lightgbm/lgbm_train_pl.py
run src/xgboost/xgb_train_pl.py

echo "===== BLEND (post-hoc on saved preds) ====="
run src/blend.py

echo "===== PIPELINE COMPLETE ====="
