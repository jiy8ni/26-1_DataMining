#!/usr/bin/env bash
# Full tune+train pipeline for the 4 new pairwise models, both engines.
# Tuners run first (write {model}_best_params.json), then trainers load them.
set -u
cd "$(dirname "$0")"
export PYTHONUTF8=1
mkdir -p logs

run() {  # run <logname> <cmd...>
  local log="logs/$1.log"; shift
  echo ">>> $* | -> $log"
  if "$@" >"$log" 2>&1; then
    echo "    OK"; tail -n 12 "$log"
  else
    echo "    FAILED (exit $?) — see $log"; tail -n 25 "$log"
  fi
  echo
}

for ENGINE in openai anthropic; do
  if [ "$ENGINE" = "anthropic" ]; then export DM_ENGINE=anthropic; else unset DM_ENGINE; fi
  echo "############## ENGINE=$ENGINE ##############"

  # --- tuning (order: fast -> slow) ---
  run "${ENGINE}_tune_logreg"  python tune/tune_logreg.py
  run "${ENGINE}_tune_ranksvm" python tune/tune_ranksvm.py
  run "${ENGINE}_tune_rf"      python tune/tune_rf.py
  run "${ENGINE}_tune_ebm"     python tune/tune_ebm.py

  # --- training (loads tuned params, saves preds) ---
  run "${ENGINE}_train_logreg"  python logreg/logreg_train.py
  run "${ENGINE}_train_ranksvm" python ranksvm/ranksvm_train.py
  run "${ENGINE}_train_rf"      python rforest/rf_train.py
  run "${ENGINE}_train_ebm"     python ebm/ebm_train.py

  # --- blend (sanity, non-blocking on failure) ---
  run "${ENGINE}_blend" python blend_new.py
done

echo "ALL DONE"
