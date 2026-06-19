"""Collate the 4 new pairwise models' tuning + test results into integrated
report rows (per engine), matching the column layout of the canonical
artifacts/tuning{,_anthropic}/TUNING_RESULTS*.md tables.

Sources of truth (kept faithful to what the trainers actually reported):
  * CV row  : artifacts/tuning{_anthropic}/{model}_best_params.json
              -> selected params, candidate count, per-fold-mean metrics.
  * Test row: logs/{engine}_train_{model}.log
              -> the "=== {model} Test Results (avg T=..) ===" block the trainer
                 printed (its nll/brier use the mean-of-per-seed temperature, which
                 the saved npz alone cannot reproduce).

Integrity check: the temperature-independent ranking metrics (top1, pairwise,
ndcg@3, kendall_tau) are recomputed from the saved npz and asserted to match the
parsed log to 4 decimals. nll/brier/temp are taken from the log.

Run:  python collect_results.py
Outputs (per engine): artifacts/new_models_results_{engine}.json
                      artifacts/new_models_fragment_{engine}.md
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import json
import re

import numpy as np

from metrics import evaluate_all
from preds_io import load_scores

HANDOFF_ROOT = _os.path.dirname(_os.path.abspath(__file__))
LOGS = _os.path.join(HANDOFF_ROOT, "logs")
ART = _os.path.join(HANDOFF_ROOT, "artifacts")

MODELS = ["ranksvm", "rf", "logreg", "ebm"]
LABELS = {"ranksvm": "RankSVM", "rf": "RandomForest",
          "logreg": "LogisticRegression", "ebm": "EBM"}
ENGINES = {
    "openai":    dict(tuning="tuning",            preds="preds"),
    "anthropic": dict(tuning="tuning_anthropic",  preds="preds_anthropic"),
}

# params that are boilerplate / fixed and add noise to the "selected params" cell
_HIDE_PARAMS = {"class_weight", "n_jobs", "fit_intercept", "solver", "max_iter",
                "max_bins", "outer_bags"}

_TEST_HDR = re.compile(r"===\s+(\w+)\s+Test Results \(avg T=([\d.]+)\)\s+===")
_METRIC_LINE = re.compile(r"^\s+([\w@]+)\s+(-?[\d.]+)\s*$")
_TEST_KEYS = ["top1_accuracy", "pairwise_accuracy", "ndcg@3",
              "kendall_tau", "nll", "brier_score"]


def _fmt_params(params: dict) -> str:
    items = [(k, v) for k, v in params.items() if k not in _HIDE_PARAMS]
    return ", ".join(f"{k}={v}" for k, v in items)


def _parse_test_log(path: str) -> dict | None:
    if not _os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    temp, out, capturing = None, {}, False
    for ln in lines:
        m = _TEST_HDR.search(ln)
        if m:
            temp = float(m.group(2)); out, capturing = {}, True; continue
        if capturing:
            mm = _METRIC_LINE.match(ln)
            if mm and mm.group(1) in _TEST_KEYS:
                out[mm.group(1)] = float(mm.group(2))
            elif out and not mm:
                capturing = False
    if not out:
        return None
    out["avg_temp"] = temp
    return out


def _recompute_ranking(preds_dir: str, model: str) -> dict | None:
    vp = _os.path.join(preds_dir, f"{model}_test.npz")
    if not _os.path.exists(vp):
        return None
    scores, ranks = load_scores(preds_dir, model, "test")
    return evaluate_all(scores, ranks)  # ranking-only (probs=None)


def collect_engine(engine: str) -> list[dict]:
    cfgdirs = ENGINES[engine]
    tuning_dir = _os.path.join(ART, cfgdirs["tuning"])
    preds_dir = _os.path.join(ART, cfgdirs["preds"])
    rows = []
    for model in MODELS:
        row = {"model": model, "label": LABELS[model], "engine": engine}
        # --- CV row from best_params.json ---
        bp = _os.path.join(tuning_dir, f"{model}_best_params.json")
        if _os.path.exists(bp):
            with open(bp, encoding="utf-8") as f:
                payload = json.load(f)
            row["params"] = payload["params"]
            row["params_str"] = _fmt_params(payload["params"])
            row["n_candidates"] = len(payload.get("all_candidates", []))
            row["cv_balanced"] = payload.get("cv_balanced")
            row["cv"] = payload.get("per_metric", {})
        else:
            row["error_cv"] = f"missing {bp}"
        # --- test row from trainer log (+ npz cross-check) ---
        test = _parse_test_log(_os.path.join(LOGS, f"{engine}_train_{model}.log"))
        recomp = _recompute_ranking(preds_dir, model)
        if test:
            row["test"] = test
            if recomp:
                mism = {k: (test[k], recomp[k]) for k in
                        ["top1_accuracy", "pairwise_accuracy", "ndcg@3", "kendall_tau"]
                        if abs(test[k] - recomp[k]) > 1e-4}
                row["npz_check"] = "OK" if not mism else f"MISMATCH {mism}"
        else:
            row["error_test"] = f"no test block in logs/{engine}_train_{model}.log"
        rows.append(row)
    return rows


def _cv_table(rows: list[dict]) -> str:
    head = ("| model | candidates | selected params | cv_balanced | top1 | "
            "pairwise | ndcg@3 | tau | nll | brier | temp |\n"
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    out = [head]
    for r in rows:
        if "cv" not in r:
            out.append(f"| `{r['model']}` | — | (missing) |" + " — |" * 8)
            continue
        c = r["cv"]
        out.append(
            f"| `{r['model']}` | {r['n_candidates']} | `{r['params_str']}` | "
            f"{r['cv_balanced']:.4f} | {c['top1_accuracy']:.4f} | {c['pairwise_accuracy']:.4f} | "
            f"{c['ndcg@3']:.4f} | {c['kendall_tau']:.4f} | {c['nll']:.4f} | "
            f"{c['brier_score']:.4f} | {c['temperature']:.2f} |"
        )
    return "\n".join(out)


def _test_table(rows: list[dict]) -> str:
    head = ("| model | avg temp | top1 | pairwise | ndcg@3 | tau | nll | brier |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    out = [head]
    for r in rows:
        if "test" not in r:
            out.append(f"| `{r['model']}` |" + " — |" * 7)
            continue
        t = r["test"]
        out.append(
            f"| `{r['model']}` | {t['avg_temp']:.2f} | {t['top1_accuracy']:.4f} | "
            f"{t['pairwise_accuracy']:.4f} | {t['ndcg@3']:.4f} | {t['kendall_tau']:.4f} | "
            f"{t['nll']:.4f} | {t['brier_score']:.4f} |"
        )
    return "\n".join(out)


def main():
    for engine in ENGINES:
        rows = collect_engine(engine)
        with open(_os.path.join(ART, f"new_models_results_{engine}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        frag = (f"### 신규 pairwise 모델 — CV 최종 선택 ({engine})\n\n"
                + _cv_table(rows)
                + f"\n\n### 신규 pairwise 모델 — 최종 테스트 결과 ({engine})\n\n"
                + _test_table(rows) + "\n")
        with open(_os.path.join(ART, f"new_models_fragment_{engine}.md"),
                  "w", encoding="utf-8") as f:
            f.write(frag)
        print(f"\n========== {engine} ==========")
        print(frag)
        checks = [f"{r['model']}:{r.get('npz_check', r.get('error_test', '?'))}" for r in rows]
        print("npz integrity:", "  ".join(checks))


if __name__ == "__main__":
    main()
