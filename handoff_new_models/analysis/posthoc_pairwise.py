"""Post-hoc interpretability for the four new pairwise rankers
(RankSVM / RandomForest / LogisticRegression / EBM), the pairwise analogue of
src/analysis/posthoc_lgbm.py.

KEY CAVEAT (surfaced in the report): these models are trained on the feature
DIFFERENCE dX = X_i - X_j, so every importance / coefficient below describes how
a *difference* in a feature drives "item i beats item j", not a raw feature level.

For each engine it refits each model once (tuned params, seed=cfg.seed) on the
pairwise train set, then computes:

  A. Global importance  - model-agnostic permutation importance in dX space:
     permute one feature column across the val items, re-score via the pairwise
     aggregation, measure the drop in top1 / ndcg@3. Comparable across all 4 models.
  B. Direction (native) - signed coef_ for LogReg / linear RankSVM (Bradley-Terry
     weights), feature_importances_ for RF, term importances for EBM.
  C. Block contribution - permutation importance summed over
     structural / text_pca / image_pca / position blocks.
  D. Position bias      - the position feature's share and rank.
  E. Error analysis     - top1-vs-2nd score margin of correct vs wrong picks (test npz).
  F. Cross-model agree  - Spearman of the per-model permutation-importance vectors.
  G. pl_theta cross-check - model-independent Spearman of structural features vs the
     Plackett-Luce item strength (mirrors the lgbm post-hoc).

Run:  python analysis/posthoc_pairwise.py --engine openai
      python analysis/posthoc_pairwise.py --engine anthropic
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from data import build_arrays
from paths import configure_paths, HANDOFF_ROOT, PROJECT_ROOT
from pairwise import make_pairwise_dataset, score_items_from_pairwise
from metrics import top1_accuracy, ndcg_at_k
from tune.runtime import apply_saved_semantic_config, load_tuned_params
from analysis.feature_layout import build_feature_layout

warnings.filterwarnings("ignore")

MODELS = ["ranksvm", "rf", "logreg", "ebm"]
LABELS = {"ranksvm": "RankSVM", "rf": "RandomForest",
          "logreg": "LogisticRegression", "ebm": "EBM"}
BLOCKS = ["structural", "text_pca", "image_pca", "position"]
N_PERM_REPEATS = 5


# --------------------------------------------------------------------------- #
# model plug-ins (make_model / prob_fn) imported from the tuner modules
# --------------------------------------------------------------------------- #
def _model_funcs(model: str):
    if model == "ranksvm":
        from tune.tune_ranksvm import make_model, prob_fn
    elif model == "rf":
        from tune.tune_rf import make_model, prob_fn
    elif model == "logreg":
        from tune.tune_logreg import make_model, prob_fn
    elif model == "ebm":
        from tune.tune_ebm import make_model, prob_fn
    else:
        raise ValueError(model)
    return make_model, prob_fn


def _fallback_params(model: str) -> dict:
    if model == "ranksvm":
        from ranksvm.ranksvm_train import DEFAULT_PARAMS
    elif model == "rf":
        from rforest.rf_train import DEFAULT_PARAMS
    elif model == "logreg":
        from logreg.logreg_train import DEFAULT_PARAMS
    elif model == "ebm":
        from ebm.ebm_train import DEFAULT_PARAMS
    else:
        raise ValueError(model)
    return dict(DEFAULT_PARAMS)


# --------------------------------------------------------------------------- #
# A. permutation importance (model-agnostic, in dX space)
# --------------------------------------------------------------------------- #
def permutation_importance(prob_fn, X_val, ranks_val, rng):
    """Drop in top1 / ndcg@3 when each feature column is shuffled across val items."""
    base_scores = score_items_from_pairwise(prob_fn, X_val)
    base_top1 = top1_accuracy(base_scores, ranks_val)
    base_ndcg = ndcg_at_k(base_scores, ranks_val, k=3)
    F = X_val.shape[1]
    imp_top1 = np.zeros(F)
    imp_ndcg = np.zeros(F)
    for f in range(F):
        d1, d3 = [], []
        for _ in range(N_PERM_REPEATS):
            Xp = X_val.copy()
            Xp[:, f] = Xp[rng.permutation(Xp.shape[0]), f]
            s = score_items_from_pairwise(prob_fn, Xp)
            d1.append(base_top1 - top1_accuracy(s, ranks_val))
            d3.append(base_ndcg - ndcg_at_k(s, ranks_val, k=3))
        imp_top1[f] = float(np.mean(d1))
        imp_ndcg[f] = float(np.mean(d3))
    return dict(base_top1=base_top1, base_ndcg=base_ndcg,
                imp_top1=imp_top1, imp_ndcg=imp_ndcg)


# --------------------------------------------------------------------------- #
# B. native importance / direction
# --------------------------------------------------------------------------- #
def native_importance(model, model_name, F):
    """Return (vector length F or None, kind-string). Signed where meaningful."""
    if model_name in ("logreg", "ranksvm") and hasattr(model, "coef_"):
        return np.asarray(model.coef_).reshape(-1)[:F], "signed_coef"
    if model_name == "rf" and hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_)[:F], "impurity"
    if model_name == "ebm":
        try:
            imps = np.asarray(model.term_importances())
            vec = np.zeros(F)
            for ti, feats in enumerate(model.term_features_):
                if len(feats) == 1 and feats[0] < F:   # main effects only
                    vec[feats[0]] = imps[ti]
            return vec, "term_importance(main)"
        except Exception:
            return None, "unavailable"
    return None, "unavailable"


# --------------------------------------------------------------------------- #
# E. error margin analysis from saved test predictions
# --------------------------------------------------------------------------- #
def error_margins(preds_dir, model):
    from preds_io import load_scores
    p = _os.path.join(preds_dir, f"{model}_test.npz")
    if not _os.path.exists(p):
        return None
    scores, ranks = load_scores(preds_dir, model, "test")
    srt = np.sort(scores, axis=1)[:, ::-1]
    margin = srt[:, 0] - srt[:, 1]
    correct = scores.argmax(1) == ranks.argmin(1)
    return dict(
        margin=margin, correct=correct,
        med_correct=float(np.median(margin[correct])) if correct.any() else float("nan"),
        med_wrong=float(np.median(margin[~correct])) if (~correct).any() else float("nan"),
        n=int(len(correct)), acc=float(correct.mean()),
    )


# --------------------------------------------------------------------------- #
# G. pl_theta cross-check (model-independent)
# --------------------------------------------------------------------------- #
def pl_theta_crosscheck(cfg):
    try:
        pl = pd.read_csv(cfg.pl_labels_path)
        if "resolved_url" not in pl or "pl_theta" not in pl:
            return None
        frames = []
        for split in ("train", "val", "test"):
            fp = _os.path.join(cfg.data_dir, f"{cfg.protocol}_{split}_features.csv")
            if _os.path.exists(fp):
                frames.append(pd.read_csv(fp))
        if not frames:
            return None
        feat = pd.concat(frames, ignore_index=True)
        if cfg.engine_filter and "engine" in feat:
            feat = feat[feat["engine"] == cfg.engine_filter]
        url_col = "resolved_url" if "resolved_url" in feat else None
        if url_col is None:
            return None
        cols = [c for c in cfg.feature_cols if c in feat.columns]
        items = feat[[url_col] + cols].drop_duplicates(url_col)
        merged = items.merge(pl[["resolved_url", "pl_theta"]], on="resolved_url", how="inner")
        if len(merged) < 10:
            return None
        corr = {}
        for c in cols:
            rho, _ = spearmanr(merged[c], merged["pl_theta"], nan_policy="omit")
            if np.isfinite(rho):
                corr[c] = float(rho)
        return dict(n_items=int(len(merged)),
                    corr=dict(sorted(corr.items(), key=lambda kv: -abs(kv[1]))))
    except Exception as e:
        return dict(error=str(e))


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_global_importance(per_model, names, out_dir):
    fig, axes = plt.subplots(1, len(per_model), figsize=(5 * len(per_model), 6), squeeze=False)
    for ax, (m, d) in zip(axes[0], per_model.items()):
        imp = d["imp_top1"]
        order = np.argsort(imp)[::-1][:12]
        ax.barh([names[i] for i in order][::-1], imp[order][::-1], color="#4C72B0")
        ax.set_title(f"{LABELS[m]}\n(perm. top1 drop)")
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(_os.path.join(out_dir, "global_importance.png"), dpi=110)
    plt.close(fig)


def plot_block_contribution(block_share, out_dir):
    models = list(block_share.keys())
    bottom = np.zeros(len(models))
    colors = {"structural": "#4C72B0", "text_pca": "#DD8452",
              "image_pca": "#55A868", "position": "#C44E52"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for blk in BLOCKS:
        vals = np.array([block_share[m].get(blk, 0.0) for m in models])
        ax.bar([LABELS[m] for m in models], vals, bottom=bottom, label=blk, color=colors[blk])
        bottom += vals
    ax.set_ylabel("share of total permutation importance")
    ax.set_title("Block contribution (dX permutation importance)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_os.path.join(out_dir, "block_contribution.png"), dpi=110)
    plt.close(fig)


def plot_error_margins(err, out_dir):
    models = [m for m in err if err[m]]
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), squeeze=False)
    for ax, m in zip(axes[0], models):
        e = err[m]
        ax.hist(e["margin"][e["correct"]], bins=20, alpha=0.6, label="correct", color="#55A868")
        ax.hist(e["margin"][~e["correct"]], bins=20, alpha=0.6, label="wrong", color="#C44E52")
        ax.set_title(f"{LABELS[m]} top1 margin"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_os.path.join(out_dir, "error_margins.png"), dpi=110)
    plt.close(fig)


def plot_agreement(corr_mat, models, out_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr_mat, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(models))); ax.set_xticklabels([LABELS[m] for m in models], rotation=45, ha="right")
    ax.set_yticks(range(len(models))); ax.set_yticklabels([LABELS[m] for m in models])
    for i in range(len(models)):
        for j in range(len(models)):
            ax.text(j, i, f"{corr_mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if corr_mat[i, j] < 0.6 else "black", fontsize=8)
    ax.set_title("Cross-model importance agreement (Spearman)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(_os.path.join(out_dir, "cross_model_agreement.png"), dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def write_report(engine, cfg, names, layout, per_model, native, block_share,
                 err, agree, models_present, pl_cross, out_dir, md_name):
    L = []
    a = L.append
    a(f"# 신규 pairwise 모델 사후 분석 ({engine})\n")
    a(f"대상 모델: {', '.join(LABELS[m] for m in models_present)} "
      f"(입력 = 항목 쌍의 feature 차이 dX = X_i - X_j)\n")
    a("> **해석 주의**: 네 모델 모두 **feature 차이(dX)** 로 학습합니다. 따라서 아래 모든 "
      "중요도/계수는 'feature 값'이 아니라 'feature **차이**가 두 항목의 승패에 주는 영향'을 "
      "나타냅니다.\n")
    a(f"- 엔진 필터: `{cfg.engine_filter}`  |  시맨틱: text_pca={cfg.text_pca_dim}, "
      f"image_pca={cfg.image_pca_dim}  |  feature 수: {len(names)}\n")

    # exec summary
    a("\n## 1. 요약\n")
    for m in models_present:
        top = np.argsort(per_model[m]["imp_top1"])[::-1][:3]
        toptxt = ", ".join(f"{names[i]}({per_model[m]['imp_top1'][i]:+.3f})" for i in top)
        a(f"- **{LABELS[m]}** (val top1={per_model[m]['base_top1']:.4f}, "
          f"ndcg@3={per_model[m]['base_ndcg']:.4f}): 상위 중요 feature = {toptxt}")
    a("")
    a("- 블록 기여(구조/텍스트/이미지/위치) 비중은 §3, 모델 간 중요도 일치도는 §6 참조.")

    # A global importance
    a("\n## 2. 전역 중요도 (permutation, dX 공간)\n")
    a("각 feature 열을 검증셋에서 섞었을 때 top1 정확도가 떨어지는 정도(클수록 중요). "
      "모든 모델에 동일하게 적용되어 직접 비교 가능합니다. 그림: `global_importance.png`.\n")
    a("| model | top1 feature 1 | 2 | 3 |")
    a("| --- | --- | --- | --- |")
    for m in models_present:
        top = np.argsort(per_model[m]["imp_top1"])[::-1][:3]
        cells = " | ".join(f"{names[i]} ({per_model[m]['imp_top1'][i]:+.3f})" for i in top)
        a(f"| `{m}` | {cells} |")

    # B native direction
    a("\n## 3. 방향성 (모델 고유 중요도)\n")
    for m in models_present:
        vec, kind = native[m]
        if vec is None:
            a(f"- **{LABELS[m]}**: 고유 중요도 없음 ({kind}; 예: RBF 커널은 선형 계수가 없어 "
              "permutation 중요도로만 해석).")
            continue
        order = np.argsort(np.abs(vec))[::-1][:5]
        txt = ", ".join(f"{names[i]}({vec[i]:+.3f})" for i in order)
        sign_note = " 부호는 'i가 j보다 그 feature가 클 때 승리 확률↑(+)/↓(-)'를 의미." \
            if kind == "signed_coef" else ""
        a(f"- **{LABELS[m]}** [{kind}]: {txt}.{sign_note}")

    # C block contribution
    a("\n## 4. 블록 기여도\n")
    a("permutation 중요도(top1 기준, 음수는 0으로 클립)를 블록별로 합산한 비중입니다. "
      "그림: `block_contribution.png`.\n")
    a("| model | structural | text_pca | image_pca | position |")
    a("| --- | ---: | ---: | ---: | ---: |")
    for m in models_present:
        bs = block_share[m]
        a(f"| `{m}` | {bs.get('structural',0)*100:.1f}% | {bs.get('text_pca',0)*100:.1f}% | "
          f"{bs.get('image_pca',0)*100:.1f}% | {bs.get('position',0)*100:.1f}% |")

    # D position bias
    a("\n## 5. 위치 편향(position)\n")
    if layout.position[1] > layout.position[0]:
        pidx = layout.position[0]
        for m in models_present:
            imp = per_model[m]["imp_top1"]
            rank = int((np.argsort(imp)[::-1] == pidx).argmax()) + 1
            a(f"- **{LABELS[m]}**: position 기여 {block_share[m].get('position',0)*100:.1f}% "
              f"(중요도 순위 {rank}/{len(imp)}).")
    else:
        a("- position feature 비활성.")

    # E error analysis
    a("\n## 6. 오류 분석 (top1 마진)\n")
    a("예측 1등과 2등 점수 차이(margin)를 정답/오답으로 나눈 분포. 오답이 더 작은 마진에 "
      "몰려 있으면 '접전에서만 틀린다'는 뜻입니다. 그림: `error_margins.png`.\n")
    a("| model | n | test top1 | margin(정답) | margin(오답) |")
    a("| --- | ---: | ---: | ---: | ---: |")
    for m in models_present:
        e = err.get(m)
        if e:
            a(f"| `{m}` | {e['n']} | {e['acc']:.4f} | {e['med_correct']:.3f} | {e['med_wrong']:.3f} |")

    # F agreement
    a("\n## 7. 모델 간 중요도 일치도\n")
    a("모델별 permutation 중요도 벡터 사이의 Spearman 상관. 그림: `cross_model_agreement.png`.\n")
    a("| | " + " | ".join(LABELS[m] for m in models_present) + " |")
    a("| --- | " + " | ".join("---:" for _ in models_present) + " |")
    for i, m in enumerate(models_present):
        a(f"| **{LABELS[m]}** | " + " | ".join(f"{agree[i, j]:.2f}" for j in range(len(models_present))) + " |")

    # G pl_theta
    a("\n## 8. pl_theta 교차검증 (모델 독립)\n")
    if pl_cross and "corr" in pl_cross:
        a(f"항목 단위(n={pl_cross['n_items']})로 구조적 feature와 Plackett-Luce 강도(pl_theta)의 "
          "Spearman 상관 상위:\n")
        a("| feature | spearman vs pl_theta |")
        a("| --- | ---: |")
        for c, rho in list(pl_cross["corr"].items())[:12]:
            a(f"| {c} | {rho:+.3f} |")
    else:
        a("- pl_theta 교차검증을 건너뜀(라벨/피처 컬럼 불일치 또는 데이터 부족).")

    # caveats
    a("\n## 9. 주의사항\n")
    a("- 모든 중요도는 **dX(차이) 공간**에서 산출됨: feature 자체가 아니라 '두 항목 간 차이'의 효과.")
    a("- permutation 중요도는 단일 시드(seed={}) 적합 모델에서 계산(학습 앙상블은 {}시드 평균).".format(cfg.seed, cfg.n_seeds))
    a("- RankSVM은 확률을 직접 내지 않아 decision score 기반으로 해석; RBF 커널이 선택되면 선형 계수 없음.")
    a("- EBM 고유 중요도는 main-effect 항만 feature 인덱스에 매핑(상호작용 항 제외).")

    txt = "\n".join(L) + "\n"
    with open(_os.path.join(out_dir, md_name), "w", encoding="utf-8") as f:
        f.write(txt)
    return txt


# --------------------------------------------------------------------------- #
def main(engine: str):
    _os.environ["DM_ENGINE"] = engine
    cfg = configure_paths(Config())
    apply_saved_semantic_config(cfg)

    out_dir = _os.path.join(HANDOFF_ROOT, "artifacts",
                            "analysis" if engine == "openai" else "analysis_anthropic")
    _os.makedirs(out_dir, exist_ok=True)
    md_name = "ANALYSIS_RESULTS.md" if engine == "openai" else "ANALYSIS_RESULTS_anthropic.md"

    layout = build_feature_layout(cfg)
    names = layout.names
    F = len(names)

    (X_train, _rel, ranks_train, _), (X_val, _, ranks_val, _), \
        (X_test, _, ranks_test, _), _scaler = build_arrays(cfg)
    dX, y = make_pairwise_dataset(X_train, ranks_train)
    print(f"[{engine}] features={F} pairwise_rows={dX.shape[0]} "
          f"val/test trials={len(ranks_val)}/{len(ranks_test)}")

    rng = np.random.default_rng(cfg.seed)
    per_model, native, block_share, err = {}, {}, {}, {}
    present = []
    for m in MODELS:
        try:
            make_model, prob_fn_factory = _model_funcs(m)
            params = load_tuned_params(cfg, f"{m}_best_params.json", _fallback_params(m), LABELS[m])
            model = make_model(params, cfg.seed)
            model.fit(dX, y)
            pfn = prob_fn_factory(model)
            d = permutation_importance(pfn, X_val, ranks_val, rng)
            per_model[m] = d
            native[m] = native_importance(model, m, F)
            pos_imp = np.clip(d["imp_top1"], 0, None)
            tot = pos_imp.sum() + 1e-12
            share = {}
            for blk in BLOCKS:
                lo, hi = getattr(layout, blk)
                share[blk] = float(pos_imp[lo:hi].sum() / tot)
            block_share[m] = share
            err[m] = error_margins(cfg.preds_dir, m)
            present.append(m)
            print(f"  {m}: base_top1={d['base_top1']:.4f} done")
        except Exception as e:
            print(f"  {m}: SKIPPED ({type(e).__name__}: {e})")

    # cross-model agreement
    agree = np.eye(len(present))
    for i in range(len(present)):
        for j in range(len(present)):
            rho, _ = spearmanr(per_model[present[i]]["imp_top1"],
                               per_model[present[j]]["imp_top1"])
            agree[i, j] = rho if np.isfinite(rho) else 0.0

    pl_cross = pl_theta_crosscheck(cfg)

    # plots
    plot_global_importance(per_model, names, out_dir)
    plot_block_contribution(block_share, out_dir)
    if any(err.get(m) for m in present):
        plot_error_margins(err, out_dir)
    plot_agreement(agree, present, out_dir)

    # machine-readable dump
    dump = {
        "engine": engine, "features": names,
        "per_model": {m: {
            "base_top1": per_model[m]["base_top1"],
            "base_ndcg": per_model[m]["base_ndcg"],
            "imp_top1": per_model[m]["imp_top1"].tolist(),
            "imp_ndcg": per_model[m]["imp_ndcg"].tolist(),
            "native_kind": native[m][1],
            "native": (native[m][0].tolist() if native[m][0] is not None else None),
            "block_share": block_share[m],
            "error": ({k: err[m][k] for k in ("med_correct", "med_wrong", "n", "acc")}
                      if err.get(m) else None),
        } for m in present},
        "agreement": {present[i]: {present[j]: float(agree[i, j])
                                   for j in range(len(present))} for i in range(len(present))},
        "pl_theta_crosscheck": pl_cross,
    }
    with open(_os.path.join(out_dir, "importance.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False)

    write_report(engine, cfg, names, layout, per_model, native, block_share,
                 err, agree, present, pl_cross, out_dir, md_name)
    print(f"[{engine}] -> {out_dir}/{md_name} (+ importance.json, 4 PNGs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Post-hoc analysis for the new pairwise rankers.")
    ap.add_argument("--engine", choices=["openai", "anthropic"], default="openai")
    main(ap.parse_args().engine)
