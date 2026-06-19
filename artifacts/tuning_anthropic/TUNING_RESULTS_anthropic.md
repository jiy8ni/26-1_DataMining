# 튜닝 결과 보고서 (Anthropic)

작성일: 2026-06-19 (median-imputation 수정 후 재실행)

> 이 보고서는 [TUNING_RESULTS.md](../tuning/TUNING_RESULTS.md)(OpenAI)와 동일한 파이프라인을
> `engine_filter="anthropic"` 설정으로 다시 실행한 결과입니다. 산출물이 OpenAI
> 아티팩트를 덮어쓰지 않도록 출력 경로를 `artifacts/tuning_anthropic/`,
> `artifacts/preds_anthropic/`로 분리했습니다.
>
> **재실행 노트**: 전처리의 결측치 median 대치를 split별 계산 → **train fold에서 계산해
> val/test/pool에 재사용**하도록 수정한 뒤(`src/data.py`, `src/inference.py`) 전체
> 파이프라인을 다시 실행했습니다. Anthropic 데이터의 V3 구조적 피처 역시 train fold에 결측이
> 사실상 없어, **모든 튜닝/테스트 지표는 수정 전과 수치적으로 동일**했습니다(방법론적 누수만
> 제거). 이번 실행부터는 사후 앙상블 `blend`도 표에 포함합니다.

## 범위

Anthropic 엔진 데이터(필터 후 8,610행, 고유 아이템 295개, 유효 trial 2,626개)에 대해
전체 CPU 튜닝을 다음 순서대로 실행했습니다:

- (PL 라벨 `data/processed/pl_labels_step2_anthropic.csv`는 기존 파일 재사용 — 순위 기반이라
  median 전처리 변경의 영향 없음. 필요 시 `python src/pl_fitting.py`로 재생성)
- `python src/tune/tune_semantic.py`
- `python src/tune/tune_mlp.py`
- `python src/tune/tune_lgbm.py`
- `python src/tune/tune_xgb.py`
- `python src/tune/tune_lgbm_pl.py`
- `python src/tune/tune_xgb_pl.py`

이후 자동 로드된 시맨틱 설정과 튜닝된 파라미터로 최종 트레이너를 실행하고, 마지막에
사후 앙상블(blend)을 실행했습니다:

- `python src/mlp/train.py`
- `python src/lightgbm/lgbm_train.py`
- `python src/xgboost/xgb_train.py`
- `python src/lightgbm/lgbm_train_pl.py`
- `python src/xgboost/xgb_train_pl.py`
- `python src/blend.py` (mlp/lgbm/xgb 저장 예측의 가중평균; val에서 가중치 선택)

## 시맨틱 최종 선택

LightGBM 프록시로 선택된 공통 시맨틱 설정:

| use_semantic_features | text_pca_dim | image_pca_dim | effective_feature_dim | cv_balanced | top1 | pairwise | ndcg@3 | tau | nll | brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `true` | 8 | 0 | 29 | 3.6265 | 0.7243 | 0.8079 | 0.9649 | 0.6159 | 1.1630 | 0.3925 |

동일한 시맨틱 스윕에서의 구조적 특징만 사용한 기준선(reference baseline):

| use_semantic_features | text_pca_dim | image_pca_dim | cv_balanced | top1 | pairwise | ndcg@3 | tau | nll | brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `false` | 0 | 0 | 0.8716 | 0.7223 | 0.8013 | 0.9639 | 0.6027 | 1.1704 | 0.3948 |

> OpenAI에서는 `text_pca_dim=16, image_pca_dim=8`(eff_dim=45)이 선택되었으나, Anthropic에서는
> 이미지 PCA가 도움이 되지 않았고 `text_pca_dim=8`(eff_dim=29)만 선택되었습니다. 또한 시맨틱 설정과
> 구조적 기준선의 per-metric 격차가 작아(top1 0.7243 vs 0.7223), 시맨틱 특징의 한계 이득은
> OpenAI보다 작았습니다.

## 교차검증(CV) 최종 선택

모든 튜너에서 선택 기준으로 balanced 점수를 사용했습니다.

| model | candidates | selected params | cv_balanced | top1 | pairwise | ndcg@3 | tau | nll | brier | temp |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lgbm` | 36 | `lr=0.05, leaves=31, min_child=20, reg_lambda=0.1` | 5.4127 | 0.7279 | 0.8080 | 0.9653 | 0.6161 | 1.1426 | 0.3788 | 0.48 |
| `xgb` | 72 | `eta=0.05, depth=4, min_child=3, gamma=0.5, reg_lambda=0.5` | 5.0841 | 0.7106 | 0.8027 | 0.9636 | 0.6053 | 1.1476 | 0.3908 | 0.44 |
| `xgb_pl` | 48 | `eta=0.03, depth=5, min_child=3, gamma=0.0, reg_lambda=1.0` | 4.3252 | 0.6940 | 0.7974 | 0.9621 | 0.5948 | 1.2118 | 0.4259 | 1.14 |
| `lgbm_pl` | 24 | `lr=0.05, leaves=31, min_child=30, reg_lambda=0.5` | 4.0472 | 0.7144 | 0.8067 | 0.9645 | 0.6135 | 1.1454 | 0.3992 | 0.42 |
| `mlp` | 36 | `hidden=[128,64,32], drop=0.3, wd=1e-4, lr=1e-3` | 3.9709 | 0.7125 | 0.8040 | 0.9637 | 0.6080 | 1.1888 | 0.4117 | 0.50 |

## 최종 테스트 결과

모든 최종 트레이너는 `artifacts/tuning_anthropic/semantic_best_config.json`과 각 모델별 튜닝 JSON을 자동 로드했습니다.
`blend`는 mlp/lgbm/xgb의 저장된 예측을 표준화 후 단순 가중평균하며, 가중치·온도를 **검증셋에서** 선택해 테스트에 적용합니다.

| model | avg temp | top1 | pairwise | ndcg@3 | tau | nll | brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lgbm` | 0.20 | 0.6787 | 0.7812 | 0.9575 | 0.5625 | 1.3412 | 0.4496 |
| `lgbm_pl` | 1.84 | 0.6713 | 0.7871 | 0.9589 | 0.5742 | 1.3079 | 0.4347 |
| `blend` | 0.50 | 0.6587 | 0.7729 | 0.9546 | 0.5458 | 1.5159 | 0.5061 |
| `mlp` | 0.70 | 0.6538 | 0.7729 | 0.9539 | 0.5458 | 1.6021 | 0.5100 |
| `xgb_pl` | 1.34 | 0.6450 | 0.7717 | 0.9550 | 0.5433 | 1.4283 | 0.4626 |
| `xgb` | 0.26 | 0.6288 | 0.7575 | 0.9512 | 0.5150 | 1.5946 | 0.5361 |

> `blend` 선택 가중치: `mlp=0.6, lgbm=0.0, xgb=0.4` (T*=0.5). 검증셋 balanced 지표는
> 높았으나(val top1=0.7527) 테스트에서는 0.6587로 중위권에 그쳤고, **최상위 단일 모델(lgbm
> 0.6787, lgbm_pl 0.6713)을 넘지 못했습니다.** 특히 `nll`/`brier`(캘리브레이션)는 표 내에서
> 가장 나쁜 축에 속해, OpenAI와 마찬가지로 운영 후보로는 부적합했습니다.

## 핵심 요약

- 공통 시맨틱 설정 `text_pca_dim=8, image_pca_dim=0`이 프록시 스윕에서 선택되었으나, 구조적 특징만
  사용한 기준선 대비 per-metric 이득은 미미했습니다(이미지 PCA는 선택되지 않음).
- median-imputation 수정 후에도 Anthropic 지표는 수치적으로 동일했습니다(V3 train fold 무결측 → 누수만 제거, 성능 변화 없음).
- Anthropic 데이터의 절대 지표는 전반적으로 OpenAI보다 높았습니다(예: CV top1 ~0.71–0.73, tau ~0.61).
- 최종 테스트에서는 LightGBM 계열이 가장 강력했습니다.
  - `lgbm`이 `top1_accuracy`에서 1위였습니다.
  - `lgbm_pl`(PL 목적함수 변형)이 `pairwise_accuracy`, `ndcg@3`, `kendall_tau`, `nll`, `brier_score`에서
    모두 1위로, 최종 테스트 기준 가장 균형 잡힌 모델이었습니다.
- OpenAI에서는 바닐라 트리(`lgbm`/`xgb`)가 최종 테스트를 지배했지만, Anthropic에서는 **PL 변형(`lgbm_pl`)이
  주요 랭킹·캘리브레이션 지표에서 바닐라 트리를 앞섰다**는 점이 가장 큰 차이입니다.
- `xgb`는 Anthropic 최종 테스트에서 가장 약했고, `mlp`는 랭킹 지표는 중위권이었으나 `nll`/`brier`(캘리브레이션)에서
  뒤처졌습니다.
- `blend`(mlp/lgbm/xgb 가중평균)는 검증셋 가중치 과적합으로 테스트 중위권에 머물러, 최상위 트리 모델을
  넘지 못했습니다.

## 실행 노트

- 생성된 튜닝 아티팩트(OpenAI 아티팩트와 분리됨):
  - `artifacts/tuning_anthropic/semantic_best_config.json`
  - `artifacts/tuning_anthropic/mlp_best_params.json`
  - `artifacts/tuning_anthropic/lgbm_best_params.json`
  - `artifacts/tuning_anthropic/xgb_best_params.json`
  - `artifacts/tuning_anthropic/lgbm_pl_best_params.json`
  - `artifacts/tuning_anthropic/xgb_pl_best_params.json`
  - 예측 점수: `artifacts/preds_anthropic/*.npz`
  - PL 라벨: `data/processed/pl_labels_step2_anthropic.csv` (기존 파일 재사용)
- 재현 방법: `src/config.py`에서 `engine_filter="anthropic"`, `tuning_dir`/`preds_dir`를
  `artifacts/tuning_anthropic`/`artifacts/preds_anthropic`로, `pl_labels_path`를
  `data/processed/pl_labels_step2_anthropic.csv`로 설정한 뒤 위 스크립트를 동일 순서로 실행합니다.
  (본 실행 후 `config.py`는 OpenAI 기본값으로 되돌려 두었습니다.)
- 결측치 median 대치를 train-fit/transform 패턴으로 수정(`src/data.py`의 `RankingDataset`이
  train fold median을 `self.medians`로 저장 → val/test/k-fold·`src/inference.py` pool 스코어링에서
  재사용). Anthropic V3 피처도 무결측이라 수치 변화는 없었습니다.
- 이번 실행에서는 OneDrive 파일 잠금으로 인한 `torch.save` 실패를 예방하기 위해 `cfg.ckpt_dir`을
  OneDrive 외부 경로로 임시 지정해 실행했고, `mlp/train.py`가 한 번에 정상 완료되었습니다.
  (OpenAI 실행 때는 기본 `checkpoints/` 경로에서 1차 시도가 `error code: 32`로 중단된 바 있음.)
- PL 튜닝/학습 중 `src/pl_objective.py:82-83`에서 수치 경고가 관찰되었습니다(`overflow encountered in
  multiply`/`divide`; XGBoost-PL에서는 `overflow encountered in accumulate`). 실행은 정상 완료되고
  아티팩트도 저장되었지만, 더 큰 탐색 전에 목적함수 안정화를 다루는 것이 좋습니다.
- `wandb`는 `WANDB_MODE=disabled`로 실행했습니다(오프라인 로그/네트워크 접근 없음).
