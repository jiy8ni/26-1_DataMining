# 튜닝 결과 보고서

작성일: 2026-06-18

## 범위

전체 CPU 튜닝을 다음 순서대로 실행했습니다:

- `python src/tune/tune_semantic.py`
- `python src/tune/tune_mlp.py`
- `python src/tune/tune_lgbm.py`
- `python src/tune/tune_xgb.py`
- `python src/tune/tune_lgbm_pl.py`
- `python src/tune/tune_xgb_pl.py`

이후 자동 로드된 시맨틱 설정과 튜닝된 파라미터로 최종 트레이너를 실행했습니다:

- `python src/mlp/train.py`
- `python src/lightgbm/lgbm_train.py`
- `python src/xgboost/xgb_train.py`
- `python src/lightgbm/lgbm_train_pl.py`
- `python src/xgboost/xgb_train_pl.py`

## 시맨틱 최종 선택

LightGBM 프록시로 선택된 공통 시맨틱 설정:

| use_semantic_features | text_pca_dim | image_pca_dim | effective_feature_dim | cv_balanced | top1 | pairwise | ndcg@3 | tau | nll | brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `true` | 16 | 8 | 45 | 4.0841 | 0.6955 | 0.7717 | 0.9582 | 0.5434 | 1.3085 | 0.4284 |

동일한 시맨틱 스윕에서의 구조적 특징만 사용한 기준선(reference baseline):

| use_semantic_features | text_pca_dim | image_pca_dim | cv_balanced | top1 | pairwise | ndcg@3 | tau | nll | brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `false` | 0 | 0 | -6.4649 | 0.6293 | 0.7301 | 0.9488 | 0.4602 | 1.4275 | 0.4897 |

## 교차검증(CV) 최종 선택

모든 튜너에서 선택 기준으로 balanced 점수를 사용했습니다.

| model | candidates | selected params | cv_balanced | top1 | pairwise | ndcg@3 | tau | nll | brier | temp |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lgbm` | 36 | `lr=0.05, leaves=31, min_child=50, reg_lambda=0.5` | 6.0689 | 0.6955 | 0.7717 | 0.9582 | 0.5434 | 1.3085 | 0.4284 | 0.32 |
| `xgb` | 72 | `eta=0.05, depth=5, min_child=5, gamma=0.5, reg_lambda=0.5` | 5.3997 | 0.6760 | 0.7623 | 0.9558 | 0.5247 | 1.3238 | 0.4372 | 0.54 |
| `lgbm_pl` | 24 | `lr=0.03, leaves=31, min_child=50, reg_lambda=0.1` | 5.2587 | 0.6689 | 0.7616 | 0.9557 | 0.5233 | 1.3532 | 0.4560 | 1.32 |
| `xgb_pl` | 48 | `eta=0.03, depth=5, min_child=5, gamma=0.0, reg_lambda=0.5` | 4.4769 | 0.6388 | 0.7360 | 0.9500 | 0.4720 | 1.4636 | 0.5025 | 1.98 |
| `mlp` | 36 | `hidden=[128,64,32], drop=0.1, wd=1e-4, lr=5e-4` | 4.4735 | 0.6815 | 0.7721 | 0.9579 | 0.5442 | 1.2984 | 0.4397 | 0.50 |

## 최종 테스트 결과

모든 최종 트레이너는 `artifacts/tuning/semantic_best_config.json`과 각 모델별 튜닝 JSON을 자동 로드했습니다.

| model | avg temp | top1 | pairwise | ndcg@3 | tau | nll | brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lgbm` | 0.20 | 0.7286 | 0.7832 | 0.9607 | 0.5664 | 1.4093 | 0.3972 |
| `xgb` | 0.32 | 0.7139 | 0.7755 | 0.9589 | 0.5509 | 1.2796 | 0.3948 |
| `lgbm_pl` | 1.04 | 0.6834 | 0.7677 | 0.9556 | 0.5355 | 1.5156 | 0.4431 |
| `xgb_pl` | 1.00 | 0.6614 | 0.7596 | 0.9534 | 0.5192 | 1.4327 | 0.4619 |
| `mlp` | 0.70 | 0.6394 | 0.7490 | 0.9495 | 0.4980 | 1.6971 | 0.4916 |

## 핵심 요약

- 공통 시맨틱 설정 `text_pca_dim=16, image_pca_dim=8`이 프록시 스윕에서 구조적 특징만 사용한 경우보다 확실히 우수했습니다.
- 이번 실행의 최종 테스트 지표에서는 바닐라 트리 모델 계열이 여전히 가장 강력했습니다.
- `lgbm`이 `top1_accuracy`, `pairwise_accuracy`, `ndcg@3`, `kendall_tau`에서 최고 성능을 보였습니다.
- `xgb`가 최종 `nll`과 `brier_score`에서 가장 좋았습니다.
- 튜닝된 PL 변형들은 비교용으로 유지할 만큼 경쟁력이 있었지만, 주요 테스트 지표에서는 여전히 바닐라 트리 모델에 뒤처졌습니다.
- `mlp`는 튜닝을 통해 개선되었으나, 최종 테스트 성능에서는 트리 기반 모델보다 뒤처진 상태로 남았습니다.

## 실행 노트

- 생성된 튜닝 아티팩트:
  - `artifacts/tuning/semantic_best_config.json`
  - `artifacts/tuning/mlp_best_params.json`
  - `artifacts/tuning/lgbm_best_params.json`
  - `artifacts/tuning/xgb_best_params.json`
  - `artifacts/tuning/lgbm_pl_best_params.json`
  - `artifacts/tuning/xgb_pl_best_params.json`
- PL 튜닝 중 `src/pl_objective.py:83`에서 수치 경고가 관찰되었습니다(`overflow encountered in multiply`; XGBoost-PL에서는 `overflow encountered in accumulate` 경고도 발생). 실행은 정상 완료되고 아티팩트도 저장되었지만, 더 큰 탐색을 진행하기 전에 목적함수 안정화를 다루는 것이 좋습니다.
- `wandb`가 `%LOCALAPPDATA%\\wandb\\logs`에 대해 `Access is denied`를 보고했으나, 오프라인 실행은 정상 완료되었고 로컬 요약은 저장소의 `wandb/` 디렉터리에 저장되었습니다.
- `src/mlp/train.py`에서 재실행 중 Windows 콘솔의 유니코드 대시(–) 출력 문제가 있었습니다. ASCII 전용 출력으로 변경한 뒤 정상적으로 재실행되었습니다.
