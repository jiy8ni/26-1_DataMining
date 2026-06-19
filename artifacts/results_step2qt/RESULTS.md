# step2qt 결과 보고서 (query-type-aware)

작성일: 2026-06-19

## 범위

`data/raw/anthropic_various_query_type.csv` 기반의 query-type-aware 데이터셋(protocol
`step2qt`)에 대해 baseline(디폴트 파라미터)과 튜닝 후 성능을 비교한다. `query_type`(5종)과
`persona`(4종)를 원핫 인코딩해 피처에 추가했고, 나머지는 기존 step2 파이프라인(브랜드 홀드아웃,
시맨틱 임베딩 PCA, 시드 앙상블, 온도 캘리브레이션)을 그대로 재사용했다.

- 데이터: train 1,925 / val 339 / test 736 trial (브랜드 4개 홀드아웃, 누수 없음 확인)
- 입력 차원(디폴트): 구조 20 + 텍스트 PCA 16 + 이미지 PCA 8 + position 1 + 원핫 9 = **54**
- 모델 코드는 수정 없이 `DM_DATASET=query_type` 스위치로만 전환

산출물 폴더(신규):
- `artifacts/tuning_step2qt/` — 튜닝된 파라미터 JSON 6종
- `artifacts/preds_step2qt/` — 모델별 raw score (.npz)
- `artifacts/results_step2qt/` — 본 보고서 + 실행 로그(`logs/`)

## 테스트 결과: Baseline (디폴트 파라미터)

| model | top1 | pairwise | ndcg@3 | tau | nll | brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lgbm` | 0.6698 | 0.7536 | 0.9556 | 0.5072 | 1.4066 | 0.4377 |
| `xgb` | 0.6427 | 0.7341 | 0.9518 | 0.4683 | 1.4988 | 0.4934 |
| `mlp` | 0.6372 | 0.7391 | 0.9515 | 0.4783 | 1.4921 | 0.4992 |
| `xgb_pl` | 0.5557 | 0.7006 | 0.9420 | 0.4013 | 1.6738 | 0.5919 |
| `lgbm_pl` | 0.5530 | 0.6807 | 0.9387 | 0.3614 | 1.7720 | 0.5720 |
| `blend` | 0.6698 | 0.7536 | 0.9556 | 0.5072 | 1.4476 | 0.4364 |

## 테스트 결과: 튜닝 후

각 트레이너는 `artifacts/tuning_step2qt/semantic_best_config.json`과 모델별 튜닝 JSON을 자동
로드했다.

| model | top1 | pairwise | ndcg@3 | tau | nll | brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lgbm` | **0.6916** | **0.7690** | **0.9591** | **0.5380** | 1.3516 | **0.4143** |
| `xgb` | 0.6753 | 0.7609 | 0.9571 | 0.5217 | **1.3864** | 0.4301 |
| `mlp` | 0.6168 | 0.7233 | 0.9481 | 0.4466 | 1.5885 | 0.5224 |
| `lgbm_pl` | 0.6087 | 0.7396 | 0.9508 | 0.4792 | 3.3161 | 0.5429 |
| `xgb_pl` | 0.5584 | 0.6821 | 0.9392 | 0.3641 | 1.6477 | 0.5542 |
| `blend` | 0.6916 | 0.7690 | 0.9591 | 0.5380 | 1.3611 | 0.4143 |

## 튜닝 효과 (top1, baseline → tuned)

| model | baseline | tuned | Δ |
| --- | ---: | ---: | ---: |
| `lgbm` | 0.6698 | 0.6916 | **+0.0218** |
| `xgb` | 0.6427 | 0.6753 | **+0.0326** |
| `lgbm_pl` | 0.5530 | 0.6087 | **+0.0557** |
| `xgb_pl` | 0.5557 | 0.5584 | +0.0027 |
| `mlp` | 0.6372 | 0.6168 | **-0.0204** |

## 선택된 설정

공통 시맨틱 (LightGBM 프록시 스윕):

| use_semantic | text_pca_dim | image_pca_dim |
| --- | ---: | ---: |
| `true` | 16 | **0** |

> 튜너가 **이미지 임베딩을 제거**(image_pca_dim=0)하는 쪽을 선택했다. 이 제품 풀(45종)에서는
> CLIP 이미지 임베딩이 신호보다 노이즈에 가까웠다는 의미. 튜닝 후 유효 차원은 54 → 46.

모델별 선택 파라미터 (brand-CV balanced 기준):

| model | params |
| --- | --- |
| `lgbm` | `lr=0.05, num_leaves=31, min_child=50, reg_lambda=0.5, reg_alpha=0.1` |
| `xgb` | `eta=0.05, depth=5, min_child=5, gamma=0.5, reg_lambda=1.0` |
| `mlp` | `hidden=[96,48,24], drop=0.3, wd=1e-3, lr=1e-3` |
| `lgbm_pl` | `lr=0.05, num_leaves=31, min_child=30, reg_lambda=0.1` |
| `xgb_pl` | `eta=0.03, depth=5, min_child=5, gamma=0.5, reg_lambda=0.5` |

블렌드: 두 단계 모두 simplex 탐색이 **lgbm=1.0** (mlp=0, xgb=0)을 선택 → 블렌드 == 단일 lgbm.

## 핵심 요약

- **튜닝 후 LightGBM이 모든 주요 지표에서 최고** (test top1 0.6916, ndcg@3 0.9591, brier 0.4143).
  nll만 xgb(1.3864)가 근소하게 앞선다.
- 튜닝으로 트리 모델이 확실히 개선됐다(lgbm +2.2pt, xgb +3.3pt). PL 변형 중 `lgbm_pl`은
  baseline 대비 크게 올랐으나(+5.6pt) 절대 성능은 여전히 바닐라 트리에 못 미친다. `lgbm_pl`의
  튜닝 후 nll=3.32는 온도 캘리브레이션이 과신 방향으로 빠진 이상치이므로 확률 신뢰도가 필요한
  용도엔 부적합.
- **MLP는 튜닝으로 오히려 하락**(-2pt). 튜너가 더 큰 망 `[96,48,24]`를 골랐는데 45종 제품
  풀에서는 과적합으로 보인다. MLP는 baseline 설정([64,32])이 더 낫다.
- 블렌드가 lgbm 단독을 선택했다는 건 이 데이터에서 모델 다양성 이득이 없다는 뜻 — lgbm 하나로
  충분하다.
- 참고: 같은 step2 평가에서 원핫(query_type/persona)을 뺀 구조+시맨틱 baseline은 xgb test
  top1 0.561이었다. 원핫 추가로 +8pt, 튜닝까지 더하면 lgbm 기준 0.692로, **query 문맥이
  선호 예측에 지배적인 신호**임을 재확인.

## 권장

- 운영 모델은 **튜닝된 LightGBM**(`artifacts/tuning_step2qt/lgbm_best_params.json`).
- 시맨틱은 텍스트 PCA-16만 사용, 이미지 임베딩은 끈다.
- 추가 탐색 시: query_type×item 명시적 교차항, persona별 온도 분리, lgbm_pl 목적함수
  수치 안정화(과거 overflow 경고).

## 재현 방법

```bash
# 1) 데이터 생성
python src/prep_query_type.py
DM_DATASET=query_type python src/pl_fitting.py

# 2) baseline (디폴트 파라미터)
DM_DATASET=query_type WANDB_MODE=disabled python src/mlp/train.py        # 및 lgbm/xgb(+_pl), blend

# 3) 튜닝 → artifacts/tuning_step2qt/
DM_DATASET=query_type python src/tune/tune_semantic.py                   # 및 mlp/lgbm/xgb(+_pl)

# 4) 튜닝된 파라미터로 재학습 (자동 로드)
DM_DATASET=query_type WANDB_MODE=disabled python src/lightgbm/lgbm_train.py
```
