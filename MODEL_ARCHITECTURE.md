# Model Architecture

이 문서는 화장품 상품 추천 랭킹 시스템의 전체 모델 구조를 figure 제작 수준의 세부 정보로 기술합니다.

---

## 목차

1. [문제 설정](#1-문제-설정)
2. [데이터 전처리 파이프라인](#2-데이터-전처리-파이프라인)
3. [MLP Backbone (공유 구조)](#3-mlp-backbone-공유-구조)
4. [PL 라벨 생성 파이프라인](#4-pl-라벨-생성-파이프라인)
5. [Option B — Hybrid Loss (trial-level)](#5-option-b--hybrid-loss-trial-level)
6. [Option C — Pool-level KL Divergence](#6-option-c--pool-level-kl-divergence)
7. [두 Option 비교](#7-두-option-비교)
8. [후처리: Temperature Calibration](#8-후처리-temperature-calibration)
9. [평가 지표](#9-평가-지표)
10. [Feature 목록](#10-feature-목록)

---

## 1. 문제 설정

### 데이터 구조

- 전체 상품 pool: **295개** 화장품 상품 (각 `resolved_url`로 식별)
- **Trial**: 3개 상품을 묶어 AI 심사위원에게 보여주고 순위(rank 1–3)를 매긴 단위
- 상품당 평균 **40회** trial 등장, 최소 8회
- AI 심사 엔진: OpenAI / Anthropic (실험에서는 OpenAI만 사용)

### 학습 프로토콜

| 프로토콜 | 분할 기준 | 목적 |
|----------|-----------|------|
| `step1` | trial 단위 split | 본 적 있는 상품(seen-item) 평가 |
| `step2` | brand 단위 holdout | 처음 보는 상품(unseen-item) 일반화 평가 |

기본 실험은 **step2** 프로토콜 사용.

### 핵심 목표

```
입력: 상품 feature vector x_i  (수치 피처 20개 + 선택적 position feature)
출력: 상품 점수 s_i             (unbounded real)

추론 시: s_i를 전체 pool에 softmax → pool-level 추천 확률 P(i | pool)
```

---

## 2. 데이터 전처리 파이프라인

모든 모델(MLP, LightGBM, XGBoost)이 동일한 전처리를 거칩니다.

```
Raw CSV
  │
  ├─ [필터] is_ambiguous == True 인 trial 제거
  ├─ [필터] resolved_url 결측 행 제거
  │
  ▼
Log1p Transform   (skewed count/price 피처만 적용)
  │
  │  log1p(x) — 단, x = clip(x, lower=0) 후 적용
  │  대상 18개 컬럼: text_length, image_count, price_krw, ...
  │
  ▼
Median Imputation   (피처별 중앙값으로 NaN 대체)
  │  중앙값은 train set 기준으로 계산 후 val/test에 그대로 적용
  │
  ▼
StandardScaler      (μ=0, σ=1 정규화)
  │  Scaler는 train set으로 fit, val/test에 transform만 적용
  │
  ▼
[선택] Position Feature 추가
  │  sku_pos (1/2/3) → (sku_pos - 1) / 2  → 0.0 / 0.5 / 1.0
  │  Option B에서만 사용; Option C에서는 pool에 위치 개념 없으므로 제외
  │
  ▼
Feature Vector x_i
  │  Option B: shape (D,) = (21,)  [20 피처 + 1 position]
  │  Option C: shape (D,) = (20,)  [20 피처]
```

### 배치 구성 (Option B)

```
Trial 1개 = [item_A, item_B, item_C]   각 shape (D,)
Batch B개 = (B, 3, D)

학습 forward:
  (B, 3, D)  →  reshape  →  (B×3, D)  →  MLP  →  (B×3,)  →  reshape  →  (B, 3)
```

---

## 3. MLP Backbone (공유 구조)

`src/mlp/model.py` — `RecommendationScoreModel`

### 구조 다이어그램

```
Input
  │
  │  x_i ∈ ℝ^D        (D = 20 또는 21)
  │
  ▼
┌─────────────────────────────────────┐
│  Block 1                            │
│    Linear(D → 128)                  │
│    BatchNorm1d(128)                  │
│    ReLU                             │
│    Dropout(p=0.1)                   │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│  Block 2                            │
│    Linear(128 → 64)                 │
│    BatchNorm1d(64)                  │
│    ReLU                             │
│    Dropout(p=0.1)                   │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│  Block 3                            │
│    Linear(64 → 32)                  │
│    BatchNorm1d(32)                  │
│    ReLU                             │
│    Dropout(p=0.1)                   │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│  Output Layer                       │
│    Linear(32 → 1)                   │
│    squeeze(-1)                      │
└─────────────────────────────────────┘
  │
  ▼
Output
  │
  │  s_i ∈ ℝ           (unbounded real-valued score)
```

### 파라미터 요약

| 레이어 | 입력 차원 | 출력 차원 | 추가 모듈 |
|--------|-----------|-----------|-----------|
| Block 1 | D (20 or 21) | 128 | BN + ReLU + Dropout(0.1) |
| Block 2 | 128 | 64 | BN + ReLU + Dropout(0.1) |
| Block 3 | 64 | 32 | BN + ReLU + Dropout(0.1) |
| Output | 32 | 1 | — |

전체 파라미터 수 (D=21 기준):
- Block1: 21×128 + 128 = 2,816 (+ BN 256)
- Block2: 128×64 + 64 = 8,256 (+ BN 128)
- Block3: 64×32 + 32 = 2,080 (+ BN 64)
- Output: 32×1 + 1 = 33
- **총 ≈ 13,600개**

### Optimizer / Scheduler

| 항목 | 값 |
|------|----|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Max epochs | 100 |
| Early stopping patience | 15 (val loss 기준) |
| Batch size | 64 trial (= 192 items) |

---

## 4. PL 라벨 생성 파이프라인

`src/pl_fitting.py`

두 Option 모두 Plackett-Luce(PL) 모델로 생성한 라벨을 활용합니다.

### Plackett-Luce 모델

각 상품 i에 내재적 품질 점수 θ_i 가 있다고 가정합니다.

3개 상품 {A, B, C}가 경쟁할 때 순위 (A > B > C)가 나올 확률:

```
P(A>B>C) = [exp(θ_A) / (exp(θ_A)+exp(θ_B)+exp(θ_C))]
           × [exp(θ_B) / (exp(θ_B)+exp(θ_C))]

          = ∏_{k=1}^{K-1}  exp(θ_{σ_k}) / Σ_{j≥k} exp(θ_{σ_j})
```

여기서 σ는 관측된 순위 순열.

### MLE 목적함수 (L2 정규화 포함)

```
최소화:  -log P(모든 trials) + (l2/2) × ||θ||²

       = Σ_t [ logsumexp(θ_{i0}, θ_{i1}, θ_{i2})  - θ_{i0}    ← stage 1
             + logsumexp(θ_{i1}, θ_{i2})           - θ_{i1} ]  ← stage 2
         + (l2/2) × ||θ||²
```

- l2 = 1.0 (수치 안정성: θ 범위를 -4 ~ +4 수준으로 억제)
- 최적화: L-BFGS-B (maxiter=2000)
- 수렴 후 zero-mean 정규화: θ ← θ − mean(θ)

### 출력: rec_prob

```
rec_prob_i = softmax(θ)_i = exp(θ_i) / Σ_j exp(θ_j)

단, Σ_i rec_prob_i = 1.0  (전체 pool 합계)
```

### 두 Option에서 PL fitting 범위 차이

| | Option B | Option C |
|--|----------|----------|
| PL fitting 대상 | train + val + test **전체** | **train만** |
| 출력 파일 | `pl_labels_step2_openai.csv` | 메모리 내 계산 |
| 이유 | θ는 라벨로만 사용 (leakage 없음) | pool KL 학습에 직접 사용 → leakage 방지 |

---

## 5. Option B — Hybrid Loss (trial-level)

`src/mlp/train.py`, `src/loss.py`

### 전체 학습 흐름

```
┌──────────────────────────────────────────────────────────────────┐
│  Batch: B trials × 3 items                                       │
│                                                                  │
│  feats    : (B, 3, D)   — scaled feature vectors                 │
│  ranks    : (B, 3)      — AI ranks (1=best, 3=worst)             │
│  pl_theta : (B, 3)      — PL-fitted log-strength (ground truth)  │
└──────────────────────────────────────────────────────────────────┘
         │
         │  reshape: (B×3, D)
         ▼
    ┌─────────┐
    │   MLP   │   f_θ : ℝ^D → ℝ
    └─────────┘
         │
         │  reshape: (B, 3)
         ▼
  scores : (B, 3)   — raw model scores
         │
         ├────────────────────────────────┐
         ▼                                ▼
  PL Ranking Loss                   MSE Loss
  (ListMLE)                         F.mse_loss(scores, pl_theta)
         │                                │
         └──────────┬─────────────────────┘
                    ▼
         L = L_rank + λ × L_mse
         (λ = lambda_mse = 0.5)
```

### PL Ranking Loss (ListMLE) 수식

`src/loss.py` — `plackett_luce_loss`

```
입력: scores (B, K),  ranks (B, K)

1. 관측 순위대로 scores를 정렬 (rank 1이 index 0):
   sorted_scores[b, k] = scores[b, argsort(ranks[b])[k]]

2. Reverse cumulative logsumexp:
   log_cumsum[b, k] = log( Σ_{j≥k} exp(sorted_scores[b, j]) )
                    = logcumsumexp(sorted_scores, dim=1)[b, k]

3. NLL:
   nll_b = -Σ_{k=0}^{K-2}  (sorted_scores[b,k] - log_cumsum[b,k])

4. L_rank = mean(nll_b)  over batch
```

마지막 위치(k=K-1)는 결정적이므로 제외 (K=3이면 k=0,1만 합산).

### MSE Loss 수식

```
L_mse = (1/BK) × Σ_{b,k} (scores[b,k] - pl_theta[b,k])²
```

### 최종 Loss

```
L = L_rank + 0.5 × L_mse
```

### Early Stopping

- 학습 중: train loss = hybrid loss
- Early stopping 기준: **validation PL ranking loss** (MSE 항 제외)
  - val set에 pl_theta가 없어도 동작 가능하도록 분리

---

## 6. Option C — Pool-level KL Divergence

`src/mlp/train_pool.py`

### 핵심 차이

Option B는 trial 내 3개 상품의 **상대적 순서**를 학습합니다.  
Option C는 train pool 전체(258개) 상품에 대해 **절대 확률 분포**를 직접 학습합니다.

### 전체 학습 흐름

```
┌────────────────────────────────────────────────────────────────────┐
│  Train Pool (한 번만 구성, 에폭마다 재사용)                           │
│                                                                    │
│  X_pool   : (N_train, D)   — train 상품 전체 feature matrix        │
│  rec_prob : (N_train,)     — PL-fitted 추천 확률 (train만으로 계산) │
│                              Σ rec_prob_i = 1.0                    │
└────────────────────────────────────────────────────────────────────┘
         │
         │  매 epoch: pool 전체를 한 번에 forward
         ▼
    ┌─────────┐
    │   MLP   │   f_θ : ℝ^D → ℝ
    └─────────┘
         │
         │  scores: (N_train,)
         ▼
    log_softmax(scores, dim=0)   →   log_pred: (N_train,)
         │
         ▼
  KL Divergence
  L = KL(rec_prob || softmax(scores))
    = Σ_i rec_prob_i × (log rec_prob_i - log_pred_i)
    = F.kl_div(log_pred, rec_prob, reduction="sum")
```

### KL Loss Gradient (닫힌 형태)

```
∂L / ∂scores_i = softmax(scores)_i - rec_prob_i
               = predicted_prob_i  - pl_rec_prob_i
```

해석:
- predicted_prob_i > pl_rec_prob_i → 점수를 낮춰라
- predicted_prob_i < pl_rec_prob_i → 점수를 높여라

### Data Leakage 방지 구조

```
[잘못된 구조 — leakage 발생]

  PL fitting: train+val+test 전체 → test 상품 θ 포함
       ↓
  Pool KL 학습: 전체 pool → test 상품 점수를 직접 학습
       ↓
  test 평가: 사실상 답 보고 시험 (top1_acc 0.7555 — 부풀려진 수치)


[올바른 구조 — leakage 없음]

  PL fitting: train 데이터만 → train 상품 θ만 계산
       ↓
  Pool KL 학습: train pool만 (258개) → test 상품 점수 미학습
       ↓
  test 평가: 모델이 처음 보는 상품을 feature만으로 예측
             (top1_acc 0.5831 — 정직한 수치)
```

### Position Feature 사용 불가

Option C에서 position feature(sku_pos)를 쓸 수 없는 이유:  
KL 학습 시 "pool 전체"를 한 번에 넣는데, pool 내 개별 상품에 presentation order 개념이 없습니다.  
따라서 `cfg.use_position_feature = False` 고정 → input dim = D = 20.

---

## 7. 두 Option 비교

| 항목 | Option B (Hybrid Loss) | Option C (Pool KL) |
|------|----------------------|-------------------|
| 학습 단위 | trial 64개 × 3 items | train pool 전체 (258개) |
| 배치 텐서 크기 | (64, 3, D) | (258, D) |
| 학습 신호 | 상대적 순위 + 절대 calibration | pool-level 확률 분포 직접 학습 |
| Loss | L_rank + 0.5 × L_mse | KL(pl_rec_prob \|\| softmax(scores)) |
| PL 라벨 범위 | train+val+test 전체 | train 전체 |
| position feature | 사용 (D=21) | 사용 불가 (D=20) |
| Early stopping 기준 | val PL ranking loss | train KL loss |
| 일반화 방식 | feature로 순위 패턴 학습 후 새 상품 추론 | feature로 pool 내 절대 위치 학습 후 새 상품 추론 |

---

## 8. 후처리: Temperature Calibration

`src/calibration.py` — `TemperatureCalibration`

### 목적

MLP 점수 s_i 를 pool-level 확률로 변환할 때, softmax의 confidence를 조정합니다.

```
P(i | pool) = softmax(scores / T)_i = exp(s_i / T) / Σ_j exp(s_j / T)
```

T > 1: 분포를 더 평탄하게 (uncertainty 증가)  
T < 1: 분포를 더 첨예하게 (상위 상품 확률 집중)

### Grid Search

```
후보 T: [0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

for T in candidates:
    calibrated_scores = val_scores / T       # (N_val_trials, K)
    nll = PL_ranking_loss(calibrated_scores, val_ranks)

T* = argmin_{T} nll
```

모델 파라미터는 재학습하지 않음 — 점수에 상수를 나누는 post-hoc 보정.

### Test 시 확률 변환

```
raw_scores : (N_test_trials, 3)    — MLP 출력

exp_cal    = exp(raw_scores / T*)  # (N_test_trials, 3)
test_probs = exp_cal / Σ_k exp_cal_k  # row-wise softmax
           : (N_test_trials, 3)
```

---

## 9. 평가 지표

`src/metrics.py`

| 지표 | 설명 | 범위 |
|------|------|------|
| `top1_accuracy` | trial 내 rank-1 상품을 모델이 최고 점수로 예측한 비율 | 0–1 (↑) |
| `pairwise_accuracy` | trial 내 모든 상품 쌍 중 순서를 맞힌 비율 | 0–1 (↑) |
| `kendall_tau` | 예측 순위와 실제 순위의 Kendall τ 상관계수 | -1–1 (↑) |
| `nll` | Plackett-Luce NLL (보정된 확률 기준) | 0–∞ (↓) |
| `brier_score` | rank-1 상품 확률의 Brier score | 0–1 (↓) |

### 주요 성능 수치

| 모델 | top1_acc | pairwise_acc | kendall_τ | nll | brier |
|------|:--------:|:------------:|:---------:|:---:|:-----:|
| Option C (leakage) | 0.7555 | 0.8093 | 0.6186 | 1.1462 | 0.3488 |
| Option C (수정) | 0.5831 | 0.6952 | 0.3904 | 1.8343 | 0.5891 |

Random baseline (K=3): top1_acc ≈ 0.333

---

## 10. Feature 목록

`src/config.py`

### v2 Feature (기본, 20개)

| # | 피처명 | 설명 | Log transform |
|---|--------|------|:-------------:|
| 1 | `text_length` | 상품 설명 텍스트 길이 | ✓ |
| 2 | `image_count` | 이미지 수 | ✓ |
| 3 | `table_count` | 표 수 | ✓ |
| 4 | `list_item_count` | 리스트 항목 수 | ✓ |
| 5 | `paragraph_count` | 단락 수 | ✓ |
| 6 | `section_count` | 섹션 수 | ✓ |
| 7 | `jsonld_field_count` | JSON-LD 필드 수 | ✓ |
| 8 | `explicit_number_count` | 명시적 수치 표현 수 | ✓ |
| 9 | `ambiguous_term_count` | 모호한 표현 수 | ✓ |
| 10 | `numeric_specificity_ratio` | 수치 구체성 비율 | — |
| 11 | `price_krw` | 가격 (원) | ✓ |
| 12 | `skin_type_targets_count` | 피부 타입 타겟 수 | ✓ |
| 13 | `active_ingredient_count` | 활성 성분 수 | ✓ |
| 14 | `claim_keyword_count` | 효능 키워드 수 | ✓ |
| 15 | `texture_keyword_count` | 텍스처 키워드 수 | ✓ |
| 16 | `no_list_count` | 무(無)성분 표시 수 | ✓ |
| 17 | `cosmetic_cert_count` | 인증 수 | ✓ |
| 18 | `T7_eat_score` | T7 먹방 점수 | — |
| 19 | `Q4_social_proof_count` | 소셜 프루프 언급 수 | ✓ |
| 20 | `Q9_external_authority_count` | 외부 권위 언급 수 | ✓ |

v1 → v2 제거 이유 (결측률 초과):

| 피처 | 결측률 | 제거 사유 |
|------|--------|-----------|
| `ph_value` | 98.2% | 거의 모든 상품에 데이터 없음 |
| `aggregate_rating_value` | 38.8% | >20% 결측 기준 초과 |
| `aggregate_rating_count` | 38.5% | >20% 결측 기준 초과 |
| `volume_ml` | 19.9% | >20% 결측 기준 초과 |

### 선택적 피처 (Option B만)

| # | 피처명 | 계산 방법 | 값 범위 |
|---|--------|-----------|---------|
| 21 | `sku_pos` (정규화) | (sku_pos - 1) / 2 | 0.0 / 0.5 / 1.0 |
