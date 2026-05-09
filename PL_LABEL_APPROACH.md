# PL 라벨 기반 학습 접근법

기존 모델은 "3개 상품 중 누가 더 좋냐"는 **상대적 순위**만 학습했습니다.  
이 문서는 각 상품에 **절대적인 추천 확률(rec_prob)** 을 부여하고, 이를 학습 목표로 활용하는 방법을 설명합니다.

---

## 왜 이 접근법이 필요한가

### 기존 모델의 한계

기존 모델은 3개 상품이 묶인 trial 단위로 학습합니다.

```
[A, B, C] → 모델 → [score_A, score_B, score_C] → A가 1위?
```

이 방식으로 학습한 모델에 상품 하나를 던져주면 스칼라 점수 하나가 나옵니다.  
그런데 이 점수 자체는 의미가 없습니다. **다른 상품의 점수와 비교해야만** 의미가 생깁니다.

### 목표

> "이 상품을 전체 300개 pool에 집어넣었을 때, 추천받을 확률이 얼마인가?"

이 질문에 답하려면 각 상품에 **pool 전체 기준의 절대적 확률**이 필요합니다.

---

## 핵심 개념: Plackett-Luce(PL) 모델

PL 모델은 비교 데이터로부터 각 상품의 **내재적 품질 점수(λ_i)** 를 역산합니다.

### 동작 원리

각 상품 i에 내재적 품질 점수 λ_i가 있다고 가정하면,  
3개 상품 {A, B, C}가 경쟁할 때 A가 1위를 차지할 확률은:

```
P(A가 1위) = λ_A / (λ_A + λ_B + λ_C)
```

A가 1위로 결정된 뒤, 나머지 {B, C}에서 B가 2위를 차지할 확률은:

```
P(B가 2위 | A가 1위) = λ_B / (λ_B + λ_C)
```

따라서 (A > B > C) 순위가 나올 전체 확률은:

```
P(A>B>C) = [λ_A / (λ_A+λ_B+λ_C)] × [λ_B / (λ_B+λ_C)]
```

### MLE (최대우도추정)

관측된 수천 개의 trial 순위 결과를 가장 잘 설명하는 λ_i 값들을 역산합니다.

**단순 1위 비율과의 차이:**  
단순 비율은 "몇 번 이겼나"만 봅니다.  
PL MLE는 "**누구를 상대로** 이겼나"까지 고려합니다.  
강한 상대를 이긴 승리가 약한 상대를 이긴 승리보다 더 높은 λ를 가져옵니다.

### 최종 rec_prob 계산

λ_i를 전체 pool에 softmax를 취하면 pool-level 추천 확률이 됩니다:

```
rec_prob_i = λ_i / (λ_1 + λ_2 + ... + λ_N)
```

---

## 데이터 조건 확인

PL fitting의 신뢰도는 상품당 trial 등장 횟수에 달려 있습니다.

| 항목 | 수치 |
|------|------|
| 전체 고유 상품 수 | 295개 |
| 상품당 평균 등장 횟수 | 40번 |
| 상품당 최소 등장 횟수 | 8번 |
| 10번 이상 등장한 상품 비율 | 257/258 (99.6%) |

상품당 평균 40번의 비교 데이터가 있어 PL 추정치가 안정적입니다.

---

## 구현: PL 라벨 생성 (`src/pl_fitting.py`)

전체 데이터(train + val + test)를 사용해 각 상품의 λ와 rec_prob을 추정합니다.

**왜 전체 데이터를 쓰는가?**  
train/val/test 데이터셋 자체가 실제 서비스 상품 전체를 **대표하도록 설계**되었습니다.  
따라서 "전체 pool 대비 추천 확률"을 구하려면 전체 데이터로 fitting하는 것이 맞습니다.

**수치 안정성을 위한 L2 정규화:**  
어떤 상품이 비교한 모든 trial에서 1위를 했다면 MLE가 λ → ∞ 로 발산합니다.  
이를 막기 위해 L2 정규화(`l2=1.0`)를 추가해 θ 범위를 -4 ~ +4 수준으로 억제합니다.

**출력 파일:** `data/processed/pl_labels_step2_openai.csv`

| 컬럼 | 설명 |
|------|------|
| `resolved_url` | 상품 식별자 |
| `pl_theta` | 로그 품질 점수 (zero-mean, 높을수록 좋음) |
| `rec_prob` | pool 전체 기준 추천 확률 (전 상품 합계 = 1.0) |
| `pool_rank` | pool 내 순위 (1 = 가장 추천 가능성 높음) |
| `top_pct` | 상위 몇 % (낮을수록 좋음) |

---

## Option B: Hybrid Loss (`src/train.py`)

기존 MLP 신경망에 PL 라벨을 **보조 손실 함수**로 추가합니다.

### 손실 함수

```
loss = PL_ranking_loss(triplet 내 순위) + λ × MSE(score, pl_theta)
```

- **PL ranking loss**: 기존과 동일. 3개 상품의 상대적 순위를 맞히도록 학습.
- **MSE 항**: 모델 출력 점수가 PL fitted θ에 가까워지도록 유도. 절대적 calibration 역할.
- **λ (lambda_mse)**: 두 항의 균형을 조절하는 가중치. 기본값 0.5.

### 직관적 의미

| 항 | 역할 |
|----|------|
| PL ranking loss | "A, B, C 중 순서를 맞혀라" |
| MSE(score, pl_theta) | "이 상품의 점수가 전체 pool에서 어느 위치여야 하는지를 알아라" |

두 신호가 서로 보완합니다.  
순위 신호만 있으면 점수의 절대값이 의미 없고,  
절대 calibration만 있으면 미세한 순위 차이를 놓칩니다.

### 수정된 파일

| 파일 | 변경 내용 |
|------|-----------|
| `src/loss.py` | `hybrid_loss(scores, ranks, pl_theta, lambda_mse)` 함수 추가 |
| `src/config.py` | `pl_labels_path`, `lambda_mse=0.5` 파라미터 추가 |
| `src/data.py` | 데이터셋 로딩 시 pl_theta 병합, 배치에 포함 |
| `src/train.py` | `_forward`, `train_epoch`에서 hybrid_loss 사용 |

---

## Option C: Pool-level KL Divergence (`src/train_pool.py`)

모델이 처음부터 **pool-level 확률 분포**를 직접 학습하도록 합니다.

### 손실 함수

```
predicted_prob = softmax(model scores over 전체 train pool)
loss = KL(pl_rec_prob_train || predicted_prob)
```

KL divergence의 gradient는 닫힌 형태로 계산됩니다:

```
∂loss/∂score_i = predicted_prob_i - pl_rec_prob_i
```

"내가 예측한 확률"이 "PL이 추정한 확률"보다 높으면 점수를 낮추고, 낮으면 높입니다.

### Option B와의 구조적 차이

| 항목 | Option B (Hybrid Loss) | Option C (Pool KL) |
|------|----------------------|-------------------|
| 학습 배치 단위 | trial 64개 × 3 상품 | train pool 전체 (258개) |
| 학습 신호 | 상대적 순위 + 절대 calibration | pool-level 확률 분포 직접 학습 |
| position feature | 사용 가능 | 사용 불가 (pool에 위치 개념 없음) |
| early stopping 기준 | val PL ranking loss | train KL loss (개선 없으면 멈춤) |

### Data Leakage 문제와 해결

**처음 구현의 문제:**  
PL fitting을 train+val+test 전체로 하고, 모델도 전체 pool로 학습했습니다.  
이 경우 test 상품의 품질 정보가 학습에 포함되어 **data leakage** 가 발생합니다.

```
[잘못된 구조]
PL fitting (전체 데이터) → test 상품 pl_theta 포함
       ↓
모델 학습 (전체 pool KL) → test 상품 점수를 직접 학습
       ↓
test 평가 → 사실상 "답 보고 시험 보기"
```

leakage 있을 때 top1_accuracy: **0.7555**

**수정된 구조:**  
PL fitting과 KL 학습 모두 **train 상품만** 사용합니다.  
test 상품은 모델이 feature만 보고 점수를 **예측**해야 합니다.

```
[올바른 구조]
PL fitting (train 데이터만) → train 상품 pl_theta
       ↓
모델 학습 (train pool KL) → train 상품 점수만 학습
       ↓
test 평가 → 모델이 처음 보는 상품을 feature로만 예측
```

leakage 수정 후 top1_accuracy: **0.5831** (정직한 수치)

### 최종 성능 비교

| 모델 | top1_accuracy | pairwise_accuracy | kendall_tau | nll | brier_score |
|------|:---:|:---:|:---:|:---:|:---:|
| Option C (leakage 있음) | 0.7555 | 0.8093 | 0.6186 | 1.1462 | 0.3488 |
| Option C (수정 후) | 0.5831 | 0.6952 | 0.3904 | 1.8343 | 0.5891 |

---

## 전체 파일 구조

```
src/
├── pl_fitting.py       # [신규] PL MLE fitting → rec_prob 라벨 생성
├── train.py            # [수정] Option B: PL ranking + MSE hybrid loss
├── train_pool.py       # [신규] Option C: pool-level KL divergence
├── loss.py             # [수정] hybrid_loss 함수 추가
├── config.py           # [수정] pl_labels_path, lambda_mse 추가
├── data.py             # [수정] 배치에 pl_theta 포함
│
├── lgbm_train.py       # 변경 없음
├── xgb_train.py        # 변경 없음
├── model.py            # 변경 없음
├── metrics.py          # 변경 없음
└── calibration.py      # 변경 없음

data/processed/
└── pl_labels_step2_openai.csv   # [신규] 상품별 PL 라벨
```

---

## 실험 재현 순서

```bash
# 1. PL 라벨 생성 (전체 데이터 사용)
python src/pl_fitting.py

# 2. Option B: Hybrid Loss 학습
python src/train.py

# 3. Option C: Pool-level KL 학습 (leakage 없는 버전)
python src/train_pool.py
```
