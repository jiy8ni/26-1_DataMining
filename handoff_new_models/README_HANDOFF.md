# 신규 모델 작업 폴더 (handoff_new_models)

이 폴더는 **새 모델 4개를 추가**하기 위한 작업 공간입니다.
모델링을 처음 시작하는 분도 따라 할 수 있도록 개념부터 실행까지 차근차근 설명합니다.

> 핵심 한 줄: **"세 상품 중 AI가 매긴 순위를 우리 모델이 얼마나 잘 맞히는가"** 를 푸는 문제입니다.

---

## 0. 요약

1. 설치: `pip install -r requirements.txt`
2. 튜닝(좋은 설정 찾기): `python tune/tune_logreg.py` (4개 모델 각각)
3. 학습(최종 모델 만들기): `python logreg/logreg_train.py` (4개 모델 각각)
4. 앙상블(모델 합치기): `python blend_new.py`
5. 결과 지표는 화면에 출력되고, 예측은 `artifacts/preds/`에 저장됩니다.

이 폴더는 **원본 프로젝트(`src/` 등)를 절대 건드리지 않습니다.** 데이터는 원본에서 읽기만 하고,
결과물은 모두 이 폴더 안 `artifacts/`에만 저장됩니다. 마음껏 실험해도 원본은 안전합니다.

---

## 1. 우리가 푸는 문제가 뭔가요?

데이터의 한 단위는 **trial(시도)** 입니다. 하나의 trial = **상품 3개 + AI가 매긴 1·2·3등 순위**.

예시 (trial 하나):

| 상품 | 가격 | 리뷰수 | 이미지수 | ... | **AI 순위(정답)** |
|---|---|---|---|---|---|
| A | 19,000 | 1200 | 8 | ... | **1등** |
| B | 25,000 | 300  | 3 | ... | **3등** |
| C | 21,000 | 800  | 5 | ... | **2등** |

우리 모델의 목표: 상품들의 **feature(가격·리뷰수·이미지수·텍스트 임베딩 등)** 만 보고
**AI가 매긴 순위(A > C > B)를 똑같이 재현**하는 것입니다.

테스트는 "**처음 보는 브랜드**"로 평가합니다(`step2`). 즉, 브랜드를 외우는 게 아니라
"feature가 이러면 더 높은 순위" 라는 **일반적인 규칙**을 배워야 합니다.

---

## 2. "Pairwise(쌍 비교)" 방식이 핵심 아이디어

새 모델 4개는 모두 같은 아이디어를 씁니다. 순위를 한 번에 맞히는 대신,
**"둘 중 누가 더 위인가?"** 라는 쉬운 질문으로 쪼갭니다.

### 학습할 때
한 trial의 상품 3개로 **쌍(pair)** 을 만듭니다. (A vs B, A vs C, B vs C ...)

- 입력: 두 상품의 **feature 차이** → `ΔX = X_i − X_j`
  (예: A−B = `[가격차 -6000, 리뷰차 +900, ...]`)
- 정답: i가 j보다 순위가 높으면(=ai_rank 숫자가 작으면) `y = 1`, 아니면 `y = 0`
  (A는 1등, B는 3등 → A vs B의 정답은 `1`)

모델은 "**어떤 feature 차이가 있을 때 앞 상품이 이기는가**" 를 배웁니다.

### 예측할 때
각 상품이 **나머지 상품을 이길 점수/확률을 모두 더합니다.**

```
score(A) = P(A가 B를 이김) + P(A가 C를 이김)
score(C) = P(C가 A를 이김) + P(C가 B를 이김)
score(B) = ...
```

점수가 높은 순서대로 1·2·3등으로 정렬 → 이게 모델의 최종 예측입니다.

> 💡 왜 이렇게 하나요? 순위(ranking) 문제를 익숙한 **이진 분류(둘 중 승자 맞히기)** 로
> 바꿀 수 있어서, 일반적인 분류 모델(SVM·랜덤포레스트·로지스틱회귀 등)을 그대로 쓸 수 있습니다.

이 변환(쌍 만들기 ↔ 점수 합치기)은 **`pairwise.py`** 한 파일이 전부 처리합니다.
여러분은 각 모델을 "분류기"로만 생각하면 됩니다.

---

## 3. 4개 모델 소개 (PDF의 3+1)

| # | 모델 | 폴더 | 한 줄 설명 | 비고 |
|---|---|---|---|---|
| 1 | **RankSVM** | `ranksvm/` | SVM으로 "이기는 경계"를 최대 마진으로 학습 | 확률 대신 decision score 사용 |
| 2 | **Random Forest** | `rforest/` | 여러 결정트리의 투표(Bagging) | 기존 LightGBM(Boosting)과 대비 |
| 3 | **Logistic Regression** | `logreg/` | "이길 확률"을 직접 출력(Bradley–Terry) | 가장 단순·해석 쉬운 기준선 |
| 4 | **GAM / EBM** | `ebm/` | feature별 효과를 **곡선**으로 학습 | 효과 그래프 산출(해석용) |

모두 같은 입력(`ΔX`)·같은 정답(`y`)·같은 평가 방식을 씁니다. 차이는 "분류기 종류" 뿐입니다.

---

## 4. 폴더 안에 뭐가 들어있나요?

```
handoff_new_models/
│
├─ README_HANDOFF.md     ← 지금 읽는 문서
├─ requirements.txt      ← 필요한 라이브러리 목록
│
├─ 【여러분이 주로 볼 파일】
│  ├─ pairwise.py        쌍 만들기 ↔ 점수 합치기 (핵심 아이디어 구현)
│  ├─ harness.py         튜닝/학습 공통 루프 (모든 모델이 공유)
│  ├─ paths.py           경로 설정 (원본 데이터는 읽기, 결과는 이 폴더에만 저장)
│  ├─ blend_new.py       여러 모델을 가중평균으로 합치는 앙상블
│  │
│  ├─ tune/tune_ranksvm.py   ┐  각 모델의 "설정 찾기" 스크립트
│  ├─ tune/tune_rf.py        │  (어떤 하이퍼파라미터가 좋은지 교차검증으로 탐색)
│  ├─ tune/tune_logreg.py    │
│  ├─ tune/tune_ebm.py       ┘
│  │
│  ├─ ranksvm/ranksvm_train.py  ┐  각 모델의 "최종 학습" 스크립트
│  ├─ rforest/rf_train.py       │  (찾은 설정으로 학습하고 예측 저장)
│  ├─ logreg/logreg_train.py    │
│  └─ ebm/ebm_train.py          ┘
│
├─ 【원본에서 복사해온 파일 — 수정하지 마세요】
│  ├─ config.py        모든 설정값 (feature 목록, 경로, seed 수 등)
│  ├─ data.py          데이터 로딩 + 전처리 (결측치·스케일링·임베딩 PCA)
│  ├─ metrics.py       평가 지표 계산 (top1, ndcg@3 등)
│  ├─ calibration.py   확률 보정 (temperature scaling)
│  ├─ loss.py          순위 손실함수
│  ├─ preds_io.py      예측 결과 저장/불러오기
│  └─ tune/runtime.py, tune/cv_common.py   튜닝 보조 도구
│
└─ artifacts/          【결과물이 쌓이는 곳】
   ├─ tuning/   semantic_best_config.json(미리 복사됨) + <모델>_best_params.json(생성됨)
   ├─ preds/    <모델>_val.npz, <모델>_test.npz (생성됨)
   └─ ebm/      EBM feature 효과 그래프 (생성됨)
```

각 tuner/trainer 파일은 매우 짧습니다(20~40줄). 실제 일은 `pairwise.py`와 `harness.py`가 하고,
각 파일은 **"어떤 분류기를 쓸지"** 만 정합니다. 예를 들어 `tune_logreg.py`의 핵심은 이 부분:

```python
def make_model(cand, seed):           # 어떤 모델을 만들지
    return LogisticRegression(C=cand["C"], ...)

def prob_fn(model):                   # 이길 확률을 어떻게 뽑을지
    return lambda d: model.predict_proba(d)[:, 1]
```

전처리(결측치 median 채우기 · log 변환 · StandardScaler · 임베딩 PCA · 위치 feature)는
원본 `data.py`가 **자동으로** 해줍니다. 여러분이 따로 신경 쓸 필요 없습니다.

---

## 5. 단계별 실행 가이드

먼저 이 폴더로 이동하세요. (스크립트는 어느 위치에서 실행해도 경로를 알아서 잡습니다.)

```bash
cd handoff_new_models
```

### 0단계 — 설치 (최초 1회)
```bash
pip install -r requirements.txt
```
> `interpret`(EBM용)은 설치가 조금 오래 걸릴 수 있습니다.

### 1단계 — (선택) 빠른 동작 점검 `--smoke`
실제 튜닝은 시간이 걸리므로, 먼저 작은 설정으로 "에러 없이 도는지" 확인합니다.
결과는 `_smoke.json`으로 따로 저장되어 진짜 결과를 덮어쓰지 않습니다.

```bash
python tune/tune_logreg.py --smoke      # 1~2분 안에 끝나면 정상
```

### 2단계 — 튜닝 (좋은 하이퍼파라미터 찾기)
브랜드 단위 교차검증으로 최적 설정을 찾아 `artifacts/tuning/<모델>_best_params.json`에 저장합니다.

```bash
python tune/tune_logreg.py     # 빠름 (수십 초)
python tune/tune_ranksvm.py    # 보통
python tune/tune_rf.py         # 느림 (트리가 많음, 수 분)
python tune/tune_ebm.py        # 가장 느림 (outer_bags=8, 수 분~십수 분)
```
> 시간이 부족하면 2단계를 건너뛰어도 됩니다. 그러면 3단계에서 기본 설정으로 학습합니다
> (성능은 떨어질 수 있어요).

### 3단계 — 학습 (최종 모델 + 예측 저장)
찾은 설정으로 `n_seeds`(기본 5)개 모델을 학습해 평균 내고, test 지표를 출력하며,
예측을 `artifacts/preds/<모델>_{val,test}.npz`에 저장합니다.

```bash
python logreg/logreg_train.py
python ranksvm/ranksvm_train.py
python rforest/rf_train.py
python ebm/ebm_train.py         # 추가로 artifacts/ebm/ 에 feature 효과 그래프 저장
```

화면 출력 예시:
```
=== logreg Test Results (avg T=0.700) ===
  top1_accuracy          0.5xxx
  pairwise_accuracy      0.6xxx
  ndcg@3                 0.9xxx
  ...
  -> saved logreg_val.npz / logreg_test.npz to .../artifacts/preds
```

### 4단계 — 앙상블 (모델 합치기)
저장된 예측을 불러와 가중평균 조합을 탐색하고, 합친 결과와 단일 모델 결과를 비교합니다.

```bash
python blend_new.py
```
> 예측 파일이 있는 모델만 자동으로 포함합니다(2개 이상 필요).

---

## 6. 결과 지표 읽는 법

| 지표 | 의미 | 방향 | 메모 |
|---|---|---|---|
| **ndcg@3** | 전체 순위를 얼마나 잘 맞혔나 | 높을수록 ↑ | **메인 지표** |
| **top1_accuracy** | AI의 1등 상품을 1등으로 맞혔나 | 높을수록 ↑ | 실무적으로 중요 |
| **pairwise_accuracy** | 쌍 비교(둘 중 승자)를 맞힌 비율 | 높을수록 ↑ | 이 모델들의 직접 학습 목표 |
| kendall_tau | 순위 상관도 (-1~+1) | 높을수록 ↑ | 보조 |
| nll | 확률 예측 품질(음의 로그우도) | 낮을수록 ↓ | 보조 |
| brier_score | 확률 보정 정도 | 낮을수록 ↓ | 보조 |

**비교 기준선**(기존 모델 best, step2 test): LightGBM의 top1 ≈ 0.73, ndcg@3 ≈ 0.96.
새 모델이 이 근처면 잘 나온 것이고, 많이 낮으면 튜닝(2단계)을 다시 보세요.

가장 먼저 볼 것: **pairwise_accuracy**. 이 모델들은 쌍 비교를 직접 배우므로,
여기서부터 높게 나와야 순위 지표도 따라 올라갑니다.

---

## 7. 자주 나오는 질문 / 문제 해결

**Q. 튜닝을 안 하고 학습하면?**
→ "No tuned ... params found; using fallback defaults" 가 뜨고 기본 설정으로 학습합니다.
   동작은 하지만 성능은 보통 더 낮습니다. 가능하면 2단계를 먼저 하세요.

**Q. `ModuleNotFoundError: interpret`**
→ EBM 전용 라이브러리입니다. `pip install interpret`. (다른 3개 모델은 없어도 돌아갑니다.)

**Q. 화면에 빨간 경고(ConvergenceWarning, FutureWarning)가 많이 떠요.**
→ 대부분 무시해도 되는 경고입니다. 결과 지표가 정상 출력되면 학습은 성공한 것입니다.

**Q. 한글/특수문자 출력 에러 (`UnicodeEncodeError`, cp949)**
→ 콘솔 인코딩 문제입니다. `set PYTHONUTF8=1` (Windows) 후 다시 실행하면 해결됩니다.

**Q. 결과가 원본을 망가뜨리지 않나요?**
→ 안 됩니다. `paths.py`가 결과를 이 폴더 `artifacts/`로만 저장합니다. 원본 데이터는 읽기 전용입니다.

**Q. 더 넓게 튜닝하고 싶어요.**
→ 각 `tune/tune_*.py`의 `GRID` 딕셔너리에 값을 추가하세요. (RF/EBM은 시간이 오래 걸리니 주의)

**Q. 기존 모델(lgbm/xgb/mlp)까지 같이 앙상블하고 싶어요.**
→ 원본 `artifacts/preds/{lgbm,xgb,mlp}_{val,test}.npz`를 이 폴더 `artifacts/preds/`로 복사하고,
   `blend_new.py` 위쪽 `MODELS` 리스트에 이름을 추가하세요.

---

## 8. 새 모델을 더 추가하고 싶다면 (확장 방법)

`harness.py` 덕분에 새 분류기 추가는 매우 쉽습니다. 함수 2개만 정의하면 됩니다.

1. `tune/tune_<이름>.py` 를 기존 파일 복사해서 만들고, `make_model`/`prob_fn`/`GRID`만 수정
2. `<이름>/<이름>_train.py` 를 기존 trainer 복사해서 `DEFAULT_PARAMS`와 import 줄만 수정

`prob_fn`은 "feature 차이 묶음(M, D)을 받아 '앞 상품이 이길 점수(M,)'를 돌려주는 함수"면 됩니다.
- 확률 모델: `lambda d: model.predict_proba(d)[:, 1]`
- 점수 모델: `model.decision_function`

---

## 9. 주의사항 (꼭 읽어주세요)

- **`config.py`, `data.py`, `metrics.py` 등 "원본에서 복사한 파일"은 수정하지 마세요.**
  원본과 동기화 상태를 유지해야 합니다. 동작을 바꿔야 하면 새 스크립트에서 `cfg` 값을 바꾸세요.
- **RankSVM**은 확률을 직접 내지 않습니다. nll/brier 값은 참고용이며, 평가는 ndcg@3·top1 위주로.
- **EBM**의 입력은 "feature 차이(ΔX)"이므로, feature 효과 그래프도 "**차이**가 승리 확률에 주는 효과"를
  나타냅니다. 보고서에 이 점을 적어주세요.
- 이 폴더는 원본 커밋 `428f6f4`(2026-06-18) 기준으로 공용 파일을 복사했습니다.
  이후 원본 `src/`가 바뀌면, 복사해온 파일들을 다시 복사해 동기화하세요.

---

## 10. 한눈에 보는 데이터 흐름

```
data.py (build_arrays)                      ← 원본 데이터 읽기 + 전처리
   │  X(아이템별 feature), ai_rank
   ▼
pairwise.py (make_pairwise_dataset)         ← 쌍 만들기: ΔX, y
   │
   ▼
make_model() 으로 만든 분류기.fit(ΔX, y)     ← 각 모델이 정하는 부분
   │
   ▼
pairwise.py (score_items_from_pairwise)     ← 점수 합치기 → 상품별 점수 (B,3)
   │
   ▼
calibration.py (temperature) + metrics.py   ← 확률 보정 + 지표 계산
   │
   ▼
preds_io.py (save_scores) → artifacts/preds/<모델>_{val,test}.npz
   │
   ▼
blend_new.py                                ← 여러 모델 가중평균 앙상블
```
