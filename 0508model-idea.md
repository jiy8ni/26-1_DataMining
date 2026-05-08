# 0508 모델 기획

# 폼클렌저 상품 페이지의 AI 추천 가능성 예측 모델 구축 계획

## 1. 한 줄 요약

1. 폼클렌저 3개씩 AI에게 보여주고 1·2·3위 ranking을 받습니다.
2. 이 ranking은 “AI의 선호 데이터”이므로 **Preference Learning**으로 학습합니다.
3. 모델은 상품 페이지별 추천 score를 예측하고, **Plackett–Luce ranking loss**로 실제 AI 순위와 비슷해지도록 학습합니다.
4. 학습 후 **300개 pool 내 순위**와 **추천 확률**을 계산합니다.
    
    → 모델 결과의 현실적인 유용성은 300개의 pool이 얼마나 대표성 있는 pool인지에 따라 달려 있음. (자사 상품, 주요 경쟁 상품, 가격대별 상품, 브랜드별 상품, 피부 타입, 효능별 상품, 인기 상품과 비인기 상품 등 포괄적으로 구성되어야 함)
    

---

## 2. 프로젝트 목표

### 우리가 알고 싶은 것

특정 폼클렌저 상품 페이지가 있을 때, 이 페이지가 AI에게 얼마나 추천될 가능성이 높은지 예측하고자 한다.

최종적으로 각 상품 페이지에 대해 아래 지표를 산출하는 것이 목표다.

| 산출 지표 | 의미 |
| --- | --- |
| Recommendation score | AI가 해당 상품을 추천할 만한 정도를 나타내는 모델 점수 |
| Relative ranking | 폼클렌저 pool 내 상대 순위 |
| Percentile | 전체 폼클렌저 상품 중 상위 몇 %인지 |
| Pool 기준 추천 확률 | 폼클렌저 pool 안에서 해당 상품이 선택될 확률 |
| Odds uplift | 기준 상품 또는 평균 상품 대비 추천 가능성이 몇 배 높은지 |

---

## 3. 문제 정의

우리가 직접 알고 싶은 값은 다음과 같다.

```
이 폼클렌저 상품 페이지가 AI에게 추천될 확률은 얼마인가?
```

하지만 현실적으로 모든 폼클렌저 상품을 한 번에 AI에게 보여주고 전체 순위를 매기게 하기는 어렵다.

따라서 데이터 수집은 다음 방식으로 진행한다.

```
폼클렌저 상품 3개를 후보로 제시
→ AI가 1위, 2위, 3위를 선택
→ 이 ranking 데이터를 학습에 사용
```

예시:

```
후보: A, B, C
AI ranking: A > C > B
```

이 데이터는 단순히 “A는 추천됨, B는 추천 안 됨”이 아니라 다음과 같은 상대적 선호 정보를 포함한다.

```
A가 C보다 선호됨
A가 B보다 선호됨
C가 B보다 선호됨
```

따라서 이 문제는 일반적인 classification 문제가 아니라 **Learning to Rank / Preference Learning 문제**로 보는 것이 적절하다.

Learning to Rank는 보통 pointwise, pairwise, listwise 접근으로 나뉘며, 우리의 경우처럼 후보 리스트 전체의 순서를 학습하는 경우는 **listwise ranking**에 해당한다. ([Computer Laboratory](https://www.cl.cam.ac.uk/teaching/1516/R222/l2r-overview.pdf?utm_source=chatgpt.com))

---

## 4. 핵심 접근법

### 전체 구조

모델은 2단계로 구축한다.

```
1단계: 상품 페이지 → recommendation score 예측
2단계: recommendation score → 폼클렌저 pool 기준 추천 확률 변환
```

---

## 5. 1단계: Recommendation score 모델

### 개념

각 상품 페이지의 정보를 입력하면, 모델이 해당 상품의 추천 가능성을 나타내는 score를 출력한다.

```
상품 페이지 정보
→ feature 추출
→ ML 모델
→ recommendation score
```

수식으로 표현하면 다음과 같다.

$$
s_i = f_\theta(x_i)
$$

| 기호 | 의미 |  |
| --- | --- | --- |
| $(x_i)$ | 상품 (i)의 페이지 feature |  |
| $(s_i)$ | 상품 (i)의 recommendation score |  |
| $(f_\theta)$ | 학습할 머신러닝 모델 |  |

이 score는 처음부터 확률이 아니다.

먼저 **AI가 해당 상품을 상대적으로 얼마나 선호할지 나타내는 latent score**로 해석한다.

---

## 6. 사용할 데이터

### 상품 페이지 feature

각 폼클렌저 상품 페이지에서 아래와 같은 정보를 feature로 만든다.

| Feature 유형 | 예시 |
| --- | --- |
| 텍스트 정보 | 상품명, 상세 설명, 효능 문구, 성분 설명 |
| 이미지 정보 | 대표 이미지, 상세 이미지, 이미지 embedding |
| 가격 정보 | 가격, 할인율, 용량 대비 가격 |
| 리뷰 정보 | 평점, 리뷰 수, 리뷰 키워드 |
| 브랜드 정보 | 브랜드명, 브랜드 인지도 |
| 제품 속성 | 약산성, 저자극, 피부 타입, 주요 성분, 용량 |
| 페이지 품질 | 상세 설명 길이, 이미지 품질, 정보 구성 |

---

### Ranking 데이터

AI에게 폼클렌저 상품 3개를 보여주고 1위, 2위, 3위를 받는다.

| trial_id | 후보 상품 | AI ranking |
| --- | --- | --- |
| 1 | A, B, C | A > C > B |
| 2 | D, E, F | E > D > F |
| 3 | A, G, H | H > A > G |

학습용 데이터는 아래 형태로 관리할 수 있다.

| trial_id | item_id | rank | page_features |
| --- | --- | --- | --- |
| 1 | A | 1 | A의 페이지 feature |
| 1 | C | 2 | C의 페이지 feature |
| 1 | B | 3 | B의 페이지 feature |
| 2 | E | 1 | E의 페이지 feature |
| 2 | D | 2 | D의 페이지 feature |
| 2 | F | 3 | F의 페이지 feature |

---

## 7. Plackett–Luce ranking loss

### 왜 필요한가

AI가 준 데이터는 3개 후보의 순위다.

예:

```
A > C > B
```

우리는 모델이 이 순서를 재현하도록 학습해야 한다.

즉 모델 score가 다음과 같은 관계를 갖도록 학습한다.

```
score(A) > score(C) > score(B)
```

이를 위해 사용할 수 있는 대표적인 loss가 **Plackett–Luce ranking loss**다.

Plackett–Luce 모델은 ranking 데이터를 모델링하는 대표적인 방식이며, 소비자가 여러 제품 중 선호 순위를 매기는 상황에도 사용할 수 있다. 이 모델은 item별 worth, 즉 선호도를 추정하는 방식으로 설명된다. ([Hturner](https://hturner.github.io/PlackettLuce/?utm_source=chatgpt.com))

- **Plackett–Luce ranking loss란?**
    
    우리가 수집하는 데이터는 “상품 3개 중 AI가 1위, 2위, 3위를 고른 결과”입니다.
    
    예를 들어 한 trial에서 AI가 다음과 같이 순위를 매겼다고 합시다.
    
    ```
    후보: A, B, C
    AI ranking: A > C > B
    ```
    
    모델은 각 상품에 대해 recommendation score를 예측합니다.
    
    | 상품 | 모델 score |
    | --- | --- |
    | A | $s_A$ |
    | B | $s_B$ |
    | C | $s_C$ |
    
    Plackett–Luce 모델은 이 ranking이 나올 확률을 다음처럼 계산합니다.
    
    $$
    P(A≻C≻B)=\frac{e^{s_A}}{e^{s_A}+e^{s_B}+e^{s_C}}⋅\frac{e^{s_C}}{e^{s_C}+e^{s_B}}
    
    $$
    
    이 식의 의미는 직관적입니다.
    
    먼저 A, B, C 세 개 중에서 A가 1위로 선택될 확률을 계산합니다.
    
    $$
    \frac{e^{s_A}}{e^{s_A}+e^{s_B}+e^{s_C}}
    $$
    
    그 다음 A를 제외한 나머지 B, C 중에서 C가 2위로 선택될 확률을 계산합니다.
    
    $$
    \frac{e^{s_C}}{e^{s_C}+e^{s_B}}
    $$
    
    이 둘을 곱하면 전체 ranking, 즉 A > C > B가 나올 확률이 됩니다.
    
    ---
    
    ### 이걸 loss로 어떻게 쓰는가?
    
    모델 학습에서는 실제 AI가 매긴 ranking의 확률이 높아지도록 학습합니다.
    
    즉 모델이 실제 ranking에 높은 확률을 주면 좋은 모델이고, 낮은 확률을 주면 penalty를 받습니다.
    
    그래서 loss는 보통 negative log likelihood로 정의합니다.
    
    $$
    \mathcal{L} = -\log P(A \succ C \succ B)
    $$
    
    위 식을 풀면:
    
    $$
    \mathcal{L}=-\left[s_A-\log(e^{s_A}+e^{s_B}+e^{s_C})+s_C-\log(e^{s_C}+e^{s_B})\right]
    $$
    
    쉽게 말하면:
    
    ```
    실제 ranking: A > C > B
    모델이 원하는 방향: score(A) > score(C) > score(B)
    ```
    
    가 되도록 학습하는 loss입니다.
    
    ---
    
    ### 왜 이 loss가 적절한가?
    
    이 loss는 단순히 1위만 맞히는 것이 아니라, **1위, 2위, 3위 순서 전체를 학습에 사용**합니다.
    
    예를 들어 A>C>BA > C > BA>C>B라는 ranking에는 다음 정보가 모두 들어 있습니다.
    
    ```
    A가 C보다 선호됨
    A가 B보다 선호됨
    C가 B보다 선호됨
    ```
    
    Plackett–Luce ranking loss는 이 정보를 한 번에 반영합니다.
    
    따라서 단순 classification보다 ranking 데이터의 정보를 더 효율적으로 활용할 수 있습니다.
    
    ---
    
    ### 우리 프로젝트에서의 역할
    
    우리 모델은 상품 페이지 feature를 보고 score를 냅니다.
    
    ```
    상품 페이지 feature
    → 모델
    → recommendation score
    ```
    
    그리고 같은 trial 안의 세 상품 score를 Plackett–Luce loss에 넣습니다.
    
    ```
    실제 AI ranking
    vs
    모델 score 기반 ranking probability
    ```
    
    이 과정을 반복하면 모델은 “AI가 선호할 만한 상품 페이지의 특징”을 학습하게 됩니다.
    
    즉 Plackett–Luce ranking loss는 **AI ranking 데이터를 ML 모델의 학습 신호로 바꾸는 핵심 장치**입니다.
    
- Plackett-Luce ranking loss 구체적인 숫자 예시
    
    예를 들어 한 번의 비교에서 AI가 이렇게 순위를 줬다고 해볼게요.
    
    ```
    후보: A, B, C
    AI ranking: A > C > B
    ```
    
    모델이 각 상품에 대해 score를 냅니다.
    
    | 상품 | score |
    | --- | --- |
    | A | 2.0 |
    | B | 0.5 |
    | C | 1.0 |
    
    Plackett–Luce는 ranking을 **순차 선택**으로 봅니다. 즉, 먼저 3개 중 1위를 고르고, 그다음 남은 2개 중 2위를 고르는 방식입니다. ranking 데이터에 쓰이는 대표 모델이고, 소비자 선호 순위 같은 상황에도 사용됩니다. ([Hturner](https://hturner.github.io/PlackettLuce/?utm_source=chatgpt.com))
    
    ---
    
    ## 1단계: A가 1위일 확률
    
    $$
    P(A \text{ is 1st}) = \frac{e^{2.0}}{e^{2.0}+e^{0.5}+e^{1.0}}
    $$
    
    대략 계산하면:
    
    ```
    e^2.0 = 7.39
    e^1.0 = 2.72
    e^0.5 = 1.65
    ```
    
    그래서:
    
    $$
    P(A \text{ is 1st}) = \frac{7.39}{7.39+2.72+1.65} =0.628
    $$
    
    ---
    
    ## 2단계: 남은 C, B 중 C가 2위일 확률
    
    A가 1위로 뽑혔으니 이제 남은 건 C와 B입니다.
    
    $$
    P(C \text{ is 2nd} \mid A \text{ is 1st}) = \frac{e^{1.0}}{e^{1.0}+e^{0.5}} = \frac{2.72}{2.72+1.65} = 0.622
    $$
    
    ---
    
    ## 전체 ranking 확률
    
    따라서 모델이 `A > C > B`라는 실제 AI ranking에 부여한 확률은:
    
    $$
    P(A>C>B) = 0.628 \times 0.622 = 0.391
    $$
    
    즉 모델은 실제 ranking에 약 **39.1% 확률**을 준 것입니다.
    
    ---
    
    ## Plackett–Luce loss
    
    loss는 이 확률에 `-log`를 씌운 값입니다.
    
    $$
    Loss = -\log(0.391) = 0.939
    $$
    
    즉 이 trial의 loss는 약 **0.94**입니다.
    
    ---
    
    ## 직관
    
    만약 모델 score가 실제 ranking과 잘 맞으면:
    
    ```
    A score 높음
    C score 중간
    B score 낮음
    ```
    
    실제 ranking `A > C > B`의 확률이 커지고, loss는 작아집니다.
    
    반대로 모델 score가:
    
    | 상품 | score |
    | --- | --- |
    | A | 0.5 |
    | B | 2.0 |
    | C | 1.0 |
    
    처럼 실제 ranking과 반대로 나오면 `A > C > B` 확률은 작아지고, loss는 커집니다.
    
    한 줄로 말하면:
    
    > **Plackett–Luce loss는 AI가 실제로 준 순위가 모델 score상에서도 높은 확률을 갖도록 만드는 loss입니다.**
    > 

---

### 직관적 설명

AI ranking이 다음과 같다고 하자.

```
A > C > B
```

모델이 각 상품에 대해 score를 예측한다.

| 상품 | 모델 score |
| --- | --- |
| A | $(s_A)$ |
| B | $(s_B)$ |
| C | $(s_C)$ |

Plackett–Luce 모델은 이 ranking이 나올 확률을 아래처럼 계산한다.

$$
[P(A \succ C \succ B) = \frac{e^{s_A}}{e^{s_A}+e^{s_B}+e^{s_C}}\cdot\frac{e^{s_C}}{e^{s_C}+e^{s_B}}]
$$

의미는 다음과 같다.

1. A, B, C 중 A가 1위로 선택될 확률을 계산한다.
2. A를 제외한 나머지 B, C 중 C가 2위로 선택될 확률을 계산한다.
3. 두 확률을 곱해 전체 ranking 확률을 계산한다.

---

### Loss로 사용하는 방식

모델 학습에서는 실제 AI가 매긴 ranking의 확률이 높아지도록 학습한다.

따라서 loss는 아래처럼 정의한다.

$$
\mathcal{L} = - \log P(\text{observed ranking})
$$

즉 실제 ranking에 높은 확률을 부여하면 loss가 작아지고, 낮은 확률을 부여하면 loss가 커진다.

쉽게 말하면:

```
모델이 AI가 실제로 고른 순위를 잘 설명할수록 좋은 모델
```

이 된다.

Learning to Rank 분야에서도 Plackett–Luce likelihood를 ranking loss로 사용하는 ListMLE 계열 방법이 존재한다. ListMLE는 전체 리스트의 순서를 학습하는 listwise ranking 방식으로 볼 수 있다. ([arXiv](https://arxiv.org/abs/1909.06722?utm_source=chatgpt.com))

---

### 우리 프로젝트에서의 역할

Plackett–Luce ranking loss는 다음 역할을 한다.

```
AI ranking 데이터
→ 모델 학습에 사용할 수 있는 loss로 변환
→ 상품 페이지별 recommendation score 학습
```

즉 이 loss는 **AI가 매긴 1·2·3위 순위를 ML 모델의 학습 신호로 바꾸는 핵심 장치**다.

---

## 8. 2단계: Score를 확률로 변환

### 왜 별도 단계가 필요한가

Recommendation score는 상대적 선호도는 잘 나타낼 수 있지만, 그 자체가 바로 확률은 아니다.

예를 들어 두 모델이 아래와 같은 score를 낼 수 있다.

| 상품 | 모델 1 score | 모델 2 score |
| --- | --- | --- |
| A | 10 | 2 |
| B | 9 | 1 |
| C | 8 | 0 |

두 모델 모두 순위는 동일하다.

```
A > B > C
```

하지만 softmax를 적용하면 확률값은 크게 달라질 수 있다.

즉 ranking 모델은 순서를 잘 맞혀도 확률이 과신되거나, 반대로 너무 평평할 수 있다.

그래서 score를 확률로 바꾸는 calibration 단계가 필요하다.

---

## 9. Temperature calibration

### 개념

추천 확률을 계산할 때 score를 그대로 softmax에 넣지 않고, temperature (T)로 나눠서 계산한다.

$$
P(i)=\frac{e^{s_i/T}}{\sum_{j \in U} e^{s_j/T}}
$$

| 기호 | 의미 |
| --- | --- |
| $(s_i)$ | 상품 (i)의 recommendation score |
| $(U)$ | 폼클렌저 상품 전체 pool |
| $(T)$ | temperature calibration 값 |

Temperature scaling은 softmax 기반 모델의 confidence를 조정하는 대표적인 calibration 방법이다. Guo et al.은 modern neural network가 probability calibration이 잘 안 될 수 있으며, temperature scaling이 실용적으로 효과적인 post-processing calibration 방법이라고 보고했다. ([arXiv](https://arxiv.org/abs/1706.04599?utm_source=chatgpt.com))

---

### ($T$)의 역할

- $(T < 1)$
    
    확률 분포가 더 뾰족해진다.
    
    ```
    1위 상품에 확률이 더 많이 몰림
    모델을 더 confident하게 만듦
    ```
    
- $(T > 1)$
    
    확률 분포가 더 평평해진다.
    
    ```
    상위 상품과 하위 상품 간 확률 차이가 줄어듦
    모델을 덜 confident하게 만듦
    ```
    
- $(T = 1)$
    
    기본 softmax와 동일하다.
    

---

### Temperature는 어떻게 정하나

Temperature는 학습 데이터가 아니라 **validation set**에서 정한다.

절차는 다음과 같다.

```
1. 학습된 score 모델을 고정한다.
2. validation set의 각 trial에 대해 상품별 score를 계산한다.
3. 여러 temperature 후보를 적용해 ranking 확률을 계산한다.
4. 실제 AI ranking에 대한 Negative Log Likelihood가 가장 낮은 T를 선택한다.
```

수식으로는 다음과 같다.

$$
T^*\arg\min_T\text{NLL}_{validation}(T)
$$

즉 모델을 다시 학습하는 것이 아니라, 이미 학습된 score를 확률로 변환할 때 사용하는 scale만 조정하는 과정이다.

---

### 우리 프로젝트에서의 역할

Temperature calibration은 최종적으로 “추천 확률”을 보고할 때 사용한다.

| 개념 | 목적 | 사용 시점 |
| --- | --- | --- |
| Plackett–Luce ranking loss | recommendation score 학습 | training 단계 |
| Temperature calibration | score를 calibrated probability로 변환 | validation 이후 / inference 단계 |

정리하면:

```
Plackett–Luce ranking loss
= AI ranking 데이터를 이용해 score 모델을 학습하는 방법

Temperature calibration
= 학습된 score를 폼클렌저 pool 기준 추천 확률로 변환할 때 과신/과소추정을 보정하는 방법
```

---

## 10. 최종 모델 사용 방식

학습이 끝난 뒤에는 매번 AI에게 3개 후보를 다시 보여줄 필요가 없다.

최종 모델은 상품 페이지 하나만 입력해도 score를 낼 수 있다.

```
상품 X 페이지
→ feature 추출
→ 모델
→ recommendation score
```

그 다음 폼클렌저 pool 전체 상품의 score와 비교해 아래 지표를 계산한다.

---

### Relative ranking

폼클렌저 전체 pool 내에서 상품 X가 몇 위인지 계산한다.

예:

```
상품 X는 폼클렌저 1,000개 중 42위
```

---

### Percentile

상대 순위를 percentile로 변환한다.

예:

```
상품 X는 폼클렌저 pool 내 상위 4.2%
```

---

### Odds uplift

기준 상품 또는 평균 상품 대비 추천 odds가 몇 배 높은지 계산한다.

예:

```
상품 X는 평균 폼클렌저 대비 AI 추천 odds가 2.4배 높음
```

---

### Pool 기준 추천 확률

폼클렌저 pool 전체 score를 softmax로 정규화해 추천 확률을 계산한다.

예:

```
상품 X의 AI 추천 확률은 폼클렌저 pool 기준 1.6%
```

중요한 점은 이 확률이 항상 **폼클렌저 pool 기준**이라는 것이다.

---

## 11. 모델 평가 방법

모델 평가는 두 가지 관점에서 진행한다.

```
1. Ranking을 잘 맞히는가?
2. 확률이 실제 선택 빈도와 잘 맞는가?
```

---

### Ranking 성능 평가

test set에서도 3개 후보 ranking 데이터를 만들어 평가한다.

| trial_id | 실제 AI ranking | 모델 예측 ranking |
| --- | --- | --- |
| 1 | A > C > B | A > C > B |
| 2 | E > D > F | D > E > F |
| 3 | H > A > G | H > A > G |

사용할 수 있는 지표는 다음과 같다.

| 지표 | 의미 |
| --- | --- |
| Top-1 Accuracy | AI가 1위로 고른 상품을 모델도 1위로 예측했는가 |
| Pairwise Accuracy | 후보 간 상대 순서를 얼마나 맞혔는가 |
| NDCG@3 | 상위 순위를 얼마나 잘 맞혔는가 |
| Kendall Tau | 실제 순위와 예측 순위의 상관 |

초기에는 **Top-1 Accuracy, Pairwise Accuracy, NDCG@3**를 핵심 지표로 본다.

---

### 확률 성능 평가

추천 확률까지 사용할 경우, 모델의 확률이 실제 선택 빈도와 잘 맞는지도 확인해야 한다.

| 지표 | 의미 |
| --- | --- |
| Negative Log Likelihood | 실제 ranking에 모델이 높은 확률을 부여했는가 |
| Brier Score | 예측 확률과 실제 선택 결과가 얼마나 가까운가 |
| Calibration curve | 예측 확률과 실제 정답률이 일치하는가 |
| ECE | 전체적으로 calibration error가 얼마나 되는가 |

---

## 12. 실험 설계 시 주의점

모델 성능은 후보 3개를 어떻게 구성하는지에 크게 영향을 받는다.

좋은 실험 설계는 다음 조건을 만족해야 한다.

```
폼클렌저 pool에서 후보를 균형 있게 샘플링
각 상품이 다양한 경쟁 상품과 비교되도록 구성
특정 브랜드/가격대/피부타입에 편향되지 않도록 관리
전체 상품들이 comparison network상 연결되도록 설계
```

주의해야 할 점은 다음과 같다.

```
항상 약한 상품과만 비교된 상품은 score가 과대평가될 수 있음
항상 강한 상품과만 비교된 상품은 score가 과소평가될 수 있음
특정 하위 카테고리 안에서만 비교하면 전체 pool 기준 해석이 어려움
```

따라서 후보 3개는 랜덤 샘플링을 기본으로 하되, 브랜드/가격대/제품 타입이 한쪽으로 치우치지 않도록 stratified sampling을 고려한다.

---

## 13. 기대 결과물

최종적으로 각 폼클렌저 상품 페이지에 대해 아래와 같은 리포트를 만들 수 있다.

| 상품 | Score | Pool Rank | Percentile | 평균 대비 Odds | 추천 확률 |
| --- | --- | --- | --- | --- | --- |
| A | 2.10 | 12 / 1,000 | 상위 1.2% | 3.1배 | 1.8% |
| B | 0.85 | 170 / 1,000 | 상위 17.0% | 1.4배 | 0.5% |
| C | -0.30 | 620 / 1,000 | 하위 38.0% | 0.7배 | 0.1% |

이를 통해 아래 질문에 답할 수 있다.

```
AI 추천 가능성이 높은 폼클렌저 페이지는 어떤 특징을 갖는가?
우리 상품은 경쟁 상품 대비 어느 위치에 있는가?
상세페이지 개선 전후 추천 가능성이 얼마나 달라지는가?
어떤 요소가 AI 추천 가능성에 영향을 주는가?
```

---

## 14. 전체 실행 플로우

```
1. 폼클렌저 상품 pool 구축
2. 상품 페이지 feature 수집 및 embedding 생성
3. 상품 3개씩 후보 set 구성
4. AI에게 1위, 2위, 3위 ranking 수집
5. Plackett–Luce ranking loss로 recommendation score 모델 학습
6. validation set에서 temperature calibration 수행
7. test set에서 ranking 성능 및 확률 calibration 평가
8. 폼클렌저 pool 전체에 score 부여
9. relative ranking, percentile, odds uplift, 추천 확률 산출
```

---

## 15. 최종 요약

이번 모델은 **폼클렌저 상품 페이지의 AI 추천 가능성을 예측하기 위한 Learning-to-Rank / Preference Learning 모델**이다.

핵심은 전체 상품을 한 번에 비교하지 않고, 폼클렌저 상품 3개씩의 AI ranking 데이터를 수집해 상품 페이지별 latent recommendation score를 학습하는 것이다.

이후 해당 score를 폼클렌저 pool 내에서 정렬하고 softmax 및 temperature calibration을 적용해 상대 순위와 추천 확률을 산출한다.

보고용으로 한 문장으로 정리하면:

> 전체 폼클렌저 상품을 한 번에 비교하는 대신, 3개 후보 ranking 데이터를 활용해 AI가 선호하는 상품 페이지의 특징을 학습하고, 이를 바탕으로 각 상품의 폼클렌저 pool 내 추천 순위와 추천 확률을 예측하는 모델을 구축하고자 합니다.
> 

---

## +) Train/Val/Test 나누기

## 1. 전체 운영 방식

모델 평가는 3단계로 나눠서 진행한다.

| 단계 | 목적 | 방식 |
| --- | --- | --- |
| Step 1. Seen-item test | 기본 ranking 성능 확인 | 기존 상품이 새로운 조합으로 나왔을 때 잘 맞히는지 평가 |
| Step 2. Unseen-item test | 새 상품 일반화 성능 확인 | 학습에서 보지 않은 상품도 잘 예측하는지 평가 |
| Step 3. Final training | 운영용 최종 모델 구축 | 평가 후 전체 데이터를 활용해 최종 모델 재학습 |

---

## 2. 기본 원칙

### 같은 후보 조합은 split 간 중복 금지

예를 들어 아래 조합이 train에 들어갔다면:

```
A, B, C → A > C > B
```

같은 `A, B, C` 조합은 validation이나 test에 들어가면 안 된다.

즉, 데이터는 개별 row가 아니라 **trial_id 단위**로 나눈다.

---

## 3. Step 1 — Seen-item test

### 목적

> 학습에서 본 상품이 새로운 조합으로 나와도 AI ranking을 잘 맞히는지 확인
> 

### 방식

전체 AI 응답 데이터를 `trial_id` 기준으로 나눈다.

```
Train 70%
Validation 15%
Test 15%
```

예시:

| trial_id | 후보 상품 | Split |
| --- | --- | --- |
| 1 | A, B, C | Train |
| 2 | A, D, E | Test |
| 3 | F, G, H | Train |

여기서 A는 train과 test에 모두 등장할 수 있다.

하지만 같은 조합 `A, B, C`가 train/test에 동시에 들어가면 안 된다.

### 확인하는 것

- 기존 상품의 새로운 조합 예측 성능
- Top-1 Accuracy
- Pairwise Accuracy
- NDCG@3

---

## 4. Step 2 — Unseen-item test

### 목적

> 학습에서 한 번도 본 적 없는 새 상품도 잘 예측하는지 확인
> 

### 방식

먼저 test 상품을 정한다.

예:

```
전체 상품 300개 중 test 상품 30개 선정
```

그다음 test 상품이 하나라도 포함된 trial은 모두 test로 뺀다.

예시:

| trial_id | 후보 상품 | Split |
| --- | --- | --- |
| 1 | A, B, C | Train |
| 2 | X, A, B | Test |
| 3 | Y, C, D | Test |

여기서 X, Y가 test 상품이면 해당 trial은 모두 test다.

### 중요

Unseen-item test는 train 구성이 달라지므로 **모델을 별도로 다시 학습해야 한다.**

### 확인하는 것

- 모델이 상품 ID를 외운 것이 아닌지
- 상품 페이지 feature만 보고 새 상품을 평가할 수 있는지
- 실제 운영 상황에서 일반화 가능한지

---

## 5. Step 3 — Final training

### 목적

평가가 끝난 뒤 운영에 사용할 최종 모델을 만든다.

### 방식

```
1. Seen-item / unseen-item 평가로 모델 구조 확정
2. feature, hyperparameter, calibration 방식 확정
3. 사용 가능한 전체 데이터로 최종 모델 재학습
4. 300개 폼클렌저 pool 전체에 score 부여
5. 순위, percentile, odds, 추천 확률 산출
```

단, 최종 성능 보고는 반드시 분리해둔 test set 결과 기준으로 한다.

---

## 한 줄 요약

> 먼저 trial 단위 split으로 기존 상품의 새로운 조합 예측 성능을 보고, 그다음 상품 단위 holdout으로 새 상품 일반화 성능을 확인한 뒤, 검증이 끝나면 전체 데이터를 활용해 운영용 최종 모델을 다시 학습한다.
>