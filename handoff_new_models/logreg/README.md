# logreg/ — Pairwise Logistic Regression 모델 (PDF 모델 3)

"앞 상품이 이길 확률"을 직접 출력하는 가장 단순하고 **해석하기 쉬운 기준선**입니다.
쌍별 선호 확률을 모델링하는 고전적 방법(Bradley–Terry)과 수학적으로 유사합니다.
`fit_intercept=False`로 두어 A−B와 B−A의 부호 대칭성을 유지합니다.

- `logreg_train.py` — 최종 학습 스크립트. 튜닝된 설정을 불러와 학습하고
  `artifacts/preds/logreg_{val,test}.npz`에 예측을 저장합니다.
- 하이퍼파라미터 탐색은 `../tune/tune_logreg.py` 에서 합니다.

실행:
```bash
python logreg/logreg_train.py
```
> 학습이 가장 빠른 모델입니다. 처음 동작 확인용으로 좋습니다.

자세한 내용은 상위 폴더의 `README_HANDOFF.md`를 보세요.
