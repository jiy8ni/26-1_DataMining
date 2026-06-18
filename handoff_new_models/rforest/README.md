# rforest/ — Pairwise Random Forest 모델 (PDF 모델 2)

여러 결정트리의 **투표(Bagging)** 로 "앞 상품이 이길 확률"을 예측합니다.
기존 LightGBM(Boosting)과 대비되는 앙상블 방식이며, 과적합에 비교적 강합니다.

- `rf_train.py` — 최종 학습 스크립트. 튜닝된 설정을 불러와 학습하고
  `artifacts/preds/rf_{val,test}.npz`에 예측을 저장합니다.
- 하이퍼파라미터 탐색은 `../tune/tune_rf.py` 에서 합니다.

실행:
```bash
python rforest/rf_train.py
```
> 트리 개수가 많아 다른 모델보다 학습이 느립니다(수 분).

자세한 내용은 상위 폴더의 `README_HANDOFF.md`를 보세요.
