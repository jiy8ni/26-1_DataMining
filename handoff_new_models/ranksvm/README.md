# ranksvm/ — RankSVM 모델 (PDF 모델 1)

SVM을 순위 문제에 맞게 변형한 모델. 두 상품의 feature 차이(ΔX)를 보고
"앞 상품이 이기는 경계"를 **최대 마진**으로 학습합니다. 확률 대신 decision score를 사용합니다.

- `ranksvm_train.py` — 최종 학습 스크립트. 튜닝된 설정을 불러와 학습하고
  `artifacts/preds/ranksvm_{val,test}.npz`에 예측을 저장합니다.
- 하이퍼파라미터 탐색은 `../tune/tune_ranksvm.py` 에서 합니다.

실행:
```bash
python ranksvm/ranksvm_train.py
```

자세한 내용은 상위 폴더의 `README_HANDOFF.md`를 보세요.
