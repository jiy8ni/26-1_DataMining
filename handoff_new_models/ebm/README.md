# ebm/ — GAM / EBM 모델 (PDF 모델 4)

선형 모델과 복잡한 비선형 모델의 중간. feature별 효과를 **곡선**으로 학습해
"어떤 feature가 어느 구간에서 승리에 유리한지"를 보여줄 수 있습니다(해석에 강점).

- `ebm_train.py` — 최종 학습 스크립트. 학습 후 예측을
  `artifacts/preds/ebm_{val,test}.npz`에 저장하고,
  추가로 **feature 효과 그래프**를 `artifacts/ebm/`에 저장합니다.
- 하이퍼파라미터 탐색은 `../tune/tune_ebm.py` 에서 합니다.

실행:
```bash
python ebm/ebm_train.py
```

주의:
- `interpret` 라이브러리가 필요합니다 → `pip install interpret`
- 학습이 가장 느립니다(`outer_bags=8`, 수 분~십수 분).
- 입력이 feature **차이(ΔX)** 이므로, 효과 곡선도 "차이가 승리 확률에 주는 효과"를 의미합니다.

자세한 내용은 상위 폴더의 `README_HANDOFF.md`를 보세요.
