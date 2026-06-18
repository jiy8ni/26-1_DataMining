# artifacts/ — 결과물 저장 폴더

스크립트가 만들어내는 모든 산출물이 여기에 쌓입니다.
**원본 프로젝트의 `artifacts/`와 완전히 분리**되어 있어, 여기서 무엇을 하든 원본은 안전합니다.

## tuning/
하이퍼파라미터 튜닝 결과(JSON).
- `semantic_best_config.json` — 임베딩 PCA 설정. 원본에서 미리 복사해 둔 것(모든 모델이 공유).
- `<모델>_best_params.json` — `tune/tune_*.py` 실행 시 생성. 학습 스크립트가 자동으로 불러옵니다.
- `<모델>_best_params_smoke.json` — `--smoke` 점검용 결과(진짜 결과를 덮어쓰지 않음).

## preds/
모델별 예측 점수(`.npz`). `<모델>_train.py` 실행 시 생성됩니다.
- `<모델>_val.npz`, `<모델>_test.npz` — 각 trial(상품 3개)의 점수와 정답 순위.
- `blend_new.py`가 이 파일들을 읽어 앙상블합니다.

## ebm/
EBM 모델의 **feature 효과 그래프**(해석용). `ebm/ebm_train.py` 실행 시 생성됩니다.

---
이 폴더의 파일들은 스크립트가 다시 만들 수 있으므로, 지우고 처음부터 실행해도 됩니다
(단, `tuning/semantic_best_config.json`은 복사본이니 지우지 마세요).
