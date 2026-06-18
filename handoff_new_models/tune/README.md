# tune/ — 하이퍼파라미터 튜닝 스크립트

각 모델의 **좋은 설정(하이퍼파라미터)을 찾는** 폴더입니다.
브랜드 단위 교차검증(brand-CV)으로 여러 후보 설정을 비교해, 가장 좋은 설정을
`artifacts/tuning/<모델>_best_params.json`에 저장합니다. 이 파일은 학습 스크립트가 자동으로 불러옵니다.

- `tune_ranksvm.py` · `tune_rf.py` · `tune_logreg.py` · `tune_ebm.py` — 모델별 튜너
- 각 파일은 `make_model`(어떤 모델), `prob_fn`(이길 점수 뽑는 법), `GRID`(탐색할 설정)만 정의합니다.
  실제 교차검증 루프는 상위 폴더의 `harness.py`가 처리합니다.

실행:
```bash
python tune/tune_logreg.py            # 본 튜닝
python tune/tune_logreg.py --smoke    # 빠른 동작 점검 (작은 그리드, *_smoke.json 저장)
```

함께 들어있는 파일:
- `runtime.py`, `cv_common.py` — 원본에서 복사한 튜닝 보조 도구 (**수정 금지**)
- `__init__.py` — 파이썬 패키지 표시용 빈 파일

더 넓게 탐색하려면 각 튜너의 `GRID` 딕셔너리에 값을 추가하세요(RF/EBM은 시간이 오래 걸립니다).
자세한 내용은 상위 폴더의 `README_HANDOFF.md`를 보세요.
