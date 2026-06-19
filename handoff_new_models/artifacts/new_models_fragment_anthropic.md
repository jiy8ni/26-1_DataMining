### 신규 pairwise 모델 — CV 최종 선택 (anthropic)

| model | candidates | selected params | cv_balanced | top1 | pairwise | ndcg@3 | tau | nll | brier | temp |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ranksvm` | 8 | `kernel=rbf, C=1.0` | 2.2579 | 0.6532 | 0.7601 | 0.9543 | 0.5203 | 1.3469 | 0.4665 | 2.00 |
| `rf` | 16 | `n_estimators=500, max_depth=None, min_samples_leaf=5, max_features=sqrt` | 5.0991 | 0.6593 | 0.7606 | 0.9549 | 0.5212 | 1.3384 | 0.4561 | 0.50 |
| `logreg` | 4 | `C=0.01` | 4.8369 | 0.6552 | 0.7574 | 0.9538 | 0.5149 | 1.3501 | 0.4632 | 0.54 |
| `ebm` | 8 | `interactions=0, learning_rate=0.05, min_samples_leaf=10` | 4.0687 | 0.6645 | 0.7619 | 0.9554 | 0.5238 | 1.3342 | 0.4564 | 0.62 |

### 신규 pairwise 모델 — 최종 테스트 결과 (anthropic)

| model | avg temp | top1 | pairwise | ndcg@3 | tau | nll | brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ranksvm` | 2.00 | 0.5288 | 0.6829 | 0.9350 | 0.3658 | 1.9788 | 0.6760 |
| `rf` | 0.50 | 0.5487 | 0.6929 | 0.9377 | 0.3858 | 1.5967 | 0.5747 |
| `logreg` | 0.50 | 0.5012 | 0.6725 | 0.9318 | 0.3450 | 1.8769 | 0.6815 |
| `ebm` | 0.62 | 0.5188 | 0.6821 | 0.9340 | 0.3642 | 1.7357 | 0.6343 |
