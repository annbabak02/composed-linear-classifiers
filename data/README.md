# Data

The experiments use public datasets loaded at runtime through scikit-learn/OpenML.
Raw datasets are not committed because several of them are large.

| Dataset | Loader | Source |
|---|---|---|
| Breast Cancer Wisconsin | `load_breast_cancer` | `sklearn.datasets` |
| Bank Marketing | `fetch_openml(data_id=1461)` | OpenML |
| Adult | `fetch_openml(data_id=1590)` | OpenML |
| Credit Card Fraud | `fetch_openml(data_id=1597)` | OpenML |
| SECOM | `fetch_openml(data_id=43587)`, target `Pass/Fail` | OpenML |
| SECOM Selected | `load_secom` + feature selection | Derived |

For all datasets, label `1` is forced to be the minority/content-positive class.
