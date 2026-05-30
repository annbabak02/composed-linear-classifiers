# Results Exports

- `metrics_default_stream_from_notebook.csv` contains all 168 successful default-CV rows parsed from the executed notebook log.
- `metrics_default_failures_from_notebook.csv` records 8 default-CV failures caused by unavailable optional `xgboost` and `lightgbm` packages in the original runtime.
- `metrics_default_table_preview_from_notebook.csv` is the truncated pandas HTML preview saved inside the notebook; it is preserved only for provenance.
- `metrics_tuned_partial_from_notebook.csv` is parsed from the saved Optuna stream; the saved notebook output stops during Adult/KernelSVM, so this file is intentionally marked partial.
- `notebook_default_cv.log` and `notebook_optuna_partial.log` preserve the raw execution streams for audit.
