# Export Manifest

| Path | Type | Purpose |
|---|---|---|
| `artifacts/original/VKR_Babak_PM22-6.docx` | DOCX | Исходный текст ВКР |
| `artifacts/original/Презентация_предзащита.pptx` | PPTX | Исходная презентация |
| `artifacts/original/main.ipynb` | IPYNB | Архивная копия исходного ноутбука |
| `notebooks/main.ipynb` | IPYNB | Рабочая копия ноутбука в структуре проекта |
| `exports/code/main_export.py` | Python | Полный экспорт code-cells исходного ноутбука |
| `src/composed_linear_classifiers/core.py` | Python package | Библиотечная версия кода без автозапуска экспериментов |
| `results/metrics_default_stream_from_notebook.csv` | CSV | Полная default-выгрузка из сохраненного stream-лога ноутбука |
| `results/metrics_default_failures_from_notebook.csv` | CSV | Зафиксированные default-ошибки исходной среды |
| `results/metrics_tuned_partial_from_notebook.csv` | CSV | Частичная Optuna-выгрузка из сохраненного stream-лога |
| `results/summary_tables/top_models_by_dataset.csv` | CSV | Топ моделей по датасетам |
| `docs/thesis_text_export.md` | Markdown | Текстовая версия ВКР для GitHub-просмотра |
| `docs/thesis_text_export.txt` | TXT | Plain-text версия ВКР для поиска |

Примечание: `metrics_tuned_partial_from_notebook.csv` не является полной таблицей итогового Optuna-эксперимента, потому что сохраненный output исходного ноутбука обрывается во время расчета Adult/KernelSVM.
