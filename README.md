# Составные нелинейные классификаторы на основе линейных моделей

Репозиторий содержит программную часть выпускной квалификационной работы Бабак Анны Романовны по теме систематического построения и экспериментального сравнения составных нелинейных классификаторов, в которых линейная модель остается базовым вычислительным элементом, а нелинейность возникает за счет преобразования признаков, ансамблирования, каскадирования, локализации или нелинейной агрегации скоров.

## Научная постановка

**Объект исследования:** задача бинарной классификации в машинном обучении.

**Предмет исследования:** составные нелинейные классификаторы, построенные на основе линейных моделей, и их поведение на задачах с различным размером выборки, размерностью, дисбалансом классов и нелинейностью границы решения.

**Цель:** реализовать и эмпирически сравнить девять архитектур составных классификаторов в единой методологической рамке scikit-learn.

Ключевой методологический инвариант: метка `1` всегда соответствует миноритарному, содержательно положительному классу. Это принципиально для корректного F1, Precision, Recall и PR-AUC на задачах SECOM и Credit Card Fraud.

## Состав репозитория

```text
.
├── artifacts/original/                 # исходные артефакты ВКР: docx, pptx, ipynb
├── data/README.md                      # источники датасетов и правила загрузки
├── docs/                               # текстовая выгрузка диплома и методические notes
├── exports/code/main_export.py         # полный Python-экспорт исходного ноутбука
├── notebooks/main.ipynb                # исходный исследовательский ноутбук
├── results/                            # выгрузки результатов и audit-логи
├── scripts/                            # воспроизводимые CLI-запуски экспериментов
├── src/composed_linear_classifiers/     # библиотечная версия исследовательского кода
└── tests/                              # быстрые smoke-тесты
```

## Модели

Реестр содержит 44 конфигурации:

- 4 линейных бейзлайна: Logistic Regression, Gaussian Naive Bayes, LDA, ElasticNet.
- 5 нелинейных бейзлайнов: RBF SVM, Random Forest, XGBoost, LightGBM, MLP.
- 35 составных конфигураций: 9 архитектур, параметризованных базовыми линейными моделями `logistic`, `sgd`, `lda`, `nb`; комбинация `LinearBoost + LDA` исключена, так как LDA в scikit-learn не поддерживает `sample_weight`.

Составные архитектуры: `PolynomialFeaturesClassifier`, `SplineFeaturesClassifier`, `LinearBoostClassifier`, `BaggingFeatureSubspaceClassifier`, `MaxOfLinearClassifier`, `CascadeLinearClassifier`, `PiecewiseLinearClassifier`, `MixtureOfLinearExpertsClassifier`, `FuzzyLogicLinearClassifier`.

## Данные

Датасеты загружаются публично через scikit-learn/OpenML и не коммитятся в репозиторий:

| Датасет | Источник | Назначение |
|---|---|---|
| Breast Cancer Wisconsin | `sklearn.datasets` | почти линейно разделимая медицинская задача |
| Bank Marketing | OpenML `1461` | маркетинговый отклик, слабый сигнал |
| Adult | OpenML `1590` | социально-экономическая бинарная классификация |
| Credit Card Fraud | OpenML `1597` | экстремальный дисбаланс классов |
| SECOM | OpenML `43587`, target `Pass/Fail` | дефектоскопия полупроводников |
| SECOM Selected | SECOM + отбор 25 признаков | анализ влияния предобработки |

В проектной версии исправлен загрузчик SECOM: у OpenML-объекта `43587` нет default target, поэтому целевая переменная читается явно из колонки `Pass/Fail`.

## Быстрый старт

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-minimal.txt
python scripts/smoke_test.py
python -m unittest discover -s tests
```

Полный набор зависимостей с XGBoost и LightGBM:

```bash
pip install -r requirements.txt
```

## Воспроизведение экспериментов

Быстрая проверка на одном датасете:

```bash
python scripts/run_default_cv.py --datasets breast_cancer --models BaselineLogisticRegression 'SplineFeaturesClassifier[logistic]'
```

Полный default-бенчмарк:

```bash
python scripts/run_default_cv.py --datasets breast_cancer bank_marketing adult credit_card_fraud secom secom_selected
```

Optuna-оценка с 50 испытаниями на модель:

```bash
python scripts/run_optuna_cv.py --datasets breast_cancer --n-trials 50
```

Полный Optuna-запуск на всех моделях и датасетах вычислительно дорогой: каждая модель проходит внутреннюю и внешнюю 5-fold оценку, поэтому запуск может занимать часы.

## Выгрузки результатов

Основные файлы:

- `results/metrics_default_stream_from_notebook.csv` — 168 успешных default-CV строк, восстановленных из сохраненного вывода исходного ноутбука.
- `results/metrics_default_failures_from_notebook.csv` — 8 зафиксированных отказов из исходной среды из-за отсутствующих `xgboost` и `lightgbm`.
- `results/metrics_tuned_partial_from_notebook.csv` — частичная Optuna-выгрузка из ноутбука; сохраненный вывод обрывается на Adult/KernelSVM.
- `results/summary_tables/top_models_by_dataset.csv` — топ моделей по F1 для каждого датасета из default-выгрузки.
- `results/notebook_default_cv.log`, `results/notebook_optuna_partial.log` — исходные audit-логи выполнения.

Важно: сохраненный output исходного ноутбука пропустил SECOM/SECOM Selected из-за старого загрузчика target. В библиотечной версии это исправлено; для финального пересчета используйте CLI-скрипты выше.

## Методологический протокол

- Стратифицированная 5-fold кросс-валидация.
- Фиксированный `random_state=42`.
- Целевая метрика Optuna: F1 по миноритарному классу.
- Дополнительные метрики: Precision, Recall, F1-macro, balanced accuracy, ROC-AUC, PR-AUC, время обучения.
- Для сильно несбалансированных задач PR-AUC рассматривается как приоритетная ранжирующая метрика.

## Академические артефакты

Исходные файлы комиссии находятся в `artifacts/original/`:

- `VKR_Babak_PM22-6.docx` — текст ВКР.
- `Презентация_предзащита.pptx` — презентация.
- `main.ipynb` — исходный ноутбук.

Текстовая выгрузка диплома для быстрого просмотра и поиска: `docs/thesis_text_export.md` и `docs/thesis_text_export.txt`.

## Ограничения

Репозиторий фиксирует исходные результаты настолько полно, насколько они были сохранены в ноутбуке. Для строгого финального архива рекомендуется выполнить полный CLI-пересчет в среде с установленными `xgboost` и `lightgbm`, после чего заменить частичную Optuna-выгрузку на `results/metrics_tuned.csv`.
