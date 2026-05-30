# %% [cell 3]
from __future__ import annotations

import contextlib
import inspect
import io
import time
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cluster import KMeans
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer as _sklearn_breast_cancer,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.feature_selection import (
    SelectKBest, VarianceThreshold, f_classif, mutual_info_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, SGDClassifier
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import (
    LabelEncoder, PolynomialFeatures, SplineTransformer, StandardScaler,
)
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# %% [cell 5]
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml, load_breast_cancer as _load_breast_cancer, make_moons


@dataclass
class BinaryDataset:
    """Единый формат датасета для бинарной классификации.

    Инвариант: positive class (label 1) — миноритарный/целевой класс.
    Это необходимо, чтобы все метрики для класса 1 (Precision, Recall, F1,
    PR-AUC) считались для содержательно «положительного» класса —
    дефекта в SECOM, мошенничества в Credit Card Fraud, депозита в Bank
    Marketing и т. п. См. §2.4.2 для обоснования.
    """
    X: np.ndarray
    y: np.ndarray
    name: str
    feature_names: list[str] | None = None


def _ensure_minority_positive(y: np.ndarray) -> np.ndarray:
    """Если class 1 является majority — инвертирует метки."""
    y = np.asarray(y).astype(int)
    if (y == 1).mean() > 0.5:
        y = 1 - y
    return y


def _encode_target(target) -> np.ndarray:
    """LabelEncoder + гарантия positive=minority."""
    y = LabelEncoder().fit_transform(np.asarray(target).astype(str))
    return _ensure_minority_positive(y)


def _impute_and_scale(X: np.ndarray) -> np.ndarray:
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)
    return X


def _df_to_numeric(df: pd.DataFrame) -> np.ndarray:
    df = df.copy()
    for col in df.select_dtypes(include=["category", "object"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df.to_numpy(dtype=float)


# ---------------------------------------------------------------- #
# Datasets used in the thesis                                       #
# ---------------------------------------------------------------- #
def load_breast_cancer() -> BinaryDataset:
    data = _load_breast_cancer()
    # В sklearn breast_cancer: 0 = malignant (минор.), 1 = benign (мажор.)
    y = _ensure_minority_positive(data.target)
    return BinaryDataset(X=data.data, y=y, name="breast_cancer",
                         feature_names=list(data.feature_names))


def load_adult() -> BinaryDataset:
    ds = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    X = _df_to_numeric(ds.data)
    X = _impute_and_scale(X)
    y = _encode_target(ds.target)
    return BinaryDataset(X=X, y=y, name="adult")


def load_bank_marketing() -> BinaryDataset:
    ds = fetch_openml(data_id=1461, as_frame=True, parser="auto")
    X = _df_to_numeric(ds.data)
    X = _impute_and_scale(X)
    y = _encode_target(ds.target)
    return BinaryDataset(X=X, y=y, name="bank_marketing")


def load_secom() -> BinaryDataset:
    """SECOM: дефектоскопия полупроводников. Дефект (~6%) = positive."""
    ds = fetch_openml(data_id=43587, as_frame=True, parser="auto")
    df = ds.data.copy()
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])
    X = _df_to_numeric(df)
    X = _impute_and_scale(X)
    y = _encode_target(ds.target)
    return BinaryDataset(X=X, y=y, name="secom")


def load_secom_selected(top_k: int = 25) -> BinaryDataset:
    """SECOM после многоэтапного отбора признаков (см. §2.4.1)."""
    from sklearn.feature_selection import (
        VarianceThreshold, mutual_info_classif, f_classif, SelectKBest,
    )
    from sklearn.linear_model import LogisticRegression
    base = load_secom()
    X, y = base.X, base.y
    # 1) удаление квази-константных
    sel_var = VarianceThreshold(threshold=0.01)
    X = sel_var.fit_transform(X)
    # 2) удаление сильно коррелированных
    corr = np.corrcoef(X, rowvar=False)
    upper = np.triu(np.abs(corr), k=1)
    keep = ~np.any(upper > 0.95, axis=0)
    X = X[:, keep]
    # 3) ранговая агрегация ANOVA + MI + L1
    f_scores = SelectKBest(f_classif, k="all").fit(X, y).scores_
    mi_scores = mutual_info_classif(X, y, random_state=42)
    lr = LogisticRegression(penalty="l1", solver="saga", C=0.1,
                            max_iter=2000, random_state=42).fit(X, y)
    l1_scores = np.abs(lr.coef_).ravel()
    rank = (
        pd.Series(f_scores).rank(ascending=False)
        + pd.Series(mi_scores).rank(ascending=False)
        + pd.Series(l1_scores).rank(ascending=False)
    )
    top = rank.nsmallest(top_k).index.values
    X = X[:, top]
    return BinaryDataset(X=X, y=y, name="secom_selected")


def load_credit_card_fraud() -> BinaryDataset:
    """Credit Card Fraud (OpenML id 1597). Мошенничество (~0.17%) = positive."""
    ds = fetch_openml(data_id=1597, as_frame=True, parser="auto")
    X = _df_to_numeric(ds.data)
    X = _impute_and_scale(X)
    y = _encode_target(ds.target)
    return BinaryDataset(X=X, y=y, name="credit_card_fraud")





# Реестр датасетов: сюда включены все 6 датасетов, описанные в §2.4.1.
DATASET_REGISTRY: list[Callable[[], BinaryDataset]] = [
    load_breast_cancer,
    load_bank_marketing,
    load_adult,
    load_credit_card_fraud,
    load_secom,
    load_secom_selected,
]

# %% [cell 7]
class ComposedClassifier(BaseEstimator, ClassifierMixin):
    """Базовый класс для кастомных составных классификаторов."""

    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def predict(self, X): ...

    def _suggest_params(self, trial) -> dict:
        return {}


# Конфигурация базовых моделей
_BASE_MODELS = {
    "logistic": {
        "factory": lambda **kw: LogisticRegression(max_iter=1000, random_state=42, **kw),
        "defaults": {"C": 1.0},
        "suggest": lambda trial: {"C": trial.suggest_float("base_C", 1e-4, 100.0, log=True)},
    },
    "sgd": {
        "factory": lambda **kw: SGDClassifier(loss="log_loss", max_iter=1000, random_state=42, **kw),
        "defaults": {"alpha": 0.0001},
        "suggest": lambda trial: {"alpha": trial.suggest_float("base_alpha", 1e-6, 10.0, log=True)},
    },
    "lda": {
        "factory": lambda **kw: LinearDiscriminantAnalysis(**kw),
        "defaults": {},
        "suggest": lambda trial: {},
    },
    "nb": {
        "factory": lambda **kw: GaussianNB(**kw),
        "defaults": {},
        "suggest": lambda trial: {
            "var_smoothing": trial.suggest_float("base_var_smoothing", 1e-12, 1e-2, log=True)
        },
    },
}

BASE_MODEL_NAMES = list(_BASE_MODELS.keys())


def create_base_model(name: str = "logistic", **kwargs):
    if name not in _BASE_MODELS:
        raise ValueError(f"Unknown base_model={name!r}. Choose from {list(_BASE_MODELS)}")
    config = _BASE_MODELS[name]
    params = {**config["defaults"], **kwargs}
    return config["factory"](**params)


def suggest_base_params(trial, base_model: str) -> dict:
    if base_model not in _BASE_MODELS:
        return {}
    return _BASE_MODELS[base_model]["suggest"](trial)


def get_base_defaults(base_model: str) -> dict:
    if base_model not in _BASE_MODELS:
        return {}
    return dict(_BASE_MODELS[base_model]["defaults"])


def get_proba(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(decision, -500, 500)))
        return np.column_stack([1 - sigmoid, sigmoid])
    preds = model.predict(X)
    return np.column_stack([1 - preds, preds]).astype(float)


def make_variant(cls, base_model: str):
    """Создать подкласс с другим дефолтным base_model."""
    variant_name = f"{cls.__name__}[{base_model}]"

    def new_init(self, **kwargs):
        kwargs.setdefault("base_model", base_model)
        cls.__init__(self, **kwargs)

    new_init.__signature__ = inspect.signature(cls.__init__)
    return type(variant_name, (cls,), {"__init__": new_init})

# %% [cell 9]
class BaselineLogisticRegression(ComposedClassifier):
    """Логистическая регрессия с L2-регуляризацией [Cox, 1958]. Линейный бейзлайн."""

    def __init__(self, C: float = 1.0, max_iter: int = 5000):
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model_ = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=42)
        self.model_.fit(X, y)
        return self

    def predict(self, X): return self.model_.predict(X)
    def predict_proba(self, X): return self.model_.predict_proba(X)

    def _suggest_params(self, trial) -> dict:
        return {"C": trial.suggest_float("C", 1e-4, 100.0, log=True)}


class BaselineNaiveBayes(ComposedClassifier):
    """Гауссовский наивный байесовский классификатор [Mosteller & Wallace, 1964]. Линейный бейзлайн."""

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model_ = GaussianNB(var_smoothing=self.var_smoothing)
        self.model_.fit(X, y)
        return self

    def predict(self, X): return self.model_.predict(X)
    def predict_proba(self, X): return self.model_.predict_proba(X)

    def _suggest_params(self, trial) -> dict:
        return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-1, log=True)}


class BaselineLDA(ComposedClassifier):
    """Линейный дискриминантный анализ Фишера [Fisher, 1936]. Линейный бейзлайн."""

    def __init__(self, shrinkage: float | None = None, solver: str = "svd"):
        self.shrinkage = shrinkage
        self.solver = solver

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model_ = LinearDiscriminantAnalysis(solver=self.solver, shrinkage=self.shrinkage)
        self.model_.fit(X, y)
        return self

    def predict(self, X): return self.model_.predict(X)
    def predict_proba(self, X): return self.model_.predict_proba(X)

    def _suggest_params(self, trial) -> dict:
        solver = trial.suggest_categorical("solver", ["svd", "lsqr", "eigen"])
        shrinkage = None
        if solver != "svd":
            if trial.suggest_categorical("use_shrinkage", [True, False]):
                shrinkage = trial.suggest_float("shrinkage", 0.0, 1.0)
        return {"solver": solver, "shrinkage": shrinkage}


class BaselineElasticNet(ComposedClassifier):
    """ElasticNet-регрессия с L1/L2-смесью + округление прогнозов. Линейный бейзлайн."""

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 5000):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model_ = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                                  max_iter=self.max_iter, random_state=42)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        raw = self.model_.predict(X)
        indices = np.clip(np.round(raw).astype(int), 0, len(self.classes_) - 1)
        return self.classes_[indices]

    def _suggest_params(self, trial) -> dict:
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }


class BaselineKernelSVM(ComposedClassifier):
    """Метод опорных векторов с RBF-ядром [Cortes & Vapnik, 1995]. Нелинейный бейзлайн."""

    def __init__(self, C: float = 1.0, gamma: str | float = "scale"):
        self.C = C
        self.gamma = gamma

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model_ = SVC(C=self.C, kernel="rbf", gamma=self.gamma,
                           probability=True, random_state=42)
        self.model_.fit(X, y)
        return self

    def predict(self, X): return self.model_.predict(X)
    def predict_proba(self, X): return self.model_.predict_proba(X)

    def _suggest_params(self, trial) -> dict:
        return {
            "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }


class BaselineRandomForest(ComposedClassifier):
    """Случайный лес [Breiman, 2001] из 100 деревьев. Нелинейный бейзлайн."""

    def __init__(self, n_estimators: int = 100, max_depth: int | None = None,
                 min_samples_leaf: int = 1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        self.classes_ = np.unique(y)
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf, random_state=42, n_jobs=-1)
        self.model_.fit(X, y)
        return self

    def predict(self, X): return self.model_.predict(X)
    def predict_proba(self, X): return self.model_.predict_proba(X)

    def _suggest_params(self, trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }


class BaselineXGBoost(ComposedClassifier):
    """Gradient Boosting XGBoost [Chen & Guestrin, 2016] с scale_pos_weight для дисбаланса. Нелинейный бейзлайн."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, subsample: float = 1.0,
                 colsample_bytree: float = 1.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

    def fit(self, X, y):
        import xgboost as xgb
        self.classes_ = np.unique(y)
        self.model_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
            eval_metric="logloss",
            random_state=42, verbosity=0, n_jobs=-1)
        self.model_.fit(X, y)
        return self

    def predict(self, X): return self.model_.predict(X)
    def predict_proba(self, X): return self.model_.predict_proba(X)

    def _suggest_params(self, trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }


class BaselineLightGBM(ComposedClassifier):
    """Gradient Boosting LightGBM [Ke et al., 2017] с scale_pos_weight для дисбаланса. Нелинейный бейзлайн."""

    def __init__(self, n_estimators: int = 100, num_leaves: int = 31,
                 learning_rate: float = 0.1, subsample: float = 1.0,
                 colsample_bytree: float = 1.0):
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

    def fit(self, X, y):
        import lightgbm as lgb
        self.classes_ = np.unique(y)
        self.model_ = lgb.LGBMClassifier(
            n_estimators=self.n_estimators, num_leaves=self.num_leaves,
            learning_rate=self.learning_rate, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
            random_state=42, verbosity=-1, n_jobs=-1)
        self.model_.fit(X, y)
        return self

    def predict(self, X): return self.model_.predict(X)
    def predict_proba(self, X): return self.model_.predict_proba(X)

    def _suggest_params(self, trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }


class BaselineMLP(ComposedClassifier):
    """Многослойный перцептрон [Rumelhart et al., 1986] — нелинейный бейзлайн.

    Используется sklearn-овская реализация MLPClassifier с двумя скрытыми
    слоями (64 и 32 нейрона по умолчанию), активацией ReLU и оптимизатором
    Adam. Включает раннюю остановку (10% валидация) для предотвращения
    переобучения.
    """

    def __init__(self, hidden_layer_sizes: tuple = (64, 32),
                 alpha: float = 1e-4, learning_rate_init: float = 1e-3,
                 max_iter: int = 200):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter

    def fit(self, X, y):
        from sklearn.neural_network import MLPClassifier
        self.classes_ = np.unique(y)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.model_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu", solver="adam",
            alpha=self.alpha, learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10,
            random_state=42)
        self.model_.fit(X_scaled, y)
        return self

    def predict(self, X): return self.model_.predict(self.scaler_.transform(X))
    def predict_proba(self, X): return self.model_.predict_proba(self.scaler_.transform(X))

    def _suggest_params(self, trial) -> dict:
        # Архитектура — на сетке (sklearn принимает tuple)
        arch = trial.suggest_categorical(
            "arch",
            ["(32,)", "(64,)", "(128,)", "(64, 32)", "(128, 64)", "(64, 32, 16)"],
        )
        return {
            "hidden_layer_sizes": eval(arch),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-1, log=True),
        }


# %% [cell 12]
class PolynomialFeaturesClassifier(ComposedClassifier):
    """PolynomialFeatures -> StandardScaler -> линейная модель."""

    def __init__(self, degree: int = 2, interaction_only: bool = False,
                 max_features: int | None = 100, base_model: str = "logistic",
                 base_kwargs: dict | None = None):
        self.degree = degree
        self.interaction_only = interaction_only
        self.max_features = max_features
        self.base_model = base_model
        self.base_kwargs = base_kwargs

    def _select_features(self, X):
        if self.max_features is None or X.shape[1] <= self.max_features:
            self.feature_idx_ = None
            return X
        variances = np.var(X, axis=0)
        self.feature_idx_ = np.argsort(variances)[-self.max_features:]
        return X[:, self.feature_idx_]

    def _apply_selection(self, X):
        return X if self.feature_idx_ is None else X[:, self.feature_idx_]

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X_sel = self._select_features(X)
        self.poly_ = PolynomialFeatures(degree=self.degree,
                                        interaction_only=self.interaction_only,
                                        include_bias=False)
        self.scaler_ = StandardScaler()
        X_poly = self.poly_.fit_transform(X_sel)
        X_scaled = self.scaler_.fit_transform(X_poly)
        kwargs = self.base_kwargs or get_base_defaults(self.base_model)
        self.model_ = create_base_model(self.base_model, **kwargs)
        self.model_.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_sel = self._apply_selection(X)
        return self.model_.predict(self.scaler_.transform(self.poly_.transform(X_sel)))

    def predict_proba(self, X):
        X_sel = self._apply_selection(X)
        return get_proba(self.model_, self.scaler_.transform(self.poly_.transform(X_sel)))

    def _suggest_params(self, trial) -> dict:
        params = {
            "degree": trial.suggest_int("degree", 2, 3),
            "interaction_only": trial.suggest_categorical("interaction_only", [True, False]),
        }
        base_suggested = suggest_base_params(trial, self.base_model)
        if base_suggested:
            params["base_kwargs"] = base_suggested
        return params

# %% [cell 13]
class SplineFeaturesClassifier(ComposedClassifier):
    """SplineTransformer -> StandardScaler -> линейная модель."""

    def __init__(self, n_knots: int = 5, degree: int = 3,
                 base_model: str = "logistic", base_kwargs: dict | None = None):
        self.n_knots = n_knots
        self.degree = degree
        self.base_model = base_model
        self.base_kwargs = base_kwargs

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.input_scaler_ = StandardScaler()
        X_scaled = self.input_scaler_.fit_transform(X)
        self.spline_ = SplineTransformer(n_knots=self.n_knots, degree=self.degree,
                                         include_bias=False)
        X_spline = self.spline_.fit_transform(X_scaled)
        self.scaler_ = StandardScaler()
        X_final = self.scaler_.fit_transform(X_spline)
        kwargs = self.base_kwargs or get_base_defaults(self.base_model)
        self.model_ = create_base_model(self.base_model, **kwargs)
        self.model_.fit(X_final, y)
        return self

    def _transform(self, X):
        return self.scaler_.transform(
            self.spline_.transform(self.input_scaler_.transform(X)))

    def predict(self, X): return self.model_.predict(self._transform(X))
    def predict_proba(self, X): return get_proba(self.model_, self._transform(X))

    def _suggest_params(self, trial) -> dict:
        params = {"n_knots": trial.suggest_int("n_knots", 3, 10),
                  "degree": trial.suggest_int("degree", 2, 3)}
        base_suggested = suggest_base_params(trial, self.base_model)
        if base_suggested:
            params["base_kwargs"] = base_suggested
        return params

# %% [cell 15]
class LinearBoostClassifier(ComposedClassifier):
    """AdaBoost с линейными слабыми классификаторами."""

    def __init__(self, n_estimators: int = 10, learning_rate: float = 1.0,
                 base_model: str = "logistic", base_kwargs: dict | None = None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_model = base_model
        self.base_kwargs = base_kwargs

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        kwargs = self.base_kwargs or get_base_defaults(self.base_model)
        base_estimator = create_base_model(self.base_model, **kwargs)
        self.model_ = AdaBoostClassifier(estimator=base_estimator,
                                          n_estimators=self.n_estimators,
                                          learning_rate=self.learning_rate,
                                          random_state=42)
        self.model_.fit(X_scaled, y)
        return self

    def predict(self, X): return self.model_.predict(self.scaler_.transform(X))
    def predict_proba(self, X): return self.model_.predict_proba(self.scaler_.transform(X))

    def _suggest_params(self, trial) -> dict:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 5, 50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
        }
        base_suggested = suggest_base_params(trial, self.base_model)
        if base_suggested:
            params["base_kwargs"] = base_suggested
        return params

# %% [cell 16]
class BaggingFeatureSubspaceClassifier(ComposedClassifier):
    """BaggingClassifier с линейными моделями (аналог Random Forest)."""

    def __init__(self, n_estimators: int = 10, max_features: float = 0.7,
                 max_samples: float = 0.8, base_model: str = "logistic",
                 base_kwargs: dict | None = None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.base_model = base_model
        self.base_kwargs = base_kwargs

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        kwargs = self.base_kwargs or get_base_defaults(self.base_model)
        base_estimator = create_base_model(self.base_model, **kwargs)
        self.model_ = BaggingClassifier(
            estimator=base_estimator, n_estimators=self.n_estimators,
            max_features=self.max_features, max_samples=self.max_samples,
            random_state=42)
        self.model_.fit(X_scaled, y)
        return self

    def predict(self, X): return self.model_.predict(self.scaler_.transform(X))
    def predict_proba(self, X): return self.model_.predict_proba(self.scaler_.transform(X))

    def _suggest_params(self, trial) -> dict:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 5, 50),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
        }
        base_suggested = suggest_base_params(trial, self.base_model)
        if base_suggested:
            params["base_kwargs"] = base_suggested
        return params

# %% [cell 17]
class MaxOfLinearClassifier(ComposedClassifier):
    """Скор класса = максимум из k линейных функций (§2.2.4 ВКР).

    Для каждого класса обучаются k локальных линейных моделей по схеме
    «один против остальных» с k-means подкластеризацией внутри класса.
    Глобальный скор класса — максимум k индивидуальных скоринговых
    выходов; финальная вероятность — softmax от пары скоров двух классов.
    """

    def __init__(self, n_hyperplanes: int = 3, base_model: str = "logistic",
                 base_kwargs: dict | None = None):
        self.n_hyperplanes = n_hyperplanes
        self.base_model = base_model
        self.base_kwargs = base_kwargs

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        assert len(self.classes_) == 2, "Только бинарная классификация"
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        kwargs = self.base_kwargs or get_base_defaults(self.base_model)
        self.class_models_ = {}
        self.class_kmeans_ = {}

        for c in self.classes_:
            X_c = X_scaled[y == c]
            n_clusters = min(self.n_hyperplanes, len(X_c))
            if n_clusters < 2:
                model = create_base_model(self.base_model, **kwargs)
                model.fit(X_scaled, y)
                self.class_models_[c] = [model]
                self.class_kmeans_[c] = None
                continue

            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            km.fit(X_c)
            self.class_kmeans_[c] = km
            cluster_labels = km.predict(X_c)
            indices_c = np.where(y == c)[0]
            models = []

            for k in range(n_clusters):
                cluster_mask = cluster_labels == k
                if cluster_mask.sum() < 2:
                    model = create_base_model(self.base_model, **kwargs)
                    model.fit(X_scaled, y)
                else:
                    sample_weight = np.ones(len(y))
                    sample_weight[indices_c[cluster_mask]] *= 3.0
                    model = create_base_model(self.base_model, **kwargs)
                    try:
                        model.fit(X_scaled, y, sample_weight=sample_weight)
                    except TypeError:
                        model.fit(X_scaled, y)
                models.append(model)

            self.class_models_[c] = models

        return self

    def _model_score(self, model, X_scaled, target_class):
        if hasattr(model, "decision_function"):
            score = model.decision_function(X_scaled)
            return -score if target_class == self.classes_[0] else score
        else:
            return model.predict_proba(X_scaled)[:, list(self.classes_).index(target_class)]

    def _class_scores(self, X_scaled):
        scores = np.zeros((X_scaled.shape[0], len(self.classes_)))
        for idx, c in enumerate(self.classes_):
            model_scores = np.column_stack([
                self._model_score(m, X_scaled, c) for m in self.class_models_[c]
            ])
            scores[:, idx] = np.max(model_scores, axis=1)
        return scores

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        X_scaled = self.scaler_.transform(X)
        scores = self._class_scores(X_scaled)
        scores_shifted = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def _suggest_params(self, trial) -> dict:
        params = {"n_hyperplanes": trial.suggest_int("n_hyperplanes", 2, 6)}
        base_suggested = suggest_base_params(trial, self.base_model)
        if base_suggested:
            params["base_kwargs"] = base_suggested
        return params

# %% [cell 19]
class CascadeLinearClassifier(ComposedClassifier):
    """Каскад линейных моделей с обогащением признаков на каждом уровне."""

    def __init__(self, n_stages: int = 3, base_model: str = "logistic",
                 base_kwargs: dict | None = None):
        self.n_stages = n_stages
        self.base_model = base_model
        self.base_kwargs = base_kwargs

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.input_scaler_ = StandardScaler()
        X_scaled = self.input_scaler_.fit_transform(X)
        kwargs = self.base_kwargs or get_base_defaults(self.base_model)
        self.stages_ = []
        self.scalers_ = []
        X_current = X_scaled
        for i in range(self.n_stages):
            scaler = StandardScaler()
            X_input = scaler.fit_transform(X_current)
            self.scalers_.append(scaler)
            model = create_base_model(self.base_model, **kwargs)
            model.fit(X_input, y)
            self.stages_.append(model)
            if i < self.n_stages - 1:
                X_current = np.hstack([X_scaled, get_proba(model, X_input)])
        return self

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        X_scaled = self.input_scaler_.transform(X)
        X_current = X_scaled
        for i, (scaler, model) in enumerate(zip(self.scalers_, self.stages_)):
            X_input = scaler.transform(X_current)
            if i == self.n_stages - 1:
                return get_proba(model, X_input)
            X_current = np.hstack([X_scaled, get_proba(model, X_input)])
        return get_proba(self.stages_[-1], self.scalers_[-1].transform(X_current))

    def _suggest_params(self, trial) -> dict:
        params = {"n_stages": trial.suggest_int("n_stages", 2, 5)}
        base_suggested = suggest_base_params(trial, self.base_model)
        if base_suggested:
            params["base_kwargs"] = base_suggested
        return params

# %% [cell 21]
class PiecewiseLinearClassifier(ComposedClassifier):
    """KMeans-разбиение -> локальные линейные модели в регионах."""

    def __init__(self, n_regions: int = 3, soft_assignment: bool = False,
                 base_model: str = "logistic", base_kwargs: dict | None = None):
        self.n_regions = n_regions
        self.soft_assignment = soft_assignment
        self.base_model = base_model
        self.base_kwargs = base_kwargs

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.kmeans_ = KMeans(n_clusters=self.n_regions, random_state=42, n_init=10)
        labels = self.kmeans_.fit_predict(X_scaled)
        kwargs = self.base_kwargs or get_base_defaults(self.base_model)
        self.models_ = []
        for k in range(self.n_regions):
            mask = labels == k
            model = create_base_model(self.base_model, **kwargs)
            if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
                model.fit(X_scaled, y)
            else:
                model.fit(X_scaled[mask], y[mask])
            self.models_.append(model)
        return self

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        X_scaled = self.scaler_.transform(X)
        if self.soft_assignment:
            distances = self.kmeans_.transform(X_scaled)
            weights = 1.0 / (distances + 1e-8)
            weights /= weights.sum(axis=1, keepdims=True)
            proba = np.zeros((X.shape[0], len(self.classes_)))
            for k, model in enumerate(self.models_):
                proba += weights[:, k:k+1] * get_proba(model, X_scaled)
            return proba
        else:
            cluster_labels = self.kmeans_.predict(X_scaled)
            proba = np.zeros((X.shape[0], len(self.classes_)))
            for k, model in enumerate(self.models_):
                mask = cluster_labels == k
                if mask.any():
                    proba[mask] = get_proba(model, X_scaled[mask])
            return proba

    def _suggest_params(self, trial) -> dict:
        params = {
            "n_regions": trial.suggest_int("n_regions", 2, 10),
            "soft_assignment": trial.suggest_categorical("soft_assignment", [True, False]),
        }
        base_suggested = suggest_base_params(trial, self.base_model)
        if base_suggested:
            params["base_kwargs"] = base_suggested
        return params

# %% [cell 22]
class MixtureOfLinearExpertsClassifier(ComposedClassifier):
    """Смесь линейных экспертов с мягким гейтингом (EM-алгоритм)."""

    def __init__(self, n_experts: int = 3, n_em_steps: int = 10,
                 base_model: str = "logistic", base_kwargs: dict | None = None):
        self.n_experts = n_experts
        self.n_em_steps = n_em_steps
        self.base_model = base_model
        self.base_kwargs = base_kwargs

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples = X.shape[0]
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        kwargs = self.base_kwargs or get_base_defaults(self.base_model)
        self.experts_ = [create_base_model(self.base_model, **kwargs)
                         for _ in range(self.n_experts)]
        self.gate_ = LogisticRegression(max_iter=1000, random_state=42)
        rng = np.random.RandomState(42)
        responsibilities = rng.dirichlet(np.ones(self.n_experts), size=n_samples)

        for _ in range(self.n_em_steps):
            assignments = np.argmax(responsibilities, axis=1)
            for k, expert in enumerate(self.experts_):
                w = responsibilities[:, k]
                if w.sum() < 1.0:
                    expert.fit(X_scaled, y)
                    continue
                try:
                    expert.fit(X_scaled, y, sample_weight=w)
                except TypeError:
                    mask = assignments == k
                    if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
                        expert.fit(X_scaled, y)
                    else:
                        expert.fit(X_scaled[mask], y[mask])

            gate_targets = np.argmax(responsibilities, axis=1)
            if len(np.unique(gate_targets)) >= 2:
                self.gate_.fit(X_scaled, gate_targets)

            gate_proba = self._gate_proba(X_scaled)
            expert_likelihood = np.zeros((n_samples, self.n_experts))
            for k, expert in enumerate(self.experts_):
                p = get_proba(expert, X_scaled)[:, 1]
                expert_likelihood[:, k] = np.where(y == 1, p, 1 - p)

            responsibilities = gate_proba * expert_likelihood
            row_sums = np.maximum(responsibilities.sum(axis=1, keepdims=True), 1e-10)
            responsibilities = responsibilities / row_sums

        return self

    def _gate_proba(self, X_scaled):
        if self.n_experts == 1:
            return np.ones((X_scaled.shape[0], 1))
        try:
            return self.gate_.predict_proba(X_scaled)
        except Exception:
            return np.full((X_scaled.shape[0], self.n_experts), 1.0 / self.n_experts)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        X_scaled = self.scaler_.transform(X)
        gate_proba = self._gate_proba(X_scaled)
        mixture_proba_1 = sum(
            gate_proba[:, k] * get_proba(expert, X_scaled)[:, 1]
            for k, expert in enumerate(self.experts_)
        )
        mixture_proba_1 = np.clip(mixture_proba_1, 0, 1)
        return np.column_stack([1 - mixture_proba_1, mixture_proba_1])

    def _suggest_params(self, trial) -> dict:
        params = {
            "n_experts": trial.suggest_int("n_experts", 2, 6),
            "n_em_steps": trial.suggest_int("n_em_steps", 5, 20),
        }
        base_suggested = suggest_base_params(trial, self.base_model)
        if base_suggested:
            params["base_kwargs"] = base_suggested
        return params

# %% [cell 24]
class FuzzyLogicLinearClassifier(ComposedClassifier):
    """Нечёткая логика над линейными предсказаниями (AND/OR/XOR + мета-модель)."""

    def __init__(self, n_base_models: int = 4, use_and: bool = True,
                 use_or: bool = True, use_xor: bool = True,
                 base_model: str = "logistic", base_kwargs: dict | None = None):
        self.n_base_models = n_base_models
        self.use_and = use_and
        self.use_or = use_or
        self.use_xor = use_xor
        self.base_model = base_model
        self.base_kwargs = base_kwargs

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        kwargs = self.base_kwargs or get_base_defaults(self.base_model)
        n_features = X_scaled.shape[1]
        rng = np.random.RandomState(42)
        self.base_models_ = []
        self.feature_subsets_ = []

        for i in range(self.n_base_models):
            n_selected = max(2, int(n_features * rng.uniform(0.6, 1.0)))
            subset = rng.choice(n_features, size=n_selected, replace=False)
            subset.sort()
            self.feature_subsets_.append(subset)
            self.base_models_.append(create_base_model(self.base_model, **kwargs))

        oof_predictions = np.zeros((X_scaled.shape[0], self.n_base_models))
        for i, (model, subset) in enumerate(zip(self.base_models_, self.feature_subsets_)):
            X_sub = X_scaled[:, subset]
            try:
                oof_pred = cross_val_predict(model, X_sub, y, cv=3,
                                              method="predict_proba")[:, 1]
            except Exception:
                oof_pred = cross_val_predict(model, X_sub, y, cv=3, method="predict")
            oof_predictions[:, i] = oof_pred
            model.fit(X_sub, y)

        fuzzy_features = self._build_fuzzy_features(oof_predictions)
        self.meta_scaler_ = StandardScaler()
        fuzzy_scaled = self.meta_scaler_.fit_transform(fuzzy_features)
        self.meta_model_ = LogisticRegression(max_iter=1000, random_state=42)
        self.meta_model_.fit(fuzzy_scaled, y)
        return self

    def _build_fuzzy_features(self, predictions):
        features = [predictions, 1 - predictions]
        pairs = list(combinations(range(predictions.shape[1]), 2))
        if self.use_and and pairs:
            features.append(np.column_stack([
                predictions[:, i] * predictions[:, j] for i, j in pairs]))
        if self.use_or and pairs:
            features.append(np.column_stack([
                predictions[:, i] + predictions[:, j] - predictions[:, i] * predictions[:, j]
                for i, j in pairs]))
        if self.use_xor and pairs:
            features.append(np.column_stack([
                predictions[:, i] * (1 - predictions[:, j]) +
                predictions[:, j] * (1 - predictions[:, i])
                for i, j in pairs]))
        return np.hstack([f for f in features if f.size > 0])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        X_scaled = self.scaler_.transform(X)
        predictions = np.zeros((X_scaled.shape[0], self.n_base_models))
        for i, (model, subset) in enumerate(zip(self.base_models_, self.feature_subsets_)):
            predictions[:, i] = get_proba(model, X_scaled[:, subset])[:, 1]
        fuzzy_features = self._build_fuzzy_features(predictions)
        fuzzy_scaled = self.meta_scaler_.transform(fuzzy_features)
        return self.meta_model_.predict_proba(fuzzy_scaled)

    def _suggest_params(self, trial) -> dict:
        params = {
            "n_base_models": trial.suggest_int("n_base_models", 3, 6),
            "use_and": trial.suggest_categorical("use_and", [True, False]),
            "use_or": trial.suggest_categorical("use_or", [True, False]),
            "use_xor": trial.suggest_categorical("use_xor", [True, False]),
        }
        base_suggested = suggest_base_params(trial, self.base_model)
        if base_suggested:
            params["base_kwargs"] = base_suggested
        return params

# %% [cell 26]
# Порядок соответствует §2.3 ВКР: 5 принципов нелинейности → 9 архитектур
_COMPOSED_MODELS = [
    # §2.3.1 — Преобразование признаков
    PolynomialFeaturesClassifier,
    SplineFeaturesClassifier,
    # §2.3.2 — Ансамблирование
    LinearBoostClassifier,
    BaggingFeatureSubspaceClassifier,
    MaxOfLinearClassifier,
    # §2.3.3 — Последовательная композиция
    CascadeLinearClassifier,
    # §2.3.4 — Локализация
    PiecewiseLinearClassifier,
    MixtureOfLinearExpertsClassifier,
    # §2.3.5 — Нелинейная комбинация скоров
    FuzzyLogicLinearClassifier,
]

# Несовместимые комбинации (см. §2.5 ВКР): LinearBoost требует sample_weight,
# а scikit-learn-овская реализация LDA его не поддерживает.
_INCOMPATIBLE = {(LinearBoostClassifier, "lda")}

# Базовые модели (4 линейных + 4 нелинейных) + 35 составных конфигураций
MODEL_REGISTRY: list = [
    BaselineLogisticRegression,
    BaselineNaiveBayes,
    BaselineLDA,
    BaselineElasticNet,
    BaselineKernelSVM,
    BaselineRandomForest,
    BaselineXGBoost,
    BaselineLightGBM,
    BaselineMLP,
]
for _ModelClass in _COMPOSED_MODELS:
    for _bm in BASE_MODEL_NAMES:
        if (_ModelClass, _bm) not in _INCOMPATIBLE:
            MODEL_REGISTRY.append(make_variant(_ModelClass, _bm))

print(f"Всего моделей в реестре: {len(MODEL_REGISTRY)}")
print(f"  Линейных бейзлайнов:   4")
print(f"  Нелинейных бейзлайнов: 5")
print(f"  Составных моделей:     {len(MODEL_REGISTRY) - 8}")

# %% [cell 28]
RESULTS_DIR = Path("results_notebook")
RESULTS_DIR.mkdir(exist_ok=True)

from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)


def _get_scores(model, X) -> np.ndarray | None:
    """Скоры для класса 1 (минорный/целевой)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def _compute_metrics(y_true, y_pred, y_scores) -> dict:
    """Все метрики считаются для positive=1=миноритарный класс.

    Дополнительно фиксируется balanced_accuracy и macro_f1 как меры,
    устойчивые к дисбалансу классов.  PR-AUC (average precision)
    приоритетная метрика на сильно несбалансированных датасетах
    (SECOM, Credit Card Fraud) — см. §2.4.2 ВКР.
    """
    metrics = {
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall":    recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1":        f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_macro":  f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_scores)) if y_scores is not None else float("nan")
    except ValueError:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_scores)) if y_scores is not None else float("nan")
    except ValueError:
        metrics["pr_auc"] = float("nan")
    return metrics


def evaluate_cv(model, X, y, n_folds: int = 5, random_state: int = 42):
    """Стратифицированная k-fold CV с фиксированным random_state.

    Используется один и тот же сплит для режима «без оптимизации» и
    для финальной оценки после Optuna-тюнинга (см. §2.4.3).
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_metrics, fit_times = [], []

    for train_idx, val_idx in skf.split(X, y):
        fold_model = clone(model)
        t0 = time.perf_counter()
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            fold_model.fit(X[train_idx], y[train_idx])
        fit_times.append(time.perf_counter() - t0)

        y_pred = fold_model.predict(X[val_idx])
        y_scores = _get_scores(fold_model, X[val_idx])
        fold_metrics.append(_compute_metrics(y[val_idx], y_pred, y_scores))

    result = {}
    for key in fold_metrics[0]:
        values = [m[key] for m in fold_metrics]
        result[key] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values))
    result["fit_time"] = float(np.mean(fit_times))
    result["n_folds"] = int(n_folds)
    return result

# %% [cell 30]
N_FOLDS = 5
RANDOM_STATE = 42
rows = []

for dataset_fn in DATASET_REGISTRY:
    try:
        dataset = dataset_fn()
    except Exception as e:
        print(f"[SKIP] {dataset_fn.__name__}: {e}")
        continue

    pos_rate = float(dataset.y.mean())
    print(f"\nDataset: {dataset.name} {dataset.X.shape}  pos_rate={pos_rate:.4f}")

    for ModelClass in MODEL_REGISTRY:
        model = ModelClass()
        model_name = type(model).__name__
        try:
            metrics = evaluate_cv(model, dataset.X, dataset.y,
                                   n_folds=N_FOLDS, random_state=RANDOM_STATE)
            print(f"  {model_name:<55} F1={metrics['f1']:.4f}  "
                  f"PR-AUC={metrics['pr_auc']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")
            rows.append({"model": model_name, "dataset": dataset.name,
                         "pos_rate": pos_rate, **metrics})
        except Exception as e:
            print(f"  {model_name:<55} [FAIL] {type(e).__name__}: {e}")

results_df = pd.DataFrame(rows)
results_df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
print(f"\nРезультаты сохранены в {RESULTS_DIR / 'metrics.csv'}")
results_df

# %% [cell 32]
# Байесовская оптимизация гиперпараметров (Optuna, TPE).
# Внешняя оценка и внутренняя CV для тюнинга используют ОДНУ И ТУ ЖЕ
# схему — стратифицированную 5-fold CV с фиксированным random_state.
# Это даёт корректное сравнение «без оптимизации vs после оптимизации»:
# обе ветки оцениваются на идентичных сплитах.
N_TRIALS = 50
N_FOLDS_TUNE = 5
RANDOM_STATE = 42
TUNE_DATASETS = None  # None = все датасеты


def _has_custom_suggest(model) -> bool:
    return type(model)._suggest_params is not ComposedClassifier._suggest_params


def tune_model(model, X, y, n_trials: int, n_folds: int, random_state: int = 42):
    def objective(trial):
        params = model._suggest_params(trial)
        candidate = clone(model).set_params(**params)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            m = clone(candidate)
            m.fit(X[train_idx], y[train_idx])
            scores.append(f1_score(
                y[val_idx], m.predict(X[val_idx]),
                pos_label=1, zero_division=0))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)
    best_params = model._suggest_params(study.best_trial)
    model.set_params(**best_params)
    return model, study.best_value


tuned_rows = []
for dataset_fn in DATASET_REGISTRY:
    try:
        dataset = dataset_fn()
    except Exception as e:
        print(f"[SKIP] {dataset_fn.__name__}: {e}")
        continue

    if TUNE_DATASETS is not None and dataset.name not in TUNE_DATASETS:
        continue

    pos_rate = float(dataset.y.mean())
    print(f"\nDataset: {dataset.name} {dataset.X.shape}  pos_rate={pos_rate:.4f}")

    for ModelClass in MODEL_REGISTRY:
        model = ModelClass()
        model_name = type(model).__name__
        try:
            tuned = _has_custom_suggest(model)
            if tuned:
                print(f"  {model_name:<55} [tuning {N_TRIALS} trials...]", end="", flush=True)
                model, best_val = tune_model(
                    model, dataset.X, dataset.y, N_TRIALS, N_FOLDS_TUNE, RANDOM_STATE)
                print(f" best_inner_F1={best_val:.4f}", end="")
            metrics = evaluate_cv(model, dataset.X, dataset.y,
                                   n_folds=N_FOLDS_TUNE, random_state=RANDOM_STATE)
            print(f"  ->  F1={metrics['f1']:.4f}  PR-AUC={metrics['pr_auc']:.4f}")
            tuned_rows.append({"model": model_name, "dataset": dataset.name,
                                "tuned": tuned, "pos_rate": pos_rate, **metrics})
        except Exception as e:
            print(f"  {model_name:<55} [FAIL] {type(e).__name__}: {e}")

tuned_df = pd.DataFrame(tuned_rows)
tuned_df.to_csv(RESULTS_DIR / "metrics_tuned.csv", index=False)
print(f"\nРезультаты с тюнингом сохранены в {RESULTS_DIR / 'metrics_tuned.csv'}")
tuned_df

# %% [cell 34]
# Парный bootstrap для сравнения «без оптимизации» и «после оптимизации».
# Идея: на каждом фолде сохраняем per-sample предсказания обеих моделей и
# повторяемой бутстрэп-подвыборкой получаем 95% CI для разности метрики
# Δ = metric(tuned) − metric(default).  Поскольку обе модели оцениваются
# на одних и тех же фолдах с одним и тем же random_state, это валидное
# парное сравнение.
def paired_bootstrap_delta(
    y_true, y_pred_a, y_pred_b, y_score_a=None, y_score_b=None,
    metric="f1", n_boot: int = 2000, seed: int = 42,
):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if metric == "f1":
            a = f1_score(y_true[idx], y_pred_a[idx], pos_label=1, zero_division=0)
            b = f1_score(y_true[idx], y_pred_b[idx], pos_label=1, zero_division=0)
        elif metric == "pr_auc":
            try:
                a = average_precision_score(y_true[idx], y_score_a[idx])
                b = average_precision_score(y_true[idx], y_score_b[idx])
            except Exception:
                continue
        else:
            raise ValueError(metric)
        deltas.append(b - a)
    deltas = np.asarray(deltas)
    return {
        "mean_delta": float(deltas.mean()),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "p_one_sided": float((deltas <= 0).mean()),  # H0: tuned <= default
    }


# Пример использования (в обычном расчёте обе модели обучаются на
# одних и тех же сплитах; полные предсказания агрегируются по фолдам и
# подаются в paired_bootstrap_delta).  При практическом запуске замените
# заглушки реальными массивами.
print("paired_bootstrap_delta — вспомогательная функция для §3.6 ВКР.")

