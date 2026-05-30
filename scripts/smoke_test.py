#!/usr/bin/env python3
"""Fast operational smoke test for the research package."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from composed_linear_classifiers import DATASET_LOADERS, describe_model_registry, evaluate_cv
from composed_linear_classifiers.core import BaselineLogisticRegression, SplineFeaturesClassifier


def main() -> int:
    print(describe_model_registry())
    dataset = DATASET_LOADERS["breast_cancer"]()
    for model in [BaselineLogisticRegression(), SplineFeaturesClassifier(base_model="logistic")]:
        metrics = evaluate_cv(model, dataset.X, dataset.y, n_folds=3, random_state=42)
        print(type(model).__name__, f"F1={metrics['f1']:.4f}", f"PR-AUC={metrics['pr_auc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
