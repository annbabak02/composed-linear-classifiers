from __future__ import annotations

import sys
import unittest
from pathlib import Path

from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from composed_linear_classifiers import describe_model_registry, evaluate_cv
from composed_linear_classifiers.core import BaselineLogisticRegression, _ensure_minority_positive


class ResearchCodeSmokeTest(unittest.TestCase):
    def test_registry_shape(self):
        self.assertEqual(
            describe_model_registry(),
            {
                "total": 44,
                "linear_baselines": 4,
                "nonlinear_baselines": 5,
                "composed_models": 35,
            },
        )

    def test_minority_positive_invariant(self):
        y = _ensure_minority_positive([1, 1, 1, 0])
        self.assertLessEqual(y.mean(), 0.5)

    def test_cv_runs_on_synthetic_binary_data(self):
        X, y = make_classification(
            n_samples=90,
            n_features=6,
            n_informative=4,
            n_redundant=0,
            weights=[0.8, 0.2],
            random_state=42,
        )
        metrics = evaluate_cv(BaselineLogisticRegression(), X, y, n_folds=3, random_state=42)
        self.assertIn("f1", metrics)
        self.assertGreaterEqual(metrics["balanced_acc"], 0.0)


if __name__ == "__main__":
    unittest.main()
