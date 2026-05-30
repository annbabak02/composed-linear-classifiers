#!/usr/bin/env python3
"""Run Optuna tuning followed by stratified cross-validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from composed_linear_classifiers import DATASET_LOADERS, evaluate_cv, model_name, select_model_classes, tune_model
from composed_linear_classifiers.core import ComposedClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["breast_cancer"],
        choices=sorted(DATASET_LOADERS),
        help="Datasets to evaluate. Full thesis runs are computationally expensive.",
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "metrics_tuned.csv")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def has_custom_suggest(model) -> bool:
    return type(model)._suggest_params is not ComposedClassifier._suggest_params


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_classes = select_model_classes(args.models)

    rows: list[dict] = []
    failures: list[dict] = []
    for dataset_name in args.datasets:
        dataset = DATASET_LOADERS[dataset_name]()
        pos_rate = float(dataset.y.mean())
        print(f"Dataset: {dataset.name} {dataset.X.shape} pos_rate={pos_rate:.4f}")

        for model_class in model_classes:
            model = model_class()
            name = model_name(model)
            try:
                tuned = has_custom_suggest(model)
                best_inner_f1 = None
                if tuned:
                    print(f"  {name:<55} tuning {args.n_trials} trials...", end="", flush=True)
                    model, best_inner_f1 = tune_model(
                        model,
                        dataset.X,
                        dataset.y,
                        n_trials=args.n_trials,
                        n_folds=args.n_folds,
                        random_state=args.random_state,
                    )
                    print(f" best_inner_F1={best_inner_f1:.4f}", end="")
                metrics = evaluate_cv(
                    model,
                    dataset.X,
                    dataset.y,
                    n_folds=args.n_folds,
                    random_state=args.random_state,
                )
                rows.append(
                    {
                        "dataset": dataset.name,
                        "model": name,
                        "tuned": tuned,
                        "best_inner_f1": best_inner_f1,
                        "pos_rate": pos_rate,
                        **metrics,
                    }
                )
                print(f" -> F1={metrics['f1']:.4f} PR-AUC={metrics['pr_auc']:.4f}")
            except Exception as exc:
                failures.append({"dataset": dataset.name, "model": name, "error": repr(exc)})
                print(f"  {name:<55} FAIL {type(exc).__name__}: {exc}")
                if args.fail_fast:
                    raise

    pd.DataFrame(rows).to_csv(args.output, index=False)
    if failures:
        failure_path = args.output.with_name(args.output.stem + "_failures.csv")
        pd.DataFrame(failures).to_csv(failure_path, index=False)
        print(f"Failures saved to {failure_path}")
    print(f"Metrics saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
