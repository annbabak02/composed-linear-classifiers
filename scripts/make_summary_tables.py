#!/usr/bin/env python3
"""Build compact result tables from a metrics CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/metrics_default_stream_from_notebook.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/summary_tables"))
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input)
    for column in ["f1", "pr_auc", "roc_auc", "fit_time"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    top = (
        df.sort_values(["dataset", "f1"], ascending=[True, False])
        .groupby("dataset", group_keys=False)
        .head(args.top_k)
    )
    top.to_csv(args.output_dir / "top_models_by_dataset.csv", index=False)

    family = df.assign(
        family=df["model"].str.replace(r"\[.*\]$", "", regex=True)
    )
    aggregations = {"f1_mean": ("f1", "mean")}
    if "pr_auc" in family.columns:
        aggregations["pr_auc_mean"] = ("pr_auc", "mean")
    if "fit_time" in family.columns:
        aggregations["fit_time_mean"] = ("fit_time", "mean")
    architecture = (
        family.groupby("family", as_index=False)
        .agg(**aggregations)
        .sort_values("f1_mean", ascending=False)
    )
    architecture.to_csv(args.output_dir / "mean_metrics_by_model_family.csv", index=False)
    print(f"Summary tables saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
