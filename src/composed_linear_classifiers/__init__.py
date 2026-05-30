"""Composed nonlinear classifiers based on linear models."""

from .core import (
    DATASET_LOADERS,
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    BinaryDataset,
    describe_model_registry,
    evaluate_cv,
    model_name,
    paired_bootstrap_delta,
    select_model_classes,
    tune_model,
)

__all__ = [
    "BinaryDataset",
    "DATASET_LOADERS",
    "DATASET_REGISTRY",
    "MODEL_REGISTRY",
    "describe_model_registry",
    "evaluate_cv",
    "model_name",
    "paired_bootstrap_delta",
    "select_model_classes",
    "tune_model",
]
