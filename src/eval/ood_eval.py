"""OOD evaluation helpers."""

from __future__ import annotations

from src.eval.metrics import ood_gap


def summarize_split_accuracies(
    train_accuracy: float,
    val_iid_accuracy: float,
    iid_test_accuracy: float,
    ood_test_accuracy: float,
) -> dict[str, float]:
    return {
        "train_accuracy": float(train_accuracy),
        "val_iid_accuracy": float(val_iid_accuracy),
        "iid_test_accuracy": float(iid_test_accuracy),
        "ood_test_accuracy": float(ood_test_accuracy),
        "ood_gap": ood_gap(iid_test_accuracy, ood_test_accuracy),
    }
