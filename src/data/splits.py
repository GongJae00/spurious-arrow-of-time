"""Split naming and validation for STA-Bench."""

from __future__ import annotations


SPLIT_ORDER = ("train", "val_iid", "iid_test", "ood_test")
SPLIT_SPURIOUS_MODE_DEFAULTS = {
    "train": "correlated",
    "val_iid": "correlated",
    "iid_test": "correlated",
    "ood_test": "reversed",
}


def validate_split(split: str) -> None:
    if split not in SPLIT_ORDER:
        raise ValueError(f"split must be one of {SPLIT_ORDER}, got {split!r}")


def default_spurious_mode_for_split(split: str) -> str:
    validate_split(split)
    return SPLIT_SPURIOUS_MODE_DEFAULTS[split]

