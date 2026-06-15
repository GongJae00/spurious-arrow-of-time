"""Train LENS-inspired forward/reverse classifier baseline on train split only."""

from src.train.common import run_arrow_cli


if __name__ == "__main__":
    run_arrow_cli("lens_like_arrow_classifier")
