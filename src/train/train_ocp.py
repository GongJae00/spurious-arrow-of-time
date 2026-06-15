"""Train OCP-style order/arrow pretraining baseline on train split only."""

from src.train.common import run_arrow_cli


if __name__ == "__main__":
    run_arrow_cli("ocp_style")
