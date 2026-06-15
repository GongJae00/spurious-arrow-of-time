"""EP sanity utilities."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def ep_ranking_correlations(estimated: np.ndarray, analytic_ep: np.ndarray) -> dict[str, float]:
    estimated = np.asarray(estimated, dtype=float)
    analytic_ep = np.asarray(analytic_ep, dtype=float)
    if estimated.shape[0] < 2 or np.std(estimated) < 1e-12 or np.std(analytic_ep) < 1e-12:
        return {"spearman": float("nan"), "pearson": float("nan")}
    return {
        "spearman": float(spearmanr(estimated, analytic_ep).statistic),
        "pearson": float(pearsonr(estimated, analytic_ep).statistic),
    }
