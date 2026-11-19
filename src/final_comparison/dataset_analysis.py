"""Utilities for comparing cleaned polymer datasets."""

import pandas as pd
from scipy.stats import wasserstein_distance


def compare_subset(parent, subset, label):
    """Compare a subset with the remaining samples in its parent dataset."""
    selected = parent.sample_id.isin(subset.sample_id)

    if selected.sum() != len(subset) or not 0 < selected.sum() < len(parent):
        raise ValueError(f"{label}: invalid subset")

    a = parent.loc[selected, "Tg_K"]
    b = parent.loc[~selected, "Tg_K"]

    return {
        "subset": label,
        "parent_n": len(parent),
        "subset_n": len(a),
        "subset_pct": 100 * len(a) / len(parent),
        "subset_mean_K": a.mean(),
        "complement_mean_K": b.mean(),
        "mean_difference_K": a.mean() - b.mean(),
        "wasserstein_K": wasserstein_distance(a, b),
    }