"""Classify items as decisive / redundant / counter-evidential per §6.4."""

import numpy as np


def classify_items(phi: np.ndarray, d: np.ndarray,
                   decisive_threshold: float = 0.0) -> dict:
    """Classify each item based on its Shapley value and differential.

    Categories:
      - decisive: φ_i > decisive_threshold (large positive contributors)
      - counter_evidential: φ_i < 0 (push toward a flip; necessarily d_i = -1)
      - redundant: φ_i ≈ 0 (includes d_i = 0 items and diluted d_i = +1 items)
    """
    labels = np.empty(len(phi), dtype='<U20')
    labels[:] = "redundant"
    labels[phi > decisive_threshold] = "decisive"
    labels[phi < -decisive_threshold] = "counter_evidential"

    return {
        "labels": labels,
        "n_decisive": int((labels == "decisive").sum()),
        "n_counter": int((labels == "counter_evidential").sum()),
        "n_redundant": int((labels == "redundant").sum()),
        "decisive_indices": np.where(labels == "decisive")[0],
        "counter_indices": np.where(labels == "counter_evidential")[0],
        "decisive_phi_sum": float(phi[labels == "decisive"].sum()),
        "counter_phi_sum": float(phi[labels == "counter_evidential"].sum()),
    }
