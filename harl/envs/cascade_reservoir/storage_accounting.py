"""Neutral reservoir inventory accounting, without a target or pass/fail test."""
import numpy as np


def storage_change_summary(initial, final=None):
    """Return signed final-minus-initial inventory and its net depletion."""
    start = np.asarray(initial, dtype=np.float64)
    if start.shape != (5,) or not np.isfinite(start).all():
        raise ValueError('Initial storage must be a finite five-vector')
    if final is None:
        return dict(initial_storage_m3=start.tolist(), final_storage_m3=None,
                    storage_change_m3=None, net_storage_change_m3=None,
                    total_storage_depletion_m3=None, total_storage_depletion_billion_m3=None)
    end = np.asarray(final, dtype=np.float64)
    if end.shape != (5,) or not np.isfinite(end).all():
        raise ValueError('Final storage must be a finite five-vector')
    change = end-start
    net = float(change.sum())
    return dict(initial_storage_m3=start.tolist(), final_storage_m3=end.tolist(),
                storage_change_m3=change.tolist(), net_storage_change_m3=net,
                total_storage_depletion_m3=-net, total_storage_depletion_billion_m3=-net/1e9)
