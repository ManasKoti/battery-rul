"""Feature engineering: turn a cell's first 100 cycles into a row of numbers."""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from .data import clean_capacity, cycle_life

EPS = 1e-8


def delta_q(qdlin: np.ndarray, early: int = 10, late: int = 100) -> np.ndarray:
    """ΔQ(V) = Q_late(V) - Q_early(V). Row k of qdlin is cycle k+1."""

    dq = qdlin[late - 1] - qdlin[early - 1]
    return dq[np.isfinite(dq)]


def cell_features(cell: dict, early: int = 10, late: int = 100) -> dict:
    """All candidate features for one cell, using only cycles 1..late."""

    dq = delta_q(cell["qdlin"], early, late)
    qd = clean_capacity(cell["qd"])[:late]
    ir = cell["ir"][:late]
    cycles = np.arange(2, late + 1)          # cycles 2..late (cycle 1 is often odd)
    slope, intercept = np.polyfit(cycles, qd[1:late], 1)

    return {
        # --- shape of the ΔQ(V) curve (the strongest signal) ---
        "dq_log_var": np.log10(np.var(dq) + EPS),
        "dq_log_min": np.log10(np.abs(dq.min()) + EPS),
        "dq_log_mean": np.log10(np.abs(dq.mean()) + EPS),
        "dq_log_skew": np.log10(np.abs(skew(dq)) + EPS),
        "dq_log_kurt": np.log10(np.abs(kurtosis(dq, fisher=False)) + EPS),
        # --- capacity fade curve ---
        "qd_cycle2": qd[1],
        "qd_max_minus_cycle2": qd[1:late].max() - qd[1],
        "qd_slope_2_late": slope,
        "qd_intercept_2_late": intercept,
        "qd_slope_last10": np.polyfit(np.arange(late - 9, late + 1), qd[late - 10:late], 1)[0],
        # --- charging, temperature, resistance ---
        "charge_time_first5": np.log(np.nanmean(cell["charge_time"][:5]) + EPS),
        "temp_mean_2_late": np.nanmean(cell["t_mean"][1:late]),
        "temp_max_2_late": np.nanmax(cell["t_max"][1:late]),
        "ir_min_2_late": np.nanmin(ir[1:late]),
        "ir_change": ir[late - 1] - ir[1],
    }


# Feature sets from Severson et al. (2019), mapped onto names.
VARIANCE_FEATURES = ["dq_log_var"]
DISCHARGE_FEATURES = ["dq_log_min", "dq_log_var", "dq_log_skew", "dq_log_kurt", "qd_cycle2", "qd_max_minus_cycle2"]
FULL_FEATURES = ["dq_log_min", "dq_log_var", "qd_slope_2_late", "qd_intercept_2_late", "qd_cycle2", "charge_time_first5", "temp_mean_2_late", "ir_min_2_late", "ir_change"]


def build_feature_table(cells: dict, ids: list, early: int = 10, late: int = 100) -> pd.DataFrame:
    """One row per cell: features + the target (cycle life). Missing cells are reported."""
    
    rows, missing = [], []
    for cid in ids:
        if cid not in cells:
            missing.append(cid)
            continue
        row = {"cell_id": cid, **cell_features(cells[cid], early, late)}
        row["cycle_life"] = cycle_life(cells[cid]["qd"])
        rows.append(row)
    if missing:
        print(f"Warning: {len(missing)} cells not found: {missing}")
    df = pd.DataFrame(rows).set_index("cell_id")
    return df.dropna(subset=["cycle_life"])