"""Loading BatteryML's per-cell pickles and shrinking them into a compact cache."""

import pickle
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter
from .config import CACHE_PATH, EOL_AH, NOMINAL_AH, N_EARLY_CYCLES, PROCESSED_DIR, V_GRID


def _cycle_max(values):
    """Largest value in a cycle's sample list, or NaN if the list is empty."""

    if values is None or len(values) == 0:
        return np.nan
    return float(np.nanmax(values))


def _cycle_mean(values):
    if values is None or len(values) == 0:
        return np.nan
    return float(np.nanmean(values))


def _charge_duration(current, time):
    """Total time spent with positive (charging) current in one cycle."""

    if current is None or time is None or len(current) < 2:
        return np.nan
    current, time = np.asarray(current, float), np.asarray(time, float)
    dt = np.diff(time)
    charging = current[1:] > 0.05
    return float(np.sum(dt[charging]))


def compute_qdlin(voltage, current, discharge_q, v_grid=V_GRID):
    """Build a Q(V) curve on a fixed voltage grid from raw samples."""

    v, i, q = (np.asarray(a, float) for a in (voltage, current, discharge_q))
    mask = i < -0.05
    if mask.sum() < 10:
        return np.full(len(v_grid), np.nan)
    v, q = v[mask], q[mask]
    order = np.argsort(v)
    v, q = v[order], q[order]
    return np.interp(v_grid, v, q, left=np.nan, right=np.nan)


def compact_cell(raw: dict, n_curve_cycles: int = N_EARLY_CYCLES, use_precomputed: bool = True) -> dict:
    """Turn one BatteryML cell dict into a small dict of numpy arrays."""

    cycles = raw["cycle_data"]
    qd = np.array([_cycle_max(c["discharge_capacity_in_Ah"]) for c in cycles])
    ir = np.array([c.get("internal_resistance_in_ohm") or np.nan for c in cycles], float)
    t_mean = np.array([_cycle_mean(c.get("temperature_in_C")) for c in cycles])
    t_max = np.array([_cycle_max(c.get("temperature_in_C")) for c in cycles])
    charge_time = np.array([_charge_duration(c["current_in_A"], c["time_in_s"]) for c in cycles[:n_curve_cycles]])

    curves = []
    for c in cycles[:n_curve_cycles]:
        if use_precomputed and c.get("Qdlin") is not None:
            curves.append(np.asarray(c["Qdlin"], float))
        else:
            curves.append(compute_qdlin(c["voltage_in_V"], c["current_in_A"], c["discharge_capacity_in_Ah"]))

    return {
        "cell_id": raw["cell_id"].split("_", 1)[-1],  # "MATR_b1c0" -> "b1c0"
        "cycle_number": np.array([c["cycle_number"] for c in cycles]),
        "qd": qd,
        "ir": ir,
        "t_mean": t_mean,
        "t_max": t_max,
        "charge_time": charge_time,
        "qdlin": np.stack(curves),  # shape (100, 1000): row k is cycle k+1
        "charge_protocol": raw.get("charge_protocol"),
    }


def build_cache(processed_dir: Path = PROCESSED_DIR, out_path: Path = CACHE_PATH, use_precomputed: bool = True) -> dict:
    """Read every processed cell once and save the compact versions together."""

    from tqdm import tqdm

    files = sorted(Path(processed_dir).glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl files in {processed_dir}. Run Step 2 first.")
    cells = {}
    for f in tqdm(files, desc="Compacting cells"):
        with open(f, "rb") as fh:
            raw = pickle.load(fh)
        if len(raw["cycle_data"]) < N_EARLY_CYCLES:
            continue  # can't compute 100-cycle features for very short tests
        cell = compact_cell(raw, use_precomputed=use_precomputed)
        cells[cell["cell_id"]] = cell
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(cells, fh)
    print(f"Saved {len(cells)} cells to {out_path}")
    return cells


def load_cache(path: Path = CACHE_PATH) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def clean_capacity(qd: np.ndarray, low: float = 0.5 * NOMINAL_AH, high: float = 1.2 * NOMINAL_AH) -> np.ndarray:
    """Replace physically impossible capacity readings with interpolated values."""

    qd = np.asarray(qd, float).copy()
    bad = ~np.isfinite(qd) | (qd < low) | (qd > high)
    if bad.all():
        return qd
    idx = np.arange(len(qd))
    qd[bad] = np.interp(idx[bad], idx[~bad], qd[~bad])
    return qd


def cycle_life(qd: np.ndarray, eol_ah: float = EOL_AH, window: int = 5,
               end_tol: float = 0.01) -> float:
    """Cycle number at which capacity first falls to 80% of nominal."""
    
    smooth = median_filter(clean_capacity(qd), size=window, mode="nearest")
    below = np.flatnonzero(smooth <= eol_ah)
    if below.size:
        return float(below[0] + 1)
    if smooth[-1] <= eol_ah + end_tol:
        return float(len(smooth))
    return float("nan")