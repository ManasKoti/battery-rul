"""Load the HUST dataset (Ma et al., 2022) straight from its zip into the same compact format as MATR.

BatteryML's HUST preprocessor does the same steps (integrate current -> capacity,
skip the first two cycles of cell 7-5), but importing BatteryML pulls in numba,
which isn't needed here. Reading the zip directly keeps everything in the `rul` env.
"""

import pickle
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import N_EARLY_CYCLES, ROOT
from .data import _charge_duration, compute_qdlin

HUST_ZIP = ROOT / "data" / "raw" / "HUST" / "hust_data.zip"
HUST_CACHE = ROOT / "data" / "cache" / "hust_compact.pkl"


def integrate_capacity(current_a: np.ndarray, time_s: np.ndarray, charge: bool) -> np.ndarray:
    """Running charge (Ah) passed in one direction within a cycle (same as BatteryML's calc_Q)."""

    i = np.asarray(current_a, float)
    dt = np.diff(np.asarray(time_s, float), prepend=np.nan)
    step = np.where(i > 0, i, 0.0) if charge else np.where(i < 0, -i, 0.0)
    step = np.nan_to_num(step * dt / 3600.0)
    step[0] = 0.0
    return np.cumsum(step)


def compact_hust_cell(cell_id: str, cycle_frames: dict, n_curve_cycles: int = N_EARLY_CYCLES) -> dict:
    """One HUST cell (dict of per-cycle DataFrames) -> compact dict matching src.data.compact_cell."""

    keys = sorted(cycle_frames)
    if cell_id == "7-5":
        keys = keys[2:]  # first two cycles are faulty (as in BatteryML)
    qd, charge_time, curves = [], [], []
    for n, k in enumerate(keys):
        df = cycle_frames[k]
        i = df["Current (mA)"].to_numpy(float) / 1000.0
        t = df["Time (s)"].to_numpy(float)
        v = df["Voltage (V)"].to_numpy(float)
        q_dis = integrate_capacity(i, t, charge=False)
        qd.append(float(q_dis.max()) if len(q_dis) else np.nan)
        if n < n_curve_cycles:
            charge_time.append(_charge_duration(i, t))
            curves.append(compute_qdlin(v, i, q_dis))
    nan = np.full(len(qd), np.nan)
    return {
        "cell_id": cell_id,
        "cycle_number": np.arange(1, len(qd) + 1),
        "qd": np.array(qd),
        "ir": nan.copy(),       # HUST has no internal-resistance or temperature logs
        "t_mean": nan.copy(),
        "t_max": nan.copy(),
        "charge_time": np.array(charge_time),
        "qdlin": np.stack(curves),
        "charge_protocol": None,
    }


def hust_feature_table(cells: dict, ids=None, early: int = 10, late: int = 100):
    """Feature table for HUST cells, labelled with the dataset's own cycle life.

    HUST cells were cycled until end of life and the dataset's `rul` label at cycle 1
    equals the number of recorded cycles. Our integrated capacity at test end is about
    0.90 Ah, just above the 0.88 Ah threshold `src.data.cycle_life` looks for, so we use
    the dataset label (= number of cycles kept) instead of re-deriving it.
    """
    import pandas as pd
    from .features import cell_features

    ids = list(cells) if ids is None else ids
    rows = [{"cell_id": c, **cell_features(cells[c], early, late), "cycle_life": float(len(cells[c]["qd"]))}
            for c in ids]
    return pd.DataFrame(rows).set_index("cell_id")


def build_hust_cache(zip_path: Path = HUST_ZIP, out_path: Path = HUST_CACHE) -> dict:
    """Read every cell in hust_data.zip once and save the compact versions together."""

    cells = {}
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".pkl"))
        for name in tqdm(names, desc="Compacting HUST cells"):
            cell_id = Path(name).stem
            with zf.open(name) as fh:
                frames = pickle.load(fh)[cell_id]["data"]
            if len(frames) < N_EARLY_CYCLES:
                continue
            cells[cell_id] = compact_hust_cell(cell_id, frames)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(cells, fh)
    print(f"Saved {len(cells)} HUST cells to {out_path}")
    return cells
