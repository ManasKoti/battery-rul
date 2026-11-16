"""Streamlit demo: pick an unseen test cell, see its predicted lifetime with an interval.

Run from the project root:  streamlit run app.py
(first run `python -m scripts.train_final` to create models/cycle_life_model.joblib)
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import EOL_AH, MODELS_DIR, V_GRID
from src.data import clean_capacity, cycle_life, load_cache
from src.features import cell_features
from src.splits import PRIMARY_TEST


@st.cache_resource
def load_everything():
    return joblib.load(MODELS_DIR / "cycle_life_model.joblib"), load_cache()


bundle, cells = load_everything()

st.title("Battery cycle-life predictor")
st.caption("Predicts total cycle life (to 80% capacity) from the first 100 cycles. "
           "Every cell in the list was held out from training.")

cell_id = st.selectbox("Test cell", bundle["test_ids"],
                       format_func=lambda c: f"{c}  ({'primary' if c in PRIMARY_TEST else 'secondary'} test)")
cell = cells[cell_id]

x = pd.DataFrame([cell_features(cell)])[bundle["features"]]
pred = float(bundle["model"].predict(x)[0])
f = bundle["width_factor"]
true_life = cycle_life(cell["qd"])

c1, c2, c3 = st.columns(3)
c1.metric("Predicted cycle life", f"{pred:,.0f}")
c2.metric(f"{int((1 - bundle['alpha']) * 100)}% interval", f"{pred / f:,.0f} – {pred * f:,.0f}")
c3.metric("Actual cycle life", "n/a" if np.isnan(true_life) else f"{true_life:,.0f}",
          delta=None if np.isnan(true_life) else f"{100 * (pred - true_life) / true_life:+.0f}% error",
          delta_color="off")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
dq = cell["qdlin"][99] - cell["qdlin"][9]
a1.plot(dq, V_GRID)
a1.set(xlabel="ΔQ = Q100 − Q10 (Ah)", ylabel="Voltage (V)", title="What the model sees")

qd = clean_capacity(cell["qd"])
a2.plot(np.arange(1, len(qd) + 1), qd, lw=1)
a2.axvline(100, ls="--", c="gray", lw=1)
a2.axhline(EOL_AH, ls=":", c="red", lw=1)
a2.axvspan(pred / f, pred * f, alpha=0.15)
a2.set(xlabel="Cycle", ylabel="Discharge capacity (Ah)", title="Full life (hidden from model)")
fig.tight_layout()
st.pyplot(fig)
