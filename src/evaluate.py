"""Metrics and plots shared by every model, so all models are judged identically."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred) / y_true) * 100)


def score(y_true, y_pred) -> dict:
    return {"RMSE (cycles)": round(rmse(y_true, y_pred), 1), "MAPE (%)": round(mape(y_true, y_pred), 1)}


def results_table(results: dict) -> pd.DataFrame:
    """results = {model_name: {split_name: (y_true, y_pred)}} -> tidy comparison table."""

    rows = []
    for model, splits in results.items():
        for split, (yt, yp) in splits.items():
            rows.append({"model": model, "split": split, **score(yt, yp)})
    return pd.DataFrame(rows).pivot(index="model", columns="split")


def parity_plot(y_true, y_pred, title="", ax=None, lower=None, upper=None):
    """Predicted vs true cycle life. Points on the diagonal are perfect predictions."""
    
    ax = ax or plt.gca()
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if lower is not None:
        ax.errorbar(y_true, y_pred, yerr=[y_pred - lower, upper - y_pred], fmt="none", alpha=0.3, color="gray")
    ax.scatter(y_true, y_pred, s=25)
    lim = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlim=lim, ylim=lim, xlabel="True cycle life", ylabel="Predicted cycle life",
           title=f"{title}  RMSE={rmse(y_true, y_pred):.0f}, MAPE={mape(y_true, y_pred):.1f}%")
    ax.set_aspect("equal")
    return ax