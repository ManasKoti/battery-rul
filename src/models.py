"""Classical models + conformal prediction intervals."""

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .config import SEED


def pow10(z):
    """Inverse of log10. A named function (not a lambda) so saved models can be reloaded."""

    return np.power(10.0, z)


def _log_target(regressor):
    """Train on log10(cycle life), predict back in cycles.

    Lifetimes range from ~150 to ~2,300 cycles. In log space a 10% error counts
    the same for short- and long-lived cells, and predictions can never go negative.
    """
    return TransformedTargetRegressor(regressor=regressor, func=np.log10, inverse_func=pow10)


def make_elastic_net(seed: int = SEED):
    enet = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        cv=KFold(n_splits=5, shuffle=True, random_state=seed),
        max_iter=100_000,
    )
    return _log_target(make_pipeline(StandardScaler(), enet))


def make_lgbm(seed: int = SEED):
    gbm = LGBMRegressor(
        n_estimators=400, learning_rate=0.03,
        num_leaves=7, max_depth=3, min_child_samples=5,   # small trees: we only have ~41 cells
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=seed, verbose=-1,
    )
    return _log_target(gbm)


def conformal_intervals(model_factory, X_train, y_train, X_new, alpha: float = 0.1):
    """Distribution-free prediction intervals (jackknife / leave-one-out conformal).

    1. For each training cell, fit on all the others and predict it -> honest residuals.
    2. Take the (1 - alpha) quantile of those residuals (in log space = relative error).
    3. Widen every new prediction by that amount.
    If the test cells resemble the training cells, ~ (1 - alpha) of true values land inside.
    """
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    oof = cross_val_predict(model_factory(), X_train, y_train, cv=LeaveOneOut())
    resid = np.abs(np.log10(y_train) - np.log10(oof))
    n = len(resid)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q = np.quantile(resid, level, method="higher")
    model = model_factory().fit(X_train, y_train)
    pred = model.predict(np.asarray(X_new))
    return pred, pred / 10 ** q, pred * 10 ** q


def coverage(y_true, lower, upper) -> float:
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= lower) & (y_true <= upper)) * 100)