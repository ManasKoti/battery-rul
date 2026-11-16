"""Train the chosen model on the training cells and save everything the demo needs.

Run from the project root:  python -m scripts.train_final
"""

import joblib

from src.config import MODELS_DIR
from src.data import load_cache
from src.features import DISCHARGE_FEATURES, build_feature_table
from src.models import conformal_intervals, make_elastic_net
from src.splits import PRIMARY_TEST, SECONDARY_TEST, TRAIN

FEATURES = DISCHARGE_FEATURES   # won leave-one-out CV on the training cells (notebook 06)
ALPHA = 0.1                     # 90% prediction intervals


def main():
    cells = load_cache()
    train = build_feature_table(cells, TRAIN)
    test_ids = [c for c in PRIMARY_TEST + SECONDARY_TEST if c in cells]

    # conformal_intervals returns predictions + bounds; we only need the width factor here.
    pred, lower, _ = conformal_intervals(make_elastic_net, train[FEATURES], train["cycle_life"],
                                         train[FEATURES], alpha=ALPHA)
    width_factor = float((pred / lower)[0])   # interval = [pred / f, pred * f]

    model = make_elastic_net().fit(train[FEATURES], train["cycle_life"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / "cycle_life_model.joblib"
    joblib.dump({"model": model, "features": FEATURES, "width_factor": width_factor,
                 "alpha": ALPHA, "test_ids": test_ids}, out)
    print(f"Saved {out}  (interval factor x{width_factor:.2f})")


if __name__ == "__main__":
    main()
