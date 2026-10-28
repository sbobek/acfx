"""Constants and assertion helpers shared by the acfx tests."""
import numpy as np
import pandas as pd

SEED = 42

# Small search budget: enough to exercise the whole pipeline, small enough to keep the suite fast.
FAST = dict(init_points=4, n_iter=15)

# Linear SEM over iris features (row = child, column = parent), used wherever a causal graph is needed
# without depending on lingam: x1 <- x0, x2 <- x1, x3 <- x0 + x2.
IRIS_ADJACENCY = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.8, 0.0, 0.0, 0.0],
    [0.0, 0.6, 0.0, 0.0],
    [0.5, 0.0, 0.7, 0.0],
])
IRIS_CAUSAL_ORDER = [0, 1, 2, 3]


def other_class(model, query_instance):
    """A class different from the one currently predicted for ``query_instance``."""
    current = model.predict(pd.DataFrame([query_instance], columns=model.feature_names_in_))[0]
    return int(next(c for c in model.classes_ if c != current))


def assert_within_bounds(cfs, pbounds, features_order):
    for i, feature in enumerate(features_order):
        low, high = pbounds[feature]
        assert np.all(cfs[:, i] >= low - 1e-9) and np.all(cfs[:, i] <= high + 1e-9), feature
