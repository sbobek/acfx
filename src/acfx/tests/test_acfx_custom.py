from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from overrides import overrides
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.acfx import AcfxCustom
from src.acfx.abstract import ModelBasedCounterOptimizer
from src.acfx.evaluation.casual_counterfactuals import compute_causal_penalty


@pytest.fixture
def sample_data():
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

class SomeCustomCounterOptimizer(ModelBasedCounterOptimizer):
    def __init__(self, model, X: pd.DataFrame, feature_bounds: Dict[str, Tuple[float, float]], n_iter: int = 100):
        if not hasattr(model, 'predict_proba'):
            raise AttributeError("Model must implement predict_proba()")
        self.model = model
        self.X = X
        self.feature_bounds = feature_bounds
        self.n_iter = n_iter

    @overrides
    def optimize_proba(self, target_class: int, feature_masked: list[str]) -> Dict[str, float]:
        base_instance = self.X.mean().copy()
        best_instance = base_instance.copy()
        best_score = self.model.predict_proba([base_instance])[0][target_class]

        for _ in range(self.n_iter):
            candidate = base_instance.copy()
            for feature_name in self.X.columns:
                if feature_name not in feature_masked and feature_name in self.feature_bounds:
                    min_val, max_val = self.feature_bounds[feature_name]
                    candidate[feature_name] = np.random.uniform(min_val, max_val)

            score = self.model.predict_proba([candidate])[0][target_class]
            if score > best_score:
                best_score = score
                best_instance = candidate.copy()

        return best_instance.to_dict()


def test_custom_cfx_optimizer(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    feature_masked = ["sepal width (cm)"]
    pbounds = {col: (X_train[col].min(), X_train[col].max()) for col in X_train.columns}
    optimizer = SomeCustomCounterOptimizer(model, X_test, pbounds)

    explainer = AcfxCustom(model)

    query_instance = X_test.iloc[0].values
    desired_class = model.predict([query_instance])[0]

    pbounds = {col: (X_train[col].min(), X_train[col].max()) for col in X_train.columns}
    features_order = X_train.columns.tolist()

    adjacency_matrix = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.0, 0.0],
        [0.0, 0.6, 0.0, 0.0],
        [0.5, 0.0, 0.7, 0.0]
    ])

    causal_order = list(range(len(features_order)))

    explainer.fit(X=X_train, query_instance=query_instance, adjacency_matrix=adjacency_matrix,
                  casual_order=causal_order, pbounds=pbounds,
                  features_order=features_order, optimizer=optimizer, masked_features=feature_masked)

    cf = explainer.counterfactual(desired_class=desired_class)

    cfs_casual_penalty = compute_causal_penalty(cf, adjacency_matrix, causal_order)

    assert cfs_casual_penalty is not None and cfs_casual_penalty > 0


