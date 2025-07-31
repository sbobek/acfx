import numpy as np
import pytest
from sklearn.base import ClassifierMixin
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
    def __init__(self, model: ClassifierMixin, feature_mask: np.ndarray, step_size=0.05, max_iter=100):
        self.model = model
        self.feature_mask = feature_mask
        self.step_size = step_size
        self.max_iter = max_iter

    def optimize_proba(self, target_class: int, feature_masked: np.ndarray):
        x = feature_masked.copy().astype(float)
        best_x = x.copy()
        best_proba = self.model.predict_proba([x])[0][target_class]

        for _ in range(self.max_iter):
            for i in range(len(x)):
                if not self.feature_mask[i]:
                    continue

                for delta in [-self.step_size, self.step_size]:
                    x_try = best_x.copy()
                    x_try[i] += delta
                    proba = self.model.predict_proba([x_try])[0][target_class]
                    if proba > best_proba:
                        best_proba = proba
                        best_x = x_try
        return best_x

def test_custom_cfx_optimizer(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    feature_mask = np.array([True] * len(X_train))
    optimizer = SomeCustomCounterOptimizer(model, feature_mask)

    explainer = AcfxCustom(model)

    query_instance = X_test.iloc[0].values
    original_class = model.predict([query_instance])[0]

    pbounds = {col: (X_train[col].min(), X_train[col].max()) for col in X_train.columns}
    features_order = X_train.columns.tolist()

    adjacency_matrix = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.0, 0.0],
        [0.0, 0.6, 0.0, 0.0],
        [0.5, 0.0, 0.7, 0.0]
    ])

    causal_order = list(range(len(features_order)))

    explainer.fit(X=X_train, query_instance=query_instance, adjacency_matrix=adjacency_matrix, casual_order=causal_order, pbounds=pbounds,
                  features_order=features_order, optimizer=optimizer)

    cf = explainer.counterfactual(desired_class=original_class)

    cfs_casual_penalty = compute_causal_penalty(cf, adjacency_matrix, causal_order)

    assert cfs_casual_penalty is not None and cfs_casual_penalty > 0
    assert np.linalg.norm(query_instance - cf) > 1e-2


