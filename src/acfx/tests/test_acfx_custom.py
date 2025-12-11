from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from overrides import overrides
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from RandomSearchCounterOptimizer import RandomSearchCounterOptimizer
from src.acfx import AcfxCustom
from src.acfx.abstract import ModelBasedCounterOptimizer
from src.acfx.evaluation.loss import compute_causal_penalty


@pytest.fixture
def sample_data():
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def test_custom_cfx_optimizer(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    feature_masked = ["sepal width (cm)"]
    pbounds = {col: (X_train[col].min(), X_train[col].max()) for col in X_train.columns}
    optimizer = RandomSearchCounterOptimizer(model, X_test, pbounds)

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

    explainer.fit(X=X_train, adjacency_matrix=adjacency_matrix,
                  causal_order=causal_order, pbounds=pbounds,
                  features_order=features_order, optimizer=optimizer, masked_features=feature_masked)

    cf = explainer.counterfactual(desired_class=desired_class, query_instance=query_instance)

    cfs_causal_penalty = compute_causal_penalty(cf, adjacency_matrix, causal_order)

    assert cfs_causal_penalty is not None and cfs_causal_penalty > 0


