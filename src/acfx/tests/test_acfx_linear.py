import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from src.acfx.AcfxLinear import AcfxLinear
from src.acfx.evaluation.casual_counterfactuals import compute_causal_penalty

@pytest.fixture
def sample_data():
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def test_counterfactual_has_casual_penalty(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    explainer = AcfxLinear(model)

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
                  features_order=features_order)

    cf = explainer.counterfactual(desired_class=original_class)

    cfs_casual_penalty = compute_causal_penalty(cf, adjacency_matrix, causal_order)

    assert cfs_casual_penalty is not None and cfs_casual_penalty > 0