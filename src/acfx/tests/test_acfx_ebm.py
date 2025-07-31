import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from interpret.glassbox import ExplainableBoostingClassifier

from src.acfx.AcfxEBM import AcfxEBM
from src.acfx.evaluation.casual_counterfactuals import compute_causal_penalty


@pytest.fixture
def sample_data():
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def test_init_sets_attributes():
    model = ExplainableBoostingClassifier()
    explainer = AcfxEBM(model)
    assert explainer.blackbox == model
    assert explainer.optimizer is None

def test_fit_sets_internal_state(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = ExplainableBoostingClassifier()
    explainer = AcfxEBM(model)
    query_instance = X_test.iloc[0]
    adjacency_matrix = None
    causal_order = None
    pbounds = {col: (X_train[col].min(), X_train[col].max()) for col in X_train.columns}

    explainer.fit(X_train, query_instance, adjacency_matrix, causal_order, pbounds, y_train)

    assert explainer.X.equals(X_train)
    assert explainer.query_instance is not None
    assert explainer.pbounds == pbounds
    assert explainer.optimizer is not None

def test_predict_returns_predictions(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = ExplainableBoostingClassifier()
    model.fit(X_train, y_train)
    explainer = AcfxEBM(model)
    preds = explainer.predict(X_test)
    assert len(preds) == len(X_test)

def test_counterfactual_calls_generate_cfs_returns_result(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = ExplainableBoostingClassifier()
    model.fit(X_train, y_train)
    explainer = AcfxEBM(model)
    query_instance = X_test.iloc[0].values
    pbounds = {col: (X_train[col].min(), X_train[col].max()) for col in X_train.columns}
    explainer.fit(X_train, query_instance, None, None, pbounds, y_train)

    result = explainer.counterfactual(desired_class=1)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 4)

def test_counterfactual_has_lower_causal_penalty(sample_data):

    X_train, X_test, y_train, y_test = sample_data
    model = ExplainableBoostingClassifier()
    model.fit(X_train, y_train)
    explainer = AcfxEBM(model)

    query_instance = X_test.iloc[0].values
    original_class = model.predict([query_instance])[0]

    pbounds = {col: (X_train[col].min(), X_train[col].max()) for col in X_train.columns}
    features_order = X_train.columns.tolist()

    #adjacency_matrix = np.eye(len(features_order))
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

    original_casual_penalty = compute_causal_penalty(np.array([query_instance]), adjacency_matrix, causal_order)
    cfs_casual_penalty = compute_causal_penalty(cf, adjacency_matrix, causal_order)

    print(f"Original penalty: {original_casual_penalty}, CF penalty: {cfs_casual_penalty}")
    assert cfs_casual_penalty <= original_casual_penalty
