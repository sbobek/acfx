import lingam
import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from interpret.glassbox import ExplainableBoostingClassifier

from src.acfx.AcfxEBM import AcfxEBM
from src.acfx.evaluation.loss import compute_causal_penalty


@pytest.fixture
def sample_data():
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_causal_model(X):
    causal_model = lingam.DirectLiNGAM()
    causal_model.fit(X)
    return causal_model

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
    adjacency_matrix = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.0, 0.0],
        [0.0, 0.6, 0.0, 0.0],
        [0.5, 0.0, 0.7, 0.0]
    ])
    causal_order = None
    pbounds = {col: (X_train[col].min(), X_train[col].max()) for col in X_train.columns}

    explainer.fit(X_train, pbounds, causal_order,adjacency_matrix , y_train)

    assert explainer.X.equals(X_train)
    assert explainer.pbounds == pbounds

def test_predict_returns_predictions(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = ExplainableBoostingClassifier()
    model.fit(X_train, y_train)
    explainer = AcfxEBM(model)
    preds = explainer.predict(X_test)
    assert len(preds) == len(X_test)

def test_raises_error_plausability_weight_on_missing_args(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    causal_model = train_causal_model(X_train)
    causal_order = causal_model.causal_order_
    model = ExplainableBoostingClassifier()
    model.fit(X_train, y_train)
    explainer = AcfxEBM(model)
    query_instance = X_test.iloc[0].values
    pbounds = {col: (X_train[col].min(), X_train[col].max()) for col in X_train.columns}
    adjacency_matrix = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.0, 0.0],
        [0.0, 0.6, 0.0, 0.0],
        [0.5, 0.0, 0.7, 0.0]
    ])
    with pytest.raises(ValueError):
        explainer.fit(X_train, pbounds, causal_order, None, y_train)
        explainer.counterfactual(desired_class=1, plausibility_weight=0.5, query_instance=query_instance)
    with pytest.raises(ValueError):
        explainer.fit(X_train,pbounds , None, adjacency_matrix, y_train)
        explainer.counterfactual(desired_class=1, plausibility_weight=0.5, query_instance=query_instance)
    with pytest.raises(ValueError):
        explainer.fit(X_train, pbounds, None,None , y_train)
        explainer.counterfactual(desired_class=1, plausibility_weight=0.5, query_instance=query_instance)
    with pytest.raises(ValueError):
        explainer.fit(X_train,pbounds , [1,2,3],adjacency_matrix , y_train)
        explainer.counterfactual(desired_class=1, plausibility_weight=0.5, query_instance=query_instance)


def test_works_fine_plausability_weight_off_missing_args(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    causal_model = train_causal_model(X_train)
    causal_order = causal_model.causal_order_
    model = ExplainableBoostingClassifier()
    model.fit(X_train, y_train)
    explainer = AcfxEBM(model)
    query_instance = X_test.iloc[0].values
    pbounds = {col: (X_train[col].min(), X_train[col].max()) for col in X_train.columns}
    adjacency_matrix = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.0, 0.0],
        [0.0, 0.6, 0.0, 0.0],
        [0.5, 0.0, 0.7, 0.0]
    ])
    explainer.fit(X_train, pbounds, causal_order, None, y_train)
    explainer.counterfactual(desired_class=1, plausibility_weight=0, query_instance=query_instance)

    explainer.fit(X_train,pbounds , None, adjacency_matrix, y_train)
    explainer.counterfactual(desired_class=1, plausibility_weight=0, query_instance=query_instance)

    explainer.fit(X_train, pbounds, None,None , y_train)
    explainer.counterfactual(desired_class=1, plausibility_weight=0, query_instance=query_instance)

    explainer.fit(X_train, pbounds, [1,2,3],adjacency_matrix , y_train)
    explainer.counterfactual(desired_class=1, plausibility_weight=0, query_instance=query_instance)

def test_plausibility_with_causal_order(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    causal_model = train_causal_model(X_train)
    causal_order = causal_model.causal_order_
    model = ExplainableBoostingClassifier()
    model.fit(X_train, y_train)
    explainer = AcfxEBM(model)
    query_instance = X_test.iloc[0].values
    pbounds = {col: (X_train[col].min(), X_train[col].max()) for col in X_train.columns}
    adjacency_matrix = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.0, 0.0],
        [0.0, 0.6, 0.0, 0.0],
        [0.5, 0.0, 0.7, 0.0]
    ])
    explainer.fit(X_train, pbounds, causal_order, adjacency_matrix, y_train)
    result = explainer.counterfactual(desired_class=1, plausibility_weight=0.5, query_instance=query_instance)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 4)

def test_counterfactual_calls_generate_cfs_returns_result(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = ExplainableBoostingClassifier()
    model.fit(X_train, y_train)
    explainer = AcfxEBM(model)
    query_instance = X_test.iloc[0].values
    pbounds = {col: (X_train[col].min(), X_train[col].max()) for col in X_train.columns}
    adjacency_matrix = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.0, 0.0],
        [0.0, 0.6, 0.0, 0.0],
        [0.5, 0.0, 0.7, 0.0]
    ])
    explainer.fit(X_train, pbounds, None, adjacency_matrix, y_train)

    result = explainer.counterfactual(desired_class=1, query_instance=query_instance)

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
    adjacency_matrix = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.0, 0.0],
        [0.0, 0.6, 0.0, 0.0],
        [0.5, 0.0, 0.7, 0.0]
    ])

    causal_order = list(range(len(features_order)))

    explainer.fit(X=X_train, adjacency_matrix=adjacency_matrix, causal_order=causal_order, pbounds=pbounds,
                  features_order=features_order)

    cf = explainer.counterfactual(desired_class=original_class, query_instance=query_instance)

    original_causal_penalty = compute_causal_penalty(np.array([query_instance]), adjacency_matrix, causal_order)
    cfs_causal_penalty = compute_causal_penalty(cf, adjacency_matrix, causal_order)

    print(f"Original penalty: {original_causal_penalty}, CF penalty: {cfs_causal_penalty}")
    assert cfs_causal_penalty <= original_causal_penalty
