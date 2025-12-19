import networkx as nx
import numpy as np
import pytest
from acfx import AcfxEBM, RandomSearchCounterOptimizer, AcfxCustom
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lingam

@pytest.fixture
def sample_data():
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def get_lingam(X):
    causal_model = lingam.DirectLiNGAM()
    causal_model.fit(X)
    return (causal_model.adjacency_matrix_, causal_model.causal_order_)

def get_pbounds(X):
    return {col: (X[col].min(), X[col].max()) for col in X.columns}

def get_causal_order(adjacency_matrix):
    # Build directed graph
    G = nx.DiGraph()
    n = adjacency_matrix.shape[0]

    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] != 0:
                G.add_edge(j, i)  # j → i (parent → child)
    # Topological sort
    causal_order = list(nx.topological_sort(G))
    return causal_order


def test_basic_example1(sample_data):
    model = ExplainableBoostingClassifier()
    explainer = AcfxEBM(model)
    X_train, X_test, y_train, y_test = sample_data
    pbounds = get_pbounds(X_train)
    adjacency_matrix, causal_order = get_lingam(X_train)
    features_order = X_train.columns.tolist()
    explainer.fit(X=X_train, adjacency_matrix=adjacency_matrix, causal_order=causal_order, pbounds=pbounds,
                  y=y_train, features_order=features_order)

    query_instance = X_test.iloc[0].values
    original_class = explainer.predict([query_instance])[0]
    cf = explainer.counterfactual(desired_class=original_class, query_instance=query_instance)
    print(cf)

def test_basic_example_custom_adjacency(sample_data):
    model = ExplainableBoostingClassifier()
    explainer = AcfxEBM(model)
    X_train, X_test, y_train, y_test = sample_data
    pbounds = get_pbounds(X_train)
    adjacency_matrix = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.0, 0.0],
        [0.0, 0.6, 0.0, 0.0],
        [0.5, 0.0, 0.7, 0.0]
    ])
    causal_order = get_causal_order(adjacency_matrix)
    features_order = X_train.columns.tolist()
    explainer.fit(X=X_train, adjacency_matrix=adjacency_matrix, causal_order=causal_order, pbounds=pbounds,
                  y=y_train, features_order=features_order)

    query_instance = X_test.iloc[0].values
    original_class = explainer.predict([query_instance])[0]
    cf = explainer.counterfactual(desired_class=original_class, query_instance=query_instance)
    print(cf)

def test_basic_example2(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    adjacency_matrix, causal_order = get_lingam(X_train)
    pbounds = get_pbounds(X_train)
    features_order = X_train.columns.tolist()
    feature_masked = ["sepal width (cm)"]
    optimizer = RandomSearchCounterOptimizer(model, X_test, pbounds)

    explainer = AcfxCustom(model)
    explainer.fit(X=X_train, adjacency_matrix=adjacency_matrix, causal_order=causal_order, pbounds=pbounds,
                  features_order=features_order, optimizer=optimizer, masked_features=feature_masked)

    query_instance = X_test.iloc[0].values
    original_class = explainer.predict([query_instance])[0]
    cf = explainer.counterfactual(desired_class=original_class, query_instance=query_instance)
    print(cf)