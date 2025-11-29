import pytest
from acfx import AcfxEBM
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import lingam

import networkx as nx


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

@pytest.fixture
def sample_data():
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def get_lingam(X):
    causal_model = lingam.DirectLiNGAM()
    causal_model.fit(X)
    return causal_model.adjacency_matrix_

def get_pbounds(X):
    return {col: (X[col].min(), X[col].max()) for col in X.columns}


def test_basic_example1(sample_data):
    model = ExplainableBoostingClassifier()
    explainer = AcfxEBM(model)
    X_train, X_test, y_train, y_test = sample_data
    pbounds = get_pbounds(X_train)
    adjacency_matrix = get_lingam(X_train)
    causal_order = get_causal_order(adjacency_matrix)
    features_order = X_train.columns.tolist()
    explainer.fit(X=X_train, adjacency_matrix=adjacency_matrix, casual_order=causal_order, pbounds=pbounds,
                  y=y_train, features_order=features_order)

