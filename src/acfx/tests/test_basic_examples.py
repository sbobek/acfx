"""Executable versions of the usage examples from README.md (causal graph from LiNGAM or defined by hand).

They run the default search budget used in the README and are therefore marked as slow.
"""
import lingam
import networkx as nx
import numpy as np
import pytest
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from acfx import AcfxCustom, AcfxEBM, RandomSearchCounterOptimizer
from acfx.tests.helpers import IRIS_ADJACENCY, SEED, assert_within_bounds, other_class

pytestmark = pytest.mark.slow

EXAMPLE_BUDGET = dict(init_points=10, n_iter=100)


def get_lingam(X):
    causal_model = lingam.DirectLiNGAM()
    causal_model.fit(X)
    return causal_model.adjacency_matrix_, causal_model.causal_order_


def get_causal_order(adjacency_matrix):
    graph = nx.DiGraph()
    n = adjacency_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] != 0:
                graph.add_edge(j, i)  # j -> i (parent -> child)
    return list(nx.topological_sort(graph))


def test_custom_adjacency_gives_topological_causal_order():
    assert get_causal_order(IRIS_ADJACENCY) == [0, 1, 2, 3]


def test_ebm_with_lingam_graph(X_train, y_train, X_test, pbounds, features_order):
    explainer = AcfxEBM(ExplainableBoostingClassifier(random_state=SEED))
    adjacency_matrix, causal_order = get_lingam(X_train)
    explainer.fit(X=X_train, adjacency_matrix=adjacency_matrix, causal_order=causal_order, pbounds=pbounds,
                  y=y_train, features_order=features_order)

    query_instance = X_test.iloc[0].values
    desired_class = other_class(explainer.blackbox, query_instance)
    cf = explainer.counterfactual(desired_class=desired_class, query_instance=query_instance,
                                  plausibility_weight=1, **EXAMPLE_BUDGET)

    assert explainer.predict(cf)[0] == desired_class
    assert_within_bounds(cf, pbounds, features_order)


def test_ebm_with_custom_adjacency(X_train, y_train, X_test, pbounds, features_order):
    explainer = AcfxEBM(ExplainableBoostingClassifier(random_state=SEED))
    explainer.fit(X=X_train, adjacency_matrix=IRIS_ADJACENCY, causal_order=get_causal_order(IRIS_ADJACENCY),
                  pbounds=pbounds, y=y_train, features_order=features_order)

    query_instance = X_test.iloc[0].values
    desired_class = other_class(explainer.blackbox, query_instance)
    cf = explainer.counterfactual(desired_class=desired_class, query_instance=query_instance,
                                  plausibility_weight=1, **EXAMPLE_BUDGET)

    assert explainer.predict(cf)[0] == desired_class
    assert_within_bounds(cf, pbounds, features_order)


def test_random_forest_with_custom_optimizer(X_train, y_train, X_test, pbounds, features_order):
    model = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(X_train, y_train)
    adjacency_matrix, causal_order = get_lingam(X_train)
    feature_masked = ["sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    optimizer = RandomSearchCounterOptimizer(model, X_test, pbounds)

    explainer = AcfxCustom(model)
    explainer.fit(X=X_train, adjacency_matrix=adjacency_matrix, causal_order=causal_order, pbounds=pbounds,
                  features_order=features_order, optimizer=optimizer, masked_features=feature_masked)

    query_instance = X_test.iloc[0].values
    desired_class = other_class(model, query_instance)
    cf = explainer.counterfactual(desired_class=desired_class, query_instance=query_instance, **EXAMPLE_BUDGET)

    assert explainer.predict(cf)[0] == desired_class
    np.testing.assert_array_equal(cf[0, 0], query_instance[0])  # sepal length is not actionable
