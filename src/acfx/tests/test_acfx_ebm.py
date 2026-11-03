"""End-to-end behaviour of AcfxEBM (Explainable Boosting Machine blackbox)."""
import numpy as np
import pytest
from interpret.glassbox import ExplainableBoostingClassifier

from acfx import AcfxEBM
from acfx.evaluation.loss import compute_causal_penalty
from acfx.tests.helpers import FAST, IRIS_ADJACENCY, IRIS_CAUSAL_ORDER, SEED, assert_within_bounds, other_class


def test_init_sets_attributes():
    model = ExplainableBoostingClassifier()
    explainer = AcfxEBM(model)
    assert explainer.blackbox == model
    assert explainer.optimizer is None


def test_fit_with_targets_trains_ebm_and_sets_internal_state(X_train, y_train, X_test, pbounds):
    explainer = AcfxEBM(ExplainableBoostingClassifier(interactions=0, outer_bags=2, random_state=SEED))
    explainer.fit(X_train, pbounds, None, IRIS_ADJACENCY, y_train)

    assert explainer.X.equals(X_train)
    assert explainer.pbounds == pbounds
    assert len(explainer.predict(X_test)) == len(X_test)


@pytest.mark.parametrize("causal_order, adjacency", [
    (IRIS_CAUSAL_ORDER, None), (None, IRIS_ADJACENCY), (None, None), ([1, 2, 3], IRIS_ADJACENCY)])
def test_plausibility_on_requires_complete_causal_graph(ebm_model, X_train, pbounds, query_instance,
                                                        causal_order, adjacency):
    explainer = AcfxEBM(ebm_model).fit(X_train, pbounds, causal_order, adjacency)
    with pytest.raises(ValueError):
        explainer.counterfactual(desired_class=1, plausibility_weight=0.5, query_instance=query_instance, **FAST)


@pytest.mark.parametrize("causal_order, adjacency", [
    (IRIS_CAUSAL_ORDER, None), (None, IRIS_ADJACENCY), (None, None), ([1, 2, 3], IRIS_ADJACENCY)])
def test_plausibility_off_does_not_need_causal_graph(ebm_model, X_train, pbounds, query_instance,
                                                     causal_order, adjacency):
    explainer = AcfxEBM(ebm_model).fit(X_train, pbounds, causal_order, adjacency)
    result = explainer.counterfactual(desired_class=1, plausibility_weight=0, query_instance=query_instance, **FAST)
    assert result.shape == (1, 4)


def test_plausibility_with_causal_graph_returns_counterfactual_within_bounds(ebm_model, X_train, pbounds,
                                                                            features_order, query_instance):
    explainer = AcfxEBM(ebm_model).fit(X_train, pbounds, IRIS_CAUSAL_ORDER, IRIS_ADJACENCY,
                                       features_order=features_order)
    result = explainer.counterfactual(desired_class=other_class(ebm_model, query_instance),
                                      plausibility_weight=0.5, query_instance=query_instance, **FAST)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 4)
    assert_within_bounds(result, pbounds, features_order)


@pytest.mark.slow
def test_counterfactual_flips_prediction_and_is_more_plausible_than_query(ebm_model, X_train, pbounds,
                                                                         features_order, query_instance):
    desired_class = other_class(ebm_model, query_instance)
    explainer = AcfxEBM(ebm_model).fit(X=X_train, adjacency_matrix=IRIS_ADJACENCY,
                                       causal_order=IRIS_CAUSAL_ORDER, pbounds=pbounds,
                                       features_order=features_order)

    cf = explainer.counterfactual(desired_class=desired_class, query_instance=query_instance,
                                  plausibility_weight=1, init_points=10, n_iter=200)

    assert explainer.predict(cf)[0] == desired_class
    original_penalty = compute_causal_penalty(np.array([query_instance]), IRIS_ADJACENCY, IRIS_CAUSAL_ORDER)
    assert compute_causal_penalty(cf, IRIS_ADJACENCY, IRIS_CAUSAL_ORDER) <= original_penalty
