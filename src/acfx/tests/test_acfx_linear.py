"""End-to-end behaviour of AcfxLinear (linear-additive blackbox, e.g. LogisticRegression)."""
import pytest

from acfx import AcfxLinear
from acfx.evaluation.loss import compute_causal_penalty
from acfx.tests.helpers import FAST, IRIS_ADJACENCY, IRIS_CAUSAL_ORDER, assert_within_bounds, other_class


def test_counterfactual_reaches_desired_class_within_bounds(lr_model, X_test, pbounds, features_order,
                                                           query_instance):
    desired_class = other_class(lr_model, query_instance)
    explainer = AcfxLinear(lr_model).fit(X=X_test, pbounds=pbounds, causal_order=IRIS_CAUSAL_ORDER,
                                         adjacency_matrix=IRIS_ADJACENCY, features_order=features_order)

    cf = explainer.counterfactual(desired_class=desired_class, query_instance=query_instance,
                                  plausibility_weight=1, **FAST)

    assert explainer.predict(cf)[0] == desired_class
    assert_within_bounds(cf, pbounds, features_order)


def test_counterfactual_has_causal_penalty(lr_model, X_test, pbounds, features_order, query_instance):
    explainer = AcfxLinear(lr_model).fit(X=X_test, pbounds=pbounds, causal_order=IRIS_CAUSAL_ORDER,
                                         adjacency_matrix=IRIS_ADJACENCY, features_order=features_order)

    cf = explainer.counterfactual(desired_class=other_class(lr_model, query_instance),
                                  query_instance=query_instance, **FAST)

    assert compute_causal_penalty(cf, IRIS_ADJACENCY, IRIS_CAUSAL_ORDER) > 0


@pytest.mark.parametrize("num_counterfactuals", [1, 3])
def test_returns_requested_number_of_distinct_counterfactuals(lr_model, X_train, pbounds, features_order,
                                                             query_instance, num_counterfactuals):
    explainer = AcfxLinear(lr_model).fit(X=X_train, pbounds=pbounds, features_order=features_order)
    cfs = explainer.counterfactual(desired_class=other_class(lr_model, query_instance),
                                   query_instance=query_instance, num_counterfactuals=num_counterfactuals, **FAST)
    assert cfs.shape == (num_counterfactuals, len(features_order))
    assert len({tuple(row) for row in cfs}) == num_counterfactuals
