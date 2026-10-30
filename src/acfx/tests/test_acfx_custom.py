"""End-to-end behaviour of AcfxCustom (any blackbox with predict_proba + a user-supplied counter-optimizer)."""
from acfx import AcfxCustom, RandomSearchCounterOptimizer
from acfx.evaluation.loss import compute_causal_penalty
from acfx.tests.helpers import FAST, IRIS_ADJACENCY, IRIS_CAUSAL_ORDER, other_class


def test_custom_cfx_optimizer_changes_only_masked_features(rf_model, X_train, X_test, pbounds, features_order,
                                                            query_instance):
    feature_masked = ["sepal width (cm)"]
    optimizer = RandomSearchCounterOptimizer(rf_model, X_test, pbounds)
    explainer = AcfxCustom(rf_model).fit(X=X_train, adjacency_matrix=IRIS_ADJACENCY,
                                         causal_order=IRIS_CAUSAL_ORDER, pbounds=pbounds,
                                         features_order=features_order, optimizer=optimizer,
                                         masked_features=feature_masked)

    cf = explainer.counterfactual(desired_class=other_class(rf_model, query_instance),
                                  query_instance=query_instance, **FAST)

    assert cf.shape == (1, len(features_order))
    for i, feature in enumerate(features_order):
        if feature not in feature_masked:
            assert cf[0, i] == query_instance[i]
    assert compute_causal_penalty(cf, IRIS_ADJACENCY, IRIS_CAUSAL_ORDER) > 0
