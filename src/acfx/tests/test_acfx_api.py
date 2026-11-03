"""Public API of the explainers (ACFX, AcfxLinear, AcfxEBM, AcfxCustom): fitting, validation and dispatch."""
import sys
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.base import clone

import acfx
from acfx import ACFX, AcfxCustom, AcfxEBM, AcfxLinear, RandomSearchCounterOptimizer
from acfx.abstract import OptimizerType
from acfx.tests.helpers import IRIS_ADJACENCY, IRIS_CAUSAL_ORDER

acfx_module = sys.modules["acfx.ACFX"]


def test_package_exports_public_api():
    assert set(acfx.__all__) == {ACFX, AcfxLinear, AcfxEBM, AcfxCustom, RandomSearchCounterOptimizer}


def test_base_explainer_is_abstract(lr_model):
    with pytest.raises(TypeError):
        ACFX(lr_model)


def test_new_explainer_is_not_fitted(lr_model):
    explainer = AcfxLinear(lr_model)
    assert explainer.blackbox is lr_model
    assert explainer.X is None and explainer.optimizer is None and explainer.optimizer_type is None


@pytest.mark.parametrize("explainer_cls, expected", [(AcfxLinear, OptimizerType.LinearAdditive),
                                                     (AcfxEBM, OptimizerType.EBM)])
def test_fit_selects_optimizer_matching_the_model_family(explainer_cls, expected, lr_model, X_train, pbounds):
    assert explainer_cls(lr_model).fit(X_train, pbounds).optimizer_type is expected


def test_custom_explainer_requires_optimizer(rf_model, X_train, pbounds):
    with pytest.raises(ValueError):
        AcfxCustom(rf_model).fit(X_train, pbounds)


def test_custom_explainer_stores_optimizer(rf_model, X_train, pbounds):
    optimizer = RandomSearchCounterOptimizer(rf_model, X_train, pbounds)
    explainer = AcfxCustom(rf_model).fit(X_train, pbounds, optimizer)
    assert explainer.optimizer is optimizer and explainer.optimizer_type is OptimizerType.Custom


def test_fit_stores_search_configuration(lr_model, X_train, pbounds, features_order):
    explainer = AcfxLinear(lr_model).fit(X_train, pbounds, IRIS_CAUSAL_ORDER, IRIS_ADJACENCY,
                                         masked_features=["petal width (cm)"], categorical_indicator=[False] * 4,
                                         features_order=features_order)
    assert explainer.X is X_train
    assert explainer.pbounds == pbounds
    assert explainer.causal_order == IRIS_CAUSAL_ORDER
    assert explainer.adjacency_matrix is IRIS_ADJACENCY
    assert explainer.masked_features == ["petal width (cm)"]
    assert explainer.features_order == features_order
    assert explainer.bayesian_model is None


def test_fit_with_targets_trains_the_blackbox(lr_model, X_train, y_train, X_test, pbounds):
    explainer = AcfxLinear(clone(lr_model)).fit(X_train, pbounds, y=y_train)
    np.testing.assert_array_equal(explainer.predict(X_test), lr_model.predict(X_test))


def test_bayesian_fit_trains_network_when_not_given(lr_model, X_train, pbounds):
    with patch.object(acfx_module, "train_bayesian_model", return_value="network") as train:
        explainer = AcfxLinear(lr_model).fit(X_train, pbounds, IRIS_CAUSAL_ORDER, IRIS_ADJACENCY,
                                             categorical_indicator=[False] * 4, bayesian_causality=True, num_bins=3)
    train.assert_called_once_with(X_train, [False] * 4, 3)
    assert explainer.bayesian_model == "network"
    assert explainer.adjacency_matrix is None and explainer.causal_order is None


def test_bayesian_fit_uses_given_network(lr_model, X_train, pbounds):
    with patch.object(acfx_module, "train_bayesian_model") as train:
        explainer = AcfxLinear(lr_model).fit(X_train, pbounds, bayesian_causality=True, bayesian_model="given",
                                             num_bins=3)
    train.assert_not_called()
    assert explainer.bayesian_model == "given"


# --- counterfactual() validation ----------------------------------------------------------

@pytest.mark.parametrize("causal_order, adjacency, message", [
    (None, IRIS_ADJACENCY, "Causal order"),
    (IRIS_CAUSAL_ORDER, None, "adjacency_matrix must be provided"),
    (IRIS_CAUSAL_ORDER, IRIS_ADJACENCY[:, :3], "same number of rows and columns"),
    ([0, 1, 2], IRIS_ADJACENCY, "same length as causal order"),
])
def test_plausibility_requires_consistent_causal_graph(lr_model, X_train, pbounds, query_instance,
                                                       causal_order, adjacency, message):
    explainer = AcfxLinear(lr_model).fit(X_train, pbounds, causal_order, adjacency)
    with pytest.raises(ValueError, match=message):
        explainer.counterfactual(query_instance, 1, plausibility_weight=0.5)


@pytest.mark.parametrize("network, bins, message", [(None, 3, "Bayesian model"), ("net", None, "num_bins")])
def test_bayesian_plausibility_requires_network_and_bins(lr_model, X_train, pbounds, query_instance,
                                                         network, bins, message):
    explainer = AcfxLinear(lr_model).fit(X_train, pbounds)
    explainer.bayesian_causality, explainer.bayesian_model, explainer.num_bins = True, network, bins
    with pytest.raises(ValueError, match=message):
        explainer.counterfactual(query_instance, 1, plausibility_weight=0.5)


def test_query_instance_is_required(lr_model, X_train, pbounds):
    with pytest.raises(ValueError, match="query_instance"):
        AcfxLinear(lr_model).fit(X_train, pbounds).counterfactual(None, 1)


def test_counterfactual_before_fit_is_rejected(lr_model, query_instance):
    with pytest.raises(ValueError, match="fit"):
        AcfxLinear(lr_model).counterfactual(query_instance, 1)


def test_custom_counterfactual_before_fit_is_rejected(rf_model, query_instance):
    with pytest.raises(ValueError, match="Optimizer"):
        AcfxCustom(rf_model).counterfactual(query_instance, 1)


def test_linear_explainer_requires_linear_coefficients(rf_model, X_train, pbounds, query_instance):
    with pytest.raises(AttributeError, match="coef_"):
        AcfxLinear(rf_model).fit(X_train, pbounds).counterfactual(query_instance, 1)


# --- counterfactual() dispatch ------------------------------------------------------------

def test_counterfactual_forwards_configuration_to_generator(lr_model, X_train, pbounds, features_order,
                                                            query_instance):
    explainer = AcfxLinear(lr_model).fit(X_train, pbounds, IRIS_CAUSAL_ORDER, IRIS_ADJACENCY,
                                         features_order=features_order, masked_features=["petal width (cm)"])
    with patch.object(acfx_module, "generate_cfs", return_value="cfs") as generate:
        result = explainer.counterfactual(query_instance, 2, num_counterfactuals=3, proximity_weight=0.1,
                                          sparsity_weight=0.2, plausibility_weight=0.3, diversity_weight=0.4,
                                          init_points=5, n_iter=6, sampling_from_model=False)
    assert result == "cfs"
    generate.assert_called_once_with(
        query_instance=query_instance, desired_class=2, adjacency_matrix=IRIS_ADJACENCY,
        causal_order=IRIS_CAUSAL_ORDER, proximity_weight=0.1, sparsity_weight=0.2, plausibility_weight=0.3,
        diversity_weight=0.4, bounds=pbounds, model=lr_model, features_order=features_order,
        masked_features=["petal width (cm)"], categorical_indicator=None, X=X_train, num_cfs=3, init_points=5,
        n_iter=6, sampling_from_model=False, optimizer_type=OptimizerType.LinearAdditive, optimizer=None)


def test_counterfactual_uses_bayesian_generator_when_network_is_fitted(lr_model, X_train, pbounds, query_instance):
    explainer = AcfxLinear(lr_model).fit(X_train, pbounds, bayesian_causality=True, bayesian_model="net",
                                         num_bins=4)
    with patch.object(acfx_module, "generate_cfs_bayesian", return_value="cfs") as generate, \
            patch.object(acfx_module, "generate_cfs") as plain:
        assert explainer.counterfactual(query_instance, 1, plausibility_weight=1) == "cfs"
    plain.assert_not_called()
    kwargs = generate.call_args.kwargs
    assert kwargs["bayesian_model"] == "net" and kwargs["categorical_bins_num"] == 4
