"""Counterfactual generation engine (acfx.evaluation.ccfs): input validation, search space and output contract."""
import numpy as np
import pytest

from acfx.abstract import ModelBasedCounterOptimizer, OptimizerType
from acfx.evaluation import generate_cfs, generate_cfs_bayesian, train_bayesian_model
from acfx.evaluation.ccfs import _generate_single_cf
from acfx.tests.helpers import FAST, IRIS_ADJACENCY, IRIS_CAUSAL_ORDER, assert_within_bounds, other_class


@pytest.fixture
def linear_kwargs(lr_model, X_train, pbounds, features_order, query_instance):
    return dict(query_instance=query_instance, desired_class=other_class(lr_model, query_instance),
                adjacency_matrix=IRIS_ADJACENCY, causal_order=IRIS_CAUSAL_ORDER, proximity_weight=1,
                sparsity_weight=1, plausibility_weight=1, diversity_weight=1, bounds=pbounds, model=lr_model,
                features_order=features_order, X=X_train, optimizer_type=OptimizerType.LinearAdditive, **FAST)


def test_unknown_optimizer_type_is_not_supported(linear_kwargs):
    linear_kwargs["optimizer_type"] = "unknown"
    with pytest.raises(NotImplementedError):
        generate_cfs(**linear_kwargs)


def test_training_data_is_required_to_seed_the_search(linear_kwargs):
    linear_kwargs["X"] = None
    with pytest.raises(ValueError, match="X is None"):
        generate_cfs(**linear_kwargs)


@pytest.mark.parametrize("missing", ["bayesian_model", "bins_num"])
def test_bayesian_mode_requires_network_and_number_of_bins(lr_model, pbounds, query_instance, missing):
    kwargs = dict(bayesian_model=object(), bins_num=3)
    kwargs[missing] = None
    with pytest.raises(ValueError, match=missing):
        _generate_single_cf(query_instance, 0, None, None, 1, 1, 1, 1, pbounds, lr_model,
                            bayesian_causality=True, **kwargs)


def test_generates_requested_number_of_counterfactuals_within_bounds(linear_kwargs, pbounds, features_order):
    cfs = generate_cfs(num_cfs=2, **linear_kwargs)
    assert cfs.shape == (2, len(features_order))
    assert_within_bounds(cfs, pbounds, features_order)


def test_counterfactual_reaches_desired_class(linear_kwargs, lr_model):
    cfs = generate_cfs(**linear_kwargs)
    assert lr_model.predict(cfs)[0] == linear_kwargs["desired_class"]


def test_only_masked_features_are_changed(linear_kwargs, features_order, query_instance):
    masked = ["petal length (cm)", "petal width (cm)"]
    cfs = generate_cfs(masked_features=masked, **linear_kwargs)
    for i, feature in enumerate(features_order):
        if feature not in masked:
            assert cfs[0, i] == query_instance[i]


def test_categorical_features_take_integer_values(linear_kwargs, pbounds, X_train):
    """Categorical features are searched over the integer codes within their bounds."""
    categorical = [False, False, True, False]
    linear_kwargs["X"] = X_train.assign(**{"petal length (cm)": X_train["petal length (cm)"].round()})
    linear_kwargs["bounds"] = {**pbounds, "petal length (cm)": (1.0, 7.0)}
    cfs = generate_cfs(categorical_indicator=categorical, num_cfs=2, **linear_kwargs)
    assert np.all(cfs[:, 2] == np.round(cfs[:, 2]))


def test_custom_optimizer_candidates_are_used_and_its_failures_do_not_break_the_search(
        rf_model, X_train, pbounds, features_order, query_instance):
    class CountingOptimizer(ModelBasedCounterOptimizer):
        calls = 0

        def optimize_proba(self, target_class, feature_masked):
            CountingOptimizer.calls += 1
            if CountingOptimizer.calls == 1:
                raise RuntimeError("optimizer failure is logged and skipped")
            return {f: X_train[f].mean() for f in features_order}

    cfs = generate_cfs(query_instance, other_class(rf_model, query_instance), IRIS_ADJACENCY, IRIS_CAUSAL_ORDER,
                       1, 1, 0, 1, pbounds, rf_model, features_order=features_order, X=X_train,
                       sampling_from_model=True, optimizer_type=OptimizerType.Custom,
                       optimizer=CountingOptimizer(), init_points=20, n_iter=5)

    assert CountingOptimizer.calls > 1
    assert_within_bounds(cfs, pbounds, features_order)


def test_bayesian_generation_returns_counterfactuals_within_bounds(lr_model, X_train, pbounds, features_order,
                                                                    query_instance):
    categorical = [False] * len(features_order)
    network = train_bayesian_model(X_train, categorical, bins=3)

    cfs = generate_cfs_bayesian(query_instance, other_class(lr_model, query_instance), network, 3, 1, 1, 1, 1,
                                pbounds, lr_model, features_order=features_order, categorical_indicator=categorical,
                                X=X_train, num_cfs=2, optimizer_type=OptimizerType.LinearAdditive, **FAST)

    assert cfs.shape == (2, len(features_order))
    assert_within_bounds(cfs, pbounds, features_order)
