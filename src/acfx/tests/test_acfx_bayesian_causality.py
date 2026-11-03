"""Bayesian-network plausibility on a real mixed-type dataset (transformed_ready_for_acfx.csv)."""
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from acfx import AcfxLinear
from acfx.evaluation import compute_bin_edges, discretize_ndarray, train_bayesian_model
from acfx.evaluation.loss import compute_loss_bayesian
from acfx.tests.helpers import FAST, SEED, assert_within_bounds, other_class

pytestmark = pytest.mark.slow

target_column = 'cat_value'
FEATURES = ['params_alpha', 'params_booster', 'params_colsample_bytree', 'params_gamma', 'params_grow_policy',
            'params_lambda', 'params_learning_rate', 'params_max_delta_step', 'params_max_depth',
            'params_min_child_weight', 'params_n_estimators', 'params_normalize_type',
            'params_params_scale_pos_weight', 'params_rate_drop', 'params_sample_type', 'params_skip_drop',
            'params_subsample', 'user_attrs_decision_threshold', 'user_attrs_feature_fraction',
            'user_attrs_resampling_strategy', 'user_attrs_sample_rows', 'soft_risk_preference',
            'soft_decision_speed', 'soft_train_rows']
CATEGORICAL_INDICATOR = [False, True, False, False, True, False, False, True,
                         True, False, True, True, False, False,
                         True, False, False, False, False, True,
                         False, True, True, False]
NUM_BINS = 5


@pytest.fixture(scope="module")
def sample_data():
    csv = pd.read_csv(os.path.join(os.path.dirname(__file__), 'transformed_ready_for_acfx.csv'))
    csv = csv[FEATURES + [target_column]].dropna().sample(n=600, random_state=SEED)
    train, test = train_test_split(csv, test_size=0.5, random_state=SEED)
    return train[FEATURES], train[target_column], test[FEATURES]


@pytest.fixture(scope="module")
def linear_model(sample_data):
    X_train, y_train, _ = sample_data
    return LogisticRegression(max_iter=1000).fit(X_train, y_train)


@pytest.fixture(scope="module")
def pbounds(sample_data):
    X_train, _, _ = sample_data
    return {col: (X_train[col].min(), X_train[col].max()) for col in X_train.columns}


def test_bayesian_mode_works_with_valid_params(sample_data, linear_model, pbounds):
    _, _, X_test = sample_data
    acfx = AcfxLinear(linear_model)
    acfx.fit(X=X_test, pbounds=pbounds, categorical_indicator=CATEGORICAL_INDICATOR, features_order=FEATURES,
             bayesian_causality=True, num_bins=NUM_BINS)

    query_instance = X_test.iloc[0].values
    desired_class = other_class(linear_model, query_instance)

    result = acfx.counterfactual(desired_class=desired_class, query_instance=query_instance,
                                 plausibility_weight=1, **FAST)
    assert result.shape == (1, len(FEATURES))
    assert_within_bounds(result, pbounds, FEATURES)
    for i, is_categorical in enumerate(CATEGORICAL_INDICATOR):
        if is_categorical:
            assert result[0, i] == np.round(result[0, i])

    edges = compute_bin_edges(X_test[FEATURES].to_numpy(), CATEGORICAL_INDICATOR, NUM_BINS)
    loss = compute_loss_bayesian(model=linear_model, cfs=result,
                                 cfs_categorized=discretize_ndarray(result, CATEGORICAL_INDICATOR, edges),
                                 query_instance=query_instance, desired_class=desired_class,
                                 bayesian_model=acfx.bayesian_model, proximity_weight=1, sparsity_weight=1,
                                 plausibility_weight=1, diversity_weight=1, pbounds=pbounds,
                                 features_order=FEATURES, masked_features=None,
                                 categorical=CATEGORICAL_INDICATOR, allcfs=[])
    assert loss[0, 1] > 0


def test_bayesian_mode_fits_same_network_as_train_bayesian_model(sample_data, linear_model, pbounds):
    _, _, X_test = sample_data
    acfx = AcfxLinear(linear_model)
    acfx.fit(X=X_test, pbounds=pbounds, categorical_indicator=CATEGORICAL_INDICATOR, bayesian_causality=True,
             num_bins=NUM_BINS)
    expected_bayesian_model = train_bayesian_model(X_test, CATEGORICAL_INDICATOR, NUM_BINS)

    assert set(acfx.bayesian_model.edges()) == set(expected_bayesian_model.edges())
    for node in acfx.bayesian_model.nodes():
        cpd = acfx.bayesian_model.get_cpds(node)
        expected_cpd = expected_bayesian_model.get_cpds(node)
        assert (cpd is None) == (expected_cpd is None), f"Mismatch for node {node}"
        if cpd is not None:
            np.testing.assert_array_equal(cpd.values, expected_cpd.values)
