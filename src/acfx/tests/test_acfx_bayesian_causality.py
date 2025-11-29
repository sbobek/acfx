import os
from unittest.mock import patch

import numpy as np
import pytest
import pandas as pd
from src.acfx.evaluation.bayesian_model import discretize_ndarray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.acfx.evaluation.loss import compute_loss_bayesian
from src.acfx import AcfxLinear
from src.acfx.evaluation import discretize_dataframe, train_bayesian_model

target_column = 'cat_value'

@pytest.fixture
def sample_data():
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, 'transformed_ready_for_acfx.csv')
    csv = pd.read_csv(file_path)
    # simplified dataset for test
    # csv = csv[['params_alpha', 'params_booster', 'params_colsample_bytree', 'params_gamma', target_column]]
    csv = csv[['params_alpha', 'params_booster',
       'params_colsample_bytree', 'params_gamma', 'params_grow_policy',
       'params_lambda', 'params_learning_rate', 'params_max_delta_step',
       'params_max_depth', 'params_min_child_weight', 'params_n_estimators',
       'params_normalize_type', 'params_params_scale_pos_weight',
       'params_rate_drop', 'params_sample_type', 'params_skip_drop',
       'params_subsample', 'user_attrs_decision_threshold',
       'user_attrs_feature_fraction', 'user_attrs_resampling_strategy',
       'user_attrs_sample_rows', 'soft_risk_preference',
       'soft_decision_speed', 'soft_train_rows', 'cat_value']]
    csv = csv.dropna()
    categorical_indicator = [False,True, False, False, True, False, False,True,
                             True, False, True, True, False, False,
                             True, False, False, False, False, True,
                             False, True, True, False]

    train,test = train_test_split(csv, test_size=0.2)
    return train.drop(target_column, axis=1), train[target_column], test.drop(target_column, axis=1), test[target_column], categorical_indicator

def test_bayesianmode_workswithvalidparams(sample_data):
    X_train,y_train,X_test,y_test, categorical_indicator = sample_data

    linear_model = LogisticRegression()
    linear_model.fit(X_train, y_train)
    acfx = AcfxLinear(linear_model)
    pbounds = {col: (X_train[col].min(), X_train[col].max()) for col in X_train.columns}
    features_order = list(X_test.columns)
    acfx.fit(X=X_test, pbounds=pbounds, causal_order=None, adjacency_matrix=None, y=None,
             masked_features=None, categorical_indicator=categorical_indicator,
             features_order=features_order, bayesian_causality=True,num_bins=5)

    query_instance = X_test.iloc[0].values
    desired_class = linear_model.predict([query_instance])[0]

    result = acfx.counterfactual(desired_class=desired_class, query_instance=query_instance)
    assert result is not None
    assert query_instance.shape[0] == result.shape[1]

    bayesian_causality_loss = compute_loss_bayesian(model=linear_model, cfs=result,
                                                    cfs_categorized=discretize_ndarray(result, categorical_indicator),
                                                    query_instance=query_instance, desired_class=desired_class,
                                                    bayesian_model=acfx.bayesian_model, proximity_weight=1,
                                                    sparsity_weight=1, plausibility_weight=1,
                                                    diversity_weight=1,pbounds=pbounds,
                                                    features_order=features_order, masked_features=None,
                                                    categorical=categorical_indicator, allcfs=[])

    bayesian_causality_loss = bayesian_causality_loss[0, 1]
    assert bayesian_causality_loss > 0


def test_bayesianmode_fitsbayesiannetwork(sample_data):
    X_train, y_train, X_test, y_test, categorical_indicator = sample_data
    linear_model = LogisticRegression()
    linear_model.fit(X_train, y_train)
    acfx = AcfxLinear(linear_model)
    pbounds = {col: (X_train[col].min(), X_train[col].max()) for col in X_train.columns}
    acfx.fit(X=X_test, pbounds=pbounds, causal_order=None, adjacency_matrix=None, y=None,
             masked_features=None, categorical_indicator=categorical_indicator,
             features_order=None, bayesian_causality=True,num_bins=5)
    expected_bayesian_model = train_bayesian_model(X_test, categorical_indicator, 5)
    acfx_bayesian_model = acfx.bayesian_model

    # Create a table of all features and their CPTs
    for node in acfx_bayesian_model.nodes():
        cpd = acfx_bayesian_model.get_cpds(node)
        expected_cpd = expected_bayesian_model.get_cpds(node)

        # Assert that both are None or both are not None
        assert (cpd is None and expected_cpd is None) or (cpd is not None and expected_cpd is not None), \
            f"Mismatch for node {node}: cpd={cpd}, expected_cpd={expected_cpd}"

        if cpd is not None:
            assert np.array_equal(cpd.values, expected_cpd.values), f"cpd={cpd}, expected_cpd={expected_cpd}"
