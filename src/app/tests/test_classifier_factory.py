"""Classifier selection logic: feature selection, available models and widget-to-parameter mapping (page 2)."""
import warnings

import pandas as pd
import pytest
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from utils.classifier_factory import (CLASSIFIER_LIST, available_classifiers, build_classifier, build_ebm_params,
                                      build_knn_params, build_logistic_regression_params,
                                      build_random_forest_params, parse_interactions, select_features)
from utils.data_loading import load_builtin_dataset


def test_select_features_keeps_available_columns_in_order():
    X = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    feature_types = pd.DataFrame([
        {"Column Name": "c", "Type": "nominal", "is_on": True},
        {"Column Name": "a", "Type": "continuous", "is_on": True},
        {"Column Name": "b", "Type": "continuous", "is_on": False},
        {"Column Name": "label", "Type": "nominal", "is_on": True},  # not a feature of X
    ])
    selected_X, selected_types = select_features(feature_types, X)
    assert selected_X.columns.tolist() == ["c", "a"]
    assert selected_types == ["nominal", "continuous"]


def test_only_ebm_is_available_for_categorical_features():
    assert available_classifiers(0) == CLASSIFIER_LIST
    assert available_classifiers(2) == ["ExplainableBoostingClassifier (glassbox)"]


@pytest.mark.parametrize("name, expected_cls", list(zip(CLASSIFIER_LIST, [
    LogisticRegression, RandomForestClassifier, KNeighborsClassifier, ExplainableBoostingClassifier])))
def test_build_classifier_maps_name_to_model(name, expected_cls):
    assert isinstance(build_classifier(name, {}, ["continuous"]), expected_cls)


def test_ebm_receives_feature_types():
    model = build_classifier(CLASSIFIER_LIST[3], {}, ["continuous", "nominal"])
    assert model.feature_types == ["continuous", "nominal"]


def test_unknown_classifier_is_rejected():
    with pytest.raises(IndexError):
        build_classifier("SVC", {}, [])


def test_logistic_regression_params():
    params = build_logistic_regression_params(tol=1e-6, fit_intercept=False, max_iter=100)
    assert params == {'tol': 1e-6, 'fit_intercept': False, 'max_iter': 100}


def test_logistic_regression_converges_on_unscaled_data_with_higher_max_iter():
    """Data is never scaled in the app (counterfactuals must stay in original units); max_iter is the remedy."""
    data = load_builtin_dataset('wine')
    default = build_classifier(CLASSIFIER_LIST[0], build_logistic_regression_params(1e-6, False, 100), [])
    with pytest.warns(ConvergenceWarning):
        default.fit(data.data, data.target)

    increased = build_classifier(CLASSIFIER_LIST[0], build_logistic_regression_params(1e-6, False, 5000), [])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        increased.fit(data.data, data.target)


def test_random_forest_widget_defaults_map_to_sklearn_defaults():
    params = build_random_forest_params(n_estimators=100, criterion='gini', max_depth=0, min_samples_split=2.0,
                                        min_samples_leaf=1.0, min_weight_fraction_leaf=0.0, max_features='None',
                                        max_leaf_nodes=0, min_impurity_decrease=0.0)
    assert params['max_depth'] is None
    assert params['max_leaf_nodes'] is None
    assert params['max_features'] is None
    assert params['min_samples_split'] == 2 and isinstance(params['min_samples_split'], int)
    assert params['min_samples_leaf'] == 1 and isinstance(params['min_samples_leaf'], int)
    RandomForestClassifier(**params)  # accepted by sklearn


def test_random_forest_fractions_below_one_stay_float():
    params = build_random_forest_params(n_estimators=10, criterion='entropy', max_depth=3, min_samples_split=0.5,
                                        min_samples_leaf=0.25, min_weight_fraction_leaf=0.0, max_features='sqrt',
                                        max_leaf_nodes=8, min_impurity_decrease=0.0)
    assert params['min_samples_split'] == 0.5 and params['min_samples_leaf'] == 0.25
    assert params['max_depth'] == 3 and params['max_leaf_nodes'] == 8 and params['max_features'] == 'sqrt'


@pytest.mark.parametrize("weights, expected", [('None', None), ('distance', 'distance')])
def test_knn_params(weights, expected):
    params = build_knn_params(n_neighbors=3, weights=weights, algorithm='auto', leaf_size=30, p=2)
    assert params == {'n_neighbors': 3, 'weights': expected, 'algorithm': 'auto', 'leaf_size': 30, 'p': 2}


@pytest.mark.parametrize("text, expected", [
    (None, 0),
    ("5", 5),
    ("0.5", 0.5),
    ("2.5x", "2.5x"),   # regression: '<float>x' (multiple of the number of features) used to be rejected
    ("3x", "3x"),
    ("abcx", 0),
    ("abc", 0),
])
def test_parse_interactions(text, expected):
    assert parse_interactions(text) == expected


def test_ebm_params_zero_random_state_means_non_repeatable():
    params = build_ebm_params(random_state=0, max_bins=256, max_interaction_bins=32, interactions="2.5x",
                              validation_size=0.15, outer_bags=1, inner_bags=0, learning_rate=0.015)
    assert params['random_state'] is None
    assert params['interactions'] == "2.5x"
    ExplainableBoostingClassifier(**params)  # accepted by interpret
