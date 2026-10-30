"""Model-based counter-optimizers that seed the counterfactual search with promising candidates."""
import numpy as np
import pandas as pd
import pytest
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.svm import SVC

from acfx import RandomSearchCounterOptimizer
from acfx.abstract import ModelBasedCounterOptimizer
from acfx.evaluation import EBMCounterOptimizer, LogisticRegressionCounterOptimizer


def test_counter_optimizer_cannot_be_instantiated_without_optimize_proba():
    with pytest.raises(TypeError):
        ModelBasedCounterOptimizer()


# --- LogisticRegressionCounterOptimizer ----------------------------------------------------

def test_linear_optimizer_requires_fitted_linear_model(X_train, pbounds):
    with pytest.raises(AttributeError):
        LogisticRegressionCounterOptimizer(object(), X_train, pbounds)


def test_linear_optimizer_requires_bounds(lr_model, X_train):
    with pytest.raises(ValueError):
        LogisticRegressionCounterOptimizer(lr_model, X_train, None)


@pytest.mark.parametrize("bad_bounds", [{"a": [0, 1]}, {"a": (0, 1, 2)}, {"a": ("0", "1")}, {1: (0, 1)}])
def test_linear_optimizer_rejects_malformed_bounds(lr_model, X_train, bad_bounds):
    with pytest.raises(AttributeError):
        LogisticRegressionCounterOptimizer(lr_model, X_train, bad_bounds)


def test_linear_optimizer_moves_masked_features_to_bound_favouring_target_class(lr_model, X_test, pbounds):
    target_class = 0
    optimizer = LogisticRegressionCounterOptimizer(lr_model, X_test, pbounds)
    masked = list(X_test.columns[:2])

    result = optimizer.optimize_proba(target_class, masked)

    for idx, feature in enumerate(X_test.columns):
        if feature in masked:
            low, high = pbounds[feature]
            assert result[feature] == pytest.approx(high if lr_model.coef_[target_class][idx] > 0 else low)
        else:
            assert result[feature] == pytest.approx(X_test[feature].mean())


# --- RandomSearchCounterOptimizer ----------------------------------------------------------

def test_random_search_requires_predict_proba(X_train, pbounds):
    with pytest.raises(AttributeError):
        RandomSearchCounterOptimizer(SVC(probability=False), X_train, pbounds)


def test_random_search_changes_only_masked_features_within_bounds(rf_model, X_test, pbounds):
    masked = ["petal length (cm)"]
    result = RandomSearchCounterOptimizer(rf_model, X_test, pbounds, n_iter=30).optimize_proba(2, masked)

    low, high = pbounds[masked[0]]
    assert low <= result[masked[0]] <= high
    for feature in X_test.columns.drop(masked):
        assert result[feature] == pytest.approx(X_test[feature].mean())


def test_random_search_never_decreases_target_probability(rf_model, X_test, pbounds):
    masked = list(X_test.columns)
    result = RandomSearchCounterOptimizer(rf_model, X_test, pbounds, n_iter=30).optimize_proba(2, masked)

    base = rf_model.predict_proba(pd.DataFrame([X_test.mean()]))[0][2]
    found = rf_model.predict_proba(pd.DataFrame([result]))[0][2]
    assert found >= base


# --- EBMCounterOptimizer ------------------------------------------------------------------

def test_ebm_optimizer_rejects_unknown_class(ebm_model, X_test):
    with pytest.raises(KeyError):
        EBMCounterOptimizer(ebm_model, X_test.iloc[[0]]).optimize_proba(99, list(X_test.columns))


def test_ebm_optimizer_returns_value_for_every_feature(ebm_model, X_test):
    query = X_test.iloc[[0]]
    masked = ["petal length (cm)", "petal width (cm)"]

    result = EBMCounterOptimizer(ebm_model, query).optimize_proba(2, masked)

    assert set(result) == set(X_test.columns)
    for feature in X_test.columns.drop(masked):
        assert result[feature] == query[feature].iloc[0]
    for feature in masked:
        low, high = ebm_model.feature_bounds_[list(X_test.columns).index(feature)]
        assert low <= result[feature] <= high


@pytest.fixture(scope="module")
def binary_ebm_with_categorical():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"income": rng.uniform(0, 100, 400), "segment": rng.integers(0, 3, 400)})
    y = ((X["income"] > 50) & (X["segment"] == 2)).astype(int)
    model = ExplainableBoostingClassifier(feature_types=["continuous", "nominal"], interactions=0, outer_bags=2,
                                          random_state=0).fit(X, y)
    return model, X


def test_ebm_optimizer_supports_binary_classification_with_categorical_features(binary_ebm_with_categorical):
    model, X = binary_ebm_with_categorical
    query = pd.DataFrame({"income": [10.0], "segment": [0]})

    result = EBMCounterOptimizer(model, query).optimize_proba(1, ["income", "segment"])

    assert result["segment"] in {0, 1, 2, "0", "1", "2"}
    assert model.predict_proba(pd.DataFrame([result]).astype({"segment": int}))[0][1] > \
        model.predict_proba(query)[0][1]


def test_ebm_optimizer_moves_masked_features_towards_target_class(ebm_model, X_test):
    query = X_test.iloc[[0]]
    target = int(next(c for c in ebm_model.classes_ if c != ebm_model.predict(query)[0]))

    result = EBMCounterOptimizer(ebm_model, query).optimize_proba(target, list(X_test.columns))

    before = ebm_model.predict_proba(query)[0][target]
    after = ebm_model.predict_proba(pd.DataFrame([result])[X_test.columns])[0][target]
    assert after > before
