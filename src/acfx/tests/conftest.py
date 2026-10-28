"""Shared fixtures for the acfx test-suite.

The counterfactual search is stochastic (random sampling + Optuna TPE), so every test runs with fixed seeds,
which makes failures reproducible and lets the tests act as regression checks.
"""
import random

import numpy as np
import optuna
import pytest
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from acfx.tests.helpers import SEED


@pytest.fixture(autouse=True)
def seeded(monkeypatch):
    np.random.seed(SEED)
    random.seed(SEED)
    original_create_study = optuna.create_study

    def create_seeded_study(*args, **kwargs):
        kwargs.setdefault("sampler", optuna.samplers.TPESampler(seed=SEED))
        return original_create_study(*args, **kwargs)

    monkeypatch.setattr(optuna, "create_study", create_seeded_study)


@pytest.fixture(scope="session")
def iris_split():
    data = load_iris(as_frame=True)
    return train_test_split(data.data, data.target, test_size=0.2, random_state=SEED)


@pytest.fixture(scope="session")
def X_train(iris_split):
    return iris_split[0]


@pytest.fixture(scope="session")
def X_test(iris_split):
    return iris_split[1]


@pytest.fixture(scope="session")
def y_train(iris_split):
    return iris_split[2]


@pytest.fixture(scope="session")
def pbounds(X_train):
    return {col: (float(X_train[col].min()), float(X_train[col].max())) for col in X_train.columns}


@pytest.fixture(scope="session")
def features_order(X_train):
    return X_train.columns.tolist()


@pytest.fixture(scope="session")
def query_instance(X_test):
    return X_test.iloc[0].values


@pytest.fixture(scope="session")
def lr_model(X_train, y_train):
    return LogisticRegression(max_iter=500).fit(X_train, y_train)


@pytest.fixture(scope="session")
def rf_model(X_train, y_train):
    return RandomForestClassifier(n_estimators=20, random_state=SEED).fit(X_train, y_train)


@pytest.fixture(scope="session")
def ebm_model(X_train, y_train):
    return ExplainableBoostingClassifier(interactions=0, outer_bags=2, random_state=SEED).fit(X_train, y_train)
