"""Shared fixtures for the Streamlit app tests.

Pages are driven with streamlit.testing.v1.AppTest. Each page expects the session state produced by the previous
pages, so `iris_state` builds that state with the same helper functions the pages use.
"""
import random

import numpy as np
import pytest
import streamlit
import streamlit_sortables

from utils.classifier_factory import CLASSIFIER_LIST, build_classifier, select_features
from utils.data_loading import builtin_feature_types, load_builtin_dataset
from utils.key_helper import calc_pbounds, get_pbounds_key
from app.tests.helpers import FakeSessionState


@pytest.fixture
def session_state(monkeypatch):
    state = FakeSessionState()
    monkeypatch.setattr(streamlit, "session_state", state)
    return state


@pytest.fixture(autouse=True)
def isolated_run(monkeypatch, tmp_path):
    """Seeds randomness, keeps files written by pages (temp_graph.html) out of the repo
    and replaces the drag & drop component, which cannot be rendered by AppTest, with the identity."""
    np.random.seed(0)
    random.seed(0)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(streamlit_sortables, "sort_items", lambda items, **kwargs: list(items))


@pytest.fixture
def iris_state():
    """Session state after selecting iris (page 1) and fitting LogisticRegression on all features (page 2)."""
    data = load_builtin_dataset('iris')
    feature_types = builtin_feature_types('iris', data)
    selected_X, selected_types = select_features(feature_types, data.data)
    classifier_name = CLASSIFIER_LIST[0]
    classifier = build_classifier(classifier_name, {'max_iter': 500}, selected_types).fit(selected_X, data.target)
    return {
        'X': data.data, 'y': data.target, 'data': data, 'data_source_name': 'iris', 'data_loaded': True,
        'feature_types': feature_types, 'label_column': None,
        'selected_X': selected_X, 'selected_feature_types': selected_types,
        'classifier_name': classifier_name, 'classifier_instance': classifier, 'classifier_params': {},
    }


@pytest.fixture
def evaluation_state(iris_state):
    """Session state after pages 3 (plausibility off) and 4 (default bounds and weights)."""
    X = iris_state['selected_X']
    pbounds = calc_pbounds(X, [])
    state = dict(iris_state, plausibility_loss_on=False, plausibility_loss=0, pbounds=pbounds,
                 proximity_weight=0.5, diversity_weight=0.5, sparsity_weight=0.5)
    state.update({get_pbounds_key(f): bounds for f, bounds in pbounds.items()})
    return state
