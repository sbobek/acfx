"""Session-state helpers shared by all pages: widget value persistence and feature lookup by type."""
import pandas as pd
import pytest

from utils.features_by_type import (get_all_columns, get_categorical_indicator, get_continuous_cols,
                                    get_nominal_cols, get_ordinal_cols)
from utils.key_helper import get_masked_feature_key, get_pbounds_key
from utils.session_state import load_value, store_value


def test_load_value_initialises_value_and_its_widget_key(session_state):
    load_value("n_iter", 100)
    assert session_state == {"n_iter": 100, "_n_iter": 100}


def test_load_value_restores_widget_from_persisted_value(session_state):
    """Widget keys are dropped by Streamlit when leaving a page; the persisted value brings them back."""
    session_state["n_iter"] = 50
    load_value("n_iter", 100)
    assert session_state["_n_iter"] == 50


def test_store_value_persists_widget_value(session_state):
    session_state["_n_iter"] = 7
    store_value("n_iter")
    assert session_state["n_iter"] == 7


def test_state_keys_for_feature_settings():
    assert get_pbounds_key("Age") == "Age_pbounds"
    assert get_masked_feature_key("Age") == "Age_is_masked"


@pytest.fixture
def typed_state(session_state):
    session_state.feature_types = pd.DataFrame({
        "Column Name": ["a", "b", "c", "d"],
        "Type": ["continuous", "nominal", "ordinal", "continuous"],
        "is_on": [True, True, True, False],
    })
    session_state.data = pd.DataFrame(columns=["a", "b", "c", "d"])
    session_state.selected_X = pd.DataFrame(columns=["a", "b", "c"])
    return session_state


def test_columns_by_type_are_limited_to_selected_features(typed_state):
    assert get_all_columns() == ["a", "b", "c"]
    assert get_continuous_cols() == ["a"]
    assert get_nominal_cols() == ["b"]
    assert get_ordinal_cols() == ["c"]


def test_categorical_indicator_follows_selected_columns(typed_state):
    assert get_categorical_indicator() == [False, True, True]


@pytest.mark.parametrize("missing", ["selected_X", "feature_types"])
def test_columns_by_type_require_selection(typed_state, missing):
    del typed_state[missing]
    with pytest.raises(Exception, match=missing):
        get_continuous_cols()
