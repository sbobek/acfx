"""Data selection logic: built-in datasets and automatic detection of feature types (page 1)."""
import numpy as np
import pandas as pd
import pytest

from utils.data_loading import (BUILTIN_DATASETS, GERMAN_CREDIT_TYPES, builtin_feature_types, infer_feature_types,
                                is_categorical_like, load_builtin_dataset, load_german_credit_data,
                                split_features_and_label)


@pytest.mark.parametrize("series, expected", [
    (pd.Series([0.1, 2.3, 4.5, 6.7] * 20), False),           # float measurements
    (pd.Series([1, 2, 3, 1] * 20), True),                    # few integer codes
    (pd.Series([1.0, 2.0, 1.0, 2.0] * 20), True),            # integer codes stored as float
    (pd.Series(["a", "b"] * 40), True),                      # strings
    (pd.Series(pd.Categorical(["x", "y"])), True),           # pandas categorical
    (pd.Series(range(1000)), False),                         # many distinct integers (an id or a count)
])
def test_is_categorical_like(series, expected):
    assert is_categorical_like(series) == expected


def test_infer_feature_types_marks_only_continuous_features_as_available():
    df = pd.DataFrame({"height": np.linspace(150, 200, 100), "group": [1, 2] * 50})
    types = infer_feature_types(df, df.columns)
    assert types.to_dict("records") == [
        {"Column Name": "height", "Type": "continuous", "is_on": True},
        {"Column Name": "group", "Type": "nominal", "is_on": False},
    ]


def test_infer_feature_types_prefers_predefined_types():
    df = pd.DataFrame({"height": np.linspace(150, 200, 100), "group": [1, 2] * 50})
    types = infer_feature_types(df, df.columns, {"group": "ordinal", "height": "continuous"})
    assert types["Type"].tolist() == ["continuous", "ordinal"]
    assert types["is_on"].tolist() == [True, False]


def test_infer_feature_types_rejects_non_numerical_features():
    df = pd.DataFrame({"name": ["a", "b"]})
    with pytest.raises(ValueError, match="non-numerical feature found: name"):
        infer_feature_types(df, df.columns)


@pytest.mark.parametrize("name", BUILTIN_DATASETS)
def test_every_builtin_dataset_loads_with_numeric_features(name):
    data = load_builtin_dataset(name)
    types = builtin_feature_types(name, data)
    assert len(data.data) == len(data.target)
    assert types["Column Name"].tolist() == list(data.feature_names)


def test_unknown_builtin_dataset_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        load_builtin_dataset("mnist")


def test_german_credit_uses_documented_feature_types():
    data = load_german_credit_data()
    types = builtin_feature_types('german credit', data).set_index("Column Name")["Type"]
    assert "Credit granted" not in data.feature_names
    assert data.target_names.tolist() == ['Not Granted', 'Granted']
    for column in data.feature_names:
        assert types[column] == GERMAN_CREDIT_TYPES[column]


def test_split_features_and_label():
    df = pd.DataFrame({"a": [1, 2], "label": [0, 1]})
    X, y = split_features_and_label(df, "label")
    assert X.columns.tolist() == ["a"]
    assert y.tolist() == [0, 1]
