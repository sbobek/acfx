"""Discretisation and Bayesian-network training used by the Bayesian plausibility loss (acfx.evaluation.bayesian_model)."""
import numpy as np
import pandas as pd
import pytest
from pgmpy.models import DiscreteBayesianNetwork

from acfx.evaluation import discretize_dataframe, discretize_ndarray, train_bayesian_model


@pytest.fixture
def mixed_df():
    return pd.DataFrame({"cont": [0.0, 2.5, 5.0, 7.5, 10.0], "cat": [3, 1, 2, 1, 3]})


def test_discretize_dataframe_bins_continuous_columns_into_right_closed_equal_width_bins(mixed_df):
    result = discretize_dataframe(mixed_df, [False, True], bins=4)
    assert result["cont"].tolist() == [0, 0, 1, 2, 3]


def test_discretize_dataframe_leaves_categorical_columns_and_input_untouched(mixed_df):
    original = mixed_df.copy()
    result = discretize_dataframe(mixed_df, [False, True], bins=4)
    assert result["cat"].tolist() == mixed_df["cat"].tolist()
    pd.testing.assert_frame_equal(mixed_df, original)


def test_discretize_dataframe_rejects_flags_of_wrong_length(mixed_df):
    with pytest.raises(ValueError):
        discretize_dataframe(mixed_df, [False])


def test_discretize_ndarray_is_consistent_with_discretize_dataframe(mixed_df):
    """The network is trained on discretize_dataframe output and queried with discretize_ndarray output."""
    result = discretize_ndarray(mixed_df.values, [False, True], bins=4)
    expected = discretize_dataframe(mixed_df, [False, True], bins=4)
    assert result[:, 0].tolist() == expected["cont"].tolist()
    assert result[:, 1].tolist() == [3, 1, 2, 1, 3]


def test_discretize_ndarray_rejects_flags_of_wrong_length(mixed_df):
    with pytest.raises(ValueError):
        discretize_ndarray(mixed_df.values, [False, True, True])


def test_train_bayesian_model_learns_dependency_between_correlated_features():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 3, 400)
    df = pd.DataFrame({"a": a, "b": a.copy(), "noise": rng.integers(0, 2, 400)})

    model = train_bayesian_model(df, [True, True, True], bins=3)

    assert isinstance(model, DiscreteBayesianNetwork)
    assert set(model.nodes()) == set(df.columns)
    assert set(model.edges()) in ({("a", "b")}, {("b", "a")})
    # every connected node has a fitted CPD (pgmpy leaves isolated nodes without one; the loss skips them)
    assert {cpd.variable for cpd in model.get_cpds()} == {"a", "b"}
