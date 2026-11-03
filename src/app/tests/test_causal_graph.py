"""Causal graph logic of the adjacency generation page (page 3)."""
import numpy as np
import pandas as pd
import pytest

from utils.causal_graph import (default_adjacency_generator, find_order_violations, order_indices_to_names,
                                train_causal_model)
from utils.const import ADJACENCY_OPTION_BAYESIAN, ADJACENCY_OPTION_DIRECTLINGAM


@pytest.mark.parametrize("continuous, expected", [
    (["a", "b"], ADJACENCY_OPTION_DIRECTLINGAM),  # only continuous features
    (["a"], ADJACENCY_OPTION_BAYESIAN),           # mixed features
    ([], ADJACENCY_OPTION_BAYESIAN),              # only discrete features
])
def test_default_generator_is_lingam_only_for_purely_continuous_data(continuous, expected):
    """Regression: the default used to be inverted (Bayesian for continuous data, LiNGAM for discrete)."""
    assert default_adjacency_generator(["a", "b"], continuous) == expected


@pytest.fixture
def linear_sem_data():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=1000)
    y = 2.0 * x + rng.uniform(size=1000) * 0.1
    return pd.DataFrame({"cat": rng.integers(0, 3, 1000), "x": x, "y": y})


def test_lingam_graph_covers_all_features_with_zero_edges_for_categorical(linear_sem_data):
    _, adjacency, causal_order = train_causal_model(linear_sem_data, ["x", "y"])

    assert adjacency.shape == (3, 3)
    assert not adjacency[0, :].any() and not adjacency[:, 0].any()
    assert adjacency[2, 1] == pytest.approx(2.0, abs=0.1)  # y <- x
    assert causal_order == [0, 1, 2]  # categorical first, then the LiNGAM order x -> y


def test_order_indices_to_names():
    assert order_indices_to_names(pd.Index(["a", "b", "c"]), [2, 0, 1]) == ["c", "a", "b"]


def test_order_violations_lists_edges_from_child_placed_before_parent():
    # acfx convention: adjacency[child, parent] != 0, here b <- a
    adjacency = pd.DataFrame([[0, 0], [0.5, 0]], index=["a", "b"], columns=["a", "b"])
    assert find_order_violations(adjacency, ["a", "b"]) == []
    assert find_order_violations(adjacency, ["b", "a"]) == [("b", "a")]
