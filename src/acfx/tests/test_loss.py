"""Unit tests of the loss components minimised by the counterfactual search (acfx.evaluation.loss).

Expected values are computed by hand on tiny inputs, so each test documents the exact formula.
"""
import warnings
from unittest.mock import Mock

import numpy as np
import pytest
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from acfx.evaluation.loss import (compute_causal_penalty, compute_causal_penalty_bayesian_adjacency,
                                  compute_diversity_loss, compute_loss, compute_loss_bayesian,
                                  compute_loss_generic, compute_proximity_loss, compute_sparsity_loss,
                                  compute_yloss)
from acfx.tests.helpers import IRIS_ADJACENCY, IRIS_CAUSAL_ORDER


class FixedProbaModel:
    """Classifier stub returning the same class probabilities for every row."""

    def __init__(self, proba):
        self.proba = np.asarray(proba)
        self.classes_ = np.arange(len(proba))

    def predict_proba(self, X):
        return np.tile(self.proba, (len(X), 1))


def sem_consistent_sample(x0=1.0):
    x1 = 0.8 * x0
    x2 = 0.6 * x1
    x3 = 0.5 * x0 + 0.7 * x2
    return np.array([[x0, x1, x2, x3]])


# --- plausibility (linear SEM) ---------------------------------------------------------------

def test_causal_penalty_is_zero_for_sample_consistent_with_sem():
    penalty = compute_causal_penalty(sem_consistent_sample(), IRIS_ADJACENCY, IRIS_CAUSAL_ORDER)
    assert penalty == pytest.approx(0.0)


def test_causal_penalty_is_rmse_of_residuals_divided_by_number_of_features():
    sample = sem_consistent_sample()
    sample[0, 1] += 1.0  # x1 residual = 1, which also makes x2 residual = -0.6
    penalty = compute_causal_penalty(sample, IRIS_ADJACENCY, IRIS_CAUSAL_ORDER)
    assert penalty == pytest.approx(np.sqrt(1.0 + 0.36) / 4)


def test_causal_penalty_skips_categorical_features_by_default():
    sample = sem_consistent_sample()
    sample[0, 1] += 1.0
    categorical = [False, True, False, False]  # x1 is skipped, only the x2 residual (-0.6) remains
    penalty = compute_causal_penalty(sample, IRIS_ADJACENCY, IRIS_CAUSAL_ORDER, categorical=categorical)
    assert penalty == pytest.approx(np.sqrt(0.36) / 3)


def test_causal_penalty_counts_categorical_mismatch_when_not_skipped():
    sample = sem_consistent_sample(x0=2.0)  # x1 = 1.6 -> rounds to 2
    sample[0, 1] = 5.0
    categorical = [False, True, False, False]
    penalty = compute_causal_penalty(sample, IRIS_ADJACENCY, IRIS_CAUSAL_ORDER, categorical=categorical,
                                     skip_categorical=False)
    # x1: rounded mismatch -> 1, x2: (0.96 - 0.6 * 5)^2 = 2.04^2
    assert penalty == pytest.approx(np.sqrt(1 + 2.04 ** 2) / 4)


# --- plausibility (Bayesian network) -------------------------------------------------------

@pytest.fixture
def two_node_network():
    model = DiscreteBayesianNetwork([("A", "B")])
    model.add_cpds(
        TabularCPD("A", 2, [[0.5], [0.5]]),
        TabularCPD("B", 2, [[0.9, 0.2], [0.1, 0.8]], evidence=["A"], evidence_card=[2]),
    )
    return model


def test_bayesian_penalty_is_negative_log_likelihood_of_child_given_parents(two_node_network):
    samples = np.array([[0, 1]])  # P(B=1 | A=0) = 0.1, root nodes are not penalised
    assert compute_causal_penalty_bayesian_adjacency(samples, two_node_network) == pytest.approx(-np.log(0.1))


@pytest.fixture
def network_with_gapped_states():
    """B was observed only with codes 0, 2 and 4 (e.g. n_estimators in steps of 100)."""
    model = DiscreteBayesianNetwork([("A", "B")])
    model.add_cpds(
        TabularCPD("A", 2, [[0.5], [0.5]], state_names={"A": [0, 1]}),
        TabularCPD("B", 3, [[0.7, 0.1], [0.2, 0.1], [0.1, 0.8]], evidence=["A"], evidence_card=[2],
                   state_names={"B": [0, 2, 4], "A": [0, 1]}),
    )
    return model


def test_bayesian_penalty_looks_up_states_by_name(network_with_gapped_states):
    samples = np.array([[0, 2]])  # P(B=2 | A=0) = 0.2
    assert compute_causal_penalty_bayesian_adjacency(samples, network_with_gapped_states) == pytest.approx(-np.log(0.2))


@pytest.mark.parametrize("sample", [[0, 1], [0, 7], [5, 2]])
def test_bayesian_penalty_is_maximal_for_states_never_observed(network_with_gapped_states, sample):
    """Regression: unseen states used to be read as raw indices (wrong penalty for B=1, IndexError for B=7)."""
    with pytest.warns(UserWarning, match="never observed"):
        penalty = compute_causal_penalty_bayesian_adjacency(np.array([sample]), network_with_gapped_states)
    assert penalty == pytest.approx(-np.log(1e-8))


def test_bayesian_penalty_is_averaged_over_samples(two_node_network):
    samples = np.array([[0, 0], [1, 1]])
    expected = (-np.log(0.9) - np.log(0.8)) / 2
    assert compute_causal_penalty_bayesian_adjacency(samples, two_node_network) == pytest.approx(expected)


# --- validity -----------------------------------------------------------------------------

def test_yloss_is_gap_to_most_probable_other_class():
    yloss, maxvalue = compute_yloss(FixedProbaModel([0.2, 0.5, 0.3]), np.zeros((1, 2)), desired_class=2)
    assert yloss[0] == pytest.approx(0.2)
    assert maxvalue[0] == pytest.approx(0.5)


def test_yloss_is_zero_when_desired_class_is_most_probable():
    yloss, _ = compute_yloss(FixedProbaModel([0.2, 0.5, 0.3]), np.zeros((1, 2)), desired_class=1)
    assert yloss[0] == 0


# --- proximity / sparsity / diversity ------------------------------------------------------

def test_proximity_loss_is_zero_for_identical_instance():
    q = np.array([[1.0, 2.0]])
    loss = compute_proximity_loss(q.copy(), q, {"a": (0, 4), "b": (0, 10)}, features_order=["a", "b"])
    assert loss[0] == 0


def test_proximity_loss_normalises_continuous_difference_by_feature_range():
    loss = compute_proximity_loss(np.array([[1.0, 2.0]]), np.array([[0.0, 2.0]]),
                                  {"a": (0, 4), "b": (0, 10)}, features_order=["a", "b"])
    assert loss[0] == pytest.approx((1 / 4) / 2)


def test_proximity_loss_counts_any_categorical_change_as_one():
    loss = compute_proximity_loss(np.array([[3.0, 2.0]]), np.array([[1.0, 2.0]]),
                                  {"a": (0, 4), "b": (0, 10)}, features_order=["a", "b"],
                                  categorical=[True, False])
    assert loss[0] == pytest.approx(1 / 2)


def test_sparsity_loss_is_fraction_of_changed_features():
    loss = compute_sparsity_loss(np.array([[1.0, 2.0, 3.0, 4.0]]), np.array([[1.0, 0.0, 3.0, 0.0]]))
    assert loss[0] == pytest.approx(2 / 4)


def test_diversity_determinant_is_close_to_zero_for_identical_counterfactuals():
    cfs = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert compute_diversity_loss(cfs) == pytest.approx(0.0, abs=1e-4)


def test_diversity_determinant_is_close_to_one_for_distant_counterfactuals():
    cfs = np.array([[0.0, 0.0], [1000.0, 1000.0]])
    assert compute_diversity_loss(cfs) == pytest.approx(1.0, abs=1e-3)


# --- total loss ---------------------------------------------------------------------------

def test_total_loss_is_zero_for_valid_counterfactual_equal_to_query():
    q = np.array([[1.0, 2.0]])
    loss = compute_loss(FixedProbaModel([0.1, 0.9]), q.copy(), q, desired_class=1, adjency_matrix=None,
                        causal_order=None, proximity_weight=1, sparsity_weight=1, plausibility_weight=0,
                        diversity_weight=1, pbounds={"a": (0, 4), "b": (0, 10)}, features_order=["a", "b"],
                        masked_features=None)
    np.testing.assert_allclose(loss, [[0, 0.0]])


def test_total_loss_combines_components_with_conditional_term():
    model = FixedProbaModel([0.7, 0.3])  # yloss = 0.4 for desired class 1
    cfs, q = np.array([[1.0, 2.0]]), np.array([[0.0, 2.0]])
    loss = compute_loss(model, cfs, q, desired_class=1, adjency_matrix=None, causal_order=None,
                        proximity_weight=1, sparsity_weight=2, plausibility_weight=0, diversity_weight=0,
                        pbounds={"a": (0, 4), "b": (0, 10)}, features_order=["a", "b"], masked_features=None)
    proximity, sparsity = 0.125, 0.5
    conditional_term = 1 / (0.4 + 1 + 3)
    assert loss[0, 1] == pytest.approx(0.4 + conditional_term * (proximity + 2 * sparsity))


def test_plausibility_is_computed_only_when_its_weight_is_positive():
    plausibility_fn = Mock(return_value=10.0)
    args = dict(model=FixedProbaModel([0.1, 0.9]), cfs=np.array([[1.0]]), query_instance=np.array([[1.0]]),
                desired_class=1, proximity_weight=0, sparsity_weight=0, diversity_weight=0,
                pbounds={"a": (0, 1)}, features_order=["a"], plausibility_fn=plausibility_fn)

    assert compute_loss_generic(plausibility_weight=0, **args)[0, 1] == 0
    plausibility_fn.assert_not_called()

    assert compute_loss_generic(plausibility_weight=1, **args)[0, 1] == pytest.approx(10.0 / 2)
    plausibility_fn.assert_called_once()


def test_bayesian_total_loss_uses_categorised_counterfactual(two_node_network):
    cfs = np.array([[0.2, 0.9]])
    loss = compute_loss_bayesian(FixedProbaModel([0.1, 0.9]), cfs, np.array([[0, 1]]), cfs, desired_class=1,
                                 bayesian_model=two_node_network, proximity_weight=0, sparsity_weight=0,
                                 plausibility_weight=1, diversity_weight=0, pbounds={"A": (0, 1), "B": (0, 1)},
                                 features_order=["A", "B"], masked_features=None)
    assert loss[0, 1] == pytest.approx(-np.log(0.1) / 2)


def test_bayesian_penalty_does_not_warn_for_observed_states(network_with_gapped_states):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_causal_penalty_bayesian_adjacency(np.array([[0, 2]]), network_with_gapped_states)
