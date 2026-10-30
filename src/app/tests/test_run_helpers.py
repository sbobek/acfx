"""Evaluation settings (page 4) and counterfactual run (page 5) logic."""
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import Bunch

from acfx import AcfxCustom, AcfxEBM, AcfxLinear, RandomSearchCounterOptimizer
from utils.key_helper import calc_pbounds
from utils.run_helpers import (acfx_for_classifier, categorical_feature_names, causal_order_to_indices,
                               class_probabilities_text, continuous_slider_range, counterfactuals_table,
                               fit_explainer, frozen_category_errors, incomplete_query_errors, masked_features,
                               numeric_desired_class,
                               pbounds_from_state,
                               prediction_label, query_template, query_warnings, target_class_list)


def test_calc_pbounds_uses_data_range_and_integer_bounds_for_categorical():
    data = pd.DataFrame({"x": [0.5, 2.5], "c": [1.0, 4.0]})
    assert calc_pbounds(data, ["c"]) == {"x": (0.5, 2.5), "c": (1, 4)}
    assert isinstance(calc_pbounds(data, ["c"])["c"][0], int)


def test_categorical_feature_names():
    feature_types = pd.DataFrame({"Column Name": ["a", "b", "c"], "Type": ["continuous", "nominal", "ordinal"]})
    assert categorical_feature_names(feature_types) == ["b", "c"]


def test_continuous_bounds_can_be_widened_three_ranges_each_side():
    assert continuous_slider_range(1.0, 3.0) == (-5.0, 9.0)


def test_pbounds_are_read_from_slider_state_as_floats():
    state = {"a_pbounds": (1, 2), "b_pbounds": (0.5, 1.5), "unrelated": 1}
    assert pbounds_from_state(state, ["a", "b"]) == {"a": (1.0, 2.0), "b": (0.5, 1.5)}


def test_masked_features_ignore_widget_mirror_keys():
    """Regression: '_<name>_is_masked' widget keys used to be reported as extra features '_<name>'."""
    state = {"Age_is_masked": True, "_Age_is_masked": True, "Job_is_masked": False, "_Job_is_masked": False,
             "Income_is_masked": True, "_Income_is_masked": True}
    assert masked_features(state) == ["Age", "Income"]


@pytest.mark.parametrize("model, expected", [
    (LogisticRegression(), AcfxLinear), (ExplainableBoostingClassifier(), AcfxEBM),
    (RandomForestClassifier(), AcfxCustom), (KNeighborsClassifier(), AcfxCustom)])
def test_explainer_is_chosen_by_classifier_family(model, expected):
    explainer = acfx_for_classifier(model)
    assert type(explainer) is expected and explainer.blackbox is model


def test_unsupported_classifier_is_rejected():
    with pytest.raises(ValueError):
        acfx_for_classifier(SVC())


def test_causal_order_given_by_names_is_converted_to_indices():
    assert causal_order_to_indices(["c", "a", "b"], ["a", "b", "c"]) == [2, 0, 1]
    assert causal_order_to_indices([2, 0, 1], ["a", "b", "c"]) == [2, 0, 1]


@pytest.fixture
def fit_args():
    X = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 2.0]})
    return dict(X=X, pbounds={"a": (0.0, 1.0), "b": (1.0, 2.0)}, features_order=["a", "b"], masked=["a"],
                categorical_indicator=[False, False])


def test_fit_without_causal_model(fit_args):
    explainer = Mock(spec=AcfxLinear)
    fit_explainer(explainer, **fit_args)
    explainer.fit.assert_called_once_with(X=fit_args["X"], pbounds=fit_args["pbounds"], masked_features=["a"],
                                          categorical_indicator=[False, False], features_order=["a", "b"],
                                          bayesian_causality=False, adjacency_matrix=None, causal_order=None)


def test_fit_with_lingam_graph(fit_args):
    explainer = Mock(spec=AcfxLinear)
    adjacency = np.zeros((2, 2))
    fit_explainer(explainer, adjacency_matrix=adjacency, causal_order=[0, 1], **fit_args)
    kwargs = explainer.fit.call_args.kwargs
    assert kwargs["adjacency_matrix"] is adjacency and kwargs["causal_order"] == [0, 1]
    assert kwargs["bayesian_causality"] is False


def test_fit_with_bayesian_network(fit_args):
    explainer = Mock(spec=AcfxLinear)
    fit_explainer(explainer, bayesian_model="network", num_bins=4, **fit_args)
    kwargs = explainer.fit.call_args.kwargs
    assert kwargs["bayesian_causality"] is True
    assert kwargs["bayesian_model"] == "network" and kwargs["num_bins"] == 4
    assert "adjacency_matrix" not in kwargs


def test_custom_explainer_gets_random_search_optimizer_over_same_bounds(fit_args):
    model = RandomForestClassifier(n_estimators=2).fit(fit_args["X"], [0, 1])
    explainer = fit_explainer(AcfxCustom(model), **fit_args)
    assert isinstance(explainer.optimizer, RandomSearchCounterOptimizer)
    assert explainer.optimizer.feature_bounds == fit_args["pbounds"] and explainer.optimizer.n_iter == 1000


def test_target_names_are_used_when_dataset_provides_them():
    assert target_class_list(Bunch(target_names=np.array(["no", "yes"])), [0, 1]).tolist() == ["no", "yes"]
    assert target_class_list(pd.DataFrame({"x": [1]}), [0, 1]) == [0, 1]


def test_prediction_label():
    assert prediction_label(1, [0, 1], np.array(["no", "yes"])) == "yes"
    assert prediction_label(7, [0, 1], np.array(["no", "yes"])) == "7"


def test_numeric_desired_class():
    assert numeric_desired_class("yes", np.array(["no", "yes"])) == 1
    assert numeric_desired_class(0, np.array(["no", "yes"])) == 0


def test_query_template_is_a_data_row_with_integer_categorical_columns():
    data = pd.DataFrame({"x": [2.5, 3.5], "c": [1.0, 4.0]}, index=[7, 8])
    template = query_template(data, [False, True])
    assert template.shape == (1, 2)
    assert template.index.tolist() == [0]
    assert template.dtypes.tolist() == [np.dtype(float), np.dtype(int)]
    assert template.iloc[0].tolist() == [2.5, 1]


def test_query_template_skips_rows_with_missing_values():
    data = pd.DataFrame({"x": [np.nan, 3.5], "c": [1.0, 4.0]})
    assert query_template(data, [False, True]).iloc[0].tolist() == [3.5, 4]


def test_query_template_handles_data_without_complete_rows():
    data = pd.DataFrame({"x": [np.nan, 3.5], "c": [1.0, np.nan]})
    assert query_template(data, [False, True]).iloc[0].tolist() == [0.0, 1]


def test_query_warnings_accepts_the_query_template():
    data = pd.DataFrame({"x": [0.5, 2.5], "c": [1, 4]})
    pbounds = {"x": (0.5, 2.5), "c": (1, 4)}
    assert query_warnings(query_template(data, [False, True]), data, pbounds, [False, True]) == []


def test_query_warnings_reports_unknown_category_and_value_outside_bounds():
    data = pd.DataFrame({"x": [0.5, 2.5], "c": [1, 4]})
    pbounds = {"x": (0.5, 2.5), "c": (1, 4)}
    query = pd.DataFrame({"x": [9.0, 1.0], "c": [1, 0]})

    result = query_warnings(query, data, pbounds, [False, True])

    assert len(result) == 2
    assert "Query 1: 'x'" in result[0] and "bounds" in result[0]
    assert "Query 2: 'c'" in result[1] and "does not occur" in result[1]


def test_query_warnings_rejects_category_between_observed_values():
    data = pd.DataFrame({"c": [1, 4]})
    assert len(query_warnings(pd.DataFrame({"c": [2]}), data, {"c": (1, 4)}, [True])) == 1


def test_frozen_category_errors_reports_unknown_value_of_unmasked_feature():
    data = pd.DataFrame({"x": [0.5, 2.5], "c": [1, 4]})
    errors = frozen_category_errors(pd.DataFrame({"x": [1.0], "c": [0]}), data, ["x"], [False, True])
    assert len(errors) == 1 and "'c' = 0" in errors[0]


@pytest.mark.parametrize("query, masked", [
    ({"x": [1.0], "c": [1]}, ["x"]),  # the value occurs in the data
    ({"x": [1.0], "c": [0]}, ["c"]),  # the feature may change, the query value is not used
    ({"x": [1.0], "c": [0]}, []),  # no masked features: every feature may change
    ({"x": [99.0], "c": [1]}, ["c"]),  # continuous features are out of scope
], ids=["known_value", "masked_feature", "nothing_masked", "continuous_feature"])
def test_frozen_category_errors_accepts_other_cases(query, masked):
    data = pd.DataFrame({"x": [0.5, 2.5], "c": [1, 4]})
    assert frozen_category_errors(pd.DataFrame(query), data, masked, [False, True]) == []


def test_incomplete_query_errors_reports_each_incomplete_row_once():
    query = pd.DataFrame({"x": [1.0, np.nan, 2.0], "c": [1, np.nan, np.nan]})
    errors = incomplete_query_errors(query)
    assert len(errors) == 2
    assert errors[0].startswith("Query 2:") and errors[1].startswith("Query 3:")


def test_incomplete_query_errors_accepts_complete_rows():
    assert incomplete_query_errors(pd.DataFrame({"x": [1.0], "c": [1]})) == []


def test_frozen_category_errors_leaves_incomplete_rows_to_incomplete_query_errors():
    data = pd.DataFrame({"x": [0.5, 2.5], "c": [1, 4]})
    query = pd.DataFrame({"x": [1.0, 1.0], "c": [1, np.nan]})
    assert frozen_category_errors(query, data, ["x"], [False, True]) == []


def test_counterfactuals_table_puts_query_first():
    query = pd.Series({"a": 1.0, "b": 2.0}, name=0)
    table = counterfactuals_table(query, np.array([[3.0, 2.0], [1.0, 4.0]]), ["a", "b"])
    assert table.index.tolist() == ["Query Instance", "Counterfactual 1", "Counterfactual 2"]
    assert table.loc["Counterfactual 2"].tolist() == [1.0, 4.0]


def test_class_probabilities_text():
    assert class_probabilities_text([0, 1], [0.25, 0.75], np.array(["no", "yes"])) == "**no**: 25.0% | **yes**: 75.0%"
