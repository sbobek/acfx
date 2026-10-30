"""Smoke tests of the Streamlit pages driven with streamlit.testing.v1.AppTest.

Each test renders a page with the session state left by the previous pages, interacts with widgets and checks
the state the page hands over to the next one.
"""
import pandas as pd
import pytest

from app.tests.helpers import page, with_state
from utils.classifier_factory import CLASSIFIER_LIST, select_features
from utils.data_loading import builtin_feature_types, load_builtin_dataset
from utils.const import ADJACENCY_OPTION_BAYESIAN, ADJACENCY_OPTION_DIRECTLINGAM


def assert_no_exception(app):
    assert not app.exception, [e.message for e in app.exception]


def button(app, label):
    return next(b for b in app.button if b.label == label)


def persisted_state(app):
    """Values kept by the app between pages (widget keys starting with '_' are re-created by load_value)."""
    state = app.session_state
    # streamlit>=1.5x exposes keys() on AppTest's session state, older versions only filtered_state
    keys = state.keys() if hasattr(state, "keys") else state.filtered_state.keys()
    return {k: state[k] for k in keys if not k.startswith("_")}


def test_welcome_page_renders():
    app = page("welcome").run()
    assert_no_exception(app)
    assert "ACFX" in app.title[0].value


# --- page 1: data selection -------------------------------------------------------------------

def test_builtin_dataset_selection_loads_data_and_feature_types():
    app = page("1_Data_Selection.py").run()
    app.radio(key="_source").set_value("Builtin").run()
    app.radio(key="_data_source_name").set_value("iris").run()

    assert_no_exception(app)
    assert app.session_state.data_loaded
    assert app.session_state.X.shape == (150, 4)
    assert app.session_state.feature_types["Type"].tolist() == ["continuous"] * 4


def test_changing_dataset_clears_state_of_later_pages(iris_state):
    app = with_state(page("1_Data_Selection.py"), dict(iris_state, source="Builtin", pbounds={},
                                                        Age_is_masked=True))
    app.run()
    app.radio(key="_data_source_name").set_value("wine").run()

    assert_no_exception(app)
    for key in ["selected_X", "classifier_instance", "pbounds", "Age_is_masked"]:
        assert key not in app.session_state
    assert app.session_state.X.shape[1] == 13


def test_mark_all_features_as_available():
    app = page("1_Data_Selection.py").run()
    app.radio(key="_source").set_value("Builtin").run()
    app.radio(key="_data_source_name").set_value("german credit").run()
    assert not app.session_state.feature_types["is_on"].all()

    button(app, "Mark all as available for evaluation").click().run()

    assert_no_exception(app)
    assert app.session_state.feature_types["is_on"].all()


# --- page 2: classifier selection -------------------------------------------------------------

def test_classifier_page_requires_data():
    app = page("2_Classifier_Selection.py").run()
    assert_no_exception(app)
    assert "Data Selection" in app.warning[0].value


@pytest.mark.parametrize("classifier_name, extra_selectbox", [
    (CLASSIFIER_LIST[0], None),
    (CLASSIFIER_LIST[1], ("_rf_criterion", "entropy")),
    (CLASSIFIER_LIST[2], ("_knn_weights", "distance")),
    (CLASSIFIER_LIST[2], ("_knn_algorithm", "brute")),
    (CLASSIFIER_LIST[3], None),
])
def test_initialize_classifier(iris_state, classifier_name, extra_selectbox):
    """Regression for selectboxes whose on_change args were a plain string (TypeError on change)."""
    state = {k: iris_state[k] for k in ["X", "y", "data", "feature_types", "data_loaded"]}
    app = with_state(page("2_Classifier_Selection.py"), state).run()
    app.selectbox(key="_classifier_name").set_value(classifier_name).run()
    if extra_selectbox:
        key, value = extra_selectbox
        app.selectbox(key=key).set_value(value).run()
        assert_no_exception(app)
        assert value in app.session_state.classifier_params.values()

    button(app, "Initialize classifier").click().run()

    assert_no_exception(app)
    assert hasattr(app.session_state.classifier_instance, "classes_")
    assert "Classifier is initialized" in app.markdown[-1].value


def test_logistic_regression_max_iter_is_passed_to_the_model(iris_state):
    state = {k: iris_state[k] for k in ["X", "y", "data", "feature_types", "data_loaded"]}
    app = with_state(page("2_Classifier_Selection.py"), state).run()
    app.selectbox(key="_classifier_name").set_value(CLASSIFIER_LIST[0]).run()
    assert app.number_input(key="_lr_max_iter").value == 100  # sklearn default keeps previous behaviour

    app.number_input(key="_lr_max_iter").set_value(3000).run()
    button(app, "Initialize classifier").click().run()

    assert_no_exception(app)
    assert app.session_state.classifier_params["max_iter"] == 3000
    assert app.session_state.classifier_instance.max_iter == 3000


def test_only_ebm_is_offered_for_categorical_features():
    app = page("1_Data_Selection.py").run()
    app.radio(key="_source").set_value("Builtin").run()
    app.radio(key="_data_source_name").set_value("german credit").run()
    button(app, "Mark all as available for evaluation").click().run()

    classifier_page = with_state(page("2_Classifier_Selection.py"), persisted_state(app)).run()

    assert_no_exception(classifier_page)
    assert classifier_page.selectbox(key="_classifier_name").options == [CLASSIFIER_LIST[3]]


# --- page 3: adjacency generation -------------------------------------------------------------

def test_adjacency_page_requires_classifier():
    app = page("3_Adjacency_Generation.py").run()
    assert "Classifier selection" in app.warning[0].value


def test_lingam_graph_is_generated_by_default_for_continuous_data(iris_state):
    app = with_state(page("3_Adjacency_Generation.py"), iris_state).run()
    app.checkbox(key="_plausibility_loss_on").check().run()
    app.checkbox(key="_generate_graph").check().run()

    assert_no_exception(app)
    assert app.session_state.adjacency_generator_name == ADJACENCY_OPTION_DIRECTLINGAM
    assert app.session_state.adjacency_matrix.shape == (4, 4)
    assert sorted(app.session_state.causal_order) == sorted(iris_state["selected_X"].columns)
    assert not app.error  # LiNGAM returns a DAG consistent with its causal order


def test_bayesian_network_can_be_selected(iris_state):
    app = with_state(page("3_Adjacency_Generation.py"), iris_state).run()
    app.checkbox(key="_plausibility_loss_on").check().run()
    app.selectbox(key="_adjacency_generator_name").set_value(ADJACENCY_OPTION_BAYESIAN).run()
    app.checkbox(key="_generate_graph").check().run()

    assert_no_exception(app)
    assert app.session_state.num_bins == 5
    assert set(app.session_state.bayesian_model.nodes()) <= set(iris_state["selected_X"].columns)


# --- page 4: evaluation settings --------------------------------------------------------------

def test_settings_page_requires_adjacency_step(iris_state):
    app = with_state(page("4_Evaluation_Settings.py"), iris_state).run()
    assert "Adjacency Generation" in app.warning[0].value


def test_bounds_and_weights_are_initialised_from_data(iris_state):
    app = with_state(page("4_Evaluation_Settings.py"), dict(iris_state, plausibility_loss_on=False)).run()
    app.checkbox(key="_petal width (cm)_is_masked").check().run()

    assert_no_exception(app)
    X = iris_state["selected_X"]
    assert app.session_state.pbounds == {f: (X[f].min(), X[f].max()) for f in X.columns}
    assert app.session_state["petal width (cm)_is_masked"] is True
    assert app.session_state.plausibility_loss == 0
    assert app.session_state.proximity_weight == 0.5


def test_categorical_features_get_integer_bounds():
    data = load_builtin_dataset('german credit')
    feature_types = builtin_feature_types('german credit', data).assign(is_on=True)
    selected_X, _ = select_features(feature_types, data.data)
    selected_X = selected_X.assign(Telephone=1)  # a feature with a single observed value
    state = dict(classifier_name=CLASSIFIER_LIST[3], plausibility_loss_on=False, selected_X=selected_X,
                 feature_types=feature_types)

    app = with_state(page("4_Evaluation_Settings.py"), state).run()

    assert_no_exception(app)
    assert app.session_state.pbounds["Job"] == (selected_X["Job"].min(), selected_X["Job"].max())
    assert isinstance(app.session_state.pbounds["Job"][0], int)
    assert "only one value" in app.info[0].value
    assert app.slider(key="_Job_pbounds").value == app.session_state.pbounds["Job"]


# --- page 5: run ------------------------------------------------------------------------------

def test_run_page_requires_settings():
    app = page("5_Run.py").run()
    assert "Evaluation Settings" in app.warning[0].value


def test_predict_shows_class_probabilities(evaluation_state):
    app = with_state(page("5_Run.py"), evaluation_state).run()
    button(app, "PREDICT").click().run()

    assert_no_exception(app)
    assert "Predicted Class" in app.info[0].value
    assert "setosa" in app.info[0].value


@pytest.mark.slow
def test_counterfactual_run_shows_query_and_counterfactuals(evaluation_state):
    app = with_state(page("5_Run.py"), evaluation_state).run()
    app.selectbox(key="_desired_class").set_value("virginica").run()
    app.number_input(key="_num_counterfactuals").set_value(2).run()
    app.number_input(key="_n_iter").set_value(10).run()
    app.number_input(key="_init_points").set_value(2).run()
    button(app, "COUNTERFACTUAL").click().run()

    assert_no_exception(app)
    result = app.dataframe[-1].value
    assert result.index.tolist() == ["Query Instance", "Counterfactual 1", "Counterfactual 2"]
    assert result.columns.tolist() == evaluation_state["selected_X"].columns.tolist()


@pytest.mark.slow
def test_counterfactual_run_warns_about_query_outside_bounds(evaluation_state):
    feature = evaluation_state["selected_X"].columns[0]
    value = evaluation_state["selected_X"][feature].iloc[0]
    state = {**evaluation_state, f"{feature}_pbounds": (value + 1, value + 2)}
    app = with_state(page("5_Run.py"), state).run()
    app.selectbox(key="_desired_class").set_value("virginica").run()
    app.number_input(key="_n_iter").set_value(10).run()
    app.number_input(key="_init_points").set_value(2).run()
    button(app, "COUNTERFACTUAL").click().run()

    assert_no_exception(app)
    assert any(feature in warning.value and "outside the bounds" in warning.value for warning in app.warning)


@pytest.mark.slow
@pytest.mark.parametrize("generator", [ADJACENCY_OPTION_DIRECTLINGAM, ADJACENCY_OPTION_BAYESIAN])
def test_full_flow_through_all_pages(generator):
    """Data selection -> classifier -> causal model -> settings -> counterfactuals, carrying the session state."""
    app = page("1_Data_Selection.py").run()
    app.radio(key="_source").set_value("Builtin").run()
    app.radio(key="_data_source_name").set_value("iris").run()

    app = with_state(page("2_Classifier_Selection.py"), persisted_state(app)).run()
    app.selectbox(key="_classifier_name").set_value(CLASSIFIER_LIST[1]).run()
    button(app, "Initialize classifier").click().run()

    app = with_state(page("3_Adjacency_Generation.py"), persisted_state(app)).run()
    app.checkbox(key="_plausibility_loss_on").check().run()
    app.selectbox(key="_adjacency_generator_name").set_value(generator).run()

    app = with_state(page("4_Evaluation_Settings.py"), persisted_state(app)).run()
    app.checkbox(key="_sepal length (cm)_is_masked").check().run()

    app = with_state(page("5_Run.py"), persisted_state(app)).run()
    app.selectbox(key="_desired_class").set_value("versicolor").run()
    app.number_input(key="_n_iter").set_value(10).run()
    app.number_input(key="_init_points").set_value(2).run()
    button(app, "COUNTERFACTUAL").click().run()

    assert_no_exception(app)
    result = app.dataframe[-1].value
    assert isinstance(result, pd.DataFrame)
    assert result.index.tolist() == ["Query Instance", "Counterfactual 1"]
    # only the masked (actionable) feature may differ from the query
    unchanged = result.columns.drop("sepal length (cm)")
    assert (result.loc["Counterfactual 1", unchanged] == result.loc["Query Instance", unchanged]).all()
