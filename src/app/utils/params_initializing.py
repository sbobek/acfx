from streamlit import session_state
from utils.session_state import store_value, load_value
import streamlit as st

def init_randomforest_params():
    st.markdown("**see requirements:** https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier")
    load_value('rf_n_estimators', 100)
    st.number_input(
        label="n_estimators",
        min_value=1,
        format="%d",
        step=1,
        key="_rf_n_estimators",
        on_change=store_value, args=['rf_n_estimators']
    )
    load_value("rf_criterion", 'gini')
    st.selectbox('criterion', ['gini', 'entropy', 'log_loss'], key='_rf_criterion', on_change=store_value, args='rf_criterion')

    load_value('rf_max_depth', 0)
    st.number_input(
        label="max_depth (0 is None)",
        min_value=0,
        format="%d",
        step=1,
        key="_rf_max_depth",
        on_change=store_value, args=['rf_max_depth']
    )

    load_value('rf_min_samples_split', 2.0)
    st.number_input(
        label="min_samples_split",
        min_value=0.0,
        format="%.3f",
        step=0.01,
        key="_rf_min_samples_split",
        on_change=store_value, args=['rf_min_samples_split']
    )

    load_value('rf_min_samples_leaf', 1.0)
    st.number_input(
        label="min_samples_leaf",
        min_value=0.0,
        format="%.3f",
        step=0.01,
        key="_rf_min_samples_leaf",
        on_change=store_value, args=['rf_min_samples_leaf']
    )

    load_value('rf_min_weight_fraction_leaf', 0.0)
    st.number_input(
        label="min_weight_fraction_leaf",
        min_value=0.0,
        format="%.3f",
        step=0.01,
        key="_rf_min_weight_fraction_leaf",
        on_change=store_value, args=['rf_min_weight_fraction_leaf']
    )
    load_value('rf_max_features', 'sqrt')
    st.selectbox('max_features', ['None', 'sqrt', 'log2'],
                 key='_rf_max_features', on_change=store_value, args=['rf_max_features'])

    load_value('rf_max_leaf_nodes', 0)
    st.number_input(
        label="max_leaf_nodes (0 is None)",
        min_value=0,
        format="%d",
        step=1,
        key="_rf_max_leaf_nodes",
        on_change=store_value, args=['rf_max_leaf_nodes']
    )

    load_value('rf_min_impurity_decrease', 0.0)
    st.number_input(
        label="min_impurity_decrease",
        min_value=0.0,
        format="%.3f",
        step=0.01,
        key="_rf_min_impurity_decrease",
        on_change=store_value, args=['rf_min_impurity_decrease']
    )

    if st.session_state.rf_n_estimators is not None and \
            st.session_state.rf_criterion is not None and \
            st.session_state.rf_max_depth is not None and \
            st.session_state.rf_min_samples_split is not None and \
            st.session_state.rf_min_samples_leaf is not None and \
            st.session_state.rf_min_weight_fraction_leaf is not None and \
            st.session_state.rf_max_leaf_nodes is not None and \
            st.session_state.rf_min_impurity_decrease is not None and \
            st.session_state.rf_max_features is not None:

        max_features = st.session_state.rf_max_features
        if max_features == 'None':
            max_features = None
        max_depth = st.session_state.rf_max_depth
        if max_depth == 0:
            max_depth = None
        max_leaf_nodes = st.session_state.rf_max_leaf_nodes
        if max_leaf_nodes is 0:
            max_leaf_nodes = None
        st.session_state.classifier_params = {
            'n_estimators': st.session_state.rf_n_estimators,
            'criterion': st.session_state.rf_criterion,
            'max_depth': max_depth,
            'min_samples_split': st.session_state.rf_min_samples_split,
            'min_samples_leaf': st.session_state.rf_min_samples_leaf,
            'min_weight_fraction_leaf': st.session_state.rf_min_weight_fraction_leaf,
            'max_features': max_features,
            'max_leaf_nodes': max_leaf_nodes,
            'min_impurity_decrease' : st.session_state.rf_min_impurity_decrease,
        }


def init_logisticregression_params():
    st.markdown("**see requirements:** https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html")
    load_value('lr_tol', 1e-6)
    st.number_input(
        label="Tolerance (tol)",
        min_value=1e-10,
        max_value=1.0,
        format="%.10f",
        step=1e-6,
        key="_lr_tol",
        on_change=store_value, args=['lr_tol']
    )
    load_value('lr_fit_intercept', False)
    st.checkbox(label="fit_intercept", key="_lr_fit_intercept", on_change=store_value, args=['lr_fit_intercept'])
    if st.session_state.lr_tol is not None and st.session_state.lr_fit_intercept is not None:
        st.session_state.classifier_params = {
            'tol': st.session_state.lr_tol,
            'fit_intercept': st.session_state.lr_fit_intercept
        }

def init_KNeighborsClassifier_params():
    st.markdown("**see requirements:** https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html")
    load_value('knn_n_neighbors', 5)
    st.number_input(
        label="n_neighbors",
        min_value=1,
        format="%d",
        step=1,
        key="_knn_n_neighbors",
        on_change=store_value, args=['knn_n_neighbors']
    )
    load_value('knn_weights', 'uniform')
    st.selectbox('weights', ['distance', 'uniform', 'None'], key='_knn_weights', on_change=store_value, args='knn_weights')
    load_value('knn_algorithm', 'auto')
    st.selectbox('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'], key='_knn_algorithm', on_change=store_value, args='knn_algorithm')
    load_value('knn_leaf_size',30)
    st.number_input(
        label="leaf_size",
        min_value=1,
        format="%d",
        step=1,
        key="_knn_leaf_size",
        on_change=store_value, args=['knn_leaf_size']
    )
    load_value('knn_p',2)
    st.number_input(
        label="Power parameter for the Minkowski metric (p)",
        min_value=1,
        format="%d",
        step=1,
        key="_knn_p",
        on_change=store_value, args=['knn_p']
    )

    if st.session_state.knn_n_neighbors is not None \
        and st.session_state.knn_weights is not None \
        and st.session_state.knn_algorithm is not None \
        and st.session_state.knn_leaf_size is not None \
        and st.session_state.knn_p is not None:
        weights = st.session_state.knn_weights
        if weights == 'None':
            weights = None
        st.session_state.classifier_params = {
            'n_neighbors': st.session_state.knn_n_neighbors,
            'weights': weights,
            'algorithm': st.session_state.knn_algorithm,
            'leaf_size': st.session_state.knn_leaf_size,
            'p': st.session_state.knn_p
        }

def init_ExplainableBoostingClassifier_params():
    st.markdown("**see requirements:** https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html")
    def parse_interactions(interactions_instance):
        def assert_interaction_string(s: str) -> int:
            if isinstance(s, str) and s.endswith("x"):
                factor = float(s[:-1])
            raise ValueError("Invalid format. Expected format: '<float>x'")
        if interactions_instance is None:
            return 0
        if interactions_instance.isdigit():
            return int(interactions_instance)
        try:
            assert_interaction_string(interactions_instance)
            return interactions_instance
        except:
            try:
                _interactions = float(interactions_instance)
                return _interactions
            except:
                return 0

    load_value('ebm_random_state', 42)
    st.number_input(
        label="random_state (0 means uses device_random and generates non-repeatable sequences)",
        min_value=0,
        format="%d",
        step=1,
        key="_ebm_random_state",
        on_change=store_value, args=['ebm_random_state']
    )
    load_value('ebm_max_bins', 1024)
    st.number_input(
        label="max_bins",
        min_value=1,
        format="%d",
        step=1,
        key="_ebm_max_bins",
        on_change=store_value, args=['ebm_max_bins']
    )
    load_value('ebm_max_interaction_bins', 64)
    st.number_input(
        label="max_interaction_bins",
        min_value=1,
        format="%d",
        step=1,
        key="_ebm_max_interaction_bins",
        on_change=store_value, args=['ebm_max_interaction_bins']
    )
    load_value('ebm_interactions', '0')
    st.text_input(
        label="interactions",
        key="_ebm_interactions",
        on_change=store_value, args=['ebm_interactions'])
    load_value('ebm_validation_size', 0.15)
    st.number_input(
        label="validation_size",
        min_value=0.0,
        format="%.2f",
        step=0.01,
        key="_ebm_validation_size",
        on_change=store_value, args=['ebm_validation_size']
    )
    load_value('ebm_outer_bags', 1)
    st.number_input(
        label="outer_bags",
        min_value=1,
        format="%d",
        step=1,
        key="_ebm_outer_bags",
        on_change=store_value, args=['ebm_outer_bags']
    )
    load_value('ebm_inner_bags', 0)
    st.number_input(
        label="inner_bags",
        min_value=0,
        format="%d",
        step=1,
        key="_ebm_inner_bags",
        on_change=store_value, args=['ebm_inner_bags']
    )
    load_value('ebm_learning_rate', 0.015)
    st.number_input(
        label="learning_rate",
        min_value=0.001,
        format="%.3f",
        step=0.001,
        key="_ebm_learning_rate",
        on_change=store_value, args=['ebm_learning_rate']
    )

    if st.session_state.ebm_random_state is not None \
        and st.session_state.ebm_max_bins is not None \
            and st.session_state.ebm_max_interaction_bins is not None \
            and st.session_state.ebm_interactions is not None \
            and st.session_state.ebm_validation_size is not None \
            and st.session_state.ebm_outer_bags is not None \
            and st.session_state.ebm_inner_bags is not None \
            and st.session_state.ebm_learning_rate is not None:
        random_state = st.session_state.ebm_random_state
        if random_state == 0:
            random_state = None
        interactions = parse_interactions(st.session_state.ebm_interactions)

        st.session_state.classifier_params ={
            'random_state': random_state,
            'max_bins': st.session_state.ebm_max_bins,
            'max_interaction_bins' : st.session_state.ebm_max_interaction_bins,
            'interactions' : interactions,
            'validation_size' : st.session_state.ebm_validation_size,
            'outer_bags' : st.session_state.ebm_outer_bags,
            'inner_bags' : st.session_state.ebm_inner_bags,
            'learning_rate' : st.session_state.ebm_learning_rate,
        }