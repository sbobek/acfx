from utils.session_state import store_value, load_value
import streamlit as st

def init_randomforest_params():
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
    st.selectbox('criterion', ['gini', 'entropy', 'log_loss'], key='_rf_criterion', on_change=store_value, param='rf_criterion')

    load_value('rf_max_depth', 0)
    st.number_input(
        label="max_depth (0 is None)",
        min_value=0,
        format="%d",
        step=1,
        key="_rf_max_depth",
        on_change=store_value, args=['rf_max_depth']
    )

    load_value('rf_min_samples_split', 2)
    st.number_input(
        label="min_samples_split",
        min_value=0,
        format="%.3f",
        step=0.01,
        key="_rf_min_samples_split",
        on_change=store_value, args=['rf_min_samples_split']
    )

    load_value('rf_min_samples_leaf', 1)
    st.number_input(
        label="min_samples_leaf",
        min_value=0,
        format="%.3f",
        step=0.01,
        key="_rf_min_samples_leaf",
        on_change=store_value, args=['rf_min_samples_leaf']
    )

    load_value('rf_min_weight_fraction_leaf', 0)
    st.number_input(
        label="min_weight_fraction_leaf",
        min_value=0,
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
        format="%.3f",
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
            'rf_criterion': st.session_state.rf_rf_criterion,
            'max_depth': max_depth,
            'min_samples_split': st.session_state.rf_min_samples_split,
            'min_samples_leaf': st.session_state.rf_min_samples_leaf,
            'min_weight_fraction_leaf': st.session_state.rf_min_weight_fraction_leaf,
            'max_features': max_features,
            'max_leaf_nodes': max_leaf_nodes,
            'min_impurity_decrease' : st.session_state.rf_min_impurity_decrease,
        }


def init_logisticregression_params():
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
            'lr_tol': st.session_state.lr_tol,
            'fit_intercept': st.session_state.lr_fit_intercept
        }

def init_KNeighborsClassifier_params():
    pass