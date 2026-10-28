import warnings
import pandas as pd
import streamlit as st
from utils.session_state import store_value, load_value
from acfx import AcfxCustom, ACFX
from utils.features_by_type import get_categorical_indicator, get_all_columns
from utils.const import ADJACENCY_OPTION_DIRECTLINGAM,ADJACENCY_OPTION_BAYESIAN
from utils.run_helpers import (acfx_for_classifier, causal_order_to_indices, class_probabilities_text,
                               counterfactuals_table, fit_explainer, masked_features, numeric_desired_class,
                               pbounds_from_state, prediction_label, query_template, target_class_list)

def get_pbounds() -> dict[str,tuple[float,float]]:
    return pbounds_from_state(st.session_state, st.session_state.pbounds.keys())

def get_masked_features() -> list[str]:
    return masked_features(st.session_state)

# @st.cache_resource
def get_acfx():
    return acfx_for_classifier(st.session_state.classifier_instance)

# @st.cache_resource
def fit_acfx(features_order) -> None | AcfxCustom | ACFX:
    def fit(acfx_instance: ACFX, bounds, **causal_model):
        return fit_explainer(acfx_instance, X=st.session_state.selected_X, pbounds=bounds,
                             features_order=features_order, masked=get_masked_features(),
                             categorical_indicator=get_categorical_indicator(), **causal_model)

    acfx_instance = get_acfx()
    adjacency_matrix = None
    causal_order = None
    if 'pbounds' not in st.session_state or not isinstance(st.session_state.pbounds, dict):
        raise TypeError("pbounds must be initialized in session state and be dict")
    elif 'feature_types' not in st.session_state or st.session_state.feature_types is None:
        raise KeyError('feature_types must be initialized in session state here')
    bounds = get_pbounds()
    if not st.session_state.plausibility_loss_on or st.session_state.plausibility_loss == 0.0:
        return fit(acfx_instance, bounds)
    elif st.session_state.plausibility_loss_on and st.session_state.plausibility_loss > 0:
        if st.session_state.adjacency_generator_name == ADJACENCY_OPTION_DIRECTLINGAM:
            if 'adjacency_matrix' in st.session_state and st.session_state.adjacency_matrix is not None:
                adjacency_matrix = st.session_state.adjacency_matrix.to_numpy()
            if 'causal_order' in st.session_state and st.session_state.causal_order is not None:
                causal_order = causal_order_to_indices(st.session_state.causal_order, get_all_columns())

            if 'plausibility_loss' in st.session_state and st.session_state.plausibility_loss > 0.0:
                if causal_order is None or adjacency_matrix is None:
                    raise KeyError('causal_order and adjacency_matrix must be both specified')

            return fit(acfx_instance, bounds, adjacency_matrix=adjacency_matrix, causal_order=causal_order)
        elif st.session_state.adjacency_generator_name == ADJACENCY_OPTION_BAYESIAN:
            if 'bayesian_model' not in st.session_state or st.session_state.bayesian_model is None:
                raise KeyError('bayesian_model must be specified')
            if 'num_bins' not in st.session_state or st.session_state.num_bins is None:
                raise KeyError('num_bins must be specified')
            return fit(acfx_instance, bounds, num_bins=st.session_state.num_bins,
                       bayesian_model=st.session_state.bayesian_model)
        else:
            raise IndexError('adjacency_generator_name')
    return None


def get_target_class_list():
    return target_class_list(st.session_state.data, st.session_state.classifier_instance.classes_)

def get_prediction_label(pred) -> str:
    try:
        classifier_instance = st.session_state.classifier_instance
        if hasattr(classifier_instance, "classes_") and classifier_instance.classes_ is not None:
            return prediction_label(pred, classifier_instance.classes_, get_target_class_list())
    except Exception:
        pass
    return str(pred)

def get_numeric_desired_class():
    return numeric_desired_class(st.session_state['desired_class'], get_target_class_list())

def show_desired_class_input():
    load_value('desired_class')
    target_class_list=get_target_class_list()
    st.selectbox("Target:", target_class_list, key="_desired_class", on_change=store_value, args=['desired_class'])

def show_how_many_cfs_per_query_instance():
    load_value('num_counterfactuals',1)
    st.number_input(
        label="Num of counterfactuals generated per query instance",
        min_value=1,
        max_value=10,
        format="%d",
        step=1,
        key="_num_counterfactuals",
        on_change=store_value, args=['num_counterfactuals']
    )

def show_init_points_choice():
    load_value('init_points',10)
    st.number_input(
        label="Number of initial points for Bayesian Optimization",
        min_value=1,
        format="%d",
        step=1,
        key="_init_points",
        on_change=store_value, args=['init_points']
    )

def show_n_iter_choice():
    load_value('n_iter',100)
    st.number_input(
        label="Number of iterations for Bayesian Optimization",
        min_value=1,
        format="%d",
        step=1,
        key="_n_iter",
        on_change=store_value, args=['n_iter']
    )

def show_sampling_from_model_choice():
    load_value('sampling_from_model',True)
    st.checkbox("Sampling from model",
                key="_sampling_from_model", on_change=store_value, args=['sampling_from_model'])

if 'proximity_weight' not in st.session_state \
    or 'diversity_weight' not in st.session_state \
    or 'sparsity_weight' not in st.session_state or 'data' not in st.session_state:
        st.warning("⚠️ Start by running 'Evaluation Settings'")
elif 'classifier_name' not in st.session_state or st.session_state.classifier_name is None or \
        'classifier_instance' not in st.session_state or st.session_state.classifier_instance is None:
    st.warning("⚠️ Start by running 'Classifier Selection'")
elif 'plausibility_loss_on' not in st.session_state or \
      ('adjacency_generator_name' not in st.session_state
       and st.session_state.plausibility_loss_on == True) \
      or 'plausibility_loss' not in st.session_state:
    st.warning("⚠️ Start by running 'Adjacency Generation'")
elif 'plausibility_loss_on' in st.session_state and \
        st.session_state.plausibility_loss_on == True and \
        st.session_state.adjacency_generator_name != ADJACENCY_OPTION_DIRECTLINGAM and \
        st.session_state.adjacency_generator_name != ADJACENCY_OPTION_BAYESIAN:
            raise ValueError("'adjacency_generator_name' must indicate bayesian or lingam adjacency")
else:
    show_desired_class_input()
    show_how_many_cfs_per_query_instance()
    show_init_points_choice()
    show_n_iter_choice()
    show_sampling_from_model_choice()

    categorical_indicator = get_categorical_indicator()
    all_columns = get_all_columns()
    df_init = query_template(all_columns, categorical_indicator)

    query_instances = st.data_editor(
        data=df_init,
        num_rows='dynamic',
        use_container_width=True,
    )

    is_button_disabled = query_instances is None

    col1, col2, _ = st.columns([1, 1, 3])
    with col2:
        evaluate_clicked = st.button("COUNTERFACTUAL", disabled=is_button_disabled, use_container_width=True)
    with col1:
        check_blackbox_clicked = st.button("PREDICT", disabled=is_button_disabled, use_container_width=True)

    if evaluate_clicked:
        if query_instances is None:
            st.warning("⚠️ Query instance must be provided first")
        elif 'desired_class' not in st.session_state or st.session_state.desired_class is None:
            st.warning("⚠️ Choose desired class first")
        else:
            acfx = fit_acfx(list(query_instances.columns))
            with st.spinner("Evaluating counterfactuals..."):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    for i, query_instance in query_instances.iterrows():
                        cfs = acfx.counterfactual(
                                         query_instance=query_instance.array,
                                         desired_class=get_numeric_desired_class(),
                                         num_counterfactuals=st.session_state.num_counterfactuals,
                                         proximity_weight=st.session_state.proximity_weight,
                                         plausibility_weight=st.session_state.plausibility_loss,
                                         diversity_weight=st.session_state.diversity_weight,
                                         sparsity_weight=st.session_state.sparsity_weight,
                                         init_points=st.session_state.init_points,
                                         n_iter=st.session_state.n_iter,
                                         sampling_from_model=st.session_state.sampling_from_model)

                        st.subheader(f"📊 RESULT {i+1}")
                        st.dataframe(counterfactuals_table(query_instance, cfs, query_instances.columns),
                                     use_container_width=True)

    elif check_blackbox_clicked:
        if query_instances is None:
            st.warning("⚠️ Query instance must be provided first")
        else:
            classifier_instance = st.session_state.classifier_instance
            with st.spinner("Checking black-box predictions..."):
                for i, query_instance in query_instances.iterrows():
                    query_instance_df = pd.DataFrame([query_instance])
                    query_instance_for_pred = query_instance_df[st.session_state.selected_X.columns]
                    
                    pred = classifier_instance.predict(query_instance_for_pred)[0]
                    pred_label = get_prediction_label(pred)
                    
                    if not hasattr(classifier_instance, "predict_proba"):
                        raise AttributeError(f"The classifier '{type(classifier_instance).__name__}' does not support 'predict_proba'.")
                    
                    probs = classifier_instance.predict_proba(query_instance_for_pred)[0]
                    classes = list(classifier_instance.classes_)
                    pred_idx = classes.index(pred)
                    pred_prob = probs[pred_idx]
                    
                    prob_info = class_probabilities_text(classes, probs, get_target_class_list())
                        
                    query_instance_with_index = pd.DataFrame([query_instance])
                    query_instance_with_index.index = ["Query Instance"]
                    
                    st.subheader(f"📊 BLACK-BOX PREDICTION {i+1}")
                    st.dataframe(query_instance_with_index, use_container_width=True)
                    
                    st.info(f"🔮 **Predicted Class:** `{pred_label}` ({pred_prob:.1%})  \n**Class Probabilities:** {prob_info}")

