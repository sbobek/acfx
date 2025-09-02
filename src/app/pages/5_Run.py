import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from utils.session_state import store_value, load_value
from acfx import AcfxCustom, AcfxEBM, AcfxLinear, ACFX
from utils.key_helper import get_pbounds_key

def get_pbounds() -> dict[str,tuple[float,float]]:
    return {key: (float(st.session_state[get_pbounds_key(key)][0]), float(st.session_state[get_pbounds_key(key)][1]))
            for key in st.session_state.pbounds.keys()}

def get_masked_features() -> list[str]:
    suffix = "_is_masked"
    masked_features_keys = [key for key in st.session_state.keys()
                            if key.endswith(suffix) and st.session_state[key] == True]
    return [feature_masked[:-len(suffix)] for feature_masked in masked_features_keys]

def get_all_columns() -> list[str]:
    return list(st.session_state.selected_X.columns)

def get_categorical_indicator() -> list[bool]:
    is_categorical = {row['Column Name']: row['Type'] != 'continuous'  for _, row in st.session_state.feature_types.iterrows()}
    if 'data' in st.session_state.data:
        assert list(st.session_state.data.data.columns) == list(is_categorical.keys())
    else:
        assert list(st.session_state.data.columns) == list(is_categorical.keys())
    return [is_categorical[feature_name] for feature_name in get_all_columns()]


@st.cache_resource
def get_acfx() -> ACFX:
    classifier_instance = st.session_state.classifier_instance
    if isinstance(classifier_instance, LogisticRegression):
        acfx = AcfxLinear(classifier_instance)
    elif isinstance(classifier_instance, ExplainableBoostingClassifier):
        acfx = AcfxEBM(classifier_instance)
    elif (isinstance(classifier_instance, RandomForestClassifier)
          or isinstance(classifier_instance, KNeighborsClassifier)):
        acfx = AcfxCustom(classifier_instance)
    else:
        raise ValueError("classifier_instance out of range")
    return acfx

@st.cache_resource
def fit_acfx(features_order) -> ACFX:
    acfx_instance = get_acfx()
    adjacency_matrix = None
    casual_order = None
    if 'adjacency_matrix' in st.session_state and st.session_state.adjacency_matrix is not None:
        adjacency_matrix = st.session_state.adjacency_matrix
    if 'casual_order' in st.session_state and st.session_state.casual_order is not None:
        casual_order = st.session_state.casual_order
        if all(isinstance(item, str) for item in casual_order):
            all_columns = get_all_columns()
            temp = [all_columns.index(column) for column in casual_order]
            casual_order = temp
    if 'pbounds' not in st.session_state or not isinstance(st.session_state.pbounds, dict):
        raise TypeError("pbounds must be initialized in session state and be dict")
    if (casual_order is None and adjacency_matrix is not None) or (casual_order is not None and adjacency_matrix is None):
        raise KeyError('casual_order and adjacency_matrix must be specified together')
    elif casual_order is not None and not isinstance(casual_order,list):
        raise TypeError('casual_order must be a list')
    elif adjacency_matrix is not None and not isinstance(adjacency_matrix,np.ndarray):
        raise TypeError('adjacency_matrix must be a numpy array')
    elif 'feature_types' not in st.session_state or st.session_state.feature_types is None:
        raise KeyError('feature_types must be initialized in session state here')

    return acfx_instance.fit(X= st.session_state.selected_X,
                             adjacency_matrix=adjacency_matrix,
                             casual_order=casual_order,
                             pbounds=get_pbounds(),
                             masked_features=get_masked_features(),
                             categorical_indicator=get_categorical_indicator(),
                             features_order=features_order)
def get_target_class_list():
    if 'target_names' in st.session_state.data:
        assert len(st.session_state.classifier_instance.classes_) == len(st.session_state.data.target_names)
        target_class_list = st.session_state.data.target_names
    else:
        target_class_list = st.session_state.classifier_instance.classes_
    return target_class_list

def get_numeric_desired_class():
    if isinstance(st.session_state['desired_class'], str):
        desired_class = np.argwhere(get_target_class_list() == st.session_state['desired_class'])[0][0]
    else:
        desired_class = st.session_state['desired_class']
    return desired_class

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
else:
    if 'classifier_name' not in st.session_state or st.session_state.classifier_name is None:
        raise ValueError('classifier_name must be in session state')
    if 'classifier_instance' not in st.session_state or st.session_state.classifier_instance is None:
        raise ValueError('classifier_instance must be in session state')

    show_desired_class_input()
    show_how_many_cfs_per_query_instance()
    show_init_points_choice()
    show_n_iter_choice()
    show_sampling_from_model_choice()
    query_instances = st.data_editor(
        data=pd.DataFrame([0] * len(get_all_columns()), index=get_all_columns()).T,
        num_rows='dynamic',
        use_container_width=True,
    )
    is_button_disabled = query_instances is None
    if st.button("EVALUATE", disabled=is_button_disabled):
        if query_instances is None:
            st.warning("⚠️ Query instance must be provided first")
        elif 'desired_class' not in st.session_state or st.session_state.desired_class is None:
            st.warning("⚠️ Choose desired class first")
        else:
            acfx = fit_acfx(list(query_instances.columns))
            with st.spinner("Evaluating counterfactuals..."):
                for i, query_instance in query_instances.iterrows():
                    cfs = acfx.counterfactual(
                                     query_instance=query_instance.array,
                                     desired_class=get_numeric_desired_class(),
                                     num_counterfactuals=st.session_state.num_counterfactuals,
                                     proximity_weight=st.session_state.proximity_weight,
                                     plausibility_weight=st.session_state.plausibility_loss,
                                     diversity_weight=st.session_state.diversity_weight,
                                     init_points=st.session_state.init_points,
                                     n_iter=st.session_state.n_iter,
                                     sampling_from_model=st.session_state.sampling_from_model)

                    cfs = pd.DataFrame(cfs, columns=list(query_instances.columns))
                    cfs.index = [f"Counterfactual {i+1}" for i in range(len(cfs))]
                    query_instance_with_index = pd.DataFrame([query_instance])
                    query_instance_with_index.index = ["Query Instance"]
                    st.subheader(f"📊 RESULT {i+1}")
                    st.dataframe(pd.concat([query_instance_with_index,cfs]), use_container_width=True)


