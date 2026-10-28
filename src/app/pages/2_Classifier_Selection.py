import streamlit as st

from utils.session_state import store_value, load_value
from utils.params_initializing import (init_logisticregression_params,init_randomforest_params,
                                       init_KNeighborsClassifier_params, init_ExplainableBoostingClassifier_params)
from utils.features_by_type import get_nominal_cols, get_ordinal_cols
from utils.classifier_factory import (CLASSIFIER_LIST as classifier_list, available_classifiers as classifiers_for,
                                      build_classifier, select_features)

def init_selected_x():
    st.session_state.selected_X, st.session_state.selected_feature_types = select_features(
        st.session_state.feature_types, st.session_state.X)

def init_classifier():
    if st.session_state.X is None:
        raise ValueError("Data for classifier fit is not available (st.session_state.X)")
    if st.session_state.y is None:
        raise ValueError("Data for classifier fit is not available (st.session_state.y)")
    classifier = None
    params = st.session_state.classifier_params

    if params is not None:
        classifier = build_classifier(st.session_state.classifier_name, params,
                                      st.session_state.selected_feature_types)
        classifier = classifier.fit(st.session_state.selected_X, st.session_state.y)

    st.session_state.classifier_instance = classifier

def init_classifier_params():
    if st.session_state.classifier_name == classifier_list[0]:
        init_logisticregression_params()
    elif st.session_state.classifier_name == classifier_list[1]:
        init_randomforest_params()
    elif st.session_state.classifier_name == classifier_list[2]:
        init_KNeighborsClassifier_params()
    elif st.session_state.classifier_name == classifier_list[3]:
        init_ExplainableBoostingClassifier_params()

def get_available_classifiers():
    return classifiers_for(len(get_nominal_cols()) + len(get_ordinal_cols()))


if 'X' not in st.session_state or st.session_state.X is None:
    st.warning("⚠️ Start by selecting dataset in 'Data Selection'")
else:
    if 'classifier_instance' not in st.session_state:
        st.session_state['classifier_instance'] = None
    if 'classifier_params' not in st.session_state:
        st.session_state['classifier_params'] = None

    st.title("⚙️ Select classifier")
    init_selected_x()

    if 'data_loaded' in st.session_state and st.session_state.data_loaded:
        load_value('classifier_name')
        available_classifiers = get_available_classifiers()
        if available_classifiers != classifier_list:
            st.info("Some of the blackbox classifiers are not available when using categorical features")

        st.selectbox("Classifier:", get_available_classifiers(), key="_classifier_name", on_change=store_value, args=['classifier_name'])
        if st.session_state.classifier_name is not None:
            init_classifier_params()

        if st.button("Initialize classifier"):
            if st.session_state.classifier_name is None:
                st.warning('Select classifier name first')
            init_classifier()
        if st.button("Reset classifier"):
            st.session_state["classifier_instance"] = None
        if st.session_state['classifier_instance'] is not None:
            st.write(f'Classifier is initialized: {type(st.session_state["classifier_instance"])}')
        else:
            st.write('Classifier is not initialized')
    else:
        st.warning("⚠️ Start by selecting data in 'Data Selection'")