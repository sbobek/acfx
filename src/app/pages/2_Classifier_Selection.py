import streamlit as st
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from utils.session_state import store_value, load_value
from utils.params_initializing import (init_logisticregression_params,init_randomforest_params,
                                       init_KNeighborsClassifier_params, init_ExplainableBoostingClassifier_params)

classifier_list = [
        "LogisticRegression",
        "RandomForestClassifier",
        "KNeighborsClassifier",
        "ExplainableBoostingClassifier"
    ]

def init_classifier():
    if st.session_state.X is None:
        raise ValueError("Data for classifier fit is not available (st.session_state.X)")
    if st.session_state.y is None:
        raise ValueError("Data for classifier fit is not available (st.session_state.y)")
    classifier = None
    params = st.session_state.classifier_params
    selected_feature_labels = [item["Column Name"] for _, item in st.session_state.feature_types.iterrows()
                               if item["is_on"] and item["Column Name"] in st.session_state.X.columns]
    selected_feature_types = [item["Type"] for _, item in st.session_state.feature_types.iterrows()
                              if item["is_on"] and item["Column Name"] in selected_feature_labels]
    st.session_state.selected_X = st.session_state.X[selected_feature_labels]
    if params is not None:
        if st.session_state.classifier_name == classifier_list[0]:
            classifier = LogisticRegression(**params)
        elif st.session_state.classifier_name == classifier_list[1]:
            classifier = RandomForestClassifier(**params)
        elif st.session_state.classifier_name == classifier_list[2]:
            classifier = KNeighborsClassifier(**params)
        elif st.session_state.classifier_name == classifier_list[3]:
            classifier = ExplainableBoostingClassifier(feature_types= selected_feature_types, **params)
        else:
            raise IndexError("classifier_name out of range")

        classifier = classifier.fit(st.session_state.selected_X.to_numpy(),st.session_state.y)

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


if 'classifier_instance' not in st.session_state:
    st.session_state['classifier_instance'] = None
if 'classifier_params' not in st.session_state:
    st.session_state['classifier_params'] = None

st.title("⚙️ Select classifier")

if 'data_loaded' in st.session_state and st.session_state.data_loaded:
    load_value('classifier_name')
    st.selectbox("Classifier:", classifier_list, key="_classifier_name", on_change=store_value, args=['classifier_name'])
    if st.session_state.classifier_name is not None:
        init_classifier_params()

    if st.button("Initialize classifier"):
        init_classifier()
    if st.button("Reset classifier"):
        st.session_state["classifier_instance"] = None
    if st.session_state['classifier_instance'] is not None:
        st.write(f'Classifier is initialized: {type(st.session_state["classifier_instance"])}')
    else:
        st.write('Classifier is not initialized')
else:
    st.warning("⚠️ Start by selecting data in 'Data Selection'")