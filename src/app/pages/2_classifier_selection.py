import streamlit as st
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from utils.session_state import store_value, load_value
from utils.params_initializing import init_logisticregression_params, init_randomforest_params, init_KNeighborsClassifier_params

classifier_list = [
        "LogisticRegression",
        "RandomForestClassifier",
        "KNeighborsClassifier",
        "ExplainableBoostingClassifier"
    ]

def init_classifier():
    classifier = None
    if st.session_state.classifier_name == classifier_list[0]:
        classifier = LogisticRegression
    elif st.session_state.classifier_name == classifier_list[1]:
        classifier = RandomForestClassifier
    elif st.session_state.classifier_name == classifier_list[2]:
        classifier = KNeighborsClassifier
    elif st.session_state.classifier_name == classifier_list[3]:
        classifier = ExplainableBoostingClassifier
    st.session_state.classifier_choice_type = classifier

def init_classifier_params():
    if st.session_state.classifier_name == classifier_list[0]:
        init_logisticregression_params()
    if st.session_state.classifier_name == classifier_list[1]:
        init_randomforest_params()
    if st.session_state.classifier_name == classifier_list[2]:
        init_KNeighborsClassifier_params()


st.title("⚙️ Wybierz klasyfikator")

if st.session_state.data_loaded:
    load_value('classifier_name')
    st.selectbox("Klasyfikator:", classifier_list, key="_classifier_name", on_change=store_value, args=['classifier_name'])
    if st.session_state.classifier is not None:
        st.write(f"Wybrano: **{st.session_state.classifier}**")
    init_classifier_params()
    init_classifier()
else:
    st.warning("⚠️ Najpierw wybierz dane w zakładce 'Wybór danych'.")