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
    classifier = None
    params = st.session_state.classifier_params
    if params is not None:
        if st.session_state.classifier_name == classifier_list[0]:
            classifier = LogisticRegression(**params)
        elif st.session_state.classifier_name == classifier_list[1]:
            classifier = RandomForestClassifier(**params)
        elif st.session_state.classifier_name == classifier_list[2]:
            classifier = KNeighborsClassifier(**params)
        elif st.session_state.classifier_name == classifier_list[3]:
            classifier = ExplainableBoostingClassifier(**params)
        else:
            raise IndexError("classifier_name out of range")
        if st.session_state.X is None:
            raise ValueError("Data for classifier fit is not available (st.session_state.X)")
        if st.session_state.y is None:
            raise ValueError("Data for classifier fit is not available (st.session_state.y)")
        classifier = classifier.fit(st.session_state.X,st.session_state.y)

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

st.title("⚙️ Wybierz klasyfikator")

if st.session_state.data_loaded:
    load_value('classifier_name')
    st.selectbox("Klasyfikator:", classifier_list, key="_classifier_name", on_change=store_value, args=['classifier_name'])
    if st.session_state.classifier_name is not None:
        init_classifier_params()

    if st.button("Utwórz klasyfikator"):
        init_classifier()
    if st.button("Resetuj klasyfikator"):
        st.session_state["classifier_instance"] = None
    if st.session_state['classifier_instance'] is not None:
        st.write(f'Klasyfikator jest zainicjalizowany: {type(st.session_state["classifier_instance"])}')
    else:
        st.write('Klasyfikator nie jest zainicjalizowany')
else:
    st.warning("⚠️ Najpierw wybierz dane w zakładce 'Wybór danych'.")