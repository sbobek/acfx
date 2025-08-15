import streamlit as st

def store_value(key):
    st.session_state[key] = st.session_state["_"+key]
def load_value(key):
    if key in st.session_state:
        st.session_state["_"+key] = st.session_state[key]
    else:
        st.session_state["_"+key] = None
        st.session_state[key] = None

def init_session_state():
    if "X" not in st.session_state:
        st.session_state.X = None
    if "y" not in st.session_state:
        st.session_state.y = None
    if "data" not in st.session_state:
        st.session_state.data = None
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "data_source_name" not in st.session_state:
        st.session_state.data_source_name = None
    if 'label_column' not in st.session_state:
        st.session_state.label_column = None
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None