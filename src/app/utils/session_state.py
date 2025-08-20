import streamlit as st

def store_value(key):
    st.session_state[key] = st.session_state["_"+key]
def load_value(key, default_value=None):
    if key in st.session_state:
        st.session_state["_"+key] = st.session_state[key]
    else:
        st.session_state["_"+key] = default_value
        st.session_state[key] = default_value