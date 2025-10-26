import streamlit as st

def get_continuous_cols():
    return _get_cols_by_type("continuous")

def get_nominal_cols():
    return _get_cols_by_type("nominal")

def get_ordinal_cols():
    return _get_cols_by_type("ordinal")

def _get_cols_by_type(type_name: str):
    if 'selected_X' not in st.session_state:
        raise Exception('selected_X not set in session state')
    if 'feature_types' not in st.session_state:
        raise Exception('feature_types types not set in session state')
    feature_types = st.session_state.feature_types
    X = st.session_state.selected_X
    return [col for col in X.columns if
                       col in feature_types.loc[feature_types['Type'] == type_name, 'Column Name'].values]
