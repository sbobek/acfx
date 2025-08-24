# streamlit_app: 📂 Data Upload

import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype, is_categorical_dtype
from sklearn.datasets import load_iris, load_wine, load_diabetes
import streamlit as st
from utils.session_state import load_value, store_value

def show_dane_wejsciowe():
    load_value('data_loaded')
    if st.session_state.data_loaded:
        st.subheader("📈 Input data")
        st.dataframe(st.session_state.X)
        st.subheader("🏷️ Labels")
        st.dataframe(st.session_state.y)
    if st.session_state.feature_types is not None:
        st.subheader("Typy kolumn")
        updated_types = []
        for i, row in st.session_state.feature_types.iterrows():
            key_type_select = f"type_select_{i}"
            selected_type = st.selectbox(
                row['Column Name'],
                options=["continuous", "nominal", "ordinal"],
                index=0 if row['Type'] == "continuous" else 1,
                key=key_type_select
            )
            key_is_on_select = f"is_on_{i}"
            if st.session_state.label_column is not None and row["Column Name"] == st.session_state.label_column:
                st.markdown(f"**{row['Column Name']}** is marked as target feature")
                updated_types.append({
                    "Column Name": row["Column Name"],
                    "Type": selected_type,
                    "is_on": True
                })
            else:
                if key_is_on_select not in st.session_state:
                    st.session_state[key_is_on_select] = row['is_on']
                is_on = st.checkbox(label=f"**{row['Column Name']}** is available for evaluation", key=key_is_on_select)
                updated_types.append({
                    "Column Name": row["Column Name"],
                    "Type": selected_type,
                    "is_on": is_on
                })
        st.session_state.feature_types = pd.DataFrame(updated_types)



def save_input_data(X, y, data_source_name, data):
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.data_source_name = data_source_name
    st.session_state.data = data

def set_feature_types(data_instances, feature_names):
    feature_types = []
    for col in feature_names:
        if is_numeric_dtype(data_instances[col]):
            feature_types.append({'Column Name': col, 'Type': 'continuous', "is_on": True})
        else:
            feature_types.append({'Column Name': col, 'Type': 'nominal', "is_on": True})
    st.session_state.feature_types = pd.DataFrame(feature_types)

def init_session_state():
    if "feature_types" not in st.session_state:
        st.session_state.feature_types = None
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

def clear_features_session_state():
    if st.session_state.feature_types is not None:
        for i, row in st.session_state.feature_types.iterrows():
            key_type_select = f"type_select_{i}"
            key_is_on_select = f"is_on_{i}"
            if key_is_on_select in st.session_state:
                del st.session_state[key_is_on_select]
            if key_type_select in st.session_state:
                del st.session_state[key_type_select]
        st.session_state.feature_types = None
    delete_future_session_state()

def delete_future_session_state():
    if 'selected_X' in st.session_state:
        del st.session_state['selected_X']
    if 'classifier_instance' in st.session_state:
        del st.session_state['classifier_instance']
    if 'classifier_params' in st.session_state:
        del st.session_state['classifier_params']
    if 'data_loaded' in st.session_state:
        st.session_state.data_loaded = False

def data_source_changed(data_source_name):
    clear_features_session_state()
    store_value(data_source_name)

init_session_state()
st.title("🔍 Select data")

load_value('source'); st.radio("Data source:", ["Builtin", "Plik CSV"], key="_source", on_change=data_source_changed, args=['source'])
if st.session_state.source == "Builtin":
    st.session_state.label_column = None
    st.session_state.data_loaded = False
    load_value('data_source_name'); st.radio("Dataset:", ['iris', 'wine', 'diabetes'],
                                             key='_data_source_name', on_change=data_source_changed, args=['data_source_name'])
    data = None
    if st.session_state.data_source_name == 'iris':
        data = load_iris(as_frame=True)
    elif st.session_state.data_source_name == 'wine':
        data = load_wine(as_frame=True)
    elif st.session_state.data_source_name == 'diabetes':
        data = load_diabetes(as_frame=True)

    if st.session_state.data_source_name is not None:
        X = data.data
        y = data.target
        save_input_data(X, y, st.session_state.data_source_name, data)
        st.session_state.data_loaded = True

        set_feature_types(data_instances=data.data, feature_names=data.feature_names)


elif st.session_state.source == "CSV file":
    uploaded_file = st.file_uploader("Load CSV", type=["csv"], on_change=clear_features_session_state)
    if not st.session_state.data_loaded or st.session_state.data_source_name is not None:
        data = None
        X = None
        y = None
        st.session_state.feature_types = None
    else:
        data = st.session_state.data
        X = st.session_state.X
        y = st.session_state.y

    if uploaded_file is not None:
        st.session_state.data_loaded = False
        data = pd.read_csv(uploaded_file).dropna(axis=1, how='all')
        if data.columns is None or len(list(data.columns)) == 0:
            raise ValueError("Csv file must have columns")
        if st.session_state.label_column not in data.columns:
            st.session_state.label_column = None
        load_value('label_column'); st.selectbox("Select label feature column:", data.columns, key="_label_column", on_change=store_value, args=['label_column'])
        if st.session_state.label_column is not None:
            X = data.drop(columns=[st.session_state.label_column])
            y = data[st.session_state.label_column]
            set_feature_types(data_instances=data, feature_names=data.columns)
            st.session_state.data_loaded = True

    save_input_data(X, y, None, data)


show_dane_wejsciowe()