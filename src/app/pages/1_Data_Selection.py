import pandas as pd
import streamlit as st
from utils.session_state import load_value, store_value
from utils.data_loading import (BUILTIN_DATASETS, builtin_feature_types, infer_feature_types, load_builtin_dataset,
                                split_features_and_label)

def mark_all_on():
    for key in st.session_state.keys():
        if key.startswith("is_on_"):
            st.session_state[key] = True


def show_dane_wejsciowe():
    if 'classifier_instance' in st.session_state:
        del st.session_state['classifier_instance']
    if 'classifier_name' in st.session_state:
        del st.session_state['classifier_name']
    load_value('data_loaded')
    if st.session_state.data_loaded:
        st.subheader("📈 Input data")
        st.dataframe(st.session_state.X)
        st.subheader("🏷️ Labels")
        if 'target_names' in st.session_state.data:
            st.dataframe(st.session_state.y.map(lambda x: st.session_state.data.target_names[x]))
        else:
            st.dataframe(st.session_state.y)
    if st.session_state.feature_types is not None:
        if st.button('Mark all as available for evaluation'):
            mark_all_on()
        st.subheader("Column types")
        updated_types = []
        for i, row in st.session_state.feature_types.iterrows():
            key_type_select = f"type_select_{i}"
            type_options = ["continuous", "nominal", "ordinal"]
            # The initial value goes through session state, not `index`: since streamlit 1.5x a keyed widget keeps its
            # browser-side value when only `index` changes, so a new dataset would inherit the previous dataset's types.
            if key_type_select not in st.session_state:
                st.session_state[key_type_select] = row['Type']
            selected_type = st.selectbox(
                row['Column Name'],
                options= type_options,
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
    if 'selected_feature_types' in st.session_state:
        del st.session_state['selected_feature_types']
    if 'classifier_instance' in st.session_state:
        del st.session_state['classifier_instance']
    if 'classifier_params' in st.session_state:
        del st.session_state['classifier_params']
    if 'data_loaded' in st.session_state:
        st.session_state.data_loaded = False

    if 'adjacency_matrix' in st.session_state:
        del st.session_state['adjacency_matrix']
    if 'causal_order' in st.session_state:
        del st.session_state['causal_order']
    if 'plausibility_loss_on' in st.session_state:
        del st.session_state['plausibility_loss_on']

    if 'causal_order_features' in st.session_state:
        del st.session_state['causal_order_features']
    if 'pbounds' in st.session_state:
        del st.session_state['pbounds']
    if 'classifier_instance' in st.session_state:
        del st.session_state["classifier_instance"]
    if 'desired_class' in st.session_state:
        del st.session_state['desired_class']
    if 'num_bins' in st.session_state:
        del st.session_state['num_bins']
    if 'bayesian_model' in st.session_state:
        del st.session_state['bayesian_model']

    pbounds_is_masked_to_delete = [key for key in st.session_state.keys() if key.endswith('_pbounds') or key.endswith('_is_masked')]
    for key in pbounds_is_masked_to_delete:
        del st.session_state[key]

def data_source_changed(data_source_name):
    clear_features_session_state()
    store_value(data_source_name)

init_session_state()
st.title("🔍 Select data")

load_value('source'); st.radio("Data source:", ["Builtin", "CSV file"], key="_source", on_change=data_source_changed, args=['source'])
if st.session_state.source == "Builtin":
    st.session_state.label_column = None
    st.session_state.drop_na = False
    st.session_state.data_loaded = False
    load_value('data_source_name')
    st.radio("Dataset:", BUILTIN_DATASETS,
                                             key='_data_source_name', on_change=data_source_changed, args=['data_source_name'])
    if st.session_state.data_source_name is not None:
        data = load_builtin_dataset(st.session_state.data_source_name)
        X = data.data
        y = data.target

        try:
            st.session_state.feature_types = builtin_feature_types(st.session_state.data_source_name, data)
            save_input_data(X, y, st.session_state.data_source_name, data)
            st.session_state.data_loaded = True
        except ValueError as e:
            st.error(e)


elif st.session_state.source == "CSV file":  # pragma: no cover - st.file_uploader cannot be driven by AppTest
    st.info(
        "Please preprocess your data to meet the classifier's fit() requirements. "
        "Without preprocessing, fitting may fail or be in inefficient depending on the classifier.")
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
            load_value('drop_na', False)
            if st.checkbox("Drop NaN", key="_drop_na", on_change=store_value, args=['drop_na']):
                data.dropna(inplace=True)
            try:
                X, y = split_features_and_label(data, st.session_state.label_column)
                st.session_state.feature_types = infer_feature_types(data_instances=data, feature_names=data.columns)
                st.session_state.data_loaded = True
            except ValueError as e:
                st.error(e)


    save_input_data(X, y, None, data)

show_dane_wejsciowe()