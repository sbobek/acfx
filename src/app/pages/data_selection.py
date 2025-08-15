import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_diabetes
import streamlit as st
from utils.session_state import load_value, store_value
from utils.const import SELECT_BUTTON

def show_dane_wejsciowe():
    if st.session_state.data_loaded:
        st.subheader("📈 Dane wejściowe")
        st.dataframe(st.session_state.X)
        st.subheader("🏷️ Etykiety")
        st.dataframe(st.session_state.y)

def save_dane_wejsciowe(X,y,data_source_name, data):
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.data_source_name = data_source_name
    st.session_state.data = data


st.title("🔍 Wybierz dane")

load_value('source'); st.radio("Źródło danych:", ["Wbudowane", "Plik CSV"], key="_source", on_change=store_value, args=['source'])
if st.session_state.source == "Wbudowane":
    st.session_state.label_column = None
    st.session_state.data_loaded = False
    load_value('data_source_name'); st.radio("Zbiór danych:", ['iris', 'wine', 'diabetes'],
             key='_data_source_name', on_change=store_value, args=['data_source_name'])
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
        save_dane_wejsciowe(X, y, st.session_state.data_source_name, data)
        st.session_state.data_loaded = True

elif st.session_state.source == "Plik CSV":
    uploaded_file = st.file_uploader("Załaduj plik CSV", type=["csv"])
    if not st.session_state.data_loaded or st.session_state.data_source_name is not None:
        data = None
        X = None
        y = None
    else:
        data = st.session_state.data
        X = st.session_state.X
        y = st.session_state.y

    if uploaded_file is not None:
        st.session_state.data_loaded = False
        data = pd.read_csv(uploaded_file)
        if st.session_state.label_column not in data.columns:
            st.session_state.label_column = None
        load_value('label_column'); st.selectbox("Wybierz kolumnę z etykietami:", data.columns, key="_label_column", on_change=store_value, args=['label_column'])
        if st.session_state.label_column is not None:
            X = data.drop(columns=[st.session_state.label_column])
            y = data[st.session_state.label_column]
            st.session_state.data_loaded = True

    save_dane_wejsciowe(X, y, None, data)


show_dane_wejsciowe()