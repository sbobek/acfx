import streamlit as st

from utils.session_state import store_value, load_value
from utils.const import SELECT_BUTTON

st.title("⚙️ Wybierz klasyfikator")

if st.session_state.data_loaded:
    load_value('classifier')
    st.selectbox("Klasyfikator:", [
        SELECT_BUTTON,
        "Logistic Regression",
        "Random Forest",
        "SVM",
        "KNN"
    ], key="_classifier", on_change=store_value, args=['classifier'])
    if st.session_state.classifier is not None and st.session_state.classifier != SELECT_BUTTON:
        st.write(f"Wybrano: **{st.session_state.classifier}**")
        st.write("📈 Dane wejściowe:")
        st.dataframe(st.session_state.X)
        st.write("🏷️ Etykiety:")
        st.dataframe(st.session_state.y)
else:
    st.warning("⚠️ Najpierw wybierz dane w zakładce 'Wybór danych'.")