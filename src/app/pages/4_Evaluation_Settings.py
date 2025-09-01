import streamlit as st
from utils.session_state import store_value, load_value
from utils.key_helper import calc_pbounds, get_pbounds_key, get_masked_feature_key


if 'classifier_name' not in st.session_state or st.session_state.classifier_name is None:
    st.warning("⚠️ Start by initializing classifier in 'Classifier selection'")
elif 'plausibility_loss_on' not in st.session_state or st.session_state.plausibility_loss_on is None:
    st.warning("⚠️ Start by visiting 'Adjacency Generation'")
elif st.session_state.plausibility_loss_on \
      and ('adjacency_matrix' not in st.session_state or st.session_state.adjacency_matrix is None):
    st.warning("⚠️ Start by initializing adjacency matrix in 'Adjacency Generation'")
else:
    if st.session_state.selected_X is None:
        raise ValueError("selected_X must be initialized in session state")
    categorical_indicator = categorical_columns = st.session_state.feature_types[
        st.session_state.feature_types["Type"] == "nominal"]["Column Name"].tolist()
    initial_pbounds = calc_pbounds(st.session_state.selected_X, categorical_indicator)
    load_value('pbounds', initial_pbounds)
    st.subheader("The bounds for each feature to search over")
    for feature_name, interval in st.session_state.pbounds.items():
        min = interval[0]
        max = interval[1]
        pbounds_key = get_pbounds_key(feature_name)
        load_value(pbounds_key, (min, max))
        start,end=st.slider(
            feature_name,
            min_value=float(min - 3*abs(max-min)),
            max_value=float(max + 3*abs(max-min)),
            key=f"_{pbounds_key}",
            on_change=store_value,
            args= [pbounds_key]
        )
        masked_feature_key = get_masked_feature_key(feature_name)
        load_value(masked_feature_key, False)
        st.checkbox(label=f"Mark **{feature_name}** as masked feature", key=f'_{masked_feature_key}',
                    on_change=store_value, args=[masked_feature_key])

    st.subheader("The weights for loss function components")
    if st.session_state.plausibility_loss_on:
        load_value('plausibility_loss', 0.5)
        st.slider(label="Plausibility loss weight", key="_plausibility_loss",
                  min_value=0.0, max_value=1., step=0.05, on_change=store_value, args=['plausibility_loss'])
    else:
        st.session_state.plausibility_loss = 0
    load_value('proximity_weight', 0.5)
    st.slider(label="Proximity loss weight", key="_proximity_weight",
              min_value=0.0, max_value=1., step=0.05, on_change=store_value, args=['proximity_weight'])

    load_value('diversity_weight', 0.5)
    st.slider(label="Diversity loss weight", key="_diversity_weight",
              min_value=0.0, max_value=1., step=0.05, on_change=store_value, args=['diversity_weight'])

    load_value('sparsity_weight', 0.5)
    st.slider(label="Sparsity loss weight", key="_sparsity_weight",
              min_value=0.0, max_value=1., step=0.05, on_change=store_value, args=['sparsity_weight'])

