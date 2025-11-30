import lingam
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork, DiscreteBayesianNetwork
from sklearn.metrics import mutual_info_score
from streamlit_sortables import sort_items
from utils.features_by_type import get_continuous_cols, get_categorical_indicator, get_all_columns
from utils.session_state import store_value, load_value
from acfx.evaluation.bayesian_model import train_bayesian_model

ADJACENCY_OPTION_DIRECTLINGAM = 'DirectLiNGAM'
ADJACENCY_OPTION_BAYESIAN = 'Discrete Bayesian network'

def reset_adjacency_bayesian():
    # if 'num_bins' in st.session_state:
    #     st.session_state.num_bins = None
    if 'bayesian_model' in st.session_state:
        st.session_state.bayesian_model = None
    store_value('plausibility_loss_on')

def reset_adjacency_lingam():
    if 'adjacency_matrix' in st.session_state:
        del st.session_state.adjacency_matrix
    if 'casual_order' in st.session_state:
        del st.session_state.casual_order
    store_value('plausibility_loss_on')

def lingam_causality_display():
    def generate_adjacency(graph, fig, ax):
        pos = nx.spring_layout(graph, k=15)
        nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
        # Display edge weights
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax)
        st.pyplot(fig)

    def log_not_dag(graph: nx.DiGraph):
        st.error('Input adjacency matrix is not a directed acyclic graph!\nFound cycle(s):')
        for i, cycle in enumerate(list(nx.simple_cycles(graph))):
            st.error(f"\nCycle {i}: {' → '.join(cycle)}")

    def validate_adjacency_order():
        graph = nx.DiGraph(st.session_state.adjacency_matrix)
        position = {node: i for i, node in enumerate(st.session_state.casual_order)}
        violations = []
        for u, v in graph.edges():
            if position[u] <= position[v]:
                violations.append((u, v))
        if violations:
            st.error("Adjacency order is NOT valid. Violations found:")
            for u, v in violations:
                st.error(f"Edge {u} → {v} violates the order.")

    def edit_adjacency_order(casual_order):
        if 'casual_order' not in st.session_state:
            feature_names = st.session_state.selected_X.columns
            ordered_features = list(map(lambda x: feature_names[x], casual_order))
            st.session_state.casual_order = ordered_features.copy()
        casual_order = sort_items(st.session_state.casual_order, direction="vertical")
        if casual_order != st.session_state.casual_order:
            st.session_state.casual_order = casual_order.copy()
        # st.write("Casual order:", st.session_state.casual_order)

    def train_causal_model(X):
        continuous_cols = get_continuous_cols()
        X_continuous = X[continuous_cols]
        causal_model = lingam.DirectLiNGAM()
        causal_model.fit(X_continuous)
        adjacency_matrix = causal_model.adjacency_matrix_
        causal_order = causal_model.causal_order_

        full_adj_matrix = np.zeros((X.shape[1], X.shape[1]))

        original_indices = [X.columns.get_loc(col) for col in continuous_cols]
        causal_order_mapped = [original_indices[i] for i in causal_order]

        for row in causal_order:
            for col in causal_order:
                full_adj_matrix[original_indices[row], original_indices[col]] = adjacency_matrix[row, col]

        missing_indices = list(set(range(X.shape[1])) - set(causal_order_mapped))
        full_causal_order = sorted(missing_indices) + causal_order_mapped

        return causal_model, full_adj_matrix, full_causal_order

    reset_adjacency_bayesian()
    causal_model, full_adj_matrix, causal_order_mapped = train_causal_model(st.session_state.selected_X)

    st.subheader("Edit Adjacency Matrix")
    adjacency_matrix_with_features = pd.DataFrame(full_adj_matrix, columns=st.session_state.selected_X.columns,
                                                  index=st.session_state.selected_X.columns)
    load_value('adjacency_matrix', adjacency_matrix_with_features)

    edited_adjacency_matrix = st.data_editor(adjacency_matrix_with_features)
    if not np.array_equal(edited_adjacency_matrix, st.session_state.adjacency_matrix):
        st.session_state.adjacency_matrix = edited_adjacency_matrix
    if len(st.session_state.selected_X.columns) > 4:
        st.info(
            f"Generating Adjacency Matrix might be not too pretty for large amount of features. You have {len(st.session_state.selected_X.columns)}.")
    if st.button("🔄 Generate adjacency graph"):
        fig, ax = plt.subplots()
        G = nx.DiGraph(st.session_state.adjacency_matrix)
        if not nx.is_directed_acyclic_graph(G):
            log_not_dag(G)
        else:
            generate_adjacency(G, fig, ax)

    st.subheader("Edit Casual Order")
    if len(get_continuous_cols()) > 0:
        st.info(f"For categorical features, the adjacency is set to 0. "
                f"These features are skipped for causality calculation as {ADJACENCY_OPTION_DIRECTLINGAM} doesn't support categorical features. "
                f"Order of categorical features is irrelevant for evaluation. "
                f"We suggest selecting 'Discrete Bayesian network' for adjacency generation "
                f"as it allows to include both categorical and continuous features.")
    edit_adjacency_order(causal_order_mapped)
    validate_adjacency_order()

def bayesian_causality_display():
    def plot_cpd_table(bayesian_model: DiscreteBayesianNetwork):
        st.title("Bayesian Network Visualization")

        nodes = list(bayesian_model.nodes)
        if len(nodes) == 0:
            st.warning(f'No nodes found in {ADJACENCY_OPTION_BAYESIAN}.')
            return

        edges = list(bayesian_model.edges)
        if len(edges) == 0:
            st.warning(f'No edges found in {ADJACENCY_OPTION_BAYESIAN}. Structure of the DAG indicates no causality between nodes.')
            return

        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        fig, ax = plt.subplots(figsize=(6, 4))
        pos = nx.spring_layout(G)
        nx.draw(G, pos,
                with_labels=True,
                node_size=2000,
                node_color='lightblue',
                font_size=12,
                font_weight='bold',
                arrows=True,
                ax=ax)
        ax.set_title("Bayesian Network Graph")

        st.pyplot(fig)
        num_bins = st.session_state['num_bins']
        st.info(f'All continuous features were discretized to {num_bins} bins.')
        st.subheader("Conditional Probability Distributions Table:")
        for cpd in bayesian_model.get_cpds():
            if not cpd:
                continue
            try:
                print(cpd)
                df = cpd.to_dataframe()
            except:
                continue
            # todo: multidimentional cpd. I only tested for now when there is only one parent
            st.write(f"**CPD of {cpd.variable} (column-valued):**")
            st.write(f"**Conditionality on {cpd.variables[-1]} (row-valued):**")
            st.table(df)

    reset_adjacency_lingam()
    categorical_indicator = get_categorical_indicator()
    load_value('num_bins', 5)
    st.slider("Number of bins", min_value=3, max_value=20, step=1, key='_num_bins', on_change=store_value, args=['num_bins'])
    bayesian_model = train_bayesian_model(st.session_state.selected_X, categorical_indicator, st.session_state.num_bins)
    load_value('bayesian_model', bayesian_model)
    plot_cpd_table(bayesian_model)

def get_default_adjacency_generator_name():
    if len(get_continuous_cols()) > 0:
        return ADJACENCY_OPTION_BAYESIAN
    else:
        return ADJACENCY_OPTION_DIRECTLINGAM

if 'classifier_instance' not in st.session_state or 'selected_X' not in st.session_state:
    st.warning("⚠️ Start by initializing classifier in 'Classifier selection'")
else:
    load_value('plausibility_loss_on', False)
    if st.checkbox(label="I want plausibility loss to be calculated.",
                   key="_plausibility_loss_on", on_change=reset_adjacency_lingam):
        if st.session_state.selected_X is None:
            raise ValueError("selected_X must be initialized in session state")
        if 'casual_order_features' not in st.session_state:
            st.session_state['casual_order_features'] = None

        load_value('adjacency_generator_name', get_default_adjacency_generator_name())
        st.selectbox("Adjacency generator:", [ADJACENCY_OPTION_DIRECTLINGAM, ADJACENCY_OPTION_BAYESIAN],
                     key="_adjacency_generator_name", on_change=store_value,
                     args=['adjacency_generator_name'])
        if st.session_state.adjacency_generator_name == ADJACENCY_OPTION_DIRECTLINGAM:
            st.title(F"Feature Relationship (generated by {ADJACENCY_OPTION_DIRECTLINGAM})")
            st.text("see more about the method: https://pypi.org/project/lingam/")
            lingam_causality_display()
        elif st.session_state.adjacency_generator_name == ADJACENCY_OPTION_BAYESIAN:
            st.title(f"Feature Relationship (generated by {ADJACENCY_OPTION_DIRECTLINGAM} + pgmpy)")
            st.text("see more (lingam): https://pypi.org/project/lingam/ ")
            st.text("see more (pgmpy): https://pgmpy.org/models/bayesiannetwork.html ")
            bayesian_causality_display()
