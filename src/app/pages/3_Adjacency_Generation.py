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
from utils.features_by_type import get_continuous_cols, get_categorical_indicator, get_all_columns, get_ordinal_cols,get_nominal_cols
from utils.session_state import store_value, load_value
from acfx.evaluation.bayesian_model import train_bayesian_model
from utils.const import ADJACENCY_OPTION_DIRECTLINGAM,ADJACENCY_OPTION_BAYESIAN
import io
import streamlit.components.v1 as components
from pyvis.network import Network

def reset_adjacency_bayesian():
    if 'bayesian_model' in st.session_state:
        st.session_state.bayesian_model = None
    store_value('plausibility_loss_on')

def reset_adjacency_lingam():
    if 'adjacency_matrix' in st.session_state:
        del st.session_state.adjacency_matrix
    if 'causal_order' in st.session_state:
        del st.session_state.causal_order
    store_value('plausibility_loss_on')


def set_zoom(ax):
    load_value('zoom_factor', 1)
    st.slider(
        "Zoom Level (Adjust to zoom in/out)",
        min_value=0.1,
        max_value=2.0,
        step=0.05, key="_zoom_factor", on_change=store_value, args=['zoom_factor']
    )
    limit = 1.0 * st.session_state.zoom_factor
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

@st.cache_data
def get_graph_data(_G:nx.DiGraph) -> bytes:
    buffer = io.BytesIO()
    nx.write_graphml(_G, buffer)
    graphml_data = buffer.getvalue()
    return graphml_data

def lingam_causality_display():
    import streamlit as st
    import streamlit.components.v1 as components
    from pyvis.network import Network
    import networkx as nx
    import os
    def generate_interactive_graph(graph):
        # 1. Inicjalizacja sieci Pyvis
        # directed=True zachowuje strzałki relacji Bayesowskich
        net = Network(
            height="600px",
            width="100%",
            bgcolor="#222222",
            font_color="white",
            directed=True
        )

        for node in graph.nodes():
            net.add_node(node, label=node, title=node)  # title to tooltip po najechaniu

        for u, v, d in graph.edges(data=True):
            weight = d.get('weight', 1.0)
            label = f"{weight:.2f}"
            net.add_edge(u, v, value=1, label=label, title=label)

        net.toggle_physics(True)

        path = "temp_graph.html"
        net.save_graph(path)

        try:
            with open(path, 'r', encoding='utf-8') as f:
                html_data = f.read()

            st.subheader("Interactive Bayesian Network Structure")
            components.html(html_data, height=650, width=1000, scrolling=True)

        except Exception as e:
            st.error(f"Error rendering graph: {e}")
        finally:
            if os.path.exists(path):
                os.remove(path)

    def log_not_dag(graph: nx.DiGraph):
        st.error('Input adjacency matrix is not a directed acyclic graph!\nFound cycle(s):')
        for i, cycle in enumerate(list(nx.simple_cycles(graph))):
            st.error(f"\nCycle {i}: {' → '.join(cycle)}")

    def validate_adjacency_order():
        graph = nx.DiGraph(st.session_state.adjacency_matrix)
        position = {node: i for i, node in enumerate(st.session_state.causal_order)}
        violations = []
        for u, v in graph.edges():
            if position[u] <= position[v]:
                violations.append((u, v))
        if violations:
            st.error("Adjacency order is NOT valid. Violations found:")
            for u, v in violations:
                st.error(f"Edge {u} → {v} violates the order.")

    def edit_adjacency_order(causal_order):
        if 'causal_order' not in st.session_state:
            feature_names = st.session_state.selected_X.columns
            ordered_features = list(map(lambda x: feature_names[x], causal_order))
            st.session_state.causal_order = ordered_features.copy()
        causal_order = sort_items(st.session_state.causal_order, direction="vertical")
        if causal_order != st.session_state.causal_order:
            st.session_state.causal_order = causal_order.copy()
        # st.write("causal order:", st.session_state.causal_order)

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
    info_graph_interactive()
    if st.checkbox(label="🔄 Generate adjacency graph",
                       key="_generate_graph", on_change=store_value, args=['generate_graph']):
        G = nx.DiGraph(st.session_state.adjacency_matrix)
        if not nx.is_directed_acyclic_graph(G):
            log_not_dag(G)
        else:
            generate_interactive_graph(G)
            st.download_button(
                label="Download Graph as GraphML",
                data=get_graph_data(G),
                file_name="network_export_lingam.graphml",
                mime="application/xml"
            )

    st.subheader("Edit Causal Order")
    if len(get_continuous_cols()) > 0 and (len(get_ordinal_cols()) > 0 or len(get_nominal_cols()) > 0):
        st.info(f"For categorical features, the adjacency is set to 0. "
                f"These features are skipped for causality calculation as {ADJACENCY_OPTION_DIRECTLINGAM} doesn't support categorical features. "
                f"Order of categorical features is irrelevant for evaluation. "
                f"We suggest selecting 'Discrete Bayesian network' for adjacency generation "
                f"as it allows to include both categorical and continuous features.")
    edit_adjacency_order(causal_order_mapped)
    validate_adjacency_order()


def info_graph_interactive():
        st.info(
            f"Note that the generated DAG is interactive (try to zoom in/out or touch/move the nodes).")


def bayesian_causality_display():

    def draw_interactive_bayesian_net(G):
        net = Network(
            height="500px",
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            directed=True,
            cdn_resources='remote'
        )
        net.from_nx(G)

        for node in net.nodes:
            node['color'] = '#ADD8E6'
            node['size'] = 25
            node['font'] = {'size': 12, 'weight': 'bold'}

        net.set_options("""
        var options = {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -30000,
              "centralGravity": 0.3,
              "springLength": 100
            },
            "minVelocity": 0.75
          },
          "interaction": {
            "zoomView": true,
            "dragView": true
          }
        }
        """)

        html_content = net.generate_html()

        st.subheader("Interactive Bayesian Network Graph")
        components.html(html_content, height=550, scrolling=True, width=1000)

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

        info_graph_interactive()
        if st.checkbox(label="🔄 Generate Bayesian Network graph",
                       key="_generate_graph", on_change=store_value, args=['generate_graph']):
            G = nx.DiGraph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)

            draw_interactive_bayesian_net(G)

            st.download_button(
                label="Download Graph as GraphML",
                data=get_graph_data(G),
                file_name="network_export_bayesian.graphml",
                mime="application/xml"
            )

        num_bins = st.session_state['num_bins']
        st.info(f'All continuous features were discretized to {num_bins} bins.')
        st.subheader("Conditional Probability Distributions Table:")
        for cpd in bayesian_model.get_cpds():
            if not cpd:
                continue
            try:
                # print(cpd)
                df = cpd.to_dataframe()
            except:
                continue
            st.write(f"**CPD of {cpd.variable} (column-valued):**")
            st.write(f"**Conditionality on {cpd.get_evidence()} (row-valued):**")
            st.table(df)

    reset_adjacency_lingam()
    categorical_indicator = get_categorical_indicator()
    load_value('num_bins', 5)
    st.slider("Number of bins", min_value=3, max_value=20, step=1, key='_num_bins', on_change=store_value, args=['num_bins'])
    bayesian_model = train_bayesian_model(st.session_state.selected_X, categorical_indicator, st.session_state.num_bins)
    st.session_state.bayesian_model = bayesian_model
    plot_cpd_table(st.session_state.bayesian_model)

def get_default_adjacency_generator_name():
    if len(get_continuous_cols()) > 0:
        return ADJACENCY_OPTION_BAYESIAN
    else:
        return ADJACENCY_OPTION_DIRECTLINGAM

load_value('generate_graph', False)

if 'classifier_instance' not in st.session_state or 'selected_X' not in st.session_state:
    st.warning("⚠️ Start by initializing classifier in 'Classifier selection'")
else:
    load_value('generate_graph', False)
    load_value('plausibility_loss_on', False)
    if st.checkbox(label="I want plausibility loss to be calculated.",
                   key="_plausibility_loss_on", on_change=reset_adjacency_lingam):
        if st.session_state.selected_X is None:
            raise ValueError("selected_X must be initialized in session state")
        if 'causal_order_features' not in st.session_state:
            st.session_state['causal_order_features'] = None

        load_value('adjacency_generator_name', get_default_adjacency_generator_name())
        st.selectbox("Adjacency generator:", [ADJACENCY_OPTION_DIRECTLINGAM, ADJACENCY_OPTION_BAYESIAN],
                     key="_adjacency_generator_name", on_change=store_value,
                     args=['adjacency_generator_name'])
        if st.session_state.adjacency_generator_name == ADJACENCY_OPTION_DIRECTLINGAM:
            st.title(F"Feature Relationship (generated by {ADJACENCY_OPTION_DIRECTLINGAM})")
            st.text("see more (lingam): https://pypi.org/project/lingam/ ")
            lingam_causality_display()
        elif st.session_state.adjacency_generator_name == ADJACENCY_OPTION_BAYESIAN:
            st.title(f"Feature Relationship (generated by {ADJACENCY_OPTION_BAYESIAN} of pgmpy package)")
            st.text("see more (pgmpy): https://pgmpy.org/models/bayesiannetwork.html ")
            bayesian_causality_display()
