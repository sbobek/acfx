import lingam
import networkx as nx
import numpy as np

from utils.const import ADJACENCY_OPTION_BAYESIAN, ADJACENCY_OPTION_DIRECTLINGAM


def default_adjacency_generator(selected_columns, continuous_cols):
    """DirectLiNGAM handles continuous features only; mixed or purely discrete data needs the Bayesian network."""
    if all(col in continuous_cols for col in selected_columns):
        return ADJACENCY_OPTION_DIRECTLINGAM
    return ADJACENCY_OPTION_BAYESIAN


def train_causal_model(X, continuous_cols):
    """
    Fits DirectLiNGAM on the continuous columns and embeds the result in the full feature space:
    edges of non-continuous features are 0 and those features are placed first in the causal order.
    """
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


def order_indices_to_names(columns, causal_order):
    return [columns[i] for i in causal_order]


def find_order_violations(adjacency_matrix, causal_order):
    """Edges u -> v of the (named) adjacency matrix for which u is not placed after v in causal_order."""
    graph = nx.DiGraph(adjacency_matrix)
    position = {node: i for i, node in enumerate(causal_order)}
    return [(u, v) for u, v in graph.edges() if position[u] <= position[v]]
