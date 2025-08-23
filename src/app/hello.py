import streamlit as st
import lingam
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx

def train_causal_model(X):
    causal_model = lingam.DirectLiNGAM()
    causal_model.fit(X)
    return causal_model

def sample_data():
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Title
st.title("Feature Relationship Editor and Visualizer")

X_train, X_test, y_train, y_test = sample_data()
casual_model = train_causal_model(X_train)
casual_order = casual_model.causal_order_
adjacency_matrix = casual_model.adjacency_matrix_

st.subheader("Edit Adjacency Matrix")
edited_matrix = st.data_editor(adjacency_matrix, num_rows="dynamic")


fig, ax = plt.subplots()

G = nx.DiGraph(edited_matrix)
pos = nx.spring_layout(G, k=15)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)

# Display edge weights
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

st.pyplot(fig)

