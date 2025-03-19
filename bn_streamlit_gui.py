import streamlit as st
import networkx as nx
from pyvis.network import Network
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Sample Bayesian Network
model = BayesianNetwork([
    ('seasons', 'temperature'),
    ('seasons', 'wind'),
    ('seasons', 'rain'),
    ('seasons', 'sunlight'),
    ('temperature', 'walk'),
    ('rain', 'mood'),
    ('sunlight', 'mood'),
    ('wind', 'walk'),
    ('mood', 'walk')
])

# Placeholder for dataset loading (replace with actual data)
data = pd.DataFrame({
    'seasons': ['Winter', 'Summer', 'Spring', 'Autumn'] * 50,
    'temperature': ['cold', 'hot', 'mild', 'cool'] * 50,
    'rain': [1, 0, 1, 0] * 50,
    'sunlight': [0, 1, 0, 1] * 50,
    'wind': ['breeze', 'strong', 'calm', 'gentle'] * 50,
    'mood': ['happy', 'sad', 'neutral', 'happy'] * 50,
    'walk': [1, 0, 1, 0] * 50
})

# Train model with Maximum Likelihood Estimation
model.fit(data, estimator=MaximumLikelihoodEstimator)
infer = VariableElimination(model)

def draw_graph():
    G = nx.DiGraph()
    G.add_edges_from(model.edges())
    net = Network(height='400px', width='100%', directed=True)
    for node in G.nodes():
        net.add_node(node, label=node)
    for edge in G.edges():
        net.add_edge(edge[0], edge[1])
    net.show("graph.html")
    return "graph.html"

# Streamlit UI
st.title("Interactive Bayesian Network")

# Show Bayesian Network
st.subheader("Bayesian Network Structure")
html_file = draw_graph()
st.components.v1.html(open(html_file, 'r', encoding='utf-8').read(), height=450)

# Select node to visualize CPD
node = st.selectbox("Select a node to view CPD:", model.nodes())
cpd = model.get_cpds(node)
cpd_df = pd.DataFrame(cpd.values.reshape(-1, len(cpd.state_names[node])), columns=cpd.state_names[node])
st.write("### Conditional Probability Distribution")
st.dataframe(cpd_df)

# Visualize CPD as Heatmap
plt.figure(figsize=(6, 3))
sns.heatmap(cpd_df, annot=True, cmap='coolwarm', fmt='.2f')
st.pyplot(plt)

# User input for node states
st.subheader("Update Node States")
evidence = {}
for node in model.nodes():
    state = st.selectbox(f"Select state for {node}", [None] + list(cpd.state_names[node]))
    if state:
        evidence[node] = state

if st.button("Update Probabilities"):
    result = infer.map_query(variables=["walk"], evidence=evidence)
    st.write("### Updated Prediction:", result)
