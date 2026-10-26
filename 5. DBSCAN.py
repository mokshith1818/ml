# 5. DBSCAN

# pip install pandas matplotlib networkx scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from io import StringIO
from sklearn.cluster import DBSCAN
from sklearn.neighbors import kneighbors_graph

# Create an inline CSV dataset
csv_data = StringIO("""X,Y
1,1
1.5,2
2,1.5
2,2
3,3
3.5,3
3,4
8,8
8.5,8
9,8.5
9,9
15,15
""")

# Read inline CSV data
data = pd.read_csv(csv_data)
X = data[["X", "Y"]]

# -------------------------------
# 1. Graph-Based Clustering
# -------------------------------

# Create a graph using nearest neighbors
graph = kneighbors_graph(
    X,
    n_neighbors=3,
    mode="connectivity",
    include_self=False
)

G = nx.from_scipy_sparse_array(graph)

# Find connected components
components = list(nx.connected_components(G))

# Assign cluster numbers
graph_labels = [-1] * len(X)

for cluster_no, component in enumerate(components):
    for point in component:
        graph_labels[point] = cluster_no

print("Graph-Based Clustering:")
print(graph_labels)

# Graph-Based Clustering Plot
fig1 = plt.figure(figsize=(7, 5))

fig1.canvas.manager.set_window_title(
    "ML Exp5 - Figure 1 - Graph-Based Clustering (moba)"
)

plt.scatter(
    X["X"],
    X["Y"],
    c=graph_labels,
    cmap="viridis",
    s=100
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Graph-Based Clustering")
plt.grid()
plt.show()

# -------------------------------
# 2. DBSCAN
# -------------------------------

dbscan = DBSCAN(eps=2, min_samples=2)
dbscan_labels = dbscan.fit_predict(X)

print("\nDBSCAN Clustering:")
print(dbscan_labels)

# DBSCAN Plot
fig2 = plt.figure(figsize=(7, 5))

fig2.canvas.manager.set_window_title(
    "ML Exp5 - Figure 2 - DBSCAN Clustering (moba)"
)

plt.scatter(
    X["X"],
    X["Y"],
    c=dbscan_labels,
    cmap="viridis",
    s=100
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("DBSCAN Clustering")
plt.grid()
plt.show()