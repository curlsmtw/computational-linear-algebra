import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

  
df = pd.read_csv("projects/project-one/scores.csv", header=None)

labels = df.iloc[0, 1:].tolist()  # Gets the team names from the first row, then Removes headers to keep only numerical data
df = df.iloc[1:, 1:]
df.index = labels  # Sets row index as team names
df.columns = labels  # Sets column names as team names


matrix = df.to_numpy(dtype=int)

# Constructs the adjacency matrix (victory-based)
n = len(matrix)
adj_matrix_victory = np.zeros((n, n), dtype=int)  # Initializes adjacency matrix

# Fills the adjacency matrix (only for wins)
for i in range(n):
    for j in range(n):
        if matrix[i, j] > matrix[j, i]: 
            adj_matrix_victory[i, j] = matrix[i, j] - matrix[j, i]  # Stores score difference

adj_matrix_df = pd.DataFrame(adj_matrix_victory, index=labels, columns=labels)

# Prints the adjacency matrix
print("Adjacency Matrix (Victory-Based):")
print(adj_matrix_df)

is_symmetric = np.array_equal(adj_matrix_victory, adj_matrix_victory.T)

#Prints justification for symmetry
if is_symmetric:
    print("\nThe adjacency matrix is symmetric.")
else:
    print("\nThe adjacency matrix is NOT symmetric.")
    print("Justification: A directed edge exists only when a team wins. If Team A defeats Team B, an edge (A → B) exists, but the reverse edge (B → A) does not exist unless B also wins another game against A. Therefore, A ≠ A^T, making it asymmetric.")

# Creates a directed graph based on victories
G = nx.DiGraph()

for i in range(n):
    for j in range(n):
        if adj_matrix_victory[i, j] > 0:  # Adds edge only for wins
            G.add_edge(labels[i], labels[j], weight=adj_matrix_victory[i, j])

# Computes win-loss records
team_wins = {labels[i]: np.count_nonzero(adj_matrix_victory[i, :]) for i in range(n)}
team_losses = {labels[i]: np.count_nonzero(adj_matrix_victory[:, i]) for i in range(n)}

# Computes Win-Loss Ratio
win_loss_ratio = {
    team: team_wins[team] / (team_wins[team] + team_losses[team]) if (team_wins[team] + team_losses[team]) > 0 else 0
    for team in labels
}

# Sorts by Win-Loss Ratio (Higher is better)
sorted_win_loss = sorted(win_loss_ratio.items(), key=lambda x: x[1], reverse=True)

# Converts to DataFrame for better visualization
ranking_win_loss_df = pd.DataFrame(sorted_win_loss, columns=["Team", "Win-Loss Ratio"])

# Prints Win-Loss Rankings
print("\nWin-Loss Ratio-Based Team Rankings:")
print(ranking_win_loss_df)

# Visualizes the graph
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)  # Layout for visualization
nx.draw(G, pos, with_labels=True, node_color='lightblue', font_weight='bold', arrowsize=20)

# Adds edge labels (score differences)
edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

plt.title("Directed Graph of Team Victories (Weighted by Score Difference)")
plt.show()