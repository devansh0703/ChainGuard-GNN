import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
nodes_df = pd.read_csv('nodes_with_bg_degree.csv')
edges_df = pd.read_csv('edges.csv')

# Map clId to indices
clids = nodes_df['clId'].unique()
clid_to_idx = {clid: idx for idx, clid in enumerate(clids)}
idx_to_clid = {idx: clid for clid, idx in clid_to_idx.items()}

# Features
feat_cols = [col for col in nodes_df.columns if col.startswith('feat#')]
feat_cols.append('background_degree')
X = nodes_df[feat_cols].values

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = torch.tensor(X, dtype=torch.float)

# Labels
le = LabelEncoder()
y = torch.tensor(le.fit_transform(nodes_df['ccLabel']), dtype=torch.long)

# Edges
edge_index = []
for _, row in edges_df.iterrows():
    if row['clId1'] in clid_to_idx and row['clId2'] in clid_to_idx:
        edge_index.append([clid_to_idx[row['clId1']], clid_to_idx[row['clId2']]])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Create Data
data = Data(x=X, edge_index=edge_index, y=y)

# Oversample minority class
minority_mask = (data.y == 1)
minority_indices = minority_mask.nonzero().squeeze()
num_minority = minority_indices.size(0)
oversample_factor = 10
if num_minority > 0:
    new_x = torch.cat([data.x, data.x[minority_indices].repeat(oversample_factor - 1, 1)], dim=0)
    new_y = torch.cat([data.y, data.y[minority_indices].repeat(oversample_factor - 1)], dim=0)
    num_nodes = data.x.size(0)
    new_nodes = num_minority * (oversample_factor - 1)
    new_indices = torch.arange(num_nodes, num_nodes + new_nodes)
    self_loops = torch.stack([new_indices, new_indices], dim=0)
    new_edge_index = torch.cat([data.edge_index, self_loops], dim=1)
    data.x = new_x
    data.y = new_y
    data.edge_index = new_edge_index

# Save
torch.save(data, 'graph_data.pt')
torch.save({'clid_to_idx': clid_to_idx, 'idx_to_clid': idx_to_clid, 'le': le, 'scaler': scaler}, 'mappings.pt')

print("Data prepared with oversampling.")