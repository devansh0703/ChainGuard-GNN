import pandas as pd
from collections import defaultdict

# Get clIds
nodes_df = pd.read_csv('nodes_labeled.csv')
clids = set(nodes_df['clId'])
degree_dict = defaultdict(int)

chunk_size = 100000
for chunk in pd.read_csv('background_edges.csv', chunksize=chunk_size):
    for _, row in chunk.iterrows():
        clid1 = row['clId1']
        clid2 = row['clId2']
        if clid1 in clids:
            degree_dict[clid1] += 1
        if clid2 in clids:
            degree_dict[clid2] += 1
nodes_df['background_degree'] = nodes_df['clId'].map(degree_dict).fillna(0)

# Save
nodes_df.to_csv('nodes_with_bg_degree.csv', index=False)

print("Background degrees added.")