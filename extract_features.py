import pandas as pd
import os

# Load the main nodes to get clIds
nodes_df = pd.read_csv('nodes.csv')
clids = set(nodes_df['clId'])

# Now, read background_nodes in chunks and filter
chunk_size = 100000  # Adjust based on RAM
features_list = []

for chunk in pd.read_csv('background_nodes.csv', chunksize=chunk_size):
    filtered = chunk[chunk['clId'].isin(clids)]
    features_list.append(filtered)

# Concatenate
features_df = pd.concat(features_list, ignore_index=True)

# Merge with nodes_df to add ccId
result_df = pd.merge(nodes_df, features_df, on='clId', how='left')

# Save to csv
result_df.to_csv('nodes_with_features.csv', index=False)

print("Features extracted and saved.")