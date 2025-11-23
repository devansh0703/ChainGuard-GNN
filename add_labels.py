import pandas as pd

# Load nodes_with_features
nodes_df = pd.read_csv('nodes_with_features.csv')

# Load connected_components
cc_df = pd.read_csv('connected_components.csv')

# Merge on ccId
labeled_df = pd.merge(nodes_df, cc_df, on='ccId', how='left')

# Save
labeled_df.to_csv('nodes_labeled.csv', index=False)

print("Labels added.")