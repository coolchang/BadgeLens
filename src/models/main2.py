import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load datasets
badge_df = pd.read_csv('data/processed/vectorized_badge_dataset_10.csv')  # Badge dataset
user_df = pd.read_csv('data/processed/vectorized_user_dataset_10.csv')    # User dataset

# Normalize user and badge features
scaler = StandardScaler()
user_features = scaler.fit_transform(user_df.values)  # Normalize user features
badge_features = scaler.fit_transform(badge_df.values)  # Normalize badge features

# Convert to torch tensors
user_features_tensor = torch.tensor(user_features, dtype=torch.float)
badge_features_tensor = torch.tensor(badge_features, dtype=torch.float)

# Concatenate user and badge features
# Treating user and badge features as separate node sets in the graph
all_features = torch.cat([user_features_tensor, badge_features_tensor], dim=0)

# Print to check the dimensions of the concatenated tensor
print(all_features.shape)
