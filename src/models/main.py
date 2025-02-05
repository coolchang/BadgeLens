import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load datasets
badge_df = pd.read_csv('data/processed/vectorized_badge_dataset_10.csv')  # Badge dataset
user_df = pd.read_csv('data/processed/vectorized_user_dataset_10.csv')    # User dataset
edges_df = pd.read_csv('data/processed/edges_dataset.csv')  # Edge dataset

# 2. Load metadata (goal matching and acquired badges)
with open('data/processed/vectorized_user_dataset_metadata_10.json', 'r') as f:
    user_metadata = json.load(f)

with open('data/processed/vectorized_badge_dataset_metadata_10.json', 'r') as g:
    badge_metadata = json.load(g)

# 3. Prepare the node features
# Normalize node features (for both users and badges)
scaler = StandardScaler()
user_features = scaler.fit_transform(user_df.values)  # Normalize user features
badge_features = scaler.fit_transform(badge_df.values)  # Normalize badge features

# Concatenate user and badge features
all_features = torch.tensor(np.concatenate([user_features, badge_features], axis=0), dtype=torch.float)

# 4. Create edge indices (user-badge relations)
edges = edges_df[['source', 'target']].values
edge_index = torch.tensor(edges.T, dtype=torch.long)

# 5. Create a PyTorch Geometric Data object
data = Data(x=all_features, edge_index=edge_index)

# 6. Build the GNN model
class GNNModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(in_channels, 64)
        self.conv2 = torch_geometric.nn.GCNConv(64, out_channels)
        self.fc = nn.Linear(out_channels, 1)  # Output layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        out = self.fc(x)
        return out

# Initialize the model
model = GNNModel(in_channels=all_features.shape[1], out_channels=16)

# 7. Set up training (loss function, optimizer)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 8. Train the model
def train(model, data, optimizer, criterion, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)  # Assuming labels are provided
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

train(model, data, optimizer, criterion)

# 9. Save the model after training
torch.save(model.state_dict(), 'gnn_model.pth')
