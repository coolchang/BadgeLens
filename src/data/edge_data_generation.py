import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load the user and badge dataset
badge_df = pd.read_csv('C:/Users/USER/Documents/Badgelens/data/processed/vectorized_badge_dataset_10.csv')  # Badge dataset
user_df = pd.read_csv('C:/Users/USER/Documents/Badgelens/data/processed/vectorized_user_dataset_10.csv')    # User dataset

# Load user metadata to get goal matching and acquired badges
with open('C:/Users/USER/Documents/Badgelens/data/processed/vectorized_user_dataset_metadata_10.json', 'r') as f:
    user_metadata = json.load(f)

with open('C:/Users/USER/Documents/Badgelens/data/processed/vectorized_badge_dataset_metadata_10.json', 'r') as f:
    badge_metadata = json.load(f)

# 1. Extract goal matching badges from user metadata
goal_matching_badges = user_metadata["goal_matching_badges_features"]

# 2. Extract acquired badges from user metadata
acquired_badges = user_metadata["acquired_badges_features"]

# 3. Create edges based on goal matching badges
edges = []
for user_id in range(len(user_df)):
    for badge_id in goal_matching_badges:
        edges.append((user_id, badge_id, 1))  # Weight 1 for goal matching badge

# 4. Create edges based on acquired badges
for user_id in range(len(user_df)):
    for badge_id in acquired_badges:
        edges.append((user_id, badge_id, 1))  # Weight 1 for acquired badge

# 5. Compute badge similarity using cosine similarity
badge_features = badge_df.values  # Badge feature vectors
cosine_sim = cosine_similarity(badge_features)

# 6. Add edges between similar badges
threshold = 0.8  # Define a threshold for similarity
for i in range(len(badge_features)):
    for j in range(i + 1, len(badge_features)):
        if cosine_sim[i, j] >= threshold:
            edges.append((i, j, cosine_sim[i, j]))  # Create edge based on cosine similarity between badges

# 7. Convert edges to DataFrame for easy visualization
edges_df = pd.DataFrame(edges, columns=['source', 'target', 'weight'])

# Ensure the directory exists for saving the file
output_path = 'C:/Users/USER/Documents/Badgelens/edges_dataset.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save the edges DataFrame to CSV
edges_df.to_csv(output_path, index=False)

# Inform the user that the file was saved successfully
print(f"Edges data saved successfully to {output_path}")
