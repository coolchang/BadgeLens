import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer

# Load the user dataset
user_file_path = "data/user_dataset_10000.csv"
user_df = pd.read_csv(user_file_path)

# 1️⃣ Label Encoding (user_id)
user_label_encoder = LabelEncoder()
user_df['user_id_encoded'] = user_label_encoder.fit_transform(user_df['user_id'])

# 2️⃣ One-Hot Encoding (goal)
goal_onehot_encoder = OneHotEncoder(sparse_output=False)
goal_encoded = goal_onehot_encoder.fit_transform(user_df[['goal']])

# 3️⃣ Multi-Hot Encoding (skills)
skills_list = user_df['skills'].apply(lambda x: eval(x) if isinstance(x, str) else [])
mlb_skills = MultiLabelBinarizer()
skills_encoded = mlb_skills.fit_transform(skills_list)

# 4️⃣ Multi-Hot Encoding (goal_required_skills)
goal_skills_list = user_df['goal_required_skills'].apply(lambda x: eval(x) if isinstance(x, str) else [])
mlb_goal_skills = MultiLabelBinarizer()
goal_skills_encoded = mlb_goal_skills.fit_transform(goal_skills_list)

# 5️⃣ Multi-Hot Encoding (goal_matching_badges)
goal_badges_list = user_df['goal_matching_badges'].apply(lambda x: eval(x) if isinstance(x, str) else [])
mlb_goal_badges = MultiLabelBinarizer()
goal_badges_encoded = mlb_goal_badges.fit_transform(goal_badges_list)

# 6️⃣ Multi-Hot Encoding (acquired_badges)
acquired_badges_list = user_df['acquired_badges'].apply(lambda x: eval(x) if isinstance(x, str) else [])
mlb_acquired_badges = MultiLabelBinarizer()
acquired_badges_encoded = mlb_acquired_badges.fit_transform(acquired_badges_list)

# 7️⃣ Combine all features into a single dataset
user_feature_matrix = np.hstack([
    goal_encoded,
    skills_encoded,
    goal_skills_encoded,
    goal_badges_encoded,
    acquired_badges_encoded
])

# 8️⃣ Convert to DataFrame
vectorized_user_df = pd.DataFrame(user_feature_matrix)

# 9️⃣ Save vectorized dataset
vectorized_user_df.to_csv("vectorized_user_dataset.csv", index=False, encoding='utf-8')

# 🔟 Save metadata

metadata_filename = "vectorized_user_dataset_metadata.json"  # 백터화된 데이터세트와 연계된 이름으로 메타데이터 저장

metadata = {
    "user_id": "Label Encoded user ID",
    "goal_features": list(goal_onehot_encoder.get_feature_names_out(['goal'])),
    "skills_features": list(mlb_skills.classes_),
    "goal_required_skills_features": list(mlb_goal_skills.classes_),
    "goal_matching_badges_features": list(mlb_goal_badges.classes_),
    "acquired_badges_features": list(mlb_acquired_badges.classes_)
}

with open(metadata_filename, "w") as f:
    json.dump(metadata, f, indent=4)

print("Vectorization complete. Saved as vectorized_user_dataset.csv and vectorized_user_dataset_metadata.json")
