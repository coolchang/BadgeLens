import pandas as pd
import numpy as np
import json  # 🚀 json 모듈 추가

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer



# 데이터 불러오기
df = pd.read_csv("data/badge_dataset_10000.csv")

# 1️⃣ Label Encoding (배지 ID)
label_encoder = LabelEncoder()
df['badge_id_encoded'] = label_encoder.fit_transform(df['badge_id'])

# 2️⃣ TF-IDF (배지 이름, 설명)
tfidf_vectorizer_name = TfidfVectorizer()
badge_name_vectors = tfidf_vectorizer_name.fit_transform(df['badge_name']).toarray()

tfidf_vectorizer_desc = TfidfVectorizer()
description_vectors = tfidf_vectorizer_desc.fit_transform(df['description']).toarray()

# 3️⃣ One-Hot Encoding (발급 기관)
onehot_encoder = OneHotEncoder(sparse_output=False)
issuer_encoded = onehot_encoder.fit_transform(df[['issuer']])

# 4️⃣ Multi-Hot Encoding (스킬 및 목표)
mlb_skills = MultiLabelBinarizer()
skills_encoded = mlb_skills.fit_transform(df['skillsValidated'].apply(eval))

mlb_goals = MultiLabelBinarizer()
goals_encoded = mlb_goals.fit_transform(df['relevant_goals'].apply(eval))

# 5️⃣ Ordinal Encoding (난이도)
difficulty_mapping = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
df['difficulty_level_encoded'] = df['difficulty_level'].map(difficulty_mapping)

# 6️⃣ 벡터화된 데이터셋 병합
feature_matrix = np.hstack([
    df[['badge_id_encoded', 'difficulty_level_encoded']].values,
    badge_name_vectors,
    issuer_encoded,
    skills_encoded,
    goals_encoded,
    description_vectors
])

vectorized_df = pd.DataFrame(feature_matrix)

# 7️⃣ 메타데이터 저장 (JSON)

metadata_filename = "vectorized_badge_dataset_metadata.json"  # 백터화된 데이터세트와 연계된 이름으로 메타데이터 저장

metadata = {
    "badge_id": "Original Badge ID",
    "issuer_features": list(onehot_encoder.get_feature_names_out(['issuer'])),
    "skills_features": list(mlb_skills.classes_),
    "goals_features": list(mlb_goals.classes_),
    "badge_name_tfidf_features": list(tfidf_vectorizer_name.get_feature_names_out()),
    "description_tfidf_features": list(tfidf_vectorizer_desc.get_feature_names_out())
}

with open(metadata_filename, "w") as f:
    json.dump(metadata, f, indent=4)

# 8️⃣ 벡터화된 데이터 저장
vectorized_df.to_csv("vectorized_badge_dataset.csv", index=False)
