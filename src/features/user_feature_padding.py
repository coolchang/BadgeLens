import pandas as pd
import json


# 벡터화된 배지 데이터셋 로드
badge_df = pd.read_csv('data/processed/vectorized_badge_dataset_10.csv')

# 벡터화된 유저 데이터셋 로드
user_df = pd.read_csv('data/processed/vectorized_user_dataset_10.csv')

# 배지 데이터셋의 피처 목록
badge_features = badge_df.columns.tolist()

# 유저 데이터셋의 피처 목록
user_features = user_df.columns.tolist()

# 배지 데이터셋에는 있지만 유저 데이터셋에는 없는 피처 식별
missing_features = [feature for feature in badge_features if feature not in user_features]

# 부족한 피처를 유저 데이터셋에 추가하고 값은 0으로 채움
for feature in missing_features:
    user_df[feature] = 0

# 피처 순서를 배지 데이터셋의 피처 순서에 맞게 정렬
user_df = user_df[badge_features]

# 결과 저장
user_df.to_csv('padded_vectorized_user_dataset.csv', index=False)

# 메타데이터 파일 로드
metadata_file_path = 'data/processed/vectorized_user_dataset_metadata_10.json'
with open(metadata_file_path, 'r') as f:
    metadata = json.load(f)

# 추가된 피처를 메타데이터에 추가
for feature in missing_features:
    metadata[feature] = 'Added to match badge dataset; originally not present in user dataset.'

# 업데이트된 메타데이터 저장
with open(metadata_file_path, 'w') as f:
    json.dump(metadata, f, indent=4)

print("유저 데이터셋의 피처 수가 배지 데이터셋과 동일하게 맞춰졌으며, 메타데이터도 업데이트되었습니다.")
