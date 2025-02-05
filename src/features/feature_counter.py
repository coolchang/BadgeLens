import pandas as pd

# 벡터화된 배지 데이터셋 로드
badge_df = pd.read_csv('data/processed/vectorized_badge_dataset_10.csv')
# 벡터화된 유저 데이터셋 로드
user_df = pd.read_csv('data/processed/vectorized_user_dataset_10.csv')

# 각 데이터셋의 피처(열) 수 확인
num_badge_features = badge_df.shape[1]
num_user_features = user_df.shape[1]

print(f'벡터화된 배지 데이터셋의 피처 수: {num_badge_features}')
print(f'벡터화된 유저 데이터셋의 피처 수: {num_user_features}')