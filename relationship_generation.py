import pandas as pd
import ast

# 사용자 데이터 로드
user_df = pd.read_excel("data/User_Dataset.xlsx", engine='openpyxl')


# 'recommendation_history' 필드를 리스트로 변환
user_df['recommendation_history'] = user_df['recommendation_history'].apply(ast.literal_eval)

# 사용자-배지 관계 데이터 생성
user_badge_relationships = []
for index, row in user_df.iterrows():
    user_id = row['user_id']
    for badge_id in row['recommendation_history']:
        user_badge_relationships.append({'user_id': user_id, 'BADGE_ID': badge_id})

# 데이터프레임 생성
user_badge_df = pd.DataFrame(user_badge_relationships)

# 결과 저장
user_badge_df.to_excel("data/user_badge_relationship.xlsx", index=False, engine='openpyxl')

print("사용자-배지 관계 데이터셋이 'user_badge_relationship.xlsx' 파일로 저장되었습니다.")
