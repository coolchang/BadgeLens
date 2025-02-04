import pandas as pd
import random
from faker import Faker

fake = Faker()

# 사용자 데이터 증강: user_id를 "U001", "U002", ... 형식으로 생성
def augment_user_data(user_df, target_count=1000):
    augmented_users = [user_df]
    while sum(len(df) for df in augmented_users) < target_count:
        new_rows = []
        current_total = sum(len(df) for df in augmented_users)
        for _, row in user_df.iterrows():
            new_row = row.copy()
            new_row['user_id'] = f"U{(current_total + len(new_rows) + 1):03d}"
            new_row['name'] = fake.name()
            new_row['email'] = fake.email()
            new_rows.append(new_row)
            if current_total + len(new_rows) >= target_count:
                break
        augmented_users.append(pd.DataFrame(new_rows))
    return pd.concat(augmented_users, ignore_index=True)

# 배지 데이터 증강: badge_id를 "B001", "B002", ... 형식으로 생성
def augment_badge_data(badge_df, target_count=1000):
    augmented_badges = [badge_df]
    while sum(len(df) for df in augmented_badges) < target_count:
        new_rows = []
        current_total = sum(len(df) for df in augmented_badges)
        for _, row in badge_df.iterrows():
            new_row = row.copy()
            new_row['badge_id'] = f"B{(current_total + len(new_rows) + 1):03d}"
            new_row['badge_name'] = row['badge_name'] + ' ' + fake.word()
            new_row['description'] = row['description'] + ' ' + fake.sentence()
            new_rows.append(new_row)
            if current_total + len(new_rows) >= target_count:
                break
        augmented_badges.append(pd.DataFrame(new_rows))
    return pd.concat(augmented_badges, ignore_index=True)

# 사용자-배지 관계 데이터 증강: 원본 관계 데이터셋과 동일하게 user_id, goal, badge_id, goal_matching_score를 생성
def augment_relationship_data(user_df, badge_df, relationship_df, target_count=1000):
    augmented_relationships = [relationship_df]
    # 사용자별 goal 정보를 딕셔너리로 미리 준비 (user_id -> goal)
    user_goal_dict = user_df.set_index('user_id')['goal'].to_dict()
    user_ids = user_df['user_id'].tolist()
    badge_ids = badge_df['badge_id'].tolist()
    while sum(len(df) for df in augmented_relationships) < target_count:
        new_rows = []
        current_total = sum(len(df) for df in augmented_relationships)
        for _ in range(target_count - current_total):
            chosen_user = random.choice(user_ids)
            new_relationship = {
                'user_id': chosen_user,
                'goal': user_goal_dict.get(chosen_user, ''),
                'badge_id': random.choice(badge_ids),
                'goal_matching_score': round(random.uniform(0.85, 0.95), 2)
            }
            new_rows.append(new_relationship)
            if current_total + len(new_rows) >= target_count:
                break
        augmented_relationships.append(pd.DataFrame(new_rows))
    return pd.concat(augmented_relationships, ignore_index=True)

# 데이터 로드 (fillna를 적용하여 빈 셀 문제 방지)
user_df = pd.read_excel('data/user_dataset_updated.xlsx').fillna('')
badge_df = pd.read_excel('data/badge_dataset_updated.xlsx').fillna('')
relationship_df = pd.read_excel('data/relationship_dataset_updated.xlsx').fillna('')

# 데이터 증강
augmented_user_df = augment_user_data(user_df)
augmented_badge_df = augment_badge_data(badge_df)
augmented_relationship_df = augment_relationship_data(augmented_user_df, augmented_badge_df, relationship_df)

# 증강된 데이터 저장
augmented_user_df.to_excel('augmented_user_dataset.xlsx', index=False)
augmented_badge_df.to_excel('augmented_badge_dataset.xlsx', index=False)
augmented_relationship_df.to_excel('augmented_relationship_dataset.xlsx', index=False)
