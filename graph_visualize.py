import pandas as pd
from pyvis.network import Network

# 사용자-배지 관계 데이터 로드
user_badge_df = pd.read_excel("data/user_badge_relationship.xlsx", engine='openpyxl')

# Pyvis 네트워크 객체 생성
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

# 사용자-배지 노드 및 엣지 추가
for index, row in user_badge_df.iterrows():
    user_id = row['user_id']
    badge_id = row['BADGE_ID']
    
    # 사용자 노드 추가 (하늘색)
    net.add_node(user_id, label=user_id, color='skyblue', title=f"User: {user_id}")
    
    # 배지 노드 추가 (주황색)
    net.add_node(badge_id, label=badge_id, color='orange', title=f"Badge: {badge_id}")
    
    # 엣지 추가
    net.add_edge(user_id, badge_id)

# 물리적 레이아웃 설정
net.barnes_hut()

# 그래프 저장 및 시각화
net.show("user_badge_relationship.html", notebook=False)
