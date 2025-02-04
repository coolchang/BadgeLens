import pandas as pd
from pyvis.network import Network


# 사용자 데이터 로드
user_df = pd.read_excel("data/user_dataset.xlsx", engine='openpyxl')

# 배지 데이터 로드
badge_df = pd.read_excel("data/openbadge_dataset.xlsx", engine='openpyxl')

# 사용자-배지 관계 데이터 로드
user_badge_df = pd.read_excel("data/relationship_dataset.xlsx", engine='openpyxl')



# 열 이름을 소문자로 변환하여 일관성 유지
user_badge_df.columns = user_badge_df.columns.str.lower()

# Pyvis 네트워크 객체 생성
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

# 사용자 정보를 딕셔너리로 변환 (user_id를 키로 사용)
user_info = user_df.set_index('user_id').T.to_dict()

# 배지 정보를 딕셔너리로 변환 (badge_id를 키로 사용)
badge_info = badge_df.set_index('badge_id').T.to_dict()

# 사용자-배지 노드 및 엣지 추가
for index, row in user_badge_df.iterrows():
    user_id = str(row['user_id'])  # 문자열로 변환하여 일관성 유지
    badge_id = str(row['badge_id'])  # 문자열로 변환하여 일관성 유지
    
    # 사용자 노드 추가 (하늘색, 크기 15)
    if user_id not in net.node_ids:
        user_details = user_info.get(user_id, {})
        user_title = f"User ID: {user_id}<br>" + "<br>".join([f"{key}: {value}" for key, value in user_details.items()])
        net.add_node(user_id, label=user_id, color='skyblue', size=15, title=user_title)
    
    # 배지 노드 추가 (주황색, 크기 15)
    if badge_id not in net.node_ids:
        badge_details = badge_info.get(badge_id, {})
        badge_title = f"Badge ID: {badge_id}<br>" + "<br>".join([f"{key}: {value}" for key, value in badge_details.items()])
        net.add_node(badge_id, label=badge_id, color='orange', size=15, title=badge_title)
    
    # 엣지 추가
    net.add_edge(user_id, badge_id)

# 노드 레이블 폰트 크기 설정 및 물리 엔진 옵션 설정
options = """
var options = {
  "nodes": {
    "font": {
      "size": 12
    }
  },
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -8000,
      "centralGravity": 0.3,
      "springLength": 95,
      "springConstant": 0.04,
      "damping": 0.09
    },
    "minVelocity": 0.75
  }
}
"""
net.set_options(options)

# 그래프 저장 및 시각화
net.show("user_badge_relationship.html", notebook=False)


