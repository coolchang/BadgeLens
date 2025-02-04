import pandas as pd
from pyvis.network import Network

# 오리지널 데이터 로드  (10개 데이터)
#user_file_path = "data/user_dataset_updated.xlsx"
#badge_file_path = "data/badge_dataset_updated.xlsx"
#relationship_file_path = "data/relationship_dataset_updated.xlsx"


# 증강 데이터 로드 오리지널 (1000개 데이터)
user_file_path = "data/augmented_user_dataset.xlsx"
badge_file_path = "data/augmented_badge_dataset.xlsx"
relationship_file_path = "data/augmented_relationship_dataset.xlsx"


user_df = pd.read_excel(user_file_path, engine='openpyxl')
badge_df = pd.read_excel(badge_file_path, engine='openpyxl')
user_badge_df = pd.read_excel(relationship_file_path, engine='openpyxl')

# 컬럼명 소문자로 변환하여 일관성 유지
user_badge_df.columns = user_badge_df.columns.str.lower()

# PyVis 네트워크 그래프 생성
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

# 사용자 및 배지 데이터를 딕셔너리로 변환 (빠른 조회를 위해)
user_info = user_df.set_index('user_id').T.to_dict()
badge_info = badge_df.set_index('badge_id').T.to_dict()

# 사용자-배지 관계 추가
for _, row in user_badge_df.iterrows():
    user_id = str(row['user_id'])
    badge_id = str(row['badge_id'])

    # 사용자 노드 추가
    if user_id not in net.get_nodes():
        user_details = user_info.get(user_id, {})
        user_title = f"User ID: {user_id}<br>" + "<br>".join([f"{key}: {value}" for key, value in user_details.items()])
        net.add_node(user_id, label=user_id, color='skyblue', size=15, title=user_title)

    # 배지 노드 추가
    if badge_id not in net.get_nodes():
        badge_details = badge_info.get(badge_id, {})
        badge_title = f"Badge ID: {badge_id}<br>" + "<br>".join([f"{key}: {value}" for key, value in badge_details.items()])
        net.add_node(badge_id, label=badge_id, color='orange', size=15, title=badge_title)

    # 사용자-배지 엣지 추가
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
net.show("user_badge_relationship_pyvis.html", notebook=False)
