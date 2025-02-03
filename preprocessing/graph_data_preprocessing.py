import pandas as pd
import numpy as np
import ast
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MultiLabelBinarizer

def load_data():
    """데이터 불러오기"""
    badge_df = pd.read_excel("data/OpenBadge_Dataset.xlsx")
    user_df = pd.read_excel("data/User_Dataset.xlsx")
    return badge_df, user_df

def create_graph_edges(user_df, badge_df):
    """User-Badge 관계 및 Badge-Badge 관계 생성"""
    user_badge_edges = []
    for _, row in user_df.iterrows():
        acquired_badges = ast.literal_eval(row['acquired_badges'])
        for badge in acquired_badges:
            user_badge_edges.append((row['user_id'], badge))

    badge_badge_edges = []
    for _, row in badge_df.iterrows():
        related_badges = ast.literal_eval(row['related_badges'])
        for related in related_badges:
            badge_badge_edges.append((row['badge_id'], related))

    return user_badge_edges, badge_badge_edges

def process_features(user_df, badge_df):
    """사용자 및 배지 노드 특성 벡터화"""
    # 사용자 목표(goal) One-Hot Encoding
    goal_encoder = OneHotEncoder(sparse_output=False)
    goal_encoded = goal_encoder.fit_transform(user_df[['goal']])
    
    # 사용자 기술(skills) Multi-Hot Encoding
    mlb_skills = MultiLabelBinarizer()
    user_df['skills'] = user_df['skills'].apply(ast.literal_eval)
    skills_encoded = mlb_skills.fit_transform(user_df['skills'])
    
    # 숙련도(competency_level) Ordinal Encoding
    competency_encoder = LabelEncoder()
    competency_encoded = competency_encoder.fit_transform(user_df['competency_level'])
    
    # 사용자 특성 벡터 결합: goal + skills + competency_level
    user_feature_vectors = np.hstack([goal_encoded, skills_encoded, competency_encoded.reshape(-1, 1)])
    
    # 배지 역량(competency) Multi-Hot Encoding
    mlb_competency = MultiLabelBinarizer()
    if 'competency' in badge_df.columns and not badge_df['competency'].isnull().all():
        # 문자열이면 리스트로 감싸서 처리
        badge_df['competency'] = badge_df['competency'].apply(lambda x: [x] if isinstance(x, str) else x)
        competency_encoded_badge = mlb_competency.fit_transform(badge_df['competency'])
    else:
        print("❌ Warning: 'competency' 열이 비어 있음. 기본값 적용")
        competency_encoded_badge = np.zeros((len(badge_df), 1))  # 기본값 설정
    
    # 학습 기회(learningOpportunity) One-Hot Encoding (동일한 인코더 사용)
    learning_opportunity_encoded = goal_encoder.fit_transform(badge_df[['learningOpportunity']])
    
    # 배지 특성 벡터 결합: competency + learningOpportunity
    badge_feature_vectors = np.hstack([competency_encoded_badge, learning_opportunity_encoded])
    
    print(f"✅ competency_encoded_badge shape: {competency_encoded_badge.shape}")  
    print(f"✅ learning_opportunity_encoded shape: {learning_opportunity_encoded.shape}")  
    
    return user_feature_vectors, badge_feature_vectors

def create_pyg_graph(user_badge_edges, badge_badge_edges, user_feature_vectors, badge_feature_vectors):
    """PyTorch Geometric을 위한 그래프 데이터 변환"""
    # 사용자 및 배지 노드를 고유 정수 인덱스로 매핑 (사용자 먼저, 이후 배지)
    # 매핑 결과가 올바르게 생성되었는지 디버깅 출력
    user_ids = set(u for u, _ in user_badge_edges)
    badge_ids = set(b for _, b in user_badge_edges)
    
    user_mapping = {uid: i for i, uid in enumerate(user_ids)}
    badge_mapping = {bid: i + len(user_mapping) for i, bid in enumerate(badge_ids)}
    
    print("Unique 사용자 수 (user_mapping 길이):", len(user_mapping))  # 기대: 100
    print("Unique 배지 수 (badge_mapping 길이):", len(badge_mapping))    # 기대: 100

    # User-Badge 관계 인덱스 변환
    user_badge_index = torch.tensor(
        [[user_mapping[u], badge_mapping[b]] for u, b in user_badge_edges],
        dtype=torch.long
    ).t()

    # Badge-Badge 관계 인덱스 변환 (매핑된 배지 ID만 사용)
    badge_badge_index = torch.tensor(
        [[badge_mapping[b1], badge_mapping[b2]]
         for b1, b2 in badge_badge_edges if b1 in badge_mapping and b2 in badge_mapping],
        dtype=torch.long
    ).t()

    # 모든 엣지를 하나의 텐서로 합침
    edge_index = torch.cat([user_badge_index, badge_badge_index], dim=1)

    # 노드 특성 텐서 변환
    user_feature_tensor = torch.tensor(user_feature_vectors, dtype=torch.float)
    badge_feature_tensor = torch.tensor(badge_feature_vectors, dtype=torch.float)

    # 특성 차원 맞추기 (패딩)
    user_feature_dim = user_feature_tensor.shape[1]
    badge_feature_dim = badge_feature_tensor.shape[1]
    if badge_feature_dim < user_feature_dim:
        pad = torch.zeros(badge_feature_tensor.shape[0], user_feature_dim - badge_feature_dim)
        badge_feature_tensor = torch.cat([badge_feature_tensor, pad], dim=1)
    elif badge_feature_dim > user_feature_dim:
        pad = torch.zeros(user_feature_tensor.shape[0], badge_feature_dim - user_feature_dim)
        user_feature_tensor = torch.cat([user_feature_tensor, pad], dim=1)

    # 전체 노드 특성 결합: 사용자 노드와 배지 노드 모두 포함 (순서: 사용자 노드, 그 다음 배지 노드)
    x = torch.cat([user_feature_tensor, badge_feature_tensor], dim=0)

    # PyG Data 객체 생성
    graph_data = Data(x=x, edge_index=edge_index)

    # Data 객체의 딕셔너리 형태에 사용자 노드 개수를 추가하여 직렬화 시 보존합니다.
    data_dict = graph_data.to_dict()
    data_dict['num_user_nodes'] = user_feature_tensor.shape[0]
    graph_data = Data(**data_dict)

    return graph_data

if __name__ == "__main__":
    badge_df, user_df = load_data()
    user_badge_edges, badge_badge_edges = create_graph_edges(user_df, badge_df)
    user_feature_vectors, badge_feature_vectors = process_features(user_df, badge_df)
    
    graph_data = create_pyg_graph(user_badge_edges, badge_badge_edges, user_feature_vectors, badge_feature_vectors)
    
    # 생성된 Data 객체 정보 출력 및 검증
    print("총 노드 수:", graph_data.x.size(0))            # 기대: 200 (사용자 100 + 배지 100)
    print("전체 특성 차원:", graph_data.x.size(1))         # 예: 13
    print("연결된 엣지 수:", graph_data.edge_index.size(1))  # 예: 446 (엣지 수는 데이터에 따라 달라짐)
    print(graph_data)  # Data 객체 요약 출력
    
    # 데이터 저장
    torch.save(graph_data, "data/graph_data.pt")
    print("✅ PyTorch Geometric 그래프 데이터 저장 완료! 🚀")
