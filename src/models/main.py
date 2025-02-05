import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json

# 1. 데이터셋 로드
badge_df = pd.read_csv('data/processed/vectorized_badge_dataset_10.csv')  # 배지 데이터셋
user_df = pd.read_csv('data/processed/vectorized_user_dataset_10.csv')    # 사용자 데이터셋
edges_df = pd.read_csv('data/processed/edges_dataset.csv')  # 엣지 데이터셋


# 2. 메타데이터 로드 (목표 매칭 및 획득한 배지)
with open('data/processed/vectorized_user_dataset_metadata_10.json', 'r') as f:
    user_metadata = json.load(f)

with open('data/processed/vectorized_badge_dataset_metadata_10.json', 'r') as g:
    badge_metadata = json.load(g)

# 3. 노드 피처 준비
# 사용자 및 배지 피처 정규화
scaler = StandardScaler()
user_features = scaler.fit_transform(user_df.values)  # 사용자 피처 정규화
badge_features = scaler.fit_transform(badge_df.values)  # 배지 피처 정규화

# 사용자와 배지 피처 결합
all_features = torch.tensor(np.concatenate([user_features, badge_features], axis=0), dtype=torch.float)

# 4. 엣지 인덱스 생성 (사용자-배지 관계)
edges = edges_df[['source', 'target']].values
edges = edges.astype(str)  # 모든 값을 문자열로 변환

# 숫자가 아닌 값이 있는지 확인
non_numeric_edges = edges[~np.char.isnumeric(edges[:, 0]) | ~np.char.isnumeric(edges[:, 1])]
if non_numeric_edges.size > 0:
    print(f"경고: 'source' 또는 'target' 열에 숫자가 아닌 값이 포함되어 있습니다. 해당 행을 제거합니다.")
    print(non_numeric_edges)

# 숫자형으로 변환
edges = edges.astype(int)

# 엣지 인덱스 생성
#edge_index = torch.tensor(edges.T, dtype=torch.long)

# 'source'와 'target' 열을 사용하여 에지 인덱스를 생성합니다.
edge_index = torch.tensor(edges_df[['source', 'target']].values.T, dtype=torch.long)


# 5. PyTorch Geometric 데이터 객체 생성
data = Data(x=all_features, edge_index=edge_index)



# 6. GNN 모델 구축
class GNNModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(in_channels, 64)
        self.conv2 = torch_geometric.nn.GCNConv(64, out_channels)
        self.fc = nn.Linear(out_channels, 1)  # 출력층

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        out = self.fc(x)
        return out

# 모델 초기화
model = GNNModel(in_channels=all_features.shape[1], out_channels=16)

# 7. 학습 설정 (손실 함수, 옵티마이저)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 8. 모델 학습
def train(model, data, optimizer, criterion, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        # 레이블 데이터가 data.y에 있어야 합니다.
        if data.y is None:
            print("경고: 레이블 데이터가 없습니다. 학습을 진행할 수 없습니다.")
            return
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

train(model, data, optimizer, criterion)

# 9. 학습 후 모델 저장
torch.save(model.state_dict(), 'gnn_model.pth')
