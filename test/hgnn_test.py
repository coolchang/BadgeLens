import time
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset


# HeteroData 객체 생성
data = HeteroData()

# 사용자 노드 특징 행렬 (10명의 사용자, 각 사용자당 16개의 특징)
data['user'].x = torch.randn(10, 16)

# 배지 노드 특징 행렬 (10개의 배지, 각 배지당 8개의 특징)
data['badge'].x = torch.randn(10, 8)

# 사용자와 배지 간의 관계를 나타내는 엣지 인덱스
# 예를 들어, 사용자 0이 배지 0을 가지고 있고, 사용자 1이 배지 1을 가지고 있다고 가정
data['user', 'has', 'badge'].edge_index = torch.tensor([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # 사용자 인덱스
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # 배지 인덱스
], dtype=torch.long)

# 그래프를 무방향으로 변환
data = T.ToUndirected()(data)


# 사용자 노드의 클래스 레이블 (예: 2개의 클래스)
data['user'].y = torch.randint(0, 2, (10,), dtype=torch.long)

# GNN 모델 정의
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv((-1, -1), 16)
        self.conv2 = SAGEConv((16, 16), 2)  # 사용자 노드의 클래스 수에 맞게 출력 차원 설정

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 모델 초기화 및 이질 그래프에 맞게 변환
model = GNN()
model = to_hetero(model, data.metadata(), aggr='sum')

# 옵티마이저 설정
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 학습 루프
num_epochs = 200
for epoch in range(num_epochs):
    start_time = time.time()  # 에폭 시작 시간 기록

    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    out = out['user']  # 사용자 노드의 출력만 관심 대상
    loss = F.cross_entropy(out, data['user'].y)  # 손실 함수 계산
    loss.backward()
    optimizer.step()

    # 에폭 종료 시간 기록 및 경과 시간 계산
    epoch_time = time.time() - start_time

    # 정확도 계산
    model.eval()
    _, pred = out.max(dim=1)
    correct = (pred == data['user'].y).sum().item()
    accuracy = correct / len(data['user'].y)

    # 에폭 번호, 경과 시간, 손실 값, 정확도 출력
    print(f'Epoch {epoch+1}/{num_epochs}, Time: {epoch_time:.4f}s, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
