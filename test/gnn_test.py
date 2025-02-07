import time
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# 노드 특징 행렬 (10개의 노드, 각 노드는 10개의 특징을 가짐)
x = torch.tensor([
    [0.5, 1.2, 3.1, 0.0, 2.2, 1.1, 0.3, 0.8, 1.5, 2.3],
    [1.1, 0.7, 0.0, 2.1, 1.3, 0.4, 2.2, 1.0, 0.9, 1.8],
    [0.3, 2.2, 1.1, 0.5, 1.7, 2.0, 0.6, 1.4, 0.2, 0.9],
    [2.0, 0.1, 1.5, 1.8, 0.6, 1.9, 2.1, 0.7, 1.3, 0.4],
    [1.0, 1.1, 0.5, 2.0, 1.8, 0.3, 1.7, 0.6, 2.2, 1.9],
    [0.6, 2.0, 1.3, 0.7, 1.5, 2.1, 0.4, 1.2, 0.8, 1.0],
    [1.9, 0.4, 2.2, 1.1, 0.3, 1.6, 2.0, 0.5, 1.4, 0.7],
    [0.2, 1.5, 0.9, 2.3, 1.0, 0.8, 1.3, 2.1, 0.6, 1.7],
    [1.4, 0.6, 1.8, 0.9, 2.1, 0.5, 1.2, 0.3, 2.0, 1.5],
    [0.7, 2.1, 0.4, 1.6, 0.8, 1.3, 2.2, 0.9, 1.0, 1.4]
], dtype=torch.float)

# 엣지 목록 (노드 간의 연결 관계를 정의)
edge_index = torch.tensor([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 4, 6, 8],
    [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 2, 4, 6, 8, 0]
], dtype=torch.long)

# 노드 레이블 (각 노드의 클래스 레이블: 0부터 4까지 총 5개 클래스)
y = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=torch.long)

# 데이터 객체 생성
data = Data(x=x, edge_index=edge_index, y=y)

# GCN 모델 정의
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, 5)  # 클래스 수에 맞게 출력 차원 설정

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 모델 초기화
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 학습 루프
num_epochs = 200
for epoch in range(num_epochs):
    start_time = time.time()  # 에폭 시작 시간 기록

    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out, data.y)  # 손실 함수 계산
    loss.backward()
    optimizer.step()

    # 에폭 종료 시간 기록 및 경과 시간 계산
    epoch_time = time.time() - start_time

    # 정확도 계산
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = (pred == data.y).sum().item()
    accuracy = correct / len(data.y)

    # 에폭 번호, 경과 시간, 손실 값, 정확도 출력
    print(f'Epoch {epoch+1}/{num_epochs}, Time: {epoch_time:.4f}s, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
