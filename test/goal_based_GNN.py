import time
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
import torch_geometric.transforms as T
import matplotlib.pyplot as plt

# HeteroData 객체 생성
data = HeteroData()

# 사용자 노드 특징 행렬 (1,000명의 사용자, 각 사용자당 100개의 특징)
data['user'].x = torch.randn(10000, 100)

# 배지 노드 특징 행렬 (1,500개의 배지, 각 배지당 100개의 특징)
data['badge'].x = torch.randn(15000, 100)

# 사용자와 배지 간의 매칭 스코어를 나타내는 엣지 인덱스 및 특징
edge_index = torch.tensor([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # 사용자 인덱스
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # 배지 인덱스
], dtype=torch.long)

matching_scores = torch.tensor([0.8, 0.6, 0.9, 0.7, 0.5, 0.85, 0.65, 0.75, 0.95, 0.55], dtype=torch.float)

data['user', 'matches_with', 'badge'].edge_index = edge_index
data['user', 'matches_with', 'badge'].edge_attr = matching_scores.unsqueeze(1)  # 엣지 특징으로 매칭 스코어 추가

# 사용자 노드의 클래스 레이블 (예: 2개의 클래스)
data['user'].y = torch.randint(0, 2, (10000,), dtype=torch.long)


# 그래프를 무방향으로 변환
data = T.ToUndirected()(data)

# GNN 모델 정의
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv((-1, -1), 16)
        self.conv2 = SAGEConv((16, 16), 2)  # 사용자 노드의 클래스 수에 맞게 출력 차원 설정

    def forward(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict, edge_index_dict)
        x = F.relu(x)
        x = self.conv2(x, edge_index_dict)
        return x

# 모델 초기화 및 이질 그래프에 맞게 변환
model = GNN()
model = to_hetero(model, data.metadata(), aggr='sum')

# 옵티마이저 설정
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 학습 손실 및 정확도를 저장할 리스트 초기화
train_losses = []
train_accuracies = []

# 학습 루프
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    out = out['user']  # 사용자 노드의 출력만 관심 대상
    loss = F.cross_entropy(out, data['user'].y)  # 손실 함수 계산
    loss.backward()
    optimizer.step()

    # 손실 값 저장
    train_losses.append(loss.item())

    # 정확도 계산
    model.eval()
    _, pred = out.max(dim=1)
    correct = (pred == data['user'].y).sum().item()
    accuracy = correct / len(data['user'].y)
    train_accuracies.append(accuracy)

    # 에폭 번호, 손실 값, 정확도 출력
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

# 최종 손실 값 및 정확도 출력
print(f'Final Loss: {train_losses[-1]:.4f}, Final Accuracy: {train_accuracies[-1]:.4f}')

# 손실 및 정확도 그래프 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()