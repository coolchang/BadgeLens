import time
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv
import torch_geometric.transforms as T
import matplotlib.pyplot as plt

# HeteroData 객체 생성
data = HeteroData()

# 사용자 노드 특징 행렬 (100,000명의 사용자, 각 사용자당 100개의 특징)
data['user'].x = torch.randn(100000, 100)

# 배지 노드 특징 행렬 (15,000개의 배지, 각 배지당 100개의 특징)
data['badge'].x = torch.randn(15000, 100)

# 사용자와 배지 간의 매칭 관계를 나타내는 엣지 인덱스 및 특징
edge_index_user_to_badge = torch.stack([
    torch.randint(0, 100000, (50000,)),  # 사용자 노드 (0~99,999 범위)
    torch.randint(0, 15000, (50000,))   # 배지 노드 (0~14,999 범위)
])
matching_scores = torch.rand(50000)  # 0~1 사이의 랜덤한 매칭 스코어

# 사용자 -> 배지 관계 추가
data['user', 'matches_with', 'badge'].edge_index = edge_index_user_to_badge
data['user', 'matches_with', 'badge'].edge_attr = matching_scores.unsqueeze(1)

# 배지 -> 사용자 역방향 관계 추가 (양방향 학습 가능하도록)
edge_index_badge_to_user = torch.stack([
    edge_index_user_to_badge[1],  # 배지 노드
    edge_index_user_to_badge[0]   # 사용자 노드
])
data['badge', 'matched_by', 'user'].edge_index = edge_index_badge_to_user
data['badge', 'matched_by', 'user'].edge_attr = matching_scores.unsqueeze(1)

# 사용자 노드의 클래스 레이블 (예: 2개의 클래스)
data['user'].y = torch.randint(0, 2, (100000,), dtype=torch.long)

# 그래프를 무방향으로 변환 (추가적인 안정성 확보)
data = T.ToUndirected()(data)

# GNN 모델 정의
class HeteroGNN(torch.nn.Module):
    def __init__(self):
        super(HeteroGNN, self).__init__()
        self.conv1 = HeteroConv({
            ('user', 'matches_with', 'badge'): GATConv((-1, -1), 16, add_self_loops=False),
            ('badge', 'matched_by', 'user'): GATConv((-1, -1), 16, add_self_loops=False)
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('user', 'matches_with', 'badge'): GATConv((16, 16), 2, add_self_loops=False),
            ('badge', 'matched_by', 'user'): GATConv((16, 16), 2, add_self_loops=False)
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict, edge_weight_dict):
        x_dict = self.conv1(x_dict, edge_index_dict, edge_weight_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}  # 활성화 함수 적용
        x_dict = self.conv2(x_dict, edge_index_dict, edge_weight_dict)
        return x_dict

# 모델 초기화
model = HeteroGNN()

# 옵티마이저 설정
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 학습 손실 및 정확도를 저장할 리스트 초기화
train_losses = []
train_accuracies = []

# 학습 루프
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # edge_weight 전달 (모든 관계에 대해 추가)
    edge_weight_dict = {
        ('user', 'matches_with', 'badge'): data['user', 'matches_with', 'badge'].edge_attr,
        ('badge', 'matched_by', 'user'): data['badge', 'matched_by', 'user'].edge_attr
    }

    # 모델 실행
    out = model(data.x_dict, data.edge_index_dict, edge_weight_dict)

    # 출력 크기 검증
    if out['user'].shape[0] != data['user'].y.shape[0]:
        raise ValueError(f"Output size mismatch! Expected {data['user'].y.shape[0]}, but got {out['user'].shape[0]}")

    # 손실 함수 계산
    loss = F.cross_entropy(out['user'], data['user'].y)
    loss.backward()
    optimizer.step()

    # 정확도 계산
    _, pred = out['user'].max(dim=1)
    accuracy = (pred == data['user'].y).sum().item() / len(data['user'].y)

    train_losses.append(loss.item())
    train_accuracies.append(accuracy)

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

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
