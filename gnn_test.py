import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

# 1. Cora 데이터셋 로드
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# 2. GCN 모델 정의
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 3. 모델 초기화 및 학습 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(dataset.num_node_features, 16, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 4. 모델 학습 함수
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# 5. 모델 평가 함수
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc

# 6. 그래프 시각화 함수
def visualize_graph(data):
    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)
    labels = data.y.cpu().numpy()
    nx.draw(G, pos, node_color=labels, with_labels=True, node_size=50, cmap=plt.cm.rainbow, alpha=0.7)
    plt.title('Cora Dataset Graph Visualization')
    plt.show()

# 7. 그래프 시각화
visualize_graph(data)

# 8. 모델 학습
for epoch in range(200):
    loss = train()
    if epoch % 20 == 0:
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')

# 9. 모델 테스트
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
