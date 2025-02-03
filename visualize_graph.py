import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

# 안전한 글로벌 객체들을 허용하여 파일을 로드합니다.
from torch.serialization import safe_globals
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

with safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage]):
    # weights_only=False 옵션을 명시하여 전체 객체를 로드합니다.
    graph_data = torch.load("data/graph_data.pt", weights_only=False)

# PyG Data 객체를 undirected NetworkX 그래프로 변환합니다.
G = to_networkx(graph_data, to_undirected=True)

# 전체 노드 수 (예: 200개)
total_nodes = graph_data.x.size(0)
# 누락된(고립된) 노드가 있다면 추가합니다.
for i in range(total_nodes):
    if i not in G:
        G.add_node(i)

# 출력해서 전체 노드 수와 엣지 수를 확인합니다.
print("Total nodes in NetworkX graph:", G.number_of_nodes())
print("Total edges in NetworkX graph:", G.number_of_edges())

# 저장 시 Data 객체의 딕셔너리에 포함시킨 'num_user_nodes' 값을 읽어옵니다.
if hasattr(graph_data, 'num_user_nodes'):
    num_user_nodes = graph_data.num_user_nodes
else:
    raise AttributeError(
        "graph_data에 'num_user_nodes' 속성이 존재하지 않습니다. "
        "graph_data_preprocessing.py에서 Data 객체에 해당 정보가 올바르게 저장되었는지 확인하세요."
    )

# 전체 노드 번호를 정렬합니다.
sorted_nodes = sorted(G.nodes())

# 사용자 노드와 배지 노드를 색상으로 구분합니다.
# 사용자 노드는 하늘색 (skyblue), 배지 노드는 오렌지 (orange)로 표시합니다.
node_colors = ['skyblue' if node < num_user_nodes else 'orange' for node in sorted_nodes]

# 사용자와 배지 노드를 별도의 인덱스로 라벨링합니다.
# 사용자 노드: U1, U2, ... (0부터 시작하므로 node+1)
# 배지 노드: B1, B2, ... (전체 노드에서 사용자 노드 수를 뺀 값에 1을 더함)
labels = {}
for node in sorted_nodes:
    if node < num_user_nodes:
        labels[node] = f"U{node + 1}"
    else:
        labels[node] = f"B{node - num_user_nodes + 1}"

# 보다 분산된 레이아웃을 위해 Kamada-Kawai layout 사용
pos = nx.kamada_kawai_layout(G)

# 그래프 시각화
plt.figure(figsize=(12, 12))
nx.draw_networkx_nodes(G, pos, nodelist=sorted_nodes, node_color=node_colors, node_size=300)
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos, labels, font_size=8)
plt.title("Graph Visualization\nSkyblue: User Nodes, Orange: Badge Nodes")
plt.axis("off")
plt.show()
