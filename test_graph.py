import networkx as nx
import matplotlib.pyplot as plt

# 그래프 객체 생성
G = nx.Graph()

# 노드 추가
G.add_node('A')
G.add_node('B')
G.add_node('C')

# 엣지 추가
G.add_edge('A', 'B')
G.add_edge('B', 'C')
G.add_edge('C', 'A')

# 그래프 시각화
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=16)
plt.title('Simple Graph Visualization')
plt.show()
