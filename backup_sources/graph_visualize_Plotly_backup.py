import pandas as pd
import networkx as nx
import plotly.graph_objects as go

# 사용자 데이터 로드
user_df = pd.read_excel("data/user_dataset.xlsx", engine='openpyxl')

# 배지 데이터 로드
badge_df = pd.read_excel("data/openbadge_dataset.xlsx", engine='openpyxl')

# 사용자-배지 관계 데이터 로드
user_badge_df = pd.read_excel("data/relationship_dataset.xlsx", engine='openpyxl')

# 열 이름을 소문자로 변환하여 일관성 유지
user_badge_df.columns = user_badge_df.columns.str.lower()

# NetworkX 그래프 객체 생성
G = nx.Graph()

# 사용자 정보를 딕셔너리로 변환 (user_id를 키로 사용)
user_info = user_df.set_index('user_id').T.to_dict()

# 배지 정보를 딕셔너리로 변환 (badge_id를 키로 사용)
badge_info = badge_df.set_index('badge_id').T.to_dict()

# 사용자-배지 노드 및 엣지 추가
for index, row in user_badge_df.iterrows():
    user_id = str(row['user_id'])  # 문자열로 변환하여 일관성 유지
    badge_id = str(row['badge_id'])  # 문자열로 변환하여 일관성 유지
    
    # 사용자 노드 추가
    if user_id not in G:
        user_details = user_info.get(user_id, {})
        user_title = f"User ID: {user_id}<br>" + "<br>".join([f"{key}: {value}" for key, value in user_details.items()])
        G.add_node(user_id, label=user_id, color='skyblue', title=user_title, size=15)
    
    # 배지 노드 추가
    if badge_id not in G:
        badge_details = badge_info.get(badge_id, {})
        badge_title = f"Badge ID: {badge_id}<br>" + "<br>".join([f"{key}: {value}" for key, value in badge_details.items()])
        G.add_node(badge_id, label=badge_id, color='orange', title=badge_title, size=15)
    
    # 엣지 추가
    G.add_edge(user_id, badge_id)

# 노드 위치 계산 (spring layout 사용)
pos = nx.spring_layout(G)

# 엣지 추출
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

# 노드 추출
node_x = []
node_y = []
node_text = []
node_color = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(G.nodes[node]['title'])
    node_color.append(G.nodes[node]['color'])

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    hoverinfo='text',
    textposition="top center",
    marker=dict(
        showscale=False,
        color=node_color,
        size=15,
        line_width=2),
    text=[G.nodes[node]['label'] for node in G.nodes()],
    textfont=dict(size=10, color='white')
)

# 레이아웃 설정
layout = go.Layout(
    title=dict(
        text='User-Badge Relationship Network',
        font=dict(size=16)
    ),
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=40),
    annotations=[dict(
        text="",
        showarrow=False,
        xref="paper", yref="paper"
    )],
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False)
)


# 그래프 생성
fig = go.Figure(data=[edge_trace, node_trace], layout=layout)

# 그래프 저장 및 시각화
fig.write_html("user_badge_relationship.html")
fig.show()
