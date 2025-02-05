from preprocessing.graph_data_preprocessing import load_data, create_graph_edges, process_features, create_pyg_graph

if __name__ == "__main__":
    badge_df, user_df = load_data()
    user_badge_edges, badge_badge_edges = create_graph_edges(user_df, badge_df)
    user_feature_vectors, badge_feature_vectors = process_features(user_df, badge_df)
    
    graph_data = create_pyg_graph(user_badge_edges, badge_badge_edges, user_feature_vectors, badge_feature_vectors)
    
    # 생성된 그래프 데이터 출력 및 저장
    print(graph_data)
    print("✅ 그래프 데이터 변환이 완료되었습니다!")
    
    # 그래프 데이터를 파일로 저장 (시각화 전에 한 번 실행)
    # (이미 graph_data_preprocessing.py에서도 저장 중이라 중복 저장하지 않아도 됩니다.)
