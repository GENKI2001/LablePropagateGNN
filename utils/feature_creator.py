import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
import random
from collections import defaultdict
from torch_geometric.utils import to_networkx
import networkx as nx
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

def display_node_features(data, adj_matrix, one_hot_labels, dataset_name, max_hops=2, sample_nodes=None):
    """
    ノードの特徴量を表示する関数
    
    Args:
        data: PyTorch Geometric データオブジェクト
        adj_matrix: 隣接行列
        one_hot_labels: ワンホットエンコーディングされたラベル
        dataset_name (str): データセット名
        max_hops (int): 最大hop数
        sample_nodes (list): 表示するノードのリスト（デフォルト: [0, 10, 50, 100, 200]）
    """
    
    if sample_nodes is None:
        sample_nodes = [0, 10, 50, 100, 200]
    
    print(f"\n=== {dataset_name} ノード特徴量の確認 ===")
    print(f"特徴量の形状: {data.x.shape}")
    print(f"クラス数: {one_hot_labels.shape[1]}")
    print(f"最大hop数: {max_hops}")
    
    # unknownクラスの情報を表示
    unknown_mask = ~data.train_mask
    if unknown_mask.sum() > 0:
        print(f"unknownクラス: あり（テスト・検証ノード用）")
        print(f"unknownノード数: {unknown_mask.sum().item()}")
    else:
        print(f"unknownクラス: なし")
    
    for hop in range(1, max_hops + 1):
        print(f"{hop}-hop特徴量の次元: {one_hot_labels.shape[1]}")

    # いくつかのノードの特徴量を表示
    for node_idx in sample_nodes:
        if node_idx < data.num_nodes:
            # 正しい次数を計算（隣接行列から）
            degree = adj_matrix[node_idx].sum().item()
            
            print(f"\nノード {node_idx}:")
            print(f"  実際のラベル: {data.y[node_idx].item()}")
            print(f"  ノードの種類: {'訓練' if data.train_mask[node_idx] else 'テスト・検証'}")
            print(f"  ノードの次数: {degree}")
            
            # 各hopの特徴量を表示
            start_idx = 0
            for hop in range(1, max_hops + 1):
                end_idx = start_idx + one_hot_labels.shape[1]
                hop_feat = data.x[node_idx, start_idx:end_idx].cpu().numpy()
                print(f"  {hop}-hop隣接ノードの平均特徴量: {hop_feat}")
                start_idx = end_idx
            
            # 結合された特徴量
            combined_feat = data.x[node_idx].cpu().numpy()
            print(f"  結合された特徴量: {combined_feat}")

def get_feature_info(data, one_hot_labels, max_hops=2):
    """
    特徴量の情報を取得する関数
    
    Args:
        data: PyTorch Geometric データオブジェクト
        one_hot_labels: ワンホットエンコーディングされたラベル
        max_hops (int): 最大hop数
    
    Returns:
        dict: 特徴量の情報を含む辞書
    """
    return {
        'feature_dim': data.x.shape[1],
        'num_classes': one_hot_labels.shape[1],
        'max_hops': max_hops,
        'hop_dims': [one_hot_labels.shape[1]] * max_hops
    }

def create_pca_features(data, device, pca_components=50, original_features=None):
    """
    ノード特徴量に対してPCAで次元圧縮した新たな特徴量を返す関数

    Args:
        data (torch_geometric.data.Data): PyGのデータオブジェクト
        device (torch.device): 使用デバイス（cuda または cpu）
        pca_components (int): 圧縮後の次元数
        original_features (torch.Tensor or None): 圧縮対象の特徴量（指定がなければ data.x）

    Returns:
        data: PCA特徴量が設定されたデータオブジェクト
        pca_features (torch.Tensor): PCA圧縮後の特徴量テンソル（[num_nodes, pca_components]）
        pca: PCAオブジェクト
    """
    print(f"PCA特徴量を作成中...（圧縮次元: {pca_components}）")

    # 入力特徴量を指定 or デフォルトは data.x
    if original_features is None:
        if not hasattr(data, 'x') or data.x is None:
            raise ValueError("original_features が None の場合、data.x が必要です。")
        features = data.x.cpu().numpy()
    else:
        features = original_features.cpu().numpy()

    # PCAによる次元圧縮
    pca = PCA(n_components=pca_components, random_state=42)
    reduced_features = pca.fit_transform(features)

    print(f"PCA完了: 元の次元数 = {features.shape[1]}, 圧縮後 = {pca_components}")
    print(f"累積寄与率（explained variance ratio）: {pca.explained_variance_ratio_.sum():.4f}")

    # PCA特徴量をテンソルに変換
    pca_features = torch.tensor(reduced_features, dtype=torch.float32).to(device)
    
    # データオブジェクトの特徴量を更新
    data.x = pca_features

    return data, pca_features, pca

def create_label_features(data, device, max_hops=2, exclude_test_labels=True, use_neighbor_label_features=True, temperature=1.0, label_smoothing=0.):
    print(f"現在の特徴量の形状: {data.x.shape}")

    # ワンホットエンコーディングの作成
    train_labels = data.y[data.train_mask].cpu().numpy().reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_labels)
    all_labels = data.y.cpu().numpy().reshape(-1, 1)
    one_hot_labels = encoder.transform(all_labels)

    if exclude_test_labels:
        one_hot_labels[~data.train_mask.cpu().numpy()] = 0
    else:
        unknown_mask = ~data.train_mask
        if unknown_mask.sum() > 0:
            original_classes = one_hot_labels.shape[1]
            unknown_encoding = np.zeros((one_hot_labels.shape[0], original_classes + 1))
            unknown_encoding[data.train_mask.cpu().numpy(), :original_classes] = one_hot_labels[data.train_mask.cpu().numpy()]
            unknown_encoding[unknown_mask.cpu().numpy(), original_classes] = 1
            one_hot_labels = unknown_encoding
            print(f"Unknownクラスを追加: {original_classes} → {original_classes + 1} クラス")
            print(f"テスト・検証ノード数: {unknown_mask.sum().item()}")
            print(f"訓練ノード数: {data.train_mask.sum().item()}")
        else:
            print("テスト・検証ノードがないため、unknownクラスは追加しません")

    # === ✅ ラベルスムージング適用 ===
    if label_smoothing > 0.0:
        num_classes = one_hot_labels.shape[1]
        one_hot_labels = one_hot_labels * (1 - label_smoothing) + (label_smoothing / num_classes)
        print(f"ラベルスムージングを適用: ε = {label_smoothing}")

    edge_index = data.edge_index.cpu()
    num_nodes = data.num_nodes
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj_matrix[edge_index[0], edge_index[1]] = 1.0
    adj_matrix[edge_index[1], edge_index[0]] = 1.0
    combined_features = data.x
    neighbor_label_features = None

    if use_neighbor_label_features:
        print("隣接ノードのラベル特徴量を結合します")
        print(f"温度パラメータ: {temperature}")
        one_hot_labels_tensor = torch.tensor(one_hot_labels, dtype=torch.float32)
        hop_features_list = []
        edge_index_np = edge_index.numpy()

        A = torch.zeros((num_nodes, num_nodes))
        A[edge_index[0], edge_index[1]] = 1
        A[edge_index[1], edge_index[0]] = 1
        A = A.bool()
        A.fill_diagonal_(False)

        prev_mask = A.clone()
        for hop in range(1, max_hops + 1):
            mask = prev_mask
            neighbor_labels = mask.float() @ one_hot_labels_tensor
            hop_features = F.softmax(neighbor_labels / temperature, dim=1)
            hop_features_list.append(hop_features)
            prev_mask = torch.matmul(mask.float(), A.float()).bool()

        neighbor_label_features = torch.cat(hop_features_list, dim=1)
        combined_features = torch.cat([data.x, neighbor_label_features], dim=1)
        print(f"結合後の特徴量の形状: {combined_features.shape}")
        print(f"  - 現在の特徴量: {data.x.shape[1]}次元")
        print(f"  - ラベル分布特徴量: {neighbor_label_features.shape[1]}次元")
    else:
        print("隣接ノードのラベル特徴量は結合しません")

    data.x = combined_features.to(device)
    return data, adj_matrix, one_hot_labels, neighbor_label_features



def compute_extended_structural_features(data, device,
                                         include_degree=True,
                                         include_clustering=True,
                                         include_triangle=True,
                                         include_depth=False,
                                         include_avg_neighbor_degree=True,
                                         include_pagerank=True,
                                         include_eigenvector=True,
                                         include_kcore=True,
                                         include_l2_stats=True
                                         ):
    """
    各ノードに対して構造的特徴量を計算し、data.x に結合する関数（拡張版）

    含まれる構造的特徴量（任意選択）:
        - degree: ノードの次数
        - clustering: クラスタ係数
        - triangle: 三角形数
        - depth: sourceノードからのBFS深さ
        - avg_neighbor_degree: 隣接ノードの平均次数
        - pagerank: PageRankスコア
        - eigenvector: 固有ベクトル中心性（影響力）
        - kcore: コア番号（中核的密度レベル）
        - l2_stats: L2ノルム差分統計量（平均、標準偏差、最大値、最小値）

    Returns:
        data: 特徴量が結合された PyG データ
        structural_features: 構造的特徴量テンソル（torch.Tensor）
    """
    
    num_nodes = data.num_nodes

    # PyTorch Geometric データを NetworkX グラフに変換（無向）
    G = to_networkx(data, to_undirected=True)
    
    # 自己ループを削除（NetworkXの一部の関数で必要）
    G.remove_edges_from(nx.selfloop_edges(G))

    # すべての特徴量をここに格納（NumPy配列として）
    features = []

    if include_degree:
        # 🔢 各ノードの次数（degree）: 単純な接続数
        degree_feat = np.array([val for _, val in sorted(G.degree())], dtype=np.float32).reshape(-1, 1)
        features.append(degree_feat)

    if include_clustering:
        # 🔁 クラスタ係数: ノードの近傍内で三角形が形成されている割合
        clustering_dict = nx.clustering(G)
        clustering_feat = np.array([clustering_dict[i] for i in range(num_nodes)], dtype=np.float32).reshape(-1, 1)
        features.append(clustering_feat)

    if include_triangle:
        # 🔺 三角形数: ノードが属する三角形構造の数（局所密度の指標）
        triangle_dict = nx.triangles(G)
        triangle_feat = np.array([triangle_dict[i] for i in range(num_nodes)], dtype=np.float32).reshape(-1, 1)
        features.append(triangle_feat)

    if include_depth:
        # 🧭 ノード0をルートとしたときの BFS 深さ（参考情報として）
        depth_dict = nx.single_source_shortest_path_length(G, source=0)
        depth_feat = np.zeros((num_nodes, 1), dtype=np.float32)
        for i in range(num_nodes):
            depth_feat[i, 0] = depth_dict.get(i, 0)
        features.append(depth_feat)

    if include_avg_neighbor_degree:
        # 🔄 隣接ノードの平均次数: 周囲のノードの密度の平均
        avg_deg_dict = nx.average_neighbor_degree(G)
        avg_deg_feat = np.array([avg_deg_dict[i] for i in range(num_nodes)], dtype=np.float32).reshape(-1, 1)
        features.append(avg_deg_feat)

    if include_pagerank:
        # 🌐 PageRank: グローバルな重要度スコア（スパム検出などでも有用）
        pr = nx.pagerank(G)
        pr_feat = np.array([pr[i] for i in range(num_nodes)], dtype=np.float32).reshape(-1, 1)
        features.append(pr_feat)

    if include_eigenvector:
        try:
            # 📈 固有ベクトル中心性: 影響力のあるノードとつながっているほどスコアが高い
            eig = nx.eigenvector_centrality_numpy(G)
            eig_feat = np.array([eig[i] for i in range(num_nodes)], dtype=np.float32).reshape(-1, 1)
            features.append(eig_feat)
        except:
            print("eigenvector_centrality の計算に失敗しました。スキップします。")

    if include_kcore:
        # 🏗️ k-core 番号: ノードが属する最大の k-core の k 値（中核性の指標）
        kcore = nx.core_number(G)
        kcore_feat = np.array([kcore[i] for i in range(num_nodes)], dtype=np.float32).reshape(-1, 1)
        features.append(kcore_feat)

    if include_l2_stats:
        # ======== L2ノルム差分統計量を追加 ========
        x_np = data.x.cpu().numpy()  # 全ノードの特徴量ベクトル
        l2_stats = np.zeros((num_nodes, 4), dtype=np.float32)  # mean, std, max, min

        for i in range(num_nodes):
            neighbors = list(G.neighbors(i))
            if not neighbors:
                continue
            neighbor_feats = x_np[neighbors]
            diff_norms = np.linalg.norm(x_np[i] - neighbor_feats, axis=1)
            l2_stats[i, 0] = diff_norms.mean()
            l2_stats[i, 1] = diff_norms.std()
            l2_stats[i, 2] = diff_norms.max()
            l2_stats[i, 3] = diff_norms.min()

        # 正規化（平均0・分散1）
        l2_stats = StandardScaler().fit_transform(l2_stats)
        l2_tensor = torch.tensor(l2_stats, dtype=torch.float32).to(device)
        features.append(l2_tensor)

    # 💡 全構造特徴量を結合（NumPy配列 [num_nodes, total_features]）
    structural_features = np.concatenate(features, axis=1)

    # ⚖️ 特徴量を標準化（平均0・分散1）→ 比較可能にするため重要
    scaler = StandardScaler()
    structural_features = scaler.fit_transform(structural_features)

    # 🔁 NumPy → torch.Tensor へ変換（デバイスへ送る）
    structural_features = torch.tensor(structural_features, dtype=torch.float32).to(device)

    # 📎 既存の特徴量（data.x）に構造特徴量を結合
    data.x = torch.cat([data.x, structural_features], dim=1)

    return data, structural_features

