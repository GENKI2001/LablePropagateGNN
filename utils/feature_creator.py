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

def create_label_features(data, device, max_hops=2, calc_neighbor_label_features=True, temperature=1.0, label_smoothing=0.):
    print(f"現在の特徴量の形状: {data.x.shape}")

    # ワンホットエンコーディングの作成
    train_labels = data.y[data.train_mask].cpu().numpy().reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_labels)
    all_labels = data.y.cpu().numpy().reshape(-1, 1)
    one_hot_labels = encoder.transform(all_labels)

    # テスト・検証ノードのラベルを0に設定（常にTrueの動作）
    one_hot_labels[~data.train_mask.cpu().numpy()] = 0

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

    if calc_neighbor_label_features:
        one_hot_labels_tensor = torch.tensor(one_hot_labels, dtype=torch.float32)
        hop_features_list = []
        edge_index_np = edge_index.numpy()

        A = torch.zeros((num_nodes, num_nodes))
        A[edge_index[0], edge_index[1]] = 1
        A[edge_index[1], edge_index[0]] = 1
        A = A.bool()
        A.fill_diagonal_(False)

        # 各hopまでの到達可能性を追跡
        reachable_nodes = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
        
        for hop in range(1, max_hops + 1):
            if hop == 1:
                # 1hop: 直接隣接ノードのみ
                mask = A.clone()
                reachable_nodes = mask.clone()
            else:
                # 2hop以上: 前のhopまでの到達可能性を除外
                # 現在のhopで到達可能なノードを計算
                current_reachable = torch.matmul(reachable_nodes.float(), A.float()).bool()
                # 前のhopまでの到達可能性を除外
                mask = current_reachable & (~reachable_nodes)
                # 到達可能性を更新
                reachable_nodes = reachable_nodes | current_reachable
            
            # 各ノードについて、そのhopで到達可能な隣接ノードのラベルを集約
            neighbor_labels = mask.float() @ one_hot_labels_tensor
            hop_features = F.softmax(neighbor_labels / temperature, dim=1)
            hop_features_list.append(hop_features)
            
            print(f"  {hop}hop: {mask.sum().item()}個の接続（前のhopを除外）")

        neighbor_label_features = torch.cat(hop_features_list, dim=1)
        combined_features = torch.cat([data.x, neighbor_label_features], dim=1)
        print(f"  - 生の特徴量: {data.x.shape[1]}次元")
        print(f"  - ラベル分布特徴量: {neighbor_label_features.shape[1]}次元")
    else:
        print("隣接ノードのラベル特徴量は結合しません")

    return adj_matrix, one_hot_labels, neighbor_label_features

def create_positional_random_walk_label_features(data, device, walk_length=4, use_train_only=True):
    from torch_sparse import spmm
    from torch_geometric.utils import add_self_loops, degree

    num_nodes = data.num_nodes
    y = data.y.cpu().numpy()

    print(f"\n=== 順序付きランダムウォーク特徴量作成 ===")
    print(f"ノード数: {num_nodes}")
    print(f"ランダムウォーク長: {walk_length}")
    print(f"訓練ノードのみ使用: {use_train_only}")

    # === ラベルのワンホット化 ===
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(y[data.train_mask.cpu().numpy()].reshape(-1, 1))
    one_hot_labels = encoder.transform(y.reshape(-1, 1))
    num_classes = one_hot_labels.shape[1]
    print(f"クラス数: {num_classes}")
    
    one_hot_labels[~data.train_mask.cpu().numpy()] = 0
    print(f"テスト・検証ノードのラベルを0に設定")
    
    Y = torch.tensor(one_hot_labels, dtype=torch.float32).to(device)  # [N, C]

    # === 正規化隣接行列（スパース）を作成 ===
    edge_index, _ = add_self_loops(data.edge_index)  # 自己ループあり
    row, col = edge_index
    deg = degree(row, num_nodes, dtype=torch.float32)  # 出次数
    deg_inv = 1.0 / deg
    deg_inv[deg_inv == float('inf')] = 0

    norm = deg_inv[row]  # 転置なし (row-normalized)
    A_hat = torch.sparse_coo_tensor(
        indices=edge_index,
        values=norm,
        size=(num_nodes, num_nodes)
    ).to(device)

    print(f"正規化隣接行列作成完了: {A_hat.shape}")

    # === 各 hop における伝播結果を記録 ===
    H = Y
    label_hops = []
    print(f"\n各hopでの特徴量次元:")
    for t in range(1, walk_length + 1):
        H = torch.sparse.mm(A_hat, H)  # t-step伝播
        label_hops.append(H)
        print(f"  {t}hop: {H.shape[1]}次元 (クラス数: {num_classes})")

    # === [N, walk_length, C] に変換して結合 ===
    position_label_tensor = torch.stack(label_hops, dim=1)  # [N, T, C]
    flattened = position_label_tensor.view(num_nodes, -1)   # [N, T*C]

    print(f"\n=== ランダムウォーク特徴量の詳細 ===")
    print(f"元の特徴量次元: {data.x.shape[1]}")
    print(f"ランダムウォーク特徴量次元: {flattened.shape[1]}")
    print(f"  - 各hop: {num_classes}次元 × {walk_length}hop = {num_classes * walk_length}次元")
    print(f"  - テンソル形状: {position_label_tensor.shape} → 平坦化: {flattened.shape}")

    # === data.x に結合 ===
    original_x_shape = data.x.shape
    
    return position_label_tensor

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

def create_similarity_based_edges(features, threshold=0.5, device='cpu'):
    """
    特徴量のコサイン類似度に基づいて新しいエッジを作成する関数
    
    Args:
        features (torch.Tensor): ノード特徴量 [num_nodes, feature_dim]（生の特徴量またはラベル分布特徴量など）
        threshold (float): コサイン類似度の閾値（0.0-1.0）
        device (str): 使用デバイス
    
    Returns:
        torch.Tensor: 新しいエッジインデックス [2, num_new_edges]
        torch.Tensor: 新しい隣接行列 [num_nodes, num_nodes]
        int: 作成されたエッジ数
    """
    print(f"\n=== 類似度ベースエッジ作成 ===")
    print(f"特徴量形状: {features.shape}")
    print(f"類似度閾値: {threshold}")
    
    num_nodes = features.shape[0]
    
    # 特徴量を正規化（コサイン類似度計算のため）
    features_normalized = F.normalize(features, p=2, dim=1)
    
    # 全ノード間のコサイン類似度を計算
    similarity_matrix = torch.mm(features_normalized, features_normalized.t())
    
    # 対角成分（自己類似度）を0に設定
    similarity_matrix.fill_diagonal_(0.0)
    
    # 閾値を超える類似度を持つペアを抽出
    edge_mask = similarity_matrix > threshold
    
    # 上三角行列のみを考慮（重複を避けるため）
    upper_triangle_mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
    edge_mask = edge_mask & upper_triangle_mask
    
    # エッジのインデックスを取得
    edge_indices = torch.nonzero(edge_mask, as_tuple=False)
    
    # 双方向のエッジを作成（無向グラフのため）
    if len(edge_indices) > 0:
        # 元のエッジ
        source_nodes = edge_indices[:, 0]
        target_nodes = edge_indices[:, 1]
        
        # 逆方向のエッジ
        reverse_source = target_nodes
        reverse_target = source_nodes
        
        # 両方向を結合
        all_source = torch.cat([source_nodes, reverse_source])
        all_target = torch.cat([target_nodes, reverse_target])
        
        new_edge_index = torch.stack([all_source, all_target], dim=0)
    else:
        # エッジがない場合
        new_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    
    # 新しい隣接行列を作成
    new_adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    if len(new_edge_index) > 0:
        new_adj_matrix[new_edge_index[0], new_edge_index[1]] = 1.0
    
    num_new_edges = new_edge_index.shape[1] // 2 if len(new_edge_index) > 0 else 0
    
    print(f"作成されたエッジ数: {num_new_edges}")
    print(f"新しいエッジインデックス形状: {new_edge_index.shape}")
    print(f"新しい隣接行列形状: {new_adj_matrix.shape}")
    
    # 類似度の統計情報を表示
    if len(similarity_matrix) > 0:
        valid_similarities = similarity_matrix[upper_triangle_mask]
        print(f"類似度統計:")
        print(f"  平均: {valid_similarities.mean():.4f}")
        print(f"  標準偏差: {valid_similarities.std():.4f}")
        print(f"  最小値: {valid_similarities.min():.4f}")
        print(f"  最大値: {valid_similarities.max():.4f}")
        print(f"  閾値 {threshold} を超えるペア数: {edge_mask.sum().item()}")
    
    return new_edge_index, new_adj_matrix, num_new_edges


def create_similarity_based_edges_with_original(original_edge_index, features, 
                                              threshold=0.5, device='cpu', 
                                              combine_with_original=True):
    """
    特徴量のコサイン類似度に基づいて新しいエッジを作成し、
    元のエッジと結合する関数
    
    Args:
        original_edge_index (torch.Tensor): 元のエッジインデックス [2, num_edges]
        features (torch.Tensor): ノード特徴量 [num_nodes, feature_dim]（生の特徴量またはラベル分布特徴量など）
        threshold (float): コサイン類似度の閾値（0.0-1.0）
        device (str): 使用デバイス
        combine_with_original (bool): 元のエッジと結合するかどうか
    
    Returns:
        torch.Tensor: 結合されたエッジインデックス [2, num_total_edges]
        torch.Tensor: 結合された隣接行列 [num_nodes, num_nodes]
        int: 元のエッジ数
        int: 新しく作成されたエッジ数
        int: 総エッジ数
    """
    print(f"\n=== 類似度ベースエッジ作成（元エッジ結合） ===")
    print(f"元のエッジ数: {original_edge_index.shape[1]}")
    print(f"特徴量形状: {features.shape}")
    print(f"類似度閾値: {threshold}")
    print(f"元エッジと結合: {combine_with_original}")
    
    num_nodes = features.shape[0]
    
    # 新しいエッジを作成
    new_edge_index, new_adj_matrix, num_new_edges = create_similarity_based_edges(
        features, threshold, device
    )
    
    if combine_with_original:
        # 元のエッジと新しいエッジを結合
        combined_edge_index = torch.cat([original_edge_index, new_edge_index], dim=1)
        
        # 重複エッジを除去（元のエッジと新しいエッジが重複する可能性）
        edge_pairs = combined_edge_index.t()
        unique_edges, inverse_indices = torch.unique(edge_pairs, dim=0, return_inverse=True)
        combined_edge_index = unique_edges.t()
        
        # 結合された隣接行列を作成
        combined_adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
        combined_adj_matrix[combined_edge_index[0], combined_edge_index[1]] = 1.0
        
        num_original_edges = original_edge_index.shape[1]
        num_total_edges = combined_edge_index.shape[1]
        num_actual_new_edges = num_total_edges - num_original_edges
        
        print(f"結合結果:")
        print(f"  元のエッジ数: {num_original_edges}")
        print(f"  新しく追加されたエッジ数: {num_actual_new_edges}")
        print(f"  総エッジ数: {num_total_edges}")
        print(f"  エッジ増加率: {num_actual_new_edges / num_original_edges * 100:.2f}%")
        
        return combined_edge_index, combined_adj_matrix, num_original_edges, num_actual_new_edges, num_total_edges
    else:
        # 新しいエッジのみを返す
        num_original_edges = original_edge_index.shape[1]
        num_total_edges = new_edge_index.shape[1]
        
        return new_edge_index, new_adj_matrix, num_original_edges, num_new_edges, num_total_edges

