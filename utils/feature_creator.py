import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
import random
from collections import defaultdict

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
<<<<<<< HEAD
    PCAで次元圧縮した元の特徴量と隣接ノードのラベル特徴量を結合する関数
    
    Args:
        data: PyTorch Geometric データオブジェクト
        device (torch.device): デバイス
        max_hops (int): 最大hop数（デフォルト: 2）
        exclude_test_labels (bool): テスト・検証ノードのラベルを隣接ノードの特徴量計算から除外するか（デフォルト: True）
        pca_components (int): PCAで圧縮する次元数（デフォルト: 50）
        original_features (torch.Tensor, optional): 元の特徴量（Noneの場合はdata.xを使用）
    
    Returns:
        data: 特徴量が設定されたデータオブジェクト
        adj_matrix: 隣接行列
        one_hot_labels: ワンホットエンコーディングされたラベル
        pca_features: PCA圧縮された特徴量
    """
    
    # 元の特徴量を取得
    if original_features is None:
        original_features = data.x
    
    print(f"元の特徴量の形状: {original_features.shape}")
    print(f"PCA圧縮後の次元数: {pca_components}")
    
    # 訓練データのラベルのみを使用してワンホットエンコーディングを作成
    train_labels = data.y[data.train_mask].cpu().numpy().reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_labels)
    
    # 全ノードのラベルをワンホットエンコーディング
    all_labels = data.y.cpu().numpy().reshape(-1, 1)
    one_hot_labels = encoder.transform(all_labels)
    
    if exclude_test_labels:
        # テスト・検証ノードのラベル情報を隣接ノードの特徴量計算から除外する場合
        print("テスト・検証ノードのラベル情報を隣接ノードの特徴量計算から除外します")
        print(f"テスト・検証ノード数: {(~data.train_mask).sum().item()}")
        print(f"訓練ノード数: {data.train_mask.sum().item()}")
        
        # テスト・検証ノードのラベルを無効化（特徴量計算時に使用しない）
        test_val_mask = ~data.train_mask
        if test_val_mask.sum() > 0:
            # テスト・検証ノードのワンホットエンコーディングをゼロベクトルに設定
            one_hot_labels[test_val_mask] = 0
    else:
        # 従来の方法: テスト・検証ノードをunknownクラスとして明示的に設定
        unknown_mask = ~data.train_mask
        if unknown_mask.sum() > 0:
            # unknownクラスのワンホットエンコーディングを作成
            original_classes = one_hot_labels.shape[1]
            unknown_encoding = np.zeros((one_hot_labels.shape[0], original_classes + 1))
            
            # 訓練データのノードは元のエンコーディングを保持
            unknown_encoding[data.train_mask, :original_classes] = one_hot_labels[data.train_mask]
            
            # テスト・検証ノードはunknownクラスとして設定 [0,0,...,0,1]
            unknown_encoding[unknown_mask, original_classes] = 1
            
            one_hot_labels = unknown_encoding
            print(f"Unknownクラスを追加: {original_classes} → {original_classes + 1} クラス")
            print(f"テスト・検証ノード数: {unknown_mask.sum().item()}")
            print(f"訓練ノード数: {data.train_mask.sum().item()}")
            
        else:
            # テスト・検証ノードがない場合は、unknownクラスを追加しない
            print("テスト・検証ノードがないため、unknownクラスは追加しません")
    
    # 訓練データのみを使用してPCAを学習
    train_features = original_features[data.train_mask].cpu().numpy()
    
    # PCAを適用
    pca = PCA(n_components=min(pca_components, train_features.shape[1], train_features.shape[0]))
    pca.fit(train_features)
    
    # 全ノードの特徴量をPCAで圧縮
    all_features = original_features.cpu().numpy()
    pca_features = pca.transform(all_features)
    
    print(f"PCA圧縮後の特徴量の形状: {pca_features.shape}")
    print(f"説明分散比: {pca.explained_variance_ratio_.sum():.4f}")
    
    # 隣接ノードのラベル特徴量を作成
    data_with_neighbor_features, adj_matrix, _ = create_neighbor_lable_features(
        data, device, max_hops, exclude_test_labels
    )
    
    # 特徴量を結合: PCA + 隣接ノード特徴量
    pca_tensor = torch.tensor(pca_features, dtype=torch.float32)
    neighbor_features = data_with_neighbor_features.x
    
    combined_features = torch.cat([pca_tensor, neighbor_features], dim=1)
    
    print(f"結合後の特徴量の形状: {combined_features.shape}")
    print(f"  - PCA特徴量: {pca_tensor.shape[1]}次元")
    print(f"  - 隣接ノード特徴量: {neighbor_features.shape[1]}次元")
    
    # 特徴量を設定
    data.x = combined_features.to(device)
    
    return data, adj_matrix, one_hot_labels, pca_features

def create_co_label_embeddings(data, device, embedding_dim=32, window_size=1, max_hops=1, exclude_test_labels=True):
    """
    ラベルの共起パターンをエンベディング（"co-label" embedding）を作成する関数
    
    - 隣接ノード間のラベル共起パターンを学習
    - 訓練データのラベルのみを使用して共起パターンを構築
    - 各ノードの隣接ノードのラベルシーケンスから共起特徴量を生成
    - 多hop対応（1-hop, 2-hop, ...）でより広範囲の共起パターンを学習
=======
    PCAで次元圧縮した特徴量を作成する関数（実験前に実行）
>>>>>>> 3c0a64f3ac2f336559879e8599b3d689d84e14a1
    
    Args:
        data: PyTorch Geometric データオブジェクト
        device (torch.device): デバイス
        embedding_dim (int): エンベディングの次元数（デフォルト: 32）
        window_size (int): 共起を計算するウィンドウサイズ（デフォルト: 1）
        max_hops (int): 最大hop数（デフォルト: 1）
        exclude_test_labels (bool): テスト・検証ノードのラベルを共起計算から除外するか（デフォルト: True）
    
    Returns:
        co_label_features: 共起ラベルエンベディング特徴量
        label_cooccurrence_matrix: ラベル共起行列
    """
    
    print(f"共起ラベルエンベディングを作成中...")
    print(f"エンベディング次元: {embedding_dim}")
    print(f"ウィンドウサイズ: {window_size}")
    print(f"最大hop数: {max_hops}")
    
    # エッジリストから隣接関係を取得
    edge_index = data.edge_index.cpu()
    num_nodes = data.num_nodes
    
    # 各ノードの隣接ノードリストを作成
    neighbors_dict = {}
    for i in range(num_nodes):
        neighbors_dict[i] = []
    
    # エッジから隣接関係を構築
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if src != dst:  # 自己ループを除外
            neighbors_dict[src].append(dst)
            neighbors_dict[dst].append(src)  # 無向グラフとして扱う
    
    # 訓練データのラベルのみを使用してラベル辞書を作成
    train_labels = data.y[data.train_mask].cpu().numpy()
    unique_labels = np.unique(train_labels)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    
    print(f"ラベル数: {num_classes}")
    print(f"ラベル: {unique_labels}")
    
    # ラベル共起行列を初期化
    label_cooccurrence_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    
    # 各ノードの隣接ノードのラベルシーケンスを作成（多hop対応）
    label_sequences = []
    valid_node_indices = []
    
    for node_idx in range(num_nodes):
        # 各hopの隣接ノードを取得
        all_neighbors = set()
        
        if max_hops == 1:
            # 1-hop: 直接隣接
            all_neighbors = set(neighbors_dict[node_idx])
        else:
            # 多hop: 各hopの隣接ノードを収集
            current_neighbors = set(neighbors_dict[node_idx])
            all_neighbors.update(current_neighbors)
            
            for hop in range(2, max_hops + 1):
                next_neighbors = set()
                for neighbor in current_neighbors:
                    next_neighbors.update(neighbors_dict[neighbor])
                # 前のhopの隣接ノードを除外
                next_neighbors = next_neighbors - all_neighbors
                all_neighbors.update(next_neighbors)
                current_neighbors = next_neighbors
        
        # 自分自身を除外
        all_neighbors.discard(node_idx)
        
        if exclude_test_labels:
            # テスト・検証ノードのラベル情報を除外する場合
            # 訓練ノードのみのラベル情報を使用
            train_neighbors = [n for n in all_neighbors if data.train_mask[n]]
            
            if len(train_neighbors) > 0:
                # 隣接ノードのラベルを取得
                neighbor_labels = []
                for neighbor in train_neighbors:
                    label = data.y[neighbor].item()
                    if label in label_to_idx:  # 訓練データに存在するラベルのみ
                        neighbor_labels.append(label)
                
                if len(neighbor_labels) > 0:
                    label_sequences.append(neighbor_labels)
                    valid_node_indices.append(node_idx)
        else:
            # 全ての隣接ノードのラベル情報を使用
            neighbor_labels = []
            for neighbor in all_neighbors:
                label = data.y[neighbor].item()
                if label in label_to_idx:  # 訓練データに存在するラベルのみ
                    neighbor_labels.append(label)
            
            if len(neighbor_labels) > 0:
                label_sequences.append(neighbor_labels)
                valid_node_indices.append(node_idx)
    
    print(f"有効なノード数: {len(valid_node_indices)}")
    print(f"ラベルシーケンス数: {len(label_sequences)}")
    
    # ラベル共起行列を構築
    for sequence in label_sequences:
        for i in range(len(sequence)):
            for j in range(max(0, i - window_size), min(len(sequence), i + window_size + 1)):
                if i != j:
                    label_i = sequence[i]
                    label_j = sequence[j]
                    label_cooccurrence_matrix[label_to_idx[label_i], label_to_idx[label_j]] += 1
    
    # 対称化（無向グラフのため）
    label_cooccurrence_matrix = (label_cooccurrence_matrix + label_cooccurrence_matrix.T) / 2
    
    # 正規化
    row_sums = label_cooccurrence_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # ゼロ除算を防ぐ
    label_cooccurrence_matrix = label_cooccurrence_matrix / row_sums
    
    print(f"ラベル共起行列の形状: {label_cooccurrence_matrix.shape}")
    print(f"共起行列の非ゼロ要素数: {np.count_nonzero(label_cooccurrence_matrix)}")
    
    # 共起行列からエンベディングを生成（SVDを使用）
    try:
        from sklearn.decomposition import TruncatedSVD
        # クラス数に基づいて動的にエンベディング次元を決定
        # クラス数が少ない場合（≤10）: クラス数に近い値を使用（無駄なゼロパディングを避ける）
        # クラス数が多い場合（>10）: 指定された次元数を使用（情報を十分に保持）
        if num_classes <= 10:
            # クラス数が少ない場合: クラス数と同じか、クラス数の半分程度を使用
            # 例: クラス数7の場合、min(32, 7) = 7
            dynamic_embedding_dim = min(embedding_dim, num_classes)
        else:
            # クラス数が多い場合は、指定された次元数を使用
            dynamic_embedding_dim = min(embedding_dim, num_classes)
        
        # SVDのn_componentsは特徴量数（クラス数）以下である必要がある
        svd_components = min(dynamic_embedding_dim, num_classes)
        svd = TruncatedSVD(n_components=svd_components, random_state=42)
        co_label_embeddings = svd.fit_transform(label_cooccurrence_matrix)
        actual_embedding_dim = co_label_embeddings.shape[1]
        
        print(f"SVDによるエンベディング作成完了 (クラス数: {num_classes}, 要求次元: {dynamic_embedding_dim}, 実際次元: {actual_embedding_dim})")
        print(f"説明分散比: {svd.explained_variance_ratio_.sum():.4f}")
            
    except ImportError:
        # SVDが利用できない場合は、共起行列をそのまま使用
        dynamic_embedding_dim = min(embedding_dim, num_classes)
        co_label_embeddings = label_cooccurrence_matrix[:, :dynamic_embedding_dim]
        actual_embedding_dim = co_label_embeddings.shape[1]
        print(f"共起行列を直接使用してエンベディング作成 (クラス数: {num_classes}, 次元: {actual_embedding_dim})")
    
    # 各ノードの共起特徴量を計算（実際の次元数に合わせて作成）
    co_label_features = torch.zeros((num_nodes, actual_embedding_dim), dtype=torch.float32)
    
    for node_idx in range(num_nodes):
        neighbors = neighbors_dict[node_idx]
        
        if exclude_test_labels:
            # テスト・検証ノードのラベル情報を除外する場合
            train_neighbors = [n for n in neighbors if data.train_mask[n]]
            
            if len(train_neighbors) > 0:
                # 隣接ノードのラベルエンベディングを取得
                neighbor_embeddings = []
                for neighbor in train_neighbors:
                    label = data.y[neighbor].item()
                    if label in label_to_idx:
                        embedding = co_label_embeddings[label_to_idx[label]]
                        neighbor_embeddings.append(embedding)
                
                if len(neighbor_embeddings) > 0:
                    # 平均を計算
                    avg_embedding = np.mean(neighbor_embeddings, axis=0)
                    co_label_features[node_idx] = torch.tensor(avg_embedding, dtype=torch.float32)
        else:
            # 全ての隣接ノードのラベル情報を使用
            neighbor_embeddings = []
            for neighbor in neighbors:
                label = data.y[neighbor].item()
                if label in label_to_idx:
                    embedding = co_label_embeddings[label_to_idx[label]]
                    neighbor_embeddings.append(embedding)
            
            if len(neighbor_embeddings) > 0:
                # 平均を計算
                avg_embedding = np.mean(neighbor_embeddings, axis=0)
                co_label_features[node_idx] = torch.tensor(avg_embedding, dtype=torch.float32)
    
    print(f"共起ラベルエンベディング特徴量の形状: {co_label_features.shape}")
    
    return co_label_features.to(device), label_cooccurrence_matrix

def create_combined_features_with_pca_and_co_label(data, device, max_hops=2, exclude_test_labels=True, 
                                                 pca_components=50, co_label_embedding_dim=32, 
                                                 co_label_window_size=1, co_label_max_hops=1, original_features=None):
    """
    PCAで次元圧縮した元の特徴量、隣接ノードのラベル特徴量、共起ラベルエンベディングを結合する関数
    
    Args:
        data: PyTorch Geometric データオブジェクト
        device (torch.device): デバイス
        max_hops (int): 最大hop数（デフォルト: 2）
        exclude_test_labels (bool): テスト・検証ノードのラベルを隣接ノードの特徴量計算から除外するか（デフォルト: True）
        pca_components (int): PCAで圧縮する次元数（デフォルト: 50）
        co_label_embedding_dim (int): 共起ラベルエンベディングの次元数（デフォルト: 32）
        co_label_window_size (int): 共起を計算するウィンドウサイズ（デフォルト: 1）
        co_label_max_hops (int): 共起ラベルエンベディングの最大hop数（デフォルト: 1）
        original_features (torch.Tensor, optional): 元の特徴量（Noneの場合はdata.xを使用）
    
    Returns:
<<<<<<< HEAD
        data: 特徴量が設定されたデータオブジェクト
        adj_matrix: 隣接行列
        one_hot_labels: ワンホットエンコーディングされたラベル
=======
        data: PCA特徴量が設定されたデータオブジェクト
>>>>>>> 3c0a64f3ac2f336559879e8599b3d689d84e14a1
        pca_features: PCA圧縮された特徴量
        co_label_features: 共起ラベルエンベディング特徴量
        label_cooccurrence_matrix: ラベル共起行列
    """
    if original_features is None:
        original_features = data.x
<<<<<<< HEAD
    
    print(f"=== 統合特徴量作成開始 ===")
=======

>>>>>>> 3c0a64f3ac2f336559879e8599b3d689d84e14a1
    print(f"元の特徴量の形状: {original_features.shape}")
    print(f"PCA圧縮後の次元数: {pca_components}")
    print(f"共起ラベルエンベディング次元: {co_label_embedding_dim}")
    
    # 訓練データのラベルのみを使用してワンホットエンコーディングを作成
    train_labels = data.y[data.train_mask].cpu().numpy().reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_labels)
    
    # 全ノードのラベルをワンホットエンコーディング
    all_labels = data.y.cpu().numpy().reshape(-1, 1)
    one_hot_labels = encoder.transform(all_labels)
    
    if exclude_test_labels:
        # テスト・検証ノードのラベル情報を隣接ノードの特徴量計算から除外する場合
        print("テスト・検証ノードのラベル情報を隣接ノードの特徴量計算から除外します")
        print(f"テスト・検証ノード数: {(~data.train_mask).sum().item()}")
        print(f"訓練ノード数: {data.train_mask.sum().item()}")
        
        # テスト・検証ノードのラベルを無効化（特徴量計算時に使用しない）
        test_val_mask = ~data.train_mask
        if test_val_mask.sum() > 0:
            # テスト・検証ノードのワンホットエンコーディングをゼロベクトルに設定
            one_hot_labels[test_val_mask] = 0
    else:
        # 従来の方法: テスト・検証ノードをunknownクラスとして明示的に設定
        unknown_mask = ~data.train_mask
        if unknown_mask.sum() > 0:
            # unknownクラスのワンホットエンコーディングを作成
            original_classes = one_hot_labels.shape[1]
            unknown_encoding = np.zeros((one_hot_labels.shape[0], original_classes + 1))
            
            # 訓練データのノードは元のエンコーディングを保持
            unknown_encoding[data.train_mask, :original_classes] = one_hot_labels[data.train_mask]
            
            # テスト・検証ノードはunknownクラスとして設定 [0,0,...,0,1]
            unknown_encoding[unknown_mask, original_classes] = 1
            
            one_hot_labels = unknown_encoding
            print(f"Unknownクラスを追加: {original_classes} → {original_classes + 1} クラス")
            print(f"テスト・検証ノード数: {unknown_mask.sum().item()}")
            print(f"訓練ノード数: {data.train_mask.sum().item()}")
            
        else:
            # テスト・検証ノードがない場合は、unknownクラスを追加しない
            print("テスト・検証ノードがないため、unknownクラスは追加しません")
    
    # 全ノードの特徴量を使用してPCAを学習（実験前なので全ノード使用可能）
    all_features = original_features.cpu().numpy()
    
    # PCAを適用
    max_components = min(pca_components, all_features.shape[1])
    pca = PCA(n_components=max_components)
    pca.fit(all_features)
    
    # 全ノードの特徴量をPCAで圧縮
    pca_features = pca.transform(all_features)
    
    print(f"PCA圧縮後の特徴量の形状: {pca_features.shape}")
    print(f"説明分散比: {pca.explained_variance_ratio_.sum():.4f}")
    
    # 隣接ノードのラベル特徴量を作成
    data_with_neighbor_features, adj_matrix, _ = create_neighbor_lable_features(
        data, device, max_hops, exclude_test_labels
    )
    
<<<<<<< HEAD
    # 共起ラベルエンベディングを作成
    co_label_features, label_cooccurrence_matrix = create_co_label_embeddings(
        data, device, co_label_embedding_dim, co_label_window_size, co_label_max_hops, exclude_test_labels
    )
    
    # 特徴量を結合: PCA + 隣接ノード特徴量 + 共起ラベルエンベディング
    pca_tensor = torch.tensor(pca_features, dtype=torch.float32)
    neighbor_features = data_with_neighbor_features.x
    
    combined_features = torch.cat([pca_tensor, neighbor_features, co_label_features], dim=1)
    
    print(f"結合後の特徴量の形状: {combined_features.shape}")
    print(f"  - PCA特徴量: {pca_tensor.shape[1]}次元")
    print(f"  - 隣接ノード特徴量: {neighbor_features.shape[1]}次元")
    print(f"  - 共起ラベルエンベディング: {co_label_features.shape[1]}次元")
    
    # 特徴量を設定
    data.x = combined_features.to(device)
    
    return data, adj_matrix, one_hot_labels, pca_features, co_label_features, label_cooccurrence_matrix

def display_co_label_embeddings_info(co_label_features, label_cooccurrence_matrix, dataset_name, sample_nodes=None):
    """
    共起ラベルエンベディングの情報を表示する関数
    
    Args:
        co_label_features: 共起ラベルエンベディング特徴量
        label_cooccurrence_matrix: ラベル共起行列
        dataset_name (str): データセット名
        sample_nodes (list): 表示するノードのリスト（デフォルト: [0, 10, 50, 100, 200]）
    """
    
    if sample_nodes is None:
        sample_nodes = [0, 10, 50, 100, 200]
    
    print(f"\n=== {dataset_name} 共起ラベルエンベディング情報 ===")
    print(f"共起ラベルエンベディングの形状: {co_label_features.shape}")
    print(f"ラベル共起行列の形状: {label_cooccurrence_matrix.shape}")
    
    # 共起行列の統計情報
    print(f"共起行列の非ゼロ要素数: {np.count_nonzero(label_cooccurrence_matrix)}")
    print(f"共起行列の最大値: {label_cooccurrence_matrix.max():.4f}")
    print(f"共起行列の最小値: {label_cooccurrence_matrix.min():.4f}")
    print(f"共起行列の平均値: {label_cooccurrence_matrix.mean():.4f}")
    
    # 最も強い共起関係を持つラベルペアを表示
    cooccurrence_pairs = []
    for i in range(label_cooccurrence_matrix.shape[0]):
        for j in range(i+1, label_cooccurrence_matrix.shape[1]):
            if label_cooccurrence_matrix[i, j] > 0:
                cooccurrence_pairs.append((i, j, label_cooccurrence_matrix[i, j]))
    
    # 共起強度でソート
    cooccurrence_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n最も強い共起関係を持つラベルペア（上位5個）:")
    for i, (label1, label2, strength) in enumerate(cooccurrence_pairs[:5]):
        print(f"  ラベル {label1} - ラベル {label2}: {strength:.4f}")
    
    # いくつかのノードの共起特徴量を表示
    print(f"\nサンプルノードの共起ラベルエンベディング:")
    for node_idx in sample_nodes:
        if node_idx < co_label_features.shape[0]:
            embedding = co_label_features[node_idx].cpu().numpy()
            print(f"ノード {node_idx}: {embedding[:10]}...")  # 最初の10次元のみ表示

# 使用例:
"""
# 1. 基本的な共起ラベルエンベディングの作成（1-hop）
co_label_features, label_cooccurrence_matrix = create_co_label_embeddings(
    data, device, embedding_dim=32, window_size=1, max_hops=1, exclude_test_labels=True
)

# 2. 多hop共起ラベルエンベディングの作成（2-hop）
co_label_features, label_cooccurrence_matrix = create_co_label_embeddings(
    data, device, embedding_dim=32, window_size=1, max_hops=2, exclude_test_labels=True
)

# 3. 統合特徴量の作成（PCA + 隣接ノード特徴量 + 多hop共起ラベルエンベディング）
data, adj_matrix, one_hot_labels, pca_features, co_label_features, label_cooccurrence_matrix = \
    create_combined_features_with_pca_and_co_label(
        data, device, 
        max_hops=2, 
        exclude_test_labels=True,
        pca_components=50, 
        co_label_embedding_dim=32, 
        co_label_window_size=1,
        co_label_max_hops=2
    )

# 4. 共起ラベルエンベディングの情報表示
display_co_label_embeddings_info(co_label_features, label_cooccurrence_matrix, "Dataset Name")

# 5. 特徴量の情報取得
feature_info = get_feature_info(data, one_hot_labels, max_hops=2)
print(f"特徴量情報: {feature_info}")
"""
=======
    return data, pca_features, pca

def create_label_features(data, device, max_hops=2, exclude_test_labels=True, use_neighbor_label_features=True):
    """
    隣接ノードのラベル特徴量を作成する関数（実験中に実行）
    
    Args:
        data: PyTorch Geometric データオブジェクト
        device (torch.device): デバイス
        max_hops (int): 最大hop数（デフォルト: 2）
        exclude_test_labels (bool): テスト・検証ノードのラベルを隣接ノードの特徴量計算から除外するか（デフォルト: True）
        use_neighbor_label_features (bool): Trueなら隣接ノードのラベル特徴量を結合、Falseなら結合しない
    
    Returns:
        data: 特徴量が設定されたデータオブジェクト
        adj_matrix: 隣接行列
        one_hot_labels: ワンホットエンコーディングされたラベル
    """
    print(f"現在の特徴量の形状: {data.x.shape}")
    
    # ワンホットエンコーディングの作成
    train_labels = data.y[data.train_mask].cpu().numpy().reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_labels)
    all_labels = data.y.cpu().numpy().reshape(-1, 1)
    one_hot_labels = encoder.transform(all_labels)
    
    if exclude_test_labels:
        test_val_mask = ~data.train_mask
        if test_val_mask.sum() > 0:
            one_hot_labels[test_val_mask] = 0
    else:
        unknown_mask = ~data.train_mask
        if unknown_mask.sum() > 0:
            original_classes = one_hot_labels.shape[1]
            unknown_encoding = np.zeros((one_hot_labels.shape[0], original_classes + 1))
            unknown_encoding[data.train_mask, :original_classes] = one_hot_labels[data.train_mask]
            unknown_encoding[unknown_mask, original_classes] = 1
            one_hot_labels = unknown_encoding
            print(f"Unknownクラスを追加: {original_classes} → {original_classes + 1} クラス")
            print(f"テスト・検証ノード数: {unknown_mask.sum().item()}")
            print(f"訓練ノード数: {data.train_mask.sum().item()}")
        else:
            print("テスト・検証ノードがないため、unknownクラスは追加しません")
    
    # 隣接ノード特徴量の処理
    if use_neighbor_label_features:
        print("隣接ノードのラベル特徴量を結合します")
        
        # 隣接ノード特徴量を作成
        edge_index = data.edge_index.cpu()
        num_nodes = data.num_nodes
        
        # 各ノードの隣接ノードリストを作成
        neighbors_dict = {}
        for i in range(num_nodes):
            neighbors_dict[i] = []
        
        # エッジから隣接関係を構築
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src != dst:  # 自己ループを除外
                neighbors_dict[src].append(dst)
                neighbors_dict[dst].append(src)  # 無向グラフとして扱う

        # 各hopの特徴量を格納するリスト
        hop_features_list = []

        # 各hopの特徴量を計算
        for hop in range(1, max_hops + 1):
            hop_features = torch.zeros((num_nodes, one_hot_labels.shape[1]), dtype=torch.float32)
            
            for i in range(num_nodes):
                # n-hop隣接ノードを取得
                if hop == 1:
                    # 1-hop: 直接隣接
                    neighbors = neighbors_dict[i]
                else:
                    # 2-hop: 隣接ノードの隣接ノード（重複を除く）
                    neighbors = set()
                    for neighbor in neighbors_dict[i]:
                        neighbors.update(neighbors_dict[neighbor])
                    # 1-hop隣接ノードを除外
                    neighbors = neighbors - set(neighbors_dict[i])
                    neighbors = list(neighbors)
                
                # 自分自身を除外
                if i in neighbors:
                    neighbors.remove(i)
                
                if len(neighbors) > 0:
                    if exclude_test_labels:
                        # テスト・検証ノードのラベル情報を除外する場合
                        # 訓練ノードのみのラベル情報を使用
                        train_neighbors = [n for n in neighbors if data.train_mask[n]]
                        
                        if len(train_neighbors) > 0:
                            # 訓練ノードのワンホットエンコーディングを取得
                            neighbor_one_hot = torch.tensor(one_hot_labels[train_neighbors], dtype=torch.float32)
                            # 平均を計算
                            hop_features[i] = neighbor_one_hot.mean(dim=0)
                        else:
                            # 訓練ノードの隣接ノードがない場合はゼロベクトル
                            hop_features[i] = torch.zeros(one_hot_labels.shape[1], dtype=torch.float32)
                    else:
                        # 従来の方法: 全ての隣接ノードのラベル情報を使用
                        neighbor_one_hot = torch.tensor(one_hot_labels[neighbors], dtype=torch.float32)
                        # 平均を計算
                        hop_features[i] = neighbor_one_hot.mean(dim=0)
            
            hop_features_list.append(hop_features)

        # 全てのhopの特徴量を結合
        neighbor_features = torch.cat(hop_features_list, dim=1)
        
        # 隣接行列を作成
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        adj_matrix[edge_index[0], edge_index[1]] = 1.0
        
        # 現在の特徴量（PCA済み）と隣接ノード特徴量を結合
        combined_features = torch.cat([data.x, neighbor_features], dim=1)
        print(f"結合後の特徴量の形状: {combined_features.shape}")
        print(f"  - 現在の特徴量: {data.x.shape[1]}次元")
        print(f"  - 隣接ノード特徴量: {neighbor_features.shape[1]}次元")
    else:
        print("隣接ノードのラベル特徴量は結合しません")
        # 隣接行列のみ作成（GSLモデルで必要）
        edge_index = data.edge_index.cpu()
        num_nodes = data.num_nodes
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        adj_matrix[edge_index[0], edge_index[1]] = 1.0
        combined_features = data.x
        print(f"最終特徴量の形状: {combined_features.shape}")
        print(f"  - 現在の特徴量: {data.x.shape[1]}次元")
    
    data.x = combined_features.to(device)
    return data, adj_matrix, one_hot_labels

def create_combined_features(data, device, max_hops=2, exclude_test_labels=True, 
                            pca_components=50, use_pca=True, use_neighbor_label_features=True, original_features=None):
    """
    PCAで次元圧縮した元の特徴量または生の特徴量と隣接ノードのラベル特徴量を結合する関数（後方互換性のため残す）
    
    Args:
        data: PyTorch Geometric データオブジェクト
        device (torch.device): デバイス
        max_hops (int): 最大hop数（デフォルト: 2）
        exclude_test_labels (bool): テスト・検証ノードのラベルを隣接ノードの特徴量計算から除外するか（デフォルト: True）
        pca_components (int): PCAで圧縮する次元数（デフォルト: 50）
        use_pca (bool): TrueならPCA圧縮、Falseなら生の特徴量を使用
        use_neighbor_label_features (bool): Trueなら隣接ノードのラベル特徴量を結合、Falseなら結合しない
        original_features (torch.Tensor, optional): 元の特徴量（Noneの場合はdata.xを使用）
    
    Returns:
        data: 特徴量が設定されたデータオブジェクト
        adj_matrix: 隣接行列
        one_hot_labels: ワンホットエンコーディングされたラベル
        pca_features: PCA圧縮された特徴量またはNone
    """
    if original_features is None:
        original_features = data.x

    print(f"元の特徴量の形状: {original_features.shape}")
    
    # ワンホットエンコーディングの作成（隣接ノード特徴量を使う場合のみ必要）
    if use_neighbor_label_features:
        train_labels = data.y[data.train_mask].cpu().numpy().reshape(-1, 1)
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(train_labels)
        all_labels = data.y.cpu().numpy().reshape(-1, 1)
        one_hot_labels = encoder.transform(all_labels)
        
        if exclude_test_labels:
            test_val_mask = ~data.train_mask
            if test_val_mask.sum() > 0:
                one_hot_labels[test_val_mask] = 0
        else:
            unknown_mask = ~data.train_mask
            if unknown_mask.sum() > 0:
                original_classes = one_hot_labels.shape[1]
                unknown_encoding = np.zeros((one_hot_labels.shape[0], original_classes + 1))
                unknown_encoding[data.train_mask, :original_classes] = one_hot_labels[data.train_mask]
                unknown_encoding[unknown_mask, original_classes] = 1
                one_hot_labels = unknown_encoding
                print(f"Unknownクラスを追加: {original_classes} → {original_classes + 1} クラス")
                print(f"テスト・検証ノード数: {unknown_mask.sum().item()}")
                print(f"訓練ノード数: {data.train_mask.sum().item()}")
            else:
                print("テスト・検証ノードがないため、unknownクラスは追加しません")
    else:
        # 隣接ノード特徴量を使わない場合は、最小限のワンホットエンコーディングを作成
        train_labels = data.y[data.train_mask].cpu().numpy().reshape(-1, 1)
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(train_labels)
        all_labels = data.y.cpu().numpy().reshape(-1, 1)
        one_hot_labels = encoder.transform(all_labels)
    
    # 元の特徴量の処理
    if use_pca:
        print(f"PCA圧縮後の次元数: {pca_components}")
        train_features = original_features[data.train_mask].cpu().numpy()
        max_components = min(pca_components, train_features.shape[1])
        if train_features.shape[0] < max_components:
            print(f"警告: 訓練ノード数({train_features.shape[0]})が要求次元数({max_components})より少ないため、")
            print(f"元の特徴量次元({train_features.shape[1]})まで使用可能にします")
            max_components = min(pca_components, train_features.shape[1])
        pca = PCA(n_components=max_components)
        pca.fit(train_features)
        all_features = original_features.cpu().numpy()
        pca_features = pca.transform(all_features)
        print(f"PCA圧縮後の特徴量の形状: {pca_features.shape}")
        print(f"説明分散比: {pca.explained_variance_ratio_.sum():.4f}")
        pca_tensor = torch.tensor(pca_features, dtype=torch.float32)
    else:
        print("PCAを使わず生の特徴量を結合します")
        pca_tensor = original_features.cpu().float()
        pca_features = None
    
    # 隣接ノード特徴量の処理
    if use_neighbor_label_features:
        print("隣接ノードのラベル特徴量を結合します")
        
        # 隣接ノード特徴量を作成
        edge_index = data.edge_index.cpu()
        num_nodes = data.num_nodes
        
        # 各ノードの隣接ノードリストを作成
        neighbors_dict = {}
        for i in range(num_nodes):
            neighbors_dict[i] = []
        
        # エッジから隣接関係を構築
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src != dst:  # 自己ループを除外
                neighbors_dict[src].append(dst)
                neighbors_dict[dst].append(src)  # 無向グラフとして扱う

        # 各hopの特徴量を格納するリスト
        hop_features_list = []

        # 各hopの特徴量を計算
        for hop in range(1, max_hops + 1):
            hop_features = torch.zeros((num_nodes, one_hot_labels.shape[1]), dtype=torch.float32)
            
            for i in range(num_nodes):
                # n-hop隣接ノードを取得
                if hop == 1:
                    # 1-hop: 直接隣接
                    neighbors = neighbors_dict[i]
                else:
                    # 2-hop: 隣接ノードの隣接ノード（重複を除く）
                    neighbors = set()
                    for neighbor in neighbors_dict[i]:
                        neighbors.update(neighbors_dict[neighbor])
                    # 1-hop隣接ノードを除外
                    neighbors = neighbors - set(neighbors_dict[i])
                    neighbors = list(neighbors)
                
                # 自分自身を除外
                if i in neighbors:
                    neighbors.remove(i)
                
                if len(neighbors) > 0:
                    if exclude_test_labels:
                        # テスト・検証ノードのラベル情報を除外する場合
                        # 訓練ノードのみのラベル情報を使用
                        train_neighbors = [n for n in neighbors if data.train_mask[n]]
                        
                        if len(train_neighbors) > 0:
                            # 訓練ノードのワンホットエンコーディングを取得
                            neighbor_one_hot = torch.tensor(one_hot_labels[train_neighbors], dtype=torch.float32)
                            # 平均を計算
                            hop_features[i] = neighbor_one_hot.mean(dim=0)
                        else:
                            # 訓練ノードの隣接ノードがない場合はゼロベクトル
                            hop_features[i] = torch.zeros(one_hot_labels.shape[1], dtype=torch.float32)
                    else:
                        # 従来の方法: 全ての隣接ノードのラベル情報を使用
                        neighbor_one_hot = torch.tensor(one_hot_labels[neighbors], dtype=torch.float32)
                        # 平均を計算
                        hop_features[i] = neighbor_one_hot.mean(dim=0)
            
            hop_features_list.append(hop_features)

        # 全てのhopの特徴量を結合
        neighbor_features = torch.cat(hop_features_list, dim=1)
        
        # 隣接行列を作成
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        adj_matrix[edge_index[0], edge_index[1]] = 1.0
        
        combined_features = torch.cat([pca_tensor, neighbor_features], dim=1)
        print(f"結合後の特徴量の形状: {combined_features.shape}")
        print(f"  - {'PCA' if use_pca else '生'}特徴量: {pca_tensor.shape[1]}次元")
        print(f"  - 隣接ノード特徴量: {neighbor_features.shape[1]}次元")
    else:
        print("隣接ノードのラベル特徴量は結合しません")
        # 隣接行列のみ作成（GSLモデルで必要）
        edge_index = data.edge_index.cpu()
        num_nodes = data.num_nodes
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        adj_matrix[edge_index[0], edge_index[1]] = 1.0
        combined_features = pca_tensor
        print(f"最終特徴量の形状: {combined_features.shape}")
        print(f"  - {'PCA' if use_pca else '生'}特徴量: {pca_tensor.shape[1]}次元")
    
    data.x = combined_features.to(device)
    return data, adj_matrix, one_hot_labels, pca_features 
>>>>>>> 3c0a64f3ac2f336559879e8599b3d689d84e14a1
