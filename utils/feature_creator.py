import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def create_neighbor_lable_features(data, device, max_hops=2, exclude_test_labels=True):
    """
    適切なデータリーク対策を実装した特徴量作成関数（高速版）
    
    - 訓練データのラベルのみを使用してワンホットエンコーディングを作成
    - exclude_test_labels=Trueの場合: テスト・検証ノードのラベルは隣接ノードの特徴量計算に使用しない
    - exclude_test_labels=Falseの場合: テスト・検証ノードのラベルはunknownクラスとして扱う
    
    Args:
        data: PyTorch Geometric データオブジェクト
        device (torch.device): デバイス
        max_hops (int): 最大hop数（デフォルト: 2）
        exclude_test_labels (bool): テスト・検証ノードのラベルを隣接ノードの特徴量計算から除外するか（デフォルト: True）
    
    Returns:
        data: 特徴量が設定されたデータオブジェクト
        adj_matrix: 隣接行列
        one_hot_labels: ワンホットエンコーディングされたラベル
    """
    
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
            
            print("ワンホットエンコーディング例:")
            for i in range(min(10, data.num_nodes)):
                node_type = "訓練" if data.train_mask[i] else "テスト・検証"
                print(f"  ノード{i} ({node_type}): {one_hot_labels[i]}")
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
            
            # デバッグ: いくつかのノードのワンホットエンコーディングを表示
            print("ワンホットエンコーディング例:")
            for i in range(min(10, data.num_nodes)):
                node_type = "訓練" if data.train_mask[i] else "テスト・検証"
                print(f"  ノード{i} ({node_type}): {one_hot_labels[i]}")
        else:
            # テスト・検証ノードがない場合は、unknownクラスを追加しない
            print("テスト・検証ノードがないため、unknownクラスは追加しません")

    # エッジリストから隣接関係を取得（高速化のため隣接行列は作成しない）
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
    combined_features = torch.cat(hop_features_list, dim=1)

    # 特徴量を設定
    data.x = combined_features.to(device)
    
    # 隣接行列は後で必要に応じて作成
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj_matrix[edge_index[0], edge_index[1]] = 1.0
    
    return data, adj_matrix, one_hot_labels

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