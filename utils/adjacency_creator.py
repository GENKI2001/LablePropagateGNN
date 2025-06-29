import torch
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data


def create_normalized_adjacency_matrices(data: Data, device: torch.device, max_hops: int = 2):
    """
    正規化された隣接行列を作成する関数
    
    Args:
        data: PyTorch Geometricのデータオブジェクト
        device: 計算デバイス
        max_hops: 作成する最大hop数（デフォルト: 2）
    
    Returns:
        dict: 各hopの正規化された隣接行列を含む辞書
        {
            'adj_1hop': 1hop隣接行列（スパーステンソル）,
            'adj_2hop': 2hop隣接行列（スパーステンソル）,
            ...
        }
    """
    print(f"=== 正規化隣接行列作成 ===")
    print(f"ノード数: {data.num_nodes}")
    print(f"エッジ数: {data.edge_index.shape[1]}")
    print(f"最大hop数: {max_hops}")
    
    adjacency_matrices = {}
    
    # 1hop隣接行列（自己ループ付き）
    print(f"1hop隣接行列を作成中...")
    edge_index_1hop, _ = add_self_loops(data.edge_index)
    row, col = edge_index_1hop
    deg_1hop = degree(row, data.num_nodes, dtype=torch.float32)
    deg_inv_1hop = 1.0 / deg_1hop
    deg_inv_1hop[deg_inv_1hop == float('inf')] = 0
    norm_1hop = deg_inv_1hop[row]
    adj_1hop = torch.sparse_coo_tensor(
        indices=edge_index_1hop,
        values=norm_1hop,
        size=(data.num_nodes, data.num_nodes)
    ).to(device)
    
    adjacency_matrices['adj_1hop'] = adj_1hop
    print(f"  1hop隣接行列: {adj_1hop.shape}, 非ゼロ要素: {adj_1hop._nnz()}")
    
    # 2hop以上の隣接行列を作成
    if max_hops >= 2:
        print(f"2hop隣接行列を作成中...")
        adj_1hop_dense = adj_1hop.to_dense()
        adj_2hop_dense = torch.mm(adj_1hop_dense, adj_1hop_dense)
        
        # 正規化（行方向の合計が1になるように）
        row_sums_2hop = adj_2hop_dense.sum(dim=1, keepdim=True)
        row_sums_2hop[row_sums_2hop == 0] = 1  # ゼロ除算を防ぐ
        adj_2hop_normalized = adj_2hop_dense / row_sums_2hop
        adj_2hop = adj_2hop_normalized.to_sparse().to(device)
        
        adjacency_matrices['adj_2hop'] = adj_2hop
        print(f"  2hop隣接行列: {adj_2hop.shape}, 非ゼロ要素: {adj_2hop._nnz()}")
    
    # 3hop以上の隣接行列を作成（必要に応じて）
    if max_hops >= 3:
        print(f"3hop隣接行列を作成中...")
        adj_3hop_dense = torch.mm(adj_2hop_dense, adj_1hop_dense)
        
        # 正規化
        row_sums_3hop = adj_3hop_dense.sum(dim=1, keepdim=True)
        row_sums_3hop[row_sums_3hop == 0] = 1
        adj_3hop_normalized = adj_3hop_dense / row_sums_3hop
        adj_3hop = adj_3hop_normalized.to_sparse().to(device)
        
        adjacency_matrices['adj_3hop'] = adj_3hop
        print(f"  3hop隣接行列: {adj_3hop.shape}, 非ゼロ要素: {adj_3hop._nnz()}")
    
    # 4hop以上の隣接行列を作成（必要に応じて）
    if max_hops >= 4:
        print(f"4hop隣接行列を作成中...")
        adj_4hop_dense = torch.mm(adj_3hop_dense, adj_1hop_dense)
        
        # 正規化
        row_sums_4hop = adj_4hop_dense.sum(dim=1, keepdim=True)
        row_sums_4hop[row_sums_4hop == 0] = 1
        adj_4hop_normalized = adj_4hop_dense / row_sums_4hop
        adj_4hop = adj_4hop_normalized.to_sparse().to(device)
        
        adjacency_matrices['adj_4hop'] = adj_4hop
        print(f"  4hop隣接行列: {adj_4hop.shape}, 非ゼロ要素: {adj_4hop._nnz()}")
    
    print(f"隣接行列作成完了 - {len(adjacency_matrices)}個の隣接行列が利用可能")
    return adjacency_matrices


def get_adjacency_matrix(adjacency_matrices: dict, hop: int):
    """
    指定されたhop数の隣接行列を取得する関数
    
    Args:
        adjacency_matrices: create_normalized_adjacency_matricesで作成された辞書
        hop: 取得したいhop数（1, 2, 3, 4, ...）
    
    Returns:
        torch.Tensor: 指定されたhop数の隣接行列（スパーステンソル）
    """
    key = f'adj_{hop}hop'
    if key in adjacency_matrices:
        return adjacency_matrices[key]
    else:
        raise KeyError(f"Hop {hop}の隣接行列が見つかりません。利用可能なhop: {list(adjacency_matrices.keys())}")


def apply_adjacency_to_features(adjacency_matrix: torch.Tensor, features: torch.Tensor):
    """
    隣接行列を特徴量に適用する関数
    
    Args:
        adjacency_matrix: 正規化された隣接行列（スパーステンソル）
        features: 入力特徴量テンソル
    
    Returns:
        torch.Tensor: 隣接ノードの特徴量を集約したテンソル
    """
    return torch.sparse.mm(adjacency_matrix, features)


def combine_hop_features(adjacency_matrices: dict, features: torch.Tensor, hops: list = [1, 2]):
    """
    複数のhopの特徴量を結合する関数
    
    Args:
        adjacency_matrices: create_normalized_adjacency_matricesで作成された辞書
        features: 入力特徴量テンソル
        hops: 結合するhop数のリスト（デフォルト: [1, 2]）
    
    Returns:
        torch.Tensor: 複数hopの特徴量を結合したテンソル
    """
    hop_features = []
    
    for hop in hops:
        try:
            adj_matrix = get_adjacency_matrix(adjacency_matrices, hop)
            hop_feature = apply_adjacency_to_features(adj_matrix, features)
            hop_features.append(hop_feature)
        except KeyError as e:
            print(f"警告: {e}")
            continue
    
    if hop_features:
        return torch.cat(hop_features, dim=1)
    else:
        return features


def print_adjacency_info(adjacency_matrices: dict):
    """
    隣接行列の情報を表示する関数
    
    Args:
        adjacency_matrices: create_normalized_adjacency_matricesで作成された辞書
    """
    print(f"\n=== 隣接行列情報 ===")
    for key, adj_matrix in adjacency_matrices.items():
        print(f"{key}: {adj_matrix.shape}, 非ゼロ要素: {adj_matrix._nnz()}")
        if hasattr(adj_matrix, 'density'):
            density = adj_matrix._nnz() / (adj_matrix.shape[0] * adj_matrix.shape[1])
            print(f"  密度: {density:.6f}") 