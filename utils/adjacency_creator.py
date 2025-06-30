import torch
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data


def make_undirected(data: Data, device: torch.device):
    """
    データオブジェクトのedge_indexを無向グラフに修正する関数
    
    Args:
        data: PyTorch Geometricのデータオブジェクト
        device: 計算デバイス
    
    Returns:
        Data: 無向グラフに修正されたデータオブジェクト
    """
    print(f"=== エッジを無向グラフに修正 ===")
    print(f"修正前のエッジ数: {data.edge_index.shape[1]}")
    
    # 無向グラフにするために、エッジの逆方向を追加
    edge_index_undirected = torch.cat([
        data.edge_index,
        data.edge_index.flip(0)  # 逆方向のエッジを追加
    ], dim=1)
    
    # 重複エッジを除去
    edge_index_undirected = torch.unique(edge_index_undirected, dim=1)
    
    # データオブジェクトのedge_indexを更新
    data.edge_index = edge_index_undirected.to(device)
    
    print(f"修正後のエッジ数: {data.edge_index.shape[1]}")
    print(f"エッジ数増加: {data.edge_index.shape[1] - (data.edge_index.shape[1] // 2)}")
    
    return data


def create_normalized_adjacency_matrices(data: Data, device: torch.device, max_hops: int = 2):
    """
    H2GCNスタイルの正規化された隣接行列を作成する関数
    各k-hop隣接行列は「exactly k hops away」のノードのみを含み、
    他のhopの隣接行列と重複しない（disjoint）
    
    Args:
        data: PyTorch Geometricのデータオブジェクト
        device: 計算デバイス
        max_hops: 作成する最大hop数（デフォルト: 2）
    
    Returns:
        dict: 各hopの正規化された隣接行列を含む辞書
        {
            'adj_1hop': 1hop隣接行列（自己ループなし、スパーステンソル）,
            'adj_2hop': 2hop隣接行列（1hopを除く、スパーステンソル）,
            ...
        }
    """
    print(f"=== H2GCNスタイル正規化隣接行列作成 ===")
    print(f"ノード数: {data.num_nodes}")
    print(f"エッジ数: {data.edge_index.shape[1]}")
    print(f"最大hop数: {max_hops}")
    
    adjacency_matrices = {}
    
    # 無向グラフにするために、エッジの逆方向を追加
    edge_index_undirected = torch.cat([
        data.edge_index,
        data.edge_index.flip(0)  # 逆方向のエッジを追加
    ], dim=1)
    
    # 重複エッジを除去
    edge_index_undirected = torch.unique(edge_index_undirected, dim=1)
    
    print(f"無向グラフ化後のエッジ数: {edge_index_undirected.shape[1]}")
    
    # 1hop隣接行列（自己ループなし）
    print(f"1hop隣接行列を作成中...")
    # 自己ループを追加せずに直接隣接ノードのみ
    edge_index_1hop = edge_index_undirected
    
    # 次数行列の計算
    row, col = edge_index_1hop
    deg_1hop = degree(row, data.num_nodes, dtype=torch.float32)
    
    # D^(-1/2) の計算
    deg_inv_sqrt_1hop = torch.pow(deg_1hop, -0.5)
    deg_inv_sqrt_1hop[deg_inv_sqrt_1hop == float('inf')] = 0
    
    # 正規化重みの計算: D^(-1/2) * A * D^(-1/2)
    norm_1hop = deg_inv_sqrt_1hop[row] * deg_inv_sqrt_1hop[col]
    
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
        # 元の隣接行列（自己ループなし、無向グラフ）
        adj_original = torch.sparse_coo_tensor(
            indices=edge_index_undirected,
            values=torch.ones(edge_index_undirected.shape[1]),
            size=(data.num_nodes, data.num_nodes)
        ).to(device)
        
        # A^2 を計算
        adj_2hop_dense = torch.mm(adj_original.to_dense(), adj_original.to_dense())
        
        # 1-hopの接続を除去（exactly 2-hopのみを残す）
        adj_1hop_mask = adj_original.to_dense()
        adj_2hop_exact = adj_2hop_dense * (1 - adj_1hop_mask)
        
        # 自己ループも除去
        adj_2hop_exact.fill_diagonal_(0)
        
        # 2-hop隣接行列をスパース形式に変換
        adj_2hop_sparse = adj_2hop_exact.to_sparse()
        
        # 2-hop隣接行列の次数を計算
        row_2hop, col_2hop = adj_2hop_sparse.indices()
        deg_2hop = degree(row_2hop, data.num_nodes, dtype=torch.float32)
        
        # D^(-1/2) の計算（2-hop用）
        deg_inv_sqrt_2hop = torch.pow(deg_2hop, -0.5)
        deg_inv_sqrt_2hop[deg_inv_sqrt_2hop == float('inf')] = 0
        
        # 正規化重みの計算: D^(-1/2) * A^2 * D^(-1/2)
        norm_2hop = deg_inv_sqrt_2hop[row_2hop] * deg_inv_sqrt_2hop[col_2hop]
        
        adj_2hop = torch.sparse_coo_tensor(
            indices=adj_2hop_sparse.indices(),
            values=norm_2hop,
            size=(data.num_nodes, data.num_nodes)
        ).to(device)
        
        adjacency_matrices['adj_2hop'] = adj_2hop
        print(f"  2hop隣接行列: {adj_2hop.shape}, 非ゼロ要素: {adj_2hop._nnz()}")
    
    # 3hop以上の隣接行列を作成（必要に応じて）
    if max_hops >= 3:
        print(f"3hop隣接行列を作成中...")
        # A^3 を計算
        adj_3hop_dense = torch.mm(adj_2hop_dense, adj_original.to_dense())
        
        # 1-hopと2-hopの接続を除去（exactly 3-hopのみを残す）
        adj_1hop_mask = adj_original.to_dense()
        adj_2hop_mask = adj_2hop_exact
        adj_3hop_exact = adj_3hop_dense * (1 - adj_1hop_mask) * (1 - adj_2hop_mask)
        
        # 自己ループも除去
        adj_3hop_exact.fill_diagonal_(0)
        
        # 3-hop隣接行列をスパース形式に変換
        adj_3hop_sparse = adj_3hop_exact.to_sparse()
        
        # 3-hop隣接行列の次数を計算
        row_3hop, col_3hop = adj_3hop_sparse.indices()
        deg_3hop = degree(row_3hop, data.num_nodes, dtype=torch.float32)
        
        # D^(-1/2) の計算（3-hop用）
        deg_inv_sqrt_3hop = torch.pow(deg_3hop, -0.5)
        deg_inv_sqrt_3hop[deg_inv_sqrt_3hop == float('inf')] = 0
        
        # 正規化重みの計算: D^(-1/2) * A^3 * D^(-1/2)
        norm_3hop = deg_inv_sqrt_3hop[row_3hop] * deg_inv_sqrt_3hop[col_3hop]
        
        adj_3hop = torch.sparse_coo_tensor(
            indices=adj_3hop_sparse.indices(),
            values=norm_3hop,
            size=(data.num_nodes, data.num_nodes)
        ).to(device)
        
        adjacency_matrices['adj_3hop'] = adj_3hop
        print(f"  3hop隣接行列: {adj_3hop.shape}, 非ゼロ要素: {adj_3hop._nnz()}")
    
    # 4hop以上の隣接行列を作成（必要に応じて）
    if max_hops >= 4:
        print(f"4hop隣接行列を作成中...")
        # A^4 を計算
        adj_4hop_dense = torch.mm(adj_3hop_dense, adj_original.to_dense())
        
        # 1-hop、2-hop、3-hopの接続を除去（exactly 4-hopのみを残す）
        adj_1hop_mask = adj_original.to_dense()
        adj_2hop_mask = adj_2hop_exact
        adj_3hop_mask = adj_3hop_exact
        adj_4hop_exact = adj_4hop_dense * (1 - adj_1hop_mask) * (1 - adj_2hop_mask) * (1 - adj_3hop_mask)
        
        # 自己ループも除去
        adj_4hop_exact.fill_diagonal_(0)
        
        # 4-hop隣接行列をスパース形式に変換
        adj_4hop_sparse = adj_4hop_exact.to_sparse()
        
        # 4-hop隣接行列の次数を計算
        row_4hop, col_4hop = adj_4hop_sparse.indices()
        deg_4hop = degree(row_4hop, data.num_nodes, dtype=torch.float32)
        
        # D^(-1/2) の計算（4-hop用）
        deg_inv_sqrt_4hop = torch.pow(deg_4hop, -0.5)
        deg_inv_sqrt_4hop[deg_inv_sqrt_4hop == float('inf')] = 0
        
        # 正規化重みの計算: D^(-1/2) * A^4 * D^(-1/2)
        norm_4hop = deg_inv_sqrt_4hop[row_4hop] * deg_inv_sqrt_4hop[col_4hop]
        
        adj_4hop = torch.sparse_coo_tensor(
            indices=adj_4hop_sparse.indices(),
            values=norm_4hop,
            size=(data.num_nodes, data.num_nodes)
        ).to(device)
        
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

