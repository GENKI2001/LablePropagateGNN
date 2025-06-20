import torch
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, Actor
from custom_dataset_creator import create_custom_dataset

def load_dataset(dataset_name, device, **kwargs):
    """
    データセットを読み込む関数
    
    Args:
        dataset_name (str): データセット名
        device (torch.device): デバイス
        **kwargs: カスタムデータセット用の追加パラメータ
    
    Returns:
        data: PyTorch Geometric データオブジェクト
        dataset: データセットオブジェクト
    """
    
    # カスタムデータセットの処理
    if dataset_name == 'CustomGraph':
        num_nodes = kwargs.get('num_nodes', 1200)
        target_avg_degree = kwargs.get('target_avg_degree', 3.0)
        feature_dim = kwargs.get('feature_dim', 128)
        connection_patterns = kwargs.get('connection_patterns', None)
        dataset = create_custom_dataset(
            num_nodes=num_nodes, 
            name="CustomGraph",
            target_avg_degree=target_avg_degree,
            feature_dim=feature_dim,
            connection_patterns=connection_patterns
        )
        data = dataset[0].to(device)
        return data, dataset
    
    # 既存のデータセット読み込み
    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name)
    elif dataset_name in ['Cornell', 'Texas', 'Wisconsin', 'Washington']:
        dataset = WebKB(root=f'/tmp/WebKB', name=dataset_name)
    elif dataset_name in ['Chameleon', 'Squirrel']:
        dataset = WikipediaNetwork(root=f'/tmp/WikipediaNetwork', name=dataset_name)
    elif dataset_name == 'Actor':
        dataset = Actor(root=f'/tmp/Actor')
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    data = dataset[0].to(device)

    # WebKBデータセットの場合、マスクの形式を確認して修正
    if dataset_name in ['Cornell', 'Texas', 'Wisconsin', 'Washington']:
        print(f"=== {dataset_name} データ構造確認 ===")
        print(f"data.train_mask shape: {data.train_mask.shape}")
        print(f"data.val_mask shape: {data.val_mask.shape}")
        print(f"data.test_mask shape: {data.test_mask.shape}")
        
        # WebKBデータセットでは、マスクが2次元の場合があるので1次元に変換
        if len(data.train_mask.shape) > 1:
            # 2次元マスクの場合、各行の最初の要素を使用するか、適切に変換
            if data.train_mask.shape[1] > 1:
                # 複数のマスクがある場合、最初のマスクを使用
                data.train_mask = data.train_mask[:, 0]
                data.val_mask = data.val_mask[:, 0]
                data.test_mask = data.test_mask[:, 0]
                print("複数マスクから最初のマスクを選択しました")
            else:
                data.train_mask = data.train_mask.squeeze()
                data.val_mask = data.val_mask.squeeze()
                data.test_mask = data.test_mask.squeeze()
                print("マスクを1次元に変換しました")
        
        print(f"修正後のマスク形状:")
        print(f"  train_mask: {data.train_mask.shape}")
        print(f"  val_mask: {data.val_mask.shape}")
        print(f"  test_mask: {data.test_mask.shape}")

    # WikipediaNetworkとActorデータセットの場合、マスクが存在しない場合があるので作成
    if dataset_name in ['Chameleon', 'Squirrel', 'Actor']:
        print(f"=== {dataset_name} データ構造確認 ===")
        print(f"ノード数: {data.num_nodes}")
        print(f"エッジ数: {data.edge_index.shape[1]}")
        print(f"特徴量次元: {data.x.shape[1] if hasattr(data, 'x') else 'None'}")
        print(f"ラベル数: {len(torch.unique(data.y)) if hasattr(data, 'y') else 'None'}")
        
        # マスクが存在しない場合、ランダムに分割を作成
        if not hasattr(data, 'train_mask') or data.train_mask is None:
            print("マスクが存在しないため、ランダムに分割を作成します")
            num_nodes = data.num_nodes
            indices = torch.randperm(num_nodes)
            
            # 60% 訓練、20% 検証、20% テスト
            train_size = int(0.6 * num_nodes)
            val_size = int(0.2 * num_nodes)
            
            data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            data.train_mask[indices[:train_size]] = True
            data.val_mask[indices[train_size:train_size + val_size]] = True
            data.test_mask[indices[train_size + val_size:]] = True
            
            print(f"作成されたマスク:")
            print(f"  訓練ノード数: {data.train_mask.sum().item()}")
            print(f"  検証ノード数: {data.val_mask.sum().item()}")
            print(f"  テストノード数: {data.test_mask.sum().item()}")
        
        # マスクが2次元の場合、1次元に変換
        if hasattr(data, 'train_mask') and data.train_mask is not None:
            print(f"data.train_mask shape: {data.train_mask.shape}")
            print(f"data.val_mask shape: {data.val_mask.shape}")
            print(f"data.test_mask shape: {data.test_mask.shape}")
            
            if len(data.train_mask.shape) > 1:
                # 2次元マスクの場合、各行の最初の要素を使用するか、適切に変換
                if data.train_mask.shape[1] > 1:
                    # 複数のマスクがある場合、最初のマスクを使用
                    data.train_mask = data.train_mask[:, 0]
                    data.val_mask = data.val_mask[:, 0]
                    data.test_mask = data.test_mask[:, 0]
                    print("複数マスクから最初のマスクを選択しました")
                else:
                    data.train_mask = data.train_mask.squeeze()
                    data.val_mask = data.val_mask.squeeze()
                    data.test_mask = data.test_mask.squeeze()
                    print("マスクを1次元に変換しました")
            
            print(f"修正後のマスク形状:")
            print(f"  train_mask: {data.train_mask.shape}")
            print(f"  val_mask: {data.val_mask.shape}")
            print(f"  test_mask: {data.test_mask.shape}")

    print(f"=== {dataset_name} データセット情報 ===")
    print(f"ノード数: {data.num_nodes}")
    print(f"エッジ数: {data.edge_index.shape[1]}")
    print(f"クラス数: {dataset.num_classes}")
    print(f"訓練ノード数: {data.train_mask.sum().item()}")
    print(f"検証ノード数: {data.val_mask.sum().item()}")
    print(f"テストノード数: {data.test_mask.sum().item()}")
    
    return data, dataset

def get_supported_datasets():
    """
    サポートされているデータセットのリストを返す
    
    Returns:
        dict: データセットカテゴリとそのデータセット名の辞書
    """
    return {
        'Planetoid': ['Cora', 'Citeseer', 'Pubmed'],
        'WebKB': ['Cornell', 'Texas', 'Wisconsin', 'Washington'],
        'WikipediaNetwork': ['Chameleon', 'Squirrel'],
        'Actor': ['Actor'],
        'Custom': ['CustomGraph']
    } 