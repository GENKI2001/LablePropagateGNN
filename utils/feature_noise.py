import torch
import numpy as np


def add_feature_noise(features, noise_percentage, device=None):
    """
    特徴量にノイズを追加する関数（0と1を入れ替え）
    
    Args:
        features (torch.Tensor): 入力特徴量テンソル (num_nodes, num_features)
        noise_percentage (float): ノイズを追加する特徴量の割合 (0.0-1.0)
        device (torch.device, optional): デバイス
    
    Returns:
        torch.Tensor: ノイズが追加された特徴量テンソル
        dict: ノイズ情報（追加されたノイズの数、総特徴量数など）
    """
    if device is None:
        device = features.device
    
    # 特徴量をデバイスに移動
    features = features.to(device)
    
    # 特徴量の形状を取得
    num_nodes, num_features = features.shape
    
    # ノイズを追加する特徴量の数を計算
    num_noise_features = int(noise_percentage * num_features)
    
    # ノイズ情報を初期化
    noise_info = {
        'noise_percentage': noise_percentage,
        'num_features': num_features,
        'num_noise_features': num_noise_features,
        'noise_indices': [],
        'noise_type': 'per_node_flip'
    }
    
    # ノイズを追加する特徴量のインデックスをランダムに選択
    if num_noise_features > 0:
        # 各ノードでランダムに特徴量を選択して0と1を入れ替える
        noise_features = features.clone()
        
        # 各ノードでランダムに特徴量インデックスを選択
        for node_idx in range(num_nodes):
            # ランダムに特徴量インデックスを選択
            feature_indices = torch.randperm(num_features, device=device)[:num_noise_features]
            
            # 選択された特徴量で0と1を入れ替え
            for feat_idx in feature_indices:
                if noise_features[node_idx, feat_idx] == 0:
                    noise_features[node_idx, feat_idx] = 1
                elif noise_features[node_idx, feat_idx] == 1:
                    noise_features[node_idx, feat_idx] = 0
                # 0でも1でもない値はそのまま保持
            
            # ノイズ情報を記録（最初のノードのみ）
            if node_idx == 0:
                noise_info['noise_indices'] = feature_indices.cpu().tolist()
        
        return noise_features, noise_info
    else:
        # ノイズを追加しない場合は元の特徴量をそのまま返す
        return features, noise_info


def add_feature_noise_uniform(features, noise_percentage, device=None):
    """
    特徴量にノイズを追加する関数（一様分布版、0と1を入れ替え）
    全ノードで同じ特徴量インデックスにノイズを追加
    
    Args:
        features (torch.Tensor): 入力特徴量テンソル (num_nodes, num_features)
        noise_percentage (float): ノイズを追加する特徴量の割合 (0.0-1.0)
        device (torch.device, optional): デバイス
    
    Returns:
        torch.Tensor: ノイズが追加された特徴量テンソル
        dict: ノイズ情報（追加されたノイズの数、総特徴量数など）
    """
    if device is None:
        device = features.device
    
    # 特徴量をデバイスに移動
    features = features.to(device)
    
    # 特徴量の形状を取得
    num_nodes, num_features = features.shape
    
    # ノイズを追加する特徴量の数を計算
    num_noise_features = int(noise_percentage * num_features)
    
    # ノイズ情報を初期化
    noise_info = {
        'noise_percentage': noise_percentage,
        'num_features': num_features,
        'num_noise_features': num_noise_features,
        'noise_indices': [],
        'noise_type': 'uniform_flip'
    }
    
    # ノイズを追加する特徴量のインデックスをランダムに選択
    if num_noise_features > 0:
        # ランダムに特徴量インデックスを選択（全ノードで同じ）
        feature_indices = torch.randperm(num_features, device=device)[:num_noise_features]
        
        # 特徴量をコピーしてノイズを追加
        noise_features = features.clone()
        
        # 選択された特徴量で0と1を入れ替え
        for feat_idx in feature_indices:
            # 0の場合は1に、1の場合は0に変更
            mask_0 = noise_features[:, feat_idx] == 0
            mask_1 = noise_features[:, feat_idx] == 1
            noise_features[mask_0, feat_idx] = 1
            noise_features[mask_1, feat_idx] = 0
            # 0でも1でもない値はそのまま保持
        
        # ノイズ情報を記録
        noise_info['noise_indices'] = feature_indices.cpu().tolist()
        
        return noise_features, noise_info
    else:
        # ノイズを追加しない場合は元の特徴量をそのまま返す
        return features, noise_info


def add_feature_noise_random(features, noise_percentage, device=None):
    """
    特徴量にノイズを追加する関数（完全ランダム版、0と1を入れ替え）
    各ノード・各特徴量で独立にノイズを追加
    
    Args:
        features (torch.Tensor): 入力特徴量テンソル (num_nodes, num_features)
        noise_percentage (float): ノイズを追加する特徴量の割合 (0.0-1.0)
        device (torch.device, optional): デバイス
    
    Returns:
        torch.Tensor: ノイズが追加された特徴量テンソル
        dict: ノイズ情報（追加されたノイズの数、総特徴量数など）
    """
    if device is None:
        device = features.device
    
    # 特徴量をデバイスに移動
    features = features.to(device)
    
    # 特徴量の形状を取得
    num_nodes, num_features = features.shape
    
    # ノイズ情報を初期化
    noise_info = {
        'noise_percentage': noise_percentage,
        'num_features': num_features,
        'total_elements': num_nodes * num_features,
        'noise_type': 'random_flip'
    }
    
    # ランダムマスクを作成（ノイズを追加する要素を選択）
    mask = torch.rand(num_nodes, num_features, device=device) < noise_percentage
    
    # 特徴量をコピーしてノイズを追加
    noise_features = features.clone()
    
    # マスクされた要素で0と1を入れ替え
    # 0の場合は1に、1の場合は0に変更
    mask_0 = (noise_features == 0) & mask
    mask_1 = (noise_features == 1) & mask
    noise_features[mask_0] = 1
    noise_features[mask_1] = 0
    # 0でも1でもない値はそのまま保持
    
    # ノイズ情報を記録
    noise_info['num_noise_elements'] = int(mask.sum().item())
    noise_info['actual_noise_percentage'] = noise_info['num_noise_elements'] / noise_info['total_elements']
    
    return noise_features, noise_info


def print_noise_info(noise_info, dataset_name="", run_num=None):
    """
    ノイズ情報を表示する関数
    
    Args:
        noise_info (dict): ノイズ情報
        dataset_name (str): データセット名
        run_num (int): 実験回数
    """
    prefix = ""
    if dataset_name:
        prefix += f"[{dataset_name}] "
    if run_num is not None:
        prefix += f"Run {run_num}: "
    
    print(f"{prefix}特徴量ノイズ追加:")
    print(f"  ノイズ割合: {noise_info['noise_percentage']:.1%}")
    print(f"  総特徴量数: {noise_info['num_features']}")
    
    if 'noise_type' in noise_info:
        print(f"  ノイズタイプ: {noise_info['noise_type']}")
    
    if 'num_noise_features' in noise_info:
        print(f"  ノイズ特徴量数: {noise_info['num_noise_features']}")
    
    if 'num_noise_elements' in noise_info:
        print(f"  ノイズ要素数: {noise_info['num_noise_elements']}")
        print(f"  実際のノイズ割合: {noise_info['actual_noise_percentage']:.1%}")
    
    if 'noise_indices' in noise_info and len(noise_info['noise_indices']) > 0:
        print(f"  ノイズ特徴量インデックス（最初の10個）: {noise_info['noise_indices'][:10]}")
        if len(noise_info['noise_indices']) > 10:
            print(f"    ... 他 {len(noise_info['noise_indices']) - 10} 個") 