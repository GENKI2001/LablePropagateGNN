import torch
import numpy as np

def add_feature_missingness(features, missing_percentage, device=None):
    """
    特徴量に欠損を追加する関数（指定割合の特徴量を0にマスキング）

    Args:
        features (torch.Tensor): 入力特徴量テンソル (num_nodes, num_features)
        missing_percentage (float): 欠損させる割合 (0.0 - 1.0)
        device (torch.device, optional): 使用するデバイス

    Returns:
        torch.Tensor: 欠損が追加された特徴量テンソル
        dict: 欠損情報（欠損数、割合など）
    """
    if device is None:
        device = features.device

    features = features.to(device)
    num_nodes, num_features = features.shape
    total_entries = num_nodes * num_features
    num_missing = int(missing_percentage * total_entries)

    # 欠損情報を保持
    missing_info = {
        'missing_percentage': missing_percentage,
        'num_features': num_features,
        'num_nodes': num_nodes,
        'total_entries': total_entries,
        'num_missing_entries': num_missing,
        'missing_indices': []
    }

    if num_missing == 0:
        return features, missing_info

    # 欠損を加える特徴量テンソルをクローン
    missing_features = features.clone()

    # ランダムに欠損位置を選択（フラットなインデックス）
    flat_indices = torch.randperm(total_entries, device=device)[:num_missing]
    row_indices = flat_indices // num_features
    col_indices = flat_indices % num_features

    # 欠損値を0に置換（あるいは torch.nan にしたい場合は torch.float に変換必須）
    missing_features[row_indices, col_indices] = 0.0

    # 欠損インデックスの記録（最初の数個）
    missing_info['missing_indices'] = list(zip(row_indices.tolist()[:10], col_indices.tolist()[:10]))

    return missing_features, missing_info

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

def apply_feature_modifications(features, modifications, device=None):
    """
    複数の特徴量改変を順次適用する関数
    
    Args:
        features (torch.Tensor): 入力特徴量テンソル (num_nodes, num_features)
        modifications (list): 改変設定のリスト
        device (torch.device, optional): 使用するデバイス
    
    Returns:
        torch.Tensor: 改変が適用された特徴量テンソル
        dict: 改変情報の辞書
    """
    if device is None:
        device = features.device
    
    features = features.to(device)
    modified_features = features.clone()
    modification_info = {
        'original_shape': features.shape,
        'modifications_applied': [],
        'total_modifications': len(modifications)
    }
    
    for i, mod_config in enumerate(modifications):
        mod_type = mod_config.get('type', 'unknown')
        percentage = mod_config.get('percentage', 0.0)
        
        print(f"  改変 {i+1}/{len(modifications)}: {mod_type} (割合: {percentage:.1%})")
        
        if mod_type == 'noise':
            method = mod_config.get('method', 'per_node')
            if method == 'uniform':
                modified_features, noise_info = add_feature_noise_uniform(modified_features, percentage, device)
            elif method == 'random':
                modified_features, noise_info = add_feature_noise_random(modified_features, percentage, device)
            elif method == 'per_node':
                modified_features, noise_info = add_feature_noise(modified_features, percentage, device)
            else:
                print(f"    警告: 未知のノイズ方法 '{method}' のため、スキップします")
                continue
            
            modification_info['modifications_applied'].append({
                'type': 'noise',
                'method': method,
                'percentage': percentage,
                'info': noise_info
            })
            
        elif mod_type == 'missingness':
            modified_features, missingness_info = add_feature_missingness(modified_features, percentage, device)
            
            modification_info['modifications_applied'].append({
                'type': 'missingness',
                'percentage': percentage,
                'info': missingness_info
            })
            
        else:
            print(f"    警告: 未知の改変タイプ '{mod_type}' のため、スキップします")
            continue
    
    modification_info['final_shape'] = modified_features.shape
    
    return modified_features, modification_info


def print_modification_info(modification_info, dataset_name="", run_num=None):
    """
    改変情報を表示する関数
    
    Args:
        modification_info (dict): 改変情報
        dataset_name (str): データセット名
        run_num (int): 実験回数
    """
    prefix = ""
    if dataset_name:
        prefix += f"[{dataset_name}] "
    if run_num is not None:
        prefix += f"Run {run_num}: "
    
    print(f"{prefix}特徴量改変適用:")
    print(f"  元の形状: {modification_info['original_shape']}")
    print(f"  最終形状: {modification_info['final_shape']}")
    print(f"  適用された改変数: {len(modification_info['modifications_applied'])}/{modification_info['total_modifications']}")
    
    for i, mod in enumerate(modification_info['modifications_applied']):
        mod_type = mod['type']
        percentage = mod['percentage']
        
        if mod_type == 'noise':
            method = mod['method']
            info = mod['info']
            print(f"  改変 {i+1}: ノイズ ({method}) - 割合: {percentage:.1%}")
            if 'num_noise_features' in info:
                print(f"    ノイズ特徴量数: {info['num_noise_features']}")
            elif 'num_noise_elements' in info:
                print(f"    ノイズ要素数: {info['num_noise_elements']}")
                print(f"    実際のノイズ割合: {info['actual_noise_percentage']:.1%}")
                
        elif mod_type == 'missingness':
            info = mod['info']
            print(f"  改変 {i+1}: 欠損 - 割合: {percentage:.1%}")
            print(f"    欠損要素数: {info['num_missing_entries']}")
            print(f"    総要素数: {info['total_entries']}") 