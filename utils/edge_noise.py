import torch

def add_random_edges(edge_index, num_nodes, add_ratio=0.05, device=None):
    """
    ランダムにエッジを追加する関数

    Args:
        edge_index (torch.LongTensor): 元のエッジインデックス [2, num_edges]
        num_nodes (int): ノード数
        add_ratio (float): エッジを追加する割合 (0.0 - 1.0)
        device (torch.device): 使用するデバイス（省略時は edge_index に合わせる）

    Returns:
        torch.LongTensor: エッジを追加した新しい edge_index
        dict: 追加されたエッジ情報
    """
    if device is None:
        device = edge_index.device

    existing_edges = edge_index.size(1)
    num_add = int(existing_edges * add_ratio)

    # 元のエッジ集合をセットにして保持（無向グラフ前提でソート済みに）
    existing_set = set(map(tuple, edge_index.cpu().numpy().T))

    new_edges = set()
    while len(new_edges) < num_add:
        i = torch.randint(0, num_nodes, (1,)).item()
        j = torch.randint(0, num_nodes, (1,)).item()
        if i != j:
            e = (i, j)
            if e not in existing_set and e not in new_edges:
                new_edges.add(e)

    new_edges_tensor = torch.tensor(list(new_edges), dtype=torch.long, device=device).T
    combined_edge_index = torch.cat([edge_index, new_edges_tensor], dim=1)

    return combined_edge_index, {
        'num_added_edges': len(new_edges),
        'add_ratio': add_ratio,
        'original_edges': existing_edges,
        'final_edges': existing_edges + len(new_edges)
    }


def remove_random_edges(edge_index, remove_ratio=0.05, device=None):
    """
    ランダムにエッジを削除する関数

    Args:
        edge_index (torch.LongTensor): 元のエッジインデックス [2, num_edges]
        remove_ratio (float): 削除するエッジの割合 (0.0 - 1.0)
        device (torch.device): 使用するデバイス（省略時は edge_index に合わせる）

    Returns:
        torch.LongTensor: エッジを削除した新しい edge_index
        dict: 削除されたエッジ情報
    """
    if device is None:
        device = edge_index.device

    num_edges = edge_index.size(1)
    num_remove = int(num_edges * remove_ratio)

    # ランダムに削除するインデックスを選択
    keep_indices = torch.randperm(num_edges, device=device)[num_remove:]
    new_edge_index = edge_index[:, keep_indices]

    return new_edge_index, {
        'num_removed_edges': num_remove,
        'remove_ratio': remove_ratio,
        'original_edges': num_edges,
        'final_edges': num_edges - num_remove
    }


def apply_edge_modifications(edge_index, num_nodes, modifications, device=None):
    """
    複数のエッジ改変を順次適用する関数
    
    Args:
        edge_index (torch.LongTensor): 元のエッジインデックス [2, num_edges]
        num_nodes (int): ノード数
        modifications (list): 改変設定のリスト
        device (torch.device, optional): 使用するデバイス
    
    Returns:
        torch.LongTensor: 改変が適用されたエッジインデックス
        dict: 改変情報の辞書
    """
    if device is None:
        device = edge_index.device
    
    edge_index = edge_index.to(device)
    modified_edge_index = edge_index.clone()
    modification_info = {
        'original_edges': edge_index.size(1),
        'modifications_applied': [],
        'total_modifications': len(modifications)
    }
    
    for i, mod_config in enumerate(modifications):
        mod_type = mod_config.get('type', 'unknown')
        percentage = mod_config.get('percentage', 0.0)
        
        print(f"  エッジ改変 {i+1}/{len(modifications)}: {mod_type} (割合: {percentage:.1%})")
        
        if mod_type == 'add':
            modified_edge_index, add_info = add_random_edges(modified_edge_index, num_nodes, percentage, device)
            
            modification_info['modifications_applied'].append({
                'type': 'add',
                'percentage': percentage,
                'info': add_info
            })
            
        elif mod_type == 'remove':
            modified_edge_index, remove_info = remove_random_edges(modified_edge_index, percentage, device)
            
            modification_info['modifications_applied'].append({
                'type': 'remove',
                'percentage': percentage,
                'info': remove_info
            })
            
        else:
            print(f"    警告: 未知のエッジ改変タイプ '{mod_type}' のため、スキップします")
            continue
    
    modification_info['final_edges'] = modified_edge_index.size(1)
    
    return modified_edge_index, modification_info


def print_edge_modification_info(modification_info, dataset_name="", run_num=None):
    """
    エッジ改変情報を表示する関数
    
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
    
    print(f"{prefix}エッジ改変適用:")
    print(f"  元のエッジ数: {modification_info['original_edges']}")
    print(f"  最終エッジ数: {modification_info['final_edges']}")
    print(f"  適用された改変数: {len(modification_info['modifications_applied'])}/{modification_info['total_modifications']}")
    
    for i, mod in enumerate(modification_info['modifications_applied']):
        mod_type = mod['type']
        percentage = mod['percentage']
        info = mod['info']
        
        if mod_type == 'add':
            print(f"  改変 {i+1}: エッジ追加 - 割合: {percentage:.1%}")
            print(f"    追加エッジ数: {info['num_added_edges']}")
            print(f"    元のエッジ数: {info['original_edges']}")
            print(f"    最終エッジ数: {info['final_edges']}")
                
        elif mod_type == 'remove':
            print(f"  改変 {i+1}: エッジ削除 - 割合: {percentage:.1%}")
            print(f"    削除エッジ数: {info['num_removed_edges']}")
            print(f"    元のエッジ数: {info['original_edges']}")
            print(f"    最終エッジ数: {info['final_edges']}")
