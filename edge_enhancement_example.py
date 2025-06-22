#!/usr/bin/env python3
"""
エッジ追加機能の使用例

このスクリプトは、特徴量の類似度とラベルの分布類似度に基づいてエッジを追加する機能の使用方法を示します。
"""

import torch
import numpy as np
from utils.dataset_loader import load_dataset
from utils.edge_enhancer import (
    EdgeEnhancer,
    enhance_edges_by_similarity,
    enhance_edges_by_label_similarity,
    enhance_edges_combined,
    analyze_similarity_distribution,
    analyze_label_similarity_distribution
)

def main():
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # データセット読み込み
    dataset_name = 'Cora'
    data, dataset = load_dataset(dataset_name, device)
    
    print(f"\n=== データセット情報 ===")
    print(f"データセット: {dataset_name}")
    print(f"ノード数: {data.num_nodes}")
    print(f"元のエッジ数: {data.edge_index.shape[1]}")
    print(f"特徴量次元: {data.x.shape[1]}")
    print(f"クラス数: {dataset.num_classes}")
    
    # データ分割を作成（例として）
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    data.train_mask[indices[:train_size]] = True
    data.val_mask[indices[train_size:train_size + val_size]] = True
    data.test_mask[indices[train_size + val_size:]] = True
    
    print(f"データ分割: 訓練={data.train_mask.sum().item()}, 検証={data.val_mask.sum().item()}, テスト={data.test_mask.sum().item()}")
    
    # 特徴量類似度分布分析
    print(f"\n=== 特徴量類似度分布分析 ===")
    similarity_stats = analyze_similarity_distribution(
        data, 
        similarity_method='cosine',
        normalize_features=True
    )
    
    # ラベル類似度分布分析
    print(f"\n=== ラベル類似度分布分析 ===")
    label_similarity_stats = analyze_label_similarity_distribution(
        data,
        label_similarity_method='cosine',
        use_train_val_only=True
    )
    
    # 特徴量類似度ベースのエッジ追加例
    print(f"\n=== 特徴量類似度ベースのエッジ追加例 ===")
    
    # 例1: コサイン類似度ベース
    print(f"\n--- 例1: コサイン類似度 (閾値=0.8) ---")
    enhanced_data1, edge_info1 = enhance_edges_by_similarity(
        data,
        similarity_method='cosine',
        threshold=0.8,
        max_edges_per_node=None,
        symmetric=True,
        normalize_features=True
    )
    
    # 例2: Jaccard類似度ベース
    print(f"\n--- 例2: Jaccard類似度 (閾値=0.5) ---")
    enhanced_data2, edge_info2 = enhance_edges_by_similarity(
        data,
        similarity_method='jaccard',
        threshold=0.5,
        max_edges_per_node=10,
        symmetric=True,
        normalize_features=True
    )
    
    # ラベル類似度ベースのエッジ追加例
    print(f"\n=== ラベル類似度ベースのエッジ追加例 ===")
    
    # 例3: ラベル類似度ベース（訓練・検証データのみ使用）
    print(f"\n--- 例3: ラベル類似度 (閾値=0.8, 訓練・検証データのみ) ---")
    enhanced_data3, label_edge_info = enhance_edges_by_label_similarity(
        data,
        threshold=0.8,
        max_edges_per_node=None,
        symmetric=True,
        label_similarity_method='cosine',
        use_train_val_only=True
    )
    
    # 結合エッジ追加例
    print(f"\n=== 結合エッジ追加例 ===")
    
    # 例4: 特徴量類似度 + ラベル類似度
    print(f"\n--- 例4: 特徴量類似度 + ラベル類似度 ---")
    enhanced_data4, combined_info = enhance_edges_combined(
        data,
        feature_similarity_method='cosine',
        feature_threshold=0.7,
        label_similarity_method='cosine',
        label_threshold=0.8,
        max_edges_per_node=5,
        symmetric=True,
        normalize_features=True,
        use_train_val_only=True
    )
    
    # 結果の比較
    print(f"\n=== 結果比較 ===")
    print(f"元のエッジ数: {data.edge_index.shape[1]}")
    print(f"特徴量類似度（コサイン）追加後: {enhanced_data1.edge_index.shape[1]} (+{edge_info1['num_added_edges']})")
    print(f"特徴量類似度（Jaccard）追加後: {enhanced_data2.edge_index.shape[1]} (+{edge_info2['num_added_edges']})")
    print(f"ラベル類似度追加後: {enhanced_data3.edge_index.shape[1]} (+{label_edge_info['num_added_edges']})")
    print(f"結合エッジ追加後: {enhanced_data4.edge_index.shape[1]} (+{combined_info['total_added_edges']})")
    
    # 異なる閾値での実験
    print(f"\n=== 閾値による影響実験 ===")
    
    # 特徴量類似度の閾値実験
    print(f"\n--- 特徴量類似度閾値実験 ---")
    feature_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in feature_thresholds:
        enhanced_data, edge_info = enhance_edges_by_similarity(
            data,
            similarity_method='cosine',
            threshold=threshold,
            max_edges_per_node=None,
            symmetric=True,
            normalize_features=True
        )
        print(f"特徴量類似度閾値 {threshold}: +{edge_info['num_added_edges']} エッジ")
    
    # ラベル類似度の閾値実験
    print(f"\n--- ラベル類似度閾値実験 ---")
    label_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in label_thresholds:
        enhanced_data, edge_info = enhance_edges_by_label_similarity(
            data,
            threshold=threshold,
            max_edges_per_node=None,
            symmetric=True,
            label_similarity_method='cosine',
            use_train_val_only=True
        )
        print(f"ラベル類似度閾値 {threshold}: +{edge_info['num_added_edges']} エッジ")
    
    print(f"\n=== 実験完了 ===")

if __name__ == "__main__":
    main() 