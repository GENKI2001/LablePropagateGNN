#!/usr/bin/env python3
"""
実験実行のサンプルスクリプト
main.pyのmain関数を使用して複数の実験を並列実行する例を示します
"""

import multiprocessing as mp
import os
import time
from main import main

def run_single_experiment(exp_config, exp_index, total_experiments):
    """
    単一の実験を実行する関数（並列実行用）
    """
    print(f"\n{'='*80}")
    print(f"=== 実験 {exp_index+1}/{total_experiments} (PID: {os.getpid()}) ===")
    print(f"データセット: {exp_config['dataset_name']}")
    print(f"モデル: {exp_config['model_name']}")
    if exp_config.get('use_feature_modification') and exp_config.get('feature_modifications'):
        print(f"特徴量改変: {exp_config['feature_modifications']}")
    print(f"{'='*80}")
    
    try:
        # 実験実行
        main(**exp_config)
        print(f"実験 {exp_index+1} 完了 (PID: {os.getpid()})")
        return True
    except Exception as e:
        print(f"実験 {exp_index+1} でエラーが発生しました (PID: {os.getpid()}): {e}")
        return False

def run_multiple_experiments_parallel(max_workers=None):
    """
    複数の実験を並列実行する関数
    異なるデータセット、モデル、設定で実験を実行できます
    """

    # データセット選択
    # Planetoid: 'Cora', 'Citeseer', 'Pubmed'
    # WebKB: 'Cornell', 'Texas', 'Wisconsin'
    # WikipediaNetwork: 'Chameleon', 'Squirrel'
    # Actor: 'Actor'

    # モデル選択
    # MLP, GCN, GAT, GraphSAGE, H2GCN, RobustH2GCN, RGCN
    
    # 実験設定を動的に生成
    dataset_name = 'Squirrel'
    models = ['MLP', 'GCN', 'GAT', 'GraphSAGE', 'H2GCN', 'RGCN', 'RobustH2GCN']
    percentages = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    experiments = []
    for model_name in models:
        for percentage in percentages:
            experiments.append({
                'dataset_name': dataset_name,
                'model_name': model_name,
                'calc_neighbor_label_features': model_name == 'RobustH2GCN' if True else False,
                'use_feature_modification': False,
                'feature_modifications': None,
            })
            experiments.append({
                'dataset_name': dataset_name,
                'model_name': model_name,
                'calc_neighbor_label_features': model_name == 'RobustH2GCN' if True else False,
                'use_feature_modification': True,
                'feature_modifications': [
                    {'type': 'missingness', 'percentage': percentage, 'method': 'per_node'}
                ],
            })
    
    print("=== 並列実験実行開始 ===")
    print(f"実行予定実験数: {len(experiments)}")
    
    # 並列実行のワーカー数を決定
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(experiments))
    print(f"並列ワーカー数: {max_workers}")
    
    # 実験の開始時刻を記録
    start_time = time.time()
    
    # 並列実行
    with mp.Pool(processes=max_workers) as pool:
        # 各実験を並列実行
        results = []
        for i, exp_config in enumerate(experiments):
            result = pool.apply_async(
                run_single_experiment, 
                args=(exp_config, i, len(experiments))
            )
            results.append(result)
        
        # 全ての実験の完了を待機
        successful_experiments = 0
        for i, result in enumerate(results):
            try:
                success = result.get()  # 実験の完了を待機
                if success:
                    successful_experiments += 1
            except Exception as e:
                print(f"実験 {i+1} で予期しないエラーが発生しました: {e}")
    
    # 実行時間を計算
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\n{'='*80}")
    print(f"=== 全実験完了 ===")
    print(f"成功した実験数: {successful_experiments}/{len(experiments)}")
    print(f"総実行時間: {execution_time:.2f}秒")
    print(f"平均実行時間: {execution_time/len(experiments):.2f}秒/実験")
    print(f"{'='*80}")

if __name__ == "__main__":
    # 並列実行（推奨）
    run_multiple_experiments_parallel()