#!/usr/bin/env python3
"""
隣接ノード間ラベル相関分析の実行スクリプト

使用方法:
    python run_label_analysis.py                    # 全データセットを分析
    python run_label_analysis.py --dataset Cora     # 特定のデータセットを分析
    python run_label_analysis.py --dataset Chameleon --no-save  # プロットを保存しない
"""

import argparse
from label_correlation_analyzer import LabelCorrelationAnalyzer

def main():
    parser = argparse.ArgumentParser(description='隣接ノード間ラベル相関分析')
    parser.add_argument('--dataset', type=str, default=None,
                       help='分析するデータセット名（指定しない場合は全データセット）')
    parser.add_argument('--no-save', action='store_true',
                       help='プロットを保存しない')
    parser.add_argument('--output-dir', type=str, default='./',
                       help='出力ディレクトリ（デフォルト: ./）')
    
    args = parser.parse_args()
    
    # アナライザーを初期化
    analyzer = LabelCorrelationAnalyzer()
    
    if args.dataset:
        # 特定のデータセットを分析
        print(f"=== {args.dataset} データセットの分析を開始 ===")
        try:
            analyzer.analyze_dataset(
                args.dataset, 
                save_plots=not args.no_save, 
                output_dir=args.output_dir
            )
        except Exception as e:
            print(f"エラー: {args.dataset} の分析中にエラーが発生しました: {e}")
    else:
        # 全データセットを分析
        print("=== 全データセットの分析を開始 ===")
        analyzer.analyze_all_datasets(
            save_plots=not args.no_save, 
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main() 