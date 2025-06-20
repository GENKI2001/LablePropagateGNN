import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict, Counter
from dataset_loader import load_dataset, get_supported_datasets
import warnings
import os
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

class LabelCorrelationAnalyzer:
    """
    隣接ノード間のラベル相関を分析・可視化するクラス
    """
    
    def __init__(self, device=None):
        """
        初期化
        
        Args:
            device: 使用するデバイス（Noneの場合は自動選択）
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def analyze_dataset(self, dataset_name, save_plots=True, output_dir='./'):
        """
        指定されたデータセットのラベル相関を分析
        
        Args:
            dataset_name (str): データセット名
            save_plots (bool): プロットを保存するかどうか
            output_dir (str): 出力ディレクトリ
        """
        print(f"\n=== {dataset_name} データセットのラベル相関分析 ===")
        
        # データセット読み込み
        data, dataset = load_dataset(dataset_name, self.device)
        
        # 分析実行
        analysis_result = self._analyze_label_correlations(data, dataset)
        
        # 結果を保存
        self.results[dataset_name] = analysis_result
        
        # 可視化
        self._visualize_correlations(dataset_name, analysis_result, save_plots, output_dir)
        
        return analysis_result
    
    def _analyze_label_correlations(self, data, dataset):
        """
        ラベル相関を分析
        
        Args:
            data: PyTorch Geometric データオブジェクト
            dataset: データセットオブジェクト
            
        Returns:
            dict: 分析結果
        """
        # エッジ情報を取得
        edge_index = data.edge_index.cpu().numpy()
        labels = data.y.cpu().numpy()
        
        # 隣接ノードのラベルペアを収集（無向グラフなので重複を除去）
        label_pairs = []
        unique_edges = set()
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            # 無向グラフの場合、同じエッジが2回格納されている可能性があるので重複を除去
            edge_tuple = tuple(sorted([src, dst]))
            if edge_tuple not in unique_edges:
                unique_edges.add(edge_tuple)
                src_label = labels[src]
                dst_label = labels[dst]
                # 順序を統一（小さいラベルを先に）
                if src_label <= dst_label:
                    label_pairs.append((src_label, dst_label))
                else:
                    label_pairs.append((dst_label, src_label))
        
        # ラベルペアの頻度を計算
        pair_counts = Counter(label_pairs)
        
        # 各ラベルの出現頻度を計算
        label_counts = Counter(labels)
        total_nodes = len(labels)
        
        # 相関行列を作成（実際の頻度のみ）
        correlation_matrix = np.zeros((dataset.num_classes, dataset.num_classes))
        
        for (label1, label2), actual_count in pair_counts.items():
            correlation_matrix[label1, label2] = actual_count
            correlation_matrix[label2, label1] = actual_count  # 対称行列
        
        # 統計情報を計算
        total_edges_analyzed = len(label_pairs)
        unique_pairs = len(pair_counts)
        
        # 最も頻繁なラベルペアを特定
        most_frequent_pairs = sorted(
            [(pair, count) for pair, count in pair_counts.items()],
            key=lambda x: x[1], reverse=True
        )
        
        # 最も稀なラベルペアを特定
        least_frequent_pairs = sorted(
            [(pair, count) for pair, count in pair_counts.items()],
            key=lambda x: x[1]
        )
        
        return {
            'correlation_matrix': correlation_matrix,
            'pair_counts': dict(pair_counts),
            'label_counts': dict(label_counts),
            'total_edges': total_edges_analyzed,
            'unique_pairs': unique_pairs,
            'most_frequent_pairs': most_frequent_pairs[:5],
            'least_frequent_pairs': least_frequent_pairs[:5],
            'dataset_info': {
                'num_nodes': data.num_nodes,
                'num_edges': len(unique_edges),  # 重複を除去した実際のエッジ数
                'num_classes': dataset.num_classes
            }
        }
    
    def _visualize_correlations(self, dataset_name, analysis_result, save_plots, output_dir):
        """
        相関結果を可視化
        
        Args:
            dataset_name (str): データセット名
            analysis_result (dict): 分析結果
            save_plots (bool): プロットを保存するかどうか
            output_dir (str): 出力ディレクトリ
        """
        # サブプロットを作成（2x2から3つのプロットに変更）
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(f'{dataset_name} Dataset: Label Correlation Analysis', fontsize=18, fontweight='bold')
        
        # 1. ラベル分布の棒グラフ（左）
        label_counts = analysis_result['label_counts']
        labels = list(label_counts.keys())
        counts = list(label_counts.values())
        
        bars = axes[0].bar(labels, counts, color='skyblue', alpha=0.7)
        axes[0].set_title('Label Distribution', fontweight='bold', fontsize=14)
        axes[0].set_xlabel('Label', fontsize=12)
        axes[0].set_ylabel('Number of Nodes', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # バーの上に値を表示
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{count:,}', ha='center', va='bottom', fontsize=11)
        
        # 2. 相関行列のヒートマップ（真ん中、大きく）
        correlation_matrix = analysis_result['correlation_matrix']
        num_classes = correlation_matrix.shape[0]
        
        # ヒートマップ用のデータフレームを作成
        df_corr = pd.DataFrame(
            correlation_matrix,
            index=[f'Label {i}' for i in range(num_classes)],
            columns=[f'Label {i}' for i in range(num_classes)]
        )
        
        sns.heatmap(df_corr, annot=True, fmt='.0f', cmap='Blues', ax=axes[1], 
                   square=True, cbar_kws={'shrink': 0.8}, annot_kws={'size': 12})
        axes[1].set_title('Adjacent Node Label Pair Frequency', fontweight='bold', fontsize=14)
        axes[1].set_xlabel('Label', fontsize=12)
        axes[1].set_ylabel('Label', fontsize=12)
        
        # 3. ラベル分布の詳細情報（右）
        dataset_info = analysis_result['dataset_info']
        
        # Homophilyを計算
        total_edges = analysis_result['total_edges']
        same_label_edges = 0
        for (label1, label2), count in analysis_result['pair_counts'].items():
            if label1 == label2:
                same_label_edges += count
        
        homophily = same_label_edges / total_edges if total_edges > 0 else 0
        
        # データセット情報をテキストで表示（デザイン改善）
        info_text = f"""DATASET INFORMATION
{'='*35}
Nodes: {dataset_info['num_nodes']:,}
Edges: {dataset_info['num_edges']:,}
Classes: {dataset_info['num_classes']}
Density: {(2 * dataset_info['num_edges']) / (dataset_info['num_nodes'] * (dataset_info['num_nodes'] - 1)):.6f}
Homophily: {homophily:.4f}

LABEL DISTRIBUTION
{'='*35}"""
        for label, count in sorted(label_counts.items()):
            percentage = (count / dataset_info['num_nodes']) * 100
            info_text += f"\nLabel {label:2d}: {count:8,} ({percentage:6.1f}%)"
        
        # テキストボックスのスタイルを改善（文字サイズを大きく）
        axes[2].text(0.05, 0.95, info_text, transform=axes[2].transAxes, 
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', 
                             alpha=0.9, edgecolor='navy', linewidth=2))
        axes[2].set_title('Dataset Details', fontweight='bold', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_plots:
            # label_correlation_imagesフォルダを作成
            save_dir = os.path.join(output_dir, 'label_correlation_images')
            os.makedirs(save_dir, exist_ok=True)
            
            plt.savefig(f'{save_dir}/{dataset_name}_label_correlation.png', 
                       dpi=300, bbox_inches='tight')
            print(f"プロットを保存しました: {save_dir}/{dataset_name}_label_correlation.png")
        
        # plt.show()を削除して画像を表示しない
        plt.close()  # メモリを節約するためにプロットを閉じる
        
        # 統計情報を表示
        self._print_statistics(dataset_name, analysis_result)
    
    def _print_statistics(self, dataset_name, analysis_result):
        """
        統計情報を表示
        
        Args:
            dataset_name (str): データセット名
            analysis_result (dict): 分析結果
        """
        print(f"\n=== {dataset_name} データセット情報 ===")
        
        dataset_info = analysis_result['dataset_info']
        print(f"📊 基本統計:")
        print(f"  ノード数: {dataset_info['num_nodes']:,}")
        print(f"  エッジ数: {dataset_info['num_edges']:,}")
        print(f"  クラス数: {dataset_info['num_classes']}")
        print(f"  分析したエッジ数: {analysis_result['total_edges']:,}")
        print(f"  ユニークなラベルペア数: {analysis_result['unique_pairs']}")
        
        # グラフの密度を計算
        density = (2 * dataset_info['num_edges']) / (dataset_info['num_nodes'] * (dataset_info['num_nodes'] - 1))
        print(f"  グラフ密度: {density:.6f}")
        
        # Homophilyを計算
        total_edges = analysis_result['total_edges']
        same_label_edges = 0
        for (label1, label2), count in analysis_result['pair_counts'].items():
            if label1 == label2:
                same_label_edges += count
        
        homophily = same_label_edges / total_edges if total_edges > 0 else 0
        print(f"  同質性 (Homophily): {homophily:.4f}")
        
        print(f"\n🏷️ ラベル分布:")
        total_nodes = dataset_info['num_nodes']
        for label, count in sorted(analysis_result['label_counts'].items()):
            percentage = (count / total_nodes) * 100
            print(f"  ラベル {label}: {count:,} ノード ({percentage:.1f}%)")
        
        # ラベル分布の統計
        label_counts_list = list(analysis_result['label_counts'].values())
        print(f"\n📈 ラベル分布統計:")
        print(f"  最大ラベルサイズ: {max(label_counts_list):,} ノード")
        print(f"  最小ラベルサイズ: {min(label_counts_list):,} ノード")
        print(f"  平均ラベルサイズ: {np.mean(label_counts_list):.1f} ノード")
        print(f"  ラベルサイズの標準偏差: {np.std(label_counts_list):.1f}")
        
        # クラス不均衡度を計算
        max_count = max(label_counts_list)
        min_count = min(label_counts_list)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"  クラス不均衡比: {imbalance_ratio:.2f}")
        
        print(f"\n🔗 隣接ノード間ラベル相関:")
        print(f"  最も頻繁なラベルペア:")
        # 実際の頻度でソート
        sorted_pairs = sorted(
            analysis_result['pair_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (pair, count) in enumerate(sorted_pairs[:5]):
            percentage = (count / analysis_result['total_edges']) * 100
            print(f"    {i+1}. ラベル {pair[0]} ↔ ラベル {pair[1]}: {count:,} エッジ ({percentage:.1f}%)")
        
        print(f"\n  最も稀なラベルペア:")
        for i, (pair, count) in enumerate(sorted_pairs[-5:]):
            percentage = (count / analysis_result['total_edges']) * 100
            print(f"    {i+1}. ラベル {pair[0]} ↔ ラベル {pair[1]}: {count:,} エッジ ({percentage:.1f}%)")
    
    def analyze_all_datasets(self, save_plots=True, output_dir='./'):
        """
        すべてのサポートされているデータセットを分析
        
        Args:
            save_plots (bool): プロットを保存するかどうか
            output_dir (str): 出力ディレクトリ
        """
        supported_datasets = get_supported_datasets()
        all_dataset_names = []
        
        for category, datasets in supported_datasets.items():
            all_dataset_names.extend(datasets)
        
        print(f"=== 全データセットのラベル相関分析 ===")
        print(f"分析対象: {len(all_dataset_names)} データセット")
        print(f"データセット: {', '.join(all_dataset_names)}")
        
        for dataset_name in all_dataset_names:
            try:
                self.analyze_dataset(dataset_name, save_plots, output_dir)
            except Exception as e:
                print(f"エラー: {dataset_name} の分析中にエラーが発生しました: {e}")
                continue
        
        # 全データセットの比較
        self._compare_all_datasets()
    
    def _compare_all_datasets(self):
        """
        全データセットの結果を比較
        """
        if len(self.results) < 2:
            print("比較するには少なくとも2つのデータセットの結果が必要です")
            return
        
        print(f"\n=== 全データセット比較 ===")
        
        # 各データセットの統計情報を収集
        dataset_comparison = []
        
        for dataset_name, result in self.results.items():
            dataset_info = result['dataset_info']
            label_counts_list = list(result['label_counts'].values())
            
            # グラフ密度を計算
            density = (2 * dataset_info['num_edges']) / (dataset_info['num_nodes'] * (dataset_info['num_nodes'] - 1))
            
            # クラス不均衡度を計算
            max_count = max(label_counts_list)
            min_count = min(label_counts_list)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            # ラベル分布の統計
            mean_label_size = np.mean(label_counts_list)
            std_label_size = np.std(label_counts_list)
            
            # Homophilyを計算
            total_edges = result['total_edges']
            same_label_edges = 0
            for (label1, label2), count in result['pair_counts'].items():
                if label1 == label2:
                    same_label_edges += count
            
            homophily = same_label_edges / total_edges if total_edges > 0 else 0
            
            dataset_comparison.append({
                'dataset': dataset_name,
                'num_nodes': dataset_info['num_nodes'],
                'num_edges': dataset_info['num_edges'],
                'num_classes': dataset_info['num_classes'],
                'density': density,
                'imbalance_ratio': imbalance_ratio,
                'mean_label_size': mean_label_size,
                'std_label_size': std_label_size,
                'homophily': homophily
            })
        
        # 結果を表示
        print(f"{'データセット':<12} {'ノード数':<8} {'エッジ数':<8} {'クラス数':<6} {'密度':<8} {'不均衡比':<8} {'平均ラベル':<10} {'同質性':<8}")
        print("-" * 70)
        
        for comp in sorted(dataset_comparison, key=lambda x: x['num_nodes'], reverse=True):
            print(f"{comp['dataset']:<12} {comp['num_nodes']:<8,} {comp['num_edges']:<8,} {comp['num_classes']:<6} "
                  f"{comp['density']:<8.6f} {comp['imbalance_ratio']:<8.2f} {comp['mean_label_size']:<10.1f} {comp['homophily']:.4f}")
        
        print(f"\n📊 データセット特性の詳細:")
        for comp in dataset_comparison:
            print(f"\n{comp['dataset']}:")
            print(f"  ノード数: {comp['num_nodes']:,}")
            print(f"  エッジ数: {comp['num_edges']:,}")
            print(f"  クラス数: {comp['num_classes']}")
            print(f"  グラフ密度: {comp['density']:.6f}")
            print(f"  クラス不均衡比: {comp['imbalance_ratio']:.2f}")
            print(f"  平均ラベルサイズ: {comp['mean_label_size']:.1f} ± {comp['std_label_size']:.1f}")
            print(f"  同質性: {comp['homophily']:.4f}")


def main():
    """
    メイン関数
    """
    # アナライザーを初期化
    analyzer = LabelCorrelationAnalyzer()
    
    # 分析対象のデータセットを指定
    # 個別のデータセットを分析する場合
    # analyzer.analyze_dataset('Cora')
    # analyzer.analyze_dataset('Chameleon')
    
    # 全データセットを分析する場合
    analyzer.analyze_all_datasets(save_plots=True, output_dir='./')


if __name__ == "__main__":
    main() 