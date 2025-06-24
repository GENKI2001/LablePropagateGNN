import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict, Counter
from .dataset_loader import load_dataset, get_supported_datasets
import warnings
import os
warnings.filterwarnings('ignore')

class LabelCorrelationAnalyzer:
    """
    Class for analyzing and visualizing label correlations between adjacent nodes
    """
    
    def __init__(self, device=None):
        """
        Initialize the analyzer
        
        Args:
            device: Device to use (None for auto-selection)
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def analyze_dataset(self, dataset_name, save_plots=True, output_dir='./'):
        """
        Analyze label correlations for the specified dataset
        
        Args:
            dataset_name (str): Dataset name
            save_plots (bool): Whether to save plots
            output_dir (str): Output directory
        """
        print(f"\n=== {dataset_name} Dataset Label Correlation Analysis ===")
        
        # Load dataset
        data, dataset = load_dataset(dataset_name, self.device)
        
        # Perform analysis
        analysis_result = self._analyze_label_correlations(data, dataset)
        
        # Save results
        self.results[dataset_name] = analysis_result
        
        # Visualize
        self._visualize_correlations(dataset_name, analysis_result, save_plots, output_dir)
        
        return analysis_result
    
    def analyze_gsl_adjacency(self, model, data, dataset, threshold=0.1, save_plots=True, output_dir='./'):
        """
        Analyze and visualize the GSL-generated adjacency matrix
        
        Args:
            model: GSL model instance
            data: PyTorch Geometric data object
            dataset: Dataset object
            threshold (float): Threshold for converting probabilities to binary edges
            save_plots (bool): Whether to save plots
            output_dir (str): Output directory
        """
        print(f"\n=== GSL Adjacency Matrix Analysis ===")
        
        # Get the learned adjacency matrix from GSL model
        A_hat = model.get_learned_adjacency()
        A_hat_np = A_hat.detach().cpu().numpy()
        
        # Convert to binary adjacency matrix using threshold
        A_binary = (A_hat_np > threshold).astype(np.float32)
        
        # Remove self-loops
        np.fill_diagonal(A_binary, 0)
        
        # Create edge_index from binary adjacency matrix
        edge_indices = np.where(A_binary > 0)
        edge_index = np.vstack([edge_indices[0], edge_indices[1]])
        
        # Create a temporary data object with the GSL-generated edges
        temp_data = type('TempData', (), {})()
        temp_data.edge_index = torch.tensor(edge_index, dtype=torch.long)
        temp_data.y = data.y
        temp_data.num_nodes = data.num_nodes
        temp_data.train_mask = data.train_mask
        temp_data.val_mask = data.val_mask
        temp_data.test_mask = data.test_mask
        
        # Analyze the GSL-generated adjacency matrix
        analysis_result = self._analyze_label_correlations(temp_data, dataset)
        
        # Add GSL-specific information
        analysis_result['gsl_info'] = {
            'original_adjacency': A_hat_np,
            'binary_adjacency': A_binary,
            'threshold': threshold,
            'sparsity': 1.0 - (np.sum(A_binary) / (A_binary.shape[0] * A_binary.shape[1])),
            'max_probability': np.max(A_hat_np),
            'min_probability': np.min(A_hat_np),
            'mean_probability': np.mean(A_hat_np)
        }
        
        # Visualize the GSL adjacency matrix
        self._visualize_gsl_adjacency(dataset.name, analysis_result, save_plots, output_dir)
        
        return analysis_result
    
    def _analyze_label_correlations(self, data, dataset):
        """
        Analyze label correlations
        
        Args:
            data: PyTorch Geometric data object
            dataset: Dataset object
            
        Returns:
            dict: Analysis results
        """
        # Get edge information
        edge_index = data.edge_index.cpu().numpy()
        labels = data.y.cpu().numpy()
        
        # Collect label pairs of adjacent nodes (remove duplicates for undirected graphs)
        label_pairs = []
        unique_edges = set()
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            # For undirected graphs, same edge might be stored twice, so remove duplicates
            edge_tuple = tuple(sorted([src, dst]))
            if edge_tuple not in unique_edges:
                unique_edges.add(edge_tuple)
                src_label = labels[src]
                dst_label = labels[dst]
                # Standardize order (smaller label first)
                if src_label <= dst_label:
                    label_pairs.append((src_label, dst_label))
                else:
                    label_pairs.append((dst_label, src_label))
        
        # Calculate label pair frequencies
        pair_counts = Counter(label_pairs)
        
        # Calculate frequency of each label
        label_counts = Counter(labels)
        total_nodes = len(labels)
        
        # Create correlation matrix (actual frequencies only)
        correlation_matrix = np.zeros((dataset.num_classes, dataset.num_classes))
        
        for (label1, label2), actual_count in pair_counts.items():
            correlation_matrix[label1, label2] = actual_count
            correlation_matrix[label2, label1] = actual_count  # Symmetric matrix
        
        # Calculate statistics
        total_edges_analyzed = len(label_pairs)
        unique_pairs = len(pair_counts)
        
        # Identify most frequent label pairs
        most_frequent_pairs = sorted(
            [(pair, count) for pair, count in pair_counts.items()],
            key=lambda x: x[1], reverse=True
        )
        
        # Identify least frequent label pairs
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
                'num_edges': len(unique_edges),  # Actual edge count after removing duplicates
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
    
    def _visualize_gsl_adjacency(self, dataset_name, analysis_result, save_plots, output_dir):
        """
        Visualize the GSL adjacency matrix analysis results
        
        Args:
            dataset_name (str): Dataset name
            analysis_result (dict): Analysis result
            save_plots (bool): Whether to save plots
            output_dir (str): Output directory
        """
        # Create subplots (2x2 layout)
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'{dataset_name} Dataset: GSL Adjacency Matrix Analysis', fontsize=18, fontweight='bold')
        
        # 1. Original adjacency matrix heatmap (top left)
        A_hat = analysis_result['gsl_info']['original_adjacency']
        sns.heatmap(A_hat, cmap='viridis', ax=axes[0, 0], cbar_kws={'shrink': 0.8})
        axes[0, 0].set_title('GSL Learned Adjacency Matrix (Probabilities)', fontweight='bold', fontsize=14)
        axes[0, 0].set_xlabel('Node Index', fontsize=12)
        axes[0, 0].set_ylabel('Node Index', fontsize=12)
        
        # 2. Binary adjacency matrix heatmap (top right)
        A_binary = analysis_result['gsl_info']['binary_adjacency']
        sns.heatmap(A_binary, cmap='Blues', ax=axes[0, 1], cbar_kws={'shrink': 0.8})
        axes[0, 1].set_title(f'Binary Adjacency Matrix (Threshold={analysis_result["gsl_info"]["threshold"]})', 
                           fontweight='bold', fontsize=14)
        axes[0, 1].set_xlabel('Node Index', fontsize=12)
        axes[0, 1].set_ylabel('Node Index', fontsize=12)
        
        # 3. Label correlation matrix (bottom left)
        correlation_matrix = analysis_result['correlation_matrix']
        num_classes = correlation_matrix.shape[0]
        
        df_corr = pd.DataFrame(
            correlation_matrix,
            index=[f'Label {i}' for i in range(num_classes)],
            columns=[f'Label {i}' for i in range(num_classes)]
        )
        
        sns.heatmap(df_corr, annot=True, fmt='.0f', cmap='Blues', ax=axes[1, 0], 
                   square=True, cbar_kws={'shrink': 0.8}, annot_kws={'size': 10})
        axes[1, 0].set_title('GSL-Generated Edge Label Pair Frequency', fontweight='bold', fontsize=14)
        axes[1, 0].set_xlabel('Label', fontsize=12)
        axes[1, 0].set_ylabel('Label', fontsize=12)
        
        # 4. GSL statistics and information (bottom right)
        gsl_info = analysis_result['gsl_info']
        dataset_info = analysis_result['dataset_info']
        
        # Calculate homophily for GSL-generated edges
        total_edges = analysis_result['total_edges']
        same_label_edges = 0
        for (label1, label2), count in analysis_result['pair_counts'].items():
            if label1 == label2:
                same_label_edges += count
        
        homophily = same_label_edges / total_edges if total_edges > 0 else 0
        
        # Create information text
        info_text = f"""GSL ADJACENCY MATRIX INFO
{'='*40}
Original Edges: {dataset_info['num_edges']:,}
GSL Generated Edges: {analysis_result['total_edges']:,}
Sparsity: {gsl_info['sparsity']:.4f}
Threshold: {gsl_info['threshold']}

PROBABILITY STATISTICS
{'='*40}
Max Probability: {gsl_info['max_probability']:.4f}
Min Probability: {gsl_info['min_probability']:.4f}
Mean Probability: {gsl_info['mean_probability']:.4f}

GSL-GENERATED GRAPH PROPERTIES
{'='*40}
Nodes: {dataset_info['num_nodes']:,}
Classes: {dataset_info['num_classes']}
Density: {(2 * analysis_result['total_edges']) / (dataset_info['num_nodes'] * (dataset_info['num_nodes'] - 1)):.6f}
Homophily: {homophily:.4f}"""
        
        axes[1, 1].text(0.05, 0.95, info_text, transform=axes[1, 1].transAxes, 
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', 
                                alpha=0.9, edgecolor='navy', linewidth=2))
        axes[1, 1].set_title('GSL Analysis Details', fontweight='bold', fontsize=14)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_plots:
            # Create gsl_adjacency_images folder
            save_dir = os.path.join(output_dir, 'gsl_adjacency_images')
            os.makedirs(save_dir, exist_ok=True)
            
            plt.savefig(f'{save_dir}/{dataset_name}_gsl_adjacency.png', 
                       dpi=300, bbox_inches='tight')
            print(f"GSL adjacency plot saved: {save_dir}/{dataset_name}_gsl_adjacency.png")
        
        plt.close()
        
        # Print GSL-specific statistics
        self._print_gsl_statistics(dataset_name, analysis_result)
    
    def _print_statistics(self, dataset_name, analysis_result):
        """
        Print statistics
        
        Args:
            dataset_name (str): Dataset name
            analysis_result (dict): Analysis results
        """
        print(f"\n=== {dataset_name} Dataset Information ===")
        
        dataset_info = analysis_result['dataset_info']
        print(f"📊 Basic Statistics:")
        print(f"  Number of nodes: {dataset_info['num_nodes']:,}")
        print(f"  Number of edges: {dataset_info['num_edges']:,}")
        print(f"  Number of classes: {dataset_info['num_classes']}")
        print(f"  Analyzed edges: {analysis_result['total_edges']:,}")
        print(f"  Unique label pairs: {analysis_result['unique_pairs']}")
        
        # Calculate graph density
        density = (2 * dataset_info['num_edges']) / (dataset_info['num_nodes'] * (dataset_info['num_nodes'] - 1))
        print(f"  Graph density: {density:.6f}")
        
        # Calculate Homophily
        total_edges = analysis_result['total_edges']
        same_label_edges = 0
        for (label1, label2), count in analysis_result['pair_counts'].items():
            if label1 == label2:
                same_label_edges += count
        
        homophily = same_label_edges / total_edges if total_edges > 0 else 0
        print(f"  Homophily: {homophily:.4f}")
        
        print(f"\n🏷️ Label Distribution:")
        total_nodes = dataset_info['num_nodes']
        for label, count in sorted(analysis_result['label_counts'].items()):
            percentage = (count / total_nodes) * 100
            print(f"  Label {label}: {count:,} nodes ({percentage:.1f}%)")
        
        # Label distribution statistics
        label_counts_list = list(analysis_result['label_counts'].values())
        print(f"\n📈 Label Distribution Statistics:")
        print(f"  Maximum label size: {max(label_counts_list):,} nodes")
        print(f"  Minimum label size: {min(label_counts_list):,} nodes")
        print(f"  Average label size: {np.mean(label_counts_list):.1f} nodes")
        print(f"  Label size standard deviation: {np.std(label_counts_list):.1f}")
        
        # Calculate class imbalance ratio
        max_count = max(label_counts_list)
        min_count = min(label_counts_list)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"  Class imbalance ratio: {imbalance_ratio:.2f}")
        
        print(f"\n🔗 Adjacent Node Label Correlations:")
        print(f"  Most frequent label pairs:")
        # Sort by actual frequency
        sorted_pairs = sorted(
            analysis_result['pair_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (pair, count) in enumerate(sorted_pairs[:5]):
            percentage = (count / analysis_result['total_edges']) * 100
            print(f"    {i+1}. Label {pair[0]} ↔ Label {pair[1]}: {count:,} edges ({percentage:.1f}%)")
        
        print(f"\n  Least frequent label pairs:")
        for i, (pair, count) in enumerate(sorted_pairs[-5:]):
            percentage = (count / analysis_result['total_edges']) * 100
            print(f"    {i+1}. Label {pair[0]} ↔ Label {pair[1]}: {count:,} edges ({percentage:.1f}%)")
    
    def _print_gsl_statistics(self, dataset_name, analysis_result):
        """
        Print GSL-specific statistics
        
        Args:
            dataset_name (str): Dataset name
            analysis_result (dict): Analysis result
        """
        print(f"\n=== {dataset_name} GSL Adjacency Matrix Statistics ===")
        
        gsl_info = analysis_result['gsl_info']
        dataset_info = analysis_result['dataset_info']
        
        print(f"📊 GSL Matrix Properties:")
        print(f"  Original edges: {dataset_info['num_edges']:,}")
        print(f"  GSL generated edges: {analysis_result['total_edges']:,}")
        print(f"  Sparsity: {gsl_info['sparsity']:.4f}")
        print(f"  Threshold: {gsl_info['threshold']}")
        
        print(f"\n📈 Probability Statistics:")
        print(f"  Max probability: {gsl_info['max_probability']:.4f}")
        print(f"  Min probability: {gsl_info['min_probability']:.4f}")
        print(f"  Mean probability: {gsl_info['mean_probability']:.4f}")
        
        # Calculate homophily
        total_edges = analysis_result['total_edges']
        same_label_edges = 0
        for (label1, label2), count in analysis_result['pair_counts'].items():
            if label1 == label2:
                same_label_edges += count
        
        homophily = same_label_edges / total_edges if total_edges > 0 else 0
        print(f"  Homophily: {homophily:.4f}")
        
        print(f"\n🔗 GSL-Generated Edge Label Pairs:")
        sorted_pairs = sorted(
            analysis_result['pair_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        print(f"  Most frequent pairs:")
        for i, (pair, count) in enumerate(sorted_pairs[:5]):
            percentage = (count / analysis_result['total_edges']) * 100
            print(f"    {i+1}. Label {pair[0]} ↔ Label {pair[1]}: {count:,} edges ({percentage:.1f}%)")
        
        print(f"\n  Least frequent pairs:")
        for i, (pair, count) in enumerate(sorted_pairs[-5:]):
            percentage = (count / analysis_result['total_edges']) * 100
            print(f"    {i+1}. Label {pair[0]} ↔ Label {pair[1]}: {count:,} edges ({percentage:.1f}%)")

    def analyze_all_datasets(self, save_plots=True, output_dir='./'):
        """
        Analyze all supported datasets
        
        Args:
            save_plots (bool): Whether to save plots
            output_dir (str): Output directory
        """
        supported_datasets = get_supported_datasets()
        all_dataset_names = []
        
        for category, datasets in supported_datasets.items():
            all_dataset_names.extend(datasets)
        
        print(f"=== All Datasets Label Correlation Analysis ===")
        print(f"Target: {len(all_dataset_names)} datasets")
        print(f"Datasets: {', '.join(all_dataset_names)}")
        
        for dataset_name in all_dataset_names:
            try:
                self.analyze_dataset(dataset_name, save_plots, output_dir)
            except Exception as e:
                print(f"Error: Error occurred while analyzing {dataset_name}: {e}")
                continue
        
        # Compare all datasets
        self._compare_all_datasets()
    
    def _compare_all_datasets(self):
        """
        Compare results from all datasets
        """
        if len(self.results) < 2:
            print("At least 2 dataset results are needed for comparison")
            return
        
        print(f"\n=== All Datasets Comparison ===")
        
        # Collect statistics for each dataset
        dataset_comparison = []
        
        for dataset_name, result in self.results.items():
            dataset_info = result['dataset_info']
            label_counts_list = list(result['label_counts'].values())
            
            # Calculate graph density
            density = (2 * dataset_info['num_edges']) / (dataset_info['num_nodes'] * (dataset_info['num_nodes'] - 1))
            
            # Calculate class imbalance ratio
            max_count = max(label_counts_list)
            min_count = min(label_counts_list)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            # Label distribution statistics
            mean_label_size = np.mean(label_counts_list)
            std_label_size = np.std(label_counts_list)
            
            # Calculate Homophily
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
        
        # Display results
        print(f"{'Dataset':<12} {'Nodes':<8} {'Edges':<8} {'Classes':<6} {'Density':<8} {'Imbalance':<8} {'Avg Label':<10} {'Homophily':<8}")
        print("-" * 70)
        
        for comp in sorted(dataset_comparison, key=lambda x: x['num_nodes'], reverse=True):
            print(f"{comp['dataset']:<12} {comp['num_nodes']:<8,} {comp['num_edges']:<8,} {comp['num_classes']:<6} "
                  f"{comp['density']:<8.6f} {comp['imbalance_ratio']:<8.2f} {comp['mean_label_size']:<10.1f} {comp['homophily']:.4f}")
        
        print(f"\n📊 Detailed Dataset Characteristics:")
        for comp in dataset_comparison:
            print(f"\n{comp['dataset']}:")
            print(f"  Number of nodes: {comp['num_nodes']:,}")
            print(f"  Number of edges: {comp['num_edges']:,}")
            print(f"  Number of classes: {comp['num_classes']}")
            print(f"  Graph density: {comp['density']:.6f}")
            print(f"  Class imbalance ratio: {comp['imbalance_ratio']:.2f}")
            print(f"  Average label size: {comp['mean_label_size']:.1f} ± {comp['std_label_size']:.1f}")
            print(f"  Homophily: {comp['homophily']:.4f}")


def main():
    """
    Main function
    """
    # Initialize analyzer
    analyzer = LabelCorrelationAnalyzer()
    
    # Specify target datasets for analysis
    # For individual dataset analysis
    # analyzer.analyze_dataset('Cora')
    # analyzer.analyze_dataset('Chameleon')
    
    # For all datasets analysis
    analyzer.analyze_all_datasets(save_plots=True, output_dir='./')


if __name__ == "__main__":
    main() 