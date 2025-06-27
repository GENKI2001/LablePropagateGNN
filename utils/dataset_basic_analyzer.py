import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from .dataset_loader import load_dataset, get_supported_datasets
import warnings
import os
warnings.filterwarnings('ignore')

class DatasetBasicAnalyzer:
    """
    Class for analyzing basic dataset information and creating visualizations
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
        Analyze basic information for the specified dataset
        
        Args:
            dataset_name (str): Dataset name
            save_plots (bool): Whether to save plots
            output_dir (str): Output directory
        """
        print(f"\n=== {dataset_name} Dataset Basic Analysis ===")
        
        # Load dataset
        data, dataset = load_dataset(dataset_name, self.device)
        
        # Convert to undirected graph
        data = self._make_undirected(data)
        
        # Perform analysis
        analysis_result = self._analyze_basic_info(data, dataset)
        
        # Save results
        self.results[dataset_name] = analysis_result
        
        # Visualize
        self._visualize_basic_info(dataset_name, analysis_result, save_plots, output_dir)
        
        return analysis_result
    
    def _make_undirected(self, data):
        """
        Convert directed graph to undirected graph by removing duplicate edges
        
        Args:
            data: PyTorch Geometric data object
            
        Returns:
            data: Undirected graph data object
        """
        edge_index = data.edge_index.cpu().numpy()
        
        # Remove self-loops
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        
        # Remove duplicate edges for undirected graph
        edge_set = set()
        unique_edges = []
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            edge_tuple = tuple(sorted([src, dst]))
            if edge_tuple not in edge_set:
                edge_set.add(edge_tuple)
                unique_edges.append([src, dst])
        
        # Create new edge_index
        new_edge_index = torch.tensor(unique_edges, dtype=torch.long).t().contiguous()
        data.edge_index = new_edge_index.to(self.device)
        
        print(f"Converted to undirected graph: {len(unique_edges)} unique edges")
        
        return data
    
    def _analyze_basic_info(self, data, dataset):
        """
        Analyze basic dataset information
        
        Args:
            data: PyTorch Geometric data object
            dataset: Dataset object
            
        Returns:
            dict: Analysis results
        """
        # Get basic information
        num_nodes = data.num_nodes
        num_edges = data.edge_index.shape[1]
        num_classes = dataset.num_classes
        
        # Calculate average degree
        avg_degree = (2 * num_edges) / num_nodes
        
        # Get labels and calculate class distribution
        labels = data.y.cpu().numpy()
        label_counts = Counter(labels)
        
        # Calculate class ratios
        class_ratios = {}
        for label in range(num_classes):
            count = label_counts.get(label, 0)
            ratio = count / num_nodes
            class_ratios[label] = ratio
        
        # Calculate homophily
        edge_index = data.edge_index.cpu().numpy()
        same_label_edges = 0
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            if labels[src] == labels[dst]:
                same_label_edges += 1
        
        homophily = same_label_edges / num_edges if num_edges > 0 else 0
        
        # Calculate graph density
        density = (2 * num_edges) / (num_nodes * (num_nodes - 1))
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_classes': num_classes,
            'avg_degree': avg_degree,
            'density': density,
            'class_ratios': class_ratios,
            'label_counts': dict(label_counts),
            'homophily': homophily,
            'dataset_name': dataset.name if hasattr(dataset, 'name') else 'Unknown'
        }
    
    def _visualize_basic_info(self, dataset_name, analysis_result, save_plots, output_dir):
        """
        Visualize basic dataset information
        
        Args:
            dataset_name (str): Dataset name
            analysis_result (dict): Analysis result
            save_plots (bool): Whether to save plots
            output_dir (str): Output directory
        """
        # Create figure with 2 subplots (1x2 layout)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'{dataset_name} Dataset: Basic Information Analysis', fontsize=18, fontweight='bold')
        
        # 1. Class distribution pie chart (left)
        label_counts = analysis_result['label_counts']
        class_labels = [f'Class {label}' for label in label_counts.keys()]
        class_sizes = list(label_counts.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_sizes)))
        wedges, texts, autotexts = axes[0].pie(class_sizes, labels=class_labels, 
                                               autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0].set_title('Class Distribution', fontweight='bold', fontsize=14)
        
        # 2. Basic statistics table (right)
        stats_text = f"""DATASET STATISTICS
{'='*30}
Nodes: {analysis_result['num_nodes']:,}
Edges: {analysis_result['num_edges']:,}
Classes: {analysis_result['num_classes']}
Average Degree: {analysis_result['avg_degree']:.2f}
Graph Density: {analysis_result['density']:.6f}
Homophily: {analysis_result['homophily']:.4f}"""
        
        axes[1].text(0.05, 0.95, stats_text, transform=axes[1].transAxes, 
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', 
                             alpha=0.9, edgecolor='darkgreen', linewidth=2))
        axes[1].set_title('Basic Statistics', fontweight='bold', fontsize=14)
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_plots:
            # Create dataset_basic_images folder
            save_dir = os.path.join(output_dir, 'dataset_basic_images')
            os.makedirs(save_dir, exist_ok=True)
            
            plt.savefig(f'{save_dir}/{dataset_name}_basic_info.png', 
                       dpi=300, bbox_inches='tight')
            print(f"Basic info plot saved: {save_dir}/{dataset_name}_basic_info.png")
        
        plt.close()
        
        # Print statistics
        self._print_basic_statistics(dataset_name, analysis_result)
    
    def _print_basic_statistics(self, dataset_name, analysis_result):
        """
        Print basic statistics
        
        Args:
            dataset_name (str): Dataset name
            analysis_result (dict): Analysis results
        """
        print(f"\n=== {dataset_name} Basic Statistics ===")
        print(f"📊 Dataset Information:")
        print(f"  Number of nodes: {analysis_result['num_nodes']:,}")
        print(f"  Number of edges: {analysis_result['num_edges']:,}")
        print(f"  Number of classes: {analysis_result['num_classes']}")
        print(f"  Average degree: {analysis_result['avg_degree']:.2f}")
        print(f"  Graph density: {analysis_result['density']:.6f}")
        print(f"  Homophily: {analysis_result['homophily']:.4f}")
        
        print(f"\n🏷️ Class Distribution:")
        for label in sorted(analysis_result['label_counts'].keys()):
            count = analysis_result['label_counts'][label]
            ratio = analysis_result['class_ratios'][label]
            print(f"  Class {label}: {count:,} nodes ({ratio*100:.1f}%)")
    
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
        
        print(f"=== All Datasets Basic Analysis ===")
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
            dataset_comparison.append({
                'dataset': dataset_name,
                'num_nodes': result['num_nodes'],
                'num_edges': result['num_edges'],
                'num_classes': result['num_classes'],
                'avg_degree': result['avg_degree'],
                'density': result['density'],
                'homophily': result['homophily']
            })
        
        # Display results in table format
        print(f"{'Dataset':<12} {'Nodes':<8} {'Edges':<8} {'Classes':<6} {'Avg Deg':<8} {'Density':<8} {'Homophily':<8}")
        print("-" * 60)
        
        for comp in sorted(dataset_comparison, key=lambda x: x['num_nodes'], reverse=True):
            print(f"{comp['dataset']:<12} {comp['num_nodes']:<8,} {comp['num_edges']:<8,} {comp['num_classes']:<6} "
                  f"{comp['avg_degree']:<8.2f} {comp['density']:<8.6f} {comp['homophily']:.4f}")


def main():
    """
    Main function
    """
    # Initialize analyzer
    analyzer = DatasetBasicAnalyzer()
    
    # Analyze all datasets
    analyzer.analyze_all_datasets(save_plots=True, output_dir='./')


if __name__ == "__main__":
    main() 