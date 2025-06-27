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
        
        # Analyze class distribution vectors
        class_dist_analysis = self._analyze_class_distribution_vectors(data, dataset)
        analysis_result['class_distribution_vectors'] = class_dist_analysis
        
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
    
    def _analyze_class_distribution_vectors(self, data, dataset):
        """
        Analyze class distribution vectors for each node based on 1-hop, 2-hop, 3-hop, and 4-hop neighbors
        
        Args:
            data: PyTorch Geometric data object
            dataset: Dataset object
            
        Returns:
            dict: Class distribution analysis results
        """
        print(f"\n=== Class Distribution Vectors Analysis ===")
        
        # Get basic information
        num_nodes = data.num_nodes
        num_classes = dataset.num_classes
        labels = data.y.cpu().numpy()
        edge_index = data.edge_index.cpu().numpy()
        
        # Create one-hot encoding for all nodes
        one_hot_labels = np.zeros((num_nodes, num_classes))
        for i in range(num_nodes):
            one_hot_labels[i, labels[i]] = 1.0
        
        # Calculate class distribution vectors for different hop distances
        class_dist_vectors_1hop = self._calculate_hop_distribution_vectors(edge_index, one_hot_labels, num_nodes, num_classes, hop=1)
        class_dist_vectors_2hop = self._calculate_hop_distribution_vectors(edge_index, one_hot_labels, num_nodes, num_classes, hop=2)
        class_dist_vectors_3hop = self._calculate_hop_distribution_vectors(edge_index, one_hot_labels, num_nodes, num_classes, hop=3)
        class_dist_vectors_4hop = self._calculate_hop_distribution_vectors(edge_index, one_hot_labels, num_nodes, num_classes, hop=4)
        
        # Calculate average class distribution vector for each class for all hop distances
        class_avg_distributions_1hop = self._calculate_class_averages(class_dist_vectors_1hop, labels, num_classes)
        class_avg_distributions_2hop = self._calculate_class_averages(class_dist_vectors_2hop, labels, num_classes)
        class_avg_distributions_3hop = self._calculate_class_averages(class_dist_vectors_3hop, labels, num_classes)
        class_avg_distributions_4hop = self._calculate_class_averages(class_dist_vectors_4hop, labels, num_classes)
        
        # Analyze 3-hop frequent patterns
        frequent_3hop_patterns = self._analyze_3hop_frequent_patterns(edge_index, labels, num_nodes, num_classes)
        
        # Print results
        self._print_distribution_results(class_avg_distributions_1hop, class_avg_distributions_2hop, 
                                       class_avg_distributions_3hop, class_avg_distributions_4hop, 
                                       num_nodes, num_classes, frequent_3hop_patterns)
        
        return {
            'class_dist_vectors_1hop': class_dist_vectors_1hop,
            'class_dist_vectors_2hop': class_dist_vectors_2hop,
            'class_dist_vectors_3hop': class_dist_vectors_3hop,
            'class_dist_vectors_4hop': class_dist_vectors_4hop,
            'class_avg_distributions_1hop': class_avg_distributions_1hop,
            'class_avg_distributions_2hop': class_avg_distributions_2hop,
            'class_avg_distributions_3hop': class_avg_distributions_3hop,
            'class_avg_distributions_4hop': class_avg_distributions_4hop,
            'frequent_3hop_patterns': frequent_3hop_patterns,
            'one_hot_labels': one_hot_labels
        }
    
    def _calculate_hop_distribution_vectors(self, edge_index, one_hot_labels, num_nodes, num_classes, hop=1):
        """
        Calculate class distribution vectors for specified hop distance
        
        Args:
            edge_index: Edge index array
            one_hot_labels: One-hot encoded labels
            num_nodes: Number of nodes
            num_classes: Number of classes
            hop: Hop distance (1, 2, 3, or 4)
            
        Returns:
            numpy.ndarray: Class distribution vectors
        """
        class_dist_vectors = np.zeros((num_nodes, num_classes))
        neighbor_counts = np.zeros(num_nodes)
        
        if hop == 1:
            # 1-hop neighbors
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[:, i]
                class_dist_vectors[src] += one_hot_labels[dst]
                class_dist_vectors[dst] += one_hot_labels[src]
                neighbor_counts[src] += 1
                neighbor_counts[dst] += 1
                
        elif hop >= 2:
            # Create adjacency list for efficient neighbor lookup
            adj_list = [[] for _ in range(num_nodes)]
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[:, i]
                adj_list[src].append(dst)
                adj_list[dst].append(src)
            
            # Calculate k-hop neighbors
            for node in range(num_nodes):
                # Get all neighbors up to k-hop distance
                all_neighbors = set()
                current_neighbors = {node}
                
                for h in range(hop):
                    next_neighbors = set()
                    for neighbor in current_neighbors:
                        next_neighbors.update(adj_list[neighbor])
                    all_neighbors.update(next_neighbors)
                    current_neighbors = next_neighbors
                
                # Remove self and all neighbors with distance < hop
                all_neighbors.discard(node)
                
                # For hop > 1, we need to exclude neighbors from previous hops
                if hop > 1:
                    # Get neighbors from previous hops
                    prev_neighbors = set()
                    current_neighbors = {node}
                    
                    for h in range(hop - 1):
                        next_neighbors = set()
                        for neighbor in current_neighbors:
                            next_neighbors.update(adj_list[neighbor])
                        prev_neighbors.update(next_neighbors)
                        current_neighbors = next_neighbors
                    
                    # Only keep neighbors at exactly hop distance
                    target_neighbors = all_neighbors - prev_neighbors
                else:
                    target_neighbors = all_neighbors
                
                # Add one-hot vectors of target neighbors
                for target_neighbor in target_neighbors:
                    class_dist_vectors[node] += one_hot_labels[target_neighbor]
                    neighbor_counts[node] += 1
        
        # Normalize by number of neighbors (get average)
        for i in range(num_nodes):
            if neighbor_counts[i] > 0:
                class_dist_vectors[i] /= neighbor_counts[i]
        
        return class_dist_vectors
    
    def _calculate_class_averages(self, class_dist_vectors, labels, num_classes):
        """
        Calculate average class distribution vector for each class
        
        Args:
            class_dist_vectors: Class distribution vectors for all nodes
            labels: Node labels
            num_classes: Number of classes
            
        Returns:
            dict: Average distributions for each class
        """
        class_avg_distributions = {}
        class_node_indices = {}
        
        for class_label in range(num_classes):
            # Find nodes belonging to this class
            class_nodes = np.where(labels == class_label)[0]
            class_node_indices[class_label] = class_nodes
            
            if len(class_nodes) > 0:
                # Calculate average distribution vector for this class
                avg_dist = np.mean(class_dist_vectors[class_nodes], axis=0)
                class_avg_distributions[class_label] = avg_dist
            else:
                class_avg_distributions[class_label] = np.zeros(num_classes)
        
        return class_avg_distributions
    
    def _print_distribution_results(self, class_avg_distributions_1hop, class_avg_distributions_2hop, 
                                   class_avg_distributions_3hop, class_avg_distributions_4hop, 
                                   num_nodes, num_classes, frequent_3hop_patterns):
        """
        Print distribution results for all four hop distances
        
        Args:
            class_avg_distributions_1hop: 1-hop average distributions
            class_avg_distributions_2hop: 2-hop average distributions
            class_avg_distributions_3hop: 3-hop average distributions
            class_avg_distributions_4hop: 4-hop average distributions
            num_nodes: Number of nodes
            num_classes: Number of classes
            frequent_3hop_patterns: Frequent 3-hop patterns analysis results
        """
        print(f"📊 Class Distribution Vectors Analysis:")
        print(f"  Number of nodes: {num_nodes}")
        print(f"  Number of classes: {num_classes}")
        
        print(f"\n🏷️ Average Class Distribution Vectors (1-hop):")
        print(f"{'Class':<6} {'Distribution Vector':<50}")
        print("-" * 60)
        
        for class_label in sorted(class_avg_distributions_1hop.keys()):
            avg_dist = class_avg_distributions_1hop[class_label]
            dist_str = "[" + ", ".join([f"{val:.3f}" for val in avg_dist]) + "]"
            print(f"{class_label:<6} {dist_str}")
        
        print(f"\n🏷️ Average Class Distribution Vectors (2-hop):")
        print(f"{'Class':<6} {'Distribution Vector':<50}")
        print("-" * 60)
        
        for class_label in sorted(class_avg_distributions_2hop.keys()):
            avg_dist = class_avg_distributions_2hop[class_label]
            dist_str = "[" + ", ".join([f"{val:.3f}" for val in avg_dist]) + "]"
            print(f"{class_label:<6} {dist_str}")
        
        print(f"\n🏷️ Average Class Distribution Vectors (3-hop):")
        print(f"{'Class':<6} {'Distribution Vector':<50}")
        print("-" * 60)
        
        for class_label in sorted(class_avg_distributions_3hop.keys()):
            avg_dist = class_avg_distributions_3hop[class_label]
            dist_str = "[" + ", ".join([f"{val:.3f}" for val in avg_dist]) + "]"
            print(f"{class_label:<6} {dist_str}")
        
        print(f"\n🏷️ Average Class Distribution Vectors (4-hop):")
        print(f"{'Class':<6} {'Distribution Vector':<50}")
        print("-" * 60)
        
        for class_label in sorted(class_avg_distributions_4hop.keys()):
            avg_dist = class_avg_distributions_4hop[class_label]
            dist_str = "[" + ", ".join([f"{val:.3f}" for val in avg_dist]) + "]"
            print(f"{class_label:<6} {dist_str}")
        
        # Print 3-hop frequent patterns
        self._print_3hop_frequent_patterns(frequent_3hop_patterns)
    
    def _print_3hop_frequent_patterns(self, frequent_3hop_patterns):
        """
        Print 3-hop frequent patterns analysis results
        
        Args:
            frequent_3hop_patterns: Frequent 3-hop patterns analysis results
        """
        if not frequent_3hop_patterns:
            return
        
        print(f"\n🎯 3-Hop Frequent Pattern Analysis Summary:")
        print(f"  Sampled nodes: {frequent_3hop_patterns['sampled_nodes_count']:,} / {frequent_3hop_patterns['total_nodes']:,}")
        print(f"  Sampled 3-hop paths: {frequent_3hop_patterns['actual_sampled_paths']:,}")
        print(f"  Estimated total 3-hop paths: {frequent_3hop_patterns['total_paths']:,}")
        print(f"  Unique patterns: {frequent_3hop_patterns['unique_patterns']:,}")
        print(f"  Pattern diversity: {frequent_3hop_patterns['pattern_diversity']:.6f}")
        print(f"  Top 10 coverage: {frequent_3hop_patterns['top_10_coverage']:.2f}%")
        
        if frequent_3hop_patterns['scale_factor'] > 1.0:
            print(f"  Scale factor: {frequent_3hop_patterns['scale_factor']:.2f}x")
        
        print(f"\n🏆 Top 10 Most Frequent 3-Hop Patterns:")
        print(f"{'Rank':<4} {'Pattern':<20} {'Count':<8} {'Percentage':<10}")
        print("-" * 45)
        
        for rank, (pattern, count) in enumerate(frequent_3hop_patterns['top_10_patterns'], 1):
            # Use actual sampled count for percentage calculation
            actual_count = frequent_3hop_patterns['original_pattern_counts'].get(pattern, 0)
            percentage = (actual_count / frequent_3hop_patterns['actual_sampled_paths']) * 100
            print(f"{rank:<4} {pattern:<20} {count:<8,} {percentage:<10.2f}%")
    
    def _analyze_3hop_frequent_patterns(self, edge_index, labels, num_nodes, num_classes):
        """
        Analyze frequent 3-hop patterns (class1-class2-class3)
        
        Args:
            edge_index: Edge index array
            labels: Node labels
            num_nodes: Number of nodes
            num_classes: Number of classes
            
        Returns:
            dict: Frequent pattern analysis results
        """
        print(f"\n=== 3-Hop Frequent Pattern Analysis ===")
        
        # Create adjacency list
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            adj_list[src].append(dst)
            adj_list[dst].append(src)
        
        # Find all 3-hop paths and their class patterns using sampling
        pattern_counts = {}
        total_paths = 0
        actual_sampled_paths = 0
        
        # Determine sampling size based on dataset size
        if num_nodes > 10000:
            # For large datasets, sample 1000 nodes
            sample_size = min(4000, num_nodes // 3)
            sampled_nodes = np.random.choice(num_nodes, sample_size, replace=False)
            print(f"  Large dataset detected ({num_nodes:,} nodes). Sampling {sample_size:,} nodes for analysis.")
        elif num_nodes > 5000:
            # For medium datasets, sample 500 nodes
            sample_size = min(2000, num_nodes // 4)
            sampled_nodes = np.random.choice(num_nodes, sample_size, replace=False)
            print(f"  Medium dataset detected ({num_nodes:,} nodes). Sampling {sample_size:,} nodes for analysis.")
        else:
            # For small datasets, use all nodes
            sampled_nodes = np.arange(num_nodes)
            print(f"  Small dataset detected ({num_nodes:,} nodes). Using all nodes for analysis.")
        
        print(f"  Analyzing 3-hop patterns from {len(sampled_nodes):,} nodes...")
        
        for i, start_node in enumerate(sampled_nodes):
            if i % 100 == 0 and i > 0:
                print(f"    Progress: {i}/{len(sampled_nodes)} nodes processed")
            
            # Use BFS to find all 3-hop paths from start_node
            paths = self._find_3hop_paths(adj_list, start_node, labels)
            
            for path in paths:
                # Create class pattern (e.g., "0-1-2")
                class_pattern = "-".join([str(labels[node]) for node in path])
                pattern_counts[class_pattern] = pattern_counts.get(class_pattern, 0) + 1
                total_paths += 1
                actual_sampled_paths += 1
        
        # Store original counts before scaling
        original_pattern_counts = pattern_counts.copy()
        original_total_paths = actual_sampled_paths
        
        if len(sampled_nodes) < num_nodes:
            # Scale up the counts to estimate total patterns
            scale_factor = num_nodes / len(sampled_nodes)
            pattern_counts = {pattern: int(count * scale_factor) for pattern, count in pattern_counts.items()}
            total_paths = int(actual_sampled_paths * scale_factor)
            print(f"  Scaled up counts by factor {scale_factor:.2f} to estimate full dataset patterns.")
        
        # Sort patterns by frequency
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Get top 10 patterns
        top_10_patterns = sorted_patterns[:10]
        
        print(f"📊 3-Hop Pattern Analysis:")
        print(f"  Sampled 3-hop paths found: {actual_sampled_paths:,}")
        print(f"  Estimated total 3-hop paths: {total_paths:,}")
        print(f"  Unique patterns found: {len(pattern_counts)}")
        
        print(f"\n🏆 Top 10 Most Frequent 3-Hop Patterns:")
        print(f"{'Rank':<4} {'Pattern':<20} {'Count':<8} {'Percentage':<10}")
        print("-" * 45)
        
        stats_text = ""
        for i, (pattern, count) in enumerate(top_10_patterns, 1):
            # Use actual sampled count for percentage calculation
            actual_count = original_pattern_counts.get(pattern, 0)
            percentage = (actual_count / actual_sampled_paths) * 100
            stats_text += f"\n{i:2d}. {pattern:<15} {count:>8,} ({percentage:>5.1f}%)"
        
        # Calculate pattern diversity statistics
        pattern_diversity = len(pattern_counts) / total_paths if total_paths > 0 else 0
        most_common_pattern = top_10_patterns[0] if top_10_patterns else ("", 0)
        
        # Calculate coverage based on actual sampled paths
        top_10_actual_coverage = (sum(original_pattern_counts.get(pattern, 0) for pattern, _ in top_10_patterns) / actual_sampled_paths) * 100 if actual_sampled_paths > 0 else 0
        
        print(f"\n📈 Pattern Diversity Statistics:")
        print(f"  Pattern diversity ratio: {pattern_diversity:.6f}")
        print(f"  Most common pattern: {most_common_pattern[0]} ({most_common_pattern[1]:,} occurrences)")
        print(f"  Top 10 patterns cover: {top_10_actual_coverage:.2f}% of sampled paths")
        
        return {
            'pattern_counts': pattern_counts,
            'original_pattern_counts': original_pattern_counts,
            'top_10_patterns': top_10_patterns,
            'total_paths': total_paths,
            'actual_sampled_paths': actual_sampled_paths,
            'unique_patterns': len(pattern_counts),
            'pattern_diversity': pattern_diversity,
            'top_10_coverage': top_10_actual_coverage,
            'sampled_nodes_count': len(sampled_nodes),
            'total_nodes': num_nodes,
            'scale_factor': num_nodes / len(sampled_nodes) if len(sampled_nodes) < num_nodes else 1.0,
            'stats_text': stats_text
        }
    
    def _find_3hop_paths(self, adj_list, start_node, labels):
        """
        Find all 3-hop paths from start_node using BFS with limits
        
        Args:
            adj_list: Adjacency list
            start_node: Starting node
            labels: Node labels
            
        Returns:
            list: List of 3-hop paths
        """
        paths = []
        max_paths_per_node = 1000  # Limit to prevent explosion
        
        # Use BFS to find all paths of length 3
        queue = [(start_node, [start_node])]  # (current_node, path_so_far)
        
        while queue and len(paths) < max_paths_per_node:
            current_node, path = queue.pop(0)
            
            # If we have a 3-hop path, add it to results
            if len(path) == 4:  # 4 nodes = 3 hops
                paths.append(path)
                continue
            
            # If path is shorter than 3 hops, continue exploring
            if len(path) < 4:
                # Limit the number of neighbors to explore to prevent explosion
                neighbors = adj_list[current_node]
                if len(neighbors) > 50:  # If node has too many neighbors, sample them
                    neighbors = np.random.choice(neighbors, min(50, len(neighbors)), replace=False)
                
                for neighbor in neighbors:
                    # Avoid cycles (don't revisit nodes in the current path)
                    if neighbor not in path:
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path))
        
        return paths
    
    def analyze_class_distribution_vectors_only(self, dataset_name):
        """
        Analyze only class distribution vectors for the specified dataset
        
        Args:
            dataset_name (str): Dataset name
            
        Returns:
            dict: Class distribution analysis results
        """
        print(f"\n=== {dataset_name} Class Distribution Vectors Analysis ===")
        
        # Load dataset
        data, dataset = load_dataset(dataset_name, self.device)
        
        # Convert to undirected graph
        data = self._make_undirected(data)
        
        # Analyze class distribution vectors
        class_dist_analysis = self._analyze_class_distribution_vectors(data, dataset)
        
        return class_dist_analysis
    
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
        
        # Visualize class distribution vectors if available
        if 'class_distribution_vectors' in analysis_result:
            self._visualize_class_distribution_vectors(dataset_name, analysis_result, save_plots, output_dir)
        
        # Visualize 3-hop frequent patterns if available
        if 'class_distribution_vectors' in analysis_result and 'frequent_3hop_patterns' in analysis_result['class_distribution_vectors']:
            self._visualize_3hop_frequent_patterns(dataset_name, analysis_result, save_plots, output_dir)
        
        # Print statistics
        self._print_basic_statistics(dataset_name, analysis_result)
    
    def _visualize_class_distribution_vectors(self, dataset_name, analysis_result, save_plots, output_dir):
        """
        Visualize class distribution vectors
        
        Args:
            dataset_name (str): Dataset name
            analysis_result (dict): Analysis result
            save_plots (bool): Whether to save plots
            output_dir (str): Output directory
        """
        class_dist_data = analysis_result['class_distribution_vectors']
        frequent_3hop_patterns = class_dist_data['frequent_3hop_patterns']
        
        if not frequent_3hop_patterns or not frequent_3hop_patterns['top_10_patterns']:
            return
        
        # Create figure with 2 subplots (1x2 layout)
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f'{dataset_name} Dataset: 3-Hop Frequent Pattern Analysis', fontsize=18, fontweight='bold')
        
        # 1. Bar chart of top 10 patterns (left)
        patterns = [pattern for pattern, _ in frequent_3hop_patterns['top_10_patterns']]
        counts = [count for _, count in frequent_3hop_patterns['top_10_patterns']]
        # Use actual sampled counts for percentages
        actual_counts = [frequent_3hop_patterns['original_pattern_counts'].get(pattern, 0) for pattern in patterns]
        percentages = [(actual_count / frequent_3hop_patterns['actual_sampled_paths']) * 100 for actual_count in actual_counts]
        
        # Create color map based on pattern diversity
        colors = plt.cm.viridis(np.linspace(0, 1, len(patterns)))
        
        bars = axes[0].bar(range(len(patterns)), counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        axes[0].set_title('Top 10 Most Frequent 3-Hop Patterns', fontweight='bold', fontsize=14)
        axes[0].set_xlabel('Pattern Rank')
        axes[0].set_ylabel('Number of Occurrences (Estimated)')
        axes[0].set_xticks(range(len(patterns)))
        axes[0].set_xticklabels([f'#{i+1}' for i in range(len(patterns))], rotation=45)
        
        # Add value labels on bars
        for i, (bar, count, percentage) in enumerate(zip(bars, counts, percentages)):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{count:,}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=9)
        
        # Add pattern labels below x-axis
        for i, pattern in enumerate(patterns):
            axes[0].text(i, -max(counts)*0.05, pattern, ha='center', va='top', 
                        fontsize=8, rotation=45, fontweight='bold')
        
        # 2. Statistics and pattern information (right)
        stats_text = f"""3-HOP PATTERN STATISTICS
{'='*40}
Sampling Method:
  Nodes: {frequent_3hop_patterns['sampled_nodes_count']:,} / {frequent_3hop_patterns['total_nodes']:,}
  Scale factor: {frequent_3hop_patterns['scale_factor']:.2f}x

Path Counts:
  Sampled paths: {frequent_3hop_patterns['actual_sampled_paths']:,}
  Estimated total: {frequent_3hop_patterns['total_paths']:,}
  Unique patterns: {frequent_3hop_patterns['unique_patterns']:,}

Statistics:
  Pattern diversity: {frequent_3hop_patterns['pattern_diversity']:.6f}
  Top 10 coverage: {frequent_3hop_patterns['top_10_coverage']:.2f}%

TOP 10 PATTERNS
{'='*20}"""
        
        for i, (pattern, count) in enumerate(frequent_3hop_patterns['top_10_patterns'], 1):
            # Use actual sampled count for percentage calculation
            actual_count = frequent_3hop_patterns['original_pattern_counts'].get(pattern, 0)
            percentage = (actual_count / frequent_3hop_patterns['actual_sampled_paths']) * 100
            stats_text += f"\n{i:2d}. {pattern:<15} {count:>8,} ({percentage:>5.1f}%)"
        
        axes[1].text(0.05, 0.95, stats_text, transform=axes[1].transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', 
                             alpha=0.9, edgecolor='darkblue', linewidth=2))
        axes[1].set_title('Pattern Statistics', fontweight='bold', fontsize=14)
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_plots:
            # Create label_distribution_images folder
            save_dir = os.path.join(output_dir, 'label_distribution_images')
            os.makedirs(save_dir, exist_ok=True)
            
            plt.savefig(f'{save_dir}/{dataset_name}_3hop_frequent_patterns.png', 
                       dpi=300, bbox_inches='tight')
            print(f"3-hop frequent patterns plot saved: {save_dir}/{dataset_name}_3hop_frequent_patterns.png")
        
        plt.close()
    
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