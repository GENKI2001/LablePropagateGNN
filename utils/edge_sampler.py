import torch
import numpy as np
import random
from typing import Tuple, Optional, Dict, Any
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
from collections import defaultdict
import torch.nn.functional as F


class EdgeSampler:
    """
    エッジサンプリングを行うクラス
    様々なサンプリング手法を提供
    """
    
    def __init__(self, data: Data, device: torch.device):
        """
        Args:
            data: PyTorch Geometric データオブジェクト
            device: 使用デバイス
        """
        self.data = data
        self.device = device
        self.edge_index = data.edge_index
        self.num_edges = data.edge_index.shape[1]
        self.num_nodes = data.num_nodes
        
        # 隣接行列を作成
        self.adj_matrix = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32)
        self.adj_matrix[self.edge_index[0], self.edge_index[1]] = 1.0
        self.adj_matrix[self.edge_index[1], self.edge_index[0]] = 1.0
        
        # ノードの次数を計算
        self.degrees = self.adj_matrix.sum(dim=1)
        
    def random_sampling(self, sampling_ratio: float = 0.5, seed: Optional[int] = None) -> Data:
        """
        ランダムエッジサンプリング
        
        Args:
            sampling_ratio: サンプリングするエッジの割合 (0.0-1.0)
            seed: 乱数シード
            
        Returns:
            サンプリングされたエッジを含むデータオブジェクト
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            
        num_edges_to_sample = int(self.num_edges * sampling_ratio)
        sampled_indices = torch.randperm(self.num_edges)[:num_edges_to_sample]
        
        sampled_edge_index = self.edge_index[:, sampled_indices]
        
        # 新しいデータオブジェクトを作成
        sampled_data = self.data.clone()
        sampled_data.edge_index = sampled_edge_index
        
        print(f"ランダムサンプリング完了: {self.num_edges} → {num_edges_to_sample} エッジ (比率: {sampling_ratio:.2f})")
        
        return sampled_data
    
    def degree_based_sampling(self, sampling_ratio: float = 0.5, 
                            strategy: str = 'high_degree', seed: Optional[int] = None) -> Data:
        """
        次数ベースのエッジサンプリング
        
        Args:
            sampling_ratio: サンプリングするエッジの割合
            strategy: 'high_degree', 'low_degree', 'balanced'
            seed: 乱数シード
            
        Returns:
            サンプリングされたエッジを含むデータオブジェクト
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            
        num_edges_to_sample = int(self.num_edges * sampling_ratio)
        
        # 各エッジの重みを計算（両端ノードの次数の平均）
        edge_weights = []
        for i in range(self.num_edges):
            src, dst = self.edge_index[0, i], self.edge_index[1, i]
            avg_degree = (self.degrees[src] + self.degrees[dst]) / 2
            edge_weights.append(avg_degree.item())
        
        edge_weights = torch.tensor(edge_weights)
        
        if strategy == 'high_degree':
            # 高次数ノードを優先
            weights = edge_weights
        elif strategy == 'low_degree':
            # 低次数ノードを優先
            weights = 1.0 / (edge_weights + 1e-8)
        elif strategy == 'balanced':
            # バランス型（次数の逆数で重み付け）
            weights = 1.0 / (edge_weights + 1e-8)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # 重み付きサンプリング
        probs = weights / weights.sum()
        sampled_indices = torch.multinomial(probs, num_edges_to_sample, replacement=False)
        
        sampled_edge_index = self.edge_index[:, sampled_indices]
        
        # 新しいデータオブジェクトを作成
        sampled_data = self.data.clone()
        sampled_data.edge_index = sampled_edge_index
        
        print(f"次数ベースサンプリング完了 ({strategy}): {self.num_edges} → {num_edges_to_sample} エッジ")
        
        return sampled_data
    
    def class_balanced_sampling(self, sampling_ratio: float = 0.5, 
                              strategy: str = 'inter_class', seed: Optional[int] = None) -> Data:
        """
        クラスバランスを考慮したエッジサンプリング
        
        Args:
            sampling_ratio: サンプリングするエッジの割合
            strategy: 'inter_class', 'intra_class', 'balanced'
            seed: 乱数シード
            
        Returns:
            サンプリングされたエッジを含むデータオブジェクト
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            
        num_edges_to_sample = int(self.num_edges * sampling_ratio)
        
        # 各エッジのクラス情報を取得
        edge_class_info = []
        for i in range(self.num_edges):
            src, dst = self.edge_index[0, i], self.edge_index[1, i]
            src_class = self.data.y[src].item()
            dst_class = self.data.y[dst].item()
            
            # 訓練ノードのみを考慮
            if self.data.train_mask[src] and self.data.train_mask[dst]:
                is_same_class = (src_class == dst_class)
                edge_class_info.append({
                    'index': i,
                    'src_class': src_class,
                    'dst_class': dst_class,
                    'is_same_class': is_same_class
                })
            else:
                # テスト・検証ノードを含むエッジは重みを下げる
                edge_class_info.append({
                    'index': i,
                    'src_class': src_class,
                    'dst_class': dst_class,
                    'is_same_class': (src_class == dst_class)
                })
        
        # エッジを分類
        inter_class_edges = [e for e in edge_class_info if not e['is_same_class']]
        intra_class_edges = [e for e in edge_class_info if e['is_same_class']]
        
        if strategy == 'inter_class':
            # 異クラス間のエッジを優先
            target_edges = inter_class_edges
        elif strategy == 'intra_class':
            # 同クラス内のエッジを優先
            target_edges = intra_class_edges
        elif strategy == 'balanced':
            # バランス型
            min_edges = min(len(inter_class_edges), len(intra_class_edges))
            target_edges = inter_class_edges[:min_edges] + intra_class_edges[:min_edges]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # サンプリング
        if len(target_edges) >= num_edges_to_sample:
            sampled_edge_info = random.sample(target_edges, num_edges_to_sample)
        else:
            # 目標数に達しない場合は、残りのエッジをランダムに追加
            remaining_edges = [e for e in edge_class_info if e not in target_edges]
            additional_needed = num_edges_to_sample - len(target_edges)
            additional_edges = random.sample(remaining_edges, min(additional_needed, len(remaining_edges)))
            sampled_edge_info = target_edges + additional_edges
        
        sampled_indices = torch.tensor([e['index'] for e in sampled_edge_info])
        sampled_edge_index = self.edge_index[:, sampled_indices]
        
        # 新しいデータオブジェクトを作成
        sampled_data = self.data.clone()
        sampled_data.edge_index = sampled_edge_index
        
        print(f"クラスバランスサンプリング完了 ({strategy}): {self.num_edges} → {len(sampled_edge_info)} エッジ")
        
        return sampled_data
    
    def structural_sampling(self, sampling_ratio: float = 0.5, 
                          strategy: str = 'bridge', seed: Optional[int] = None) -> Data:
        """
        構造的特徴を考慮したエッジサンプリング
        
        Args:
            sampling_ratio: サンプリングするエッジの割合
            strategy: 'bridge', 'clustering', 'centrality'
            seed: 乱数シード
            
        Returns:
            サンプリングされたエッジを含むデータオブジェクト
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            
        num_edges_to_sample = int(self.num_edges * sampling_ratio)
        
        # NetworkXグラフに変換
        G = to_networkx(self.data, to_undirected=True)
        
        if strategy == 'bridge':
            # ブリッジエッジを優先（クラスタリング係数が低いエッジ）
            edge_scores = []
            for i in range(self.num_edges):
                src, dst = self.edge_index[0, i].item(), self.edge_index[1, i].item()
                try:
                    # エッジを削除した場合のクラスタリング係数の変化を計算
                    G_temp = G.copy()
                    G_temp.remove_edge(src, dst)
                    clustering_before = nx.average_clustering(G)
                    clustering_after = nx.average_clustering(G_temp)
                    score = clustering_before - clustering_after
                    edge_scores.append((i, score))
                except:
                    edge_scores.append((i, 0.0))
            
            # スコアでソート（ブリッジエッジを優先）
            edge_scores.sort(key=lambda x: x[1], reverse=True)
            sampled_indices = torch.tensor([idx for idx, _ in edge_scores[:num_edges_to_sample]])
            
        elif strategy == 'clustering':
            # クラスタリング係数が高いエッジを優先
            edge_scores = []
            for i in range(self.num_edges):
                src, dst = self.edge_index[0, i].item(), self.edge_index[1, i].item()
                try:
                    # 両端ノードのクラスタリング係数の平均
                    clustering_src = nx.clustering(G, src)
                    clustering_dst = nx.clustering(G, dst)
                    score = (clustering_src + clustering_dst) / 2
                    edge_scores.append((i, score))
                except:
                    edge_scores.append((i, 0.0))
            
            edge_scores.sort(key=lambda x: x[1], reverse=True)
            sampled_indices = torch.tensor([idx for idx, _ in edge_scores[:num_edges_to_sample]])
            
        elif strategy == 'centrality':
            # 中心性が高いノードを接続するエッジを優先
            edge_scores = []
            betweenness_centrality = nx.betweenness_centrality(G)
            
            for i in range(self.num_edges):
                src, dst = self.edge_index[0, i].item(), self.edge_index[1, i].item()
                score = (betweenness_centrality.get(src, 0) + betweenness_centrality.get(dst, 0)) / 2
                edge_scores.append((i, score))
            
            edge_scores.sort(key=lambda x: x[1], reverse=True)
            sampled_indices = torch.tensor([idx for idx, _ in edge_scores[:num_edges_to_sample]])
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        sampled_edge_index = self.edge_index[:, sampled_indices]
        
        # 新しいデータオブジェクトを作成
        sampled_data = self.data.clone()
        sampled_data.edge_index = sampled_edge_index
        
        print(f"構造的サンプリング完了 ({strategy}): {self.num_edges} → {num_edges_to_sample} エッジ")
        
        return sampled_data
    
    def adaptive_sampling(self, sampling_ratio: float = 0.5, 
                         alpha: float = 0.5, seed: Optional[int] = None) -> Data:
        """
        適応的エッジサンプリング（複数の指標を組み合わせ）
        
        Args:
            sampling_ratio: サンプリングするエッジの割合
            alpha: 次数とクラスバランスの重み (0.0-1.0)
            seed: 乱数シード
            
        Returns:
            サンプリングされたエッジを含むデータオブジェクト
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            
        num_edges_to_sample = int(self.num_edges * sampling_ratio)
        
        # 複数のスコアを計算
        edge_scores = []
        for i in range(self.num_edges):
            src, dst = self.edge_index[0, i], self.edge_index[1, i]
            
            # 次数スコア
            degree_score = (self.degrees[src] + self.degrees[dst]) / 2
            
            # クラスバランススコア
            src_class = self.data.y[src].item()
            dst_class = self.data.y[dst].item()
            class_score = 1.0 if src_class != dst_class else 0.5  # 異クラス間を優先
            
            # 訓練ノードスコア
            train_score = 1.0
            if not (self.data.train_mask[src] and self.data.train_mask[dst]):
                train_score = 0.3  # テスト・検証ノードを含むエッジは重みを下げる
            
            # 総合スコア
            total_score = alpha * degree_score + (1 - alpha) * class_score * train_score
            edge_scores.append((i, total_score))
        
        # スコアでソート
        edge_scores.sort(key=lambda x: x[1], reverse=True)
        sampled_indices = torch.tensor([idx for idx, _ in edge_scores[:num_edges_to_sample]])
        
        sampled_edge_index = self.edge_index[:, sampled_indices]
        
        # 新しいデータオブジェクトを作成
        sampled_data = self.data.clone()
        sampled_data.edge_index = sampled_edge_index
        
        print(f"適応的サンプリング完了 (α={alpha}): {self.num_edges} → {num_edges_to_sample} エッジ")
        
        return sampled_data
    
    def get_sampling_statistics(self, sampled_data: Data) -> Dict[str, Any]:
        """
        サンプリング結果の統計情報を取得
        
        Args:
            sampled_data: サンプリングされたデータ
            
        Returns:
            統計情報の辞書
        """
        original_edges = self.edge_index
        sampled_edges = sampled_data.edge_index
        
        # 基本的な統計
        stats = {
            'original_edges': self.num_edges,
            'sampled_edges': sampled_edges.shape[1],
            'sampling_ratio': sampled_edges.shape[1] / self.num_edges,
            'original_nodes': self.num_nodes,
            'sampled_nodes': len(torch.unique(sampled_edges))
        }
        
        # クラス間エッジの統計
        inter_class_count = 0
        intra_class_count = 0
        
        for i in range(sampled_edges.shape[1]):
            src, dst = sampled_edges[0, i], sampled_edges[1, i]
            src_class = self.data.y[src].item()
            dst_class = self.data.y[dst].item()
            
            if src_class == dst_class:
                intra_class_count += 1
            else:
                inter_class_count += 1
        
        stats['inter_class_edges'] = inter_class_count
        stats['intra_class_edges'] = intra_class_count
        stats['inter_class_ratio'] = inter_class_count / sampled_edges.shape[1]
        
        # 次数分布の統計
        sampled_adj = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32)
        sampled_adj[sampled_edges[0], sampled_edges[1]] = 1.0
        sampled_adj[sampled_edges[1], sampled_edges[0]] = 1.0
        sampled_degrees = sampled_adj.sum(dim=1)
        
        stats['avg_degree'] = sampled_degrees.mean().item()
        stats['max_degree'] = sampled_degrees.max().item()
        stats['min_degree'] = sampled_degrees.min().item()
        
        return stats


def sample_edges(data: Data, device: torch.device, 
                method: str = 'random', 
                sampling_ratio: float = 0.5,
                **kwargs) -> Tuple[Data, Dict[str, Any]]:
    """
    エッジサンプリングのメイン関数
    
    Args:
        data: PyTorch Geometric データオブジェクト
        device: 使用デバイス
        method: サンプリング手法 ('random', 'degree', 'class', 'structural', 'adaptive')
        sampling_ratio: サンプリングするエッジの割合
        **kwargs: 各手法固有のパラメータ
        
    Returns:
        サンプリングされたデータと統計情報のタプル
    """
    sampler = EdgeSampler(data, device)
    
    if method == 'random':
        sampled_data = sampler.random_sampling(sampling_ratio, **kwargs)
    elif method == 'degree':
        strategy = kwargs.get('strategy', 'high_degree')
        sampled_data = sampler.degree_based_sampling(sampling_ratio, strategy, **kwargs)
    elif method == 'class':
        strategy = kwargs.get('strategy', 'inter_class')
        sampled_data = sampler.class_balanced_sampling(sampling_ratio, strategy, **kwargs)
    elif method == 'structural':
        strategy = kwargs.get('strategy', 'bridge')
        sampled_data = sampler.structural_sampling(sampling_ratio, strategy, **kwargs)
    elif method == 'adaptive':
        alpha = kwargs.get('alpha', 0.5)
        sampled_data = sampler.adaptive_sampling(sampling_ratio, alpha, **kwargs)
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    # 統計情報を取得
    stats = sampler.get_sampling_statistics(sampled_data)
    
    return sampled_data, stats


def print_sampling_statistics(stats: Dict[str, Any]):
    """
    サンプリング統計情報を表示
    
    Args:
        stats: 統計情報の辞書
    """
    print(f"\n=== エッジサンプリング統計 ===")
    print(f"元のエッジ数: {stats['original_edges']}")
    print(f"サンプリング後エッジ数: {stats['sampled_edges']}")
    print(f"サンプリング比率: {stats['sampling_ratio']:.3f}")
    print(f"元のノード数: {stats['original_nodes']}")
    print(f"サンプリング後ノード数: {stats['sampled_nodes']}")
    print(f"クラス間エッジ: {stats['inter_class_edges']} ({stats['inter_class_ratio']:.3f})")
    print(f"クラス内エッジ: {stats['intra_class_edges']}")
    print(f"平均次数: {stats['avg_degree']:.2f}")
    print(f"最大次数: {stats['max_degree']}")
    print(f"最小次数: {stats['min_degree']}") 