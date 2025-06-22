import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import warnings


class EdgeEnhancer:
    """
    特徴量の類似度に基づいてエッジを追加するクラス
    """
    
    def __init__(self, similarity_method='cosine', threshold=0.8, max_edges_per_node=None, 
                 symmetric=True, normalize_features=True):
        """
        Args:
            similarity_method (str): 類似度計算方法 ('cosine', 'euclidean', 'pearson', 'jaccard')
            threshold (float): エッジ追加の閾値 (0.0-1.0)
            max_edges_per_node (int): ノードあたりの最大エッジ数 (Noneの場合は制限なし)
            symmetric (bool): 対称的なエッジ追加を行うか
            normalize_features (bool): 特徴量を正規化するか
        """
        self.similarity_method = similarity_method
        self.threshold = threshold
        self.max_edges_per_node = max_edges_per_node
        self.symmetric = symmetric
        self.normalize_features = normalize_features
        
        # 類似度計算方法の検証
        valid_methods = ['cosine', 'euclidean', 'pearson', 'jaccard']
        if similarity_method not in valid_methods:
            raise ValueError(f"similarity_method must be one of {valid_methods}")
    
    def _normalize_features(self, features):
        """特徴量を正規化"""
        if self.normalize_features:
            # L2正規化
            norm = torch.norm(features, p=2, dim=1, keepdim=True)
            norm = torch.where(norm == 0, 1.0, norm)  # ゼロ除算を避ける
            return features / norm
        return features
    
    def _compute_similarity_matrix(self, features):
        """類似度行列を計算"""
        features_np = features.detach().cpu().numpy()
        
        if self.similarity_method == 'cosine':
            # コサイン類似度
            similarity_matrix = cosine_similarity(features_np)
        
        elif self.similarity_method == 'euclidean':
            # ユークリッド距離を類似度に変換
            distances = euclidean_distances(features_np)
            max_distance = np.max(distances)
            similarity_matrix = 1 - (distances / max_distance)
        
        elif self.similarity_method == 'pearson':
            # ピアソン相関係数
            similarity_matrix = np.corrcoef(features_np)
            # NaN値を0に置換
            similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0)
        
        elif self.similarity_method == 'jaccard':
            # Jaccard類似度（バイナリ特徴量用）
            # 特徴量をバイナリ化
            binary_features = (features_np > 0).astype(float)
            similarity_matrix = np.zeros((len(features_np), len(features_np)))
            
            for i in range(len(features_np)):
                for j in range(len(features_np)):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        intersection = np.sum(np.minimum(binary_features[i], binary_features[j]))
                        union = np.sum(np.maximum(binary_features[i], binary_features[j]))
                        if union > 0:
                            similarity_matrix[i, j] = intersection / union
                        else:
                            similarity_matrix[i, j] = 0.0
        
        return torch.tensor(similarity_matrix, dtype=torch.float32, device=features.device)
    
    def _find_similar_pairs(self, similarity_matrix, existing_edges=None):
        """類似度が閾値を超えるノードペアを見つける"""
        num_nodes = similarity_matrix.shape[0]
        
        # 既存のエッジをセットとして保存
        existing_edge_set = set()
        if existing_edges is not None:
            edge_index = existing_edges.t().tolist()
            existing_edge_set = set((u, v) for u, v in edge_index)
            if self.symmetric:
                existing_edge_set.update((v, u) for u, v in edge_index)
        
        # 類似度が閾値を超えるペアを収集
        similar_pairs = []
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # 対称性を考慮して半分だけ
                if similarity_matrix[i, j] >= self.threshold:
                    # 既存のエッジでない場合のみ追加
                    if (i, j) not in existing_edge_set and (j, i) not in existing_edge_set:
                        similar_pairs.append((i, j, similarity_matrix[i, j].item()))
        
        # 類似度で降順ソート
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return similar_pairs
    
    def _limit_edges_per_node(self, similar_pairs, num_nodes):
        """ノードあたりのエッジ数を制限"""
        if self.max_edges_per_node is None:
            return similar_pairs
        
        edge_counts = {i: 0 for i in range(num_nodes)}
        filtered_pairs = []
        
        for i, j, similarity in similar_pairs:
            if (edge_counts[i] < self.max_edges_per_node and 
                edge_counts[j] < self.max_edges_per_node):
                filtered_pairs.append((i, j, similarity))
                edge_counts[i] += 1
                edge_counts[j] += 1
        
        return filtered_pairs
    
    def enhance_edges(self, data, original_features=None):
        """
        特徴量の類似度に基づいてエッジを追加
        
        Args:
            data: PyTorch GeometricのDataオブジェクト
            original_features: 元の特徴量（Noneの場合はdata.xを使用）
        
        Returns:
            enhanced_data: エッジが追加されたDataオブジェクト
            added_edges: 追加されたエッジの情報
        """
        if original_features is None:
            original_features = data.x
        
        # 特徴量の正規化
        normalized_features = self._normalize_features(original_features)
        
        # 類似度行列の計算
        similarity_matrix = self._compute_similarity_matrix(normalized_features)
        
        # 類似度が閾値を超えるペアを見つける
        similar_pairs = self._find_similar_pairs(similarity_matrix, data.edge_index)
        
        # ノードあたりのエッジ数を制限
        filtered_pairs = self._limit_edges_per_node(similar_pairs, data.num_nodes)
        
        if not filtered_pairs:
            print(f"閾値 {self.threshold} を超える類似度を持つノードペアが見つかりませんでした。")
            return data, []
        
        # 新しいエッジを作成
        new_edges = []
        for i, j, similarity in filtered_pairs:
            new_edges.append([i, j])
            if self.symmetric:
                new_edges.append([j, i])
        
        new_edge_index = torch.tensor(new_edges, dtype=torch.long, device=data.edge_index.device).t()
        
        # 既存のエッジと新しいエッジを結合
        enhanced_edge_index = torch.cat([data.edge_index, new_edge_index], dim=1)
        
        # 重複エッジを除去（オプション）
        enhanced_edge_index = self._remove_duplicate_edges(enhanced_edge_index)
        
        # 新しいDataオブジェクトを作成
        enhanced_data = data.clone()
        enhanced_data.edge_index = enhanced_edge_index
        
        # 追加されたエッジの情報
        added_edges_info = {
            'num_original_edges': data.edge_index.shape[1],
            'num_added_edges': len(new_edges),
            'num_total_edges': enhanced_edge_index.shape[1],
            'similarity_threshold': self.threshold,
            'similarity_method': self.similarity_method,
            'edge_pairs': filtered_pairs
        }
        
        print(f"エッジ追加完了:")
        print(f"  元のエッジ数: {added_edges_info['num_original_edges']}")
        print(f"  追加エッジ数: {added_edges_info['num_added_edges']}")
        print(f"  総エッジ数: {added_edges_info['num_total_edges']}")
        print(f"  類似度閾値: {self.threshold}")
        print(f"  類似度計算方法: {self.similarity_method}")
        
        return enhanced_data, added_edges_info
    
    def _remove_duplicate_edges(self, edge_index):
        """重複エッジを除去"""
        edges = edge_index.t().tolist()
        unique_edges = list(set(map(tuple, edges)))
        return torch.tensor(unique_edges, dtype=torch.long, device=edge_index.device).t()


def enhance_edges_by_similarity(data, similarity_method='cosine', threshold=0.8, 
                               max_edges_per_node=None, symmetric=True, 
                               normalize_features=True, original_features=None):
    """
    特徴量の類似度に基づいてエッジを追加する便利関数
    
    Args:
        data: PyTorch GeometricのDataオブジェクト
        similarity_method (str): 類似度計算方法
        threshold (float): エッジ追加の閾値
        max_edges_per_node (int): ノードあたりの最大エッジ数
        symmetric (bool): 対称的なエッジ追加を行うか
        normalize_features (bool): 特徴量を正規化するか
        original_features: 元の特徴量
    
    Returns:
        enhanced_data: エッジが追加されたDataオブジェクト
        added_edges_info: 追加されたエッジの情報
    """
    enhancer = EdgeEnhancer(
        similarity_method=similarity_method,
        threshold=threshold,
        max_edges_per_node=max_edges_per_node,
        symmetric=symmetric,
        normalize_features=normalize_features
    )
    
    return enhancer.enhance_edges(data, original_features)


def analyze_similarity_distribution(data, similarity_method='cosine', 
                                  normalize_features=True, original_features=None):
    """
    特徴量の類似度分布を分析する関数
    
    Args:
        data: PyTorch GeometricのDataオブジェクト
        similarity_method (str): 類似度計算方法
        normalize_features (bool): 特徴量を正規化するか
        original_features: 元の特徴量
    
    Returns:
        similarity_stats: 類似度の統計情報
    """
    if original_features is None:
        original_features = data.x
    
    enhancer = EdgeEnhancer(
        similarity_method=similarity_method,
        threshold=0.0,  # 全ペアを分析
        normalize_features=normalize_features
    )
    
    normalized_features = enhancer._normalize_features(original_features)
    similarity_matrix = enhancer._compute_similarity_matrix(normalized_features)
    
    # 対角成分（自己類似度）を除外
    mask = ~torch.eye(similarity_matrix.shape[0], dtype=torch.bool, device=similarity_matrix.device)
    similarities = similarity_matrix[mask]
    
    stats = {
        'mean': similarities.mean().item(),
        'std': similarities.std().item(),
        'min': similarities.min().item(),
        'max': similarities.max().item(),
        'median': similarities.median().item(),
        'percentiles': {
            '25': torch.quantile(similarities, 0.25).item(),
            '50': torch.quantile(similarities, 0.50).item(),
            '75': torch.quantile(similarities, 0.75).item(),
            '90': torch.quantile(similarities, 0.90).item(),
            '95': torch.quantile(similarities, 0.95).item(),
            '99': torch.quantile(similarities, 0.99).item()
        }
    }
    
    print(f"類似度分布分析 ({similarity_method}):")
    print(f"  平均: {stats['mean']:.4f}")
    print(f"  標準偏差: {stats['std']:.4f}")
    print(f"  最小値: {stats['min']:.4f}")
    print(f"  最大値: {stats['max']:.4f}")
    print(f"  中央値: {stats['median']:.4f}")
    print(f"  75パーセンタイル: {stats['percentiles']['75']:.4f}")
    print(f"  90パーセンタイル: {stats['percentiles']['90']:.4f}")
    print(f"  95パーセンタイル: {stats['percentiles']['95']:.4f}")
    
    return stats 