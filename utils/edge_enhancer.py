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
            similarity_method (str): 類似度計算方法 ('cosine', 'euclidean', 'pearson', 'jaccard', 
                                    'manhattan', 'chebyshev', 'spearman', 'mahalanobis', 
                                    'rbf_kernel', 'polynomial_kernel', 'laplacian_kernel')
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
        valid_methods = ['cosine', 'euclidean', 'pearson', 'jaccard', 'manhattan', 
                        'chebyshev', 'spearman', 'mahalanobis', 'rbf_kernel', 
                        'polynomial_kernel', 'laplacian_kernel']
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
        
        elif self.similarity_method == 'manhattan':
            # マンハッタン距離を類似度に変換
            from sklearn.metrics.pairwise import manhattan_distances
            distances = manhattan_distances(features_np)
            max_distance = np.max(distances)
            similarity_matrix = 1 - (distances / max_distance)
        
        elif self.similarity_method == 'chebyshev':
            # チェビシェフ距離を類似度に変換
            from sklearn.metrics.pairwise import pairwise_distances
            distances = pairwise_distances(features_np, metric='chebyshev')
            max_distance = np.max(distances)
            similarity_matrix = 1 - (distances / max_distance)
        
        elif self.similarity_method == 'spearman':
            # スピアマン相関係数
            from scipy.stats import spearmanr
            similarity_matrix = np.zeros((len(features_np), len(features_np)))
            
            for i in range(len(features_np)):
                for j in range(len(features_np)):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        try:
                            corr, _ = spearmanr(features_np[i], features_np[j])
                            similarity_matrix[i, j] = corr if not np.isnan(corr) else 0.0
                        except:
                            similarity_matrix[i, j] = 0.0
        
        elif self.similarity_method == 'mahalanobis':
            # マハラノビス距離を類似度に変換
            try:
                from scipy.spatial.distance import mahalanobis
                from scipy.linalg import inv
                
                # 共分散行列を計算
                cov_matrix = np.cov(features_np.T)
                inv_cov_matrix = inv(cov_matrix)
                
                similarity_matrix = np.zeros((len(features_np), len(features_np)))
                for i in range(len(features_np)):
                    for j in range(len(features_np)):
                        if i == j:
                            similarity_matrix[i, j] = 1.0
                        else:
                            try:
                                dist = mahalanobis(features_np[i], features_np[j], inv_cov_matrix)
                                similarity_matrix[i, j] = 1.0 / (1.0 + dist)  # 距離を類似度に変換
                            except:
                                similarity_matrix[i, j] = 0.0
            except:
                # マハラノビス距離が計算できない場合はユークリッド距離を使用
                print("警告: マハラノビス距離の計算に失敗しました。ユークリッド距離を使用します。")
                distances = euclidean_distances(features_np)
                max_distance = np.max(distances)
                similarity_matrix = 1 - (distances / max_distance)
        
        elif self.similarity_method == 'rbf_kernel':
            # RBFカーネル（ガウシアンカーネル）
            from sklearn.metrics.pairwise import rbf_kernel
            # gammaパラメータを自動調整（特徴量の次元に基づいて）
            gamma = 1.0 / features_np.shape[1]
            similarity_matrix = rbf_kernel(features_np, gamma=gamma)
        
        elif self.similarity_method == 'polynomial_kernel':
            # 多項式カーネル
            from sklearn.metrics.pairwise import polynomial_kernel
            similarity_matrix = polynomial_kernel(features_np, degree=2, coef0=1)
        
        elif self.similarity_method == 'laplacian_kernel':
            # ラプラシアンカーネル
            from sklearn.metrics.pairwise import laplacian_kernel
            # gammaパラメータを自動調整
            gamma = 1.0 / features_np.shape[1]
            similarity_matrix = laplacian_kernel(features_np, gamma=gamma)
        
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
            - 'cosine': コサイン類似度（ベクトル間の角度）
            - 'euclidean': ユークリッド距離ベース類似度
            - 'pearson': ピアソン相関係数（線形相関）
            - 'jaccard': Jaccard類似度（バイナリ特徴量用）
            - 'manhattan': マンハッタン距離ベース類似度
            - 'chebyshev': チェビシェフ距離ベース類似度
            - 'spearman': スピアマン相関係数（順位相関）
            - 'mahalanobis': マハラノビス距離ベース類似度（共分散考慮）
            - 'rbf_kernel': RBFカーネル（ガウシアンカーネル）
            - 'polynomial_kernel': 多項式カーネル
            - 'laplacian_kernel': ラプラシアンカーネル
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


def enhance_edges_by_label_distribution(data, feature_info, similarity_method='cosine', threshold=0.8,
                                       max_edges_per_node=None, symmetric=True, normalize_features=True,
                                       data_without_pca=None, top_percentile=None):
    """
    ラベル分布に関わる部分（ラベル分布ベクトルと共起ラベル分布）のみを使用してエッジを追加する関数
    
    Args:
        data: PyTorch GeometricのDataオブジェクト
        feature_info: 特徴量の情報を含む辞書（get_feature_infoで取得）
        similarity_method (str): 類似度計算方法 ('cosine', 'euclidean', 'pearson', 'jaccard')
        threshold (float): エッジ追加の閾値 (0.0-1.0)
        max_edges_per_node (int): ノードあたりの最大エッジ数 (Noneの場合は制限なし)
        symmetric (bool): 対称的なエッジ追加を行うか
        normalize_features (bool): 特徴量を正規化するか
        data_without_pca: PCA特徴量のないデータオブジェクト（ラベル分布特徴量のみ）
        top_percentile (float): 類似度の上位何%のみを選択するか (0.0-1.0, Noneの場合は制限なし)
    
    Returns:
        enhanced_data: エッジが追加されたDataオブジェクト
        added_edges_info: 追加されたエッジの情報
    """
    
    print(f"ラベル分布ベースのエッジ追加を開始...")
    print(f"類似度計算方法: {similarity_method}")
    print(f"類似度閾値: {threshold}")
    if top_percentile is not None:
        print(f"上位パーセンタイル制限: {top_percentile:.1%}")
    
    # PCA特徴量のないデータオブジェクトがある場合は、それを使用
    if data_without_pca is not None:
        label_distribution_features = data_without_pca.x
    else:
        # data_without_pcaが提供されていない場合はエラー
        raise ValueError("data_without_pcaが提供されていません。ラベル分布特徴量の抽出には正確な情報が必要です。")
    
    # ラベル分布特徴量を使用してエッジを追加
    enhancer = EdgeEnhancer(
        similarity_method=similarity_method,
        threshold=threshold,
        max_edges_per_node=max_edges_per_node,
        symmetric=symmetric,
        normalize_features=normalize_features
    )
    
    # 類似度行列を計算
    normalized_features = enhancer._normalize_features(label_distribution_features)
    similarity_matrix = enhancer._compute_similarity_matrix(normalized_features)
    
    # 上位パーセンタイル制限がある場合の処理
    if top_percentile is not None:
        # 対角成分（自己類似度）を除外
        mask = ~torch.eye(similarity_matrix.shape[0], dtype=torch.bool, device=similarity_matrix.device)
        similarities = similarity_matrix[mask]
        
        # 上位パーセンタイルの閾値を計算
        percentile_threshold = torch.quantile(similarities, 1.0 - top_percentile).item()
        print(f"上位{top_percentile:.1%}の閾値: {percentile_threshold:.4f}")
        
        # より厳しい閾値を使用
        final_threshold = max(threshold, percentile_threshold)
        print(f"最終閾値: {final_threshold:.4f}")
        
        # 閾値を更新
        enhancer.threshold = final_threshold
    
    enhanced_data, added_edges_info = enhancer.enhance_edges(data, label_distribution_features)
    
    # 追加情報を更新
    added_edges_info['label_distribution_features_shape'] = label_distribution_features.shape
    added_edges_info['feature_type'] = 'label_distribution'
    added_edges_info['used_data_without_pca'] = data_without_pca is not None
    if top_percentile is not None:
        added_edges_info['top_percentile'] = top_percentile
    
    print(f"ラベル分布ベースのエッジ追加完了:")
    print(f"  元のエッジ数: {added_edges_info['num_original_edges']}")
    print(f"  追加エッジ数: {added_edges_info['num_added_edges']}")
    print(f"  総エッジ数: {added_edges_info['num_total_edges']}")
    print(f"  エッジ増加率: {added_edges_info['num_added_edges'] / added_edges_info['num_original_edges']:.1%}")
    
    return enhanced_data, added_edges_info


def extract_label_distribution_features(data, feature_info, pca_components=None, co_label_embedding_dim=None):
    """
    特徴量からラベル分布に関わる部分（ラベル分布ベクトルと共起ラベル分布）のみを抽出する関数
    
    Args:
        data: PyTorch GeometricのDataオブジェクト
        feature_info: 特徴量の情報を含む辞書（get_feature_infoで取得）
        pca_components (int, optional): PCAで圧縮する次元数（正確な情報がある場合）
        co_label_embedding_dim (int, optional): 共起ラベルエンベディングの次元数（正確な情報がある場合）
    
    Returns:
        label_distribution_features: ラベル分布に関わる特徴量のみ
    """
    
    # 正確な情報がある場合は、それを使用
    if pca_components is not None and co_label_embedding_dim is not None:
        try:
            from utils.feature_creator import extract_label_distribution_features_precise
            label_distribution_features, indices = extract_label_distribution_features_precise(
                data, feature_info, pca_components, co_label_embedding_dim
            )
            return label_distribution_features
        except ImportError:
            print("警告: feature_creatorから正確な抽出関数をインポートできません。")
            return None
    
    # 正確な情報がない場合はNoneを返す（推定は行わない）
    print("警告: 正確な特徴量情報が提供されていません。data_without_pcaを使用してください。")
    return None


def analyze_label_distribution_similarity(data, feature_info, similarity_method='cosine', 
                                        normalize_features=True, data_without_pca=None):
    """
    ラベル分布特徴量の類似度分布を分析する関数
    
    Args:
        data: PyTorch GeometricのDataオブジェクト
        feature_info: 特徴量の情報を含む辞書（get_feature_infoで取得）
        similarity_method (str): 類似度計算方法
        normalize_features (bool): 特徴量を正規化するか
        data_without_pca: PCA特徴量のないデータオブジェクト（ラベル分布特徴量のみ）
    
    Returns:
        similarity_stats: 類似度の統計情報
    """
    
    print(f"ラベル分布特徴量の類似度分布を分析中...")
    
    # ラベル分布特徴量を抽出
    if data_without_pca is not None:
        label_distribution_features = data_without_pca.x
    else:
        raise ValueError("data_without_pcaが提供されていません。ラベル分布特徴量の分析には正確な情報が必要です。")
    
    enhancer = EdgeEnhancer(
        similarity_method=similarity_method,
        threshold=0.0,  # 全ペアを分析
        normalize_features=normalize_features
    )
    
    normalized_features = enhancer._normalize_features(label_distribution_features)
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
    
    return stats 