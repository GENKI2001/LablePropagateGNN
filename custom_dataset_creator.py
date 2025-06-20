import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
import numpy as np
import networkx as nx
import os
from typing import Optional, Callable
import random

class CustomGraphDataset(Dataset):
    """
    カスタムグラフデータセットクラス
    指定された接続パターンを持つグラフを生成
    """
    
    def __init__(self, root: str, name: str = "CustomGraph", 
                 num_nodes: int = 1200, num_classes: int = 5,
                 feature_dim: int = 128, target_avg_degree: float = 3.0,
                 connection_patterns: dict = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        """
        Args:
            root (str): データセットの保存ディレクトリ
            name (str): データセット名
            num_nodes (int): ノード数（1000以上）
            num_classes (int): クラス数（5）
            feature_dim (int): 特徴量の次元数
            target_avg_degree (float): 目標平均次数
            connection_patterns (dict): 接続パターン辞書
                {source_class: [target_classes]} の形式
                例: {0: [2, 3, 4], 1: [3, 4], 2: [0], 3: [1, 4], 4: [0, 1, 3]}
            transform: データ変換関数
            pre_transform: 事前変換関数
        """
        self.name = name
        self.num_nodes = num_nodes
        self.n_classes = num_classes
        self.feature_dim = feature_dim
        self.target_avg_degree = target_avg_degree
        
        # デフォルトの接続パターン
        if connection_patterns is None:
            connection_patterns = {
                0: [2, 3, 4],  # クラス1 (インデックス0) は クラス3,4,5 (インデックス2,3,4) と接続
                1: [3, 4],     # クラス2 (インデックス1) は クラス4,5 (インデックス3,4) と接続
                2: [0],        # クラス3 (インデックス2) は クラス1 (インデックス0) と接続
                3: [1, 4],     # クラス4 (インデックス3) は クラス2,5 (インデックス1,4) と接続
                4: [0, 1, 3]   # クラス5 (インデックス4) は クラス1,2,4 (インデックス0,1,3) と接続
            }
        
        self.connection_patterns = connection_patterns
        super().__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """生データファイル名（この場合は使用しない）"""
        return []
    
    @property
    def processed_file_names(self):
        """処理済みデータファイル名"""
        return [f'{self.name}_data.pt']
    
    def download(self):
        """ダウンロード処理（この場合は使用しない）"""
        pass
    
    def process(self):
        """データセットの生成処理"""
        print(f"=== {self.name} データセット生成開始 ===")
        print(f"ノード数: {self.num_nodes}")
        print(f"クラス数: {self.n_classes}")
        print(f"目標平均次数: {self.target_avg_degree}")
        
        # ノードのクラス割り当て
        nodes_per_class = self.num_nodes // self.n_classes
        remaining_nodes = self.num_nodes % self.n_classes
        
        # 各クラスにノードを割り当て
        node_labels = []
        for i in range(self.n_classes):
            if i < remaining_nodes:
                class_size = nodes_per_class + 1
            else:
                class_size = nodes_per_class
            node_labels.extend([i] * class_size)
        
        # ラベルをシャッフル
        random.shuffle(node_labels)
        node_labels = torch.tensor(node_labels, dtype=torch.long)
        
        print(f"各クラスのノード数:")
        for i in range(self.n_classes):
            count = (node_labels == i).sum().item()
            print(f"  クラス {i+1}: {count} ノード")
        
        # 特徴量の生成（ランダム）
        features = torch.randn(self.num_nodes, self.feature_dim)
        
        # エッジの生成（指定された接続パターンに従う）
        edges = self._generate_edges_with_target_degree(node_labels)
        
        # マスクの生成（60% 訓練、20% 検証、20% テスト）
        train_mask, val_mask, test_mask = self._generate_masks()
        
        # PyTorch Geometric データオブジェクトの作成
        data = Data(
            x=features,
            edge_index=edges,
            y=node_labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        
        # データの保存
        torch.save(data, os.path.join(self.processed_dir, f'{self.name}_data.pt'))
        
        # 実際の平均次数を計算
        actual_avg_degree = edges.shape[1] / self.num_nodes
        
        print(f"=== {self.name} データセット生成完了 ===")
        print(f"エッジ数: {edges.shape[1]}")
        print(f"特徴量次元: {features.shape[1]}")
        print(f"実際の平均次数: {actual_avg_degree:.2f}")
        print(f"訓練ノード数: {train_mask.sum().item()}")
        print(f"検証ノード数: {val_mask.sum().item()}")
        print(f"テストノード数: {test_mask.sum().item()}")
    
    def _generate_edges_with_target_degree(self, node_labels):
        """目標平均次数を保ちながら、指定された接続パターンに従ってエッジを生成"""
        edges_set = set()  # 重複を防ぐためsetで管理
        
        # クラスごとのノードインデックスを取得
        class_nodes = {}
        for i in range(self.n_classes):
            class_nodes[i] = torch.where(node_labels == i)[0].tolist()
        
        # 目標エッジ数を計算（無向グラフなので2で割る）
        target_total_edges = int(self.num_nodes * self.target_avg_degree / 2)
        
        print(f"目標エッジ数: {target_total_edges}")
        
        # 1. クラス内接続は削除（同ラベル同士の接続を禁止）
        print(f"クラス内接続: 無効（同ラベル同士の接続を禁止）")
        
        # 2. クラス間接続を生成
        # 各クラスからターゲットクラスに最大2本ずつ接続
        max_edges_per_connection = 2  # 各接続パターンにつき最大2本
        
        for source_class, target_classes in self.connection_patterns.items():
            source_nodes = class_nodes[source_class]
            for target_class in target_classes:
                target_nodes = class_nodes[target_class]
                
                # 各接続パターンにつき最大2本のエッジを生成
                edges_to_add = min(max_edges_per_connection, len(source_nodes), len(target_nodes))
                for _ in range(edges_to_add):
                    source_node = random.choice(source_nodes)
                    target_node = random.choice(target_nodes)
                    edge = tuple(sorted([source_node, target_node]))
                    edges_set.add(edge)
        
        print(f"必須クラス間接続後エッジ数: {len(edges_set)}")
        
        # 追加：確率的なクラス間接続
        inter_prob = 0.005  # 低めの確率
        for source_class, target_classes in self.connection_patterns.items():
            source_nodes = class_nodes[source_class]
            for target_class in target_classes:
                target_nodes = class_nodes[target_class]
                for source_node in source_nodes:
                    for target_node in target_nodes:
                        if random.random() < inter_prob:
                            edge = tuple(sorted([source_node, target_node]))
                            edges_set.add(edge)
        
        print(f"確率的クラス間接続後エッジ数: {len(edges_set)}")
        
        # 3. 目標エッジ数に達していない場合、ランダムな接続を追加（異なるラベル間のみ）
        current_edges = len(edges_set)
        if current_edges < target_total_edges:
            additional_edges_needed = target_total_edges - current_edges
            print(f"追加エッジ数: {additional_edges_needed}")
            
            # ランダムな接続を追加（異なるラベル間のみ）
            attempts = 0
            max_attempts = additional_edges_needed * 20  # より多くの試行回数
            
            while len(edges_set) < target_total_edges and attempts < max_attempts:
                # ランダムに2つの異なるクラスを選択
                class1 = random.randint(0, self.n_classes - 1)
                class2 = random.randint(0, self.n_classes - 1)
                
                if class1 != class2:  # 異なるクラスのみ
                    if class_nodes[class1] and class_nodes[class2]:  # 両クラスにノードが存在
                        node1 = random.choice(class_nodes[class1])
                        node2 = random.choice(class_nodes[class2])
                        edge = tuple(sorted([node1, node2]))
                        edges_set.add(edge)
                
                attempts += 1
        
        print(f"最終エッジ数: {len(edges_set)}")
        
        # setからエッジリストに変換（無向グラフなので両方向を追加）
        edges = []
        for edge in edges_set:
            edges.append([edge[0], edge[1]])
            edges.append([edge[1], edge[0]])  # 無向グラフ
        
        # エッジリストをテンソルに変換
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return edge_index
    
    def _generate_masks(self):
        """訓練・検証・テストマスクを生成"""
        num_nodes = self.num_nodes
        indices = torch.randperm(num_nodes)
        
        # 60% 訓練、20% 検証、20% テスト
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        
        return train_mask, val_mask, test_mask
    
    def len(self):
        """データセットのサイズ"""
        return 1
    
    def get(self, idx):
        """データの取得"""
        data = torch.load(os.path.join(self.processed_dir, f'{self.name}_data.pt'))
        return data

def create_custom_dataset(num_nodes: int = 1200, 
                         name: str = "CustomGraph",
                         feature_dim: int = 128,
                         target_avg_degree: float = 3.0,
                         connection_patterns: dict = None,
                         root: str = "/tmp/CustomGraph"):
    """
    カスタムデータセットを作成する関数
    
    Args:
        num_nodes (int): ノード数（1000以上）
        name (str): データセット名
        feature_dim (int): 特徴量の次元数
        target_avg_degree (float): 目標平均次数
        connection_patterns (dict): 接続パターン辞書
            {source_class: [target_classes]} の形式
            例: {0: [2, 3, 4], 1: [3, 4], 2: [0], 3: [1, 4], 4: [0, 1, 3]}
        root (str): データセットの保存ディレクトリ
    
    Returns:
        CustomGraphDataset: 作成されたデータセット
    """
    if num_nodes < 1000:
        raise ValueError("ノード数は1000以上である必要があります")
    
    dataset = CustomGraphDataset(
        root=root,
        name=name,
        num_nodes=num_nodes,
        num_classes=5,
        feature_dim=feature_dim,
        target_avg_degree=target_avg_degree,
        connection_patterns=connection_patterns
    )
    
    return dataset

def analyze_dataset(dataset):
    """データセットの詳細分析"""
    data = dataset[0]
    
    print(f"\n=== {dataset.name} データセット詳細分析 ===")
    print(f"ノード数: {data.num_nodes}")
    print(f"エッジ数: {data.edge_index.shape[1]}")
    print(f"特徴量次元: {data.x.shape[1]}")
    print(f"クラス数: {len(torch.unique(data.y))}")
    
    # 実際の平均次数を計算
    actual_avg_degree = data.edge_index.shape[1] / data.num_nodes
    print(f"実際の平均次数: {actual_avg_degree:.2f}")
    
    
    # マスク情報
    print(f"\nマスク情報:")
    print(f"  訓練ノード数: {data.train_mask.sum().item()}")
    print(f"  検証ノード数: {data.val_mask.sum().item()}")
    print(f"  テストノード数: {data.test_mask.sum().item()}")
    
    return data

if __name__ == "__main__":
    # データセットの作成例（シェルスクリプトから呼び出される場合は、シェルスクリプト側でパラメータを指定）
    print("=== カスタムグラフデータセット作成 ===")
    print("シェルスクリプト create_custom_graoh.sh を使用してパラメータを指定してください")
    print("例: ./create_custom_graoh.sh -n 2000 -d 4.0 -p chain")
    
    # デフォルトの例（シェルスクリプトを使用しない場合）
    dataset = create_custom_dataset(num_nodes=1200, name="CustomGraph", target_avg_degree=3.0)
    data = analyze_dataset(dataset) 