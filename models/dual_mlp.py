import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import MLP

class DualMLPFusion(nn.Module):
    def __init__(self, in_dim1, in_dim2, hidden_dim, out_dim, num_layers=2, dropout=0.0):
        """
        2つの入力特徴量に対するMLPを構築し、出力を加重平均するモデル。

        Args:
            in_dim1 (int): 特徴1の入力次元（例: x_raw）
            in_dim2 (int): 特徴2の入力次元（例: label_distribution）
            hidden_dim (int): 隠れ層の次元
            out_dim (int): 出力次元（クラス数）
            num_layers (int): MLPのレイヤー数
            dropout (float): ドロップアウト率
        """
        super(DualMLPFusion, self).__init__()

        self.mlp1 = MLP(in_dim1, hidden_dim, out_dim, num_layers, dropout)
        self.mlp2 = MLP(in_dim2, hidden_dim, out_dim, num_layers, dropout)

        # 出力を加重平均するためのスカラー重み（学習可能）
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初期値0.5で平均に近い
        
        # 特徴量の次元を保存
        self.in_dim1 = in_dim1
        self.in_dim2 = in_dim2

    def forward(self, x, edge_index=None):
        """
        標準的なGNNインターフェースに合わせたforwardメソッド
        
        Args:
            x: 結合された特徴量 [num_nodes, in_dim1 + in_dim2]
            edge_index: エッジインデックス（このモデルでは使用されない）
        """
        # 特徴量を分離
        x1 = x[:, :self.in_dim1]  # 生の特徴量
        x2 = x[:, self.in_dim1:]  # ラベル分布特徴量
        
        out1 = self.mlp1(x1)
        out2 = self.mlp2(x2)

        # ソフトマックス前に重み付き加算
        alpha = torch.clamp(self.alpha, 0, 1)  # [0,1]に制限（安全対策）
        out = alpha * out1 + (1 - alpha) * out2
        return out

    def get_alpha(self):
        """
        αの値を取得する
        
        Returns:
            float: αの値（0-1の範囲）
        """
        return torch.clamp(self.alpha, 0, 1).item()
    
    def get_alpha_info(self):
        """
        αの詳細情報を取得する
        
        Returns:
            dict: αの値と特徴量の重み情報
        """
        alpha_val = torch.clamp(self.alpha, 0, 1).item()
        return {
            'alpha': alpha_val,
            'feature1_weight': alpha_val,
            'feature2_weight': 1 - alpha_val,
            'feature1_name': 'Raw Features',
            'feature2_name': 'Label Distribution Features'
        }
    
    def print_alpha_info(self):
        """
        αの情報をコンソールに表示する
        """
        info = self.get_alpha_info()
        print(f"=== DualMLPFusion Alpha Information ===")
        print(f"Alpha value: {info['alpha']:.4f}")
        print(f"Feature 1 weight ({info['feature1_name']}): {info['feature1_weight']:.4f}")
        print(f"Feature 2 weight ({info['feature2_name']}): {info['feature2_weight']:.4f}")
        print("=" * 40)
