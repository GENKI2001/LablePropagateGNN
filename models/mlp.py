import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP as PyGMLP


class MLP(torch.nn.Module):
    """
    1層のMLPモデル
    グラフ構造を考慮せず、ノード特徴量のみを使用して分類を行う
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.0):
        """
        Args:
            in_channels (int): 入力特徴量の次元
            hidden_channels (int): 隠れ層の次元（このモデルでは使用されない）
            out_channels (int): 出力特徴量の次元
            dropout (float): ドロップアウト率
        """
        super(MLP, self).__init__()
        
        # 1層のMLP: 入力から出力への直接的な線形変換
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.dropout = dropout
    
    def forward(self, x, edge_index=None):
        """
        順伝播
        
        Args:
            x (torch.Tensor): ノード特徴量 [num_nodes, in_channels]
            edge_index (torch.Tensor, optional): エッジインデックス（このモデルでは使用されない）
        
        Returns:
            torch.Tensor: 出力特徴量 [num_nodes, out_channels]
        """
        # グラフ構造は無視して、ノード特徴量のみを使用
        x = self.linear(x)
        
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class MLPWithSkip(torch.nn.Module):
    """
    スキップ接続付きの1層MLPモデル
    入力特徴量と出力特徴量を結合するスキップ接続を含む
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.0):
        """
        Args:
            in_channels (int): 入力特徴量の次元
            hidden_channels (int): 隠れ層の次元（このモデルでは使用されない）
            out_channels (int): 出力特徴量の次元
            dropout (float): ドロップアウト率
        """
        super(MLPWithSkip, self).__init__()
        
        # 1層のMLP: 入力から出力への直接的な線形変換
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.dropout = dropout
        
        # スキップ接続用の線形層（入力と出力の次元が異なる場合）
        if in_channels != out_channels:
            self.skip_linear = torch.nn.Linear(in_channels, out_channels)
        else:
            self.skip_linear = None
    
    def forward(self, x, edge_index=None):
        """
        順伝播
        
        Args:
            x (torch.Tensor): ノード特徴量 [num_nodes, in_channels]
            edge_index (torch.Tensor, optional): エッジインデックス（このモデルでは使用されない）
        
        Returns:
            torch.Tensor: 出力特徴量 [num_nodes, out_channels]
        """
        # メインの線形変換
        out = self.linear(x)
        
        # スキップ接続
        if self.skip_linear is not None:
            skip = self.skip_linear(x)
        else:
            skip = x
        
        # スキップ接続を追加
        out = out + skip
        
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        
        return out 