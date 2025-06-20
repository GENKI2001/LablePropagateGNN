import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) モデル
    
    Args:
        in_channels (int): 入力特徴量の次元
        hidden_channels (int): 隠れ層の次元
        out_channels (int): 出力特徴量の次元
        num_layers (int): レイヤー数（デフォルト: 2）
        dropout (float): ドロップアウト率（デフォルト: 0.0）
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # レイヤーの作成
        self.convs = torch.nn.ModuleList()
        
        # 最初のレイヤー
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # 中間レイヤー
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # 最後のレイヤー
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
    
    def forward(self, x, edge_index):
        """
        順伝播
        
        Args:
            x (torch.Tensor): ノード特徴量 [num_nodes, in_channels]
            edge_index (torch.Tensor): エッジインデックス [2, num_edges]
        
        Returns:
            torch.Tensor: 出力特徴量 [num_nodes, out_channels]
        """
        # 各レイヤーを順次適用
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            # 最後のレイヤー以外でReLUとドロップアウトを適用
            if i < len(self.convs) - 1:
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

class GCNWithSkip(torch.nn.Module):
    """
    Skip connection付きのGraph Convolutional Network (GCN) モデル
    
    Args:
        in_channels (int): 入力特徴量の次元
        hidden_channels (int): 隠れ層の次元
        out_channels (int): 出力特徴量の次元
        num_layers (int): レイヤー数（デフォルト: 3）
        dropout (float): ドロップアウト率（デフォルト: 0.0）
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # レイヤーの作成
        self.convs = torch.nn.ModuleList()
        
        # 最初のレイヤー
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # 中間レイヤー
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # 最後のレイヤー
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
    
    def forward(self, x, edge_index):
        """
        順伝播（Skip connection付き）
        
        Args:
            x (torch.Tensor): ノード特徴量 [num_nodes, in_channels]
            edge_index (torch.Tensor): エッジインデックス [2, num_edges]
        
        Returns:
            torch.Tensor: 出力特徴量 [num_nodes, out_channels]
        """
        # 最初の特徴量を保存（Skip connection用）
        x_skip = x
        
        # 各レイヤーを順次適用
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            # 最後のレイヤー以外でReLUとドロップアウトを適用
            if i < len(self.convs) - 1:
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Skip connection（入力と出力の次元が同じ場合のみ）
        if x.shape[1] == x_skip.shape[1]:
            x = x + x_skip
        
        return x 