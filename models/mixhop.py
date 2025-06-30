import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MixHopConv


class MixHop(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, powers=[0, 1, 2]):
        """
        MixHopモデル
        
        Args:
            in_channels (int): 入力特徴量の次元
            hidden_channels (int): 隠れ層の次元
            out_channels (int): 出力特徴量の次元
            num_layers (int): レイヤー数（デフォルト: 2）
            dropout (float): ドロップアウト率（デフォルト: 0.5）
            powers (list): 隣接行列のべき乗のリスト（デフォルト: [0, 1, 2]）
        """
        super(MixHop, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.powers = powers
        
        # MixHopレイヤー
        self.convs = nn.ModuleList()
        
        # 最初のレイヤー
        self.convs.append(MixHopConv(in_channels, hidden_channels, powers=powers))
        
        # 中間レイヤー
        for _ in range(num_layers - 2):
            self.convs.append(MixHopConv(hidden_channels, hidden_channels, powers=powers))
        
        # 最後のレイヤー（出力層）
        if num_layers > 1:
            self.convs.append(MixHopConv(hidden_channels, out_channels, powers=powers))
    
    def forward(self, x, edge_index):
        """
        順伝播
        
        Args:
            x (torch.Tensor): ノード特徴量 [N, in_channels]
            edge_index (torch.Tensor): エッジインデックス [2, E]
            
        Returns:
            torch.Tensor: 出力特徴量 [N, out_channels]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # 最後のレイヤー以外
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
