import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, aggr="mean"):
        """
        GraphSAGE モデル
        
        Args:
            in_channels (int): 入力特徴量の次元
            hidden_channels (int): 隠れ層の次元
            out_channels (int): 出力次元（クラス数など）
            num_layers (int): レイヤー数（デフォルト: 2）
            dropout (float): ドロップアウト率（デフォルト: 0.5）
            aggr (str): 集約関数（"mean", "max", "lstm" など）
        """
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()

        # 最初のレイヤー
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))

        # 中間レイヤー（あれば）
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))

        # 出力層
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x
