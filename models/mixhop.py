import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MixHopConv


class MixHop(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, powers=[0, 1, 2]):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.powers = powers
        self.num_powers = len(powers)

        self.convs = nn.ModuleList()

        # 最初のレイヤー
        self.convs.append(MixHopConv(in_channels, hidden_channels, powers=powers))
        input_dim = hidden_channels * self.num_powers  # 最初の出力次元

        # 中間レイヤー
        for _ in range(num_layers - 2):
            self.convs.append(MixHopConv(input_dim, hidden_channels, powers=powers))
            input_dim = hidden_channels * self.num_powers  # 出力次元が更新される

        # 最後のレイヤー（出力次元に注意）
        if num_layers > 1:
            self.convs.append(MixHopConv(input_dim, out_channels, powers=powers))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x
