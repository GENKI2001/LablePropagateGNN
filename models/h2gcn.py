import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    def forward(self, adj, x):
        return torch.sparse.mm(adj, x)


class H2GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        # Step 1: Initial ego feature embedding
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=False)

        # Step 2: Shared sparse propagation layers (no weights, no nonlinearity)
        self.gcn_1hop = GCNLayer()
        self.gcn_2hop = GCNLayer()

        # Step 3: Output classifier (concat: h0 + h1_1 + h2_1 + ... + h1_K + h2_K)
        total_concat_dim = (2 * num_layers + 1) * hidden_channels
        self.fc_out = nn.Linear(total_concat_dim, out_channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj1, adj2):
        # Initial embedding
        h0 = self.fc1(x)  # shape: [N, H]

        outputs = [h0]  # list of [h0, h1_1, h2_1, ..., h1_K, h2_K]
        h_prev = h0

        for _ in range(self.num_layers):
            h1 = self.gcn_1hop(adj1, h_prev)  # 1-hop
            h2 = self.gcn_2hop(adj2, h_prev)  # 2-hop
            outputs.extend([h1, h2])
            h_prev = h1  # ← 次のステップの入力には h1 を使う（論文でもこの形）

        # Concatenate all
        h = torch.cat(outputs, dim=1)
        h = self.dropout(h)

        return self.fc_out(h)
