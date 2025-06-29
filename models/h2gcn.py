import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def forward(self, adj, x):
        return torch.sparse.mm(adj, x)


class H2GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, num_layers=1):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.gcn_1hop = GCNLayer()
        self.gcn_2hop = GCNLayer()
        self.fc_out = nn.Linear(3 * hidden_channels, out_channels)

    def forward(self, x, adj1, adj2):
        # Step 1: Ego embedding (non-linear)
        h0 = self.fc1(x)
        h0 = self.dropout(h0)

        # Step 2: Neighborhood aggregation (no nonlinear, no weight)
        h1 = self.gcn_1hop(adj1, h0)  # 1-hop neighbors (no self-loops)
        h2 = self.gcn_2hop(adj2, h0)  # 2-hop neighbors (no self-loops)

        # Step 3: Combine all representations
        h = torch.cat([h0, h1, h2], dim=1)
        h = self.dropout(h)

        # Step 4: Classification
        return self.fc_out(h)
