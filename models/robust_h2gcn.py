import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    def forward(self, adj, x):
        return torch.sparse.mm(adj, x)


class H2GCNBranch(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.5, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Step 1: Initial ego feature embedding
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        
        # Step 2: Shared sparse propagation layers (no weights, no nonlinearity)
        self.gcn_1hop = GCNLayer()
        self.gcn_2hop = GCNLayer()
        
        # Step 3: Output dimension calculation (concat: h0 + h1_1 + h2_1 + ... + h1_K + h2_K)
        self.output_dim = (2 * num_layers + 1) * hidden_dim
        
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
        return h


class RobustH2GCN(nn.Module):
    def __init__(self, in_feat_dim, in_label_dim, hidden_dim, out_dim, dropout=0.5, num_layers=1):
        super().__init__()
        # 2つのH2GCN分岐
        self.h2gcn_feat = H2GCNBranch(in_feat_dim, hidden_dim, dropout, num_layers)
        self.h2gcn_label = H2GCNBranch(in_label_dim, hidden_dim, dropout, num_layers)

        # ゲート生成ネットワーク
        gate_input_dim = self.h2gcn_feat.output_dim + self.h2gcn_label.output_dim
        self.gate_fc = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 出力分類器
        self.fc_out = nn.Linear(self.h2gcn_feat.output_dim, out_dim)

    def forward(self, x_feat, x_label, adj1, adj2):
        h_feat = self.h2gcn_feat(x_feat, adj1, adj2)     # [N, (2*num_layers+1)*H]
        h_label = self.h2gcn_label(x_label, adj1, adj2)   # [N, (2*num_layers+1)*H]

        # ゲート計算: 入力は h_feat || h_label
        gate_input = torch.cat([h_feat, h_label], dim=1)  # [N, 2*(2*num_layers+1)*H]
        gate = self.gate_fc(gate_input)                  # [N, 1]
        gate = gate.expand_as(h_feat)                    # [N, (2*num_layers+1)*H]

        # ゲート融合
        h_fused = gate * h_feat + (1 - gate) * h_label   # [N, (2*num_layers+1)*H]

        # 出力
        out = self.fc_out(h_fused)  # [N, C]
        return out, gate
