import torch
import torch.nn as nn
import torch.nn.functional as F

class RGCNLayer(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.mu_linear = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.sigma_linear = nn.Linear(hidden_channels, hidden_channels, bias=False)

    def forward(self, mu, sigma, adj, gamma=1.0):
        # Variance-based attention
        alpha = torch.exp(-gamma * sigma)

        # Aggregation with attention weights
        mu_weighted = mu * alpha
        sigma_weighted = sigma * alpha * alpha

        mu_agg = torch.matmul(adj, mu_weighted)
        sigma_agg = torch.matmul(adj, sigma_weighted)

        mu_out = self.mu_linear(mu_agg)
        sigma_out = self.sigma_linear(sigma_agg)

        return mu_out, sigma_out

class RGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # First layer: convert input features to Gaussian parameters
        self.input_mu = nn.Linear(in_channels, hidden_channels)
        self.input_sigma = nn.Linear(in_channels, hidden_channels)

        # Hidden RGCN layers
        self.layers = nn.ModuleList([
            RGCNLayer(hidden_channels) for _ in range(num_layers - 1)
        ])

        # Output classifier (mean only)
        self.output_layer = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, adj):
        # Step 1: Initial transformation
        mu = F.elu(self.input_mu(x)) # 平均（μ）はノード表現の中心値で正にも負にもなり得るため、活性化関数は符号を制限しないものが適している
        sigma = F.relu(self.input_sigma(x))  # 分散（σ）は常に正の値を持つため、活性化関数は非負の値を出力するものが適している

        # Step 2: Propagate through RGCN layers
        for layer in self.layers:
            mu, sigma = layer(mu, sigma, adj)
            mu = F.elu(mu)
            sigma = F.relu(sigma)
            mu = F.dropout(mu, p=self.dropout, training=self.training)
            sigma = F.dropout(sigma, p=self.dropout, training=self.training)

        # Step 3: Reparameterization trick
        eps = torch.randn_like(sigma) # 平均0・標準偏差1の正規分布からサンプルして sigma と同じ shape の乱数テンソルを作る
        z = mu + eps * torch.sqrt(sigma + 1e-8)  # sample from N(mu, sigma)

        # Step 4: Classification
        out = self.output_layer(z)
        return F.log_softmax(out, dim=1)
