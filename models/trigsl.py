import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TriFeatureGSLGNN(nn.Module):
    def __init__(self, input_dim_pca, hidden_dim, output_dim,
                    num_nodes, num_classes,
                    adj_init=None, num_layers=2, dropout=0.5,
                    damping_alpha=0.3, adj_init_strength=0.9, combined_dim=None, model_type=None):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_dim_pca = input_dim_pca
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.damping_alpha = nn.Parameter(torch.tensor(damping_alpha, dtype=torch.float32))
        self.adj_init_strength = adj_init_strength

        struct_dim = 1  # 必要に応じて上書き
        label_dist_dim = 4 * num_classes

        # 特徴ごとの MLP
        self.pca_mlp = nn.Sequential(
            nn.Linear(input_dim_pca, hidden_dim),
            nn.ReLU()
        )
        self.struct_mlp = nn.Sequential(
            nn.Linear(struct_dim, hidden_dim),
            nn.ReLU()
        )
        self.label_mlp = nn.Sequential(
            nn.Linear(label_dist_dim, hidden_dim),
            nn.ReLU()
        )

        self.combined_dim = 3 * hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.gcn = GCNClassifier(self.combined_dim, hidden_dim, output_dim, num_layers, dropout)

        # 構造学習初期化
        if adj_init is not None:
            adj_prob = adj_init * self.adj_init_strength + (1 - self.adj_init_strength) / 2
            adj_prob = torch.clamp(adj_prob, min=1e-6, max=1.0 - 1e-6)
            logits_init = torch.logit(adj_prob, eps=1e-6)
            self.adj_logits = nn.Parameter(logits_init.clone())
        else:
            rand_logits = torch.randn(num_nodes, num_nodes) * 0.1
            self.adj_logits = nn.Parameter((rand_logits + rand_logits.T) / 2)

        # 出力統合重み
        self.mlp_weight = nn.Parameter(torch.tensor(1.0))
        self.gcn_weight = nn.Parameter(torch.tensor(1.0))

    def get_learned_adjacency(self):
        logits_sym = (self.adj_logits + self.adj_logits.T) / 2
        return torch.sigmoid(logits_sym)

    def get_edge_index_from_adjacency(self, A_hat, threshold=0.1):
        mask = A_hat > threshold
        mask.fill_diagonal_(False)
        edge_index = torch.nonzero(mask, as_tuple=False).t()
        if edge_index.size(1) == 0:
            edge_index = torch.arange(self.num_nodes, device=A_hat.device).repeat(2, 1)
        return edge_index

    def forward(self, X_pca, X_struct, y_onehot, max_hops=4):
        device = X_pca.device
        if not isinstance(y_onehot, torch.Tensor):
            y_onehot = torch.tensor(y_onehot, dtype=torch.float32, device=device)
        else:
            y_onehot = y_onehot.to(device)

        A_hat = self.get_learned_adjacency()

        # ラベル分布伝播
        label_dist_list = []
        current_dist = y_onehot
        alpha = torch.sigmoid(self.damping_alpha)
        for _ in range(max_hops):
            current_dist = alpha * (A_hat @ current_dist) + (1 - alpha) * y_onehot
            current_dist = F.softmax(current_dist, dim=1)
            label_dist_list.append(current_dist)
        label_dist_combined = torch.cat(label_dist_list, dim=1)

        # 特徴ごとにMLP通す
        pca_embed = self.pca_mlp(X_pca)
        struct_embed = self.struct_mlp(X_struct)
        label_embed = self.label_mlp(label_dist_combined)

        combined_features = torch.cat([pca_embed, struct_embed, label_embed], dim=1)

        edge_index = self.get_edge_index_from_adjacency(A_hat)

        # MLPとGCNを統合
        out_mlp = self.mlp(combined_features)
        out_gcn = self.gcn(combined_features, edge_index)

        weights = F.softmax(torch.stack([self.mlp_weight, self.gcn_weight]), dim=0)
        return weights[0] * out_mlp + weights[1] * out_gcn
        
    def get_model_weights(self):
        """
        学習したGCNとMLPの重み割合を取得する
        
        Returns:
            dict: 各モデルの重み情報
        """
        weights = F.softmax(torch.stack([self.mlp_weight, self.gcn_weight]), dim=0)
        return {
            'mlp_weight': weights[0].item(),
            'gcn_weight': weights[1].item(),
        }

    def get_feature_weights(self):
        """
        学習した特徴量の重みを取得する
        
        Returns:
            dict: 各特徴量の重み情報
        """
        return {
            'pca_weight': self.pca_weight.item(),
            'struct_weight': self.struct_weight.item(),
            'label_weight': self.label_weight.item()
        }

class GCNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.norms.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x

def compute_loss_with_tri_smooth(
    model, X_pca, X_struct, y_onehot, train_mask, B,
    lambda_sparse=0.0, 
    lambda_label_smooth=0.3, 
    lambda_feat_smooth=0.1,
    lambda_struct_smooth=0.1,
    max_hops=4
):
    if not isinstance(X_pca, torch.Tensor):
        X_pca = torch.tensor(X_pca, dtype=torch.float32)
    if not isinstance(X_struct, torch.Tensor):
        X_struct = torch.tensor(X_struct, dtype=torch.float32)
    if not isinstance(y_onehot, torch.Tensor):
        y_onehot = torch.tensor(y_onehot, dtype=torch.float32)

    device = X_pca.device
    X_pca = X_pca.to(device)
    X_struct = X_struct.to(device)
    y_onehot = y_onehot.to(device)

    # --- 通常の forward & クロスエントロピー損失 ---
    logits = model(X_pca, X_struct, y_onehot, max_hops=max_hops)
    ce_loss = F.cross_entropy(logits[train_mask], y_onehot[train_mask].argmax(dim=1))

    # --- 学習された隣接行列 ---
    A_hat = model.get_learned_adjacency()

    # --- スパース性損失（エントロピー最小化） ---
    entropy = -torch.sum(A_hat * torch.log(A_hat + 1e-8), dim=1)
    sparse_loss = torch.mean(entropy)

    # --- ラベルスムージング損失 ---
    alpha = torch.sigmoid(model.damping_alpha)
    Y1 = alpha * (A_hat @ y_onehot) + (1 - alpha) * y_onehot
    Y1 = F.softmax(Y1, dim=1)
    diff_label = Y1.unsqueeze(1) - Y1.unsqueeze(0)
    sq_dist_label = torch.sum(diff_label ** 2, dim=2)
    num_same_label_pairs = B.sum().clamp(min=1)
    label_smooth_loss = torch.sum(B * sq_dist_label) / num_same_label_pairs

    # --- 生の特徴量スムージング ---
    diff_feat = X_pca.unsqueeze(1) - X_pca.unsqueeze(0)
    sq_dist_feat = torch.sum(diff_feat ** 2, dim=2)
    feat_smooth_loss = torch.sum(A_hat * sq_dist_feat) / A_hat.sum().clamp(min=1)

    # --- 構造的特徴量スムージング（構造的に似たノードは似たラベルを持つべき） ---
    diff_struct = X_struct.unsqueeze(1) - X_struct.unsqueeze(0)
    sq_dist_struct = torch.sum(diff_struct ** 2, dim=2)
    struct_smooth_loss = torch.sum(A_hat * sq_dist_struct) / A_hat.sum().clamp(min=1)

    # --- 合計損失 ---
    total_loss = (
        ce_loss
        + lambda_sparse * sparse_loss
        + lambda_label_smooth * label_smooth_loss
        + lambda_feat_smooth * feat_smooth_loss
        + lambda_struct_smooth * struct_smooth_loss
    )

    return total_loss, {
        'ce_loss': ce_loss.item(),
        'sparse_loss': sparse_loss.item(),
        'label_smooth_loss': label_smooth_loss.item(),
        'feat_smooth_loss': feat_smooth_loss.item(),
        'struct_smooth_loss': struct_smooth_loss.item(),
    }