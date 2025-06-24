import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, LINKX

class GSLModel_LabelDistr(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_nodes, num_classes, label_embed_dim=16,
                 adj_init=None, model_type='mlp', num_layers=2, dropout=0.5,
                 damping_alpha=0.5):
        super().__init__()

        self.model_type = model_type
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.damping_alpha = nn.Parameter(torch.tensor(damping_alpha, dtype=torch.float32))

        if model_type == 'mlp':
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        elif model_type == 'gcn':
            self.classifier = GCNClassifier(
                in_channels=input_dim,
                hidden_channels=hidden_dim,
                out_channels=output_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        elif model_type == 'linkx':
            self.classifier = LINKX(
                in_channels=input_dim,
                hidden_channels=hidden_dim,
                out_channels=output_dim,
                num_layers=num_layers,
                dropout=dropout,
                num_nodes=num_nodes
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        if adj_init is not None:
            assert adj_init.shape == (num_nodes, num_nodes)
            adj_init = torch.clamp(adj_init, min=1e-6, max=1.0)
            logits_init = torch.logit(adj_init, eps=1e-6)
            self.adj_logits = nn.Parameter(logits_init.clone())
        else:
            rand_logits = torch.randn(num_nodes, num_nodes)
            self.adj_logits = nn.Parameter((rand_logits + rand_logits.T) / 2)

        self.label_embed = nn.Parameter(torch.randn(num_classes, label_embed_dim))

    def get_learned_adjacency(self):
        logits_sym = (self.adj_logits + self.adj_logits.T) / 2
        return F.softmax(logits_sym, dim=1)

    def get_edge_index_from_adjacency(self, A_hat, threshold=0.1):
        mask = A_hat > threshold
        mask.fill_diagonal_(False)
        edge_index = torch.nonzero(mask, as_tuple=False).t()
        if edge_index.size(1) == 0:
            edge_index = torch.arange(self.num_nodes, device=A_hat.device).repeat(2, 1)
        return edge_index

    def forward(self, X, onehot_labels, max_hops=4):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(onehot_labels, torch.Tensor):
            onehot_labels = torch.tensor(onehot_labels, dtype=torch.float32)

        device = X.device
        onehot_labels = onehot_labels.to(device)
        A_hat = self.get_learned_adjacency()

        label_dist_list = []
        current_dist = onehot_labels
        alpha = torch.sigmoid(self.damping_alpha)
        for _ in range(max_hops):
            current_dist = alpha * (A_hat @ current_dist) + (1 - alpha) * onehot_labels
            current_dist = F.softmax(current_dist, dim=1)
            label_dist_list.append(current_dist)

        label_dist_combined = torch.cat(label_dist_list, dim=1)
        combined_features = torch.cat([X, label_dist_combined], dim=1)

        if self.model_type == 'mlp':
            if combined_features.shape[1] != self.input_dim:
                self.classifier[0] = nn.Linear(combined_features.shape[1], self.hidden_dim).to(combined_features.device)
            return self.classifier(combined_features)

        elif self.model_type == 'gcn':
            if combined_features.shape[1] != self.input_dim:
                self.classifier.reset_input_dim(combined_features.shape[1])
            edge_index = self.get_edge_index_from_adjacency(A_hat)
            return self.classifier(combined_features, edge_index)

        elif self.model_type == 'linkx':
            edge_index = self.get_edge_index_from_adjacency(A_hat)
            return self.classifier(combined_features, edge_index)

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

    def reset_input_dim(self, new_input_dim):
        self.convs[0] = GCNConv(new_input_dim, self.convs[0].out_channels)
        self.norms[0] = nn.BatchNorm1d(self.convs[0].out_channels)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x

def compute_loss(model, X, y_onehot, train_mask, B,
                 lambda_sparse=0.0, lambda_smooth=0.3,
                 lambda_feat_smooth=0.1, max_hops=4):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y_onehot, torch.Tensor):
        y_onehot = torch.tensor(y_onehot, dtype=torch.float32)

    device = X.device
    y_onehot = y_onehot.to(device)
    X = X.to(device)

    logits = model(X, y_onehot, max_hops=max_hops)
    ce_loss = F.cross_entropy(logits[train_mask], y_onehot[train_mask].argmax(dim=1))

    A_hat = model.get_learned_adjacency()
    entropy = -torch.sum(A_hat * torch.log(A_hat + 1e-8), dim=1)
    sparse_loss = torch.mean(entropy)

    y_onehot_train = y_onehot.clone()
    H = A_hat @ y_onehot_train

    diff_label = H.unsqueeze(1) - H.unsqueeze(0)
    sq_dist_label = torch.sum(diff_label ** 2, dim=2)
    num_same_label_pairs = B.sum().clamp(min=1)
    smooth_loss = torch.sum(B * sq_dist_label) / num_same_label_pairs

    diff_feat = X.unsqueeze(1) - X.unsqueeze(0)
    sq_dist_feat = torch.sum(diff_feat ** 2, dim=2)
    feat_smooth_loss = torch.sum(A_hat * sq_dist_feat) / A_hat.sum().clamp(min=1)

    total_loss = (
        ce_loss
        + lambda_sparse * sparse_loss
        + lambda_smooth * smooth_loss
        + lambda_feat_smooth * feat_smooth_loss
    )

    return total_loss, {
        'ce_loss': ce_loss.item(),
        'sparse_loss': sparse_loss.item(),
        'smooth_loss': smooth_loss.item(),
        'feat_smooth_loss': feat_smooth_loss.item()
    }
