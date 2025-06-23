import torch
import torch.nn as nn
import torch.nn.functional as F


class GSLModel_LabelDistr(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_nodes, num_classes, label_embed_dim=16,
                 adj_init=None):
        super().__init__()

        # ノード分類用 MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # 学習対象の隣接行列（A_hat の logits）
        if adj_init is not None:
            assert adj_init.shape == (num_nodes, num_nodes)
            adj_init = torch.clamp(adj_init, min=1e-6, max=1.0)
            self.adj_logits = nn.Parameter(torch.logit(adj_init, eps=1e-6).clone())
        else:
            self.adj_logits = nn.Parameter(torch.randn(num_nodes, num_nodes))

        # ラベル埋め込み（※今は使っていないが保持可能）
        self.label_embed = nn.Parameter(torch.randn(num_classes, label_embed_dim))

    def get_learned_adjacency(self):
        # GATスタイル softmax attention（各ノードごとに重み正規化）
        return F.softmax(self.adj_logits, dim=1)  # [N, N]

    def forward(self, X, onehot_labels, max_hops=4):
        """
        X: 入力特徴量 [N, F]
        onehot_labels: one-hot ラベル [N, C]
        max_hops: ラベル分布の伝播ステップ数
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(onehot_labels, torch.Tensor):
            onehot_labels = torch.tensor(onehot_labels, dtype=torch.float32)

        device = X.device
        onehot_labels = onehot_labels.to(device)
        A_hat = self.get_learned_adjacency()

        # ラベル分布の伝播
        label_dist_list = []
        current_dist = onehot_labels

        for _ in range(max_hops):
            current_dist = A_hat @ current_dist
            current_dist = F.softmax(current_dist, dim=1)
            label_dist_list.append(current_dist)

        label_dist_combined = torch.cat(label_dist_list, dim=1)
        combined_features = torch.cat([X, label_dist_combined], dim=1)
        return self.mlp(combined_features)


def compute_loss(model, X, y_onehot, train_mask, B,
                 lambda_sparse=0.0, lambda_smooth=0.3, max_hops=4):
    """
    - X: 入力特徴量 [N, F]
    - y_onehot: one-hot ラベル [N, C]
    - train_mask: 訓練ノードを示す bool ベクトル [N]
    - B: 同じラベルを持つノードペア (i,j) に 1 の行列 [N, N]
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y_onehot, torch.Tensor):
        y_onehot = torch.tensor(y_onehot, dtype=torch.float32)

    device = X.device
    y_onehot = y_onehot.to(device)

    # 1. 分類予測
    logits = model(X, y_onehot, max_hops=max_hops)
    ce_loss = F.cross_entropy(logits[train_mask], y_onehot[train_mask].argmax(dim=1))

    # 2. スパース正則化（softmaxなので効果は薄いが残せる）
    sparse_loss = torch.tensor(0.0, device=device)

    # 3. ラベル分布のスムージング正則化
    A_hat = model.get_learned_adjacency()
    y_onehot_train = y_onehot.clone()
    y_onehot_train[~train_mask] = 0.0  # ラベルリーク防止

    H = A_hat @ y_onehot_train  # [N, C]
    H = F.normalize(H, p=1, dim=1)  # 分布として扱うため正規化

    diff = H.unsqueeze(1) - H.unsqueeze(0)  # [N, N, C]
    sq_dist = torch.sum(diff ** 2, dim=2)   # [N, N]
    num_same_label_pairs = B.sum().clamp(min=1)
    smooth_loss = torch.sum(B * sq_dist) / num_same_label_pairs

    # 4. 総合損失
    total_loss = ce_loss + lambda_sparse * sparse_loss + lambda_smooth * smooth_loss

    return total_loss, {
        'ce_loss': ce_loss.item(),
        'sparse_loss': sparse_loss.item(),
        'smooth_loss': smooth_loss.item()
    }
