import torch
import torch.nn as nn
import torch.nn.functional as F

class GSLModel_LabelDistr(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_nodes, num_classes, label_embed_dim=16,
                 adj_init=None):  # ← 追加
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # 分類用隣接行列の初期化（元の隣接行列 or ランダム）
        if adj_init is not None:
            assert adj_init.shape == (num_nodes, num_nodes)
            adj_init = torch.clamp(adj_init, min=1e-6, max=1.0)  # 安定化のため
            self.adj_logits = nn.Parameter(torch.logit(adj_init, eps=1e-6).clone())
        else:
            self.adj_logits = nn.Parameter(torch.randn(num_nodes, num_nodes))

        # ラベル埋め込み
        self.label_embed = nn.Parameter(torch.randn(num_classes, label_embed_dim))

    def forward(self, X, onehot_labels, max_hops=4):
        """
        X: 結合された特徴量 [N, pca_dim + neighbor_dim]
        onehot_labels: one-hotラベル [N, num_classes]
        max_hops: 最大hop数
        """
        # Xとonehot_labelsをPyTorchテンソルに変換
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        
        if not isinstance(onehot_labels, torch.Tensor):
            onehot_labels = torch.tensor(onehot_labels, dtype=torch.float32)
        
        # 適切なデバイスに移動
        device = X.device
        onehot_labels = onehot_labels.to(device)

        # 学習後の隣接行列を取得
        A_hat = self.get_learned_adjacency()  # [N, N]
        
        # 正規化（行方向の和で割る）
        D_inv = torch.diag(1.0 / (A_hat.sum(dim=1) + 1e-8))  # 対角行列の逆数
        A_norm = D_inv @ A_hat  # 正規化された隣接行列
        
        # 各hopのラベル分布を計算
        label_dist_list = []
        current_dist = onehot_labels
        
        for hop in range(1, max_hops + 1):
            # n-hopラベル分布: A_norm @ previous_dist
            current_dist = A_norm @ current_dist  # [N, num_classes]
            # 行方向で正規化（softmax）
            current_dist = F.softmax(current_dist, dim=1)
            label_dist_list.append(current_dist)
        
        # 全てのhopのラベル分布を結合
        label_dist_combined = torch.cat(label_dist_list, dim=1)  # [N, max_hops * num_classes]
        
        # 結合された特徴量（PCA + 隣接ノード特徴量）とラベル分布を結合
        combined_features = torch.cat([X, label_dist_combined], dim=1)  # [N, (pca_dim + neighbor_dim) + max_hops*num_classes]
        
        # MLPで最終的な予測
        return self.mlp(combined_features)

    def get_learned_adjacency(self):
        return torch.sigmoid(self.adj_logits)

    def get_label_embedding(self, y_onehot):
        return y_onehot @ self.label_embed


def compute_loss(model, X, y_onehot, train_mask, B,
                 lambda_sparse=0.3, lambda_smooth=0.3, max_hops=4):
    """
    複合損失関数
    - y_onehot: [N, C] one-hotラベル
    - B: [N, N] 同じラベルを持つノード間のみ 1（trainデータのみ）
    - max_hops: 最大hop数
    """
    # Xとy_onehotをPyTorchテンソルに変換
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    
    if not isinstance(y_onehot, torch.Tensor):
        y_onehot = torch.tensor(y_onehot, dtype=torch.float32)
    
    # 適切なデバイスに移動
    device = X.device
    y_onehot = y_onehot.to(device)
    
    # 1. ノード分類（logits）
    logits = model(X, y_onehot, max_hops=max_hops)  # [N, C]

    # 2. クロスエントロピー損失
    ce_loss = F.cross_entropy(logits[train_mask], y_onehot[train_mask].argmax(1))

    # 3. スパース正則化（L1ノルム）- 正規化版
    A_hat = model.get_learned_adjacency()  # [N, N]
    # 対角成分を除外（自己ループは常に1に近いため）
    mask = ~torch.eye(A_hat.shape[0], dtype=torch.bool, device=A_hat.device)
    sparse_loss = torch.sum(torch.abs(A_hat[mask])) / mask.sum()  # 平均値で正規化

    # 4. ラベル埋め込み & 1-hopラベル分布伝播（trainデータのみ）
    # trainデータのラベル埋め込みのみを使用
    y_onehot_train = y_onehot.clone()
    y_onehot_train[~train_mask] = 0  # trainデータ以外は0に設定
    Y_emb = model.get_label_embedding(y_onehot_train)  # [N, D]
    H = A_hat @ Y_emb  # [N, D]

    # 5. 同ラベルノード間のL2ノルム差最小化（trainデータのみ）
    diff = H.unsqueeze(1) - H.unsqueeze(0)  # [N, N, D]
    sq_dist = torch.sum(diff ** 2, dim=2)   # [N, N]
    smooth_loss = torch.sum(B * sq_dist) / B.sum().clamp(min=1)

    # 6. 総合損失
    total_loss = ce_loss + lambda_sparse * sparse_loss + lambda_smooth * smooth_loss

    return total_loss, {
        'ce_loss': ce_loss.item(),
        'sparse_loss': sparse_loss.item(),
        'smooth_loss': smooth_loss.item()
    }
