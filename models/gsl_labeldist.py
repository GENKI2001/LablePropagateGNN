import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GSLModel_LabelDistr(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_nodes, num_classes, label_embed_dim=16,
                 adj_init=None, model_type='mlp', num_layers=2, dropout=0.0):
        super().__init__()
        
        self.model_type = model_type
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # モデルタイプに応じて分類器を選択
        if model_type == 'mlp':
            # ノード分類用 MLP（入力次元は後で調整）
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        elif model_type == 'gcn':
            # GCN分類器（入力次元は後で調整）
            self.classifier = GCNClassifier(
                in_channels=input_dim,
                hidden_channels=hidden_dim,
                out_channels=output_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

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
    
    def get_edge_index_from_adjacency(self, A_hat, threshold=0.1):
        """
        学習された隣接行列からエッジインデックスを生成
        
        Args:
            A_hat: 学習された隣接行列 [N, N]
            threshold: エッジを保持する閾値
        
        Returns:
            edge_index: エッジインデックス [2, num_edges]
        """
        # 閾値以上の重みを持つエッジのみを保持
        mask = A_hat > threshold
        # 自己ループを除外（オプション）
        mask.fill_diagonal_(False)
        
        # エッジインデックスを生成
        edge_index = torch.nonzero(mask, as_tuple=False).t()
        
        # エッジが存在しない場合は自己ループのみを追加
        if edge_index.size(1) == 0:
            edge_index = torch.arange(self.num_nodes, device=A_hat.device)
            edge_index = edge_index.repeat(2, 1)
        
        return edge_index

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
        
        # モデルタイプに応じて分類を実行
        if self.model_type == 'mlp':
            # 入力次元を動的に調整
            if combined_features.shape[1] != self.input_dim:
                # 新しい入力次元に合わせて最初のレイヤーを再作成
                self.classifier[0] = nn.Linear(combined_features.shape[1], self.hidden_dim).to(combined_features.device)
            return self.classifier(combined_features)
        elif self.model_type == 'gcn':
            # 入力次元を動的に調整
            if combined_features.shape[1] != self.input_dim:
                # 新しい入力次元に合わせて最初のレイヤーを再作成
                self.classifier.convs[0] = GCNConv(combined_features.shape[1], self.hidden_dim).to(combined_features.device)
            # GCNの場合はエッジインデックスを生成
            edge_index = self.get_edge_index_from_adjacency(A_hat)
            return self.classifier(combined_features, edge_index)


class GCNClassifier(nn.Module):
    """
    GCN分類器
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # レイヤーの作成
        self.convs = nn.ModuleList()
        
        # 最初のレイヤー
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # 中間レイヤー
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # 最後のレイヤー
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
    
    def forward(self, x, edge_index):
        """
        順伝播
        
        Args:
            x (torch.Tensor): ノード特徴量 [num_nodes, in_channels]
            edge_index (torch.Tensor): エッジインデックス [2, num_edges]
        
        Returns:
            torch.Tensor: 出力特徴量 [num_nodes, out_channels]
        """
        # 各レイヤーを順次適用
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            # 最後のレイヤー以外でReLUとドロップアウトを適用
            if i < len(self.convs) - 1:
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

def compute_loss(model, X, y_onehot, train_mask, B,
                 lambda_sparse=0.0, lambda_smooth=0.3,
                 lambda_feat_smooth=0.1, max_hops=4):
    """
    - X: 入力特徴量 [N, F]
    - y_onehot: one-hot ラベル [N, C]
    - train_mask: bool [N]
    - lambda_feat_smooth: 特徴量の滑らかさの正則化強度
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y_onehot, torch.Tensor):
        y_onehot = torch.tensor(y_onehot, dtype=torch.float32)

    device = X.device
    y_onehot = y_onehot.to(device)
    X = X.to(device)

    # 1. 分類予測
    logits = model(X, y_onehot, max_hops=max_hops)
    ce_loss = F.cross_entropy(logits[train_mask], y_onehot[train_mask].argmax(dim=1))

    # 2. スパース正則化（任意）
    # sparse_loss = torch.tensor(0.0, device=device)
    
    # スパース損失を有効にする場合（オプション）
    A_hat = model.get_learned_adjacency()
    # L1正則化によるスパース化
    sparse_loss = torch.mean(torch.abs(A_hat))
    # または、エントロピー正則化によるスパース化
    entropy = -torch.sum(A_hat * torch.log(A_hat + 1e-8), dim=1)
    sparse_loss = torch.mean(entropy)

    # 3. ラベルスムージング損失
    A_hat = model.get_learned_adjacency()
    y_onehot_train = y_onehot.clone()
    H = A_hat @ y_onehot_train
    H = F.normalize(H, p=1, dim=1)

    # 与えられた B 行列を使って全ノードの同ラベル距離を測定
    diff_label = H.unsqueeze(1) - H.unsqueeze(0)  # [N, N, C]
    sq_dist_label = torch.sum(diff_label ** 2, dim=2)  # [N, N]
    num_same_label_pairs = B.sum().clamp(min=1)
    smooth_loss = torch.sum(B * sq_dist_label) / num_same_label_pairs

    # 4. 特徴量スムージング損失
    diff_feat = X.unsqueeze(1) - X.unsqueeze(0)  # [N, N, F]
    sq_dist_feat = torch.sum(diff_feat ** 2, dim=2)  # [N, N]
    feat_smooth_loss = torch.sum(A_hat * sq_dist_feat) / A_hat.sum().clamp(min=1)

    # 5. 総合損失
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