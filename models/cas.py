# correct_and_smooth.py
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class CAS(nn.Module):
    """
    Correct and Smooth (CAS) モデル
    ベースモデルの予測結果を後処理で改善する
    """
    def __init__(self, base_model, alpha=0.5, max_iter=50, autoscale=True):
        super().__init__()
        self.base_model = base_model
        self.alpha = alpha
        self.max_iter = max_iter
        self.autoscale = autoscale
    
    def forward(self, x, edge_index=None):
        # ベースモデルで予測
        if edge_index is not None:
            # PyTorch Geometricの標準的な呼び出し方法
            base_pred = self.base_model(x, edge_index)
        else:
            # データオブジェクトとして渡された場合
            base_pred = self.base_model(x)
        return base_pred
    
    def correct_and_smooth(self, data, base_pred):
        """
        CAS後処理を実行
        
        Args:
            data: PyTorch Geometric Data object or tuple (x, edge_index)
            base_pred: ベースモデルの予測結果 (n_nodes x n_classes)
        
        Returns:
            torch.Tensor: 改善された予測結果
        """
        # データをCPUに移動してNumPyに変換
        device = base_pred.device
        base_pred_np = base_pred.detach().cpu().numpy()
        
        # データオブジェクトから必要な情報を取得
        if hasattr(data, 'y'):
            # PyTorch Geometric Data object
            y = data.y.detach().cpu().numpy()
            train_mask = data.train_mask.detach().cpu().numpy()
            edge_index = data.edge_index.detach().cpu().numpy()
            n_nodes = data.num_nodes
        else:
            # タプルの場合（x, edge_index）
            raise ValueError("CAS requires a PyTorch Geometric Data object with y, train_mask, and edge_index")
        
        # ラベルをone-hot形式に変換
        n_classes = base_pred.shape[1]
        Y = np.zeros((len(y), n_classes))
        Y[np.arange(len(y)), y] = 1
        
        # 隣接行列を取得
        adj = sp.coo_matrix((np.ones(edge_index.shape[1]), 
                           (edge_index[0], edge_index[1])), 
                          shape=(n_nodes, n_nodes)).tocsr()
        
        # CAS処理を実行
        Z_final = correct_and_smooth(base_pred_np, Y, train_mask, adj, 
                                   alpha=self.alpha, max_iter=self.max_iter)
        
        # 結果をテンソルに戻す
        return torch.from_numpy(Z_final).float().to(device)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


def label_propagation(S, Y, mask, alpha=0.5, max_iter=50):
    Z = Y.copy()
    for _ in range(max_iter):
        Z = alpha * S.dot(Z) + (1 - alpha) * Y
        Z[mask] = Y[mask]
    return Z


def correct_step(S, Z, Y, train_mask, alpha=0.5, max_iter=50, autoscale=True):
    E = np.zeros_like(Z)
    E[train_mask] = Z[train_mask] - Y[train_mask]

    E_hat = E.copy()
    for _ in range(max_iter):
        E_hat = alpha * S.dot(E_hat) + (1 - alpha) * E

    if autoscale:
        norm1 = np.mean(np.abs(E[train_mask]))
        norm2 = np.mean(np.abs(E_hat))
        if norm2 > 0:
            E_hat *= (norm1 / norm2)
    return Z + E_hat


def correct_and_smooth(Z, Y, train_mask, adj, alpha=0.5, max_iter=50):
    """
    Z: base prediction (n_nodes x n_classes)
    Y: one-hot ground truth labels (n_nodes x n_classes)
    train_mask: boolean array for train nodes
    adj: scipy sparse adjacency matrix (n x n)
    """
    S = normalize_adj(adj)

    # Step 1: Correct
    Z_corr = correct_step(S, Z, Y, train_mask, alpha=alpha, max_iter=max_iter)

    # Step 2: Smooth
    G = Z_corr.copy()
    G[train_mask] = Y[train_mask]
    Z_final = label_propagation(S, G, train_mask, alpha=alpha, max_iter=max_iter)

    return Z_final
