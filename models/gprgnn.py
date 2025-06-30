import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GPRGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, alpha=0.1, K=10, Init='PPR'):
        """
        GPR-GNN モデル
        
        Args:
            in_channels (int): 入力特徴量の次元
            hidden_channels (int): 隠れ層の次元
            out_channels (int): 出力特徴量の次元
            num_layers (int): 中間MLP層の数（GPRConvは1つ）
            dropout (float): ドロップアウト率
            alpha (float): 初期のPageRank係数
            K (int): 伝播ステップ数（= GPRConv の hop 数）
            Init (str): 重みの初期化方法（'PPR', 'SGC', 'NPPR', 'Random', 'WS' など）
        """
        super().__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.K = K
        self.Init = Init
        
        # MLP部分（前処理）
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        for _ in range(num_layers - 2):
            self.mlp.append(nn.Linear(hidden_channels, hidden_channels))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(dropout))
        
        self.mlp.append(nn.Linear(hidden_channels, out_channels))
        
        # GPR重みの初期化
        self.gamma = nn.Parameter(torch.Tensor(K + 1))
        self._init_weights()
    
    def _init_weights(self):
        """GPR重みの初期化"""
        if self.Init == 'PPR':
            # Personalized PageRank
            self.gamma.data.fill_(self.alpha)
            self.gamma.data[0] = 1.0
        elif self.Init == 'SGC':
            # Simple Graph Convolution
            self.gamma.data.fill_(0.0)
            self.gamma.data[1] = 1.0
        elif self.Init == 'NPPR':
            # Normalized Personalized PageRank
            self.gamma.data.fill_(self.alpha)
            self.gamma.data[0] = 1.0
        elif self.Init == 'Random':
            # Random initialization
            nn.init.uniform_(self.gamma.data)
        elif self.Init == 'WS':
            # Warm Start
            self.gamma.data.fill_(1.0 / (self.K + 1))
        else:
            # Default to PPR
            self.gamma.data.fill_(self.alpha)
            self.gamma.data[0] = 1.0
    
    def forward(self, x, edge_index):
        # MLPで特徴量を変換
        x = self.mlp(x)
        
        # GPR伝播
        out = self.gamma[0] * x
        
        # K-hop伝播
        x_k = x
        for k in range(1, self.K + 1):
            # グラフ畳み込み（1-hop）
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_k = self._propagate(x_k, edge_index)
            out += self.gamma[k] * x_k
        
        return out
    
    def _propagate(self, x, edge_index):
        """グラフ伝播（簡易版）"""
        # 隣接行列の正規化
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # 正規化された伝播
        return F.relu(deg_inv_sqrt[row].unsqueeze(-1) * x[col] * deg_inv_sqrt[col].unsqueeze(-1)) 