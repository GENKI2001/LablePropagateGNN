import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    """
    Graph Attention Network (GAT) モデル
    
    Args:
        in_channels (int): 入力特徴量の次元
        hidden_channels (int): 隠れ層の次元
        out_channels (int): 出力特徴量の次元
        num_layers (int): レイヤー数（デフォルト: 2）
        num_heads (int): アテンションヘッド数（デフォルト: 8）
        dropout (float): ドロップアウト率（デフォルト: 0.0）
        concat (bool): アテンションヘッドの出力を結合するかどうか（デフォルト: True）
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, 
                 num_heads=8, dropout=0.0, concat=True):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat = concat
        
        # レイヤーの作成
        self.convs = torch.nn.ModuleList()
        
        # 最初のレイヤー
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, 
                                 dropout=dropout, concat=concat))
        
        # 中間レイヤー
        for _ in range(num_layers - 2):
            in_dim = hidden_channels * num_heads if concat else hidden_channels
            self.convs.append(GATConv(in_dim, hidden_channels, heads=num_heads, 
                                     dropout=dropout, concat=concat))
        
        # 最後のレイヤー
        if num_layers > 1:
            in_dim = hidden_channels * num_heads if concat else hidden_channels
            self.convs.append(GATConv(in_dim, out_channels, heads=1, 
                                     dropout=dropout, concat=False))
    
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