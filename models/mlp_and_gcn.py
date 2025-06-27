import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .mlp import MLP

class MLPAndGCNFusion(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.0, fusion_method='concat_alpha'):
        """
        MLPとGCNを並列に実行し、出力を結合するハイブリッドモデル

        Args:
            in_channels (int): 入力特徴量の次元
            hidden_channels (int): 隠れ層の次元
            out_channels (int): 出力特徴量の次元
            num_layers (int): レイヤー数（デフォルト: 2）
            dropout (float): ドロップアウト率（デフォルト: 0.0）
            fusion_method (str): 融合方法 ('concat', 'add', 'weighted', 'concat_alpha')
        """
        super(MLPAndGCNFusion, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.fusion_method = fusion_method
        
        # GCN部分
        self.gcn_convs = nn.ModuleList()
        self.gcn_convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.gcn_convs.append(GCNConv(hidden_channels, hidden_channels))
        if num_layers > 1:
            self.gcn_convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # MLP部分
        self.mlp_layers = nn.ModuleList()
        self.mlp_layers.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.mlp_layers.append(nn.Linear(hidden_channels, hidden_channels))
        if num_layers > 1:
            self.mlp_layers.append(nn.Linear(hidden_channels, hidden_channels))
        
        # 融合層
        if fusion_method == 'concat':
            self.fusion_layer = nn.Linear(hidden_channels * 2, out_channels)
        elif fusion_method == 'add':
            self.fusion_layer = nn.Linear(hidden_channels, out_channels)
        elif fusion_method == 'weighted':
            self.fusion_layer = nn.Linear(hidden_channels * 2, out_channels)
            self.weight_alpha = nn.Parameter(torch.tensor(0.5))
        elif fusion_method == 'concat_alpha':
            # αの学習可能パラメータを追加（β = 1-α）
            self.alpha = nn.Parameter(torch.tensor(0.5))  # GCNの重み
            self.fusion_layer = nn.Linear(hidden_channels * 2, out_channels)
        
    def forward(self, x, edge_index):
        # GCN処理
        gcn_out = x
        for i, conv in enumerate(self.gcn_convs):
            gcn_out = conv(gcn_out, edge_index)
            if i < len(self.gcn_convs) - 1:
                gcn_out = F.relu(gcn_out)
                if self.dropout > 0:
                    gcn_out = F.dropout(gcn_out, p=self.dropout, training=self.training)
        
        # MLP処理
        mlp_out = x
        for i, layer in enumerate(self.mlp_layers):
            mlp_out = layer(mlp_out)
            if i < len(self.mlp_layers) - 1:
                mlp_out = F.relu(mlp_out)
                if self.dropout > 0:
                    mlp_out = F.dropout(mlp_out, p=self.dropout, training=self.training)
        
        # 融合
        if self.fusion_method == 'concat':
            combined = torch.cat([gcn_out, mlp_out], dim=1)
        elif self.fusion_method == 'add':
            combined = gcn_out + mlp_out
        elif self.fusion_method == 'weighted':
            alpha = torch.clamp(self.weight_alpha, 0, 1)
            combined = torch.cat([alpha * gcn_out, (1 - alpha) * mlp_out], dim=1)
        elif self.fusion_method == 'concat_alpha':
            # αと1-αで重み付き結合
            alpha = torch.clamp(self.alpha, 0, 1)
            beta = 1 - alpha
            combined = torch.cat([alpha * gcn_out, beta * mlp_out], dim=1)
        
        out = self.fusion_layer(combined)
        return out
    
    def get_alpha(self):
        """
        αの値を取得する
        
        Returns:
            float: αの値（0-1の範囲）、適用できない場合はNone
        """
        if self.fusion_method == 'weighted':
            return torch.clamp(self.weight_alpha, 0, 1).item()
        elif self.fusion_method == 'concat_alpha':
            return torch.clamp(self.alpha, 0, 1).item()
        return None
    
    def get_beta(self):
        """
        βの値を取得する（concat_alpha融合の場合のみ）
        
        Returns:
            float: βの値（0-1の範囲）、適用できない場合はNone
        """
        if self.fusion_method == 'concat_alpha':
            return torch.clamp(1 - self.alpha, 0, 1).item()
        return None
    
    def get_alpha_info(self):
        """
        αと1-αの詳細情報を取得する
        
        Returns:
            dict: αと1-αの値と特徴量の重み情報
        """
        if self.fusion_method == 'weighted':
            alpha_val = torch.clamp(self.weight_alpha, 0, 1).item()
            return {
                'alpha': alpha_val,
                'beta': None,
                'gcn_weight': alpha_val,
                'mlp_weight': 1 - alpha_val,
                'gcn_name': 'GCN Features',
                'mlp_name': 'MLP Features',
                'fusion_method': self.fusion_method
            }
        elif self.fusion_method == 'concat_alpha':
            alpha_val = torch.clamp(self.alpha, 0, 1).item()
            beta_val = torch.clamp(1 - self.alpha, 0, 1).item()
            return {
                'alpha': alpha_val,
                'beta': beta_val,
                'gcn_weight': alpha_val,
                'mlp_weight': beta_val,
                'gcn_name': 'GCN Features',
                'mlp_name': 'MLP Features',
                'fusion_method': self.fusion_method
            }
        else:
            return {
                'alpha': None,
                'beta': None,
                'fusion_method': self.fusion_method,
                'gcn_name': 'GCN Features',
                'mlp_name': 'MLP Features'
            }
    
    def print_alpha_info(self):
        """
        αと1-αの情報をコンソールに表示する
        """
        info = self.get_alpha_info()
        print(f"=== MLPAndGCNFusion Alpha/(1-α) Information ===")
        print(f"Fusion method: {info['fusion_method']}")
        if info['alpha'] is not None:
            print(f"Alpha value (GCN weight): {info['alpha']:.4f}")
            if info['beta'] is not None:
                print(f"(1-α) value (MLP weight): {info['beta']:.4f}")
            else:
                print(f"MLP weight: {info['mlp_weight']:.4f}")
            print(f"GCN weight ({info['gcn_name']}): {info['gcn_weight']:.4f}")
            print(f"MLP weight ({info['mlp_name']}): {info['mlp_weight']:.4f}")
        else:
            print(f"Alpha/(1-α) not applicable for {info['fusion_method']} fusion")
        print("=" * 50)


class GCNAndMLPConcat(nn.Module):
    def __init__(self, xfeat_dim, xlabel_dim, hidden_dim, out_dim, dropout=0.0, gcn_hidden_dim=None, mlp_hidden_dim=None):
        super().__init__()
        self.dropout = dropout
        
        # GCNとMLPの隠れ層次元を設定（指定されていない場合は共通のhidden_dimを使用）
        self.gcn_hidden_dim = gcn_hidden_dim if gcn_hidden_dim is not None else hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim if mlp_hidden_dim is not None else hidden_dim

        # GCN: 生の特徴量だけに適用
        self.gcn = GCNConv(xfeat_dim, self.gcn_hidden_dim)

        # MLP: 生の特徴量 + ラベル分布を連結して入力
        self.mlp = nn.Sequential(
            nn.Linear(xfeat_dim + xlabel_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        )

        # 最終出力層（GCN出力とMLP出力を連結）
        self.classifier = nn.Linear(self.gcn_hidden_dim + self.mlp_hidden_dim, out_dim)

    def forward(self, xfeat, xlabel, edge_index):
        # === GCN部分 ===
        gcn_out = self.gcn(xfeat, edge_index)
        gcn_out = F.relu(gcn_out)
        if self.dropout > 0:
            gcn_out = F.dropout(gcn_out, p=self.dropout, training=self.training)

        # === MLP部分 ===
        mlp_input = torch.cat([xfeat, xlabel], dim=1)
        mlp_out = self.mlp(mlp_input)

        # === 融合と出力 ===
        combined = torch.cat([gcn_out, mlp_out], dim=1)
        out = self.classifier(combined)
        return out
    
    def get_hidden_dims_info(self):
        """
        GCNとMLPの隠れ層次元情報を取得する
        
        Returns:
            dict: GCNとMLPの隠れ層次元情報
        """
        return {
            'gcn_hidden_dim': self.gcn_hidden_dim,
            'mlp_hidden_dim': self.mlp_hidden_dim,
            'total_hidden_dim': self.gcn_hidden_dim + self.mlp_hidden_dim
        }
    
    def print_hidden_dims_info(self):
        """
        GCNとMLPの隠れ層次元情報をコンソールに表示する
        """
        info = self.get_hidden_dims_info()
        print(f"=== GCNAndMLPConcat Hidden Dimensions Information ===")
        print(f"GCN hidden dimension: {info['gcn_hidden_dim']}")
        print(f"MLP hidden dimension: {info['mlp_hidden_dim']}")
        print(f"Total hidden dimension: {info['total_hidden_dim']}")
        print("=" * 50)


class MLPAndGCNSerial(nn.Module):
    """
    GCNの後にMLPを適用する直列ハイブリッドモデル
    
    Args:
        in_channels (int): 入力特徴量の次元
        hidden_channels (int): 隠れ層の次元
        out_channels (int): 出力特徴量の次元
        gcn_layers (int): GCNのレイヤー数（デフォルト: 2）
        mlp_layers (int): MLPのレイヤー数（デフォルト: 1）
        dropout (float): ドロップアウト率（デフォルト: 0.0）
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, gcn_layers=2, 
                 mlp_layers=1, dropout=0.0):
        super(MLPAndGCNSerial, self).__init__()
        self.gcn_layers = gcn_layers
        self.mlp_layers = mlp_layers
        self.dropout = dropout
        
        # GCN部分
        self.gcn_convs = nn.ModuleList()
        self.gcn_convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(gcn_layers - 2):
            self.gcn_convs.append(GCNConv(hidden_channels, hidden_channels))
        if gcn_layers > 1:
            self.gcn_convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # MLP部分
        self.mlp_layers_list = nn.ModuleList()
        for _ in range(mlp_layers - 1):
            self.mlp_layers_list.append(nn.Linear(hidden_channels, hidden_channels))
        if mlp_layers > 0:
            self.mlp_layers_list.append(nn.Linear(hidden_channels, out_channels))
        
    def forward(self, x, edge_index):
        # GCN処理
        for i, conv in enumerate(self.gcn_convs):
            x = conv(x, edge_index)
            if i < len(self.gcn_convs) - 1:
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        
        # MLP処理
        for i, layer in enumerate(self.mlp_layers_list):
            x = layer(x)
            if i < len(self.mlp_layers_list) - 1:
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class MLPAndGCNEnsemble(nn.Module):
    """
    MLPとGCNを独立に実行し、アンサンブルするモデル
    
    Args:
        in_channels (int): 入力特徴量の次元
        hidden_channels (int): 隠れ層の次元
        out_channels (int): 出力特徴量の次元
        num_layers (int): 各モデルのレイヤー数（デフォルト: 2）
        dropout (float): ドロップアウト率（デフォルト: 0.0）
        ensemble_method (str): アンサンブル方法 ('average', 'weighted', 'voting', 'concat_alpha')
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, 
                 dropout=0.0, ensemble_method='weighted'):
        super(MLPAndGCNEnsemble, self).__init__()
        self.ensemble_method = ensemble_method
        
        # GCNモデル
        self.gcn = GCN(in_channels, hidden_channels, out_channels, num_layers, dropout)
        
        # MLPモデル
        self.mlp = MLP(in_channels, hidden_channels, out_channels, num_layers, dropout)
        
        # 重み付きアンサンブル用の重み
        if ensemble_method == 'weighted':
            self.gcn_weight = nn.Parameter(torch.tensor(0.5))
            self.mlp_weight = nn.Parameter(torch.tensor(0.5))
        elif ensemble_method == 'concat_alpha':
            # αの学習可能パラメータを追加（β = 1-α）
            self.alpha = nn.Parameter(torch.tensor(0.5))  # GCNの重み
        
    def forward(self, x, edge_index):
        # 各モデルの出力を取得
        gcn_out = self.gcn(x, edge_index)
        mlp_out = self.mlp(x, edge_index)
        
        # アンサンブル
        if self.ensemble_method == 'average':
            out = (gcn_out + mlp_out) / 2
        elif self.ensemble_method == 'weighted':
            # 重みを正規化
            total_weight = self.gcn_weight + self.mlp_weight
            gcn_weight_norm = self.gcn_weight / total_weight
            mlp_weight_norm = self.mlp_weight / total_weight
            out = gcn_weight_norm * gcn_out + mlp_weight_norm * mlp_out
        elif self.ensemble_method == 'voting':
            # ソフトマックスを適用してから平均
            gcn_probs = F.softmax(gcn_out, dim=1)
            mlp_probs = F.softmax(mlp_out, dim=1)
            out = (gcn_probs + mlp_probs) / 2
        elif self.ensemble_method == 'concat_alpha':
            # αと1-αで重み付き加算
            alpha = torch.clamp(self.alpha, 0, 1)
            beta = 1 - alpha
            out = alpha * gcn_out + beta * mlp_out
        
        return out
    
    def get_alpha(self):
        """
        αの値を取得する
        
        Returns:
            float: αの値（0-1の範囲）、適用できない場合はNone
        """
        if self.ensemble_method == 'weighted':
            total_weight = self.gcn_weight + self.mlp_weight
            return (self.gcn_weight / total_weight).item()
        elif self.ensemble_method == 'concat_alpha':
            return torch.clamp(self.alpha, 0, 1).item()
        return None
    
    def get_beta(self):
        """
        βの値を取得する（concat_alphaアンサンブルの場合のみ）
        
        Returns:
            float: βの値（0-1の範囲）、適用できない場合はNone
        """
        if self.ensemble_method == 'concat_alpha':
            return torch.clamp(1 - self.alpha, 0, 1).item()
        return None
    
    def get_alpha_info(self):
        """
        αと1-αの詳細情報を取得する
        
        Returns:
            dict: αと1-αの値と特徴量の重み情報
        """
        if self.ensemble_method == 'weighted':
            total_weight = self.gcn_weight + self.mlp_weight
            gcn_weight_norm = (self.gcn_weight / total_weight).item()
            mlp_weight_norm = (self.mlp_weight / total_weight).item()
            return {
                'alpha': gcn_weight_norm,
                'beta': None,
                'gcn_weight': gcn_weight_norm,
                'mlp_weight': mlp_weight_norm,
                'gcn_name': 'GCN Model',
                'mlp_name': 'MLP Model',
                'ensemble_method': self.ensemble_method
            }
        elif self.ensemble_method == 'concat_alpha':
            alpha_val = torch.clamp(self.alpha, 0, 1).item()
            beta_val = torch.clamp(1 - self.alpha, 0, 1).item()
            return {
                'alpha': alpha_val,
                'beta': beta_val,
                'gcn_weight': alpha_val,
                'mlp_weight': beta_val,
                'gcn_name': 'GCN Model',
                'mlp_name': 'MLP Model',
                'ensemble_method': self.ensemble_method
            }
        else:
            return {
                'alpha': None,
                'beta': None,
                'ensemble_method': self.ensemble_method,
                'gcn_name': 'GCN Model',
                'mlp_name': 'MLP Model'
            }
    
    def print_alpha_info(self):
        """
        αと1-αの情報をコンソールに表示する
        """
        info = self.get_alpha_info()
        print(f"=== MLPAndGCNEnsemble Alpha/(1-α) Information ===")
        print(f"Ensemble method: {info['ensemble_method']}")
        if info['alpha'] is not None:
            print(f"Alpha value (GCN weight): {info['alpha']:.4f}")
            if info['beta'] is not None:
                print(f"(1-α) value (MLP weight): {info['beta']:.4f}")
            else:
                print(f"MLP weight: {info['mlp_weight']:.4f}")
            print(f"GCN weight ({info['gcn_name']}): {info['gcn_weight']:.4f}")
            print(f"MLP weight ({info['mlp_name']}): {info['mlp_weight']:.4f}")
        else:
            print(f"Alpha/(1-α) not applicable for {info['ensemble_method']} ensemble")
        print("=" * 50)


# 簡易的なGCNとMLPクラス（アンサンブル用）
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.0):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x
