import torch
from torch_geometric.nn import LINKX
from .gcn import GCN
from .gat import GAT
from .mlp import MLP
from .h2gcn import H2GCN
from .mixhop import MixHop
from .graphsage import GraphSAGE
from .robust_h2gcn import RobustH2GCN

class ModelFactory:
    """
    モデルファクトリークラス
    モデル名とパラメータから適切なモデルを作成する
    """
    
    @staticmethod
    def create_model(model_name, in_channels, hidden_channels, out_channels, **kwargs):
        """
        モデルを作成する
        
        Args:
            model_name (str): モデル名 ('GCN', 'GAT', etc.)
            in_channels (int): 入力特徴量の次元
            hidden_channels (int): 隠れ層の次元
            out_channels (int): 出力特徴量の次元
            **kwargs: モデル固有のパラメータ
        
        Returns:
            torch.nn.Module: 作成されたモデル
        """
        
        # デフォルトパラメータ
        default_params = {
            'num_layers': 2,
            'dropout': 0.0,
            'num_heads': 8,
            'concat': True,
            'num_nodes': None,
            'label_embed_dim': 16,
            'adj_init': None
        }
        
        # キーワード引数でデフォルトパラメータを更新
        default_params.update(kwargs)
        
        if model_name == 'GCN':
            return GCN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=default_params['num_layers'],
                dropout=default_params['dropout']
            )
        
        elif model_name == 'GAT':
            return GAT(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=default_params['num_layers'],
                num_heads=default_params['num_heads'],
                dropout=default_params['dropout'],
                concat=default_params['concat']
            )
        
        elif model_name == 'MLP':
            return MLP(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=default_params['num_layers'],
                dropout=default_params['dropout']
            )
        
        elif model_name == 'LINKX':
            conv_type = kwargs.get('conv_type', 'gcn')
            use_batch_norm = kwargs.get('use_batch_norm', True)
            num_nodes = kwargs.get('num_nodes', None)
            
            return LINKX(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=default_params['num_layers'],
                dropout=default_params['dropout'],
                num_nodes=num_nodes
            )
        
        elif model_name == 'H2GCN':
            return H2GCN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=default_params['num_layers'],
                dropout=default_params['dropout']
            )
        
        elif model_name == 'RobustH2GCN':
            # RobustH2GCNは特徴量とラベル特徴量の両方を使用
            in_label_dim = kwargs.get('in_label_dim', in_channels)
            return RobustH2GCN(
                in_feat_dim=in_channels,
                in_label_dim=in_label_dim,
                hidden_dim=hidden_channels,
                out_dim=out_channels,
                dropout=default_params['dropout']
            )
        
        elif model_name == 'MixHop':
            powers = kwargs.get('powers', [0, 1, 2])
            return MixHop(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=default_params['num_layers'],
                dropout=default_params['dropout'],
                powers=powers
            )
        
        elif model_name == 'GraphSAGE':
            aggr = kwargs.get('aggr', 'mean')
            return GraphSAGE(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=default_params['num_layers'],
                dropout=default_params['dropout'],
                aggr=aggr
            )
        

        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    @staticmethod
    def get_model_info(model_name):
        """
        モデルの情報を取得する
        
        Args:
            model_name (str): モデル名
        
        Returns:
            dict: モデルの情報
        """
        model_info = {
            'GCN': {
                'description': 'Graph Convolutional Network',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'num_layers', 'dropout'],
                'default_hidden_channels': 16
            },
            'GAT': {
                'description': 'Graph Attention Network',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'num_layers', 'num_heads', 'dropout', 'concat'],
                'default_hidden_channels': 8
            },
            'MLP': {
                'description': 'Multi-Layer Perceptron (ignores graph structure)',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'num_layers', 'dropout'],
                'default_hidden_channels': 16
            },
            'LINKX': {
                'description': 'LINKX model',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'dropout'],
                'default_hidden_channels': 16
            },
            'H2GCN': {
                'description': 'H2GCN Model (uses 1-hop and 2-hop adjacency matrices)',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'num_layers', 'dropout'],
                'default_hidden_channels': 16
            },
            'RobustH2GCN': {
                'description': 'Robust H2GCN Model (uses feature and label features with gating mechanism)',
                'parameters': ['in_channels', 'in_label_dim', 'hidden_channels', 'out_channels', 'dropout'],
                'default_hidden_channels': 16
            },
            'MixHop': {
                'description': 'MixHop Model (mixes different powers of adjacency matrix)',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'num_layers', 'dropout', 'powers'],
                'default_hidden_channels': 16
            },
            'GraphSAGE': {
                'description': 'GraphSAGE Model (inductive learning on large graphs)',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'num_layers', 'dropout', 'aggr'],
                'default_hidden_channels': 16
            },

        }
        
        return model_info.get(model_name, {})
    
    @staticmethod
    def get_supported_models():
        """
        サポートされているモデルのリストを取得する
        
        Returns:
            list: サポートされているモデル名のリスト
        """
        return ['GCN', 'GAT', 'MLP', 'LINKX', 'H2GCN', 'MixHop', 'GraphSAGE'] 