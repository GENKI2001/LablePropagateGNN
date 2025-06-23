import torch
from .gcn import GCN, GCNWithSkip
from .gat import GAT, GATWithSkip, GATv2
from .mlp import MLP, MLPWithSkip
from .gsl import GraphStructureLearningModel

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
            model_name (str): モデル名 ('GCN', 'GAT', 'GATv2', 'GSL', etc.)
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
        
        elif model_name == 'GCNWithSkip':
            return GCNWithSkip(
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
        
        elif model_name == 'GATWithSkip':
            return GATWithSkip(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=default_params['num_layers'],
                num_heads=default_params['num_heads'],
                dropout=default_params['dropout'],
                concat=default_params['concat']
            )
        
        elif model_name == 'GATv2':
            return GATv2(
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
                dropout=default_params['dropout']
            )
        
        elif model_name == 'MLPWithSkip':
            return MLPWithSkip(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                dropout=default_params['dropout']
            )
        
        elif model_name == 'GSL':
            if default_params['num_nodes'] is None:
                raise ValueError("GSL model requires 'num_nodes' parameter")
            return GraphStructureLearningModel(
                input_dim=in_channels,
                hidden_dim=hidden_channels,
                output_dim=out_channels,
                num_nodes=default_params['num_nodes'],
                num_classes=out_channels,
                label_embed_dim=default_params['label_embed_dim'],
                adj_init=default_params['adj_init']
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
            'GCNWithSkip': {
                'description': 'GCN with Skip Connections',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'num_layers', 'dropout'],
                'default_hidden_channels': 16
            },
            'GAT': {
                'description': 'Graph Attention Network',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'num_layers', 'num_heads', 'dropout', 'concat'],
                'default_hidden_channels': 8
            },
            'GATWithSkip': {
                'description': 'GAT with Skip Connections',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'num_layers', 'num_heads', 'dropout', 'concat'],
                'default_hidden_channels': 8
            },
            'GATv2': {
                'description': 'Improved Graph Attention Network',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'num_layers', 'num_heads', 'dropout', 'concat'],
                'default_hidden_channels': 8
            },
            'MLP': {
                'description': '1-layer Multi-Layer Perceptron (ignores graph structure)',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'dropout'],
                'default_hidden_channels': 16
            },
            'MLPWithSkip': {
                'description': '1-layer MLP with Skip Connections',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'dropout'],
                'default_hidden_channels': 16
            },
            'GSL': {
                'description': 'Graph Structure Learning Model',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'num_nodes', 'label_embed_dim', 'adj_init'],
                'default_hidden_channels': 16
            }
        }
        
        return model_info.get(model_name, {})
    
    @staticmethod
    def get_supported_models():
        """
        サポートされているモデルのリストを取得する
        
        Returns:
            list: サポートされているモデル名のリスト
        """
        return ['GCN', 'GCNWithSkip', 'GAT', 'GATWithSkip', 'GATv2', 'MLP', 'MLPWithSkip', 'GSL'] 