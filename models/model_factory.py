import torch
from torch_geometric.nn import LINKX
from .gcn import GCN, GCNWithSkip
from .gat import GAT, GATWithSkip, GATv2
from .mlp import MLP, MLPWithSkip
from .mlp_and_gcn import MLPAndGCNFusion, MLPAndGCNEnsemble, GCNAndMLPConcat
from .h2gcn import H2GCN
from .mixhop import MixHop, MixHopWithSkip

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
            model_name (str): モデル名 ('GCN', 'GAT', 'GATv2', etc.)
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
                num_layers=default_params['num_layers'],
                dropout=default_params['dropout']
            )
        
        elif model_name == 'MLPWithSkip':
            return MLPWithSkip(
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
       
        # MLP-GCNハイブリッドモデル
        elif model_name == 'MLPAndGCNFusion':
            fusion_method = kwargs.get('fusion_method', 'concat')
            return MLPAndGCNFusion(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=default_params['num_layers'],
                dropout=default_params['dropout'],
                fusion_method=fusion_method
            )
        
        elif model_name == 'MLPAndGCNEnsemble':
            # MLPAndGCNEnsembleの場合は、xfeat_dimとydist_dimを別々に指定する必要がある
            xfeat_dim = kwargs.get('xfeat_dim', in_channels)
            ydist_dim = kwargs.get('ydist_dim', 0)
            ensemble_method = kwargs.get('ensemble_method', 'weighted')
            
            return MLPAndGCNEnsemble(
                xfeat_dim=xfeat_dim,
                ydist_dim=ydist_dim,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=default_params['num_layers'],
                dropout=default_params['dropout']
            )
        
        elif model_name == 'GCNAndMLPConcat':
            # GCNAndMLPConcatの場合は、xfeat_dimとxlabel_dimを別々に指定する必要がある
            xfeat_dim = kwargs.get('xfeat_dim', in_channels)
            xlabel_dim = kwargs.get('xlabel_dim', 0)
            gcn_hidden_dim = kwargs.get('gcn_hidden_dim', None)
            mlp_hidden_dim = kwargs.get('mlp_hidden_dim', None)
            
            return GCNAndMLPConcat(
                xfeat_dim=xfeat_dim,
                xlabel_dim=xlabel_dim,
                hidden_dim=hidden_channels,
                out_dim=out_channels,
                dropout=default_params['dropout'],
                gcn_hidden_dim=gcn_hidden_dim,
                mlp_hidden_dim=mlp_hidden_dim
            )
        
        elif model_name == 'H2GCN':
            return H2GCN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=default_params['num_layers'],
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
        
        elif model_name == 'MixHopWithSkip':
            powers = kwargs.get('powers', [0, 1, 2])
            return MixHopWithSkip(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=default_params['num_layers'],
                dropout=default_params['dropout'],
                powers=powers
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
                'description': 'Multi-Layer Perceptron (ignores graph structure)',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'num_layers', 'dropout'],
                'default_hidden_channels': 16
            },
            'MLPWithSkip': {
                'description': 'Multi-Layer Perceptron with Skip Connections',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'num_layers', 'dropout'],
                'default_hidden_channels': 16
            },
            'LINKX': {
                'description': 'LINKX model',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'dropout'],
                'default_hidden_channels': 16
            },
            'MLPAndGCNFusion': {
                'description': 'MLP-GCN Fusion Model',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'num_layers', 'dropout', 'fusion_method'],
                'default_hidden_channels': 16
            },
            'MLPAndGCNEnsemble': {
                'description': 'MLP-GCN Ensemble Model',
                'parameters': ['xfeat_dim', 'ydist_dim', 'hidden_channels', 'out_channels', 'num_layers', 'dropout'],
                'default_hidden_channels': 16
            },
            'GCNAndMLPConcat': {
                'description': 'GCN-MLP Concat Model (GCN for raw features, MLP for raw+label features)',
                'parameters': ['xfeat_dim', 'xlabel_dim', 'hidden_channels', 'out_channels', 'dropout', 'gcn_hidden_dim', 'mlp_hidden_dim'],
                'default_hidden_channels': 16
            },
            'H2GCN': {
                'description': 'H2GCN Model (uses 1-hop and 2-hop adjacency matrices)',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'num_layers', 'dropout'],
                'default_hidden_channels': 16
            },
            'MixHop': {
                'description': 'MixHop Model (mixes different powers of adjacency matrix)',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'num_layers', 'dropout', 'powers'],
                'default_hidden_channels': 16
            },
            'MixHopWithSkip': {
                'description': 'MixHop Model with Skip Connections',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'num_layers', 'dropout', 'powers'],
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
        return ['GCN', 'GCNWithSkip', 'GAT', 'GATWithSkip', 'GATv2', 'MLP', 'MLPWithSkip', 'LINKX', 'MLPAndGCNFusion', 'MLPAndGCNEnsemble', 'GCNAndMLPConcat', 'H2GCN', 'MixHop', 'MixHopWithSkip'] 