import torch
from torch_geometric.nn import LINKX
from .gcn import GCN, GCNWithSkip
from .gat import GAT, GATWithSkip, GATv2
from .mlp import MLP, MLPWithSkip
from .dual_mlp import DualMLPFusion
from .gsl_labeldist import GSLModel_LabelDistr
from .trigsl import TriFeatureGSLGNN

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
        
        elif model_name == 'DualMLPFusion':
            # DualMLPFusionの場合は、in_dim1とin_dim2を別々に指定する必要がある
            in_dim1 = kwargs.get('in_dim1', in_channels)
            in_dim2 = kwargs.get('in_dim2', in_channels)
            
            return DualMLPFusion(
                in_dim1=in_dim1,
                in_dim2=in_dim2,
                hidden_dim=hidden_channels,
                out_dim=out_channels,
                num_layers=default_params['num_layers'],
                dropout=default_params['dropout']
            )
        
        elif model_name == 'GSL':
            if default_params['num_nodes'] is None:
                raise ValueError("GSL model requires 'num_nodes' parameter")
            
            # GSLモデルの追加パラメータ
            model_type = kwargs.get('model_type', 'mlp')
            num_layers = kwargs.get('num_layers', 2)
            dropout = kwargs.get('dropout', 0.0)
            damping_alpha = kwargs.get('damping_alpha', 0.8)
            
            return GSLModel_LabelDistr(
                input_dim=in_channels,
                hidden_dim=hidden_channels,
                output_dim=out_channels,
                num_nodes=default_params['num_nodes'],
                num_classes=out_channels,
                label_embed_dim=default_params['label_embed_dim'],
                adj_init=default_params['adj_init'],
                model_type=model_type,
                num_layers=num_layers,
                dropout=dropout,
                damping_alpha=damping_alpha
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
        
        elif model_name == 'TriFeatureGSLGNN':
            if default_params['num_nodes'] is None:
                raise ValueError("TriFeatureGSLGNN model requires 'num_nodes' parameter")
            
            # 結合次元が指定されていない場合は計算
            combined_dim = kwargs.get('combined_dim', None)
            input_dim_struct = kwargs.get('input_dim_struct', 1)  # デフォルトは1
            if combined_dim is None:
                label_dist_dim = 4 * out_channels  # max_hops=4を仮定
                combined_dim = in_channels + input_dim_struct + label_dist_dim
            
            return TriFeatureGSLGNN(
                input_dim_pca=in_channels,
                input_dim_struct=input_dim_struct,
                hidden_dim=hidden_channels,
                output_dim=out_channels,
                num_nodes=default_params['num_nodes'],
                num_classes=out_channels,
                adj_init=default_params['adj_init'],
                model_type=kwargs.get('model_type', 'hybrid'),
                num_layers=default_params['num_layers'],
                dropout=default_params['dropout'],
                damping_alpha=kwargs.get('damping_alpha', 0.3),
                adj_init_strength=kwargs.get('adj_init_strength', 0.9),
                combined_dim=combined_dim
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
            'DualMLPFusion': {
                'description': 'Dual MLP Fusion Model (combines raw features and label distribution features)',
                'parameters': ['in_dim1', 'in_dim2', 'hidden_channels', 'out_channels', 'num_layers', 'dropout'],
                'default_hidden_channels': 16
            },
            'GSL': {
                'description': 'Graph Structure Learning Model (supports MLP and GCN classifiers)',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'num_nodes', 'label_embed_dim', 'adj_init', 'model_type', 'num_layers', 'dropout'],
                'default_hidden_channels': 16
            },
            'LINKX': {
                'description': 'LINKX model',
                'parameters': ['in_channels', 'hidden_channels', 'out_channels', 'dropout'],
                'default_hidden_channels': 16
            },
            'TriFeatureGSLGNN': {
                'description': 'Triple Feature Graph Structure Learning Model (MLP + GCN + LINKX)',
                'parameters': ['input_dim', 'hidden_dim', 'output_dim', 'num_nodes', 'num_classes', 'label_embed_dim', 'adj_init', 'model_type', 'num_layers', 'dropout', 'damping_alpha', 'adj_init_strength'],
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
        return ['GCN', 'GCNWithSkip', 'GAT', 'GATWithSkip', 'GATv2', 'MLP', 'MLPWithSkip', 'DualMLPFusion', 'GSL', 'LINKX', 'TriFeatureGSLGNN'] 