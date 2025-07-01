import torch
import torch.nn.functional as F
import numpy as np
from utils.dataset_loader import load_dataset, get_supported_datasets
from utils.feature_creator import create_pca_features, create_label_features, display_node_features, get_feature_info, create_similarity_based_edges, create_similarity_based_edges_with_original
from utils.adjacency_creator import create_normalized_adjacency_matrices, get_adjacency_matrix, apply_adjacency_to_features, combine_hop_features, print_adjacency_info, make_undirected
from utils.feature_noise import add_feature_noise, add_feature_noise_uniform, add_feature_noise_random, add_feature_missingness, apply_feature_modifications, print_noise_info, print_modification_info
from models import ModelFactory

# ============================================================================
# ハイパーパラメータなどの設定
# ============================================================================

# データセット選択
# Planetoid: 'Cora', 'Citeseer', 'Pubmed'
# WebKB: 'Cornell', 'Texas', 'Wisconsin'
# WikipediaNetwork: 'Chameleon', 'Squirrel'
# Actor: 'Actor'
DATASET_NAME = 'Cornell'  # ここを変更してデータセットを切り替え

# サポートされているモデル:
# - 'MLP', 'GCN', 'GAT', 'H2GCN', 'RobustH2GCN', 'MixHop', 'GraphSAGE'
MODEL_NAME = 'RobustH2GCN'

# 実験設定
NUM_RUNS = 10  # 実験回数（テスト用に減らす）
NUM_EPOCHS = 600  # エポック数（テスト用に減らす）

# 特徴量作成設定
CALC_NEIGHBOR_LABEL_FEATURES = True  # True: 隣接ノードのラベル特徴量を計算, False: 計算しない
COMBINE_NEIGHBOR_LABEL_FEATURES = False  # True: 元の特徴量にラベル分布ベクトルを結合, False: スキップ
DISABLE_ORIGINAL_FEATURES = False  # True: 元のノード特徴量を無効化（data.xを空にする）

# Grid Search対象パラメータの設定
GRID_SEARCH_PARAMS = {
    'HIDDEN_CHANNELS': [8, 32, 64],  # 隠れ層次元
    'NUM_LAYERS': [1, 2],                   # レイヤー数
    'MAX_HOPS': [1, 2, 3, 4],     # 最大hop数
    'TEMPERATURE': [0.5, 2.5],          # 温度パラメータ
    'DROPOUT': [0.5]          # ドロップアウト率
}

# 単一パラメータのGrid Search設定（削除予定）
# 新しい複数パラメータGrid Searchを使用してください

# 特徴量改変設定（統合版）
USE_FEATURE_MODIFICATION = False  # True: 特徴量を改変, False: スキップ
FEATURE_MODIFICATIONS = [
    # {'type': 'noise', 'percentage': 0.4, 'method': 'per_node'},  # ノイズ追加（0と1を入れ替え）
    # {'type': 'missingness', 'percentage': 0.3},  # 欠損追加（0にマスキング）
]

# 類似度ベースエッジ作成設定
USE_SIMILARITY_BASED_EDGES = False  # True: 類似度ベースエッジ作成を実行, False: スキップ
SIMILARITY_EDGE_MODE = 'add'  # 'replace': 元のエッジを置き換え, 'add': 元のエッジに追加
SIMILARITY_FEATURE_TYPE = 'raw'  # 'raw': 生の特徴量のみ, 'label': ラベル分布特徴量のみ
SIMILARITY_RAW_THRESHOLD = 0.165  # 生の特徴量の類似度閾値 (0.0-1.0)
SIMILARITY_LABEL_THRESHOLD = 0.9999997  # ラベル分布特徴量の類似度閾値 (0.0-1.0)

# MixHopモデル固有の設定
MIXHOP_POWERS = [0, 1, 2]  # 隣接行列のべき乗のリスト [0, 1, 2] または [0, 1, 2, 3] など

# GraphSAGEモデル固有の設定
GRAPHSAGE_AGGR = 'mean'  # 集約関数 ('mean', 'max', 'lstm')

# GATモデル固有の設定
GAT_NUM_HEADS = 8  # アテンションヘッド数
GAT_CONCAT = True  # アテンションヘッドの出力を結合するかどうか

# PCA設定
USE_PCA = False  # True: PCA圧縮, False: 生の特徴量
PCA_COMPONENTS = 128  # PCAで圧縮する次元数結合後の特徴量の形状:

# データ分割設定
TRAIN_RATIO = 0.6  # 訓練データの割合
VAL_RATIO = 0.2    # 検証データの割合
TEST_RATIO = 0.2   # テストデータの割合

# 最適化設定
LEARNING_RATE = 0.01  # 学習率
WEIGHT_DECAY = 5e-4   # 重み減衰

# Early Stopping設定
USE_EARLY_STOPPING = True  # True: Early stoppingを使用, False: 使用しない
EARLY_STOPPING_PATIENCE = 50  # 何エポック改善がなければ停止するか
EARLY_STOPPING_MIN_DELTA = 0.001  # 改善とみなす最小変化量

# 表示設定
DISPLAY_PROGRESS_EVERY = 100  # 何エポックごとに進捗を表示するか
SHOW_FEATURE_DETAILS = False  # 特徴量の詳細を表示するか

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# メイン処理
# ============================================================================

# データセット読み込み
data, dataset = load_dataset(DATASET_NAME, device)

# エッジを無向グラフに修正
data = make_undirected(data, device)

# 元の特徴量を無効化する場合
if DISABLE_ORIGINAL_FEATURES:
    print(f"\n=== 元のノード特徴量を無効化 ===")
    print(f"元の特徴量形状: {data.x.shape}")
    # 空の特徴量テンソルを作成（1次元のゼロベクトル）
    data.x = torch.zeros(data.num_nodes, 0, device=device)
    print(f"無効化後の特徴量形状: {data.x.shape}")
    print(f"元の特徴量は使用されません。ラベル特徴量とランダムウォーク特徴量のみを使用します。")

# 実験前にPCA処理を実行
if USE_PCA:
    print(f"\n=== 実験前PCA処理 ===")
    data, pca_features, pca = create_pca_features(data, device, pca_components=PCA_COMPONENTS)
else:
    print(f"\n=== PCA処理をスキップ ===")
    print(f"生の特徴量を使用します: {data.x.shape}")

# 実験前に特徴量改変を適用
if USE_FEATURE_MODIFICATION and data.x.shape[1] > 0:
    print(f"\n=== 実験前特徴量改変適用 ===")
    print(f"改変設定数: {len(FEATURE_MODIFICATIONS)}")
    
    data.x, modification_info = apply_feature_modifications(data.x, FEATURE_MODIFICATIONS, device)
    
    # 改変情報を表示
    print_modification_info(modification_info, DATASET_NAME)
    print(f"改変適用後の特徴量形状: {data.x.shape}")
elif USE_FEATURE_MODIFICATION and data.x.shape[1] == 0:
    print(f"\n=== 実験前特徴量改変適用 ===")
    print(f"警告: 特徴量が空のため、改変適用をスキップします")
    modification_info = {
        'original_shape': (0, 0),
        'final_shape': (0, 0),
        'modifications_applied': [],
        'total_modifications': len(FEATURE_MODIFICATIONS)
    }

# 隣接行列を作成
adjacency_matrices = create_normalized_adjacency_matrices(data, device, max_hops=2)

# 特定のhopの隣接行列を取得
adj_1hop = get_adjacency_matrix(adjacency_matrices, 1)
adj_2hop = get_adjacency_matrix(adjacency_matrices, 2)

# 隣接行列をデータオブジェクトに追加
data.adj_1hop = adj_1hop
data.adj_2hop = adj_2hop

# 類似度ベースエッジ生成（dataに保存）
if USE_SIMILARITY_BASED_EDGES and SIMILARITY_FEATURE_TYPE == 'raw':
    print(f"\n=== 類似度ベースエッジ生成（dataに保存） ===")
    print(f"エッジモード: {SIMILARITY_EDGE_MODE}")
    print(f"特徴量タイプ: {SIMILARITY_FEATURE_TYPE}")
    print(f"生の特徴量類似度閾値: {SIMILARITY_RAW_THRESHOLD}")
    
    # 生の特徴量を取得
    if USE_PCA:
        raw_features = data.x[:, :PCA_COMPONENTS]
    else:
        raw_features = data.x[:, :dataset.num_features]
    
    # 置き換え用のエッジを生成
    similarity_edge_index, similarity_adj_matrix, num_similarity_edges = create_similarity_based_edges(
        raw_features, threshold=SIMILARITY_RAW_THRESHOLD, device=device
    )
    data.raw_similarity_edge_index = similarity_edge_index
    data.raw_similarity_adj_matrix = similarity_adj_matrix
    data.raw_num_similarity_edges = num_similarity_edges
    print(f"置き換え用エッジ生成完了: {num_similarity_edges}エッジ")
    
    print(f"類似度ベースエッジ生成完了（dataに保存済み）")

# モデル情報を取得
model_info = ModelFactory.get_model_info(MODEL_NAME)
default_hidden_channels = model_info.get('default_hidden_channels', 32)  # デフォルト値

print(f"\n=== 実験設定 ===")
print(f"データセット: {DATASET_NAME}")
print(f"モデル: {MODEL_NAME}")
print(f"説明: {model_info.get('description', 'N/A')}")
print(f"ノード数: {data.num_nodes}")
print(f"エッジ数: {data.edge_index.shape[1]}")
print(f"クラス数: {dataset.num_classes}")
print(f"実験回数: {NUM_RUNS}")
print(f"エポック数: {NUM_EPOCHS}")
print(f"データ分割: 訓練={TRAIN_RATIO:.1%}, 検証={VAL_RATIO:.1%}, テスト={TEST_RATIO:.1%}")
print(f"PCA圧縮次元数: {PCA_COMPONENTS}")
print(f"PCA使用: {USE_PCA}")
print(f"元の特徴量無効化: {DISABLE_ORIGINAL_FEATURES}")
print(f"隣接ノードラベル特徴量計算: {CALC_NEIGHBOR_LABEL_FEATURES}")
print(f"隣接ノード特徴量結合: {COMBINE_NEIGHBOR_LABEL_FEATURES}")
# Grid Search判定
active_params = {k: v for k, v in GRID_SEARCH_PARAMS.items() if len(v) > 0}
total_combinations = 1
for values in active_params.values():
    total_combinations *= len(values)

if total_combinations > 1:
    print(f"Grid Search実行: はい")
    print(f"Grid Search対象パラメータ: {list(active_params.keys())}")
    for param_name, values in active_params.items():
        print(f"  {param_name}: {values}")
    print(f"総パラメータ組み合わせ数: {total_combinations}")
else:
    print(f"Grid Search実行: いいえ（単一パラメータ実行）")
print(f"特徴量改変使用: {USE_FEATURE_MODIFICATION}")
if USE_FEATURE_MODIFICATION:
    print(f"改変設定数: {len(FEATURE_MODIFICATIONS)}")
    for i, mod in enumerate(FEATURE_MODIFICATIONS):
        mod_type = mod.get('type', 'unknown')
        percentage = mod.get('percentage', 0.0)
        if mod_type == 'noise':
            method = mod.get('method', 'per_node')
            print(f"  改変 {i+1}: ノイズ ({method}) - 割合: {percentage:.1%}")
        elif mod_type == 'missingness':
            print(f"  改変 {i+1}: 欠損 - 割合: {percentage:.1%}")
        else:
            print(f"  改変 {i+1}: {mod_type} - 割合: {percentage:.1%}")
print(f"類似度ベースエッジ作成使用: {USE_SIMILARITY_BASED_EDGES}")
if USE_SIMILARITY_BASED_EDGES:
    print(f"エッジモード: {SIMILARITY_EDGE_MODE}")
    print(f"特徴量タイプ: {SIMILARITY_FEATURE_TYPE}")
    if SIMILARITY_FEATURE_TYPE == 'raw':
        print(f"生の特徴量類似度閾値: {SIMILARITY_RAW_THRESHOLD}")
    elif SIMILARITY_FEATURE_TYPE == 'label':
        print(f"ラベル分布特徴量類似度閾値: {SIMILARITY_LABEL_THRESHOLD}")
if MODEL_NAME == 'H2GCN':
    print(f"H2GCNモデル作成: 1-hopと2-hopの隣接行列を使用してグラフ構造を学習")
    print(f"1-hop隣接行列: {data.adj_1hop.shape}")
    print(f"2-hop隣接行列: {data.adj_2hop.shape}")
elif MODEL_NAME == 'MixHop':
    print(f"MixHopモデル作成: 異なるべき乗の隣接行列を混合してグラフ畳み込み")
    print(f"べき乗リスト: {MIXHOP_POWERS}")
elif MODEL_NAME == 'GraphSAGE':
    print(f"GraphSAGEモデル作成: 帰納的学習による大規模グラフ対応")
    print(f"集約関数: {GRAPHSAGE_AGGR}")
elif MODEL_NAME == 'GAT':
    print(f"GATモデル作成: アテンション機構を使用したグラフ畳み込み")
    print(f"アテンションヘッド数: {GAT_NUM_HEADS}")
    print(f"ヘッド出力結合: {GAT_CONCAT}")

print(f"学習率: {LEARNING_RATE}")
print(f"重み減衰: {WEIGHT_DECAY}")
print(f"Early Stopping使用: {USE_EARLY_STOPPING}")
if USE_EARLY_STOPPING:
    print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print(f"Early Stopping Min Delta: {EARLY_STOPPING_MIN_DELTA}")

# 結果を保存するリスト
all_results = []

# Grid Search実行（複数要素の配列があれば自動実行）
active_params = {k: v for k, v in GRID_SEARCH_PARAMS.items() if len(v) > 0}
total_combinations = 1
for values in active_params.values():
    total_combinations *= len(values)

if total_combinations > 1:
    # 複数パラメータのGrid Search設定を確認
    active_params = {k: v for k, v in GRID_SEARCH_PARAMS.items() if len(v) > 0}
    
    if len(active_params) == 0:
        print(f"\n=== Grid Search設定エラー ===")
        print(f"有効なGrid Search対象パラメータが設定されていません。")
        print(f"GRID_SEARCH_PARAMSのいずれかに値を設定してください。")
        exit(1)
    
    # パラメータの組み合わせを生成
    import itertools
    param_names = list(active_params.keys())
    param_values = list(active_params.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"\n=== 複数パラメータGrid Search開始 ===")
    print(f"対象パラメータ: {list(active_params.keys())}")
    for param_name, values in active_params.items():
        print(f"  {param_name}: {values}")
    print(f"パラメータ組み合わせ数: {len(param_combinations)}")
    print(f"総実験数: {len(param_combinations)} × {NUM_RUNS} = {len(param_combinations) * NUM_RUNS}")
    
    grid_search_results = {}
    
    for i, param_combination in enumerate(param_combinations):
        # パラメータの組み合わせを辞書形式で作成
        param_dict = dict(zip(param_names, param_combination))
        
        print(f"\n{'='*80}")
        print(f"=== 組み合わせ {i+1}/{len(param_combinations)} ===")
        for param_name, param_value in param_dict.items():
            print(f"  {param_name} = {param_value}")
        print(f"{'='*80}")
        
        # パラメータ値を設定
        current_max_hops = param_dict.get('MAX_HOPS', 3)  # デフォルト値
        current_hidden_channels = param_dict.get('HIDDEN_CHANNELS', 32)  # デフォルト値
        current_num_layers = param_dict.get('NUM_LAYERS', 1)  # デフォルト値
        current_temperature = param_dict.get('TEMPERATURE', 0.5)  # デフォルト値
        current_dropout = param_dict.get('DROPOUT', 0.5)  # デフォルト値
        
        # このパラメータ値での実験結果を保存するリスト
        param_results = []
        
        # 実験実行
        for run in range(NUM_RUNS):
            param_info = ", ".join([f"{k}={v}" for k, v in param_dict.items()])
            print(f"\n=== 実験 {run + 1}/{NUM_RUNS} (組み合わせ {i+1}: {param_info}) ===")
            
            # 各実験で独立したデータ分割を作成
            run_data = data.clone()
            
            # ランダムなデータ分割を作成
            num_nodes = run_data.num_nodes
            indices = torch.randperm(num_nodes)
            
            # データ分割サイズを計算
            train_size = int(TRAIN_RATIO * num_nodes)
            val_size = int(VAL_RATIO * num_nodes)
            
            # 新しいマスクを作成
            run_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            run_data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            run_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            run_data.train_mask[indices[:train_size]] = True
            run_data.val_mask[indices[train_size:train_size + val_size]] = True
            run_data.test_mask[indices[train_size + val_size:]] = True
            
            print(f"  データ分割: 訓練={run_data.train_mask.sum().item()}, 検証={run_data.val_mask.sum().item()}, テスト={run_data.test_mask.sum().item()}")
            
            # 実験中にラベル特徴量を作成
            adj_matrix, one_hot_labels, neighbor_label_features = create_label_features(
                run_data, device, max_hops=current_max_hops, calc_neighbor_label_features=CALC_NEIGHBOR_LABEL_FEATURES,
                temperature=current_temperature
            )

            # 隣接ノードのラベル特徴量を結合
            if COMBINE_NEIGHBOR_LABEL_FEATURES and neighbor_label_features is not None:
                print(f"  隣接ノードラベル特徴量を結合: {data.x.shape} + {neighbor_label_features.shape}")
                
                # 通常の結合
                if COMBINE_NEIGHBOR_LABEL_FEATURES:
                    run_data.x = torch.cat([run_data.x, neighbor_label_features], dim=1)
                print(f"  結合後の特徴量形状: {run_data.x.shape}")

            # 生の特徴量類似度ベースエッジを必要に応じて結合
            if USE_SIMILARITY_BASED_EDGES and SIMILARITY_FEATURE_TYPE == 'raw' and hasattr(data, 'raw_similarity_edge_index'):
                print(f"  類似度ベースエッジを結合中...")
                print(f"    エッジモード: {SIMILARITY_EDGE_MODE}")
                
                if SIMILARITY_EDGE_MODE == 'replace':
                    # 元のエッジを類似度ベースエッジで置き換え
                    original_edge_count = run_data.edge_index.shape[1]
                    run_data.edge_index = data.raw_similarity_edge_index.clone()
                    print(f"    元のエッジを類似度ベースエッジで置き換え: {original_edge_count} → {data.raw_num_similarity_edges}")
                    
                elif SIMILARITY_EDGE_MODE == 'add':
                    # 元のエッジに類似度ベースエッジを追加
                    original_edge_count = run_data.edge_index.shape[1]
                    # 元のエッジと類似度ベースエッジを結合
                    combined_edge_index = torch.cat([run_data.edge_index, data.raw_similarity_edge_index], dim=1)
                    # 重複エッジを除去
                    edge_pairs = combined_edge_index.t()
                    unique_edges, _ = torch.unique(edge_pairs, dim=0, return_inverse=True)
                    run_data.edge_index = unique_edges.t()
                    final_edge_count = run_data.edge_index.shape[1]
                    print(f"    元のエッジに類似度ベースエッジを追加: {original_edge_count} + {data.raw_num_similarity_edges} → {final_edge_count}")
                
                print(f"  エッジ結合完了: 最終エッジ数 {run_data.edge_index.shape[1]}")

            # ラベル分布特徴量類似度ベースエッジを必要に応じて結合
            if USE_SIMILARITY_BASED_EDGES and SIMILARITY_FEATURE_TYPE == 'label' and neighbor_label_features is not None:
                print(f"  ラベル分布特徴量類似度ベースエッジを結合中...")
                print(f"    エッジモード: {SIMILARITY_EDGE_MODE}")
                print(f"    ラベル分布特徴量類似度閾値: {SIMILARITY_LABEL_THRESHOLD}")
                
                # ラベル分布特徴量で類似度ベースエッジを作成
                if SIMILARITY_EDGE_MODE == 'replace':
                    # 元のエッジをラベル分布特徴量ベースエッジで置き換え
                    original_edge_count = run_data.edge_index.shape[1]
                    label_edge_index, label_adj_matrix, num_label_edges = create_similarity_based_edges(
                        neighbor_label_features, threshold=SIMILARITY_LABEL_THRESHOLD, device=device
                    )
                    run_data.edge_index = label_edge_index
                    print(f"    元のエッジをラベル分布特徴量ベースエッジで置き換え: {original_edge_count} → {num_label_edges}")
                    
                elif SIMILARITY_EDGE_MODE == 'add':
                    # 元のエッジにラベル分布特徴量ベースエッジを追加
                    original_edge_count = run_data.edge_index.shape[1]
                    combined_edge_index, combined_adj_matrix, num_orig, num_new, num_total = create_similarity_based_edges_with_original(
                        run_data.edge_index, neighbor_label_features, 
                        threshold=SIMILARITY_LABEL_THRESHOLD, device=device, combine_with_original=True
                    )
                    run_data.edge_index = combined_edge_index
                    print(f"    元のエッジにラベル分布特徴量ベースエッジを追加: {num_orig} + {num_new} → {num_total}")
                
                print(f"  ラベル分布特徴量エッジ結合完了: 最終エッジ数 {run_data.edge_index.shape[1]}")
            
            elif USE_SIMILARITY_BASED_EDGES and SIMILARITY_FEATURE_TYPE == 'label' and neighbor_label_features is None:
                print(f"  警告: neighbor_label_featuresがNoneのため、ラベル分布特徴量でのエッジ作成をスキップします")
                print(f"    CALC_NEIGHBOR_LABEL_FEATURES=Trueに設定してください")

            # 特徴量情報を取得
            feature_info = get_feature_info(run_data, one_hot_labels, max_hops=current_max_hops)
            
            # 実際の特徴量次元を使用（隣接ノード特徴量が結合されている場合）
            actual_feature_dim = run_data.x.shape[1]
            print(f"  実際の入力特徴量次元: {actual_feature_dim}")
            
            # 特徴量の詳細表示（オプション）
            if SHOW_FEATURE_DETAILS:
                display_node_features(run_data, adj_matrix, one_hot_labels, DATASET_NAME, max_hops=current_max_hops)
            
            # モデル作成
            model_kwargs = {
                'model_name': MODEL_NAME,
                'in_channels': actual_feature_dim,  # 実際の特徴量次元を使用
                'hidden_channels': current_hidden_channels,
                'out_channels': dataset.num_classes,
                'num_layers': current_num_layers,
                'dropout': current_dropout
            }
            
            # MixHopモデルの場合はべき乗パラメータを指定
            if MODEL_NAME in ['MixHop']:
                model_kwargs.update({
                    'powers': MIXHOP_POWERS
                })
                
                print(f"  {MODEL_NAME}モデル作成:")
                print(f"    べき乗リスト: {MIXHOP_POWERS}")
                print(f"    隠れ層次元: {current_hidden_channels}")
                print(f"    レイヤー数: {current_num_layers}")
                print(f"    ドロップアウト: {current_dropout}")
            
            # GraphSAGEモデルの場合は集約関数パラメータを指定
            elif MODEL_NAME == 'GraphSAGE':
                model_kwargs.update({
                    'aggr': GRAPHSAGE_AGGR
                })
                
                print(f"  GraphSAGEモデル作成:")
                print(f"    集約関数: {GRAPHSAGE_AGGR}")
                print(f"    隠れ層次元: {current_hidden_channels}")
                print(f"    レイヤー数: {current_num_layers}")
                print(f"    ドロップアウト: {current_dropout}")
            
            # GATモデルの場合はアテンションヘッドパラメータを指定
            elif MODEL_NAME == 'GAT':
                model_kwargs.update({
                    'num_heads': GAT_NUM_HEADS,
                    'concat': GAT_CONCAT
                })
                
                print(f"  GATモデル作成:")
                print(f"    アテンションヘッド数: {GAT_NUM_HEADS}")
                print(f"    ヘッド出力結合: {GAT_CONCAT}")
                print(f"    隠れ層次元: {current_hidden_channels}")
                print(f"    レイヤー数: {current_num_layers}")
                print(f"    ドロップアウト: {current_dropout}")
            
            
            # RobustH2GCNモデルの場合はパラメータを指定
            elif MODEL_NAME == 'RobustH2GCN':
                # ラベル特徴量の次元を取得
                if neighbor_label_features is not None:
                    label_feature_dim = neighbor_label_features.shape[1]
                else:
                    # ラベル特徴量がない場合は、クラス数のone-hotベクトルを使用
                    label_feature_dim = dataset.num_classes
                
                model_kwargs.update({
                    'in_label_dim': label_feature_dim
                })
                
                print(f"  RobustH2GCNモデル作成:")
                print(f"    特徴量次元: {actual_feature_dim}")
                print(f"    ラベル特徴量次元: {label_feature_dim}")
                print(f"    隠れ層次元: {current_hidden_channels}")
                print(f"    ドロップアウト: {current_dropout}")
            
            model = ModelFactory.create_model(**model_kwargs).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            
            # 学習ループ
            def train():
                model.train()
                optimizer.zero_grad()
                
                # H2GCNとRobustH2GCNの場合は特別な処理（1-hopと2-hopの隣接行列を使用）
                if MODEL_NAME == 'H2GCN':
                    out = model(run_data.x, run_data.adj_1hop, run_data.adj_2hop)
                elif MODEL_NAME == 'RobustH2GCN':
                    # RobustH2GCNは特徴量とラベル特徴量の両方を使用
                    if neighbor_label_features is not None:
                        out, gate = model(run_data.x, neighbor_label_features, run_data.adj_1hop, run_data.adj_2hop)
                    else:
                        # ラベル特徴量がない場合は、one-hotラベルを使用
                        print(f"  ラベル特徴量がないため、one-hotラベルを使用")
                        one_hot_labels_tensor = one_hot_labels.to(device)
                        out, gate = model(run_data.x, one_hot_labels_tensor, run_data.adj_1hop, run_data.adj_2hop)
                else:
                    # その他のモデルは標準的な処理
                    out = model(run_data.x, run_data.edge_index)
                
                loss = F.cross_entropy(out[run_data.train_mask], run_data.y[run_data.train_mask])
                loss.backward()
                optimizer.step()
                return loss.item()
            
            # 評価関数
            @torch.no_grad()
            def test():
                model.eval()
                
                # H2GCNとRobustH2GCNの場合は特別な処理（1-hopと2-hopの隣接行列を使用）
                if MODEL_NAME == 'H2GCN':
                    out = model(run_data.x, run_data.adj_1hop, run_data.adj_2hop)
                    gate = None
                elif MODEL_NAME == 'RobustH2GCN':
                    # RobustH2GCNは特徴量とラベル特徴量の両方を使用
                    if neighbor_label_features is not None:
                        out, gate = model(run_data.x, neighbor_label_features, run_data.adj_1hop, run_data.adj_2hop)
                    else:
                        # ラベル特徴量がない場合は、one-hotラベルを使用
                        one_hot_labels_tensor = one_hot_labels.to(device)
                        out, gate = model(run_data.x, one_hot_labels_tensor, run_data.adj_1hop, run_data.adj_2hop)
                else:
                    # その他のモデルは標準的な処理
                    out = model(run_data.x, run_data.edge_index)
                    gate = None
                
                pred = out.argmax(dim=1)
                accs = []
                for mask in [run_data.train_mask, run_data.val_mask, run_data.test_mask]:
                    correct = pred[mask] == run_data.y[mask]
                    accs.append(int(correct.sum()) / int(mask.sum()))
                
                if MODEL_NAME == 'RobustH2GCN':
                    return accs[0], accs[1], accs[2], gate
                else:
                    return accs
            

            
            # 学習実行
            best_val_acc = 0
            best_test_acc = 0
            final_train_acc = 0
            final_val_acc = 0
            final_test_acc = 0
            final_gate = None  # RobustH2GCNのgate値を保存
            
            # Early stopping用の変数
            if USE_EARLY_STOPPING:
                best_val_acc_for_early_stopping = 0
                patience_counter = 0
                early_stopped = False
            
            for epoch in range(NUM_EPOCHS + 1):
                loss = train()
                
                # RobustH2GCNの場合はgate値も取得
                if MODEL_NAME == 'RobustH2GCN':
                    train_acc, val_acc, test_acc, gate = test()
                else:
                    train_acc, val_acc, test_acc = test()
                
                # ベスト結果を記録
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                
                # Early stoppingの処理
                if USE_EARLY_STOPPING:
                    if val_acc > best_val_acc_for_early_stopping + EARLY_STOPPING_MIN_DELTA:
                        best_val_acc_for_early_stopping = val_acc
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    # Early stopping条件をチェック
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        early_stopped = True
                        print(f"Early stopping triggered at epoch {epoch} (patience: {EARLY_STOPPING_PATIENCE})")
                        break
                
                # 最終結果を記録
                if epoch == NUM_EPOCHS:
                    final_train_acc = train_acc
                    final_val_acc = val_acc
                    final_test_acc = test_acc
                    if MODEL_NAME == 'RobustH2GCN':
                        final_gate = gate
                
                # 進捗表示
                if epoch % DISPLAY_PROGRESS_EVERY == 0:
                    alpha_info = ""
                    
                    early_stop_info = ""
                    if USE_EARLY_STOPPING:
                        early_stop_info = f", Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
                    
                    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}{alpha_info}{early_stop_info}')
            
            # Early stoppingで終了した場合の最終結果を記録
            if USE_EARLY_STOPPING and early_stopped:
                final_train_acc = train_acc
                final_val_acc = val_acc
                final_test_acc = test_acc
                if MODEL_NAME == 'RobustH2GCN':
                    final_gate = gate
            
            # 結果を保存
            run_result = {
                'run': run + 1,
                'final_train_acc': final_train_acc,
                'final_val_acc': final_val_acc,
                'final_test_acc': final_test_acc,
                'best_val_acc': best_val_acc,
                'best_test_acc': best_test_acc,
                'grid_search_params': param_dict.copy()  # パラメータ情報を保存
            }
            
            # RobustH2GCNのgate値を保存
            if MODEL_NAME == 'RobustH2GCN' and final_gate is not None:
                run_result['final_gate'] = final_gate
            
            # 改変情報を保存（実験前の改変情報を使用）
            if USE_FEATURE_MODIFICATION:
                run_result['modification_info'] = modification_info
            
            # Early stopping情報を保存
            if USE_EARLY_STOPPING:
                run_result['early_stopped'] = early_stopped
                run_result['early_stopping_epoch'] = epoch if early_stopped else NUM_EPOCHS
                run_result['early_stopping_patience'] = EARLY_STOPPING_PATIENCE
                run_result['early_stopping_min_delta'] = EARLY_STOPPING_MIN_DELTA
            
            param_results.append(run_result)
            
            print(f"実験 {run + 1} 完了:")
            print(f"  最終結果 - Train: {final_train_acc:.4f}, Val: {final_val_acc:.4f}, Test: {final_test_acc:.4f}")
            print(f"  ベスト結果 - Val: {best_val_acc:.4f}, Test: {best_test_acc:.4f}")
            
            # RobustH2GCNのgate値を出力
            if MODEL_NAME == 'RobustH2GCN' and final_gate is not None:
                gate_mean = final_gate.mean().item()
                gate_std = final_gate.std().item()
                gate_min = final_gate.min().item()
                gate_max = final_gate.max().item()
                print(f"  Gate統計 - 平均: {gate_mean:.4f}, 標準偏差: {gate_std:.4f}, 範囲: [{gate_min:.4f}, {gate_max:.4f}]")
        
        grid_search_results[param_combination] = param_results
    
    # Grid Search結果の集計と表示
    print(f"\n{'='*80}")
    print(f"=== Grid Search結果サマリー ===")
    print(f"{'='*80}")
    
    # 各パラメータ組み合わせでの結果を集計
    param_summary = {}
    for param_combination, results in grid_search_results.items():
        final_test_accs = [r['final_test_acc'] for r in results]
        best_test_accs = [r['best_test_acc'] for r in results]
        final_val_accs = [r['final_val_acc'] for r in results]
        best_val_accs = [r['best_val_acc'] for r in results]
        
        param_summary[param_combination] = {
            'final_test_mean': np.mean(final_test_accs),
            'final_test_std': np.std(final_test_accs),
            'best_test_mean': np.mean(best_test_accs),
            'best_test_std': np.std(best_test_accs),
            'final_val_mean': np.mean(final_val_accs),
            'final_val_std': np.std(final_val_accs),
            'best_val_mean': np.mean(best_val_accs),
            'best_val_std': np.std(best_val_accs),
            'results': results
        }
    
    # 結果を表形式で表示
    print(f"\nパラメータ組み合わせ別結果:")
    print(f"{'='*120}")
    header = "組み合わせ | "
    for param_name in param_names:
        header += f"{param_name:>8} | "
    header += "{'Final Test':>12} | {'Best Test':>12} | {'Final Val':>12} | {'Best Val':>12}"
    print(header)
    print(f"{'='*120}")
    
    for i, param_combination in enumerate(sorted(grid_search_results.keys())):
        summary = param_summary[param_combination]
        param_dict = dict(zip(param_names, param_combination))
        
        row = f"{i+1:>9} | "
        for param_name in param_names:
            row += f"{param_dict[param_name]:>8} | "
        row += f"{summary['final_test_mean']:>10.4f}±{summary['final_test_std']:<1.4f} | "
        row += f"{summary['best_test_mean']:>10.4f}±{summary['best_test_std']:<1.4f} | "
        row += f"{summary['final_val_mean']:>10.4f}±{summary['final_val_std']:<1.4f} | "
        row += f"{summary['best_val_mean']:>10.4f}±{summary['best_val_std']:<1.4f}"
        print(row)
    
    # 最適なパラメータ組み合わせを特定（検証精度のベスト値で選択）
    best_param_by_val = max(param_summary.items(), key=lambda x: x[1]['best_val_mean'])
    best_param_by_test = max(param_summary.items(), key=lambda x: x[1]['best_test_mean'])
    
    # 最終結果Test精度の平均値で最適パラメータを特定
    best_param_by_final_test = max(param_summary.items(), key=lambda x: x[1]['final_test_mean'])
    
    print(f"\n{'='*80}")
    print(f"最適パラメータ選択結果:")
    print(f"{'='*80}")
    
    # 検証精度ベスト値による最適パラメータ
    best_val_param_dict = dict(zip(param_names, best_param_by_val[0]))
    print(f"検証精度ベスト値による最適パラメータ:")
    for param_name, param_value in best_val_param_dict.items():
        print(f"  {param_name}: {param_value}")
    print(f"  検証精度: {best_param_by_val[1]['best_val_mean']:.4f} ± {best_param_by_val[1]['best_val_std']:.4f}")
    print(f"  テスト精度: {best_param_by_val[1]['best_test_mean']:.4f} ± {best_param_by_val[1]['best_test_std']:.4f}")
    
    # テスト精度ベスト値による最適パラメータ
    best_test_param_dict = dict(zip(param_names, best_param_by_test[0]))
    print(f"\nテスト精度ベスト値による最適パラメータ:")
    for param_name, param_value in best_test_param_dict.items():
        print(f"  {param_name}: {param_value}")
    print(f"  検証精度: {best_param_by_test[1]['best_val_mean']:.4f} ± {best_param_by_test[1]['best_val_std']:.4f}")
    print(f"  テスト精度: {best_param_by_test[1]['best_test_mean']:.4f} ± {best_param_by_test[1]['best_test_std']:.4f}")
    
    # 最終結果Test精度平均値による最適パラメータ
    best_final_test_param_dict = dict(zip(param_names, best_param_by_final_test[0]))
    print(f"\n最終結果Test精度平均値による最適パラメータ:")
    for param_name, param_value in best_final_test_param_dict.items():
        print(f"  {param_name}: {param_value}")
    print(f"  最終結果Test精度: {best_param_by_final_test[1]['final_test_mean']:.4f} ± {best_param_by_final_test[1]['final_test_std']:.4f}")
    
    # 推奨パラメータ（検証精度ベスト値による選択）
    recommended_params = best_val_param_dict
    print(f"\n推奨パラメータ (検証精度ベスト値による選択):")
    for param_name, param_value in recommended_params.items():
        print(f"  {param_name}: {param_value}")
    
    # 最終結果Test精度の最大値を出力
    max_final_test_mean = best_param_by_final_test[1]['final_test_mean']
    print(f"\n最終結果Test精度の最大平均値: {max_final_test_mean:.4f}")
    print(f"最適パラメータ:")
    for param_name, param_value in best_final_test_param_dict.items():
        print(f"  {param_name}: {param_value}")
    
    # 全結果をall_resultsに統合
    all_results = []
    for param_combination, results in grid_search_results.items():
        param_dict = dict(zip(param_names, param_combination))
        for result in results:
            result['grid_search_params'] = param_dict.copy()
        all_results.extend(results)
    
    # 詳細な結果表示
    print(f"\n=== 詳細結果 ===")
    for i, result in enumerate(all_results):
        alpha_info = ""
        
        modification_info = ""
        if USE_FEATURE_MODIFICATION and 'modification_info' in result and result['modification_info'] is not None:
            mod_count = len(result['modification_info'].get('modifications_applied', []))
            modification_info = f", 改変={mod_count}個"
        
        early_stop_info = ""
        if USE_EARLY_STOPPING and result.get('early_stopped', False):
            early_stop_info = f", ES@{result.get('early_stopping_epoch', 'N/A')}"
        
        grid_search_info = ""
        if 'grid_search_params' in result:
            param_str = ", ".join([f"{k}={v}" for k, v in result['grid_search_params'].items()])
            grid_search_info = f", パラメータ: {param_str}"
        
        print(f"実験 {i+1:2d}: Final Test={result['final_test_acc']:.4f}, Best Test={result['best_test_acc']:.4f}{alpha_info}{modification_info}{early_stop_info}{grid_search_info}")
    
    # 結果の集計と表示
    print(f"\n=== 実験結果統計 ({NUM_RUNS}回の平均) ===")

# 単一パラメータ実行（Grid Searchを使用しない場合）
else:
    print(f"\n=== 単一パラメータ実行 ===")
    print(f"Grid Searchを使用しない場合の設定値:")
    print(f"  注意: HIDDEN_CHANNELS, NUM_LAYERS, MAX_HOPS, TEMPERATURE, DROPOUTはGRID_SEARCH_PARAMSで設定してください")
    print(f"  デフォルト値: HIDDEN_CHANNELS=32, NUM_LAYERS=1, MAX_HOPS=3, TEMPERATURE=0.5, DROPOUT=0.5")
    
    # 実験実行
    for run in range(NUM_RUNS):
        print(f"\n=== 実験 {run + 1}/{NUM_RUNS} ===")
        
        # 各実験で独立したデータ分割を作成
        run_data = data.clone()
        
        # ランダムなデータ分割を作成
        num_nodes = run_data.num_nodes
        indices = torch.randperm(num_nodes)
        
        # データ分割サイズを計算
        train_size = int(TRAIN_RATIO * num_nodes)
        val_size = int(VAL_RATIO * num_nodes)
        
        # 新しいマスクを作成
        run_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        run_data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        run_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        run_data.train_mask[indices[:train_size]] = True
        run_data.val_mask[indices[train_size:train_size + val_size]] = True
        run_data.test_mask[indices[train_size + val_size:]] = True
        
        print(f"  データ分割: 訓練={run_data.train_mask.sum().item()}, 検証={run_data.val_mask.sum().item()}, テスト={run_data.test_mask.sum().item()}")
        
        # 実験中にラベル特徴量を作成（デフォルト値を使用）
        adj_matrix, one_hot_labels, neighbor_label_features = create_label_features(
            run_data, device, max_hops=3, calc_neighbor_label_features=CALC_NEIGHBOR_LABEL_FEATURES,
            temperature=0.5
        )

        # 隣接ノードのラベル特徴量を結合
        if COMBINE_NEIGHBOR_LABEL_FEATURES and neighbor_label_features is not None:
            print(f"  隣接ノードラベル特徴量を結合: {data.x.shape} + {neighbor_label_features.shape}")
            
            # 通常の結合
            if COMBINE_NEIGHBOR_LABEL_FEATURES:
                run_data.x = torch.cat([run_data.x, neighbor_label_features], dim=1)
            print(f"  結合後の特徴量形状: {run_data.x.shape}")

        # 生の特徴量類似度ベースエッジを必要に応じて結合
        if USE_SIMILARITY_BASED_EDGES and SIMILARITY_FEATURE_TYPE == 'raw' and hasattr(data, 'raw_similarity_edge_index'):
            print(f"  類似度ベースエッジを結合中...")
            print(f"    エッジモード: {SIMILARITY_EDGE_MODE}")
            
            if SIMILARITY_EDGE_MODE == 'replace':
                # 元のエッジを類似度ベースエッジで置き換え
                original_edge_count = run_data.edge_index.shape[1]
                run_data.edge_index = data.raw_similarity_edge_index.clone()
                print(f"    元のエッジを類似度ベースエッジで置き換え: {original_edge_count} → {data.raw_num_similarity_edges}")
                
            elif SIMILARITY_EDGE_MODE == 'add':
                # 元のエッジに類似度ベースエッジを追加
                original_edge_count = run_data.edge_index.shape[1]
                # 元のエッジと類似度ベースエッジを結合
                combined_edge_index = torch.cat([run_data.edge_index, data.raw_similarity_edge_index], dim=1)
                # 重複エッジを除去
                edge_pairs = combined_edge_index.t()
                unique_edges, _ = torch.unique(edge_pairs, dim=0, return_inverse=True)
                run_data.edge_index = unique_edges.t()
                final_edge_count = run_data.edge_index.shape[1]
                print(f"    元のエッジに類似度ベースエッジを追加: {original_edge_count} + {data.raw_num_similarity_edges} → {final_edge_count}")
                
            print(f"  エッジ結合完了: 最終エッジ数 {run_data.edge_index.shape[1]}")

        # ラベル分布特徴量類似度ベースエッジを必要に応じて結合
        if USE_SIMILARITY_BASED_EDGES and SIMILARITY_FEATURE_TYPE == 'label' and neighbor_label_features is not None:
            print(f"  ラベル分布特徴量類似度ベースエッジを結合中...")
            print(f"    エッジモード: {SIMILARITY_EDGE_MODE}")
            print(f"    ラベル分布特徴量類似度閾値: {SIMILARITY_LABEL_THRESHOLD}")
            
            # ラベル分布特徴量で類似度ベースエッジを作成
            if SIMILARITY_EDGE_MODE == 'replace':
                # 元のエッジをラベル分布特徴量ベースエッジで置き換え
                original_edge_count = run_data.edge_index.shape[1]
                label_edge_index, label_adj_matrix, num_label_edges = create_similarity_based_edges(
                    neighbor_label_features, threshold=SIMILARITY_LABEL_THRESHOLD, device=device
                )
                run_data.edge_index = label_edge_index
                print(f"    元のエッジをラベル分布特徴量ベースエッジで置き換え: {original_edge_count} → {num_label_edges}")
                
            elif SIMILARITY_EDGE_MODE == 'add':
                # 元のエッジにラベル分布特徴量ベースエッジを追加
                original_edge_count = run_data.edge_index.shape[1]
                combined_edge_index, combined_adj_matrix, num_orig, num_new, num_total = create_similarity_based_edges_with_original(
                    run_data.edge_index, neighbor_label_features, 
                    threshold=SIMILARITY_LABEL_THRESHOLD, device=device, combine_with_original=True
                )
                run_data.edge_index = combined_edge_index
                print(f"    元のエッジにラベル分布特徴量ベースエッジを追加: {num_orig} + {num_new} → {num_total}")
                
            print(f"  ラベル分布特徴量エッジ結合完了: 最終エッジ数 {run_data.edge_index.shape[1]}")
        
        elif USE_SIMILARITY_BASED_EDGES and SIMILARITY_FEATURE_TYPE == 'label' and neighbor_label_features is None:
            print(f"  警告: neighbor_label_featuresがNoneのため、ラベル分布特徴量でのエッジ作成をスキップします")
            print(f"    CALC_NEIGHBOR_LABEL_FEATURES=Trueに設定してください")

        # 特徴量情報を取得
        feature_info = get_feature_info(run_data, one_hot_labels, max_hops=3)
        
        # 実際の特徴量次元を使用（隣接ノード特徴量が結合されている場合）
        actual_feature_dim = run_data.x.shape[1]
        print(f"  実際の入力特徴量次元: {actual_feature_dim}")
        
        # 特徴量の詳細表示（オプション）
        if SHOW_FEATURE_DETAILS:
            display_node_features(run_data, adj_matrix, one_hot_labels, DATASET_NAME, max_hops=3)
        
        # モデル作成（デフォルト値を使用）
        model_kwargs = {
            'model_name': MODEL_NAME,
            'in_channels': actual_feature_dim,  # 実際の特徴量次元を使用
            'hidden_channels': 32,  # デフォルト値
            'out_channels': dataset.num_classes,
            'num_layers': 1,  # デフォルト値
            'dropout': 0.5  # デフォルト値
        }
        # MixHopモデルの場合はべき乗パラメータを指定
        if MODEL_NAME in ['MixHop']:
            model_kwargs.update({
                'powers': MIXHOP_POWERS
            })
            
            print(f"  {MODEL_NAME}モデル作成:")
            print(f"    べき乗リスト: {MIXHOP_POWERS}")
            print(f"    隠れ層次元: 32 (デフォルト)")
            print(f"    レイヤー数: 1 (デフォルト)")
            print(f"    ドロップアウト: 0.5 (デフォルト)")
        
        # GraphSAGEモデルの場合は集約関数パラメータを指定
        elif MODEL_NAME == 'GraphSAGE':
            model_kwargs.update({
                'aggr': GRAPHSAGE_AGGR
            })
            
            print(f"  GraphSAGEモデル作成:")
            print(f"    集約関数: {GRAPHSAGE_AGGR}")
            print(f"    隠れ層次元: 32 (デフォルト)")
            print(f"    レイヤー数: 1 (デフォルト)")
            print(f"    ドロップアウト: 0.5 (デフォルト)")
        
        # GATモデルの場合はアテンションヘッドパラメータを指定
        elif MODEL_NAME == 'GAT':
            model_kwargs.update({
                'num_heads': GAT_NUM_HEADS,
                'concat': GAT_CONCAT
            })
            
            print(f"  GATモデル作成:")
            print(f"    アテンションヘッド数: {GAT_NUM_HEADS}")
            print(f"    ヘッド出力結合: {GAT_CONCAT}")
            print(f"    隠れ層次元: 32 (デフォルト)")
            print(f"    レイヤー数: 1 (デフォルト)")
            print(f"    ドロップアウト: 0.5 (デフォルト)")
        

        
        # RobustH2GCNモデルの場合はパラメータを指定
        elif MODEL_NAME == 'RobustH2GCN':
            # ラベル特徴量の次元を取得
            if neighbor_label_features is not None:
                label_feature_dim = neighbor_label_features.shape[1]
            else:
                # ラベル特徴量がない場合は、クラス数のone-hotベクトルを使用
                label_feature_dim = dataset.num_classes
            
            model_kwargs.update({
                'in_label_dim': label_feature_dim
            })
            
            print(f"  RobustH2GCNモデル作成:")
            print(f"    特徴量次元: {actual_feature_dim}")
            print(f"    ラベル特徴量次元: {label_feature_dim}")
            print(f"    隠れ層次元: 32 (デフォルト)")
            print(f"    ドロップアウト: 0.5 (デフォルト)")
        
        model = ModelFactory.create_model(**model_kwargs).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # 学習ループ
        def train():
            model.train()
            optimizer.zero_grad()
            
            # H2GCNとRobustH2GCNの場合は特別な処理（1-hopと2-hopの隣接行列を使用）
            if MODEL_NAME == 'H2GCN':
                out = model(run_data.x, run_data.adj_1hop, run_data.adj_2hop)
            elif MODEL_NAME == 'RobustH2GCN':
                # RobustH2GCNは特徴量とラベル特徴量の両方を使用
                if neighbor_label_features is not None:
                    out, gate = model(run_data.x, neighbor_label_features, run_data.adj_1hop, run_data.adj_2hop)
                else:
                    # ラベル特徴量がない場合は、one-hotラベルを使用
                    one_hot_labels_tensor = one_hot_labels.to(device)
                    out, gate = model(run_data.x, one_hot_labels_tensor, run_data.adj_1hop, run_data.adj_2hop)
            else:
                # その他のモデルは標準的な処理
                out = model(run_data.x, run_data.edge_index)
            
            loss = F.cross_entropy(out[run_data.train_mask], run_data.y[run_data.train_mask])
            loss.backward()
            optimizer.step()
            return loss.item()
        
        # 評価関数
        @torch.no_grad()
        def test():
            model.eval()
            
            # H2GCNとRobustH2GCNの場合は特別な処理（1-hopと2-hopの隣接行列を使用）
            if MODEL_NAME == 'H2GCN':
                out = model(run_data.x, run_data.adj_1hop, run_data.adj_2hop)
                gate = None
            elif MODEL_NAME == 'RobustH2GCN':
                # RobustH2GCNは特徴量とラベル特徴量の両方を使用
                if neighbor_label_features is not None:
                    out, gate = model(run_data.x, neighbor_label_features, run_data.adj_1hop, run_data.adj_2hop)
                else:
                    # ラベル特徴量がない場合は、one-hotラベルを使用
                    one_hot_labels_tensor = one_hot_labels.to(device)
                    out, gate = model(run_data.x, one_hot_labels_tensor, run_data.adj_1hop, run_data.adj_2hop)
            else:
                # その他のモデルは標準的な処理
                out = model(run_data.x, run_data.edge_index)
                gate = None
            
            pred = out.argmax(dim=1)
            accs = []
            for mask in [run_data.train_mask, run_data.val_mask, run_data.test_mask]:
                correct = pred[mask] == run_data.y[mask]
                accs.append(int(correct.sum()) / int(mask.sum()))
            
            if MODEL_NAME == 'RobustH2GCN':
                return accs[0], accs[1], accs[2], gate
            else:
                return accs
        

        
        # 学習実行
        best_val_acc = 0
        best_test_acc = 0
        final_train_acc = 0
        final_val_acc = 0
        final_test_acc = 0
        final_gate = None  # RobustH2GCNのgate値を保存
        
        # Early stopping用の変数
        if USE_EARLY_STOPPING:
            best_val_acc_for_early_stopping = 0
            patience_counter = 0
            early_stopped = False
        
        for epoch in range(NUM_EPOCHS + 1):
            loss = train()
            
            # RobustH2GCNの場合はgate値も取得
            if MODEL_NAME == 'RobustH2GCN':
                train_acc, val_acc, test_acc, gate = test()
            else:
                train_acc, val_acc, test_acc = test()
            
            # ベスト結果を記録
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
            
            # Early stoppingの処理
            if USE_EARLY_STOPPING:
                if val_acc > best_val_acc_for_early_stopping + EARLY_STOPPING_MIN_DELTA:
                    best_val_acc_for_early_stopping = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping条件をチェック
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    early_stopped = True
                    print(f"Early stopping triggered at epoch {epoch} (patience: {EARLY_STOPPING_PATIENCE})")
                    break
            
            # 最終結果を記録
            if epoch == NUM_EPOCHS:
                final_train_acc = train_acc
                final_val_acc = val_acc
                final_test_acc = test_acc
                if MODEL_NAME == 'RobustH2GCN':
                    final_gate = gate
            
            # 進捗表示
            if epoch % DISPLAY_PROGRESS_EVERY == 0:
                alpha_info = ""
                
                early_stop_info = ""
                if USE_EARLY_STOPPING:
                    early_stop_info = f", Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
                
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}{alpha_info}{early_stop_info}')
        
        # Early stoppingで終了した場合の最終結果を記録
        if USE_EARLY_STOPPING and early_stopped:
            final_train_acc = train_acc
            final_val_acc = val_acc
            final_test_acc = test_acc
            if MODEL_NAME == 'RobustH2GCN':
                final_gate = gate
        
        # 結果を保存
        run_result = {
            'run': run + 1,
            'final_train_acc': final_train_acc,
            'final_val_acc': final_val_acc,
            'final_test_acc': final_test_acc,
            'best_val_acc': best_val_acc,
            'best_test_acc': best_test_acc
        }
        
        # RobustH2GCNのgate値を保存
        if MODEL_NAME == 'RobustH2GCN' and final_gate is not None:
            run_result['final_gate'] = final_gate
        
        # 改変情報を保存（実験前の改変情報を使用）
        if USE_FEATURE_MODIFICATION:
            run_result['modification_info'] = modification_info
        
        # Early stopping情報を保存
        if USE_EARLY_STOPPING:
            run_result['early_stopped'] = early_stopped
            run_result['early_stopping_epoch'] = epoch if early_stopped else NUM_EPOCHS
            run_result['early_stopping_patience'] = EARLY_STOPPING_PATIENCE
            run_result['early_stopping_min_delta'] = EARLY_STOPPING_MIN_DELTA
        
        all_results.append(run_result)
        
        print(f"実験 {run + 1} 完了:")
        print(f"  最終結果 - Train: {final_train_acc:.4f}, Val: {final_val_acc:.4f}, Test: {final_test_acc:.4f}")
        print(f"  ベスト結果 - Val: {best_val_acc:.4f}, Test: {best_test_acc:.4f}")
        
        # RobustH2GCNのgate値を出力
        if MODEL_NAME == 'RobustH2GCN' and final_gate is not None:
            gate_mean = final_gate.mean().item()
            gate_std = final_gate.std().item()
            gate_min = final_gate.min().item()
            gate_max = final_gate.max().item()
            print(f"  Gate統計 - 平均: {gate_mean:.4f}, 標準偏差: {gate_std:.4f}, 範囲: [{gate_min:.4f}, {gate_max:.4f}]")
    
    # 詳細な結果表示
    print(f"\n=== 詳細結果 ===")
    for i, result in enumerate(all_results):
        alpha_info = ""
        
        modification_info = ""
        if USE_FEATURE_MODIFICATION and 'modification_info' in result and result['modification_info'] is not None:
            mod_count = len(result['modification_info'].get('modifications_applied', []))
            modification_info = f", 改変={mod_count}個"
        
        early_stop_info = ""
        if USE_EARLY_STOPPING and result.get('early_stopped', False):
            early_stop_info = f", ES@{result.get('early_stopping_epoch', 'N/A')}"
        
        print(f"実験 {i+1:2d}: Final Test={result['final_test_acc']:.4f}, Best Test={result['best_test_acc']:.4f}{alpha_info}{modification_info}{early_stop_info}")
    
    # 結果の集計と表示
    print(f"\n=== 実験結果統計 ({NUM_RUNS}回の平均) ===")

# ============================================================================
# 結果の集計と表示
# ============================================================================

# 結果の統計を計算
print(f"\n=== 実験結果統計 ({NUM_RUNS}回の平均) ===")

# 最終結果の統計
final_train_accs = [r['final_train_acc'] for r in all_results]
final_val_accs = [r['final_val_acc'] for r in all_results]
final_test_accs = [r['final_test_acc'] for r in all_results]

# ベスト結果の統計
best_val_accs = [r['best_val_acc'] for r in all_results]
best_test_accs = [r['best_test_acc'] for r in all_results]

print(f"最終結果:")
print(f"  Train: {np.mean(final_train_accs):.4f} ± {np.std(final_train_accs):.4f}")
print(f"  Val:   {np.mean(final_val_accs):.4f} ± {np.std(final_val_accs):.4f}")
print(f"  Test:  {np.mean(final_test_accs):.4f} ± {np.std(final_test_accs):.4f}")

print(f"\nベスト結果:")
print(f"  Val:   {np.mean(best_val_accs):.4f} ± {np.std(best_val_accs):.4f}")
print(f"  Test:  {np.mean(best_test_accs):.4f} ± {np.std(best_test_accs):.4f}")

# Early stopping統計
if USE_EARLY_STOPPING:
    early_stopped_count = sum(1 for r in all_results if r.get('early_stopped', False))
    early_stopping_epochs = [r.get('early_stopping_epoch', NUM_EPOCHS) for r in all_results]
    print(f"Early Stopping統計:")
    print(f"  早期停止した実験数: {early_stopped_count}/{NUM_RUNS} ({early_stopped_count/NUM_RUNS:.1%})")
    print(f"  平均停止エポック: {np.mean(early_stopping_epochs):.1f} ± {np.std(early_stopping_epochs):.1f}")
    print(f"  停止エポック範囲: [{min(early_stopping_epochs)}, {max(early_stopping_epochs)}]")

# 改変情報の統計表示
if USE_FEATURE_MODIFICATION:
    print(f"改変設定: {len(FEATURE_MODIFICATIONS)}個の改変")
    for i, mod in enumerate(FEATURE_MODIFICATIONS):
        mod_type = mod.get('type', 'unknown')
        percentage = mod.get('percentage', 0.0)
        if mod_type == 'noise':
            method = mod.get('method', 'per_node')
            print(f"  改変 {i+1}: ノイズ ({method}) - 割合: {percentage:.1%}")
        elif mod_type == 'missingness':
            print(f"  改変 {i+1}: 欠損 - 割合: {percentage:.1%}")
        else:
            print(f"  改変 {i+1}: {mod_type} - 割合: {percentage:.1%}")

# RobustH2GCNのgate統計情報を表示
if MODEL_NAME == 'RobustH2GCN':
    gate_results = [r for r in all_results if 'final_gate' in r and r['final_gate'] is not None]
    if gate_results:
        print(f"\nRobustH2GCN Gate統計:")
        gate_means = [r['final_gate'].mean().item() for r in gate_results]
        gate_stds = [r['final_gate'].std().item() for r in gate_results]
        gate_mins = [r['final_gate'].min().item() for r in gate_results]
        gate_maxs = [r['final_gate'].max().item() for r in gate_results]
        
        print(f"  Gate平均値: {np.mean(gate_means):.4f} ± {np.std(gate_means):.4f}")
        print(f"  Gate標準偏差: {np.mean(gate_stds):.4f} ± {np.std(gate_stds):.4f}")
        print(f"  Gate最小値: {np.mean(gate_mins):.4f} ± {np.std(gate_mins):.4f}")
        print(f"  Gate最大値: {np.mean(gate_maxs):.4f} ± {np.std(gate_maxs):.4f}")
        print(f"  Gate値範囲: [{min(gate_mins):.4f}, {max(gate_maxs):.4f}]")


print(f"\n=== 実験完了 ===")
print(f"データセット: {DATASET_NAME}")
print(f"モデル: {MODEL_NAME}")
print(f"最終テスト精度: {np.mean(final_test_accs):.4f} ± {np.std(final_test_accs):.4f}")
print(f"ベストテスト精度: {np.mean(best_test_accs):.4f} ± {np.std(best_test_accs):.4f}")

# Grid Search結果の最終サマリー
if total_combinations > 1 and 'grid_search_params' in all_results[0]:
    best_params = max(all_results, key=lambda x: x['best_val_acc'])['grid_search_params']
    best_params_by_final_test = max(all_results, key=lambda x: x['final_test_acc'])['grid_search_params']
    max_final_test_acc = max(r['final_test_acc'] for r in all_results)
    
    # 最終結果Test精度ベストのパラメータでの詳細統計を計算
    best_final_test_results = [r for r in all_results if r['grid_search_params'] == best_params_by_final_test]
    best_final_test_accs = [r['final_test_acc'] for r in best_final_test_results]
    best_final_test_mean = np.mean(best_final_test_accs)
    best_final_test_std = np.std(best_final_test_accs)
    best_final_test_variance = np.var(best_final_test_accs)
    
    print(f"最適パラメータ (検証精度ベスト):")
    for param_name, param_value in best_params.items():
        print(f"  {param_name}: {param_value}")
    print(f"最適パラメータ (最終結果Test精度ベスト):")
    for param_name, param_value in best_params_by_final_test.items():
        print(f"  {param_name}: {param_value}")
    
    # 最終テスト精度の最大値を達成した組み合わせを特定
    best_single_result = max(all_results, key=lambda x: x['final_test_acc'])
    best_single_params = best_single_result['grid_search_params']
    best_single_acc = best_single_result['final_test_acc']
    
    print(f"\n=== 最終テスト精度最大値達成組み合わせ ===")
    print(f"最終テスト精度: {best_single_acc:.4f}")
    print(f"パラメータ:")
    for param_name, param_value in best_single_params.items():
        print(f"  {param_name}: {param_value}")
    print(f"実験回数: {best_single_result['run']}")
    
    # この組み合わせでの全実験結果を取得して統計を計算
    best_combination_results = [r for r in all_results if r['grid_search_params'] == best_single_params]
    best_combination_accs = [r['final_test_acc'] for r in best_combination_results]
    
    if len(best_combination_results) > 1:
        best_combination_mean = np.mean(best_combination_accs)
        best_combination_std = np.std(best_combination_accs)
        best_combination_var = np.var(best_combination_accs)
        
        print(f"この組み合わせでの統計:")
        print(f"  平均値: {best_combination_mean:.4f}")
        print(f"  標準偏差: {best_combination_std:.4f}")
        print(f"  分散: {best_combination_var:.6f}")
        print(f"  範囲: [{min(best_combination_accs):.4f}, {max(best_combination_accs):.4f}]")
        print(f"  実験回数: {len(best_combination_results)}回")
    else:
        print(f"この組み合わせでの実験回数: {len(best_combination_results)}回")
    
    print(f"\n=== 最適パラメータの詳細統計 ===")
    if len(best_final_test_accs) > 1:
        print(f"最終結果Test精度:")
        print(f"  平均値: {best_final_test_mean:.4f}")
        print(f"  標準偏差: {best_final_test_std:.4f}")
        print(f"  分散: {best_final_test_variance:.6f}")
        print(f"  範囲: [{min(best_final_test_accs):.4f}, {max(best_final_test_accs):.4f}]")
        print(f"  実験回数: {len(best_final_test_accs)}回")
    else:
        print(f"最終結果Test精度: {best_final_test_accs[0]:.4f}")
        print(f"実験回数: {len(best_final_test_accs)}回")
    
    # 全組み合わせの最終テスト精度を上位順に表示
    print(f"\n=== 全組み合わせの最終テスト精度ランキング（上位10位） ===")
    # 各組み合わせの最大最終テスト精度を計算
    combination_best_results = {}
    for result in all_results:
        param_key = tuple(sorted(result['grid_search_params'].items()))
        if param_key not in combination_best_results:
            combination_best_results[param_key] = []
        combination_best_results[param_key].append(result['final_test_acc'])
    
    # 各組み合わせの最大値と統計を取得してソート
    combination_max_results = []
    for param_key, accs in combination_best_results.items():
        max_acc = max(accs)
        mean_acc = np.mean(accs)
        std_acc = np.std(accs) if len(accs) > 1 else 0.0
        param_dict = dict(param_key)
        combination_max_results.append((max_acc, mean_acc, std_acc, param_dict))
    
    # 最終テスト精度の平均値で降順ソート
    combination_max_results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'順位':>3} | {'最終テスト精度 平均±標準偏差 (最大値)':>25} | {'パラメータ':>10}")
    print(f"{'='*110}")
    for i, (max_acc, mean_acc, std_acc, param_dict) in enumerate(combination_max_results[:10]):
        param_str = ", ".join([f"{k}={v}" for k, v in param_dict.items()])
        mean_std_str = f"{mean_acc:.4f}±{std_acc:.4f} ({max_acc:.4f})"
        print(f"{i+1:>3}| {mean_std_str:>23} | {param_str}")
    
    if len(combination_max_results) > 10:
        print(f"... 他 {len(combination_max_results) - 10} 組み合わせ")
    
    # RobustH2GCNの場合は最適パラメータでのgate統計も表示
    if MODEL_NAME == 'RobustH2GCN':
        best_param_gate_results = [r for r in best_final_test_results if 'final_gate' in r and r['final_gate'] is not None]
        if best_param_gate_results:
            if len(best_param_gate_results) > 1:
                print(f"\n最適パラメータでのGate統計:")
                gate_means = [r['final_gate'].mean().item() for r in best_param_gate_results]
                gate_stds = [r['final_gate'].std().item() for r in best_param_gate_results]
                gate_mins = [r['final_gate'].min().item() for r in best_param_gate_results]
                gate_maxs = [r['final_gate'].max().item() for r in best_param_gate_results]
                
                print(f"  Gate平均値: {np.mean(gate_means):.4f} ± {np.std(gate_means):.4f}")
                print(f"  Gate標準偏差: {np.mean(gate_stds):.4f} ± {np.std(gate_stds):.4f}")
                print(f"  Gate最小値: {np.mean(gate_mins):.4f} ± {np.std(gate_mins):.4f}")
                print(f"  Gate最大値: {np.mean(gate_maxs):.4f} ± {np.std(gate_maxs):.4f}")
                print(f"  Gate値範囲: [{min(gate_mins):.4f}, {max(gate_maxs):.4f}]")
                print(f"  Gate統計対象実験数: {len(best_param_gate_results)}回")
            else:
                gate = best_param_gate_results[0]['final_gate']
                print(f"\n最適パラメータでのGate値:")
                print(f"  Gate平均値: {gate.mean().item():.4f}")
                print(f"  Gate標準偏差: {gate.std().item():.4f}")
                print(f"  Gate最小値: {gate.min().item():.4f}")
                print(f"  Gate最大値: {gate.max().item():.4f}")
                print(f"  Gate値範囲: [{gate.min().item():.4f}, {gate.max().item():.4f}]")
        