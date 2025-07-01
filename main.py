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
# CustomGraph: 'CustomGraph_Chain'
# Planetoid: 'Cora', 'Citeseer', 'Pubmed'
# WebKB: 'Cornell', 'Texas', 'Wisconsin'
# WikipediaNetwork: 'Chameleon', 'Squirrel'
# Actor: 'Actor'
DATASET_NAME = 'Chameleon'  # ここを変更してデータセットを切り替え

# モデル選択（MLPまたはGCN）
# サポートされているモデル:
# - 'MLP': 1-layer Multi-Layer Perceptron (グラフ構造を無視)
# - 'GCN': Graph Convolutional Network (グラフ構造を活用)
# - 'GAT': Graph Attention Network (アテンション機構を使用したグラフ畳み込み)
# - 'H2GCN': H2GCN Model (1-hopと2-hopの隣接行列を使用してグラフ構造を学習)
# - 'RobustH2GCN': Robust H2GCN Model (特徴量とラベル特徴量をゲート機構で融合)
# - 'MixHop': MixHop Model (異なるべき乗の隣接行列を混合してグラフ畳み込み)
# - 'GraphSAGE': GraphSAGE Model (帰納的学習による大規模グラフ対応)
# - 'GPRGNN': GPR-GNN Model (Generalized PageRank Graph Neural Network)

MODEL_NAME = 'RobustH2GCN'  # ここを変更してモデルを切り替え ('MLP', 'GCN', 'GAT', 'H2GCN', 'RobustH2GCN', 'MixHop', 'GraphSAGE', 'GPRGNN')

# 実験設定
NUM_RUNS = 10  # 実験回数（テスト用に減らす）
NUM_EPOCHS = 600  # エポック数（テスト用に減らす）

# データ分割設定
TRAIN_RATIO = 0.6  # 訓練データの割合
VAL_RATIO = 0.2    # 検証データの割合
TEST_RATIO = 0.2   # テストデータの割合

# 特徴量作成設定
MAX_HOPS = 3       # 最大hop数（1, 2, 3, ...）
CALC_NEIGHBOR_LABEL_FEATURES = True  # True: 隣接ノードのラベル特徴量を計算, False: 計算しない
COMBINE_NEIGHBOR_LABEL_FEATURES = False  # True: 元の特徴量にラベル分布ベクトルを結合, False: スキップ
TEMPERATURE = 0.5  # 温度パラメータ homophily高い時は 0.5 で、heteroは 2.5
DISABLE_ORIGINAL_FEATURES = False  # True: 元のノード特徴量を無効化（data.xを空にする）

# Grid Search設定
USE_GRID_SEARCH = True  # True: Grid searchを実行, False: 単一パラメータで実行
GRID_SEARCH_PARAM = 'HIDDEN_CHANNELS'  # Grid search対象パラメータ ('MAX_HOPS', 'HIDDEN_CHANNELS')
GRID_SEARCH_VALUES = [16]  # Grid searchで試す値

# 特徴量改変設定（統合版）
USE_FEATURE_MODIFICATION = False  # True: 特徴量を改変, False: スキップ

# 改変タイプの説明:
# - 'noise': 特徴量の値を0と1で入れ替える（バイナリ特徴量用）
#   - method: 'uniform'（全ノードで同じ特徴量）, 'random'（各要素独立）, 'per_node'（各ノードでランダム選択）
# - 'missingness': 特徴量を0にマスキング（欠損値として扱う）
#   - percentage: 改変する要素の割合 (0.0-1.0)
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

# モデルハイパーパラメータ
HIDDEN_CHANNELS = 32  # 隠れ層の次元
NUM_LAYERS = 1       # レイヤー数
DROPOUT = 0.5        # ドロップアウト率

# MixHopモデル固有の設定
MIXHOP_POWERS = [0, 1, 2]  # 隣接行列のべき乗のリスト [0, 1, 2] または [0, 1, 2, 3] など

# GraphSAGEモデル固有の設定
GRAPHSAGE_AGGR = 'mean'  # 集約関数 ('mean', 'max', 'lstm')

# GATモデル固有の設定
GAT_NUM_HEADS = 8  # アテンションヘッド数
GAT_CONCAT = True  # アテンションヘッドの出力を結合するかどうか

# GPRGNNモデル固有の設定
GPRGNN_ALPHA = 0.1  # 初期のPageRank係数
GPRGNN_K = 10       # 伝播ステップ数（= GPRConv の hop 数）
GPRGNN_INIT = 'PPR' # 重みの初期化方法（'PPR', 'SGC', 'NPPR', 'Random', 'WS' など）



# PCA設定
USE_PCA = False  # True: PCA圧縮, False: 生の特徴量
PCA_COMPONENTS = 128  # PCAで圧縮する次元数結合後の特徴量の形状:

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
default_hidden_channels = model_info.get('default_hidden_channels', HIDDEN_CHANNELS)

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
print(f"最大hop数: {MAX_HOPS}")
print(f"PCA圧縮次元数: {PCA_COMPONENTS}")
print(f"PCA使用: {USE_PCA}")
print(f"元の特徴量無効化: {DISABLE_ORIGINAL_FEATURES}")
print(f"隣接ノードラベル特徴量計算: {CALC_NEIGHBOR_LABEL_FEATURES}")
print(f"隣接ノード特徴量結合: {COMBINE_NEIGHBOR_LABEL_FEATURES}")
print(f"Grid Search使用: {USE_GRID_SEARCH}")
if USE_GRID_SEARCH:
    print(f"Grid Search対象パラメータ: {GRID_SEARCH_PARAM}")
    print(f"Grid Search値: {GRID_SEARCH_VALUES}")
else:
    if GRID_SEARCH_PARAM == 'MAX_HOPS':
        print(f"単一パラメータ実行: {GRID_SEARCH_PARAM} = {MAX_HOPS}")
    elif GRID_SEARCH_PARAM == 'HIDDEN_CHANNELS':
        print(f"単一パラメータ実行: {GRID_SEARCH_PARAM} = {HIDDEN_CHANNELS}")
    else:
        print(f"単一パラメータ実行: {GRID_SEARCH_PARAM} = {MAX_HOPS}")
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
print(f"隠れ層次元: {HIDDEN_CHANNELS if not USE_GRID_SEARCH else f'Grid Search ({GRID_SEARCH_VALUES})'}")
print(f"レイヤー数: {NUM_LAYERS}")
print(f"ドロップアウト: {DROPOUT}")
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
elif MODEL_NAME == 'GPRGNN':
    print(f"GPRGNNモデル作成: Generalized PageRank Graph Neural Network")
    print(f"PageRank係数: {GPRGNN_ALPHA}")
    print(f"伝播ステップ数: {GPRGNN_K}")
    print(f"重み初期化方法: {GPRGNN_INIT}")

print(f"学習率: {LEARNING_RATE}")
print(f"重み減衰: {WEIGHT_DECAY}")
print(f"Early Stopping使用: {USE_EARLY_STOPPING}")
if USE_EARLY_STOPPING:
    print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print(f"Early Stopping Min Delta: {EARLY_STOPPING_MIN_DELTA}")

# 結果を保存するリスト
all_results = []

# Grid Search実行
if USE_GRID_SEARCH:
    print(f"\n=== Grid Search開始 ===")
    print(f"対象パラメータ: {GRID_SEARCH_PARAM}")
    print(f"試行値: {GRID_SEARCH_VALUES}")
    print(f"総実験数: {len(GRID_SEARCH_VALUES)} × {NUM_RUNS} = {len(GRID_SEARCH_VALUES) * NUM_RUNS}")
    
    grid_search_results = {}
    
    for param_value in GRID_SEARCH_VALUES:
        print(f"\n{'='*60}")
        print(f"=== {GRID_SEARCH_PARAM} = {param_value} ===")
        print(f"{'='*60}")
        
        # パラメータ値を設定
        if GRID_SEARCH_PARAM == 'MAX_HOPS':
            current_max_hops = param_value
            current_hidden_channels = HIDDEN_CHANNELS
        elif GRID_SEARCH_PARAM == 'HIDDEN_CHANNELS':
            current_hidden_channels = param_value
            current_max_hops = MAX_HOPS
        else:
            current_max_hops = MAX_HOPS
            current_hidden_channels = HIDDEN_CHANNELS
        
        # このパラメータ値での実験結果を保存するリスト
        param_results = []
        
        # 実験実行
        for run in range(NUM_RUNS):
            print(f"\n=== 実験 {run + 1}/{NUM_RUNS} ({GRID_SEARCH_PARAM}={param_value}) ===")
            
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
                temperature=TEMPERATURE
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
                'num_layers': NUM_LAYERS,
                'dropout': DROPOUT
            }
            
            # MixHopモデルの場合はべき乗パラメータを指定
            if MODEL_NAME in ['MixHop']:
                model_kwargs.update({
                    'powers': MIXHOP_POWERS
                })
                
                print(f"  {MODEL_NAME}モデル作成:")
                print(f"    べき乗リスト: {MIXHOP_POWERS}")
                print(f"    隠れ層次元: {current_hidden_channels}")
                print(f"    レイヤー数: {NUM_LAYERS}")
                print(f"    ドロップアウト: {DROPOUT}")
            
            # GraphSAGEモデルの場合は集約関数パラメータを指定
            elif MODEL_NAME == 'GraphSAGE':
                model_kwargs.update({
                    'aggr': GRAPHSAGE_AGGR
                })
                
                print(f"  GraphSAGEモデル作成:")
                print(f"    集約関数: {GRAPHSAGE_AGGR}")
                print(f"    隠れ層次元: {current_hidden_channels}")
                print(f"    レイヤー数: {NUM_LAYERS}")
                print(f"    ドロップアウト: {DROPOUT}")
            
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
                print(f"    レイヤー数: {NUM_LAYERS}")
                print(f"    ドロップアウト: {DROPOUT}")
            
            # GPRGNNモデルの場合はパラメータを指定
            elif MODEL_NAME == 'GPRGNN':
                model_kwargs.update({
                    'alpha': GPRGNN_ALPHA,
                    'K': GPRGNN_K,
                    'Init': GPRGNN_INIT
                })
                
                print(f"  GPRGNNモデル作成:")
                print(f"    PageRank係数: {GPRGNN_ALPHA}")
                print(f"    伝播ステップ数: {GPRGNN_K}")
                print(f"    重み初期化方法: {GPRGNN_INIT}")
            

            
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
                print(f"    ドロップアウト: {DROPOUT}")
            
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
        
        grid_search_results[param_value] = param_results
    
    # Grid Search結果の集計と表示
    print(f"\n{'='*80}")
    print(f"=== Grid Search結果サマリー ===")
    print(f"{'='*80}")
    
    # 各パラメータ値での結果を集計
    param_summary = {}
    for param_value, results in grid_search_results.items():
        final_test_accs = [r['final_test_acc'] for r in results]
        best_test_accs = [r['best_test_acc'] for r in results]
        final_val_accs = [r['final_val_acc'] for r in results]
        best_val_accs = [r['best_val_acc'] for r in results]
        
        param_summary[param_value] = {
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
    print(f"\n{GRID_SEARCH_PARAM}値別結果:")
    print(f"{'='*60}")
    print(f"{GRID_SEARCH_PARAM:>8} | {'Final Test':>12} | {'Best Test':>12} | {'Final Val':>12} | {'Best Val':>12}")
    print(f"{'='*60}")
    
    for param_value in sorted(grid_search_results.keys()):
        summary = param_summary[param_value]
        print(f"{param_value:>8} | {summary['final_test_mean']:>10.4f}±{summary['final_test_std']:<1.4f} | "
              f"{summary['best_test_mean']:>10.4f}±{summary['best_test_std']:<1.4f} | "
              f"{summary['final_val_mean']:>10.4f}±{summary['final_val_std']:<1.4f} | "
              f"{summary['best_val_mean']:>10.4f}±{summary['best_val_std']:<1.4f}")
    
    # 最適なパラメータ値を特定（検証精度のベスト値で選択）
    best_param_by_val = max(param_summary.items(), key=lambda x: x[1]['best_val_mean'])
    best_param_by_test = max(param_summary.items(), key=lambda x: x[1]['best_test_mean'])
    
    # 最終結果Test精度の平均値で最適パラメータを特定
    best_param_by_final_test = max(param_summary.items(), key=lambda x: x[1]['final_test_mean'])
    
    print(f"\n{'='*60}")
    print(f"最適パラメータ選択結果:")
    print(f"{'='*60}")
    print(f"検証精度ベスト値による最適{GRID_SEARCH_PARAM}: {best_param_by_val[0]}")
    print(f"  検証精度: {best_param_by_val[1]['best_val_mean']:.4f} ± {best_param_by_val[1]['best_val_std']:.4f}")
    print(f"  テスト精度: {best_param_by_val[1]['best_test_mean']:.4f} ± {best_param_by_val[1]['best_test_std']:.4f}")
    print(f"テスト精度ベスト値による最適{GRID_SEARCH_PARAM}: {best_param_by_test[0]}")
    print(f"  検証精度: {best_param_by_test[1]['best_val_mean']:.4f} ± {best_param_by_test[1]['best_val_std']:.4f}")
    print(f"  テスト精度: {best_param_by_test[1]['best_test_mean']:.4f} ± {best_param_by_test[1]['best_test_std']:.4f}")
    print(f"最終結果Test精度平均値による最適{GRID_SEARCH_PARAM}: {best_param_by_final_test[0]}")
    print(f"  最終結果Test精度: {best_param_by_final_test[1]['final_test_mean']:.4f} ± {best_param_by_final_test[1]['final_test_std']:.4f}")
    
    # 推奨パラメータ（検証精度ベスト値による選択）
    recommended_param = best_param_by_val[0]
    print(f"\n推奨{GRID_SEARCH_PARAM}: {recommended_param} (検証精度ベスト値による選択)")
    
    # 最終結果Test精度の最大値を出力
    max_final_test_mean = best_param_by_final_test[1]['final_test_mean']
    print(f"\n最終結果Test精度の最大平均値: {max_final_test_mean:.4f} ({GRID_SEARCH_PARAM}={best_param_by_final_test[0]})")
    
    # 全結果をall_resultsに統合
    all_results = []
    for param_value, results in grid_search_results.items():
        for result in results:
            result['grid_search_param'] = param_value
            result['grid_search_param_name'] = GRID_SEARCH_PARAM
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
        if 'grid_search_param' in result:
            grid_search_info = f", {result['grid_search_param_name']}={result['grid_search_param']}"
        
        print(f"実験 {i+1:2d}: Final Test={result['final_test_acc']:.4f}, Best Test={result['best_test_acc']:.4f}{alpha_info}{modification_info}{early_stop_info}{grid_search_info}")
    
    # 結果の集計と表示
    print(f"\n=== 実験結果統計 ({NUM_RUNS}回の平均) ===")

# 単一パラメータ実行（Grid Searchを使用しない場合）
else:
    print(f"\n=== 単一パラメータ実行 ===")
    if GRID_SEARCH_PARAM == 'MAX_HOPS':
        print(f"{GRID_SEARCH_PARAM} = {MAX_HOPS}")
    elif GRID_SEARCH_PARAM == 'HIDDEN_CHANNELS':
        print(f"{GRID_SEARCH_PARAM} = {HIDDEN_CHANNELS}")
    else:
        print(f"{GRID_SEARCH_PARAM} = {MAX_HOPS}")
    
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
        
        # 実験中にラベル特徴量を作成
        adj_matrix, one_hot_labels, neighbor_label_features = create_label_features(
            run_data, device, max_hops=MAX_HOPS, calc_neighbor_label_features=CALC_NEIGHBOR_LABEL_FEATURES,
            temperature=TEMPERATURE
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
        feature_info = get_feature_info(run_data, one_hot_labels, max_hops=MAX_HOPS)
        
        # 実際の特徴量次元を使用（隣接ノード特徴量が結合されている場合）
        actual_feature_dim = run_data.x.shape[1]
        print(f"  実際の入力特徴量次元: {actual_feature_dim}")
        
        # 特徴量の詳細表示（オプション）
        if SHOW_FEATURE_DETAILS:
            display_node_features(run_data, adj_matrix, one_hot_labels, DATASET_NAME, max_hops=MAX_HOPS)
        
        # モデル作成
        model_kwargs = {
            'model_name': MODEL_NAME,
            'in_channels': actual_feature_dim,  # 実際の特徴量次元を使用
            'hidden_channels': HIDDEN_CHANNELS,
            'out_channels': dataset.num_classes,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT
        }
        # MixHopモデルの場合はべき乗パラメータを指定
        if MODEL_NAME in ['MixHop']:
            model_kwargs.update({
                'powers': MIXHOP_POWERS
            })
            
            print(f"  {MODEL_NAME}モデル作成:")
            print(f"    べき乗リスト: {MIXHOP_POWERS}")
            print(f"    隠れ層次元: {HIDDEN_CHANNELS}")
            print(f"    レイヤー数: {NUM_LAYERS}")
            print(f"    ドロップアウト: {DROPOUT}")
        
        # GraphSAGEモデルの場合は集約関数パラメータを指定
        elif MODEL_NAME == 'GraphSAGE':
            model_kwargs.update({
                'aggr': GRAPHSAGE_AGGR
            })
            
            print(f"  GraphSAGEモデル作成:")
            print(f"    集約関数: {GRAPHSAGE_AGGR}")
            print(f"    隠れ層次元: {HIDDEN_CHANNELS}")
            print(f"    レイヤー数: {NUM_LAYERS}")
            print(f"    ドロップアウト: {DROPOUT}")
        
        # GATモデルの場合はアテンションヘッドパラメータを指定
        elif MODEL_NAME == 'GAT':
            model_kwargs.update({
                'num_heads': GAT_NUM_HEADS,
                'concat': GAT_CONCAT
            })
            
            print(f"  GATモデル作成:")
            print(f"    アテンションヘッド数: {GAT_NUM_HEADS}")
            print(f"    ヘッド出力結合: {GAT_CONCAT}")
            print(f"    隠れ層次元: {HIDDEN_CHANNELS}")
            print(f"    レイヤー数: {NUM_LAYERS}")
            print(f"    ドロップアウト: {DROPOUT}")
        
        # GPRGNNモデルの場合はパラメータを指定
        elif MODEL_NAME == 'GPRGNN':
            model_kwargs.update({
                'alpha': GPRGNN_ALPHA,
                'K': GPRGNN_K,
                'Init': GPRGNN_INIT
            })
            
            print(f"  GPRGNNモデル作成:")
            print(f"    PageRank係数: {GPRGNN_ALPHA}")
            print(f"    伝播ステップ数: {GPRGNN_K}")
            print(f"    重み初期化方法: {GPRGNN_INIT}")
        

        
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
            print(f"    隠れ層次元: {HIDDEN_CHANNELS}")
            print(f"    ドロップアウト: {DROPOUT}")
        
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
if USE_GRID_SEARCH and 'grid_search_param' in all_results[0]:
    best_param = max(all_results, key=lambda x: x['best_val_acc'])['grid_search_param']
    best_param_by_final_test = max(all_results, key=lambda x: x['final_test_acc'])['grid_search_param']
    max_final_test_acc = max(r['final_test_acc'] for r in all_results)
    
    # 最終結果Test精度ベストのパラメータでの詳細統計を計算
    best_final_test_results = [r for r in all_results if r['grid_search_param'] == best_param_by_final_test]
    best_final_test_accs = [r['final_test_acc'] for r in best_final_test_results]
    best_final_test_mean = np.mean(best_final_test_accs)
    best_final_test_std = np.std(best_final_test_accs)
    best_final_test_variance = np.var(best_final_test_accs)
    
    print(f"最適{GRID_SEARCH_PARAM} (検証精度ベスト): {best_param}")
    print(f"最適{GRID_SEARCH_PARAM} (最終結果Test精度ベスト): {best_param_by_final_test}")
    print(f"最終結果Test精度の最大値: {max_final_test_acc:.4f}")
    
    print(f"\n=== 最適{GRID_SEARCH_PARAM} ({best_param_by_final_test}) の詳細統計 ===")
    print(f"最終結果Test精度:")
    print(f"  平均値: {best_final_test_mean:.4f}")
    print(f"  標準偏差: {best_final_test_std:.4f}")
    print(f"  分散: {best_final_test_variance:.6f}")
    print(f"  範囲: [{min(best_final_test_accs):.4f}, {max(best_final_test_accs):.4f}]")
    print(f"  実験回数: {len(best_final_test_accs)}回")
    
    # RobustH2GCNの場合は最適パラメータでのgate統計も表示
    if MODEL_NAME == 'RobustH2GCN':
        best_param_gate_results = [r for r in best_final_test_results if 'final_gate' in r and r['final_gate'] is not None]
        if best_param_gate_results:
            print(f"\n最適{GRID_SEARCH_PARAM} ({best_param_by_final_test}) でのGate統計:")
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