import torch
import torch.nn.functional as F
import numpy as np
from utils.dataset_loader import load_dataset, get_supported_datasets
from utils.feature_creator import create_pca_features, create_label_features, display_node_features, get_feature_info, create_similarity_based_edges, create_similarity_based_edges_with_original
from utils.adjacency_creator import create_normalized_adjacency_matrices, get_adjacency_matrix, apply_adjacency_to_features, combine_hop_features, print_adjacency_info
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
DATASET_NAME = 'Cora'  # ここを変更してデータセットを切り替え

# モデル選択（MLPまたはGCN）
# サポートされているモデル:
# - 'MLP': 1-layer Multi-Layer Perceptron (グラフ構造を無視)
# - 'GCN': Graph Convolutional Network (グラフ構造を活用)
# - 'MLPAndGCNFusion': MLP-GCN Fusion Model (MLPとGCNを並列実行し融合)
# - 'MLPAndGCNEnsemble': MLP-GCN Ensemble Model (MLPとGCNを独立実行しアンサンブル)
# - 'GCNAndMLPConcat': GCN-MLP Concat Model (GCNで生の特徴量、MLPで生の特徴量+ラベル分布特徴量を処理)
MODEL_NAME = 'GCN'  # ここを変更してモデルを切り替え ('MLP', 'GCN', 'MLPAndGCNFusion', 'MLPAndGCNEnsemble', 'GCNAndMLPConcat')

# 実験設定
NUM_RUNS = 10  # 実験回数
NUM_EPOCHS = 200  # エポック数

# データ分割設定
TRAIN_RATIO = 0.6  # 訓練データの割合
VAL_RATIO = 0.2    # 検証データの割合
TEST_RATIO = 0.2   # テストデータの割合

# 特徴量作成設定
MAX_HOPS = 6       # 最大hop数（1, 2, 3, ...）
CALC_NEIGHBOR_LABEL_FEATURES = False  # True: 隣接ノードのラベル特徴量を計算, False: 計算しない
COMBINE_NEIGHBOR_LABEL_FEATURES = False  # True: 元の特徴量にラベル分布ベクトルを結合, False: スキップ
TEMPERATURE = 1.0  # 温度パラメータ
DISABLE_ORIGINAL_FEATURES = False  # True: 元のノード特徴量を無効化（data.xを空にする）

# 類似度ベースエッジ作成設定
USE_SIMILARITY_BASED_EDGES = True  # True: 類似度ベースエッジ作成を実行, False: スキップ
SIMILARITY_EDGE_MODE = 'add'  # 'replace': 元のエッジを置き換え, 'add': 元のエッジに追加
SIMILARITY_FEATURE_TYPE = 'raw'  # 'raw': 生の特徴量のみ, 'label': ラベル分布特徴量のみ
SIMILARITY_RAW_THRESHOLD = 0.165  # 生の特徴量の類似度閾値 (0.0-1.0)
SIMILARITY_LABEL_THRESHOLD = 0.9999997  # ラベル分布特徴量の類似度閾値 (0.0-1.0)

# モデルハイパーパラメータ
HIDDEN_CHANNELS = 32  # 隠れ層の次元
NUM_LAYERS = 2        # レイヤー数
DROPOUT = 0.5         # ドロップアウト率

# GCNAndMLPConcatモデル固有の設定
GCN_HIDDEN_DIM = 16   # GCNの隠れ層次元（Noneの場合はHIDDEN_CHANNELSを使用）
MLP_HIDDEN_DIM = 16   # MLPの隠れ層次元（Noneの場合はHIDDEN_CHANNELSを使用）

# MLP-GCNハイブリッドモデル設定
FUSION_METHOD = 'concat_alpha'  # 'concat', 'add', 'weighted', 'concat_alpha'
ENSEMBLE_METHOD = 'concat_alpha'  # 'average', 'weighted', 'voting', 'concat_alpha'

# PCA設定
USE_PCA = False  # True: PCA圧縮, False: 生の特徴量
PCA_COMPONENTS = 128  # PCAで圧縮する次元数結合後の特徴量の形状:

# 最適化設定
LEARNING_RATE = 0.01  # 学習率
WEIGHT_DECAY = 5e-4   # 重み減衰

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
print(f"類似度ベースエッジ作成使用: {USE_SIMILARITY_BASED_EDGES}")
if USE_SIMILARITY_BASED_EDGES:
    print(f"エッジモード: {SIMILARITY_EDGE_MODE}")
    print(f"特徴量タイプ: {SIMILARITY_FEATURE_TYPE}")
    if SIMILARITY_FEATURE_TYPE == 'raw':
        print(f"生の特徴量類似度閾値: {SIMILARITY_RAW_THRESHOLD}")
    elif SIMILARITY_FEATURE_TYPE == 'label':
        print(f"ラベル分布特徴量類似度閾値: {SIMILARITY_LABEL_THRESHOLD}")
print(f"隠れ層次元: {default_hidden_channels}")
print(f"レイヤー数: {NUM_LAYERS}")
print(f"ドロップアウト: {DROPOUT}")
if MODEL_NAME == 'MLPAndGCNFusion':
    print(f"融合方法: {FUSION_METHOD}")
elif MODEL_NAME == 'MLPAndGCNEnsemble':
    print(f"アンサンブル方法: {ENSEMBLE_METHOD}")
    if ENSEMBLE_METHOD == 'concat_alpha':
        print(f"    学習可能パラメータ: α (GCN重み), 1-α (MLP重み)")
elif MODEL_NAME == 'GCNAndMLPConcat':
    print(f"GCNAndMLPConcatモデル作成: GCNで生の特徴量、MLPで生の特徴量+ラベル分布特徴量を処理")
    print(f"GCN隠れ層次元: {GCN_HIDDEN_DIM}")
    print(f"MLP隠れ層次元: {MLP_HIDDEN_DIM}")
print(f"学習率: {LEARNING_RATE}")
print(f"重み減衰: {WEIGHT_DECAY}")

# 結果を保存するリスト
all_results = []

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
        'hidden_channels': default_hidden_channels,
        'out_channels': dataset.num_classes,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT
    }
    
    # MLPAndGCNFusionの場合は融合方法を指定
    if MODEL_NAME == 'MLPAndGCNFusion':
        model_kwargs.update({
            'fusion_method': FUSION_METHOD
        })
        print(f"  MLPAndGCNFusionモデル作成:")
        print(f"    融合方法: {FUSION_METHOD}")
        if FUSION_METHOD == 'concat_alpha':
            print(f"    学習可能パラメータ: α (GCN重み), 1-α (MLP重み)")
    
    # MLPAndGCNEnsembleの場合は特別な処理
    elif MODEL_NAME == 'MLPAndGCNEnsemble':
        # 生の特徴量とラベル分布特徴量を分離
        if USE_PCA:
            raw_features = run_data.x[:, :PCA_COMPONENTS]
        else:
            raw_features = run_data.x[:, :dataset.num_features]
        
        if neighbor_label_features is not None and CALC_NEIGHBOR_LABEL_FEATURES:
            # ラベル分布特徴量 + 生の特徴量を結合
            label_features = torch.cat([neighbor_label_features, raw_features], dim=1)
        else:
            # 生の特徴量のみを使用
            label_features = raw_features
        
        out = model(raw_features, label_features, run_data.edge_index)
    
    # GCNAndMLPConcatの場合は、生の特徴量とラベル分布特徴量の次元を指定
    elif MODEL_NAME == 'GCNAndMLPConcat':
        # 元の特徴量次元（PCA処理前の生の特徴量）
        if USE_PCA:
            raw_feature_dim = PCA_COMPONENTS
        else:
            raw_feature_dim = dataset.num_features
        
        # ラベル分布特徴量の次元
        label_dist_dim = neighbor_label_features.shape[1] if neighbor_label_features is not None else 0
        
        model_kwargs.update({
            'xfeat_dim': raw_feature_dim,  # 生の特徴量の次元
            'xlabel_dim': label_dist_dim,  # ラベル分布特徴量の次元
            'gcn_hidden_dim': GCN_HIDDEN_DIM,  # GCNの隠れ層次元
            'mlp_hidden_dim': MLP_HIDDEN_DIM   # MLPの隠れ層次元
        })
        
        print(f"  GCNAndMLPConcatモデル作成:")
        print(f"    生の特徴量次元: {raw_feature_dim}")
        print(f"    ラベル分布特徴量次元: {label_dist_dim}")
        print(f"    総特徴量次元: {actual_feature_dim}")
        print(f"    GCN隠れ層次元: {GCN_HIDDEN_DIM}")
        print(f"    MLP隠れ層次元: {MLP_HIDDEN_DIM}")
    
    model = ModelFactory.create_model(**model_kwargs).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 学習ループ
    def train():
        model.train()
        optimizer.zero_grad()
        
        # GCNAndMLPConcatの場合は特別な処理
        if MODEL_NAME == 'GCNAndMLPConcat':
            # 生の特徴量とラベル分布特徴量を分離
            if USE_PCA:
                raw_features = run_data.x[:, :PCA_COMPONENTS]
            else:
                raw_features = run_data.x[:, :dataset.num_features]
            
            if neighbor_label_features is not None:
                label_features = neighbor_label_features
            else:
                label_features = torch.zeros(run_data.x.shape[0], 0, device=device)
            
            out = model(raw_features, label_features, run_data.edge_index)
        # MLPAndGCNEnsembleの場合は特別な処理
        elif MODEL_NAME == 'MLPAndGCNEnsemble':
            # 生の特徴量とラベル分布特徴量を分離
            if USE_PCA:
                raw_features = run_data.x[:, :PCA_COMPONENTS]
            else:
                raw_features = run_data.x[:, :dataset.num_features]
            
            if neighbor_label_features is not None:
                label_features = torch.concat([neighbor_label_features, raw_features], dim=1)
            else:
                label_features = torch.zeros(run_data.x.shape[0], 0, device=device)
            
            out = model(raw_features, label_features, run_data.edge_index)
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
        
        # GCNAndMLPConcatの場合は特別な処理
        if MODEL_NAME == 'GCNAndMLPConcat':
            # 生の特徴量とラベル分布特徴量を分離
            if USE_PCA:
                raw_features = run_data.x[:, :PCA_COMPONENTS]
            else:
                raw_features = run_data.x[:, :dataset.num_features]
            
            if neighbor_label_features is not None:
                label_features = neighbor_label_features
            else:
                label_features = torch.zeros(run_data.x.shape[0], 0, device=device)
            
            out = model(raw_features, label_features, run_data.edge_index)
        # MLPAndGCNEnsembleの場合は特別な処理
        elif MODEL_NAME == 'MLPAndGCNEnsemble':
            # 生の特徴量とラベル分布特徴量を分離
            if USE_PCA:
                raw_features = run_data.x[:, :PCA_COMPONENTS]
            else:
                raw_features = run_data.x[:, :dataset.num_features]
            
            if neighbor_label_features is not None:
                label_features = torch.concat([neighbor_label_features, raw_features], dim=1)
            else:
                label_features = torch.zeros(run_data.x.shape[0], 0, device=device)
            
            out = model(raw_features, label_features, run_data.edge_index)
        else:
            # その他のモデルは標準的な処理
            out = model(run_data.x, run_data.edge_index)
        
        pred = out.argmax(dim=1)
        accs = []
        for mask in [run_data.train_mask, run_data.val_mask, run_data.test_mask]:
            correct = pred[mask] == run_data.y[mask]
            accs.append(int(correct.sum()) / int(mask.sum()))
        return accs
    
    # α値を取得する関数
    def get_alpha_value():
        if MODEL_NAME in ['MLPAndGCNFusion', 'MLPAndGCNEnsemble'] and hasattr(model, 'alpha'):
            return torch.clamp(model.alpha, 0, 1).item()
        return None
    
    # β値を取得する関数
    def get_beta_value():
        if MODEL_NAME in ['MLPAndGCNFusion', 'MLPAndGCNEnsemble'] and hasattr(model, 'alpha'):
            return torch.clamp(1 - model.alpha, 0, 1).item()
        return None
    
    # 学習実行
    best_val_acc = 0
    best_test_acc = 0
    final_train_acc = 0
    final_val_acc = 0
    final_test_acc = 0
    
    for epoch in range(NUM_EPOCHS + 1):
        loss = train()
        train_acc, val_acc, test_acc = test()
        
        # ベスト結果を記録
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        
        # 最終結果を記録
        if epoch == NUM_EPOCHS:
            final_train_acc = train_acc
            final_val_acc = val_acc
            final_test_acc = test_acc
        
        # 進捗表示
        if epoch % DISPLAY_PROGRESS_EVERY == 0:
            alpha_info = ""
            if MODEL_NAME in ['MLPAndGCNFusion', 'MLPAndGCNEnsemble']:
                alpha_val = get_alpha_value()
                beta_val = get_beta_value()
                if alpha_val is not None and beta_val is not None:
                    alpha_info = f", α={alpha_val:.4f}, 1-α={beta_val:.4f}"
                elif alpha_val is not None:
                    alpha_info = f", α={alpha_val:.4f}"
            
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}{alpha_info}')
    
    # MLPAndGCNFusion/MLPAndGCNEnsembleモデルの最終αとβ値を表示
    if MODEL_NAME in ['MLPAndGCNFusion', 'MLPAndGCNEnsemble']:
        final_alpha = get_alpha_value()
        final_beta = get_beta_value()
        if final_alpha is not None or final_beta is not None:
            print(f"\n=== {MODEL_NAME} 最終α・(1-α)値 ===")
            if hasattr(model, 'print_alpha_info'):
                model.print_alpha_info()
            else:
                print(f"α (GCN重み): {final_alpha:.4f}")
                print(f"(1-α) (MLP重み): {final_beta:.4f}")
    
    # GCNAndMLPConcatモデルの最終隠れ層次元情報を表示
    elif MODEL_NAME == 'GCNAndMLPConcat':
        print(f"\n=== GCNAndMLPConcat 最終隠れ層次元情報 ===")
        model.print_hidden_dims_info()
    
    # 結果を保存
    run_result = {
        'run': run + 1,
        'final_train_acc': final_train_acc,
        'final_val_acc': final_val_acc,
        'final_test_acc': final_test_acc,
        'best_val_acc': best_val_acc,
        'best_test_acc': best_test_acc
    }
    
    # MLPAndGCNFusion/MLPAndGCNEnsembleの場合はαとβ値も保存
    if MODEL_NAME in ['MLPAndGCNFusion', 'MLPAndGCNEnsemble']:
        final_alpha = get_alpha_value()
        final_beta = get_beta_value()
        if final_alpha is not None:
            run_result['final_alpha'] = final_alpha
        if final_beta is not None:
            run_result['final_1_minus_alpha'] = final_beta
        
        # α情報を取得
        if MODEL_NAME == 'MLPAndGCNFusion' and hasattr(model, 'get_alpha_info'):
            alpha_info = model.get_alpha_info()
        elif MODEL_NAME == 'MLPAndGCNEnsemble' and hasattr(model, 'get_alpha_info'):
            alpha_info = model.get_alpha_info()
        else:
            # フォールバック用のα情報
            alpha_info = {
                'alpha': final_alpha,
                'beta': final_beta,
                'gcn_weight': final_alpha,
                'mlp_weight': final_beta,
                'gcn_name': 'GCN Features',
                'mlp_name': 'MLP Features',
                'fusion_method': 'ensemble'
            }
        run_result['alpha_info'] = alpha_info
    
    # GCNAndMLPConcatの場合は隠れ層次元情報も保存
    elif MODEL_NAME == 'GCNAndMLPConcat':
        hidden_dims_info = model.get_hidden_dims_info()
        run_result['hidden_dims_info'] = hidden_dims_info
    
    all_results.append(run_result)
    
    print(f"実験 {run + 1} 完了:")
    print(f"  最終結果 - Train: {final_train_acc:.4f}, Val: {final_val_acc:.4f}, Test: {final_test_acc:.4f}")
    print(f"  ベスト結果 - Val: {best_val_acc:.4f}, Test: {best_test_acc:.4f}")

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

# MLPAndGCNFusion/MLPAndGCNEnsembleモデルのαとβ値統計
if MODEL_NAME in ['MLPAndGCNFusion', 'MLPAndGCNEnsemble'] and 'final_alpha' in all_results[0]:
    final_alphas = [r['final_alpha'] for r in all_results]
    final_betas = [r['final_1_minus_alpha'] for r in all_results]
    print(f"\n{MODEL_NAME} α・(1-α)値統計:")
    print(f"  最終α値: {np.mean(final_alphas):.4f} ± {np.std(final_alphas):.4f}")
    print(f"  最終(1-α)値: {np.mean(final_betas):.4f} ± {np.std(final_betas):.4f}")
    print(f"  α値範囲: [{min(final_alphas):.4f}, {max(final_alphas):.4f}]")
    print(f"  (1-α)値範囲: [{min(final_betas):.4f}, {max(final_betas):.4f}]")
    
    # 特徴量の重み統計
    gcn_weights = [r['alpha_info']['gcn_weight'] for r in all_results]
    mlp_weights = [r['alpha_info']['mlp_weight'] for r in all_results]
    print(f"  GCN重み: {np.mean(gcn_weights):.4f} ± {np.std(gcn_weights):.4f}")
    print(f"  MLP重み: {np.mean(mlp_weights):.4f} ± {np.std(mlp_weights):.4f}")

# 詳細な結果表示
print(f"\n=== 詳細結果 ===")
for i, result in enumerate(all_results):
    alpha_info = ""
    if MODEL_NAME in ['MLPAndGCNFusion', 'MLPAndGCNEnsemble'] and 'final_alpha' in result:
        alpha_info = f", α={result['final_alpha']:.4f}"
        if 'final_1_minus_alpha' in result:
            alpha_info += f", 1-α={result['final_1_minus_alpha']:.4f}"
    print(f"実験 {i+1:2d}: Final Test={result['final_test_acc']:.4f}, Best Test={result['best_test_acc']:.4f}{alpha_info}")

print(f"\n=== 実験完了 ===")
print(f"データセット: {DATASET_NAME}")
print(f"モデル: {MODEL_NAME}")
print(f"最終テスト精度: {np.mean(final_test_accs):.4f} ± {np.std(final_test_accs):.4f}")
print(f"ベストテスト精度: {np.mean(best_test_accs):.4f} ± {np.std(best_test_accs):.4f}")

# MLPAndGCNFusion/MLPAndGCNEnsembleモデルの最終αとβ値情報
if MODEL_NAME in ['MLPAndGCNFusion', 'MLPAndGCNEnsemble'] and 'final_alpha' in all_results[0]:
    print(f"最終α値: {np.mean(final_alphas):.4f} ± {np.std(final_alphas):.4f}")
    print(f"最終(1-α)値: {np.mean(final_betas):.4f} ± {np.std(final_betas):.4f}") 