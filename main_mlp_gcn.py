import torch
import torch.nn.functional as F
import numpy as np
from utils.dataset_loader import load_dataset, get_supported_datasets
from utils.feature_creator import create_pca_features, create_label_features, display_node_features, get_feature_info
from utils.edge_sampler import sample_edges, print_sampling_statistics
from models import ModelFactory

# ============================================================================
# ハイパーパラメータなどの設定
# ============================================================================

# データセット選択
# サポートされているデータセット:
# CustomGraph: 'CustomGraph_Chain'
# Planetoid: 'Cora', 'Citeseer', 'Pubmed'
# WebKB: 'Cornell', 'Texas', 'Wisconsin'
# WikipediaNetwork: 'Chameleon', 'Squirrel'
# Actor: 'Actor'
DATASET_NAME = 'Pubmed'  # ここを変更してデータセットを切り替え

# モデル選択（MLPまたはGCN）
# サポートされているモデル:
# - 'MLP': 1-layer Multi-Layer Perceptron (グラフ構造を無視)
# - 'GCN': Graph Convolutional Network (グラフ構造を活用)
MODEL_NAME = 'MLP'  # ここを変更してモデルを切り替え ('MLP' または 'GCN')

# 実験設定
NUM_RUNS = 100  # 実験回数
NUM_EPOCHS = 400  # エポック数

# データ分割設定
TRAIN_RATIO = 0.6  # 訓練データの割合
VAL_RATIO = 0.2    # 検証データの割合
TEST_RATIO = 0.2   # テストデータの割合

# 特徴量作成設定
MAX_HOPS = 4       # 最大hop数（1, 2, 3, ...）
EXCLUDE_TEST_LABELS = True  # テスト・検証ノードのラベルを隣接ノードの特徴量計算から除外するか(Falseの場合はunknownラベルとして登録する)
USE_PCA = False  # True: PCA圧縮, False: 生の特徴量
PCA_COMPONENTS = 128  # PCAで圧縮する次元数
USE_NEIGHBOR_LABEL_FEATURES = True  # True: 隣接ノードのラベル特徴量を利用
TEMPERATURE = 1.0  # 温度パラメータ

# エッジサンプリング設定
USE_EDGE_SAMPLING = True  # True: エッジサンプリングを実行, False: スキップ
EDGE_SAMPLING_METHOD = 'random'  # 'random', 'degree', 'class', 'structural', 'adaptive'
EDGE_SAMPLING_RATIO = 0.5  # サンプリングするエッジの割合 (0.0-1.0)
EDGE_SAMPLING_STRATEGY = 'high_degree'  # 各手法の戦略
EDGE_SAMPLING_ALPHA = 0.5  # 適応的サンプリングの重みパラメータ

# モデルハイパーパラメータ
HIDDEN_CHANNELS = 16  # 隠れ層の次元
NUM_LAYERS = 2        # レイヤー数
DROPOUT = 0.5         # ドロップアウト率

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

# 実験前にPCA処理を実行
if USE_PCA:
    print(f"\n=== 実験前PCA処理 ===")
    data, pca_features, pca = create_pca_features(data, device, pca_components=PCA_COMPONENTS)
else:
    print(f"\n=== PCA処理をスキップ ===")
    print(f"生の特徴量を使用します: {data.x.shape}")

# エッジサンプリング処理を実行
if USE_EDGE_SAMPLING:
    print(f"\n=== エッジサンプリング処理 ===")
    print(f"サンプリング手法: {EDGE_SAMPLING_METHOD}")
    print(f"サンプリング比率: {EDGE_SAMPLING_RATIO}")
    
    # サンプリングパラメータを設定
    sampling_kwargs = {}
    if EDGE_SAMPLING_METHOD == 'degree':
        sampling_kwargs['strategy'] = EDGE_SAMPLING_STRATEGY
    elif EDGE_SAMPLING_METHOD == 'class':
        sampling_kwargs['strategy'] = EDGE_SAMPLING_STRATEGY
    elif EDGE_SAMPLING_METHOD == 'structural':
        sampling_kwargs['strategy'] = EDGE_SAMPLING_STRATEGY
    elif EDGE_SAMPLING_METHOD == 'adaptive':
        sampling_kwargs['alpha'] = EDGE_SAMPLING_ALPHA
    
    # エッジサンプリングを実行
    data, sampling_stats = sample_edges(
        data, device, 
        method=EDGE_SAMPLING_METHOD,
        sampling_ratio=EDGE_SAMPLING_RATIO,
        **sampling_kwargs
    )
    
    # サンプリング統計を表示
    print_sampling_statistics(sampling_stats)
else:
    print(f"\n=== エッジサンプリングをスキップ ===")
    print(f"元のエッジを使用します: {data.edge_index.shape[1]} エッジ")

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
print(f"テストラベル除外: {EXCLUDE_TEST_LABELS}")
print(f"PCA圧縮次元数: {PCA_COMPONENTS}")
print(f"PCA使用: {USE_PCA}")
print(f"隣接ノード特徴量使用: {USE_NEIGHBOR_LABEL_FEATURES}")
print(f"エッジサンプリング使用: {USE_EDGE_SAMPLING}")
if USE_EDGE_SAMPLING:
    print(f"サンプリング手法: {EDGE_SAMPLING_METHOD}")
    print(f"サンプリング比率: {EDGE_SAMPLING_RATIO}")
    if EDGE_SAMPLING_METHOD in ['degree', 'class', 'structural']:
        print(f"サンプリング戦略: {EDGE_SAMPLING_STRATEGY}")
    elif EDGE_SAMPLING_METHOD == 'adaptive':
        print(f"適応的サンプリング重み: {EDGE_SAMPLING_ALPHA}")
print(f"隠れ層次元: {default_hidden_channels}")
print(f"レイヤー数: {NUM_LAYERS}")
print(f"ドロップアウト: {DROPOUT}")
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
    run_data, adj_matrix, one_hot_labels, neighbor_label_features = create_label_features(
        run_data, device, max_hops=MAX_HOPS, exclude_test_labels=EXCLUDE_TEST_LABELS, 
        use_neighbor_label_features=USE_NEIGHBOR_LABEL_FEATURES,
        temperature=TEMPERATURE
    )

    # 隣接ノードのラベル特徴量を結合
    if USE_NEIGHBOR_LABEL_FEATURES and neighbor_label_features is not None:
        print(f"  隣接ノードラベル特徴量を結合: {run_data.x.shape} + {neighbor_label_features.shape}")
        run_data.x = torch.cat([run_data.x, neighbor_label_features], dim=1)
        print(f"  結合後の特徴量形状: {run_data.x.shape}")

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
    
    model = ModelFactory.create_model(**model_kwargs).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 学習ループ
    def train():
        model.train()
        optimizer.zero_grad()
        
        # MLPとGCNモデルの場合は標準的な損失関数を使用
        out = model(run_data.x, run_data.edge_index)
        loss = F.cross_entropy(out[run_data.train_mask], run_data.y[run_data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()
    
    # 評価関数
    @torch.no_grad()
    def test():
        model.eval()
        out = model(run_data.x, run_data.edge_index)
        pred = out.argmax(dim=1)
        accs = []
        for mask in [run_data.train_mask, run_data.val_mask, run_data.test_mask]:
            correct = pred[mask] == run_data.y[mask]
            accs.append(int(correct.sum()) / int(mask.sum()))
        return accs
    
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
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    
    # 結果を保存
    run_result = {
        'run': run + 1,
        'final_train_acc': final_train_acc,
        'final_val_acc': final_val_acc,
        'final_test_acc': final_test_acc,
        'best_val_acc': best_val_acc,
        'best_test_acc': best_test_acc
    }
    
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

# 詳細な結果表示
print(f"\n=== 詳細結果 ===")
for i, result in enumerate(all_results):
    print(f"実験 {i+1:2d}: Final Test={result['final_test_acc']:.4f}, Best Test={result['best_test_acc']:.4f}")

print(f"\n=== 実験完了 ===")
print(f"データセット: {DATASET_NAME}")
print(f"モデル: {MODEL_NAME}")
print(f"最終テスト精度: {np.mean(final_test_accs):.4f} ± {np.std(final_test_accs):.4f}")
print(f"ベストテスト精度: {np.mean(best_test_accs):.4f} ± {np.std(best_test_accs):.4f}") 