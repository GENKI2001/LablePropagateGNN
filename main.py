import torch
import torch.nn.functional as F
import numpy as np
from utils.dataset_loader import load_dataset, get_supported_datasets
from utils.feature_creator import (
    create_neighbor_lable_features, 
    create_combined_features_with_pca, 
    create_combined_features_with_pca_and_co_label,
    display_node_features, 
    get_feature_info,
    display_co_label_embeddings_info
)
from utils.edge_enhancer import (
    enhance_edges_by_similarity,
    analyze_similarity_distribution
)
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
DATASET_NAME = 'Cora'  # ここを変更してデータセットを切り替え

# モデル選択
# サポートされているモデル: 'GCN', 'GCNWithSkip', 'GAT', 'GATWithSkip', 'GATv2', 'MLP', 'MLPWithSkip'
MODEL_NAME = 'GCN'  # ここを変更してモデルを切り替え

# 実験設定
NUM_RUNS = 20  # 実験回数
NUM_EPOCHS = 400  # エポック数

# データ分割設定
TRAIN_RATIO = 0.6  # 訓練データの割合
VAL_RATIO = 0.2    # 検証データの割合
TEST_RATIO = 0.2   # テストデータの割合

# 特徴量作成設定
MAX_HOPS = 3       # 最大hop数（1, 2, 3, ...）
EXCLUDE_TEST_LABELS = False  # テスト・検証ノードのラベルを隣接ノードの特徴量計算から除外するか(Falseの場合はunknownラベルとして登録する)
PCA_COMPONENTS = 30  # PCAで圧縮する次元数

# 共起ラベルエンベディング設定
USE_CO_LABEL_EMBEDDING = True  # 共起ラベルエンベディングを使用するか
CO_LABEL_EMBEDDING_DIM = 32    # 共起ラベルエンベディングの次元数（クラス数に応じて動的に調整される）
CO_LABEL_WINDOW_SIZE = 1       # 共起を計算するウィンドウサイズ
CO_LABEL_MAX_HOPS = 1          # 共起ラベルエンベディングの最大hop数

# エッジ追加設定
USE_EDGE_ENHANCEMENT = True    # 特徴量類似度に基づくエッジ追加を使用するか
EDGE_SIMILARITY_METHOD = 'euclidean'  # 類似度計算方法 ('cosine', 'euclidean', 'pearson', 'jaccard')
EDGE_SIMILARITY_THRESHOLD = 0.5    # エッジ追加の閾値 (0.0-1.0)
EDGE_MAX_EDGES_PER_NODE = None     # ノードあたりの最大エッジ数 (Noneの場合は制限なし)
EDGE_SYMMETRIC = True              # 対称的なエッジ追加を行うか
EDGE_NORMALIZE_FEATURES = True     # 特徴量を正規化するか

# ラベル類似度ベースのエッジ追加設定
USE_LABEL_SIMILARITY_ENHANCEMENT = False  # ラベル類似度に基づくエッジ追加を使用するか
LABEL_SIMILARITY_METHOD = 'cosine'       # ラベル類似度計算方法
LABEL_SIMILARITY_THRESHOLD = 0.99        # ラベル類似度の閾値
LABEL_USE_TRAIN_VAL_ONLY = True          # 訓練・検証データのみを使用するか（テストデータのラベルは使用しない）
LABEL_MAX_HOPS = 2                       # 隣接ノードを考慮する最大hop数

ANALYZE_SIMILARITY_DISTRIBUTION = False  # 類似度分布を分析するか

# モデルハイパーパラメータ
HIDDEN_CHANNELS = 16  # 隠れ層の次元（GCN系）/ 8（GAT系）
NUM_LAYERS = 2        # レイヤー数
DROPOUT = 0.5         # ドロップアウト率
NUM_HEADS = 8         # アテンションヘッド数（GAT系のみ）
CONCAT_HEADS = True   # アテンションヘッドの出力を結合するか（GAT系のみ）

# 最適化設定
LEARNING_RATE = 0.01  # 学習率
WEIGHT_DECAY = 5e-4   # 重み減衰

# 表示設定
DISPLAY_PROGRESS_EVERY = 100  # 何エポックごとに進捗を表示するか
SHOW_FEATURE_DETAILS = False  # 特徴量の詳細を表示するか
SHOW_CO_LABEL_INFO = False    # 共起ラベルエンベディングの詳細を表示するか

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# メイン処理
# ============================================================================

# データセット読み込み
data, dataset = load_dataset(DATASET_NAME, device)

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
print(f"共起ラベルエンベディング: {USE_CO_LABEL_EMBEDDING}")
if USE_CO_LABEL_EMBEDDING:
    print(f"  エンベディング次元: {CO_LABEL_EMBEDDING_DIM}")
    print(f"  ウィンドウサイズ: {CO_LABEL_WINDOW_SIZE}")
    print(f"  最大hop数: {CO_LABEL_MAX_HOPS}")
print(f"エッジ追加: {USE_EDGE_ENHANCEMENT}")
if USE_EDGE_ENHANCEMENT:
    print(f"  類似度計算方法: {EDGE_SIMILARITY_METHOD}")
    print(f"  類似度閾値: {EDGE_SIMILARITY_THRESHOLD}")
    print(f"  最大エッジ数/ノード: {EDGE_MAX_EDGES_PER_NODE}")
    print(f"  対称的エッジ: {EDGE_SYMMETRIC}")
    print(f"  特徴量正規化: {EDGE_NORMALIZE_FEATURES}")
print(f"ラベル類似度エッジ追加: {USE_LABEL_SIMILARITY_ENHANCEMENT}")
if USE_LABEL_SIMILARITY_ENHANCEMENT:
    print(f"  ラベル類似度計算方法: {LABEL_SIMILARITY_METHOD}")
    print(f"  ラベル類似度閾値: {LABEL_SIMILARITY_THRESHOLD}")
    print(f"  訓練・検証データのみを使用: {LABEL_USE_TRAIN_VAL_ONLY}")
print(f"隠れ層次元: {default_hidden_channels}")
print(f"レイヤー数: {NUM_LAYERS}")
print(f"ドロップアウト: {DROPOUT}")
if MODEL_NAME.startswith('GAT'):
    print(f"アテンションヘッド数: {NUM_HEADS}")
    print(f"ヘッド結合: {CONCAT_HEADS}")
print(f"学習率: {LEARNING_RATE}")
print(f"重み減衰: {WEIGHT_DECAY}")

# 類似度分布分析（オプション）
if ANALYZE_SIMILARITY_DISTRIBUTION:
    print(f"\n=== 類似度分布分析 ===")
    analyze_similarity_distribution(
        data, 
        similarity_method=EDGE_SIMILARITY_METHOD,
        normalize_features=EDGE_NORMALIZE_FEATURES
    )

# 特徴量ベースのエッジ追加（実験前）- 全実験で共通
if USE_EDGE_ENHANCEMENT:
    print(f"\n=== 特徴量ベースのエッジ追加（実験前） ===")
    original_features = data.x.clone()
    data, feature_edge_info = enhance_edges_by_similarity(
        data,
        similarity_method=EDGE_SIMILARITY_METHOD,
        threshold=EDGE_SIMILARITY_THRESHOLD,
        max_edges_per_node=EDGE_MAX_EDGES_PER_NODE,
        symmetric=EDGE_SYMMETRIC,
        normalize_features=EDGE_NORMALIZE_FEATURES,
        original_features=original_features
    )

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
    
    # 特徴量作成（共起ラベルエンベディングの使用有無に応じて分岐）
    if USE_CO_LABEL_EMBEDDING:
        run_data, adj_matrix, one_hot_labels, pca_features, co_label_features, label_cooccurrence_matrix = \
            create_combined_features_with_pca_and_co_label(
                run_data, device, 
                max_hops=MAX_HOPS, 
                exclude_test_labels=EXCLUDE_TEST_LABELS, 
                pca_components=PCA_COMPONENTS,
                co_label_embedding_dim=CO_LABEL_EMBEDDING_DIM,
                co_label_window_size=CO_LABEL_WINDOW_SIZE,
                co_label_max_hops=CO_LABEL_MAX_HOPS
            )
        
        # 共起ラベルエンベディングの情報表示（オプション）
        if SHOW_CO_LABEL_INFO:
            display_co_label_embeddings_info(co_label_features, label_cooccurrence_matrix, DATASET_NAME)
    else:
        run_data, adj_matrix, one_hot_labels, pca_features = create_combined_features_with_pca(
            run_data, device, max_hops=MAX_HOPS, exclude_test_labels=EXCLUDE_TEST_LABELS, 
            pca_components=PCA_COMPONENTS
        )

    # 特徴量情報を取得
    feature_info = get_feature_info(run_data, one_hot_labels, max_hops=MAX_HOPS)
    
    # 特徴量の詳細表示（オプション）
    if SHOW_FEATURE_DETAILS:
        display_node_features(run_data, adj_matrix, one_hot_labels, DATASET_NAME, max_hops=MAX_HOPS)
    
    # モデル作成
    model_kwargs = {
        'model_name': MODEL_NAME,
        'in_channels': feature_info['feature_dim'],
        'hidden_channels': default_hidden_channels,
        'out_channels': dataset.num_classes,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT
    }
    
    # GAT系モデルの場合は追加パラメータを設定
    if MODEL_NAME.startswith('GAT'):
        model_kwargs.update({
            'num_heads': NUM_HEADS,
            'concat': CONCAT_HEADS
        })
    
    model = ModelFactory.create_model(**model_kwargs).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 学習ループ
    def train():
        model.train()
        optimizer.zero_grad()
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
    
    # エッジ追加情報も保存
    if USE_EDGE_ENHANCEMENT:
        run_result['feature_edge_info'] = feature_edge_info
    
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
print(f"共起ラベルエンベディング: {USE_CO_LABEL_EMBEDDING}")
if USE_CO_LABEL_EMBEDDING:
    print(f"  エンベディング次元: {CO_LABEL_EMBEDDING_DIM}")
    print(f"  ウィンドウサイズ: {CO_LABEL_WINDOW_SIZE}")
    print(f"  最大hop数: {CO_LABEL_MAX_HOPS}")
print(f"エッジ追加: {USE_EDGE_ENHANCEMENT}")
if USE_EDGE_ENHANCEMENT:
    print(f"  類似度計算方法: {EDGE_SIMILARITY_METHOD}")
    print(f"  類似度閾値: {EDGE_SIMILARITY_THRESHOLD}")
print(f"最終テスト精度: {np.mean(final_test_accs):.4f} ± {np.std(final_test_accs):.4f}")
print(f"ベストテスト精度: {np.mean(best_test_accs):.4f} ± {np.std(best_test_accs):.4f}") 