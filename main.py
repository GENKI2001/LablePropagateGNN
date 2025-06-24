import torch
import torch.nn.functional as F
import numpy as np
from utils.dataset_loader import load_dataset, get_supported_datasets
from utils.feature_creator import create_neighbor_lable_features, create_combined_features_with_pca, display_node_features, get_feature_info
from models import ModelFactory
from models.gsl_labeldist import compute_loss

# ============================================================================
# ハイパーパラメータなどの設定
# ============================================================================

# データセット選択
# サポートされているデータセット:
# CustomGraph: 'CustomGraph_Chain'
# Planetoid: 'Cora', 'Citeseer', 'Pubmed'
# WebKB: 'Cornell', 'Wisconsin'
# WikipediaNetwork: 'Chameleon', 'Squirrel'
# Actor: 'Actor'
DATASET_NAME = 'Cornell'  # ここを変更してデータセットを切り替え

# モデル選択
# サポートされているモデル: 'GCN', 'GCNWithSkip', 'GAT', 'GATWithSkip', 'GATv2', 'MLP', 'MLPWithSkip', 'GSL'
MODEL_NAME = 'GSL'  # ここを変更してモデルを切り替え

# 実験設定
NUM_RUNS = 50  # 実験回数
NUM_EPOCHS = 400  # エポック数

# データ分割設定
TRAIN_RATIO = 0.7  # 訓練データの割合
VAL_RATIO = 0.01    # 検証データの割合
TEST_RATIO = 0.3   # テストデータの割合

# 特徴量作成設定
MAX_HOPS = 4       # 最大hop数（1, 2, 3, ...）
EXCLUDE_TEST_LABELS = True  # テスト・検証ノードのラベルを隣接ノードの特徴量計算から除外するか(Falseの場合はunknownラベルとして登録する)
PCA_COMPONENTS = 100  # PCAで圧縮する次元数

# モデルハイパーパラメータ
HIDDEN_CHANNELS = 16  # 隠れ層の次元（GCN系）/ 8（GAT系）
NUM_LAYERS = 2        # レイヤー数
DROPOUT = 0.5         # ドロップアウト率
NUM_HEADS = 8         # アテンションヘッド数（GAT系のみ）
CONCAT_HEADS = True   # アテンションヘッドの出力を結合するか（GAT系のみ）

# GSLモデル固有のハイパーパラメータ
LABEL_EMBED_DIM = 16  # ラベル埋め込み次元
LAMBDA_SPARSE = 0.01  # スパース正則化の重み（正規化後なので小さく）
LAMBDA_SMOOTH = 1.0   # ラベルスムース正則化の重み
LAMBDA_FEAT_SMOOTH = 0.00  # 特徴量スムージング正則化の重み
# GSLモデルの分類器タイプ（'mlp' または 'gcn' または 'linkx'）
GSL_MODEL_TYPE = 'mlp'  # ここを'mlp'、'gcn'、または'linkx'に変更して切り替え

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
print(f"隠れ層次元: {default_hidden_channels}")
print(f"レイヤー数: {NUM_LAYERS}")
print(f"ドロップアウト: {DROPOUT}")
if MODEL_NAME.startswith('GAT'):
    print(f"アテンションヘッド数: {NUM_HEADS}")
    print(f"ヘッド結合: {CONCAT_HEADS}")
if MODEL_NAME == 'GSL':
    print(f"ラベル埋め込み次元: {LABEL_EMBED_DIM}")
    print(f"スパース正則化重み: {LAMBDA_SPARSE}")
    print(f"スムース正則化重み: {LAMBDA_SMOOTH}")
    print(f"特徴量スムージング正則化重み: {LAMBDA_FEAT_SMOOTH}")
    print(f"モデルタイプ: {GSL_MODEL_TYPE}")
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
    
    # GSLモデルの場合は追加パラメータを設定
    if MODEL_NAME == 'GSL':
        # PCA特徴量 + 隣接ノード特徴量 + ラベル分布（MAX_HOPS分）の次元を計算
        combined_input_dim = feature_info['feature_dim'] + MAX_HOPS * dataset.num_classes
        model_kwargs.update({
            'in_channels': combined_input_dim,  # (PCA + 隣接ノード特徴量) + MAX_HOPS*ラベル分布の次元
            'num_nodes': num_nodes,
            'label_embed_dim': LABEL_EMBED_DIM,
            'adj_init': adj_matrix if adj_matrix is not None else None,
            'model_type': GSL_MODEL_TYPE,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'damping_alpha': 0.8,  # ラベル伝播の減衰係数
        })
    
    model = ModelFactory.create_model(**model_kwargs).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # GSLモデル用のB行列を作成（同じラベルを持つノード間のみ1）
    if MODEL_NAME == 'GSL':
        B = torch.zeros(num_nodes, num_nodes, device=device)
        # trainデータのみを使用してB行列を作成
        train_indices = torch.where(run_data.train_mask)[0]
        for i in train_indices:
            for j in train_indices:
                if run_data.y[i] == run_data.y[j]:
                    B[i, j] = 1.0
    
    # 学習ループ
    def train():
        model.train()
        optimizer.zero_grad()
        
        if MODEL_NAME == 'GSL':
            # GSLモデルの場合は独自の損失関数を使用
            total_loss, loss_dict = compute_loss(
                model, run_data.x, one_hot_labels, run_data.train_mask, B,
                lambda_sparse=LAMBDA_SPARSE, lambda_smooth=LAMBDA_SMOOTH, 
                lambda_feat_smooth=LAMBDA_FEAT_SMOOTH, max_hops=MAX_HOPS
            )
            total_loss.backward()
            optimizer.step()
            return total_loss.item(), loss_dict
        else:
            # 通常のモデルの場合は標準的な損失関数を使用
            out = model(run_data.x, run_data.edge_index)
            loss = F.cross_entropy(out[run_data.train_mask], run_data.y[run_data.train_mask])
            loss.backward()
            optimizer.step()
            return loss.item(), {}
    
    # 評価関数
    @torch.no_grad()
    def test():
        model.eval()
        if MODEL_NAME == 'GSL':
            # GSLモデルの場合は結合された特徴量とone-hotラベルを使用
            out = model(run_data.x, one_hot_labels, max_hops=MAX_HOPS)
        else:
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
        loss, loss_dict = train()
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
            if MODEL_NAME == 'GSL':
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, CE: {loss_dict.get("ce_loss", 0):.4f}, '
                      f'Sparse: {loss_dict.get("sparse_loss", 0):.4f}, Smooth: {loss_dict.get("smooth_loss", 0):.4f}, '
                      f'FeatSmooth: {loss_dict.get("feat_smooth_loss", 0):.4f}, '
                      f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
            else:
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