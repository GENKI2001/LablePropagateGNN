import torch
import torch.nn.functional as F
import numpy as np
from utils.dataset_loader import load_dataset, get_supported_datasets
from utils.feature_creator import create_pca_features, create_label_features, display_node_features, get_feature_info
from models import ModelFactory
from models.gsl_labeldist import compute_loss
from utils.label_correlation_analyzer import LabelCorrelationAnalyzer

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
NUM_RUNS = 1  # 実験回数
NUM_EPOCHS = 1000  # エポック数

# データ分割設定
TRAIN_RATIO = 0.7  # 訓練データの割合
VAL_RATIO = 0.1    # 検証データの割合
TEST_RATIO = 0.2   # テストデータの割合

# 特徴量作成設定
MAX_HOPS = 4       # 最大hop数（1, 2, 3, ...）
EXCLUDE_TEST_LABELS = True  # テスト・検証ノードのラベルを隣接ノードの特徴量計算から除外するか(Falseの場合はunknownラベルとして登録する)
PCA_COMPONENTS = 128  # PCAで圧縮する次元数
USE_PCA = True  # True: PCA圧縮, False: 生の特徴量
USE_NEIGHBOR_LABEL_FEATURES = True  # True: 隣接ノードのラベル特徴量を結合, False: 結合しない

# モデルハイパーパラメータ
HIDDEN_CHANNELS = 16  # 隠れ層の次元（GCN系）/ 8（GAT系）
NUM_LAYERS = 2        # レイヤー数
DROPOUT = 0.5         # ドロップアウト率
NUM_HEADS = 8         # アテンションヘッド数（GAT系のみ）
CONCAT_HEADS = True   # アテンションヘッドの出力を結合するか（GAT系のみ）

# GSLモデル固有のハイパーパラメータ
LABEL_EMBED_DIM = 16  # ラベル埋め込み次元
LAMBDA_SPARSE = 0  # スパース正則化の重み
LAMBDA_SMOOTH = 1.0   # ラベルスムース正則化の重み
LAMBDA_FEAT_SMOOTH = 0.00  # 特徴量スムージング正則化の重み
# GSLモデルの分類器タイプ（'mlp' または 'gcn' または 'linkx'）
GSL_MODEL_TYPE = 'mlp'  # ここを'mlp'、'gcn'、または'linkx'に変更して切り替え
# GSL隣接行列初期化の強度（0.0-1.0、大きいほど元のグラフ構造を強く反映）
GSL_ADJ_INIT_STRENGTH = 0.8  # 0.8: 0->0.1, 1->0.9 の確率で初期化

# GSL隣接行列分析設定
ANALYZE_GSL_ADJACENCY = True  # GSL隣接行列を分析するかどうか
GSL_ADJACENCY_THRESHOLD = 0.1  # 確率を01に変換するための閾値
SAVE_GSL_PLOTS = True  # GSL分析結果のプロットを保存するかどうか

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

# 実験前にPCA処理を実行
if USE_PCA:
    print(f"\n=== 実験前PCA処理 ===")
    data, pca_features, pca = create_pca_features(data, device, pca_components=PCA_COMPONENTS)
else:
    print(f"\n=== PCA処理をスキップ ===")
    print(f"生の特徴量を使用します: {data.x.shape}")

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
    print(f"隣接行列初期化強度: {GSL_ADJ_INIT_STRENGTH}")
    print(f"GSL隣接行列分析: {ANALYZE_GSL_ADJACENCY}")
    if ANALYZE_GSL_ADJACENCY:
        print(f"GSL隣接行列閾値: {GSL_ADJACENCY_THRESHOLD}")
        print(f"GSLプロット保存: {SAVE_GSL_PLOTS}")
print(f"学習率: {LEARNING_RATE}")
print(f"重み減衰: {WEIGHT_DECAY}")

# GSL隣接行列分析用のアナライザーを初期化
if MODEL_NAME == 'GSL' and ANALYZE_GSL_ADJACENCY:
    gsl_analyzer = LabelCorrelationAnalyzer(device)
    print(f"\nGSL隣接行列分析アナライザーを初期化しました")

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
    run_data, adj_matrix, one_hot_labels = create_label_features(
        run_data, device, max_hops=MAX_HOPS, exclude_test_labels=EXCLUDE_TEST_LABELS, 
        use_neighbor_label_features=USE_NEIGHBOR_LABEL_FEATURES
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
            'adj_init_strength': GSL_ADJ_INIT_STRENGTH,
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
    
    # GSL隣接行列分析（最終エポック後）
    if MODEL_NAME == 'GSL' and ANALYZE_GSL_ADJACENCY and run == 0:  # 最初の実験でのみ実行
        print(f"\n=== GSL隣接行列分析（実験 {run + 1}） ===")
        
        # 元のグラフ構造を分析
        print(f"元のグラフ構造を分析中...")
        original_result = gsl_analyzer.analyze_dataset(DATASET_NAME, save_plots=SAVE_GSL_PLOTS, output_dir='./')
        
        # GSL学習済み隣接行列を分析
        print(f"GSL学習済み隣接行列を分析中...")
        gsl_result = gsl_analyzer.analyze_gsl_adjacency(
            model, run_data, dataset, 
            threshold=GSL_ADJACENCY_THRESHOLD, 
            save_plots=SAVE_GSL_PLOTS, 
            output_dir='./'
        )
        
        # 比較結果を表示
        print(f"\n=== GSL隣接行列比較結果 ===")
        print(f"元のグラフエッジ数: {original_result['dataset_info']['num_edges']:,}")
        print(f"GSL生成エッジ数: {gsl_result['dataset_info']['num_edges']:,}")
        print(f"エッジ数差分: {gsl_result['dataset_info']['num_edges'] - original_result['dataset_info']['num_edges']:,}")
        
        # 同質性を計算して比較
        def calculate_homophily(result):
            total_edges = result['total_edges']
            same_label_edges = 0
            for (label1, label2), count in result['pair_counts'].items():
                if label1 == label2:
                    same_label_edges += count
            return same_label_edges / total_edges if total_edges > 0 else 0
        
        original_homophily = calculate_homophily(original_result)
        gsl_homophily = calculate_homophily(gsl_result)
        
        print(f"元のグラフ同質性: {original_homophily:.4f}")
        print(f"GSL生成グラフ同質性: {gsl_homophily:.4f}")
        print(f"同質性差分: {gsl_homophily - original_homophily:.4f}")
        
        # GSL隣接行列の統計情報
        gsl_info = gsl_result['gsl_info']
        print(f"\nGSL隣接行列統計:")
        print(f"  スパース性: {gsl_info['sparsity']:.4f}")
        print(f"  最大確率: {gsl_info['max_probability']:.4f}")
        print(f"  最小確率: {gsl_info['min_probability']:.4f}")
        print(f"  平均確率: {gsl_info['mean_probability']:.4f}")
        print(f"  使用閾値: {gsl_info['threshold']}")
    
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

# GSL隣接行列分析の結果サマリー
if MODEL_NAME == 'GSL' and ANALYZE_GSL_ADJACENCY:
    print(f"\n=== GSL隣接行列分析完了 ===")
    print(f"分析結果は以下のフォルダに保存されました:")
    print(f"  - 元のグラフ分析: label_correlation_images/")
    print(f"  - GSL隣接行列分析: gsl_adjacency_images/")
    print(f"閾値設定: {GSL_ADJACENCY_THRESHOLD}") 