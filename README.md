# GSL Codes

Graph Structure Learning (GSL) implementation using PyTorch and PyTorch Geometric.

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd GSLCodes
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv gcn-env
   ```

3. **Activate the virtual environment**
   - On macOS/Linux:
     ```bash
     source gcn-env/bin/activate
     ```
   - On Windows:
     ```bash
     gcn-env\Scripts\activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 基本的な実行方法

After setting up the environment, you can run the main script:

```bash
python main.py
```

### パラメータ化された実行方法

main関数は引数を受け取るように修正されており、異なる設定で複数回実行できます：

```python
from main import main

# 単一実験の実行
main(
    dataset_name='Cornell',
    model_name='RobustH2GCN',
    calc_neighbor_label_features=True,
    num_runs=10,
    num_epochs=600
)

# 特徴量ノイズ付き実験の実行
main(
    dataset_name='Texas',
    model_name='H2GCN',
    calc_neighbor_label_features=True,
    use_feature_modification=True,
    feature_modifications=[
        {'type': 'noise', 'percentage': 0.3, 'method': 'per_node'}
    ],
    num_runs=10,
    num_epochs=600
)
```

### 複数実験の連続実行

`run_experiments.py`を使用して複数の実験を連続実行できます：

```bash
python run_experiments.py
```

または、main.py内の`run_multiple_experiments()`関数を使用：

```python
from main import run_multiple_experiments

# 複数実験の実行
run_multiple_experiments()
```

## 主要なパラメータ

### データセットとモデル
- `dataset_name`: データセット名 ('Cora', 'Citeseer', 'Pubmed', 'Cornell', 'Texas', 'Wisconsin', 'Chameleon', 'Squirrel', 'Actor')
- `model_name`: モデル名 ('MLP', 'GCN', 'GAT', 'GraphSAGE', 'H2GCN', 'RobustH2GCN', 'MixHop', 'RGCN')

### 実験設定
- `num_runs`: 実験回数
- `num_epochs`: エポック数
- `calc_neighbor_label_features`: 隣接ノードのラベル特徴量を計算するかどうか

### 特徴量改変
- `use_feature_modification`: 特徴量改変を使用するかどうか
- `feature_modifications`: 特徴量改変の設定リスト
  ```python
  [
      {'type': 'noise', 'percentage': 0.3, 'method': 'per_node'},
      {'type': 'missingness', 'percentage': 0.2}
  ]
  ```

### エッジ改変
- `use_edge_modification`: エッジ改変を使用するかどうか
- `edge_modifications`: エッジ改変の設定リスト
  ```python
  [
      {'type': 'add', 'percentage': 0.1},
      {'type': 'remove', 'percentage': 0.2}
  ]
  ```

### Grid Search
- `grid_search_params`: Grid Search対象パラメータ
  ```python
  {
      'HIDDEN_CHANNELS': [64],
      'NUM_LAYERS': [1, 2],
      'MAX_HOPS': [2, 3, 4],
      'TEMPERATURE': [0.5, 1.0, 2.0],
      'DROPOUT': [0.5]
  }
  ```

## 実験例

### 1. 単一実験
```python
main(dataset_name='Cornell', model_name='RobustH2GCN')
```

### 2. 特徴量ノイズ実験
```python
main(
    dataset_name='Texas',
    model_name='H2GCN',
    use_feature_modification=True,
    feature_modifications=[{'type': 'noise', 'percentage': 0.3, 'method': 'per_node'}]
)
```

### 3. 複数データセット比較
```python
datasets = ['Cornell', 'Texas', 'Wisconsin']
for dataset in datasets:
    main(dataset_name=dataset, model_name='RobustH2GCN')
```

### 4. 複数モデル比較
```python
models = ['GCN', 'GAT', 'GraphSAGE', 'H2GCN', 'RobustH2GCN']
for model in models:
    main(dataset_name='Cornell', model_name=model)
```

## Project Structure

- `main.py` - Main entry point with parameterized main function
- `run_experiments.py` - Example scripts for running multiple experiments
- `utils/` - Utility modules
  - `feature_creator.py` - Feature creation utilities
  - `dataset_loader.py` - Dataset loading utilities
  - `custom_dataset_creator.py` - Custom dataset creation utilities
  - `label_correlation_analyzer.py` - Label correlation analysis utilities
  - `feature_noise.py` - Feature noise utilities
  - `edge_noise.py` - Edge modification utilities
- `models/` - Model implementations
  - `gcn.py` - Graph Convolutional Network
  - `gat.py` - Graph Attention Network
  - `model_factory.py` - Model factory for easy model creation
  - `h2gcn.py` - H2GCN implementation
  - `robust_h2gcn.py` - RobustH2GCN implementation
  - `mixhop.py` - MixHop implementation
  - `graphsage.py` - GraphSAGE implementation
  - `rgcn.py` - RGCN implementation
- `result/` - Experiment results directory 