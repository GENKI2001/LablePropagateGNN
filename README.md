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

After setting up the environment, you can run the main script:

```bash
python index.py
```

## Project Structure

- `main.py` - Main entry point
- `utils/` - Utility modules
  - `feature_creator.py` - Feature creation utilities
  - `dataset_loader.py` - Dataset loading utilities
  - `custom_dataset_creator.py` - Custom dataset creation utilities
  - `label_correlation_analyzer.py` - Label correlation analysis utilities
- `models/` - Model implementations
  - `gcn.py` - Graph Convolutional Network
  - `gat.py` - Graph Attention Network
  - `model_factory.py` - Model factory for easy model creation 