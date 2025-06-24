#!/usr/bin/env python3
"""
LINKXモデルのテストスクリプト
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import LINKX
from models.model_factory import ModelFactory

def test_linkx_models():
    """LINKXモデルをテスト"""
    
    # テスト用のデータを作成
    num_nodes = 100
    input_dim = 16
    hidden_dim = 32
    output_dim = 7
    
    # ランダムな特徴量とエッジ
    X = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    
    print("=== LINKX Model Test ===")
    
    # 1. 基本的なLINKXモデル
    model_linkx = LINKX(
        in_channels=input_dim,
        hidden_channels=hidden_dim,
        out_channels=output_dim,
        num_layers=2,
        dropout=0.1,
        num_nodes=num_nodes
    )
    
    with torch.no_grad():
        output = model_linkx(X, edge_index)
        print(f"LINKX output shape: {output.shape}")
    
    print("\n=== Model Parameters ===")
    print(f"LINKX parameters: {sum(p.numel() for p in model_linkx.parameters()):,}")
    print("\n=== Test Completed Successfully ===")

if __name__ == "__main__":
    test_linkx_models() 