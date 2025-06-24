import torch
import torch.nn.functional as F
import numpy as np
from utils.dataset_loader import load_dataset
from utils.feature_creator import create_combined_features
from models import ModelFactory
from models.gsl_labeldist import compute_loss
from utils.label_correlation_analyzer import LabelCorrelationAnalyzer

def run_gsl_analysis(dataset_name='Cora', num_epochs=100, threshold=0.1):
    """
    Run GSL model training and analyze the learned adjacency matrix
    
    Args:
        dataset_name (str): Dataset name to analyze
        num_epochs (int): Number of training epochs
        threshold (float): Threshold for converting probabilities to binary edges
    """
    print(f"=== GSL Adjacency Matrix Analysis for {dataset_name} ===")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    data, dataset = load_dataset(dataset_name, device)
    print(f"Dataset loaded: {dataset_name}")
    print(f"Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}, Classes: {dataset.num_classes}")
    
    # Create features (PCA + neighbor label features)
    data, adj_matrix, one_hot_labels, pca_features = create_combined_features(
        data, device, 
        max_hops=2, 
        exclude_test_labels=True,
        pca_components=128,
        use_pca=True,
        use_neighbor_label_features=True
    )
    
    # Create GSL model
    combined_input_dim = data.x.shape[1] + 2 * dataset.num_classes  # PCA + 2-hop neighbor features
    model = ModelFactory.create_model(
        model_name='GSL',
        in_channels=combined_input_dim,
        hidden_channels=16,
        out_channels=dataset.num_classes,
        num_nodes=data.num_nodes,
        label_embed_dim=16,
        adj_init=adj_matrix,
        model_type='mlp',
        num_layers=2,
        dropout=0.5,
        damping_alpha=0.8
    ).to(device)
    
    print(f"GSL model created with input dimension: {combined_input_dim}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Create B matrix for GSL loss (same label pairs only)
    B = torch.zeros(data.num_nodes, data.num_nodes, device=device)
    train_indices = torch.where(data.train_mask)[0]
    for i in train_indices:
        for j in train_indices:
            if data.y[i] == data.y[j]:
                B[i, j] = 1.0
    
    # Training loop
    print(f"\nTraining GSL model for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # GSL loss computation
        total_loss, loss_dict = compute_loss(
            model, data.x, one_hot_labels, data.train_mask, B,
            lambda_sparse=0.5, lambda_smooth=1.0, lambda_feat_smooth=0.0, max_hops=2
        )
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}, Loss: {total_loss:.4f}, CE: {loss_dict.get('ce_loss', 0):.4f}, "
                  f"Sparse: {loss_dict.get('sparse_loss', 0):.4f}, Smooth: {loss_dict.get('smooth_loss', 0):.4f}")
    
    print("Training completed!")
    
    # Analyze the learned adjacency matrix
    analyzer = LabelCorrelationAnalyzer(device)
    
    # Analyze original graph structure
    print(f"\n=== Original Graph Analysis ===")
    original_result = analyzer.analyze_dataset(dataset_name, save_plots=True, output_dir='./')
    
    # Analyze GSL-generated adjacency matrix
    print(f"\n=== GSL-Generated Adjacency Matrix Analysis ===")
    gsl_result = analyzer.analyze_gsl_adjacency(
        model, data, dataset, 
        threshold=threshold, 
        save_plots=True, 
        output_dir='./'
    )
    
    # Compare results
    print(f"\n=== Comparison Summary ===")
    print(f"Original graph edges: {original_result['dataset_info']['num_edges']:,}")
    print(f"GSL generated edges: {gsl_result['dataset_info']['num_edges']:,}")
    print(f"Edge difference: {gsl_result['dataset_info']['num_edges'] - original_result['dataset_info']['num_edges']:,}")
    
    # Calculate homophily for both
    def calculate_homophily(result):
        total_edges = result['total_edges']
        same_label_edges = 0
        for (label1, label2), count in result['pair_counts'].items():
            if label1 == label2:
                same_label_edges += count
        return same_label_edges / total_edges if total_edges > 0 else 0
    
    original_homophily = calculate_homophily(original_result)
    gsl_homophily = calculate_homophily(gsl_result)
    
    print(f"Original homophily: {original_homophily:.4f}")
    print(f"GSL homophily: {gsl_homophily:.4f}")
    print(f"Homophily difference: {gsl_homophily - original_homophily:.4f}")
    
    return model, analyzer, original_result, gsl_result

def main():
    """
    Main function to run GSL analysis on different datasets
    """
    # List of datasets to analyze
    datasets = ['Cora', 'Citeseer', 'Pubmed']
    
    for dataset in datasets:
        try:
            print(f"\n{'='*60}")
            print(f"Starting analysis for {dataset}")
            print(f"{'='*60}")
            
            model, analyzer, original_result, gsl_result = run_gsl_analysis(
                dataset_name=dataset,
                num_epochs=50,  # Reduced for faster demonstration
                threshold=0.1
            )
            
            print(f"\nAnalysis completed for {dataset}")
            
        except Exception as e:
            print(f"Error analyzing {dataset}: {e}")
            continue

if __name__ == "__main__":
    main() 