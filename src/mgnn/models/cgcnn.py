"""
Crystal Graph Convolutional Network (CGCNN) implementation.
Based on Xie & Grossman (2018) Physical Review Letters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from typing import Optional


class CGConv(nn.Module):
    """
    Crystal Graph Convolution layer.
    
    Combines node features, neighbor features, and edge features.
    """
    
    def __init__(self, node_dim: int, edge_dim: int):
        """
        Initialize CG convolution layer.
        
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
        """
        super(CGConv, self).__init__()
        
        # Edge network: combines source node, edge, and target node features
        self.edge_network = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            nn.BatchNorm1d(node_dim),
            nn.Softplus(),
            nn.Linear(node_dim, node_dim),
            nn.BatchNorm1d(node_dim),
            nn.Softplus()
        )
        
        # Node update network
        self.node_network = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.BatchNorm1d(node_dim),
            nn.Softplus()
        )
    
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
        
        Returns:
            Updated node features [num_nodes, node_dim]
        """
        row, col = edge_index
        
        # Concatenate source node, edge, and target node features
        edge_features = torch.cat([x[row], edge_attr, x[col]], dim=1)
        
        # Apply edge network
        edge_messages = self.edge_network(edge_features)
        
        # Aggregate messages (sum over incoming edges)
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, col, edge_messages)
        
        # Update node features
        x_new = self.node_network(x + aggregated)
        
        return x_new


class CGCNN(nn.Module):
    """
    Crystal Graph Convolutional Neural Network.
    
    Predicts material properties from crystal structure graphs.
    """
    
    def __init__(
        self,
        node_feature_dim: int = 12,
        edge_feature_dim: int = 1,
        comp_feature_dim: int = 71,
        hidden_dim: int = 128,
        n_conv: int = 4,
        n_fc: int = 2,
        dropout: float = 0.1,
        use_comp_features: bool = True
    ):
        """
        Initialize CGCNN model.
        
        Args:
            node_feature_dim: Dimension of input node features
            edge_feature_dim: Dimension of edge features
            comp_feature_dim: Dimension of composition features
            hidden_dim: Hidden layer dimension
            n_conv: Number of convolution layers
            n_fc: Number of fully connected layers after pooling
            dropout: Dropout probability
            use_comp_features: Whether to use composition features
        """
        super(CGCNN, self).__init__()
        
        self.use_comp_features = use_comp_features
        
        # Initial embedding of node features
        self.node_embedding = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Softplus()
        )
        
        # Edge embedding
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Softplus()
        )
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            CGConv(hidden_dim, hidden_dim)
            for _ in range(n_conv)
        ])
        
        # Composition feature embedding (if used)
        if use_comp_features:
            self.comp_embedding = nn.Sequential(
                nn.Linear(comp_feature_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Softplus(),
                nn.Dropout(dropout)
            )
        
        # Fully connected layers after pooling
        fc_layers = []
        input_dim = hidden_dim * 2 if use_comp_features else hidden_dim
        
        for i in range(n_fc):
            fc_layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Softplus(),
                nn.Dropout(dropout)
            ])
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # Output layer
        self.output = nn.Linear(hidden_dim, 1)
        
        print(f"CGCNN initialized:")
        print(f"  Node features: {node_feature_dim} -> {hidden_dim}")
        print(f"  Edge features: {edge_feature_dim} -> {hidden_dim}")
        if use_comp_features:
            print(f"  Comp features: {comp_feature_dim} -> {hidden_dim}")
        print(f"  Conv layers: {n_conv}")
        print(f"  FC layers: {n_fc}")
        print(f"  Dropout: {dropout}")
    
    # def forward(self, data):
    #     """
    #     Forward pass.
        
    #     Args:
    #         data: PyG Data batch object
        
    #     Returns:
    #         Predictions [batch_size, 1]
    #     """
    #     x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
    #     # Embed node and edge features
    #     x = self.node_embedding(x)
    #     edge_attr = self.edge_embedding(edge_attr)
        
    #     # Apply graph convolutions
    #     for conv in self.conv_layers:
    #         x = conv(x, edge_index, edge_attr)
        
    #     # Global pooling
    #     graph_features = global_mean_pool(x, batch)
        
    #     # Add composition features if available
    #     if self.use_comp_features and hasattr(data, 'comp_features'):
    #         comp_feat = self.comp_embedding(data.comp_features)
    #         graph_features = torch.cat([graph_features, comp_feat], dim=1)
        
    #     # Fully connected layers
    #     out = self.fc_layers(graph_features)
        
    #     # Output prediction
    #     out = self.output(out)
        
    #     return out

    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyG Data batch object
        
        Returns:
            Predictions [batch_size, 1]
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Embed node and edge features
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        # Apply graph convolutions
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_attr)
        
        # Global pooling
        graph_features = global_mean_pool(x, batch)
        
        # Add composition features if available
        if self.use_comp_features and hasattr(data, 'comp_features'):
            # CRITICAL FIX: comp_features should already be [batch_size, comp_dim]
            # when properly batched by PyG DataLoader
            comp_feat = data.comp_features
            
            # Handle case where comp_features might be 1D (single graph)
            if comp_feat.dim() == 1:
                comp_feat = comp_feat.unsqueeze(0)
            
            comp_feat = self.comp_embedding(comp_feat)
            graph_features = torch.cat([graph_features, comp_feat], dim=1)
        
        # Fully connected layers
        out = self.fc_layers(graph_features)
        
        # Output prediction
        out = self.output(out)
        
        return out

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_cgcnn():
    """Test CGCNN model."""
    from torch_geometric.data import Data, Batch
    
    print("\nTesting CGCNN model...")
    
    # Create dummy data
    data1 = Data(
        x=torch.randn(20, 12),
        edge_index=torch.randint(0, 20, (2, 240)),
        edge_attr=torch.randn(240, 1),
        comp_features=torch.randn(71),
        y=torch.tensor([0.15])
    )
    
    data2 = Data(
        x=torch.randn(18, 12),
        edge_index=torch.randint(0, 18, (2, 200)),
        edge_attr=torch.randn(200, 1),
        comp_features=torch.randn(71),
        y=torch.tensor([0.25])
    )
    
    # Create batch
    batch = Batch.from_data_list([data1, data2])
    
    # Initialize model
    model = CGCNN(
        node_feature_dim=12,
        edge_feature_dim=1,
        comp_feature_dim=71,
        hidden_dim=64,
        n_conv=3,
        n_fc=2,
        dropout=0.1,
        use_comp_features=True
    )
    
    print(f"\nModel parameters: {count_parameters(model):,}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(batch)
    
    print(f"\nInput batch size: 2")
    print(f"Output shape: {predictions.shape}")
    print(f"Predictions: {predictions.squeeze().tolist()}")
    print("\n✓ CGCNN test passed!")


if __name__ == '__main__':
    test_cgcnn()

# To run the test, use the following commands:
# cd /projectnb/cui-buchem/anandsahu/Softwares/Mine/multinary-gnn
# uv sync
# uv run python -m mgnn.models.cgcnn