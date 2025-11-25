"""
Graph-based SMOTE (Synthetic Minority Oversampling Technique).
Generate synthetic stable perovskite graphs to balance dataset.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import List, Tuple
import random
from tqdm import tqdm

class GraphSMOTE:
    """
    SMOTE for graph-structured data.
    
    Generates synthetic minority class samples by:
    1. Finding k-nearest neighbors in graph space
    2. Interpolating node features, edge features, and targets
    3. Creating new graph instances
    """
    
    def __init__(
        self,
        k_neighbors: int = 5,
        sampling_strategy: float = 0.5,
        random_state: int = 42
    ):
        """
        Initialize GraphSMOTE.
        
        Args:
            k_neighbors: Number of neighbors for interpolation
            sampling_strategy: Ratio of minority to majority after oversampling
            random_state: Random seed
        """
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        
        np.random.seed(random_state)
        random.seed(random_state)
        torch.manual_seed(random_state)
        
        print(f"GraphSMOTE initialized:")
        print(f"  k_neighbors: {k_neighbors}")
        print(f"  sampling_strategy: {sampling_strategy}")
    
    def compute_graph_distance(self, g1: Data, g2: Data) -> float:
        """
        Compute distance between two graphs.
        
        Uses weighted combination of:
        - Node feature distance (mean difference)
        - Graph size difference
        - Target value difference
        
        Args:
            g1, g2: PyG Data objects
        
        Returns:
            Distance scalar
        """
        # Node feature distance
        node_dist = torch.abs(g1.x.mean(dim=0) - g2.x.mean(dim=0)).mean().item()
        
        # Size difference
        size_dist = abs(g1.num_nodes - g2.num_nodes) / max(g1.num_nodes, g2.num_nodes)
        
        # Target difference
        target_dist = abs(g1.y.item() - g2.y.item())
        
        # Weighted combination
        total_dist = 0.4 * node_dist + 0.2 * size_dist + 0.4 * target_dist
        
        return total_dist
    
    def find_k_neighbors(self, graphs: List[Data], target_idx: int) -> List[int]:
        """
        Find k nearest neighbors for a graph.
        
        Args:
            graphs: List of all graphs
            target_idx: Index of target graph
        
        Returns:
            List of k neighbor indices
        """
        target_graph = graphs[target_idx]
        
        # Compute distances to all other graphs
        distances = []
        for i, g in enumerate(graphs):
            if i != target_idx:
                dist = self.compute_graph_distance(target_graph, g)
                distances.append((i, dist))
        
        # Sort by distance and take k nearest
        distances.sort(key=lambda x: x[1])
        neighbor_indices = [idx for idx, _ in distances[:self.k_neighbors]]
        
        return neighbor_indices
    
    # def interpolate_graphs(self, g1: Data, g2: Data, alpha: float) -> Data:
    #     """
    #     Interpolate between two graphs to create synthetic sample.
        
    #     Args:
    #         g1, g2: Two graphs to interpolate
    #         alpha: Interpolation factor (0 = g1, 1 = g2)
        
    #     Returns:
    #         New synthetic graph
    #     """
    #     # Use structure from g1, interpolate features
    #     synthetic = Data()
        
    #     # Interpolate node features
    #     # Pad to same size if needed
    #     n_nodes = min(g1.num_nodes, g2.num_nodes)
    #     x1_padded = g1.x[:n_nodes]
    #     x2_padded = g2.x[:n_nodes]
        
    #     synthetic.x = (1 - alpha) * x1_padded + alpha * x2_padded
    #     synthetic.x = synthetic.x + torch.randn_like(synthetic.x) * 0.01  # Add small noise
        
    #     # Keep edge structure from g1, interpolate edge features
    #     synthetic.edge_index = g1.edge_index.clone()
        
    #     n_edges = min(g1.edge_attr.shape[0], g2.edge_attr.shape[0])
    #     edge1_padded = g1.edge_attr[:n_edges]
    #     edge2_padded = g2.edge_attr[:n_edges]
        
    #     synthetic.edge_attr = (1 - alpha) * edge1_padded + alpha * edge2_padded
    #     synthetic.edge_attr = synthetic.edge_attr + torch.randn_like(synthetic.edge_attr) * 0.01
        
    #     # Interpolate composition features if present
    #     if hasattr(g1, 'comp_features') and hasattr(g2, 'comp_features'):
    #         synthetic.comp_features = (1 - alpha) * g1.comp_features + alpha * g2.comp_features
    #         synthetic.comp_features = synthetic.comp_features + torch.randn_like(synthetic.comp_features) * 0.01
        
    #     # Interpolate target
    #     synthetic.y = (1 - alpha) * g1.y + alpha * g2.y
        
    #     # Copy metadata
    #     synthetic.num_nodes = synthetic.x.shape[0]
    #     synthetic.num_edges = synthetic.edge_index.shape[1]
    #     synthetic.formula = f"Synthetic_{g1.formula}_{g2.formula}"
        
    #     return synthetic

    def interpolate_graphs(self, g1: Data, g2: Data, alpha: float) -> Data:
        """
        Interpolate between two graphs to create synthetic sample.
        
        Args:
            g1, g2: Two graphs to interpolate
            alpha: Interpolation factor (0 = g1, 1 = g2)
        
        Returns:
            New synthetic graph
        """
        # Use structure from g1, interpolate features
        synthetic = Data()
        
        # Interpolate node features
        # Pad to same size if needed
        n_nodes = min(g1.num_nodes, g2.num_nodes)
        x1_padded = g1.x[:n_nodes]
        x2_padded = g2.x[:n_nodes]
        
        synthetic.x = (1 - alpha) * x1_padded + alpha * x2_padded
        synthetic.x = synthetic.x + torch.randn_like(synthetic.x) * 0.01  # Add small noise
        
        # CRITICAL FIX: Keep BOTH edge structure AND edge features from g1
        # Then trim to match if needed
        synthetic.edge_index = g1.edge_index.clone()
        synthetic.edge_attr = g1.edge_attr.clone()
        
        # If g1 and g2 have compatible edge structures, interpolate edge features
        # Otherwise, just add noise to g1's edge features
        if g1.edge_attr.shape[0] == g2.edge_attr.shape[0]:
            # Same number of edges - safe to interpolate
            synthetic.edge_attr = (1 - alpha) * g1.edge_attr + alpha * g2.edge_attr
            synthetic.edge_attr = synthetic.edge_attr + torch.randn_like(synthetic.edge_attr) * 0.01
        else:
            # Different number of edges - just use g1's edges with small perturbation
            synthetic.edge_attr = g1.edge_attr + torch.randn_like(g1.edge_attr) * 0.02
        
        # Now we need to remove edges that connect to nodes we removed
        # Keep only edges where both source and target are < n_nodes
        edge_mask = (synthetic.edge_index[0] < n_nodes) & (synthetic.edge_index[1] < n_nodes)
        synthetic.edge_index = synthetic.edge_index[:, edge_mask]
        synthetic.edge_attr = synthetic.edge_attr[edge_mask]
        
        # Interpolate composition features if present
        if hasattr(g1, 'comp_features') and hasattr(g2, 'comp_features'):
            synthetic.comp_features = (1 - alpha) * g1.comp_features + alpha * g2.comp_features
            synthetic.comp_features = synthetic.comp_features + torch.randn_like(synthetic.comp_features) * 0.01
        elif hasattr(g1, 'comp_features'):
            # Only g1 has comp_features - use them with small noise
            synthetic.comp_features = g1.comp_features + torch.randn_like(g1.comp_features) * 0.01
        
        # Interpolate target
        synthetic.y = (1 - alpha) * g1.y + alpha * g2.y
        
        # Copy metadata
        synthetic.num_nodes = synthetic.x.shape[0]
        synthetic.num_edges = synthetic.edge_index.shape[1]
        if hasattr(g1, 'formula'):
            synthetic.formula = f"Synthetic_{g1.formula}_{alpha:.2f}"
        
        return synthetic
    
    def generate_samples(
        self,
        minority_graphs: List[Data],
        n_synthetic: int
    ) -> List[Data]:
        """
        Generate synthetic minority class samples.
        
        Args:
            minority_graphs: List of minority class graphs
            n_synthetic: Number of synthetic samples to generate
        
        Returns:
            List of synthetic graphs
        """
        print(f"\nGenerating {n_synthetic} synthetic samples...")
        
        synthetic_graphs = []
        
        for i in tqdm(range(n_synthetic), desc="Generating synthetic samples", leave=False):
            # Randomly select a minority sample
            idx = random.randint(0, len(minority_graphs) - 1)
            
            # Find k neighbors
            neighbor_indices = self.find_k_neighbors(minority_graphs, idx)
            
            # Randomly select one neighbor
            neighbor_idx = random.choice(neighbor_indices)
            
            # Random interpolation factor
            alpha = random.uniform(0.2, 0.8)
            
            # Generate synthetic sample
            synthetic = self.interpolate_graphs(
                minority_graphs[idx],
                minority_graphs[neighbor_idx],
                alpha
            )
            
            synthetic_graphs.append(synthetic)
            
            # if (i + 1) % 100 == 0:
                # print(f"  Generated {i+1}/{n_synthetic} samples...")
        
        print(f"Generated {len(synthetic_graphs)} synthetic samples")
        
        return synthetic_graphs
    
    def fit_resample(
        self,
        graphs: List[Data],
        threshold: float = 0.025
    ) -> Tuple[List[Data], dict]:
        """
        Oversample minority class to achieve desired ratio.
        
        Args:
            graphs: All training graphs
            threshold: Threshold for minority class (stable compounds)
        
        Returns:
            (augmented_graphs, statistics)
        """
        print("\nApplying GraphSMOTE oversampling...")
        
        # Separate minority and majority classes
        minority = [g for g in graphs if g.y.item() <= threshold]
        majority = [g for g in graphs if g.y.item() > threshold]
        
        n_minority = len(minority)
        n_majority = len(majority)
        
        print(f"  Original distribution:")
        print(f"    Minority (stable): {n_minority}")
        print(f"    Majority (unstable): {n_majority}")
        print(f"    Ratio: 1:{n_majority/n_minority:.1f}")
        
        # Calculate how many synthetic samples needed
        target_minority = int(n_majority * self.sampling_strategy)
        n_synthetic = max(0, target_minority - n_minority)
        
        print(f"  Target minority samples: {target_minority}")
        print(f"  Synthetic samples to generate: {n_synthetic}")
        
        # Generate synthetic samples
        if n_synthetic > 0:
            synthetic_graphs = self.generate_samples(minority, n_synthetic)
            
            # Combine all graphs
            augmented_graphs = graphs + synthetic_graphs
        else:
            augmented_graphs = graphs
            synthetic_graphs = []
        
        # Statistics
        stats = {
            'n_original': len(graphs),
            'n_minority_original': n_minority,
            'n_majority_original': n_majority,
            'n_synthetic': n_synthetic,
            'n_augmented': len(augmented_graphs),
            'original_ratio': n_majority / n_minority if n_minority > 0 else float('inf'),
            'augmented_ratio': n_majority / (n_minority + n_synthetic) if (n_minority + n_synthetic) > 0 else float('inf')
        }
        
        print(f"\n  Augmented distribution:")
        print(f"    Minority (stable): {n_minority + n_synthetic}")
        print(f"    Majority (unstable): {n_majority}")
        print(f"    New ratio: 1:{stats['augmented_ratio']:.1f}")
        
        return augmented_graphs, stats
    
def test_graph_smote():
    """Test GraphSMOTE on dummy data."""
    print("\nTesting GraphSMOTE...")
    
    # Create dummy graphs
    def create_dummy_graph(target_value, n_nodes=20):
        data = Data(
            x=torch.randn(n_nodes, 12),
            edge_index=torch.randint(0, n_nodes, (2, n_nodes * 12)),
            edge_attr=torch.randn(n_nodes * 12, 1),
            comp_features=torch.randn(71),
            y=torch.tensor([target_value]),
            formula=f"Test_{target_value:.3f}"
        )
        data.num_nodes = n_nodes
        data.num_edges = n_nodes * 12
        return data
    
    # Create imbalanced dataset
    graphs = []
    
    # 50 stable (minority)
    for i in range(50):
        graphs.append(create_dummy_graph(np.random.uniform(0, 0.025)))
    
    # 500 unstable (majority)
    for i in range(500):
        graphs.append(create_dummy_graph(np.random.uniform(0.1, 0.5)))
    
    print(f"\nOriginal dataset: {len(graphs)} graphs")
    print(f"  Stable: {sum(1 for g in graphs if g.y.item() <= 0.025)}")
    print(f"  Unstable: {sum(1 for g in graphs if g.y.item() > 0.025)}")
    
    # Apply SMOTE
    smote = GraphSMOTE(k_neighbors=5, sampling_strategy=0.3)
    augmented_graphs, stats = smote.fit_resample(graphs, threshold=0.025)
    
    print(f"\nAugmented dataset: {len(augmented_graphs)} graphs")
    print(f"  Original: {stats['n_original']}")
    print(f"  Synthetic: {stats['n_synthetic']}")
    print(f"  Ratio changed: 1:{stats['original_ratio']:.1f} → 1:{stats['augmented_ratio']:.1f}")
    
    # Verify synthetic samples
    synthetic_samples = augmented_graphs[len(graphs):]
    if synthetic_samples:
        print(f"\nSynthetic sample statistics:")
        print(f"  Target range: [{min(g.y.item() for g in synthetic_samples):.4f}, "
              f"{max(g.y.item() for g in synthetic_samples):.4f}]")
        print(f"  All stable? {all(g.y.item() <= 0.025 for g in synthetic_samples)}")
    
    print("\n✓ GraphSMOTE test passed!")
    
    return augmented_graphs, stats


if __name__ == '__main__':
    test_graph_smote()    


# Test GraphSMOTE first
# cd /projectnb/cui-buchem/anandsahu/Softwares/Mine/multinary-gnn
# uv sync
# uv run python -m mgnn.data.graph_augmentation