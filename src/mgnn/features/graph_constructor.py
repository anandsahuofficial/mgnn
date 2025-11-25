"""
Crystal graph construction for GNN.
Converts pymatgen Structure to PyTorch Geometric Data object.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from pymatgen.core import Structure
from typing import Dict, List, Optional, Tuple
from mgnn.features.atom_features import AtomFeaturizer


class CrystalGraphConstructor:
    """
    Construct crystal graphs from pymatgen Structure objects.
    
    Graph representation:
    - Nodes: atoms with features
    - Edges: bonds within cutoff radius
    - Edge features: distance, bond angle (optional)
    """
    
    def __init__(
        self,
        cutoff_radius: float = 5.0,
        max_neighbors: int = 12,
        atom_featurizer: Optional[AtomFeaturizer] = None
    ):
        """
        Initialize graph constructor.
        
        Args:
            cutoff_radius: Maximum distance for edges (Angstroms)
            max_neighbors: Maximum number of neighbors per atom
            atom_featurizer: AtomFeaturizer instance (creates if None)
        """
        self.cutoff_radius = cutoff_radius
        self.max_neighbors = max_neighbors
        self.atom_featurizer = atom_featurizer or AtomFeaturizer()
        
        print(f"CrystalGraphConstructor initialized:")
        print(f"  Cutoff radius: {cutoff_radius} Å")
        print(f"  Max neighbors: {max_neighbors}")
        print(f"  Atom features: {self.atom_featurizer.get_feature_dim()}")
    
    # def structure_to_graph(
    #     self,
    #     structure: Structure,
    #     oxidation_states: Optional[Dict[int, float]] = None,
    #     property_label: Optional[float] = None
    # ) -> Data:
    #     """
    #     Convert pymatgen Structure to PyG Data object.
        
    #     Args:
    #         structure: Pymatgen Structure object
    #         oxidation_states: Dict mapping atom index to oxidation state
    #         property_label: Target property value (e.g., formation energy)
        
    #     Returns:
    #         PyTorch Geometric Data object
    #     """
    #     # Get all neighbors within cutoff
    #     all_neighbors = structure.get_all_neighbors(
    #         self.cutoff_radius,
    #         include_index=True
    #     )
        
    #     # Build node features
    #     node_features = []
    #     for i, site in enumerate(structure):
    #         element = site.specie.symbol
    #         ox_state = oxidation_states.get(i) if oxidation_states else None
    #         feat = self.atom_featurizer.featurize(element, ox_state)
    #         node_features.append(feat)
        
    #     node_features = np.array(node_features, dtype=np.float32)
        
    #     # Build edge list and edge features
    #     edge_index = []
    #     edge_attr = []
        
    #     for i, neighbors in enumerate(all_neighbors):
    #         # Sort by distance and take top max_neighbors
    #         neighbors = sorted(neighbors, key=lambda x: x[1])[:self.max_neighbors]
            
    #         for neighbor, distance, _ in neighbors:
    #             # Add edge
    #             edge_index.append([i, neighbor])
                
    #             # Edge feature: normalized distance
    #             normalized_dist = distance / self.cutoff_radius
    #             edge_attr.append([normalized_dist])
        
    #     # Convert to tensors
    #     x = torch.tensor(node_features, dtype=torch.float)
    #     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    #     edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
    #     # Create Data object
    #     data = Data(
    #         x=x,
    #         edge_index=edge_index,
    #         edge_attr=edge_attr
    #     )
        
    #     # Add property label if provided
    #     if property_label is not None:
    #         data.y = torch.tensor([property_label], dtype=torch.float)
        
    #     # Store metadata
    #     data.num_nodes = len(structure)
    #     data.num_edges = edge_index.shape[1]
    #     data.formula = structure.composition.reduced_formula
        
    #     return data

    # def structure_to_graph(
    #     self,
    #     structure: Structure,
    #     oxidation_states: Optional[Dict[int, float]] = None,
    #     property_label: Optional[float] = None
    # ) -> Data:
    #     """
    #     Convert pymatgen Structure to PyG Data object.
        
    #     Args:
    #         structure: Pymatgen Structure object
    #         oxidation_states: Dict mapping atom index to oxidation state
    #         property_label: Target property value (e.g., formation energy)
        
    #     Returns:
    #         PyTorch Geometric Data object
    #     """
    #     # Get all neighbors within cutoff
    #     all_neighbors = structure.get_all_neighbors(
    #         self.cutoff_radius,
    #         include_index=True
    #     )
        
    #     # Build node features
    #     node_features = []
    #     for i, site in enumerate(structure):
    #         element = site.specie.symbol
    #         ox_state = oxidation_states.get(i) if oxidation_states else None
    #         feat = self.atom_featurizer.featurize(element, ox_state)
    #         node_features.append(feat)
        
    #     node_features = np.array(node_features, dtype=np.float32)
        
    #     # Build edge list and edge features
    #     edge_index = []
    #     edge_attr = []
        
    #     for i, neighbors in enumerate(all_neighbors):
    #         # Sort by distance and take top max_neighbors
    #         # Each neighbor is a PeriodicNeighbor object with .nn (neighbor site) and .image
    #         neighbors_sorted = sorted(neighbors, key=lambda x: x.nn.distance(structure[i]))[:self.max_neighbors]
            
    #         for neighbor_obj in neighbors_sorted:
    #             # Get neighbor index and distance
    #             neighbor_site = neighbor_obj.nn
    #             distance = neighbor_site.distance(structure[i])
                
    #             # Find the index of this neighbor in the structure
    #             # The neighbor site has an index attribute if include_index=True
    #             if hasattr(neighbor_obj, 'index'):
    #                 neighbor_idx = neighbor_obj.index
    #             else:
    #                 # Fallback: find by matching coordinates
    #                 for j, site in enumerate(structure):
    #                     if np.allclose(site.coords, neighbor_site.coords, atol=1e-3):
    #                         neighbor_idx = j
    #                         break
                
    #             # Add edge
    #             edge_index.append([i, neighbor_idx])
                
    #             # Edge feature: normalized distance
    #             normalized_dist = distance / self.cutoff_radius
    #             edge_attr.append([normalized_dist])
        
    #     # Convert to tensors
    #     x = torch.tensor(node_features, dtype=torch.float)
    #     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    #     edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
    #     # Create Data object
    #     data = Data(
    #         x=x,
    #         edge_index=edge_index,
    #         edge_attr=edge_attr
    #     )
        
    #     # Add property label if provided
    #     if property_label is not None:
    #         data.y = torch.tensor([property_label], dtype=torch.float)
        
    #     # Store metadata
    #     data.num_nodes = len(structure)
    #     data.num_edges = edge_index.shape[1]
    #     data.formula = structure.composition.reduced_formula
        
    #     return data

    def structure_to_graph(
        self,
        structure: Structure,
        oxidation_states: Optional[Dict[int, float]] = None,
        property_label: Optional[float] = None
    ) -> Data:
        """
        Convert pymatgen Structure to PyG Data object.
        
        Args:
            structure: Pymatgen Structure object
            oxidation_states: Dict mapping atom index to oxidation state
            property_label: Target property value (e.g., formation energy)
        
        Returns:
            PyTorch Geometric Data object
        """
        # Build node features
        node_features = []
        for i, site in enumerate(structure):
            element = site.specie.symbol
            ox_state = oxidation_states.get(i) if oxidation_states else None
            feat = self.atom_featurizer.featurize(element, ox_state)
            node_features.append(feat)
        
        node_features = np.array(node_features, dtype=np.float32)
        
        # Get distance matrix
        distance_matrix = structure.distance_matrix
        
        # Build edge list and edge features
        edge_index = []
        edge_attr = []
        
        for i in range(len(structure)):
            # Get distances from atom i to all others
            distances = distance_matrix[i]
            
            # Find neighbors within cutoff (excluding self)
            neighbor_indices = np.where((distances > 1e-6) & (distances <= self.cutoff_radius))[0]
            
            # Sort by distance and take top max_neighbors
            if len(neighbor_indices) > self.max_neighbors:
                neighbor_distances = distances[neighbor_indices]
                sorted_idx = np.argsort(neighbor_distances)[:self.max_neighbors]
                neighbor_indices = neighbor_indices[sorted_idx]
            
            # Add edges
            for j in neighbor_indices:
                distance = distances[j]
                
                # Add edge
                edge_index.append([i, j])
                
                # Edge feature: normalized distance
                normalized_dist = distance / self.cutoff_radius
                edge_attr.append([normalized_dist])
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            # Handle case of no edges (very small structures or large cutoff)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        # Add property label if provided
        if property_label is not None:
            data.y = torch.tensor([property_label], dtype=torch.float)
        
        # Store metadata
        data.num_nodes = len(structure)
        data.num_edges = edge_index.shape[1]
        data.formula = structure.composition.reduced_formula
        
        return data
        
    def get_graph_statistics(self, data: Data) -> Dict:
        """
        Compute statistics for a graph.
        
        Args:
            data: PyG Data object
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'num_nodes': data.num_nodes,
            'num_edges': data.num_edges,
            'avg_degree': data.num_edges / data.num_nodes if data.num_nodes > 0 else 0,
            'feature_dim': data.x.shape[1] if data.x is not None else 0,
            'edge_feature_dim': data.edge_attr.shape[1] if data.edge_attr is not None else 0,
            'has_target': hasattr(data, 'y') and data.y is not None
        }
        
        return stats


def test_graph_constructor():
    """Quick test of graph constructor."""
    from pymatgen.core import Lattice, Structure
    
    print("\nTesting CrystalGraphConstructor...")
    
    # Create simple cubic perovskite structure
    lattice = Lattice.cubic(4.0)
    structure = Structure(
        lattice,
        ['La', 'Mn', 'O', 'O', 'O'],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    )
    
    print(f"\nTest structure: {structure.composition.reduced_formula}")
    print(f"  Sites: {len(structure)}")
    
    # Construct graph
    constructor = CrystalGraphConstructor(cutoff_radius=5.0, max_neighbors=12)
    
    oxidation_states = {0: 3.0, 1: 4.0, 2: -2.0, 3: -2.0, 4: -2.0}
    data = constructor.structure_to_graph(
        structure,
        oxidation_states=oxidation_states,
        property_label=-2.5
    )
    
    print(f"\nGraph constructed:")
    print(f"  Nodes: {data.x.shape}")
    print(f"  Edges: {data.edge_index.shape}")
    print(f"  Edge features: {data.edge_attr.shape}")
    print(f"  Target: {data.y}")
    
    # Statistics
    stats = constructor.get_graph_statistics(data)
    print(f"\nGraph statistics:")
    for key, val in stats.items():
        print(f"  {key}: {val}")


if __name__ == '__main__':
    test_graph_constructor()


# # Test individual modules
# cd /projectnb/cui-buchem/anandsahu/Softwares/Mine/multinary-gnn

# # Test atom featurizer
# uv run python -m mgnn.features.atom_features

# # Test graph constructor
# uv run python -m mgnn.features.graph_constructor

# # Test composition featurizer
# uv run python -m mgnn.features.composition_features