# ============================================================================
# CONVENIENCE FUNCTIONS FOR INVERSE DESIGN
# ============================================================================

def compute_composition_features(structure):
    """
    Compute composition features for a structure on-the-fly.
    
    Uses matminer ElementProperty with preset='magpie' to match training.
    
    Args:
        structure: pymatgen Structure object
    
    Returns:
        numpy array of composition features (71 features)
    """
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    import numpy as np
    
    # Get composition
    comp = structure.composition
    
    try:
        # Try new API first (matminer >= 0.7.4)
        try:
            featurizer = ElementProperty.from_preset("magpie")
        except TypeError:
            # Fall back to old API (matminer < 0.7.4)
            featurizer = ElementProperty(
                data_source='magpie',
                features=['Number', 'MendeleevNumber', 'AtomicWeight', 'MeltingT',
                         'Column', 'Row', 'CovalentRadius', 'Electronegativity',
                         'NsValence', 'NpValence', 'NdValence', 'NfValence',
                         'NValence', 'NsUnfilled', 'NpUnfilled', 'NdUnfilled',
                         'NfUnfilled', 'NUnfilled', 'GSvolume_pa', 'GSbandgap',
                         'GSmagmom', 'SpaceGroupNumber'],
                stats=["minimum", "maximum", "range", "mean", "std_dev"]
            )
        
        # Compute features
        features = featurizer.featurize(comp)
        
        # Convert to numpy array
        features = np.array(features, dtype=np.float32)
        
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure we have 71 features
        if len(features) < 71:
            # Pad with zeros if too few
            features = np.pad(features, (0, 71 - len(features)), constant_values=0.0)
        elif len(features) > 71:
            # Truncate if too many
            features = features[:71]
        
        return features
        
    except Exception as e:
        print(f"Warning: Could not compute composition features: {e}")
        print(f"  Composition: {comp}")
        # Return zero vector as fallback
        return np.zeros(71, dtype=np.float32)

def structure_to_graph(structure, target=None, cutoff=8.0):
    """
    Convert pymatgen Structure to PyG graph with all features.
    
    This is a complete wrapper that:
    1. Extracts node features from atomic properties
    2. Computes edges from neighbor distances
    3. Computes composition features on-the-fly
    
    Used by inverse design workflow.
    
    Args:
        structure: pymatgen Structure object
        target: Target value (optional, use None for inverse design)
        cutoff: Neighbor cutoff distance in Angstroms
    
    Returns:
        PyG Data object with all necessary features
    """
    import torch
    from torch_geometric.data import Data
    
    # Dummy target if not provided
    if target is None:
        target = 0.0
    
    # 1. GET NODE FEATURES
    node_features = []
    for site in structure:
        elem = site.specie
        
        # 12 node features (same as training)
        features = [
            elem.Z,                          # Atomic number
            elem.atomic_mass,                # Atomic mass
            elem.atomic_radius or 1.0,       # Atomic radius
            elem.X or 1.0,                   # Electronegativity
            elem.group or 0,                 # Group
            elem.row or 0,                   # Period
            elem.mendeleev_no or 0,          # Mendeleev number
            elem.electron_affinity or 0,     # Electron affinity
            elem.ionization_energy or 0,     # Ionization energy
            elem.melting_point or 0,         # Melting point
            elem.boiling_point or 0,         # Boiling point
            elem.density_of_solid or 0       # Density
        ]
        
        node_features.append(features)
    
    node_features = torch.tensor(node_features, dtype=torch.float32)
    
    # 2. GET EDGE FEATURES
    all_neighbors = structure.get_all_neighbors(cutoff)
    
    edge_index = []
    edge_attr = []
    
    for i, neighbors in enumerate(all_neighbors):
        for neighbor in neighbors:
            j = neighbor.index
            distance = neighbor.nn_distance
            
            edge_index.append([i, j])
            edge_attr.append([distance])
    
    if len(edge_index) == 0:
        # No neighbors found - create self-loops
        n_atoms = len(structure)
        edge_index = [[i, i] for i in range(n_atoms)]
        edge_attr = [[0.0] for _ in range(n_atoms)]
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    
    # 3. GET COMPOSITION FEATURES
    comp_features = compute_composition_features(structure)
    comp_features = torch.tensor(comp_features, dtype=torch.float32)
    
    # 4. CREATE GRAPH
    graph = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        comp_features=comp_features,
        y=torch.tensor([target], dtype=torch.float32)
    )
    
    # Add metadata
    graph.formula = structure.composition.reduced_formula
    
    return graph


def test_structure_to_graph():
    """Test structure_to_graph function."""
    from pymatgen.core import Structure, Lattice
    
    print("\nTesting structure_to_graph...")
    
    # Create simple cubic perovskite
    structure = Structure(
        lattice=Lattice.cubic(4.0),
        species=['Ca', 'Ti', 'O', 'O', 'O'],
        coords=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ]
    )
    
    print(f"  Structure: {structure.composition.reduced_formula}")
    print(f"  Number of atoms: {len(structure)}")
    
    # Convert to graph
    graph = structure_to_graph(structure)
    
    print(f"  ✓ Graph created")
    print(f"  Number of nodes: {graph.x.shape[0]}")
    print(f"  Node feature dim: {graph.x.shape[1]}")
    print(f"  Number of edges: {graph.edge_index.shape[1]}")
    print(f"  Edge feature dim: {graph.edge_attr.shape[1]}")
    print(f"  Comp feature dim: {graph.comp_features.shape[0]}")
    print(f"  Target: {graph.y.item()}")
    
    # Validate dimensions
    assert graph.x.shape[1] == 12, "Node features should be 12-dim"
    assert graph.edge_attr.shape[1] == 1, "Edge features should be 1-dim"
    assert graph.comp_features.shape[0] == 71, "Comp features should be 71-dim"
    
    print("\n✓ structure_to_graph test passed!")


if __name__ == '__main__':
    # Run test if executed directly
    test_structure_to_graph()