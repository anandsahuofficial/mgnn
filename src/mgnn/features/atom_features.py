"""
Atomic feature extraction for graph neural networks.
Uses pymatgen for elemental properties.
"""

import numpy as np
from pymatgen.core import Element
from typing import List, Dict, Optional


class AtomFeaturizer:
    """
    Extract atomic features for graph nodes.
    
    Features include:
    - Atomic number (normalized)
    - Group and period
    - Electronegativity
    - Ionic radius
    - Atomic mass (normalized)
    - Number of valence electrons
    - Block (s, p, d, f)
    - Mendeleev number
    """
    
    def __init__(self):
        """Initialize featurizer with feature metadata."""
        self.feature_names = [
            'atomic_number_norm',
            'group',
            'period',
            'electronegativity',
            'ionic_radius',
            'atomic_mass_norm',
            'valence_electrons',
            'block_s', 'block_p', 'block_d', 'block_f',
            'mendeleev_number'
        ]
        
        # Normalization constants (approximate max values)
        self.atomic_number_max = 118.0
        self.atomic_mass_max = 300.0
        self.mendeleev_max = 103.0
        
        print(f"AtomFeaturizer initialized: {len(self.feature_names)} features")
    
    def featurize(self, element: str, oxidation_state: Optional[float] = None) -> np.ndarray:
        """
        Extract features for a single element.
        
        Args:
            element: Element symbol (e.g., 'Fe')
            oxidation_state: Oxidation state (optional)
        
        Returns:
            Feature vector as numpy array
        """
        try:
            el = Element(element)
        except:
            # Fallback for invalid elements
            return np.zeros(len(self.feature_names))
        
        features = []
        
        # Atomic number (normalized)
        features.append(el.Z / self.atomic_number_max)
        
        # Group and period
        features.append(el.group if el.group is not None else 0)
        features.append(el.row if el.row is not None else 0)
        
        # Electronegativity (Pauling scale)
        en = el.X if el.X is not None else 0.0
        features.append(en)
        
        # Ionic radius (use oxidation state if provided)
        if oxidation_state is not None:
            try:
                ionic_radius = el.ionic_radii.get(oxidation_state, 0.0)
            except:
                ionic_radius = el.atomic_radius if el.atomic_radius is not None else 0.0
        else:
            ionic_radius = el.atomic_radius if el.atomic_radius is not None else 0.0
        features.append(ionic_radius)
        
        # Atomic mass (normalized)
        features.append(el.atomic_mass / self.atomic_mass_max)
        
        # Valence electrons
        n_valence = el.group if el.group is not None and el.group <= 18 else 0
        if n_valence > 10:
            n_valence = n_valence - 10  # For groups 13-18
        features.append(n_valence)
        
        # Block (one-hot encoding)
        block = el.block if el.block is not None else 's'
        features.extend([
            1.0 if block == 's' else 0.0,
            1.0 if block == 'p' else 0.0,
            1.0 if block == 'd' else 0.0,
            1.0 if block == 'f' else 0.0
        ])
        
        # Mendeleev number (normalized)
        mendeleev = el.mendeleev_no if el.mendeleev_no is not None else 0
        features.append(mendeleev / self.mendeleev_max)
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_dim(self) -> int:
        """Return dimensionality of feature vector."""
        return len(self.feature_names)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names.copy()


def test_atom_featurizer():
    """Quick test of atom featurizer."""
    featurizer = AtomFeaturizer()
    
    print("\nTesting AtomFeaturizer...")
    print(f"Feature dimension: {featurizer.get_feature_dim()}")
    print(f"Features: {featurizer.get_feature_names()}")
    
    # Test a few elements
    for element, ox in [('Fe', 3.0), ('O', -2.0), ('La', 3.0), ('Mn', 4.0)]:
        feat = featurizer.featurize(element, ox)
        print(f"\n{element}^{ox:+.0f}: {feat}")
        print(f"  Shape: {feat.shape}, Type: {feat.dtype}")


if __name__ == '__main__':
    test_atom_featurizer()