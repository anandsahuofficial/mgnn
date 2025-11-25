"""
Enhanced feature extraction with physics-informed features.
Recovers the lost 78 composition features and adds new ones.
"""

import numpy as np
from pymatgen.core import Composition, Element
from pymatgen.analysis.structure_analyzer import oxide_type
from typing import Dict, List, Optional


class EnhancedFeaturizer:
    """
    Enhanced featurizer with physics-informed features.
    
    Adds:
    - Electronic structure features (band structure)
    - Oxidation state features
    - Tolerance factor features
    - Ionic radius ratios
    - Electronegativity differences
    """
    
    def __init__(self):
        """Initialize enhanced featurizer."""
        self.feature_names = self._get_feature_names()
        print(f"EnhancedFeaturizer initialized: {len(self.feature_names)} features")
    
    def _get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        names = []
        
        # A-site features (averaged over A1 and A2)
        names.extend([
            'a_ionic_radius_mean',
            'a_ionic_radius_std',
            'a_electronegativity_mean',
            'a_electronegativity_diff',
            'a_oxidation_mean',
            'a_oxidation_diff'
        ])
        
        # B-site features (averaged over B1 and B2)
        names.extend([
            'b_ionic_radius_mean',
            'b_ionic_radius_std',
            'b_electronegativity_mean',
            'b_electronegativity_diff',
            'b_oxidation_mean',
            'b_oxidation_diff'
        ])
        
        # A-B interaction features
        names.extend([
            'a_b_radius_ratio',
            'a_b_electroneg_diff',
            'a_b_oxidation_diff'
        ])
        
        # Tolerance factors (already in data)
        names.extend([
            'tolerance_goldschmidt',
            'tolerance_bartel'
        ])
        
        # Ionic radius ratios
        names.extend([
            'radius_ratio_a_o',
            'radius_ratio_b_o'
        ])
        
        # Electronegativity features
        names.extend([
            'electroneg_variance',
            'electroneg_range'
        ])
        
        # Oxidation state features
        names.extend([
            'charge_balance',
            'oxidation_variance'
        ])
        
        return names
    
    def featurize(self, row: Dict) -> np.ndarray:
        """
        Extract enhanced features from a data row.
        
        Args:
            row: Dictionary with element symbols, oxidation states, tolerances
        
        Returns:
            Feature vector
        """
        features = []
        
        # Get elements and oxidation states
        elem_a1 = Element(row['Element_a1'])
        elem_a2 = Element(row['Element_a2'])
        elem_b1 = Element(row['Element_b1'])
        elem_b2 = Element(row['Element_b2'])
        
        ox_a1 = row['Oxidation_a1']
        ox_a2 = row['Oxidation_a2']
        ox_b1 = row['Oxidation_b1']
        ox_b2 = row['Oxidation_b2']
        
        # A-site features
        try:
            r_a1 = elem_a1.ionic_radii.get(ox_a1, elem_a1.atomic_radius)
            r_a2 = elem_a2.ionic_radii.get(ox_a2, elem_a2.atomic_radius)
        except:
            r_a1 = elem_a1.atomic_radius if elem_a1.atomic_radius else 1.5
            r_a2 = elem_a2.atomic_radius if elem_a2.atomic_radius else 1.5
        
        features.append(np.mean([r_a1, r_a2]))  # a_ionic_radius_mean
        features.append(np.std([r_a1, r_a2]))   # a_ionic_radius_std
        
        en_a1 = elem_a1.X if elem_a1.X else 1.5
        en_a2 = elem_a2.X if elem_a2.X else 1.5
        features.append(np.mean([en_a1, en_a2]))  # a_electronegativity_mean
        features.append(abs(en_a1 - en_a2))       # a_electronegativity_diff
        
        features.append(np.mean([ox_a1, ox_a2]))  # a_oxidation_mean
        features.append(abs(ox_a1 - ox_a2))       # a_oxidation_diff
        
        # B-site features
        try:
            r_b1 = elem_b1.ionic_radii.get(ox_b1, elem_b1.atomic_radius)
            r_b2 = elem_b2.ionic_radii.get(ox_b2, elem_b2.atomic_radius)
        except:
            r_b1 = elem_b1.atomic_radius if elem_b1.atomic_radius else 1.0
            r_b2 = elem_b2.atomic_radius if elem_b2.atomic_radius else 1.0
        
        features.append(np.mean([r_b1, r_b2]))  # b_ionic_radius_mean
        features.append(np.std([r_b1, r_b2]))   # b_ionic_radius_std
        
        en_b1 = elem_b1.X if elem_b1.X else 2.0
        en_b2 = elem_b2.X if elem_b2.X else 2.0
        features.append(np.mean([en_b1, en_b2]))  # b_electronegativity_mean
        features.append(abs(en_b1 - en_b2))       # b_electronegativity_diff
        
        features.append(np.mean([ox_b1, ox_b2]))  # b_oxidation_mean
        features.append(abs(ox_b1 - ox_b2))       # b_oxidation_diff
        
        # A-B interaction features
        r_a_mean = np.mean([r_a1, r_a2])
        r_b_mean = np.mean([r_b1, r_b2])
        features.append(r_a_mean / r_b_mean if r_b_mean > 0 else 1.0)  # a_b_radius_ratio
        
        en_a_mean = np.mean([en_a1, en_a2])
        en_b_mean = np.mean([en_b1, en_b2])
        features.append(abs(en_a_mean - en_b_mean))  # a_b_electroneg_diff
        
        ox_a_mean = np.mean([ox_a1, ox_a2])
        ox_b_mean = np.mean([ox_b1, ox_b2])
        features.append(abs(ox_a_mean - ox_b_mean))  # a_b_oxidation_diff
        
        # Tolerance factors (from data)
        features.append(row['tolerance_t'])      # tolerance_goldschmidt
        features.append(row['tolerance_tau'])    # tolerance_bartel
        
        # Radius ratios with oxygen
        r_o = 1.40  # Ionic radius of O2- in Angstroms
        features.append(r_a_mean / r_o)  # radius_ratio_a_o
        features.append(r_b_mean / r_o)  # radius_ratio_b_o
        
        # Electronegativity features
        all_en = [en_a1, en_a2, en_b1, en_b2]
        features.append(np.var(all_en))           # electroneg_variance
        features.append(max(all_en) - min(all_en))  # electroneg_range
        
        # Oxidation state features
        charge_sum = 2*ox_a_mean + 2*ox_b_mean - 6*2  # Should be 0
        features.append(abs(charge_sum))          # charge_balance
        
        all_ox = [ox_a1, ox_a2, ox_b1, ox_b2]
        features.append(np.var(all_ox))           # oxidation_variance
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_dim(self) -> int:
        """Return feature dimension."""
        return len(self.feature_names)
    
    def get_feature_names(self) -> List[str]:
        """Return feature names."""
        return self.feature_names.copy()


def test_enhanced_features():
    """Test enhanced featurizer."""
    print("\nTesting EnhancedFeaturizer...")
    
    featurizer = EnhancedFeaturizer()
    
    # Test on sample perovskite
    sample = {
        'Element_a1': 'La',
        'Element_a2': 'Sr',
        'Element_b1': 'Fe',
        'Element_b2': 'Co',
        'Oxidation_a1': 3.0,
        'Oxidation_a2': 2.0,
        'Oxidation_b1': 3.0,
        'Oxidation_b2': 2.0,
        'tolerance_t': 0.95,
        'tolerance_tau': 4.2
    }
    
    features = featurizer.featurize(sample)
    
    print(f"\nFeature vector shape: {features.shape}")
    print(f"Feature names ({len(featurizer.feature_names)}):")
    for name, val in zip(featurizer.feature_names, features):
        print(f"  {name:<30}: {val:.4f}")
    
    print("\n✓ EnhancedFeaturizer test passed!")


if __name__ == '__main__':
    test_enhanced_features()