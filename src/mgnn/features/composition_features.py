"""
Composition-based feature extraction.
Uses matminer for compositional featurization.
"""

import numpy as np
from pymatgen.core import Composition
from matminer.featurizers.composition import (
    ElementProperty,
    Stoichiometry,
    ValenceOrbital,
    IonProperty
)
from typing import Dict, List


class CompositionFeaturizer:
    """
    Extract composition-based features using matminer.
    
    Features include:
    - Weighted average elemental properties
    - Stoichiometric features
    - Valence orbital features
    - Ion property features
    """
    
    def __init__(self, preset: str = 'magpie', impute_nan: bool = True):
        """
        Initialize composition featurizer.
        
        Args:
            preset: Feature preset ('magpie', 'matminer', 'deml')
            impute_nan: Whether to impute NaN values with element averages
        """
        self.preset = preset
        self.impute_nan = impute_nan
        
        # Initialize featurizers based on preset
        # Set impute_nan=True to handle missing elemental data gracefully
        if preset == 'magpie':
            self.featurizers = [
                ElementProperty.from_preset('magpie', impute_nan=impute_nan),
                Stoichiometry(),
                ValenceOrbital(impute_nan=impute_nan),
                IonProperty(impute_nan=impute_nan)
            ]
        elif preset == 'matminer':
            self.featurizers = [
                ElementProperty.from_preset('matminer', impute_nan=impute_nan),
                Stoichiometry()
            ]
        elif preset == 'deml':
            self.featurizers = [
                ElementProperty.from_preset('deml', impute_nan=impute_nan),
                Stoichiometry()
            ]
        else:
            raise ValueError(f"Unknown preset: {preset}")
        
        # Get feature names
        self._feature_names = []
        dummy_comp = Composition("Fe2O3")
        for f in self.featurizers:
            self._feature_names.extend(f.feature_labels())
        
        print(f"CompositionFeaturizer initialized:")
        print(f"  Preset: {preset}")
        print(f"  Impute NaN: {impute_nan}")
        print(f"  Total features: {len(self._feature_names)}")

    def featurize(self, composition: Composition) -> np.ndarray:
        """
        Extract features for a composition.
        
        Args:
            composition: Pymatgen Composition object
        
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        for featurizer in self.featurizers:
            try:
                feat = featurizer.featurize(composition)
                features.extend(feat)
            except Exception as e:
                # If featurization fails, use zeros
                n_features = len(featurizer.feature_labels())
                features.extend([0.0] * n_features)
        
        # Replace NaN with 0
        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def get_feature_dim(self) -> int:
        """Return dimensionality of feature vector."""
        return len(self._feature_names)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self._feature_names.copy()


def test_composition_featurizer():
    """Quick test of composition featurizer."""
    print("\nTesting CompositionFeaturizer...")
    
    featurizer = CompositionFeaturizer(preset='magpie')
    
    # Test compositions
    compositions = [
        Composition("LaFeO3"),
        Composition("BaTiO3"),
        Composition("SrTiO3")
    ]
    
    for comp in compositions:
        feat = featurizer.featurize(comp)
        print(f"\n{comp.reduced_formula}:")
        print(f"  Features: {feat.shape}")
        print(f"  Mean: {feat.mean():.3f}, Std: {feat.std():.3f}")
        print(f"  Min: {feat.min():.3f}, Max: {feat.max():.3f}")


if __name__ == '__main__':
    test_composition_featurizer()