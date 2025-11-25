"""
Feature engineering for multinary perovskite oxides.
Extracts 42 features from composition and structure.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pymatgen.core import Structure, Element, Composition
from pymatgen.analysis.local_env import CrystalNN
from matminer.featurizers.composition import (
    ElementProperty,
    Stoichiometry,
    ValenceOrbital,
    IonProperty
)
from matminer.featurizers.structure import (
    DensityFeatures,
    GlobalSymmetryFeatures,
    StructuralComplexity
)
from tqdm import tqdm


class FeatureExtractor:
    """
    Extract 42 physics-informed features for multinary oxides.
    
    Feature Categories:
    1. Compositional (12 features)
    2. Structural (15 features)
    3. Electronic (10 features)
    4. Energetic (5 features)
    """
    
    def __init__(self):
        """Initialize feature extractors."""
        self.feature_names = []
        self._build_feature_list()
    
    def _build_feature_list(self):
        """Define all 42 features."""
        # Compositional features (12)
        self.compositional_features = [
            'frac_a1', 'frac_a2', 'frac_b1', 'frac_b2',  # Stoichiometry
            'mean_atomic_mass', 'std_atomic_mass',
            'mean_atomic_radius', 'std_atomic_radius',
            'mean_electronegativity', 'std_electronegativity',
            'mean_ionization_energy', 'std_ionization_energy'
        ]
        
        # Structural features (15)
        self.structural_features = [
            'lattice_a', 'lattice_b', 'lattice_c',
            'lattice_alpha', 'lattice_beta', 'lattice_gamma',
            'volume', 'volume_per_atom',
            'mean_bond_length', 'std_bond_length',
            'mean_coordination', 'std_coordination',
            'packing_fraction', 'density', 'complexity'
        ]
        
        # Electronic features (10)
        self.electronic_features = [
            'mean_d_electrons', 'std_d_electrons',
            'mean_f_electrons', 'std_f_electrons',
            'mean_valence_electrons', 'std_valence_electrons',
            'mean_oxidation_state', 'std_oxidation_state',
            'charge_imbalance', 'electron_affinity_range'
        ]
        
        # Energetic features (5)
        self.energetic_features = [
            'tolerance_t',  # Goldschmidt
            'tolerance_tau',  # Bartel
            'dH_formation_ev',
            'octahedral_factor',  # μ = r_B / r_O
            'size_disorder'  # Variance in ionic radii
        ]
        
        # Combine all
        self.feature_names = (
            self.compositional_features +
            self.structural_features +
            self.electronic_features +
            self.energetic_features
        )
        
        assert len(self.feature_names) == 42, "Must have exactly 42 features"
    
    def extract_features(self, row: pd.Series) -> Dict[str, float]:
        """
        Extract all features from a single compound.
        
        Args:
            row: DataFrame row with columns:
                - Element_a1, Element_a2, Element_b1, Element_b2
                - Oxidation_a1, Oxidation_a2, Oxidation_b1, Oxidation_b2
                - tolerance_t, tolerance_tau, dH_formation_ev
                - Structure (pymatgen Structure object)
        
        Returns:
            Dictionary of 42 features
        """
        features = {}
        
        # Extract compositional features
        comp_feats = self._extract_compositional(row)
        features.update(comp_feats)
        
        # Extract structural features
        struct_feats = self._extract_structural(row['Structure'])
        features.update(struct_feats)
        
        # Extract electronic features
        elec_feats = self._extract_electronic(row)
        features.update(elec_feats)
        
        # Extract energetic features
        energ_feats = self._extract_energetic(row)
        features.update(energ_feats)
        
        return features
    
    def _extract_compositional(self, row: pd.Series) -> Dict[str, float]:
        """Extract 12 compositional features."""
        elements = [
            row['Element_a1'], row['Element_a2'],
            row['Element_b1'], row['Element_b2']
        ]
        
        # Stoichiometric fractions (A0.5A'0.5B0.5B'0.5O3)
        features = {
            'frac_a1': 0.5,
            'frac_a2': 0.5,
            'frac_b1': 0.5,
            'frac_b2': 0.5
        }
        
        # Get element objects
        elem_objs = [Element(e) for e in elements]
        
        # Atomic mass
        masses = [e.atomic_mass for e in elem_objs]
        features['mean_atomic_mass'] = np.mean(masses)
        features['std_atomic_mass'] = np.std(masses)
        
        # Atomic radius (use ionic radii if available)
        radii = []
        for elem, ox_state in zip(
            elements,
            [row['Oxidation_a1'], row['Oxidation_a2'], row['Oxidation_b1'], row['Oxidation_b2']]
        ):
            try:
                # Try to get ionic radius for oxidation state
                from pymatgen.core import Species
                species = Species(elem, ox_state)
                radius = species.ionic_radius
                if radius is None:
                    radius = Element(elem).atomic_radius
            except:
                radius = Element(elem).atomic_radius
            
            radii.append(radius if radius is not None else 1.0)
        
        features['mean_atomic_radius'] = np.mean(radii)
        features['std_atomic_radius'] = np.std(radii)
        
        # Electronegativity (Pauling scale)
        en_values = [e.X if e.X is not None else 2.0 for e in elem_objs]
        features['mean_electronegativity'] = np.mean(en_values)
        features['std_electronegativity'] = np.std(en_values)
        
        # Ionization energy
        ie_values = [e.ionization_energy if e.ionization_energy is not None else 10.0 for e in elem_objs]
        features['mean_ionization_energy'] = np.mean(ie_values)
        features['std_ionization_energy'] = np.std(ie_values)
        
        return features
    
    def _extract_structural(self, structure: Structure) -> Dict[str, float]:
        """Extract 15 structural features."""
        features = {}
        
        # Lattice parameters
        lattice = structure.lattice
        features['lattice_a'] = lattice.a
        features['lattice_b'] = lattice.b
        features['lattice_c'] = lattice.c
        features['lattice_alpha'] = lattice.alpha
        features['lattice_beta'] = lattice.beta
        features['lattice_gamma'] = lattice.gamma
        
        # Volume
        features['volume'] = lattice.volume
        features['volume_per_atom'] = lattice.volume / len(structure)
        
        # Bond lengths (using CrystalNN)
        try:
            cnn = CrystalNN()
            bond_lengths = []
            
            for i, site in enumerate(structure):
                nn_info = cnn.get_nn_info(structure, i)
                for nn in nn_info:
                    bond_lengths.append(nn['weight'])  # Distance weighted
            
            features['mean_bond_length'] = np.mean(bond_lengths) if bond_lengths else 2.5
            features['std_bond_length'] = np.std(bond_lengths) if bond_lengths else 0.5
        except:
            # Fallback if CrystalNN fails
            features['mean_bond_length'] = 2.5
            features['std_bond_length'] = 0.5
        
        # Coordination numbers
        try:
            coord_numbers = [len(cnn.get_nn_info(structure, i)) for i in range(len(structure))]
            features['mean_coordination'] = np.mean(coord_numbers)
            features['std_coordination'] = np.std(coord_numbers)
        except:
            features['mean_coordination'] = 6.0  # Typical for perovskites
            features['std_coordination'] = 1.0
        
        # Packing fraction (approximate)
        atomic_volumes = sum([site.specie.atomic_radius**3 * 4/3 * np.pi for site in structure])
        features['packing_fraction'] = atomic_volumes / lattice.volume if lattice.volume > 0 else 0.5
        
        # Density
        features['density'] = structure.density
        
        # Structural complexity (number of symmetry operations)
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            sga = SpacegroupAnalyzer(structure)
            features['complexity'] = len(sga.get_symmetry_operations())
        except:
            features['complexity'] = 48.0  # Typical for perovskites
        
        return features
    
    def _extract_electronic(self, row: pd.Series) -> Dict[str, float]:
        """Extract 10 electronic features."""
        elements = [
            row['Element_a1'], row['Element_a2'],
            row['Element_b1'], row['Element_b2']
        ]
        oxidation_states = [
            row['Oxidation_a1'], row['Oxidation_a2'],
            row['Oxidation_b1'], row['Oxidation_b2']
        ]
        
        features = {}
        
        # Get element objects
        elem_objs = [Element(e) for e in elements]
        
        # d-electrons (important for transition metals at B-site)
        d_electrons = []
        for elem, ox in zip(elem_objs, oxidation_states):
            try:
                # Get d-electron count considering oxidation state
                neutral_d = elem.electronic_structure.count('d')
                ionized_d = max(0, neutral_d - ox)  # Simplified
                d_electrons.append(ionized_d)
            except:
                d_electrons.append(0)
        
        features['mean_d_electrons'] = np.mean(d_electrons)
        features['std_d_electrons'] = np.std(d_electrons)
        
        # f-electrons
        f_electrons = []
        for elem, ox in zip(elem_objs, oxidation_states):
            try:
                neutral_f = elem.electronic_structure.count('f')
                ionized_f = max(0, neutral_f - ox)
                f_electrons.append(ionized_f)
            except:
                f_electrons.append(0)
        
        features['mean_f_electrons'] = np.mean(f_electrons)
        features['std_f_electrons'] = np.std(f_electrons)
        
        # Valence electrons
        valence = [e.group_id if e.group_id else 0 for e in elem_objs]
        features['mean_valence_electrons'] = np.mean(valence)
        features['std_valence_electrons'] = np.std(valence)
        
        # Oxidation states
        features['mean_oxidation_state'] = np.mean(oxidation_states)
        features['std_oxidation_state'] = np.std(oxidation_states)
        
        # Charge imbalance (should be close to 0 for charge-neutral)
        total_charge = sum([ox * 0.5 for ox in oxidation_states])  # A0.5A'0.5B0.5B'0.5
        oxygen_charge = -2 * 3  # O3
        features['charge_imbalance'] = abs(total_charge + oxygen_charge)
        
        # Electron affinity range
        ea_values = [e.electron_affinity if e.electron_affinity else 0 for e in elem_objs]
        features['electron_affinity_range'] = max(ea_values) - min(ea_values)
        
        return features
    
    def _extract_energetic(self, row: pd.Series) -> Dict[str, float]:
        """Extract 5 energetic features."""
        features = {}
        
        # Tolerance factors (already computed)
        features['tolerance_t'] = row['tolerance_t']
        features['tolerance_tau'] = row['tolerance_tau']
        features['dH_formation_ev'] = row['dH_formation_ev']
        
        # Octahedral factor μ = r_B / r_O
        # Get B-site radii
        try:
            from pymatgen.core import Species
            r_b1 = Species(row['Element_b1'], row['Oxidation_b1']).ionic_radius
            r_b2 = Species(row['Element_b2'], row['Oxidation_b2']).ionic_radius
            r_b_mean = (r_b1 + r_b2) / 2
            r_o = Species('O', -2).ionic_radius
            features['octahedral_factor'] = r_b_mean / r_o if r_o > 0 else 0.5
        except:
            features['octahedral_factor'] = 0.5
        
        # Size disorder (variance in ionic radii)
        try:
            radii = [
                Species(row['Element_a1'], row['Oxidation_a1']).ionic_radius,
                Species(row['Element_a2'], row['Oxidation_a2']).ionic_radius,
                Species(row['Element_b1'], row['Oxidation_b1']).ionic_radius,
                Species(row['Element_b2'], row['Oxidation_b2']).ionic_radius
            ]
            features['size_disorder'] = np.var([r for r in radii if r is not None])
        except:
            features['size_disorder'] = 0.1
        
        return features
    
    def extract_all(self, df: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
        """
        Extract features for entire DataFrame.
        
        Returns:
            DataFrame with original data + 42 feature columns
        """
        print(f"\nExtracting {len(self.feature_names)} features...")
        
        features_list = []
        iterator = tqdm(df.iterrows(), total=len(df)) if show_progress else df.iterrows()
        
        for idx, row in iterator:
            try:
                features = self.extract_features(row)
                features_list.append(features)
            except Exception as e:
                print(f"Warning: Feature extraction failed for {row['Unique_ID']}: {e}")
                # Append NaN features
                features_list.append({k: np.nan for k in self.feature_names})
        
        # Create features DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Concatenate with original
        result_df = pd.concat([df, features_df], axis=1)
        
        print(f"✓ Features extracted: {len(features_df.columns)} columns added")
        print(f"✓ Total columns: {len(result_df.columns)}")
        
        # Check for NaN values
        n_nan = features_df.isna().sum().sum()
        if n_nan > 0:
            print(f"  ⚠ Warning: {n_nan} NaN values in features")
        
        return result_df
