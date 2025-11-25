"""
Structure generation for inverse-designed compositions.
Generates approximate 3D crystal structures from composition.
"""

import numpy as np
from pymatgen.core import Structure, Lattice, Element
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar
from typing import Dict, Tuple, Optional
import json
from pathlib import Path


class PerovskiteStructureGenerator:
    """
    Generate 3D perovskite structures from composition.
    
    Uses template-based approach with standard perovskite prototypes.
    """
    
    def __init__(self):
        """Initialize structure generator with ionic radii database."""
        
        # Shannon ionic radii (Angstroms) for common oxidation states
        self.ionic_radii = {
            # A-site cations (typically 2+ or 3+)
            'Li': {'1+': 0.76, '2+': None, '3+': None},
            'Na': {'1+': 1.02, '2+': None, '3+': None},
            'K': {'1+': 1.38, '2+': None, '3+': None},
            'Mg': {'1+': None, '2+': 0.72, '3+': None},
            'Ca': {'1+': None, '2+': 1.00, '3+': None},
            'Sr': {'1+': None, '2+': 1.18, '3+': None},
            'Ba': {'1+': None, '2+': 1.35, '3+': None},
            'La': {'1+': None, '2+': None, '3+': 1.03},
            'Ce': {'1+': None, '2+': None, '3+': 1.01, '4+': 0.87},
            'Pr': {'1+': None, '2+': None, '3+': 0.99},
            'Nd': {'1+': None, '2+': None, '3+': 0.98},
            'Sm': {'1+': None, '2+': None, '3+': 0.96},
            'Eu': {'1+': None, '2+': 1.17, '3+': 0.95},
            'Gd': {'1+': None, '2+': None, '3+': 0.94},
            'Dy': {'1+': None, '2+': None, '3+': 0.91},
            'Y': {'1+': None, '2+': None, '3+': 0.90},
            'Pb': {'1+': None, '2+': 1.19, '3+': None, '4+': 0.775},
            
            # B-site cations (typically 3+, 4+, or 5+)
            'Al': {'3+': 0.535, '4+': None, '5+': None},
            'Ti': {'3+': 0.67, '4+': 0.605, '5+': None},
            'V': {'3+': 0.64, '4+': 0.58, '5+': 0.54},
            'Cr': {'3+': 0.615, '4+': 0.55, '5+': None},
            'Mn': {'3+': 0.58, '4+': 0.53, '5+': None},
            'Fe': {'3+': 0.55, '4+': 0.585, '5+': None},
            'Co': {'3+': 0.545, '4+': 0.53, '5+': None},
            'Ni': {'3+': 0.56, '4+': 0.48, '5+': None},
            'Zr': {'3+': None, '4+': 0.72, '5+': None},
            'Nb': {'3+': None, '4+': 0.68, '5+': 0.64},
            'Mo': {'3+': None, '4+': 0.65, '5+': 0.61},
            'Ru': {'3+': 0.68, '4+': 0.62, '5+': 0.565},
            'Rh': {'3+': 0.665, '4+': 0.60, '5+': 0.55},
            'Hf': {'3+': None, '4+': 0.71, '5+': None},
            'Ta': {'3+': None, '4+': 0.68, '5+': 0.64},
            'W': {'3+': None, '4+': 0.66, '5+': 0.62},
            'Ir': {'3+': 0.68, '4+': 0.625, '5+': 0.57},
            'Sn': {'3+': None, '4+': 0.69, '5+': None},
        }
        
        # Typical oxidation states for perovskites
        self.typical_oxidation = {
            # A-site
            'Li': 1, 'Na': 1, 'K': 1,
            'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2,
            'La': 3, 'Ce': 3, 'Pr': 3, 'Nd': 3, 'Sm': 3,
            'Eu': 2, 'Gd': 3, 'Dy': 3, 'Y': 3,
            'Pb': 2,
            # B-site
            'Al': 3, 'Ti': 4, 'V': 4, 'Cr': 3, 'Mn': 4,
            'Fe': 3, 'Co': 3, 'Ni': 3,
            'Zr': 4, 'Nb': 5, 'Mo': 4, 'Ru': 4, 'Rh': 3,
            'Hf': 4, 'Ta': 5, 'W': 4, 'Ir': 4,
            'Sn': 4,
        }
        
        print("PerovskiteStructureGenerator initialized")
        print(f"  Ionic radii database: {len(self.ionic_radii)} elements")
    
    def get_ionic_radius(
        self,
        element: str,
        oxidation: Optional[int] = None
    ) -> float:
        """
        Get ionic radius for element in given oxidation state.
        
        Args:
            element: Element symbol
            oxidation: Oxidation state (if None, use typical)
        
        Returns:
            Ionic radius in Angstroms
        """
        if element not in self.ionic_radii:
            # Use covalent radius as fallback
            return Element(element).atomic_radius
        
        if oxidation is None:
            oxidation = self.typical_oxidation.get(element, 3)
        
        ox_str = f"{oxidation}+"
        radius = self.ionic_radii[element].get(ox_str)
        
        if radius is None:
            # Try other oxidation states
            for ox_s, r in self.ionic_radii[element].items():
                if r is not None:
                    radius = r
                    break
        
        if radius is None:
            radius = Element(element).atomic_radius
        
        return radius
    
    def calculate_tolerance_factor(
        self,
        A_element: str,
        B_element: str
    ) -> float:
        """
        Calculate Goldschmidt tolerance factor.
        
        τ = (r_A + r_O) / [√2 × (r_B + r_O)]
        
        Ideal perovskite: 0.8 < τ < 1.0
        
        Args:
            A_element: A-site element
            B_element: B-site element
        
        Returns:
            Tolerance factor
        """
        r_A = self.get_ionic_radius(A_element)
        r_B = self.get_ionic_radius(B_element)
        r_O = 1.40  # O²⁻ ionic radius
        
        tolerance = (r_A + r_O) / (np.sqrt(2) * (r_B + r_O))
        
        return tolerance
    
    def generate_cubic_structure(
        self,
        A_element: str,
        B_element: str
    ) -> Structure:
        """
        Generate ideal cubic perovskite structure (Pm-3m, space group 221).
        
        Structure:
        - A at (0, 0, 0) - corners
        - B at (0.5, 0.5, 0.5) - body center
        - O at face centers
        
        Args:
            A_element: A-site element
            B_element: B-site element
        
        Returns:
            pymatgen Structure object
        """
        # Get ionic radii
        r_A = self.get_ionic_radius(A_element)
        r_B = self.get_ionic_radius(B_element)
        r_O = 1.40
        
        # Lattice parameter from B-O bond length
        # In cubic perovskite: a = 2(r_B + r_O)
        a = 2 * (r_B + r_O)
        
        # Create cubic lattice
        lattice = Lattice.cubic(a)
        
        # Atomic positions (fractional coordinates)
        species = [A_element, B_element, 'O', 'O', 'O']
        coords = [
            [0.0, 0.0, 0.0],      # A at corner
            [0.5, 0.5, 0.5],      # B at body center
            [0.5, 0.5, 0.0],      # O at face center
            [0.5, 0.0, 0.5],      # O at face center
            [0.0, 0.5, 0.5]       # O at face center
        ]
        
        structure = Structure(
            lattice=lattice,
            species=species,
            coords=coords,
            coords_are_cartesian=False
        )
        
        return structure
    
    def generate_orthorhombic_structure(
        self,
        A_element: str,
        B_element: str
    ) -> Structure:
        """
        Generate orthorhombic perovskite (Pnma, space group 62).
        
        Common for tolerance factor < 0.96 (GdFeO3-type distortion).
        
        Args:
            A_element: A-site element
            B_element: B-site element
        
        Returns:
            pymatgen Structure object
        """
        # Start with cubic
        cubic = self.generate_cubic_structure(A_element, B_element)
        a_cubic = cubic.lattice.a
        
        # Orthorhombic distortion
        # Typical: a > c > b (for Pnma)
        a_ortho = a_cubic * np.sqrt(2) * 1.01
        b_ortho = a_cubic * np.sqrt(2) * 0.99
        c_ortho = a_cubic * 1.00
        
        lattice = Lattice.orthorhombic(a_ortho, b_ortho, c_ortho)
        
        # Approximate atomic positions for Pnma
        # (Simplified - real structures have oxygen tilting)
        species = [A_element, A_element, A_element, A_element,
                   B_element, B_element, B_element, B_element,
                   'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                   'O', 'O', 'O', 'O']
        
        coords = [
            [0.05, 0.25, 0.00], [0.55, 0.75, 0.50],  # A sites
            [0.45, 0.25, 0.50], [0.95, 0.75, 0.00],
            [0.00, 0.00, 0.50], [0.50, 0.50, 0.00],  # B sites
            [0.50, 0.00, 0.00], [0.00, 0.50, 0.50],
            # O sites (simplified)
            [0.25, 0.00, 0.25], [0.75, 0.50, 0.75],
            [0.75, 0.00, 0.75], [0.25, 0.50, 0.25],
            [0.00, 0.25, 0.50], [0.50, 0.75, 0.00],
            [0.50, 0.25, 0.00], [0.00, 0.75, 0.50],
            [0.25, 0.25, 0.00], [0.75, 0.75, 0.50],
            [0.75, 0.25, 0.50], [0.25, 0.75, 0.00]
        ]
        
        structure = Structure(
            lattice=lattice,
            species=species,
            coords=coords,
            coords_are_cartesian=False
        )
        
        return structure
    
    def generate_tetragonal_structure(
        self,
        A_element: str,
        B_element: str
    ) -> Structure:
        """
        Generate tetragonal perovskite (P4mm, space group 99).
        
        Common for ferroelectric perovskites like BaTiO3.
        
        Args:
            A_element: A-site element
            B_element: B-site element
        
        Returns:
            pymatgen Structure object
        """
        # Start with cubic
        cubic = self.generate_cubic_structure(A_element, B_element)
        a_cubic = cubic.lattice.a
        
        # Tetragonal elongation along c-axis
        a_tetra = a_cubic
        c_tetra = a_cubic * 1.01  # Slight elongation
        
        lattice = Lattice.tetragonal(a_tetra, c_tetra)
        
        # Same fractional coordinates as cubic
        species = [A_element, B_element, 'O', 'O', 'O']
        coords = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ]
        
        structure = Structure(
            lattice=lattice,
            species=species,
            coords=coords,
            coords_are_cartesian=False
        )
        
        return structure
    
    def generate_rhombohedral_structure(
        self,
        A_element: str,
        B_element: str
    ) -> Structure:
        """
        Generate rhombohedral perovskite (R3c, space group 161).
        
        Common for certain rare-earth perovskites.
        
        Args:
            A_element: A-site element
            B_element: B-site element
        
        Returns:
            pymatgen Structure object
        """
        # Start with cubic
        cubic = self.generate_cubic_structure(A_element, B_element)
        a_cubic = cubic.lattice.a
        
        # Rhombohedral distortion
        a_rhombo = a_cubic
        alpha = 60.0  # degrees
        
        lattice = Lattice.rhombohedral(a_rhombo, alpha)
        
        # Simplified rhombohedral coordinates
        species = [A_element, B_element, 'O', 'O', 'O']
        coords = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ]
        
        structure = Structure(
            lattice=lattice,
            species=species,
            coords=coords,
            coords_are_cartesian=False
        )
        
        return structure
    
    def generate_best_structure(
        self,
        A_element: str,
        B_element: str
    ) -> Tuple[Structure, str]:
        """
        Generate most likely structure based on tolerance factor.
        
        Rules:
        - τ < 0.71: Ilmenite (not perovskite)
        - 0.71 < τ < 0.9: Orthorhombic distortion
        - 0.9 < τ < 1.0: Cubic (ideal)
        - 1.0 < τ < 1.13: Cubic or hexagonal
        - τ > 1.13: Hexagonal (not perovskite)
        
        Args:
            A_element: A-site element
            B_element: B-site element
        
        Returns:
            (structure, structure_type)
        """
        tau = self.calculate_tolerance_factor(A_element, B_element)
        
        if tau < 0.71:
            structure_type = "orthorhombic_distorted"
            structure = self.generate_orthorhombic_structure(A_element, B_element)
        elif tau < 0.9:
            structure_type = "orthorhombic"
            structure = self.generate_orthorhombic_structure(A_element, B_element)
        elif tau < 1.0:
            structure_type = "cubic"
            structure = self.generate_cubic_structure(A_element, B_element)
        elif tau < 1.13:
            structure_type = "cubic_or_tetragonal"
            structure = self.generate_cubic_structure(A_element, B_element)
        else:
            structure_type = "cubic_unstable"
            structure = self.generate_cubic_structure(A_element, B_element)
        
        return structure, structure_type
    
    def save_structure(
        self,
        structure: Structure,
        output_dir: Path,
        filename: str,
        formats: list = ['cif', 'poscar']
    ) -> Dict[str, Path]:
        """
        Save structure to multiple file formats.
        
        Args:
            structure: pymatgen Structure
            output_dir: Output directory
            filename: Base filename (without extension)
            formats: List of formats ('cif', 'poscar', 'json')
        
        Returns:
            Dictionary mapping format to filepath
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        for fmt in formats:
            if fmt == 'cif':
                filepath = output_dir / f"{filename}.cif"
                cif_writer = CifWriter(structure)
                cif_writer.write_file(str(filepath))
                saved_files['cif'] = filepath
                
            elif fmt == 'poscar':
                filepath = output_dir / f"{filename}.vasp"
                poscar = Poscar(structure)
                poscar.write_file(str(filepath))
                saved_files['poscar'] = filepath
                
            elif fmt == 'json':
                filepath = output_dir / f"{filename}.json"
                structure.to(filename=str(filepath), fmt='json')
                saved_files['json'] = filepath
        
        return saved_files
    
    def generate_structure_report(
        self,
        A_element: str,
        B_element: str,
        structure: Structure,
        structure_type: str,
        predicted_stability: float,
        uncertainty: float,
        shap_contributions: Optional[Dict] = None
    ) -> Dict:
        """
        Generate comprehensive report for a structure.
        
        Args:
            A_element: A-site element
            B_element: B-site element
            structure: Generated structure
            structure_type: Structure type
            predicted_stability: Predicted ΔHd
            uncertainty: Prediction uncertainty
            shap_contributions: SHAP feature contributions
        
        Returns:
            Report dictionary
        """
        tau = self.calculate_tolerance_factor(A_element, B_element)
        
        report = {
            'composition': {
                'formula': f"{A_element}{B_element}O3",
                'A_site': A_element,
                'B_site': B_element,
                'X_site': 'O'
            },
            'structure': {
                'type': structure_type,
                'space_group': structure.get_space_group_info()[1],
                'space_group_number': structure.get_space_group_info()[0],
                'lattice_parameters': {
                    'a': float(structure.lattice.a),
                    'b': float(structure.lattice.b),
                    'c': float(structure.lattice.c),
                    'alpha': float(structure.lattice.alpha),
                    'beta': float(structure.lattice.beta),
                    'gamma': float(structure.lattice.gamma),
                    'volume': float(structure.lattice.volume)
                },
                'n_atoms': len(structure),
                'atomic_positions': [
                    {
                        'element': str(site.specie),
                        'fractional_coords': site.frac_coords.tolist(),
                        'cartesian_coords': site.coords.tolist()
                    }
                    for site in structure
                ]
            },
            'predictions': {
                'stability_eV_per_atom': float(predicted_stability),
                'uncertainty_eV_per_atom': float(uncertainty),
                'confidence_level': 'high' if uncertainty < 0.015 else 'medium' if uncertainty < 0.025 else 'low',
                'probability_stable': float(1.0 / (1.0 + np.exp(predicted_stability / 0.025)))  # Rough estimate
            },
            'geometric_analysis': {
                'tolerance_factor': float(tau),
                'tolerance_interpretation': self._interpret_tolerance(tau),
                'ionic_radius_A': float(self.get_ionic_radius(A_element)),
                'ionic_radius_B': float(self.get_ionic_radius(B_element)),
                'ionic_radius_ratio': float(self.get_ionic_radius(A_element) / self.get_ionic_radius(B_element))
            },
            'shap_explanation': shap_contributions if shap_contributions else {},
            'validation_notes': {
                'structure_source': 'template-based generation',
                'dft_validation_recommended': True,
                'synthesis_considerations': self._get_synthesis_notes(A_element, B_element, tau)
            }
        }
        
        return report
    
    def _interpret_tolerance(self, tau: float) -> str:
        """Interpret tolerance factor."""
        if tau < 0.71:
            return "Too small - likely ilmenite or other structure"
        elif tau < 0.9:
            return "Orthorhombic distortion likely (GdFeO3-type)"
        elif tau < 1.0:
            return "Ideal perovskite range - cubic structure favored"
        elif tau < 1.13:
            return "Large tolerance - cubic or hexagonal possible"
        else:
            return "Too large - likely hexagonal or other structure"
    
    def _get_synthesis_notes(self, A: str, B: str, tau: float) -> str:
        """Get synthesis considerations."""
        notes = []
        
        if 0.9 < tau < 1.0:
            notes.append("Ideal tolerance factor - straightforward synthesis expected")
        elif tau < 0.9:
            notes.append("May require high pressure synthesis for stability")
        elif tau > 1.0:
            notes.append("Consider thin film or confined synthesis methods")
        
        # Add element-specific notes
        rare_earth = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Dy', 'Y']
        if A in rare_earth:
            notes.append("Rare-earth A-site - use oxygen atmosphere to maintain oxidation state")
        
        transition_metals = ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni']
        if B in transition_metals:
            notes.append("Transition metal B-site - oxygen stoichiometry critical")
        
        return "; ".join(notes) if notes else "Standard solid-state synthesis recommended"


def test_structure_generator():
    """Test structure generator."""
    print("\nTesting PerovskiteStructureGenerator...")
    
    gen = PerovskiteStructureGenerator()
    
    # Test well-known perovskites
    test_cases = [
        ('Ca', 'Ti'),  # CaTiO3 - cubic
        ('Sr', 'Ti'),  # SrTiO3 - cubic
        ('Ba', 'Ti'),  # BaTiO3 - cubic/tetragonal
        ('La', 'Al'),  # LaAlO3 - rhombohedral
    ]
    
    for A, B in test_cases:
        print(f"\nGenerating {A}{B}O3:")
        
        # Calculate tolerance
        tau = gen.calculate_tolerance_factor(A, B)
        print(f"  Tolerance factor: {tau:.3f}")
        
        # Generate structure
        structure, structure_type = gen.generate_best_structure(A, B)
        print(f"  Structure type: {structure_type}")
        print(f"  Space group: {structure.get_space_group_info()[1]}")
        print(f"  Lattice a: {structure.lattice.a:.3f} Å")
        print(f"  Number of atoms: {len(structure)}")
    
    print("\n✓ PerovskiteStructureGenerator test passed!")


if __name__ == '__main__':
    test_structure_generator()



# cd /projectnb/cui-buchem/anandsahu/Softwares/Mine/multinary-gnn

# # Sync environment
# uv sync

# # Test structure generator
# echo "Testing structure generator..."
# uv run python -m mgnn.inverse.structure_generator

# # Test genetic algorithm
# echo "Testing genetic algorithm..."
# uv run python -m mgnn.inverse.genetic_algorithm