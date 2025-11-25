"""
Data loader for multinary perovskite oxides.
Handles pickle loading, DataFrame creation, and initial filtering.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


def load_pickle_data(pickle_file: str) -> List[Dict]:
    """Load data from Cromulons pickle file."""
    print(f"\nLoading pickle data from: {pickle_file}")
    
    with open(pickle_file, 'rb') as f:
        data_list = pickle.load(f)
    
    print(f"✓ Loaded {len(data_list)} compounds")
    return data_list


def extract_tabular_data(data_dict: Dict) -> Dict:
    """
    Extract tabular data from single compound dictionary.
    
    Cromulons data structure:
    {
        'identifier': 'TbY_CoFeO6',
        'formula': 'TbYCoFeO6',
        'id': '6346f814660f1387211b46f5',
        'data': {
            'elements': {'A1': 'Tb', 'A2': 'Y', 'B1': 'Co', 'B2': 'Fe'},
            'oxidation': {
                'A1': {'value': 4.0},
                'A2': {'value': 3.0},
                'B1': {'value': 2.0},
                'B2': {'value': 3.0}
            },
            'tolerance': {
                't': {'value': 0.802457},    # Goldschmidt tolerance
                'tau': {'value': 4.19669}     # Bartel tolerance
            },
            'dH': {
                'formation': {'value': -2.70351, 'unit': 'eV'},
                'decomposition': {'value': 0.0441468, 'unit': 'eV'}
            },
            'perovskite': 'Yes'
        },
        'structures': [Structure object]
    }
    """
    try:
        return {
            "Unique_ID": data_dict['identifier'],
            "Formula": data_dict['formula'],
            "Contribution_ID": data_dict['id'],
            
            # Elements
            "Element_a1": data_dict['data']['elements']['A1'],
            "Element_a2": data_dict['data']['elements']['A2'],
            "Element_b1": data_dict['data']['elements']['B1'],
            "Element_b2": data_dict['data']['elements']['B2'],
            
            # Oxidation states
            "Oxidation_a1": data_dict['data']['oxidation']['A1']['value'],
            "Oxidation_a2": data_dict['data']['oxidation']['A2']['value'],
            "Oxidation_b1": data_dict['data']['oxidation']['B1']['value'],
            "Oxidation_b2": data_dict['data']['oxidation']['B2']['value'],
            
            # Tolerance factors
            "tolerance_t": data_dict['data']['tolerance']['t']['value'],
            "tolerance_tau": data_dict['data']['tolerance']['tau']['value'],
            
            # Energetics (eV for formation, eV for decomposition)
            "dH_formation_ev": data_dict['data']['dH']['formation']['value'],
            "dH_decomposition_ev": data_dict['data']['dH']['decomposition']['value'],
            
            # Classification
            "Perovskite": data_dict['data']['perovskite'],
            
            # Structure (store as object for later processing)
            "Structure": data_dict['structures'][0] if data_dict['structures'] else None
        }
    except KeyError as e:
        print(f"Warning: Missing key {e} for {data_dict.get('identifier', 'unknown')}")
        return None


def create_dataframe(data_list: List[Dict], show_progress: bool = True) -> pd.DataFrame:
    """Convert list of dicts to pandas DataFrame."""
    print("\nExtracting tabular data...")
    
    extracted_data = []
    iterator = tqdm(data_list) if show_progress else data_list
    
    for data_dict in iterator:
        row = extract_tabular_data(data_dict)
        if row is not None:
            extracted_data.append(row)
    
    df = pd.DataFrame(extracted_data)
    print(f"✓ Created DataFrame with {len(df)} rows, {len(df.columns)} columns")
    
    return df


def filter_dataframe(
    df: pd.DataFrame,
    perovskites_only: bool = True,
    min_formation_energy: Optional[float] = None,
    max_decomposition_energy: Optional[float] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Filter DataFrame based on criteria.
    
    Returns:
        Filtered DataFrame and statistics dict
    """
    print("\nApplying filters...")
    original_count = len(df)
    stats = {"original": original_count}
    
    # Filter 1: Perovskite classification
    if perovskites_only:
        df = df[df['Perovskite'] == 'Yes'].copy()
        stats['after_perovskite_filter'] = len(df)
        print(f"  After perovskite filter: {len(df)} ({len(df)/original_count*100:.1f}%)")
    
    # Filter 2: Formation energy (stability)
    if min_formation_energy is not None:
        df = df[df['dH_formation_ev'] >= min_formation_energy].copy()
        stats['after_formation_filter'] = len(df)
        print(f"  After formation energy filter (>= {min_formation_energy}): {len(df)} ({len(df)/original_count*100:.1f}%)")
    
    # Filter 3: Decomposition energy (above hull)
    if max_decomposition_energy is not None:
        df = df[df['dH_decomposition_ev'] <= max_decomposition_energy].copy()
        stats['after_decomposition_filter'] = len(df)
        print(f"  After decomposition filter (<= {max_decomposition_energy}): {len(df)} ({len(df)/original_count*100:.1f}%)")
    
    # Filter 4: Remove rows with missing structures
    df = df[df['Structure'].notna()].copy()
    stats['after_structure_filter'] = len(df)
    print(f"  After structure filter: {len(df)} ({len(df)/original_count*100:.1f}%)")
    
    # Reset index
    df = df.reset_index(drop=True)
    stats['final'] = len(df)
    
    print(f"\n✓ Final dataset: {len(df)} compounds ({len(df)/original_count*100:.1f}% of original)")
    
    return df, stats


def verify_data_integrity(df: pd.DataFrame) -> Dict:
    """
    Verify data integrity and compute statistics.
    
    Returns:
        Dictionary with verification results
    """
    print("\nVerifying data integrity...")
    
    results = {
        "total_compounds": len(df),
        "missing_values": {},
        "element_distribution": {},
        "energy_statistics": {},
        "warnings": []
    }
    
    # Check for missing values
    for col in df.columns:
        if col == 'Structure':
            continue
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            results["missing_values"][col] = n_missing
            results["warnings"].append(f"{col}: {n_missing} missing values")
    
    # Element distribution
    for site in ['a1', 'a2', 'b1', 'b2']:
        col = f'Element_{site}'
        if col in df.columns:
            results["element_distribution"][site] = df[col].value_counts().to_dict()
    
    # Energy statistics
    for col in ['dH_formation_ev', 'dH_decomposition_ev', 'tolerance_t', 'tolerance_tau']:
        if col in df.columns:
            results["energy_statistics"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median())
            }
    
    # Print summary
    print(f"  Total compounds: {results['total_compounds']}")
    print(f"  Missing values: {len(results['missing_values'])} columns affected")
    print(f"  Unique A-site elements: {len(set(df['Element_a1']) | set(df['Element_a2']))}")
    print(f"  Unique B-site elements: {len(set(df['Element_b1']) | set(df['Element_b2']))}")
    
    if results["warnings"]:
        print("\n  Warnings:")
        for warning in results["warnings"]:
            print(f"    ⚠ {warning}")
    else:
        print("  ✓ No integrity issues found")
    
    return results


def cross_verify_with_source(df: pd.DataFrame, source_url: str) -> Dict:
    """
    Cross-verify dataset against original source.
    
    Note: This is a placeholder - full implementation would query Materials Project API
    """
    print(f"\nCross-verification with source: {source_url}")
    print("  (Verification against live API not implemented - using local data only)")
    
    verification = {
        "source_url": source_url,
        "local_compounds": len(df),
        "verified": True,  # Assume verified since we trust Cromulons
        "notes": "Using Cromulons-provided pickle data as ground truth"
    }
    
    # Basic sanity checks
    checks = {
        "formulas_valid": all(df['Formula'].str.contains(r'[A-Z][a-z]?[0-9]*')),
        "energies_reasonable": (
            (df['dH_formation_ev'] < 0).all() and  # Formation should be negative
            (df['dH_decomposition_ev'] >= 0).all()  # Decomposition should be positive
        ),
        "oxidation_states_valid": (
            (df['Oxidation_a1'] > 0).all() and
            (df['Oxidation_a2'] > 0).all() and
            (df['Oxidation_b1'] > 0).all() and
            (df['Oxidation_b2'] > 0).all()
        ),
        "tolerance_factors_reasonable": (
            (df['tolerance_t'] > 0).all() and
            (df['tolerance_t'] < 2).all()  # Physical range
        )
    }
    
    verification['sanity_checks'] = checks
    
    print(f"  ✓ Sanity checks: {sum(checks.values())}/{len(checks)} passed")
    
    return verification


def get_dataset_summary(df: pd.DataFrame) -> str:
    """Generate human-readable dataset summary."""
    summary = f"""
{'='*70}
MULTINARY PEROVSKITE OXIDES DATASET SUMMARY
{'='*70}

Total Compounds: {len(df)}
Perovskite Structures: {(df['Perovskite'] == 'Yes').sum()}

Composition Space:
  A-site elements: {len(set(df['Element_a1']) | set(df['Element_a2']))} unique
  B-site elements: {len(set(df['Element_b1']) | set(df['Element_b2']))} unique
  
Energy Statistics:
  Formation Energy (ΔHf):
    Range: [{df['dH_formation_ev'].min():.3f}, {df['dH_formation_ev'].max():.3f}] eV
    Mean:  {df['dH_formation_ev'].mean():.3f} ± {df['dH_formation_ev'].std():.3f} eV
  
  Decomposition Energy (ΔHd):
    Range: [{df['dH_decomposition_ev'].min():.3f}, {df['dH_decomposition_ev'].max():.3f}] eV/atom
    Mean:  {df['dH_decomposition_ev'].mean():.3f} ± {df['dH_decomposition_ev'].std():.3f} eV/atom
    
Stability:
  Stable (ΔHd ≤ 0.025 eV/atom): {(df['dH_decomposition_ev'] <= 0.025).sum()} ({(df['dH_decomposition_ev'] <= 0.025).sum()/len(df)*100:.1f}%)
  Metastable (0.025 < ΔHd ≤ 0.1): {((df['dH_decomposition_ev'] > 0.025) & (df['dH_decomposition_ev'] <= 0.1)).sum()} ({((df['dH_decomposition_ev'] > 0.025) & (df['dH_decomposition_ev'] <= 0.1)).sum()/len(df)*100:.1f}%)
  Unstable (ΔHd > 0.1): {(df['dH_decomposition_ev'] > 0.1).sum()} ({(df['dH_decomposition_ev'] > 0.1).sum()/len(df)*100:.1f}%)

Tolerance Factors:
  Goldschmidt (t): {df['tolerance_t'].mean():.3f} ± {df['tolerance_t'].std():.3f}
  Bartel (τ): {df['tolerance_tau'].mean():.3f} ± {df['tolerance_tau'].std():.3f}

{'='*70}
"""
    return summary
