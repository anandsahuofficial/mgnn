"""
Test workflow: Data loading and verification.
Purpose: Verify pickle loading, DataFrame creation, and HDF5 storage.
"""

import json
from pathlib import Path
from mgnn.config_manager import ConfigManager
from mgnn.data_manager import DataManager
from mgnn.data.loader import (
    load_pickle_data,
    create_dataframe,
    filter_dataframe,
    verify_data_integrity,
    cross_verify_with_source,
    get_dataset_summary
)


def run(config: ConfigManager, dm: DataManager):
    """
    Run data loader test workflow.
    
    Steps:
    1. Load pickle data
    2. Create DataFrame
    3. Filter data
    4. Verify integrity
    5. Save to HDF5
    6. Generate summary report
    """
    
    print("\n" + "="*70)
    print("TEST WORKFLOW: DATA LOADER")
    print("="*70 + "\n")
    
    # Step 1: Load pickle data
    print("\n[STEP 1/6] Loading pickle data...")
    pickle_file = config.get('PICKLE_FILE', required=True, vartype='file')
    data_list = load_pickle_data(pickle_file)
    
    # Save count to HDF5
    dm.save({
        "test/loader/n_raw_compounds": {
            "data": len(data_list),
            "metadata": {"source": str(pickle_file)}
        }
    })
    
    # Step 2: Create DataFrame
    print("\n[STEP 2/6] Creating DataFrame...")
    df = create_dataframe(data_list, show_progress=True)
    
    # Save initial DataFrame info
    dm.save({
        "test/loader/n_extracted_compounds": {
            "data": len(df),
            "metadata": {"columns": list(df.columns)}
        }
    })
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df[['Unique_ID', 'Formula', 'dH_formation_ev', 'dH_decomposition_ev', 'Perovskite']].head())
    
    # Step 3: Filter data
    print("\n[STEP 3/6] Applying filters...")
    df_filtered, filter_stats = filter_dataframe(
        df,
        perovskites_only=config.get('FILTER_PEROVSKITES_ONLY', default=True),
        min_formation_energy=config.get('MIN_STABILITY', default=None),
        max_decomposition_energy=config.get('MAX_DECOMPOSITION', default=None)
    )
    
    # Convert filter stats to clean format
    filter_stats_clean = {k: int(v) for k, v in filter_stats.items()}
    
    # Save filter statistics
    dm.save({
        "test/loader/filter_stats": {
            "data": json.dumps(filter_stats_clean, indent=2),
            "metadata": filter_stats_clean
        }
    })
    
    # Step 4: Verify integrity
    print("\n[STEP 4/6] Verifying data integrity...")
    integrity_results = verify_data_integrity(df_filtered)
    
    # Convert to clean format
    integrity_clean = {
        'total_compounds': int(integrity_results['total_compounds']),
        'n_missing_value_columns': len(integrity_results['missing_values']),
        'missing_values': {k: int(v) for k, v in integrity_results['missing_values'].items()},
        'n_warnings': len(integrity_results['warnings']),
        'warnings': integrity_results['warnings']
    }
    
    # Save integrity results
    dm.save({
        "test/loader/integrity_check": {
            "data": json.dumps(integrity_clean, indent=2),
            "metadata": {
                'total_compounds': integrity_clean['total_compounds'],
                'n_warnings': integrity_clean['n_warnings']
            }
        }
    })
    
    # Step 5: Cross-verify with source
    print("\n[STEP 5/6] Cross-verifying with source...")
    verification = cross_verify_with_source(
        df_filtered,
        "https://contribs.materialsproject.org/projects/Multinary_Oxides"
    )
    
    # Convert verification dict to have proper types
    verification_clean = {
        'source_url': verification['source_url'],
        'local_compounds': int(verification['local_compounds']),
        'verified': bool(verification['verified']),
        'notes': str(verification['notes']),
        'sanity_checks': {
            k: bool(v) for k, v in verification['sanity_checks'].items()
        }
    }
    
    # Save verification results
    dm.save({
        "test/loader/verification": {
            "data": json.dumps(verification_clean, indent=2),
            "metadata": {
                'n_checks_passed': sum(verification_clean['sanity_checks'].values()),
                'n_checks_total': len(verification_clean['sanity_checks']),
                'verified': verification_clean['verified']
            }
        }
    })
    
    # Step 6: Generate and save summary
    print("\n[STEP 6/6] Generating summary report...")
    summary_text = get_dataset_summary(df_filtered)
    print(summary_text)
    
    # Save summary as text
    dm.save({
        "test/loader/dataset_summary": {
            "data": summary_text,
            "metadata": {
                "total_compounds": int(len(df_filtered)),
                "n_columns": int(len(df_filtered.columns))
            }
        }
    })
    
    # Save the filtered DataFrame (without Structure column for now)
    print("\nSaving DataFrame to HDF5...")
    df_for_save = df_filtered.drop(columns=['Structure']).copy()
    
    # Save key columns
    dm.save({
        "test/loader/dataframe/formulas": {
            "data": df_for_save['Formula'].values,
            "metadata": {"dtype": "string", "description": "Chemical formulas"}
        },
        "test/loader/dataframe/unique_ids": {
            "data": df_for_save['Unique_ID'].values,
            "metadata": {"dtype": "string", "description": "Unique identifiers"}
        },
        "test/loader/dataframe/dH_formation": {
            "data": df_for_save['dH_formation_ev'].values,
            "metadata": {"dtype": "float", "unit": "eV", "description": "Formation energy"}
        },
        "test/loader/dataframe/dH_decomposition": {
            "data": df_for_save['dH_decomposition_ev'].values,
            "metadata": {"dtype": "float", "unit": "eV/atom", "description": "Decomposition energy"}
        },
        "test/loader/dataframe/tolerance_t": {
            "data": df_for_save['tolerance_t'].values,
            "metadata": {"dtype": "float", "description": "Goldschmidt tolerance factor"}
        },
        "test/loader/dataframe/tolerance_tau": {
            "data": df_for_save['tolerance_tau'].values,
            "metadata": {"dtype": "float", "description": "Bartel tolerance factor"}
        }
    })
    
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    print(f"Raw compounds loaded:        {len(data_list)}")
    print(f"Extracted to DataFrame:      {len(df)}")
    print(f"After filtering:             {len(df_filtered)}")
    print(f"Integrity warnings:          {len(integrity_results['warnings'])}")
    print(f"Verification passed:         {verification['verified']}")
    print(f"Sanity checks passed:        {sum(verification_clean['sanity_checks'].values())}/{len(verification_clean['sanity_checks'])}")
    print(f"\nData saved to: {dm.datafilename}")
    print("="*70 + "\n")
    
    print("\nTo view saved data, run:")
    print(f"  ./run.sh --list")
    print(f"  ./run.sh --summary")
    print("")
    
    return df_filtered