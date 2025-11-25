"""
Exploratory Data Analysis Workflow.
Purpose: Comprehensive visualization and statistical analysis.
"""

import json
from pathlib import Path
from mgnn.config_manager import ConfigManager
from mgnn.data_manager import DataManager
from mgnn.data.loader import (
    load_pickle_data,
    create_dataframe,
    filter_dataframe
)
from mgnn.analysis.eda import ExploratoryAnalysis


def run(config: ConfigManager, dm: DataManager):
    """
    Run exploratory data analysis workflow.
    
    Steps:
    1. Load and filter data
    2. Generate energy distribution plots
    3. Generate stability analysis plots
    4. Generate element analysis plots
    5. Generate correlation analysis plots
    6. Compute summary statistics
    7. Save all results to HDF5
    8. Export plots if requested
    """
    
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS WORKFLOW")
    print("="*70 + "\n")
    
    # Step 1: Load and filter data
    print("\n[STEP 1/7] Loading and filtering data...")
    pickle_file = config.get('PICKLE_FILE', required=True, vartype='file')
    data_list = load_pickle_data(pickle_file)
    df = create_dataframe(data_list, show_progress=True)
    
    df_filtered, filter_stats = filter_dataframe(
        df,
        perovskites_only=config.get('FILTER_PEROVSKITES_ONLY', default=True),
        min_formation_energy=config.get('MIN_STABILITY', default=None),
        max_decomposition_energy=config.get('MAX_DECOMPOSITION', default=None)
    )
    
    print(f"Filtered dataset: {len(df_filtered)} compounds")
    
    # Initialize EDA analyzer
    eda = ExploratoryAnalysis(
        df_filtered,
        style=config.get('STYLE', default='seaborn-v0_8-darkgrid'),
        dpi=config.get('FIGURE_DPI', default=300)
    )
    
    # Step 2: Energy distributions
    print("\n[STEP 2/7] Generating energy distribution plots...")
    energy_figs = eda.plot_energy_distributions()
    
    for name, fig in energy_figs.items():
        dm.save({
            f"eda/figures/{name}": {
                "data": fig,
                "metadata": {
                    "plot_type": "energy_distribution",
                    "description": f"Energy distribution analysis: {name}"
                }
            }
        })
    
    # Step 3: Stability analysis
    print("\n[STEP 3/7] Generating stability analysis plots...")
    stability_figs = eda.plot_stability_analysis()
    
    for name, fig in stability_figs.items():
        dm.save({
            f"eda/figures/{name}": {
                "data": fig,
                "metadata": {
                    "plot_type": "stability_analysis",
                    "description": f"Stability analysis: {name}"
                }
            }
        })
    
    # Step 4: Element analysis
    print("\n[STEP 4/7] Generating element analysis plots...")
    element_figs = eda.plot_element_analysis(
        top_n=config.get('TOP_N_ELEMENTS', default=20)
    )
    
    for name, fig in element_figs.items():
        dm.save({
            f"eda/figures/{name}": {
                "data": fig,
                "metadata": {
                    "plot_type": "element_analysis",
                    "description": f"Element analysis: {name}"
                }
            }
        })
    
    # Step 5: Correlation analysis
    print("\n[STEP 5/7] Generating correlation analysis plots...")
    correlation_figs = eda.plot_correlation_analysis(
        min_corr=config.get('MIN_CORRELATION_DISPLAY', default=0.3)
    )
    
    for name, fig in correlation_figs.items():
        dm.save({
            f"eda/figures/{name}": {
                "data": fig,
                "metadata": {
                    "plot_type": "correlation_analysis",
                    "description": f"Correlation analysis: {name}"
                }
            }
        })
    
    # Step 6: Summary statistics
    print("\n[STEP 6/7] Computing summary statistics...")
    summary_stats = eda.generate_summary_statistics()
    
    dm.save({
        "eda/summary_statistics": {
            "data": json.dumps(summary_stats, indent=2),
            "metadata": {
                "n_compounds": len(df_filtered),
                "n_properties": len(summary_stats)
            }
        }
    })
    
    # Step 7: Export plots if requested
    if config.get('EXPORT_PLOTS', default=False):
        print("\n[STEP 7/7] Exporting plots to files...")
        plot_dir = config.get('PLOT_OUTPUT_DIR', default='./plots/eda')
        dm.export_plots(output_dir=plot_dir, include_archived=False)
    else:
        print("\n[STEP 7/7] Plot export skipped (EXPORT_PLOTS=False)")
    
    # Step 8: Generate PDF report (always do this)
    print("\n[STEP 8/8] Generating comprehensive PDF report...")
    pdf_path = dm.generate_pdf_report(
        output_file=config.get('OUTPUT_DIR', default='./results') + '/eda_report.pdf',
        include_archived=False,
        title="Exploratory Data Analysis Report",
        author="MGNN Pipeline"
    )
    
    # Final summary
    # total_figs = len(energy_figs) + len(stability_figs) + len(element_figs) + len(correlation_figs)
    
    # print("\n" + "="*70)
    # print("EDA WORKFLOW COMPLETE")
    # print("="*70)
    # print(f"Total figures generated: {total_figs}")
    # print(f"Figures saved to: {dm.datafilename}")
    # print(f"Summary statistics computed: {len(summary_stats)} categories")
    
    # if config.get('EXPORT_PLOTS', default=False):
    #     print(f"Plots exported to: {config.get('PLOT_OUTPUT_DIR')}")
    
    # print("\nTo view figures:")
    # print("  ./run.sh --list")
    # print("  ./run.sh --export-plots ./plots/eda")
    # print("="*70 + "\n")
    
    # return df_filtered, summary_stats

    # Final summary
    total_figs = len(energy_figs) + len(stability_figs) + len(element_figs) + len(correlation_figs)

    print("\n" + "="*70)
    print("EDA WORKFLOW COMPLETE")
    print("="*70)
    print(f"Total figures generated: {total_figs}")
    print(f"Figures saved to: {dm.datafilename}")
    print(f"PDF report saved to: {pdf_path}")
    print(f"Summary statistics computed: {len(summary_stats)} categories")

    if config.get('EXPORT_PLOTS', default=False):
        print(f"Individual plots exported to: {config.get('PLOT_OUTPUT_DIR')}")

    print("\nTo view results:")
    print(f"  Open PDF: {pdf_path}")
    print(f"  List data: ./run.sh --list --data-file {dm.datafilename}")
    print(f"  Generate new report: ./run.sh --generate-report custom_report.pdf --data-file {dm.datafilename}")
    print("="*70 + "\n")

    return df_filtered, summary_stats