"""
Command-line interface for MGNN.

Usage:
    mgnn workflow.inp
    mgnn --help
    mgnn --version
"""

import sys
import argparse
from pathlib import Path
from mgnn.config_manager import ConfigManager
from mgnn.data_manager import DataManager


# def main():
#     """Main CLI entry point."""
#     parser = argparse.ArgumentParser(
#         description="Multinary GNN: ML-accelerated perovskite oxide discovery",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#     mgnn workflow.inp                    # Run workflow
#     mgnn --export-plots results/         # Export plots
#     mgnn --summary                       # Show data summary
    
# For more information: https://github.com/rick-sanchez-lab/multinary-gnn
#         """
#     )
    
#     parser.add_argument(
#         "config",
#         nargs="?",
#         type=str,
#         help="Path to workflow configuration file (.inp)"
#     )
    
#     parser.add_argument(
#         "--version",
#         action="version",
#         version="MGNN v0.1.0"
#     )
    
#     parser.add_argument(
#         "--export-plots",
#         type=str,
#         metavar="DIR",
#         help="Export all plots to directory"
#     )
    
#     parser.add_argument(
#         "--summary",
#         action="store_true",
#         help="Show data summary"
#     )
    
#     parser.add_argument(
#         "--list",
#         action="store_true",
#         help="List all stored data"
#     )
    
#     args = parser.parse_args()
    
#     # Handle no arguments
#     if len(sys.argv) == 1:
#         parser.print_help()
#         sys.exit(0)
    
#     # Handle utility commands
#     if args.export_plots:
#         dm = DataManager()
#         dm.export_plots(args.export_plots)
#         return
    
#     if args.export_plots:
#         dm.export_plots(args.export_plots)
#         return
    
#     if args.summary:
#         dm = DataManager()
#         dm.summary()
#         return
    
#     if args.list:
#         dm = DataManager()
#         dm.list_all()
#         return
    
#     # Run workflow
#     if args.config:
#         run_workflow(args.config)
#     else:
#         parser.print_help()

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multinary GNN: ML-accelerated perovskite oxide discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    mgnn workflow.inp                    # Run workflow
    mgnn --export-plots results/         # Export plots as individual files
    mgnn --generate-report report.pdf    # Generate comprehensive PDF report
    mgnn --summary                       # Show data summary
    mgnn --list                          # List all data
    
For more information: https://github.com/rick-sanchez-lab/multinary-gnn
        """
    )
    
    parser.add_argument(
        "config",
        nargs="?",
        type=str,
        help="Path to workflow configuration file (.inp)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="MGNN v0.1.0"
    )
    
    parser.add_argument(
        "--export-plots",
        type=str,
        metavar="DIR",
        help="Export all plots to directory as individual files (SVG, PNG, PDF)"
    )
    
    parser.add_argument(
        "--generate-report",
        type=str,
        metavar="FILE",
        help="Generate comprehensive PDF report with all plots"
    )
    
    parser.add_argument(
        "--data-file",
        type=str,
        default="mgnn_data.h5",
        metavar="FILE",
        help="HDF5 data file to use (default: mgnn_data.h5)"
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show data summary"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all stored data"
    )
    
    args = parser.parse_args()
    
    # Handle no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    # Initialize data manager for utility commands
    if args.export_plots or args.generate_report or args.summary or args.list:
        data_file = args.data_file
        if not Path(data_file).exists():
            # Try default location
            data_file = "./results/mgnn_data.h5"
            if not Path(data_file).exists():
                print(f"Error: Data file not found: {args.data_file}")
                print("Hint: Use --data-file to specify location")
                sys.exit(1)
        
        dm = DataManager(data_file)
    
    # Handle utility commands
    if args.export_plots:
        dm.export_plots(args.export_plots)
        return
    
    if args.generate_report:
        dm.generate_pdf_report(args.generate_report)
        return
    
    if args.summary:
        dm.summary()
        return
    
    if args.list:
        dm.list_all()
        return
    
    # Run workflow
    if args.config:
        run_workflow(args.config)
    else:
        parser.print_help()
        
def run_workflow(config_file: str):
    """Execute workflow based on configuration."""
    print(f"\n{'='*70}")
    print(f"MGNN WORKFLOW EXECUTION")
    print(f"{'='*70}\n")
    
    # Load configuration
    config = ConfigManager(config_file)
    
    # Initialize data manager
    dm = DataManager(config.get('DATA_FILE'))
    
    # Determine workflow type from config
    workflow_type = config.get('WORKFLOW_TYPE', required=True)
    
    # Import and run appropriate workflow
    

    if workflow_type == 'test_data_loader':
        from mgnn.workflows.test_data_loader import run
        run(config, dm)
    
    elif workflow_type == 'data_preparation':
        from mgnn.workflows.data_prep import run
        run(config, dm)
    
    elif workflow_type == 'data_preparation':
        from mgnn.workflows.data_prep import run
        run(config, dm)
    
    elif workflow_type == 'train_baseline':
        from mgnn.workflows.baseline import run
        run(config, dm)
    
    elif workflow_type == 'train_cgcnn':
        from mgnn.workflows.cgcnn_train import run
        run(config, dm)
    
    elif workflow_type == 'transfer_learning':
        from mgnn.workflows.transfer import run
        run(config, dm)
    
    elif workflow_type == 'ensemble_inference':
        from mgnn.workflows.ensemble import run
        run(config, dm)
    
    elif workflow_type == 'dftb_validation':
        from mgnn.workflows.dftb_val import run
        run(config, dm)
    
    elif workflow_type == 'generate_figures':
        from mgnn.workflows.figures import run
        run(config, dm)
    
    elif workflow_type == 'full_pipeline':
        # Run all workflows in sequence
        from mgnn.workflows import run_full_pipeline
        run_full_pipeline(config, dm)
    

    elif workflow_type == 'exploratory_analysis':
        from mgnn.workflows.exploratory_analysis import run
        run(config, dm)

    elif workflow_type == 'feature_engineering':
        from mgnn.workflows.feature_engineering import run
        run(config, dm)

    elif workflow_type == 'model_training':
        from mgnn.workflows.model_training import run
        run(config, dm)
    
    elif workflow_type == 'improved_training':
        from mgnn.workflows.improved_training import run
        run(config, dm)
    
    elif workflow_type == 'augmented_training':
        from mgnn.workflows.augmented_training import run
        run(config, dm)

    elif workflow_type == 'uncertainty_analysis':
        from mgnn.workflows.uncertainty_analysis import run
        run(config, dm)
    
    elif workflow_type == 'ensemble_training':
        from mgnn.workflows.ensemble_training import run
        run(config, dm)
    elif workflow_type == 'ensemble_uncertainty':
        from mgnn.workflows.ensemble_uncertainty import run
        run(config, dm)
    
    elif workflow_type == 'calibration':
        from mgnn.workflows.calibration import run
        run(config, dm)
    
    elif workflow_type == 'shap_analysis':
        from mgnn.workflows.shap_analysis import run
        run(config, dm)
    
    elif workflow_type == 'inverse_design':
        from mgnn.workflows.inverse_design import run
        run(config, dm)

    else:
        raise ValueError(f"Unknown workflow type: {workflow_type}")
    
    print(f"\n{'='*70}")
    print(f"WORKFLOW COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
