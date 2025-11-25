"""Enhanced configuration manager for MGNN with validation and defaults."""

import os
import pprint
from typing import Any, Optional, Union
from pathlib import Path


class ConfigManager:
    """
    Configuration manager that reads ORCA-style input files.
    
    Format:
        !KEY = VALUE  # Active configuration (starts with !)
        KEY = VALUE   # Commented out (no !)
        # Comment line
    
    Example:
        !MP_API_KEY = your_api_key_here
        !OUTPUT_DIR = ./results
        !N_EPOCHS = 200
    """
    
    # Default configuration values
    DEFAULTS = {
        # Paths
        'OUTPUT_DIR': './mgnn_output',
        'DATA_FILE': './mgnn_data.h5',
        'CACHE_DIR': './cache',
        
        # Data settings
        'DATASET_SOURCE': 'multinary_oxides',  # 'multinary_oxides' or 'materials_project'
        'TRAIN_RATIO': 0.70,
        'VAL_RATIO': 0.15,
        'TEST_RATIO': 0.15,
        'RANDOM_SEED': 42,
        
        # Feature engineering
        'N_FEATURES': 42,
        'USE_STRUCTURAL_FEATURES': True,
        'USE_ELECTRONIC_FEATURES': True,
        'USE_COMPOSITIONAL_FEATURES': True,
        
        # Graph construction
        'GRAPH_CUTOFF_RADIUS': 8.0,  # Angstroms
        'GRAPH_K_NEIGHBORS': 12,
        'EDGE_GAUSSIAN_WIDTH': 0.5,
        'N_EDGE_GAUSSIANS': 32,
        
        # Model architecture
        'MODEL_TYPE': 'cgcnn',  # 'cgcnn', 'megnet', 'schnet'
        'HIDDEN_DIM': 128,
        'N_CONV_LAYERS': 4,
        'EDGE_DIM': 64,
        'N_OUTPUT_PROPERTIES': 3,  # ΔHd, Eᵥₒ, Band gap
        
        # Training
        'N_EPOCHS': 200,
        'BATCH_SIZE': 64,
        'LEARNING_RATE': 0.001,
        'WEIGHT_DECAY': 0.0,
        'EARLY_STOPPING_PATIENCE': 20,
        'LOSS_WEIGHTS': [1.0, 0.5, 0.3],  # ΔHd, Eᵥₒ, Band gap
        
        # Transfer learning
        'USE_TRANSFER_LEARNING': True,
        'PRETRAIN_EPOCHS': 100,
        'FINETUNE_LEARNING_RATE': 0.0001,
        'FREEZE_LAYERS': [0, 1],  # Freeze first 2 conv layers
        
        # Ensemble
        'N_ENSEMBLE_MODELS': 5,
        'ENSEMBLE_SEEDS': [42, 123, 456, 789, 1011],
        
        # DFTB validation
        'DFTB_EXECUTABLE': 'dftb+',
        'DFTB_PARAMETER_SET': 'matsci-0-3',
        'DFTB_N_VALIDATION': 50,
        
        # Screening criteria
        'SCREENING_DHD_MAX': 0.1,  # eV/atom
        'SCREENING_EVO_MIN': 1.5,  # eV
        'SCREENING_EVO_MAX': 3.5,  # eV
        'SCREENING_UNCERTAINTY_MAX': 0.12,  # eV
        
        # Analysis
        'UMAP_N_NEIGHBORS': 15,
        'UMAP_MIN_DIST': 0.1,
        'SHAP_N_SAMPLES': 5000,
        
        # Compute
        'DEVICE': 'cuda',  # 'cuda' or 'cpu'
        'N_WORKERS': 4,
        'PIN_MEMORY': True,
    }
    
    def __init__(self, config_filepath: Union[str, Path]):
        """Initialize configuration manager."""
        self.config_filepath = Path(config_filepath)
        self.config = self._read_config_file()
        self._validate_config()
        self._create_directories()
        
        print(f"\n{'='*70}")
        print(f"Configuration Loaded: {self.config_filepath}")
        print(f"{'='*70}\n")
        pprint.pprint(self.config)
        print(f"\n{'='*70}\n")
    
    def _read_config_file(self) -> dict:
        """Read ORCA-style configuration file."""
        if not self.config_filepath.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_filepath}")
        
        config = self.DEFAULTS.copy()  # Start with defaults
        
        with open(self.config_filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and pure comments
                if not line or line.startswith('#'):
                    continue
                
                # Only process lines starting with !
                if line.startswith('!'):
                    line = line[1:]  # Remove !
                    
                    # Split on first = only
                    if '=' not in line:
                        print(f"Warning: Malformed line {line_num}, skipping: {line}")
                        continue
                    
                    key, value = line.split('=', 1)
                    key = key.strip()
                    
                    # Remove inline comments
                    value = value.split('#', 1)[0].strip()
                    
                    # Try to evaluate as Python literal
                    try:
                        value = eval(value)
                    except:
                        # Keep as string if eval fails
                        pass
                    
                    config[key] = value
        
        return config
    
    def _validate_config(self):
        """Validate critical configuration values."""
        # Check ratios sum to 1.0
        ratio_sum = (
            self.config['TRAIN_RATIO'] + 
            self.config['VAL_RATIO'] + 
            self.config['TEST_RATIO']
        )
        if not (0.99 <= ratio_sum <= 1.01):
            raise ValueError(
                f"Train/val/test ratios must sum to 1.0, got {ratio_sum}"
            )
        
        # Check device availability
        if self.config['DEVICE'] == 'cuda':
            import torch
            if not torch.cuda.is_available():
                print("WARNING: CUDA requested but not available, falling back to CPU")
                self.config['DEVICE'] = 'cpu'
    
    def _create_directories(self):
        """Create necessary directories."""
        for key in ['OUTPUT_DIR', 'CACHE_DIR']:
            path = Path(self.config[key])
            path.mkdir(parents=True, exist_ok=True)
    
    def get(
        self, 
        key: str, 
        vartype: Optional[str] = None,
        default: Any = None,
        required: bool = False
    ) -> Any:
        """
        Get configuration value with validation.
        
        Args:
            key: Configuration key
            vartype: Expected type ('directory', 'file', or None)
            default: Default value if not found
            required: If True, raise error if missing
        
        Returns:
            Configuration value
        """
        value = self.config.get(key, default)
        
        # Check if required
        if required and (value is None or value == ""):
            raise ValueError(
                f"Required config key '{key}' is missing. "
                f"Please add '!{key} = VALUE' to {self.config_filepath}"
            )
        
        # Validate file/directory existence
        if vartype in ('directory', 'file') and value is not None:
            path = Path(value)
            
            if vartype == 'directory':
                if not path.is_dir():
                    if required:
                        raise ValueError(
                            f"Config key '{key}' must be a valid directory: {value}"
                        )
                    else:
                        print(f"Warning: Directory '{value}' does not exist")
                        return default
            
            elif vartype == 'file':
                if not path.is_file():
                    if required:
                        raise ValueError(
                            f"Config key '{key}' must be a valid file: {value}"
                        )
                    else:
                        print(f"Warning: File '{value}' does not exist")
                        return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value at runtime."""
        self.config[key] = value
    
    def save(self, filepath: Optional[Union[str, Path]] = None):
        """Save current configuration to file."""
        if filepath is None:
            filepath = self.config_filepath
        else:
            filepath = Path(filepath)
        
        with open(filepath, 'w') as f:
            f.write("# MGNN Configuration File\n")
            f.write(f"# Generated: {__import__('datetime').datetime.now()}\n\n")
            
            for key, value in self.config.items():
                if isinstance(value, str):
                    f.write(f"!{key} = '{value}'\n")
                else:
                    f.write(f"!{key} = {value}\n")
        
        print(f"Configuration saved to: {filepath}")
