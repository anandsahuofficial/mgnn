"""Enhanced HDF5 data manager for MGNN with plot storage."""

import os
import io
import json
import h5py
import cairosvg
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import SVG, display


class DataManager:
    """
    HDF5-backed data manager for MGNN.
    
    Features:
    - Automatic versioning and archiving
    - Stores arrays, scalars, metadata, and matplotlib figures
    - Single .h5 file for entire project
    - SVG and PGF export for publications
    - NumPy 2.0 compatible
    """
    
    def __init__(self, datafilename: Union[str, Path] = "mgnn_data.h5"):
        """Initialize data manager."""
        self.datafilename = Path(datafilename)
        
        # Create parent directory if needed
        self.datafilename.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize file if it doesn't exist
        if not self.datafilename.exists():
            with h5py.File(self.datafilename, 'w') as f:
                f.attrs['created_at'] = str(datetime.now())
                f.attrs['mgnn_version'] = '0.1.0'
        
        print(f"DataManager initialized: {self.datafilename}")
    
    def save(self, data: Dict[str, Dict[str, Any]], overwrite: bool = False):
        """
        Save data to HDF5 file.
        
        Args:
            data: Dictionary with structure:
                  {"path/to/data": {"data": <data>, "metadata": {...}}}
            overwrite: If True, overwrite existing data; if False, archive
        
        Example:
            dm.save({
                "training/loss": {
                    "data": np.array([1.2, 0.9, 0.7]),
                    "metadata": {"epoch": 100, "model": "cgcnn"}
                },
                "figures/parity_plot": {
                    "data": matplotlib_figure,
                    "metadata": {"property": "formation_energy"}
                }
            })
        """
        with h5py.File(self.datafilename, 'a') as f:
            f.attrs['last_updated'] = str(datetime.now())
            
            for datapath, datavalue in data.items():
                data_obj = datavalue.get("data", None)
                metadata = datavalue.get("metadata", {})
                
                # Add automatic metadata
                metadata['data_type'] = type(data_obj).__name__
                metadata['saved_at'] = str(datetime.now())
                
                self._save_single(f, datapath, data_obj, metadata, overwrite)
    
    def _save_single(
        self, 
        f: h5py.File,
        datapath: str, 
        data: Any,
        metadata: dict,
        overwrite: bool
    ):
        """Save a single data object with NumPy 2.0 compatibility."""
        parts = datapath.strip("/").split("/")
        dset_name = parts[-1]
        group_parts = parts[:-1]
        
        # Create parent groups
        grp = f["/"]
        for p in group_parts:
            if p in grp:
                if isinstance(grp[p], h5py.Dataset):
                    # Archive conflicting dataset
                    self._archive_dataset(f, grp[p].name)
                    grp = grp.create_group(p)
                else:
                    grp = grp[p]
            else:
                grp = grp.create_group(p)
        
        fullpath = f"{grp.name}/{dset_name}".replace("//", "/")
        
        # Handle existing data
        if fullpath in f and not overwrite:
            self._archive_dataset(f, fullpath)
        elif fullpath in f and overwrite:
            del f[fullpath]
        
        # Handle matplotlib figures
        if type(data).__name__ == "Figure":
            self._save_figure(f, fullpath, data, metadata)
            return
        
        # Prepare data for HDF5 storage
        processed_data, use_compression = self._prepare_data_for_hdf5(data)
        
        # Create dataset with appropriate options
        try:
            if use_compression:
                dset = f.create_dataset(
                    fullpath,
                    data=processed_data,
                    compression="gzip",
                    chunks=True,
                    fletcher32=True
                )
            else:
                # Scalar or special type - no compression/chunks
                dset = f.create_dataset(
                    fullpath,
                    data=processed_data
                )
            
            # Save metadata as attributes with proper type conversion
            for key, value in metadata.items():
                try:
                    # Convert value to JSON-serializable type
                    json_value = self._make_json_serializable(value)
                    
                    if isinstance(json_value, (dict, list)):
                        dset.attrs[key] = json.dumps(json_value)
                    elif isinstance(json_value, (int, float, bool, str)):
                        dset.attrs[key] = json_value
                    elif json_value is None:
                        dset.attrs[key] = "null"
                    else:
                        dset.attrs[key] = str(json_value)
                        
                except Exception as e:
                    print(f"Warning: Could not save metadata '{key}': {e}")
            
            print(f"Saved: {fullpath}")
            
        except Exception as e:
            print(f"Error saving {fullpath}: {e}")
            print(f"  Data type: {type(data)}")
            if hasattr(data, 'dtype'):
                print(f"  Data dtype: {data.dtype}")
            if hasattr(data, 'shape'):
                print(f"  Data shape: {data.shape}")

    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON-serializable type.
        Handles NumPy types, nested structures, etc.
        """
        # Handle None
        if obj is None:
            return None
        
        # Handle NumPy scalar types
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, 
                            np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        
        if isinstance(obj, (np.floating, np.float_, np.float16, 
                            np.float32, np.float64)):
            return float(obj)
        
        if isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        
        if isinstance(obj, (np.str_, np.unicode_)):
            return str(obj)
        
        # Handle NumPy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle dictionaries recursively
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(val) for key, val in obj.items()}
        
        # Handle lists/tuples recursively
        if isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        
        # Handle basic Python types
        if isinstance(obj, (int, float, bool, str)):
            return obj
        
        # Everything else - convert to string
        return str(obj)

    def _prepare_data_for_hdf5(self, data: Any) -> tuple:
        """
        Prepare data for HDF5 storage with NumPy 2.0 compatibility.
        
        Returns:
            (processed_data, use_compression)
        """
        # Handle None
        if data is None:
            return np.array([]), False
        
        # Handle Python scalars
        if isinstance(data, (int, float, bool)):
            return data, False
        
        # Handle Python strings
        if isinstance(data, str):
            # Use variable-length string type
            return data, False
        
        # Handle NumPy scalars
        if np.isscalar(data):
            return data, False
        
        # Handle NumPy arrays
        if isinstance(data, np.ndarray):
            # String/object arrays need special handling
            if data.dtype.kind in ('U', 'O', 'S'):
                # For string arrays, we need to avoid compression/chunking
                # Convert to fixed-length or variable-length strings
                if len(data) > 0:
                    # Use variable-length string dtype (h5py special)
                    str_dtype = h5py.string_dtype(encoding='utf-8')
                    
                    # Convert array elements to strings
                    str_array = np.array([str(x) for x in data], dtype=object)
                    
                    # Create a special dataset that h5py can handle
                    # We'll return it without compression to avoid the filter error
                    return str_array.astype(str_dtype), False  # NO compression for strings
                else:
                    return np.array([], dtype=h5py.string_dtype(encoding='utf-8')), False
            
            # Numeric arrays - can use compression
            if data.dtype.kind in ('i', 'u', 'f', 'c'):
                return data, True
            
            # Boolean arrays
            if data.dtype.kind == 'b':
                return data.astype(np.int8), True
            
            # Other types - convert to string array without compression
            str_array = np.array([str(x) for x in data], dtype=object)
            return str_array.astype(h5py.string_dtype(encoding='utf-8')), False
        
        # Handle lists
        if isinstance(data, (list, tuple)):
            # Try to convert to numpy array
            try:
                arr = np.array(data)
                return self._prepare_data_for_hdf5(arr)
            except:
                # If conversion fails, store as string
                return str(data), False
        
        # Handle dicts and other objects - convert to string
        return str(data), False
    
        
    def _archive_dataset(self, f: h5py.File, fullpath: str):
        """Archive existing dataset with versioning."""
        parts = fullpath.strip("/").split("/")
        archive_base = f"/archived/{'/'.join(parts)}"
        
        # Create archive group
        archive_grp = f.require_group(archive_base)
        
        # Find next version number
        existing_versions = [
            int(k.split("_")[0][1:]) 
            for k in archive_grp.keys() 
            if k.startswith("v")
        ]
        next_version = max(existing_versions, default=0) + 1
        
        # Move to archive
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"v{next_version}_{timestamp}"
        f.move(fullpath, f"{archive_grp.name}/{archive_name}")
        
        print(f"Archived: {fullpath} -> {archive_grp.name}/{archive_name}")
    
    def _save_figure(self, f: h5py.File, fullpath: str, fig, metadata: dict):
        """
        Save matplotlib figure as SVG and PGF (with fallback).
        
        PGF backend doesn't support raster graphics (scatter plots with colormaps).
        We'll try to save PGF, but gracefully fall back to SVG-only if it fails.
        """
        fig_grp = f.require_group(fullpath)
        
        formats_saved = []
        
        # Always try SVG first (most compatible)
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='svg', bbox_inches='tight')
            buf.seek(0)
            
            if 'svg' in fig_grp:
                del fig_grp['svg']
            
            dset = fig_grp.create_dataset(
                'svg',
                data=np.void(buf.getvalue())
            )
            
            # Save metadata
            for key, value in metadata.items():
                try:
                    if isinstance(value, (dict, list)):
                        dset.attrs[key] = json.dumps(value)
                    else:
                        dset.attrs[key] = value
                except:
                    pass
            
            buf.close()
            formats_saved.append('svg')
            
        except Exception as e:
            print(f"Warning: Failed to save SVG for {fullpath}: {e}")
        
        # Try PGF (may fail for raster graphics)
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='pgf', bbox_inches='tight')
            buf.seek(0)
            
            if 'pgf' in fig_grp:
                del fig_grp['pgf']
            
            dset = fig_grp.create_dataset(
                'pgf',
                data=np.void(buf.getvalue())
            )
            
            # Save metadata
            for key, value in metadata.items():
                try:
                    if isinstance(value, (dict, list)):
                        dset.attrs[key] = json.dumps(value)
                    else:
                        dset.attrs[key] = value
                except:
                    pass
            
            buf.close()
            formats_saved.append('pgf')
            
        except ValueError as e:
            if "pgf-code does not support raster graphics" in str(e):
                print(f"Info: Skipping PGF for {fullpath} (contains raster graphics)")
            else:
                print(f"Warning: Failed to save PGF for {fullpath}: {e}")
        except Exception as e:
            print(f"Warning: Failed to save PGF for {fullpath}: {e}")
        
        # Always try PDF as well (good for manuscripts)
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='pdf', bbox_inches='tight')
            buf.seek(0)
            
            if 'pdf' in fig_grp:
                del fig_grp['pdf']
            
            dset = fig_grp.create_dataset(
                'pdf',
                data=np.void(buf.getvalue())
            )
            
            buf.close()
            formats_saved.append('pdf')
            
        except Exception as e:
            print(f"Warning: Failed to save PDF for {fullpath}: {e}")
        
        if formats_saved:
            print(f"Saved figure: {fullpath} [{', '.join(formats_saved)}]")
        else:
            print(f"Error: Failed to save figure {fullpath} in any format")
        
        plt.close(fig)

    def load(self, datapath: str) -> Any:
        """Load data from HDF5 file."""
        with h5py.File(self.datafilename, 'r') as f:
            if datapath not in f:
                raise KeyError(f"Data not found: {datapath}")
            
            obj = f[datapath]
            
            # Handle groups (figures)
            if isinstance(obj, h5py.Group):
                if 'svg' in obj:
                    # Return as SVG bytes
                    return bytes(obj['svg'][()])
                else:
                    raise ValueError(f"{datapath} is a group, not a dataset")
            
            # Handle datasets
            data = obj[()]
            
            # Convert np.void to bytes if needed
            if isinstance(data, np.void):
                data = bytes(data)
            
            return data
    
    def list_all(self, pattern: Optional[str] = None):
        """List all datasets and groups."""
        print(f"\nContents of {self.datafilename}:\n")
        
        with h5py.File(self.datafilename, 'r') as f:
            def visitor(name, obj):
                if pattern and pattern not in name:
                    return
                
                if isinstance(obj, h5py.Group):
                    # Check if it's a figure group
                    formats = []
                    if 'svg' in obj:
                        formats.append('svg')
                    if 'pgf' in obj:
                        formats.append('pgf')
                    if 'pdf' in obj:
                        formats.append('pdf')
                    
                    if formats:
                        print(f"[FIGURE] {name} [{', '.join(formats)}]")
                    else:
                        print(f"[GROUP]  {name}")
                elif isinstance(obj, h5py.Dataset):
                    shape_str = str(obj.shape) if obj.shape else "(scalar)"
                    print(f"[DATA]   {name}  shape={shape_str}, dtype={obj.dtype}")
            
            f.visititems(visitor)
        
    def export_plots(
        self, 
        output_dir: Union[str, Path] = "plots",
        include_archived: bool = False
    ):
        """Export all plots as SVG, PDF, and PNG files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(self.datafilename, 'r') as f:
            def visitor(name, obj):
                if isinstance(obj, h5py.Group):
                    # Check if this is a figure group (has svg dataset)
                    if 'svg' not in obj:
                        return
                    
                    # Skip archived unless requested
                    if name.startswith("archived") and not include_archived:
                        return
                    
                    # Create safe filename
                    safe_name = name.replace("/", "_")
                    
                    formats_exported = []
                    
                    # Export SVG
                    if 'svg' in obj:
                        try:
                            svg_data = bytes(obj['svg'][()])
                            svg_path = output_dir / f"{safe_name}.svg"
                            with open(svg_path, 'wb') as out:
                                out.write(svg_data)
                            formats_exported.append('svg')
                            
                            # Convert SVG to PNG using cairosvg
                            try:
                                png_path = output_dir / f"{safe_name}.png"
                                cairosvg.svg2png(bytestring=svg_data, write_to=str(png_path))
                                formats_exported.append('png')
                            except Exception as e:
                                print(f"Warning: Could not convert {safe_name} to PNG: {e}")
                        except Exception as e:
                            print(f"Warning: Could not export SVG for {safe_name}: {e}")
                    
                    # Export PGF (if available)
                    if 'pgf' in obj:
                        try:
                            pgf_data = bytes(obj['pgf'][()])
                            pgf_path = output_dir / f"{safe_name}.pgf"
                            with open(pgf_path, 'wb') as out:
                                out.write(pgf_data)
                            formats_exported.append('pgf')
                        except Exception as e:
                            print(f"Warning: Could not export PGF for {safe_name}: {e}")
                    
                    # Export PDF (if available)
                    if 'pdf' in obj:
                        try:
                            pdf_data = bytes(obj['pdf'][()])
                            pdf_path = output_dir / f"{safe_name}.pdf"
                            with open(pdf_path, 'wb') as out:
                                out.write(pdf_data)
                            formats_exported.append('pdf')
                        except Exception as e:
                            print(f"Warning: Could not export PDF for {safe_name}: {e}")
                    
                    if formats_exported:
                        print(f"Exported: {safe_name} [{', '.join(formats_exported)}]")
            
            f.visititems(visitor)
        
        print(f"\nAll plots exported to: {output_dir}")    

    def generate_pdf_report(
        self,
        output_file: Union[str, Path] = "mgnn_report.pdf",
        include_archived: bool = False,
        title: str = "MGNN Analysis Report",
        author: str = "MGNN Pipeline"
    ) -> Path:
        """
        Generate a comprehensive PDF report with all plots.
        
        Args:
            output_file: Output PDF filename
            include_archived: Include archived plots
            title: Report title
            author: Report author
        
        Returns:
            Path to generated PDF file
        """
        from matplotlib.backends.backend_pdf import PdfPages
        from datetime import datetime
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating comprehensive PDF report: {output_path}")
        print("="*70)
        
        with h5py.File(self.datafilename, 'r') as f:
            # Collect all figure groups
            figure_groups = []
            
            def visitor(name, obj):
                if isinstance(obj, h5py.Group) and 'svg' in obj:
                    # Skip archived unless requested
                    if name.startswith("archived") and not include_archived:
                        return
                    figure_groups.append((name, obj))
            
            f.visititems(visitor)
            
            if not figure_groups:
                print("Warning: No figures found in data file")
                return None
            
            print(f"Found {len(figure_groups)} figures to include in report")
            
            # Create PDF with all plots
            with PdfPages(output_path) as pdf:
                # Add title page
                fig = plt.figure(figsize=(8.5, 11))
                fig.patch.set_facecolor('white')
                ax = fig.add_subplot(111)
                ax.axis('off')
                
                # Title
                ax.text(0.5, 0.7, title, 
                    ha='center', va='center', 
                    fontsize=24, weight='bold',
                    transform=ax.transAxes)
                
                # Subtitle
                ax.text(0.5, 0.6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                    ha='center', va='center', 
                    fontsize=12, style='italic',
                    transform=ax.transAxes)
                
                # Author
                ax.text(0.5, 0.5, f"Author: {author}", 
                    ha='center', va='center', 
                    fontsize=12,
                    transform=ax.transAxes)
                
                # Data file info
                ax.text(0.5, 0.35, f"Data File: {self.datafilename.name}", 
                    ha='center', va='center', 
                    fontsize=10,
                    transform=ax.transAxes)
                
                ax.text(0.5, 0.30, f"Total Figures: {len(figure_groups)}", 
                    ha='center', va='center', 
                    fontsize=10,
                    transform=ax.transAxes)
                
                # Add border
                rect = plt.Rectangle((0.1, 0.2), 0.8, 0.6, 
                                    fill=False, edgecolor='black', linewidth=2,
                                    transform=ax.transAxes)
                ax.add_patch(rect)
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
                # Add table of contents page
                fig = plt.figure(figsize=(8.5, 11))
                fig.patch.set_facecolor('white')
                ax = fig.add_subplot(111)
                ax.axis('off')
                
                ax.text(0.5, 0.95, "Table of Contents", 
                    ha='center', va='top', 
                    fontsize=18, weight='bold',
                    transform=ax.transAxes)
                
                # List all figures
                y_pos = 0.88
                for i, (name, _) in enumerate(figure_groups, 1):
                    display_name = name.replace('_', ' ').title()
                    ax.text(0.1, y_pos, f"{i}. {display_name}", 
                        ha='left', va='top', 
                        fontsize=9,
                        transform=ax.transAxes)
                    y_pos -= 0.03
                    
                    if y_pos < 0.1:
                        break  # Don't overflow page
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                SKIP_HEAVY_PLOTS = ['stability_vs_tolerance']  # Add problematic plots here

                
                # Add each figure
                for i, (name, obj) in enumerate(figure_groups, 1):
                    print(f"  Adding figure {i}/{len(figure_groups)}: {name}")

                    plot_name = name.split('/')[-1]
                    if plot_name in SKIP_HEAVY_PLOTS:
                        print(f"  Skipping heavy plot {i}/{len(figure_groups)}: {name} (use --export-plots for full resolution)")
                    else:                
                        try:
                            # Load SVG data and reconstruct figure
                            svg_data = bytes(obj['svg'][()])
                            
                            # Save SVG to temporary file and load as image
                            import tempfile
                            with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
                                tmp.write(svg_data)
                                tmp_path = tmp.name
                            
                            # Create figure with the SVG
                            fig = plt.figure(figsize=(8.5, 11))
                            fig.patch.set_facecolor('white')
                            
                            # Add title with figure name
                            fig.suptitle(name.replace('_', ' ').title(), 
                                    fontsize=12, weight='bold', y=0.98)
                            
                            # Add the image
                            ax = fig.add_subplot(111)
                            
                            # Try to load and display SVG as raster
                            try:
                                from cairosvg import svg2png
                                png_data = svg2png(bytestring=svg_data)
                                
                                import io
                                from PIL import Image
                                img = Image.open(io.BytesIO(png_data))
                                ax.imshow(img)
                                ax.axis('off')
                            except:
                                # Fallback: just show a placeholder with metadata
                                ax.axis('off')
                                
                                # Get metadata if available
                                metadata_text = f"Figure: {name}\n\n"
                                if hasattr(obj['svg'], 'attrs'):
                                    for key in obj['svg'].attrs.keys():
                                        val = obj['svg'].attrs[key]
                                        metadata_text += f"{key}: {val}\n"
                                
                                ax.text(0.5, 0.5, metadata_text,
                                    ha='center', va='center',
                                    fontsize=10,
                                    transform=ax.transAxes,
                                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                            
                            # Add page number
                            fig.text(0.95, 0.02, f"Page {i+2}", 
                                    ha='right', va='bottom', fontsize=8)
                            
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close(fig)

                            print(f"    Added figure: {name}")
                            
                            # Clean up temp file
                            import os
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                            
                        except Exception as e:
                            print(f"    Warning: Could not add figure {name}: {e}")
                            
                            # Add error page
                            fig = plt.figure(figsize=(8.5, 11))
                            fig.patch.set_facecolor('white')
                            ax = fig.add_subplot(111)
                            ax.axis('off')
                            
                            ax.text(0.5, 0.5, f"Error loading figure:\n{name}\n\n{str(e)}", 
                                ha='center', va='center', 
                                fontsize=12, color='red',
                                transform=ax.transAxes)
                            
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close(fig)
                    
                # Set PDF metadata
                d = pdf.infodict()
                d['Title'] = title
                d['Author'] = author
                d['Subject'] = 'MGNN Analysis Report'
                d['Keywords'] = 'MGNN, Perovskites, Machine Learning'
                d['CreationDate'] = datetime.now()
        
        print("="*70)
        print(f"PDF report generated: {output_path}")
        print(f"Total pages: {len(figure_groups) + 2}")  # +2 for title and TOC
        print("="*70)
        
        return output_path
    
    
    def summary(self):
        """Print summary statistics."""
        with h5py.File(self.datafilename, 'r') as f:
            n_groups = 0
            n_datasets = 0
            n_figures = 0
            
            def visitor(name, obj):
                nonlocal n_groups, n_datasets, n_figures
                
                if isinstance(obj, h5py.Group):
                    if 'svg' in obj and 'pgf' in obj:
                        n_figures += 1
                    else:
                        n_groups += 1
                elif isinstance(obj, h5py.Dataset):
                    n_datasets += 1
            
            f.visititems(visitor)
            
            print(f"\n{'='*50}")
            print(f"Data Summary: {self.datafilename}")
            print(f"{'='*50}")
            print(f"Groups:   {n_groups}")
            print(f"Datasets: {n_datasets}")
            print(f"Figures:  {n_figures}")
            print(f"{'='*50}\n")