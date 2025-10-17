import os
import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
from rasterio.warp import transform_bounds
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
from scipy.optimize import least_squares
from scipy.interpolate import RectBivariateSpline
import time
from collections import defaultdict
warnings.filterwarnings('ignore')

class GridHarmonizer:
    """
    Harmonizes embeddings across grid tiles to ensure visual continuity.
    Uses linear transformations (k*c + b) for each channel of each grid.
    """
    
    def __init__(self, base_dir, year='2022', num_channels=3):
        """
        Initialize the harmonizer.
        
        Args:
            base_dir: Base directory containing grid folders
            year: Year subfolder
            num_channels: Number of channels to process (default 3 for RGB)
        """
        self.base_dir = Path(base_dir) / year
        self.num_channels = num_channels
        self.grids = {}
        self.grid_metadata = {}
        self.overlaps = {}
        self.transformations = {}
        
    def load_grid_metadata(self):
        """Load metadata for all grids including bounds and CRS."""
        print("\n" + "="*80)
        print("[STEP 1] Loading Grid Metadata")
        print("="*80)
        
        grid_folders = sorted([d for d in self.base_dir.iterdir() 
                              if d.is_dir() and d.name.startswith("grid_")])
        
        print(f"Found {len(grid_folders)} grid folders in {self.base_dir}")
        
        valid_grids = 0
        for folder in tqdm(grid_folders, desc="Loading metadata"):
            grid_name = folder.name
            tiff_path = folder / f"{grid_name}.tiff"
            npy_path = folder / f"{grid_name}.npy"
            scales_path = folder / f"{grid_name}_scales.npy"
            
            if not tiff_path.exists() or not npy_path.exists():
                continue
                
            with rasterio.open(tiff_path) as src:
                self.grid_metadata[grid_name] = {
                    'bounds': src.bounds,
                    'transform': src.transform,
                    'crs': src.crs,
                    'shape': (src.height, src.width),
                    'folder': folder,
                    'has_scales': scales_path.exists()
                }
                valid_grids += 1
        
        print(f"\nSuccessfully loaded metadata for {valid_grids} valid grids")
        
        # Analyze CRS consistency
        crs_set = set(str(meta['crs']) for meta in self.grid_metadata.values())
        if len(crs_set) > 1:
            print(f"WARNING: Found multiple CRS in grids: {crs_set}")
        else:
            print(f"All grids use CRS: {crs_set.pop()}")
    
    def find_overlapping_grids(self):
        """Find all pairs of overlapping grids."""
        print("\n" + "="*80)
        print("[STEP 2] Finding Overlapping Grid Pairs")
        print("="*80)
        
        grid_names = list(self.grid_metadata.keys())
        n_grids = len(grid_names)
        
        print(f"Checking {n_grids * (n_grids - 1) // 2} potential grid pairs...")
        
        # Check all pairs
        overlap_count = 0
        total_overlap_area = 0
        
        for i in range(n_grids):
            for j in range(i + 1, n_grids):
                name1, name2 = grid_names[i], grid_names[j]
                bounds1 = self.grid_metadata[name1]['bounds']
                bounds2 = self.grid_metadata[name2]['bounds']
                
                # Check if bounds overlap
                if (bounds1.left < bounds2.right and bounds1.right > bounds2.left and
                    bounds1.bottom < bounds2.top and bounds1.top > bounds2.bottom):
                    
                    # Calculate overlap region
                    overlap_left = max(bounds1.left, bounds2.left)
                    overlap_bottom = max(bounds1.bottom, bounds2.bottom)
                    overlap_right = min(bounds1.right, bounds2.right)
                    overlap_top = min(bounds1.top, bounds2.top)
                    
                    overlap_bounds = (overlap_left, overlap_bottom, overlap_right, overlap_top)
                    overlap_area = (overlap_right - overlap_left) * (overlap_top - overlap_bottom)
                    
                    # Store overlap info
                    self.overlaps[(name1, name2)] = {
                        'bounds': overlap_bounds,
                        'area': overlap_area
                    }
                    overlap_count += 1
                    total_overlap_area += overlap_area
        
        print(f"\nFound {overlap_count} overlapping grid pairs")
        print(f"Total overlap area: {total_overlap_area:.2f} square units")
        
        # Analyze overlap statistics
        if overlap_count > 0:
            overlap_areas = [info['area'] for info in self.overlaps.values()]
            print(f"Average overlap area: {np.mean(overlap_areas):.2f}")
            print(f"Min overlap area: {np.min(overlap_areas):.2f}")
            print(f"Max overlap area: {np.max(overlap_areas):.2f}")
    
    def load_grid_data(self, grid_name, verbose=False):
        """Load and dequantize grid data."""
        folder = self.grid_metadata[grid_name]['folder']
        
        # Load embeddings
        npy_path = folder / f"{grid_name}.npy"
        embeddings = np.load(npy_path, mmap_mode='r')[:, :, :self.num_channels].copy()
        
        # Load scales and dequantize
        scales_path = folder / f"{grid_name}_scales.npy"
        if scales_path.exists():
            scales = np.load(scales_path, mmap_mode='r').copy()
            embeddings = embeddings.astype(np.float32) * scales[:, :, np.newaxis]
            if verbose:
                print(f"  Loaded and dequantized {grid_name}: shape={embeddings.shape}, "
                      f"range=[{np.min(embeddings):.3f}, {np.max(embeddings):.3f}]")
        else:
            embeddings = embeddings.astype(np.float32)
            if verbose:
                print(f"  Loaded {grid_name} (no scales): shape={embeddings.shape}")
            
        return embeddings
    
    def extract_overlap_data(self, grid1_name, grid2_name):
        """Extract embedding values from overlapping regions of two grids."""
        overlap_info = self.overlaps.get((grid1_name, grid2_name)) or \
                      self.overlaps.get((grid2_name, grid1_name))
        
        if not overlap_info:
            return None, None
        
        overlap_bounds = overlap_info['bounds']
        
        # Load data for both grids
        data1 = self.load_grid_data(grid1_name)
        data2 = self.load_grid_data(grid2_name)
        
        # Get transforms
        transform1 = self.grid_metadata[grid1_name]['transform']
        transform2 = self.grid_metadata[grid2_name]['transform']
        
        # Convert geographic bounds to pixel coordinates
        left, bottom, right, top = overlap_bounds
        
        # For grid1 - note: rowcol returns (row, col) for (x, y) input
        row1_min, col1_min = rowcol(transform1, left, top)
        row1_max, col1_max = rowcol(transform1, right, bottom)
        
        # For grid2
        row2_min, col2_min = rowcol(transform2, left, top)
        row2_max, col2_max = rowcol(transform2, right, bottom)
        
        # Ensure proper ordering (min < max)
        row1_min, row1_max = min(row1_min, row1_max), max(row1_min, row1_max)
        col1_min, col1_max = min(col1_min, col1_max), max(col1_min, col1_max)
        row2_min, row2_max = min(row2_min, row2_max), max(row2_min, row2_max)
        col2_min, col2_max = min(col2_min, col2_max), max(col2_min, col2_max)
        
        # Clip to valid bounds
        h1, w1 = data1.shape[:2]
        h2, w2 = data2.shape[:2]
        
        row1_min = max(0, row1_min)
        row1_max = min(h1, row1_max + 1)  # +1 for inclusive slicing
        col1_min = max(0, col1_min)
        col1_max = min(w1, col1_max + 1)
        
        row2_min = max(0, row2_min)
        row2_max = min(h2, row2_max + 1)
        col2_min = max(0, col2_min)
        col2_max = min(w2, col2_max + 1)
        
        # Extract overlap regions
        overlap1 = data1[row1_min:row1_max, col1_min:col1_max, :]
        overlap2 = data2[row2_min:row2_max, col2_min:col2_max, :]
        
        # If sizes don't match exactly, resample to common size
        if overlap1.shape[:2] != overlap2.shape[:2]:
            # Use bilinear interpolation for resampling
            target_h = min(overlap1.shape[0], overlap2.shape[0])
            target_w = min(overlap1.shape[1], overlap2.shape[1])
            
            if target_h > 1 and target_w > 1:
                overlap1_resampled = np.zeros((target_h, target_w, self.num_channels), dtype=np.float32)
                overlap2_resampled = np.zeros((target_h, target_w, self.num_channels), dtype=np.float32)
                
                for c in range(self.num_channels):
                    # Create interpolators
                    y1 = np.linspace(0, overlap1.shape[0]-1, overlap1.shape[0])
                    x1 = np.linspace(0, overlap1.shape[1]-1, overlap1.shape[1])
                    interp1 = RectBivariateSpline(y1, x1, overlap1[:, :, c], kx=1, ky=1)
                    
                    y2 = np.linspace(0, overlap2.shape[0]-1, overlap2.shape[0])
                    x2 = np.linspace(0, overlap2.shape[1]-1, overlap2.shape[1])
                    interp2 = RectBivariateSpline(y2, x2, overlap2[:, :, c], kx=1, ky=1)
                    
                    # Resample
                    y_new = np.linspace(0, overlap1.shape[0]-1, target_h)
                    x_new = np.linspace(0, overlap1.shape[1]-1, target_w)
                    overlap1_resampled[:, :, c] = interp1(y_new, x_new)
                    
                    y_new = np.linspace(0, overlap2.shape[0]-1, target_h)
                    x_new = np.linspace(0, overlap2.shape[1]-1, target_w)
                    overlap2_resampled[:, :, c] = interp2(y_new, x_new)
                
                return overlap1_resampled, overlap2_resampled
        
        return overlap1, overlap2
    
    def build_optimization_problem(self):
        """
        Build the optimization problem using a more robust approach.
        We minimize the sum of squared differences in overlapping regions.
        """
        print("\n" + "="*80)
        print("[STEP 3] Building Global Optimization Problem")
        print("="*80)
        
        grid_names = sorted(self.grid_metadata.keys())
        n_grids = len(grid_names)
        grid_idx = {name: idx for idx, name in enumerate(grid_names)}
        
        # Collect all overlap constraints
        print("Extracting overlap data...")
        overlap_data = []
        
        # Use parallel processing for faster data extraction
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            for (grid1_name, grid2_name) in self.overlaps.keys():
                future = executor.submit(self.extract_overlap_data, grid1_name, grid2_name)
                futures[future] = (grid1_name, grid2_name)
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing overlaps"):
                grid1_name, grid2_name = futures[future]
                try:
                    overlap1, overlap2 = future.result()
                    if overlap1 is not None and overlap1.size > 0:
                        valid_mask = (np.abs(overlap1).sum(axis=2) > 0) & (np.abs(overlap2).sum(axis=2) > 0)
                        if np.any(valid_mask):
                            overlap_data.append({
                                'grid1': grid1_name,
                                'grid2': grid2_name,
                                'data1': overlap1[valid_mask].reshape(-1, self.num_channels),
                                'data2': overlap2[valid_mask].reshape(-1, self.num_channels),
                                'n_pixels': valid_mask.sum()
                            })
                except Exception as e:
                    print(f"Error processing {grid1_name} - {grid2_name}: {e}")
        
        print(f"\nCollected {len(overlap_data)} valid overlap regions")
        total_pixels = sum(d['n_pixels'] for d in overlap_data)
        print(f"Total overlap pixels: {total_pixels:,}")
        
        # Define optimization function
        def residual_function(params):
            """Compute residuals for all overlap constraints."""
            residuals = []
            
            for data in overlap_data:
                idx1 = grid_idx[data['grid1']]
                idx2 = grid_idx[data['grid2']]
                
                # Extract transformation parameters
                k1 = params[idx1 * 2 * self.num_channels : idx1 * 2 * self.num_channels + self.num_channels]
                b1 = params[idx1 * 2 * self.num_channels + self.num_channels : (idx1 + 1) * 2 * self.num_channels]
                k2 = params[idx2 * 2 * self.num_channels : idx2 * 2 * self.num_channels + self.num_channels]
                b2 = params[idx2 * 2 * self.num_channels + self.num_channels : (idx2 + 1) * 2 * self.num_channels]
                
                # Apply transformations
                transformed1 = data['data1'] * k1[np.newaxis, :] + b1[np.newaxis, :]
                transformed2 = data['data2'] * k2[np.newaxis, :] + b2[np.newaxis, :]
                
                # Compute differences
                diff = transformed1 - transformed2
                residuals.append(diff.flatten())
            
            # Add regularization to prefer identity transformation
            reg_weight = 0.01 * np.sqrt(total_pixels / n_grids)
            for i in range(n_grids):
                for c in range(self.num_channels):
                    k_idx = i * 2 * self.num_channels + c
                    b_idx = i * 2 * self.num_channels + self.num_channels + c
                    
                    # Penalize deviation from k=1, b=0
                    residuals.append([(params[k_idx] - 1.0) * reg_weight])
                    residuals.append([params[b_idx] * reg_weight])
            
            return np.concatenate(residuals)
        
        # Initial parameters (k=1, b=0 for all)
        n_params = n_grids * 2 * self.num_channels
        x0 = np.zeros(n_params)
        for i in range(n_grids):
            for c in range(self.num_channels):
                k_idx = i * 2 * self.num_channels + c
                x0[k_idx] = 1.0  # Initial k = 1
        
        print(f"\nOptimization setup:")
        print(f"  Number of grids: {n_grids}")
        print(f"  Number of parameters: {n_params}")
        print(f"  Number of overlap constraints: {len(overlap_data)}")
        
        return residual_function, x0, grid_names
    
    def solve_optimization(self, residual_function, x0):
        """Solve the optimization problem using robust least squares."""
        print("\n" + "="*80)
        print("[STEP 4] Solving Global Optimization")
        print("="*80)
        
        print("Running least squares optimization...")
        start_time = time.time()
        
        # Use robust least squares with bounds
        # Constrain k to be positive and near 1, b to be small
        n_grids = len(x0) // (2 * self.num_channels)
        lower_bounds = []
        upper_bounds = []
        
        # Fixed: Set bounds correctly according to parameter layout
        for i in range(n_grids):
            # First add bounds for all k values of this grid
            for c in range(self.num_channels):
                lower_bounds.append(0.5)  # k >= 0.5
                upper_bounds.append(2.0)  # k <= 2.0
            
            # Then add bounds for all b values of this grid
            for c in range(self.num_channels):
                lower_bounds.append(-0.5)  # b >= -0.5
                upper_bounds.append(0.5)   # b <= 0.5
        
        # Verify bounds match x0 size
        assert len(lower_bounds) == len(x0), f"Bounds size mismatch: {len(lower_bounds)} vs {len(x0)}"
        assert len(upper_bounds) == len(x0), f"Bounds size mismatch: {len(upper_bounds)} vs {len(x0)}"
        
        # Verify x0 is within bounds
        for i, (val, lb, ub) in enumerate(zip(x0, lower_bounds, upper_bounds)):
            if not (lb <= val <= ub):
                print(f"WARNING: x0[{i}] = {val} not in bounds [{lb}, {ub}]")
                # Fix it
                x0[i] = np.clip(val, lb, ub)
        
        result = least_squares(
            residual_function, 
            x0,
            bounds=(lower_bounds, upper_bounds),
            method='trf',
            verbose=2,
            max_nfev=100
        )
        
        elapsed = time.time() - start_time
        print(f"\nOptimization completed in {elapsed:.2f} seconds")
        print(f"Success: {result.success}")
        print(f"Final cost: {result.cost:.6f}")
        print(f"Number of function evaluations: {result.nfev}")
        
        return result.x
    
    def extract_transformations(self, x, grid_names):
        """Extract transformation parameters from solution vector."""
        print("\n" + "="*80)
        print("[STEP 5] Extracting Transformation Parameters")
        print("="*80)
        
        # Store transformations
        for idx, grid_name in enumerate(grid_names):
            k_values = []
            b_values = []
            
            for c in range(self.num_channels):
                k_idx = idx * 2 * self.num_channels + c
                b_idx = idx * 2 * self.num_channels + self.num_channels + c
                k_values.append(x[k_idx])
                b_values.append(x[b_idx])
            
            self.transformations[grid_name] = {
                'k': np.array(k_values),
                'b': np.array(b_values)
            }
        
        # Print statistics
        print("\nTransformation statistics:")
        all_k = []
        all_b = []
        
        print("\nFirst 10 grids (Channel 0 transformations):")
        for i, grid_name in enumerate(grid_names[:10]):
            trans = self.transformations[grid_name]
            print(f"  {grid_name}: k={trans['k'][0]:.4f}, b={trans['b'][0]:.4f}")
            all_k.extend(trans['k'])
            all_b.extend(trans['b'])
        
        # Collect all values for statistics
        for grid_name in grid_names[10:]:
            trans = self.transformations[grid_name]
            all_k.extend(trans['k'])
            all_b.extend(trans['b'])
        
        all_k = np.array(all_k)
        all_b = np.array(all_b)
        
        print(f"\nGlobal statistics:")
        print(f"  Scale factors (k): mean={np.mean(all_k):.4f}, std={np.std(all_k):.4f}, "
              f"min={np.min(all_k):.4f}, max={np.max(all_k):.4f}")
        print(f"  Bias terms (b): mean={np.mean(all_b):.4f}, std={np.std(all_b):.4f}, "
              f"min={np.min(all_b):.4f}, max={np.max(all_b):.4f}")
        
        # Print all grids' channel 0 transformations for complete visibility
        print("\n" + "-"*60)
        print("All grids - Channel 0 (Red) transformations:")
        print("-"*60)
        for grid_name in sorted(grid_names):
            trans = self.transformations[grid_name]
            print(f"{grid_name}: k={trans['k'][0]:.4f}, b={trans['b'][0]:.4f}")
    
    def create_mosaic(self, output_dir, apply_transform=False, resolution_m=10):
        """Create a mosaic from all grids."""
        # Determine the common CRS from grids
        crs = list(self.grid_metadata.values())[0]['crs']
        
        # Calculate overall bounds
        all_bounds = [meta['bounds'] for meta in self.grid_metadata.values()]
        west = min(b.left for b in all_bounds)
        south = min(b.bottom for b in all_bounds)
        east = max(b.right for b in all_bounds)
        north = max(b.top for b in all_bounds)
        
        # Calculate output dimensions based on CRS
        if crs.to_string() == 'EPSG:4326':
            # For geographic CRS, convert resolution from meters to degrees
            lat_center = (south + north) / 2
            lon_resolution = resolution_m / (111320 * np.cos(np.radians(lat_center)))
            lat_resolution = resolution_m / 111320
            
            width = int((east - west) / lon_resolution)
            height = int((north - south) / lat_resolution)
        else:
            # For projected CRS, use resolution directly
            width = int((east - west) / resolution_m)
            height = int((north - south) / resolution_m)
        
        print(f"\nCreating mosaic: {width} x {height} pixels")
        print(f"Bounds: [{west:.2f}, {south:.2f}, {east:.2f}, {north:.2f}]")
        
        # Create output arrays
        mosaic = np.zeros((height, width, 3), dtype=np.float32)
        mosaic_count = np.zeros((height, width), dtype=np.int32)
        
        # Process each grid
        for grid_name in tqdm(self.grid_metadata.keys(), desc="Adding grids to mosaic"):
            # Load data
            data = self.load_grid_data(grid_name)
            
            # Apply transformation if requested
            if apply_transform and grid_name in self.transformations:
                trans = self.transformations[grid_name]
                for c in range(3):
                    data[:, :, c] = trans['k'][c] * data[:, :, c] + trans['b'][c]
            
            # Get grid bounds and transform
            bounds = self.grid_metadata[grid_name]['bounds']
            
            # Calculate position in output mosaic
            if crs.to_string() == 'EPSG:4326':
                col_start = int((bounds.left - west) / lon_resolution)
                row_start = int((north - bounds.top) / lat_resolution)
                col_end = int((bounds.right - west) / lon_resolution)
                row_end = int((north - bounds.bottom) / lat_resolution)
            else:
                col_start = int((bounds.left - west) / resolution_m)
                row_start = int((north - bounds.top) / resolution_m)
                col_end = int((bounds.right - west) / resolution_m)
                row_end = int((north - bounds.bottom) / resolution_m)
            
            # Ensure valid bounds
            col_start = max(0, col_start)
            row_start = max(0, row_start)
            col_end = min(width, col_end)
            row_end = min(height, row_end)
            
            if col_end <= col_start or row_end <= row_start:
                continue
            
            # Resize data to fit
            target_h = row_end - row_start
            target_w = col_end - col_start
            
            if target_h != data.shape[0] or target_w != data.shape[1]:
                # Use bilinear interpolation for resize
                data_resized = np.zeros((target_h, target_w, 3), dtype=np.float32)
                for c in range(3):
                    y = np.linspace(0, data.shape[0]-1, data.shape[0])
                    x = np.linspace(0, data.shape[1]-1, data.shape[1])
                    interp = RectBivariateSpline(y, x, data[:, :, c], kx=1, ky=1)
                    
                    y_new = np.linspace(0, data.shape[0]-1, target_h)
                    x_new = np.linspace(0, data.shape[1]-1, target_w)
                    data_resized[:, :, c] = interp(y_new, x_new)
                
                data = data_resized
            
            # Add to mosaic
            valid_mask = np.any(data != 0, axis=2)
            mosaic[row_start:row_end, col_start:col_end][valid_mask] = data[valid_mask]
            mosaic_count[row_start:row_end, col_start:col_end][valid_mask] += 1
        
        # Average overlapping regions
        mask = mosaic_count > 0
        for c in range(3):
            mosaic[:, :, c][mask] /= mosaic_count[mask]
        
        return mosaic
    
    def visualize_results(self, output_dir):
        """Create visualizations of before and after harmonization."""
        print("\n" + "="*80)
        print("[STEP 6] Creating Visualizations")
        print("="*80)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create mosaics
        print("\nCreating 'before' mosaic...")
        mosaic_before = self.create_mosaic(output_dir, apply_transform=False)
        
        print("\nCreating 'after' mosaic...")
        mosaic_after = self.create_mosaic(output_dir, apply_transform=True)
        
        # Convert to uint8 for visualization
        def to_uint8(data):
            result = np.zeros_like(data, dtype=np.uint8)
            for c in range(3):
                channel = data[:, :, c]
                valid = channel != 0
                if np.any(valid):
                    # Use percentile clipping for better visualization
                    vmin, vmax = np.percentile(channel[valid], [2, 98])
                    if vmax > vmin:
                        normalized = np.clip((channel - vmin) / (vmax - vmin), 0, 1)
                        result[:, :, c] = (normalized * 255).astype(np.uint8)
            return result
        
        mosaic_before_uint8 = to_uint8(mosaic_before)
        mosaic_after_uint8 = to_uint8(mosaic_after)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        
        axes[0].imshow(mosaic_before_uint8)
        axes[0].set_title('Before Harmonization', fontsize=16, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(mosaic_after_uint8)
        axes[1].set_title('After Harmonization', fontsize=16, fontweight='bold')
        axes[1].axis('off')
        
        plt.suptitle('Cambridge 2022 - Grid Harmonization Results', fontsize=20, fontweight='bold')
        plt.tight_layout()
        
        comparison_path = output_dir / 'cambridge_2022_harmonization_comparison.png'
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nSaved comparison to: {comparison_path}")
        
        # Save individual images
        plt.figure(figsize=(20, 20))
        plt.imshow(mosaic_before_uint8)
        plt.title('Cambridge 2022 - Before Harmonization', fontsize=18, fontweight='bold')
        plt.axis('off')
        before_path = output_dir / 'cambridge_2022_before_harmonization.png'
        plt.savefig(before_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved 'before' image to: {before_path}")
        
        plt.figure(figsize=(20, 20))
        plt.imshow(mosaic_after_uint8)
        plt.title('Cambridge 2022 - After Harmonization', fontsize=18, fontweight='bold')
        plt.axis('off')
        after_path = output_dir / 'cambridge_2022_after_harmonization.png'
        plt.savefig(after_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved 'after' image to: {after_path}")
        
        # Create difference map
        plt.figure(figsize=(20, 20))
        diff = np.abs(mosaic_after.astype(float) - mosaic_before.astype(float))
        diff_uint8 = to_uint8(diff)
        plt.imshow(diff_uint8)
        plt.title('Cambridge 2022 - Absolute Difference Map', fontsize=18, fontweight='bold')
        plt.axis('off')
        plt.colorbar(label='Difference Magnitude')
        diff_path = output_dir / 'cambridge_2022_difference_map.png'
        plt.savefig(diff_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved difference map to: {diff_path}")
        
        plt.close('all')
        
        # Print final statistics
        print("\n" + "="*80)
        print("HARMONIZATION COMPLETE!")
        print("="*80)
        
        valid_before = np.any(mosaic_before != 0, axis=2)
        valid_after = np.any(mosaic_after != 0, axis=2)
        
        print(f"\nMosaic statistics:")
        print(f"  Shape: {mosaic_before.shape}")
        print(f"  Coverage before: {np.sum(valid_before) / valid_before.size * 100:.1f}%")
        print(f"  Coverage after: {np.sum(valid_after) / valid_after.size * 100:.1f}%")
        
        for c, name in enumerate(['Red', 'Green', 'Blue']):
            before_vals = mosaic_before[:, :, c][valid_before]
            after_vals = mosaic_after[:, :, c][valid_after]
            
            print(f"\n  {name} channel:")
            print(f"    Before: mean={np.mean(before_vals):.3f}, std={np.std(before_vals):.3f}")
            print(f"    After:  mean={np.mean(after_vals):.3f}, std={np.std(after_vals):.3f}")
    
    def run(self, output_dir):
        """Run the complete harmonization pipeline."""
        print("\n" + "="*80)
        print("GRID HARMONIZATION PIPELINE")
        print("="*80)
        print(f"Base directory: {self.base_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Number of channels: {self.num_channels}")
        
        start_total = time.time()
        
        # Step 1: Load metadata
        self.load_grid_metadata()
        
        if not self.grid_metadata:
            print("\nERROR: No valid grids found!")
            return
        
        # Step 2: Find overlaps
        self.find_overlapping_grids()
        
        if not self.overlaps:
            print("\nERROR: No overlapping grids found!")
            return
        
        # Step 3: Build optimization problem
        residual_function, x0, grid_names = self.build_optimization_problem()
        
        # Step 4: Solve
        x = self.solve_optimization(residual_function, x0)
        
        # Step 5: Extract transformations
        self.extract_transformations(x, grid_names)
        
        # Step 6: Create visualizations
        self.visualize_results(output_dir)
        
        total_time = time.time() - start_total
        print(f"\nTotal pipeline execution time: {total_time:.2f} seconds")

def main():
    # Configuration
    base_dir = "/scratch/zf281/btfm_representation/cambridge"
    output_dir = Path(".")  # Current directory
    
    # Create output directory for results
    output_dir = output_dir / "harmonization_results"
    output_dir.mkdir(exist_ok=True)
    
    # Run harmonization
    harmonizer = GridHarmonizer(
        base_dir=base_dir,
        year='2022',
        num_channels=3  # RGB only
    )
    
    harmonizer.run(output_dir)

if __name__ == "__main__":
    main()