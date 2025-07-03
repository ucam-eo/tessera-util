import os
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from tqdm import tqdm
import tempfile
import warnings
from collections import defaultdict
import pyproj
import shutil
warnings.filterwarnings('ignore')

def analyze_dateline_crossing(bounds_list):
    """
    Analyze if dataset crosses dateline and determine which side has majority of data
    Returns: (crosses_dateline, majority_hemisphere, tiles_to_keep)
    """
    eastern_tiles = []  # Positive longitudes
    western_tiles = []  # Negative longitudes near dateline
    
    for i, (west, south, east, north) in enumerate(bounds_list):
        # Tile center longitude
        center_lon = (west + east) / 2
        
        # Check if this specific tile crosses dateline
        if west > 0 and east < 0:
            # This tile itself crosses dateline
            # Decide based on which side has more coverage
            if west > 180 + east:  # More on eastern side
                eastern_tiles.append(i)
            else:
                western_tiles.append(i)
        elif center_lon > 150:  # Eastern hemisphere near dateline
            eastern_tiles.append(i)
        elif center_lon < -150:  # Western hemisphere near dateline
            western_tiles.append(i)
        else:
            # Regular tiles far from dateline
            if center_lon > 0:
                eastern_tiles.append(i)
            else:
                western_tiles.append(i)
    
    # Check if we have tiles on both sides of dateline
    has_eastern_near_dateline = any(i for i in eastern_tiles 
                                   if bounds_list[i][0] > 150 or bounds_list[i][2] > 150)
    has_western_near_dateline = any(i for i in western_tiles 
                                   if bounds_list[i][0] < -150 or bounds_list[i][2] < -150)
    
    crosses_dateline = has_eastern_near_dateline and has_western_near_dateline
    
    if crosses_dateline:
        # Determine which side has more tiles
        if len(eastern_tiles) >= len(western_tiles):
            return True, 'eastern', eastern_tiles
        else:
            return True, 'western', western_tiles
    
    # No dateline crossing
    return False, None, list(range(len(bounds_list)))

def determine_best_projection(tiff_paths, sample_size=100, force_projected=True):
    """
    Determine the best projection for a set of TIFF files based on their geographic extent.
    
    Args:
        tiff_paths: List of TIFF file paths
        sample_size: Number of files to sample
        force_projected: If True, always use a projected CRS (UTM) instead of WGS84
    """
    print("\nDetermining best projection...")
    
    # Sample a subset of files to determine extent
    sample_paths = tiff_paths[:min(sample_size, len(tiff_paths))]
    
    # Collect bounds and CRS info
    all_bounds = []
    crs_counts = defaultdict(int)
    
    for path in tqdm(sample_paths, desc="Sampling TIFFs for projection"):
        try:
            with rasterio.open(path) as src:
                # Transform bounds to WGS84 for comparison
                if src.crs != 'EPSG:4326':
                    transformer = pyproj.Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
                    west, south = transformer.transform(src.bounds.left, src.bounds.bottom)
                    east, north = transformer.transform(src.bounds.right, src.bounds.top)
                else:
                    west, south, east, north = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
                
                all_bounds.append((west, south, east, north))
                crs_counts[str(src.crs)] += 1
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue
    
    if not all_bounds:
        # Default to most common CRS or WGS84
        if crs_counts:
            return max(crs_counts, key=crs_counts.get)
        return 'EPSG:4326'
    
    # Analyze dateline crossing
    crosses_dateline, majority_hemisphere, tiles_to_keep = analyze_dateline_crossing(all_bounds)
    
    if crosses_dateline:
        print(f"Area crosses the international dateline")
        print(f"Majority of data is in {majority_hemisphere} hemisphere")
        # For dateline crossing, use all bounds to determine center
        # This ensures we get the correct UTM zone for the majority of the data
    
    # Calculate extent from all bounds (not filtered)
    west = min(b[0] for b in all_bounds)
    south = min(b[1] for b in all_bounds)
    east = max(b[2] for b in all_bounds)
    north = max(b[3] for b in all_bounds)
    
    # Calculate center point
    if crosses_dateline:
        # For dateline crossing, calculate center based on majority hemisphere
        if majority_hemisphere == 'eastern':
            # Use tiles on eastern side
            eastern_bounds = [all_bounds[i] for i in tiles_to_keep]
            center_lon = np.mean([b[0] for b in eastern_bounds] + [b[2] for b in eastern_bounds])
        else:
            # Use tiles on western side
            western_bounds = [all_bounds[i] for i in tiles_to_keep]
            center_lon = np.mean([b[0] for b in western_bounds] + [b[2] for b in western_bounds])
        center_lat = (south + north) / 2
    else:
        center_lon = (west + east) / 2
        center_lat = (south + north) / 2
    
    print(f"Geographic extent: {west:.2f}°W to {east:.2f}°E, {south:.2f}°S to {north:.2f}°N")
    print(f"Center point: {center_lon:.2f}°, {center_lat:.2f}°")
    
    # Always use projected coordinate system for better visualization
    if not force_projected and crosses_dateline:
        print("Area crosses dateline, using WGS84 to avoid projection issues")
        return 'EPSG:4326'
    
    # Determine appropriate UTM zone
    utm_zone = int((center_lon + 180) / 6) + 1
    
    # Ensure UTM zone is valid (1-60)
    utm_zone = max(1, min(60, utm_zone))
    
    # Determine hemisphere
    if center_lat >= 0:
        epsg_code = 32600 + utm_zone  # Northern hemisphere
        hemisphere = "N"
    else:
        epsg_code = 32700 + utm_zone  # Southern hemisphere
        hemisphere = "S"
    
    target_crs = f'EPSG:{epsg_code}'
    print(f"Selected projection: {target_crs} (UTM Zone {utm_zone}{hemisphere})")
    
    # Special handling for very large areas or polar regions
    lat_span = north - south
    lon_span = east - west
    
    if lat_span > 60 or abs(center_lat) > 80:
        print("Large latitude span or polar region detected, considering alternative projections")
        
        # For polar regions
        if center_lat > 70:
            print("Using Arctic Polar Stereographic")
            return 'EPSG:3995'
        elif center_lat < -70:
            print("Using Antarctic Polar Stereographic")
            return 'EPSG:3031'
        
        # For very large areas, you might want to use a different projection
        # But for now, stick with UTM
    
    return target_crs

def create_multiband_data(grid_folder, temp_dir, downsample_factor=1, target_crs='EPSG:32630', rgb_only=False):
    """Create temporary multi-band TIFF and scales files for each grid, unified to target CRS"""
    try:
        grid_name = os.path.basename(grid_folder)
        npy_path = os.path.join(grid_folder, f"{grid_name}.npy")
        scales_path = os.path.join(grid_folder, f"{grid_name}_scales.npy")
        tiff_path = os.path.join(grid_folder, f"{grid_name}.tiff")
        
        if not os.path.exists(npy_path) or not os.path.exists(tiff_path):
            print(f"Missing files for {grid_name}")
            return None
        
        # Read geographic information from the TIFF in the same folder
        with rasterio.open(tiff_path) as src:
            src_transform = src.transform
            src_crs = src.crs
            bounds = src.bounds
        
        # Use memory mapping to read npy
        npy_mmap = np.load(npy_path, mmap_mode='r')
        h, w, c = npy_mmap.shape
        
        # Check if scales file exists
        has_scales = os.path.exists(scales_path)
        scales_data = None
        if has_scales:
            scales_mmap = np.load(scales_path, mmap_mode='r')
        
        # Determine number of bands to process
        num_bands_to_process = 3 if rgb_only else c
        
        print(f"Processing {grid_name}: shape {h}x{w}x{c}, using {num_bands_to_process} bands, scales: {has_scales}")
        
        # Downsample if needed
        if downsample_factor > 1:
            data = npy_mmap[::downsample_factor, ::downsample_factor, :num_bands_to_process].copy()
            if has_scales:
                scales_data = scales_mmap[::downsample_factor, ::downsample_factor].copy()
        else:
            data = npy_mmap[:, :, :num_bands_to_process].copy()
            if has_scales:
                scales_data = scales_mmap[:, :].copy()
        
        # Keep data as int8 - no scaling!
        data = data.astype(np.int8)
        
        # Create temporary TIFF file
        temp_tiff_path = os.path.join(temp_dir, f"{grid_name}_multiband.tif")
        temp_scales_path = os.path.join(temp_dir, f"{grid_name}_scales.npy") if has_scales else None
        
        # Adjust transform matrix for downsampling
        downsample_transform = src_transform * rasterio.Affine.scale(downsample_factor, downsample_factor)
        height, width = data.shape[:2]
        num_bands = data.shape[2]
        
        # If CRS differs from target CRS, reproject
        if str(src_crs) != str(target_crs):
            # Calculate new transform and dimensions
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs, target_crs, width, height,
                left=bounds.left, bottom=bounds.bottom, 
                right=bounds.right, top=bounds.top
            )
            
            # Create reprojected array
            dst_data = np.zeros((dst_height, dst_width, num_bands), dtype=np.int8)
            if has_scales:
                dst_scales = np.zeros((dst_height, dst_width), dtype=scales_data.dtype)
            
            # Reproject each band
            for i in range(num_bands):
                reproject(
                    source=data[:, :, i],
                    destination=dst_data[:, :, i],
                    src_transform=downsample_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )
            
            # Reproject scales if present
            if has_scales:
                reproject(
                    source=scales_data,
                    destination=dst_scales,
                    src_transform=downsample_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )
                scales_data = dst_scales
            
            # Use reprojected data
            data = dst_data
            transform = dst_transform
            crs = target_crs
            height, width = dst_height, dst_width
        else:
            transform = downsample_transform
            crs = src_crs
        
        # Write TIFF file with all bands
        with rasterio.open(
            temp_tiff_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=num_bands,
            dtype='int8',  # Keep as int8
            crs=crs,
            transform=transform,
            compress='lzw',
            tiled=True,
            blockxsize=256,
            blockysize=256
        ) as dst:
            for i in range(num_bands):
                dst.write(data[:, :, i], i + 1)
        
        # Save scales data if present
        if has_scales and temp_scales_path:
            np.save(temp_scales_path, scales_data)
        
        del npy_mmap
        if has_scales:
            del scales_mmap
            
        return {
            'path': temp_tiff_path,
            'scales_path': temp_scales_path,
            'src_crs': str(src_crs),
            'grid_name': grid_name,
            'num_bands': num_bands,
            'bounds': bounds,
            'transform': transform,
            'crs': crs,
            'has_scales': has_scales
        }
        
    except Exception as e:
        print(f"Error processing {grid_folder}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def filter_tiles_by_dateline(tiff_info_list, filter_minority=True):
    """
    Filter tiles based on dateline crossing analysis
    """
    if not filter_minority:
        return tiff_info_list
    
    # Get bounds in WGS84 for all tiles
    all_bounds_wgs84 = []
    for info in tiff_info_list:
        with rasterio.open(info['path']) as src:
            if src.crs != 'EPSG:4326':
                transformer = pyproj.Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
                west, south = transformer.transform(src.bounds.left, src.bounds.bottom)
                east, north = transformer.transform(src.bounds.right, src.bounds.top)
            else:
                west, south, east, north = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
            all_bounds_wgs84.append((west, south, east, north))
    
    # Analyze dateline crossing
    crosses_dateline, majority_hemisphere, tiles_to_keep = analyze_dateline_crossing(all_bounds_wgs84)
    
    if not crosses_dateline:
        return tiff_info_list
    
    # Filter tiles
    filtered_tiles = [tiff_info_list[i] for i in tiles_to_keep]
    
    print(f"\nDateline filtering: keeping {len(filtered_tiles)} of {len(tiff_info_list)} tiles")
    print(f"Dropped {len(tiff_info_list) - len(filtered_tiles)} tiles on minority side of dateline")
    
    return filtered_tiles

def merge_multiband_optimized(tiff_info_list, resolution, output_path, scales_output_path, target_crs, 
                            rgb_only=False, filter_cross_dateline_tiles=True, output_repr_format='tiff'):
    """Merge multi-band TIFFs using the optimal projection and also merge scales"""
    try:
        if not tiff_info_list:
            return False
        
        # Only filter for dateline crossing if we detect it
        if filter_cross_dateline_tiles:
            # Check if we need filtering
            all_bounds_wgs84 = []
            for info in tiff_info_list:
                with rasterio.open(info['path']) as src:
                    if str(src.crs) != 'EPSG:4326':
                        transformer = pyproj.Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
                        west, south = transformer.transform(src.bounds.left, src.bounds.bottom)
                        east, north = transformer.transform(src.bounds.right, src.bounds.top)
                    else:
                        west, south, east, north = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
                    all_bounds_wgs84.append((west, south, east, north))
            
            crosses_dateline, _, _ = analyze_dateline_crossing(all_bounds_wgs84)
            if crosses_dateline:
                tiff_info_list = filter_tiles_by_dateline(tiff_info_list, filter_cross_dateline_tiles)
        
        if not tiff_info_list:
            print("No tiles remaining after filtering")
            return False
            
        num_bands = tiff_info_list[0]['num_bands']
        has_scales = any(info['has_scales'] for info in tiff_info_list)
        
        # Get bounds in target CRS
        all_bounds_target = []
        
        for info in tiff_info_list:
            with rasterio.open(info['path']) as src:
                # All tiles should already be in target CRS from create_multiband_data
                if str(src.crs) != str(target_crs):
                    print(f"Warning: tile {info['grid_name']} not in target CRS")
                all_bounds_target.append(src.bounds)
        
        # Calculate mosaic bounds
        west = min(b.left for b in all_bounds_target)
        south = min(b.bottom for b in all_bounds_target)
        east = max(b.right for b in all_bounds_target)
        north = max(b.top for b in all_bounds_target)
        
        print(f"Output bounds in {target_crs}: {west:.2f}, {south:.2f}, {east:.2f}, {north:.2f}")
        
        # Calculate output dimensions based on resolution
        if target_crs == 'EPSG:4326':
            # For WGS84, convert resolution from meters to degrees
            center_lat = (south + north) / 2
            meters_per_degree_lon = 111320 * np.cos(np.radians(center_lat))
            meters_per_degree_lat = 111320
            
            resolution_x = resolution / meters_per_degree_lon
            resolution_y = resolution / meters_per_degree_lat
        else:
            # For projected CRS, use resolution directly
            resolution_x = resolution
            resolution_y = resolution
        
        width = int((east - west) / resolution_x)
        height = int((north - south) / resolution_y)
        
        print(f"Output dimensions: {width} x {height} pixels, {num_bands} bands")
        
        # Create output transform
        transform = from_bounds(west, south, east, north, width, height)
        
        # Initialize arrays for merging
        if output_repr_format == 'npy':
            # Create arrays in memory for NPY output
            merged_data = np.zeros((height, width, num_bands), dtype=np.int8)
        
        if has_scales:
            # Determine scales dtype from first file with scales
            scales_dtype = None
            for info in tiff_info_list:
                if info['has_scales'] and info['scales_path']:
                    scales_dtype = np.load(info['scales_path'], mmap_mode='r').dtype
                    break
            merged_scales = np.zeros((height, width), dtype=scales_dtype if scales_dtype else np.float32)
        
        # Process based on output format
        if output_repr_format == 'tiff':
            # Create output TIFF file
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=num_bands,
                dtype='int8',  # Keep as int8
                crs=target_crs,
                transform=transform,
                compress='lzw',
                tiled=True,
                blockxsize=512,
                blockysize=512,
                BIGTIFF='YES'
            ) as dst:
                
                # Process each band
                for band_idx in tqdm(range(1, num_bands + 1), desc="Processing bands"):
                    # Create band array
                    band_data = np.zeros((height, width), dtype=np.int8)
                    
                    # Process each tile for this band
                    for info in tiff_info_list:
                        with rasterio.open(info['path']) as src:
                            # Read the band
                            tile_band = src.read(band_idx)
                            
                            # Get tile bounds
                            tile_bounds = src.bounds
                            
                            # Calculate window in output raster
                            col_off = int((tile_bounds.left - west) / resolution_x)
                            row_off = int((north - tile_bounds.top) / resolution_y)
                            col_size = int((tile_bounds.right - tile_bounds.left) / resolution_x)
                            row_size = int((tile_bounds.top - tile_bounds.bottom) / resolution_y)
                            
                            # Clip to output bounds
                            if col_off < 0:
                                col_size += col_off
                                col_off = 0
                            if row_off < 0:
                                row_size += row_off
                                row_off = 0
                            if col_off + col_size > width:
                                col_size = width - col_off
                            if row_off + row_size > height:
                                row_size = height - row_off
                            
                            if col_size <= 0 or row_size <= 0:
                                continue
                            
                            # Resample tile to fit window if needed
                            if (tile_band.shape[0] != row_size or tile_band.shape[1] != col_size):
                                # Use rasterio's reproject for resampling
                                temp_data = np.zeros((row_size, col_size), dtype=np.int8)
                                
                                # Calculate transform for this window
                                win_west = west + col_off * resolution_x
                                win_north = north - row_off * resolution_y
                                win_transform = from_bounds(
                                    win_west, win_north - row_size * resolution_y,
                                    win_west + col_size * resolution_x, win_north,
                                    col_size, row_size
                                )
                                
                                reproject(
                                    source=tile_band,
                                    destination=temp_data,
                                    src_transform=src.transform,
                                    src_crs=src.crs,
                                    dst_transform=win_transform,
                                    dst_crs=target_crs,
                                    resampling=Resampling.nearest
                                )
                                
                                tile_data = temp_data
                            else:
                                tile_data = tile_band
                            
                            # Update band data with non-zero values
                            mask = tile_data != 0  # For int8, 0 is the nodata value
                            band_data[row_off:row_off+row_size, col_off:col_off+col_size][mask] = tile_data[mask]
                    
                    # Write band
                    dst.write(band_data, band_idx)
        
        else:  # output_repr_format == 'npy'
            # Process all tiles directly into memory array
            for info in tqdm(tiff_info_list, desc="Processing tiles"):
                with rasterio.open(info['path']) as src:
                    # Read all bands at once
                    tile_data = src.read()  # Shape: (bands, height, width)
                    
                    # Get tile bounds
                    tile_bounds = src.bounds
                    
                    # Calculate window in output raster
                    col_off = int((tile_bounds.left - west) / resolution_x)
                    row_off = int((north - tile_bounds.top) / resolution_y)
                    col_size = int((tile_bounds.right - tile_bounds.left) / resolution_x)
                    row_size = int((tile_bounds.top - tile_bounds.bottom) / resolution_y)
                    
                    # Clip to output bounds
                    if col_off < 0:
                        col_size += col_off
                        col_off = 0
                    if row_off < 0:
                        row_size += row_off
                        row_off = 0
                    if col_off + col_size > width:
                        col_size = width - col_off
                    if row_off + row_size > height:
                        row_size = height - row_off
                    
                    if col_size <= 0 or row_size <= 0:
                        continue
                    
                    # Process each band
                    for band_idx in range(num_bands):
                        tile_band = tile_data[band_idx]
                        
                        # Resample if needed
                        if (tile_band.shape[0] != row_size or tile_band.shape[1] != col_size):
                            temp_data = np.zeros((row_size, col_size), dtype=np.int8)
                            
                            win_west = west + col_off * resolution_x
                            win_north = north - row_off * resolution_y
                            win_transform = from_bounds(
                                win_west, win_north - row_size * resolution_y,
                                win_west + col_size * resolution_x, win_north,
                                col_size, row_size
                            )
                            
                            reproject(
                                source=tile_band,
                                destination=temp_data,
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=win_transform,
                                dst_crs=target_crs,
                                resampling=Resampling.nearest
                            )
                            
                            tile_band = temp_data
                        
                        # Update merged data
                        mask = tile_band != 0
                        merged_data[row_off:row_off+row_size, col_off:col_off+col_size, band_idx][mask] = tile_band[mask]
            
            # Save NPY file
            np.save(output_path, merged_data)
        
        # Process scales if present
        if has_scales:
            print("\nMerging scales data...")
            for info in tqdm(tiff_info_list, desc="Processing scales"):
                if not info['has_scales'] or not info['scales_path']:
                    continue
                
                # Load scales data
                tile_scales = np.load(info['scales_path'])
                
                with rasterio.open(info['path']) as src:
                    # Get tile bounds
                    tile_bounds = src.bounds
                    
                    # Calculate window in output raster
                    col_off = int((tile_bounds.left - west) / resolution_x)
                    row_off = int((north - tile_bounds.top) / resolution_y)
                    col_size = int((tile_bounds.right - tile_bounds.left) / resolution_x)
                    row_size = int((tile_bounds.top - tile_bounds.bottom) / resolution_y)
                    
                    # Clip to output bounds
                    if col_off < 0:
                        col_size += col_off
                        col_off = 0
                    if row_off < 0:
                        row_size += row_off
                        row_off = 0
                    if col_off + col_size > width:
                        col_size = width - col_off
                    if row_off + row_size > height:
                        row_size = height - row_off
                    
                    if col_size <= 0 or row_size <= 0:
                        continue
                    
                    # Resample scales to fit window if needed
                    if (tile_scales.shape[0] != row_size or tile_scales.shape[1] != col_size):
                        temp_scales = np.zeros((row_size, col_size), dtype=tile_scales.dtype)
                        
                        win_west = west + col_off * resolution_x
                        win_north = north - row_off * resolution_y
                        win_transform = from_bounds(
                            win_west, win_north - row_size * resolution_y,
                            win_west + col_size * resolution_x, win_north,
                            col_size, row_size
                        )
                        
                        reproject(
                            source=tile_scales,
                            destination=temp_scales,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=win_transform,
                            dst_crs=target_crs,
                            resampling=Resampling.nearest
                        )
                        
                        tile_scales = temp_scales
                    
                    # Update merged scales
                    mask = tile_scales != 0
                    merged_scales[row_off:row_off+row_size, col_off:col_off+col_size][mask] = tile_scales[mask]
            
            # Save merged scales
            np.save(scales_output_path, merged_scales)
            print(f"Scales data saved to: {scales_output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error in optimized merge: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def get_crs_name(crs_string):
    """Get a human-readable name for the CRS"""
    if crs_string == 'EPSG:4326':
        return 'wgs84'
    elif crs_string.startswith('EPSG:326'):
        zone = int(crs_string[8:])
        return f'utm{zone}n'
    elif crs_string.startswith('EPSG:327'):
        zone = int(crs_string[8:])
        return f'utm{zone}s'
    elif crs_string == 'EPSG:3995':
        return 'arctic_ps'
    elif crs_string == 'EPSG:3031':
        return 'antarctic_ps'
    else:
        return crs_string.replace(':', '_').lower()

def main(base_dir=None, downsample_factor=1, rgb_only=False, filter_cross_dateline_tiles=True, 
         force_projected=True, output_dir=None, output_repr_format='tiff'):
    """
    Main processing function
    
    Args:
        base_dir: Base directory containing grid folders
        downsample_factor: Factor for downsampling (default=1, no downsampling)
        rgb_only: If True, only process first 3 bands (RGB) instead of all 128 bands
        filter_cross_dateline_tiles: If True, filter out tiles on minority side of dateline crossing
        force_projected: If True, always use a projected CRS (like UTM) instead of WGS84
        output_dir: Output directory for results (default=base_dir)
        output_repr_format: Output format for representation ('tiff' or 'npy', default='tiff')
    """
    # Set paths
    if base_dir is None:
        base_dir = "/scratch/zf281/btfm_representation/senegal/2024"
    
    # Set output directory
    if output_dir is None:
        output_dir = base_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract location name from base_dir for output naming
    location_name = Path(base_dir).parts[-2]  # e.g., "senegal" from the path
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix=f'{location_name}_map_temp_')
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Get all grid subfolders
        grid_folders = [d for d in Path(base_dir).iterdir() if d.is_dir() and d.name.startswith("grid_")]
        print(f"Found {len(grid_folders)} grid folders")
        
        if not grid_folders:
            print("No grid folders found!")
            return
        
        # Determine best projection based on TIFF files in grid folders
        sample_tiff_paths = []
        for folder in grid_folders[:100]:  # Sample first 100
            tiff_path = os.path.join(folder, f"{folder.name}.tiff")
            if os.path.exists(tiff_path):
                sample_tiff_paths.append(tiff_path)
        
        target_crs = determine_best_projection(sample_tiff_paths, force_projected=force_projected)
        crs_name = get_crs_name(target_crs)
        
        print(f"\n*** Using target CRS: {target_crs} ({crs_name}) ***\n")
        
        # Calculate resolution based on downsample factor
        base_resolution = 10  # meters (assuming 10m base resolution)
        resolution = base_resolution * downsample_factor
        
        # Step 1: Create multi-band TIFF files in parallel
        bands_str = "RGB (3 bands)" if rgb_only else "all 128 bands"
        print(f"\nStep 1: Creating multi-band TIFF files ({bands_str}, resolution: {resolution}m)...")
        num_workers = min(48, len(grid_folders))  # Reduced workers to avoid memory issues
        
        tiff_info = []
        crs_stats = defaultdict(int)
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_grid = {
                executor.submit(create_multiband_data, str(folder), temp_dir, 
                              downsample_factor=downsample_factor, target_crs=target_crs, 
                              rgb_only=rgb_only): folder 
                for folder in grid_folders
            }
            
            with tqdm(total=len(grid_folders), desc="Creating multi-band TIFFs") as pbar:
                for future in as_completed(future_to_grid):
                    result = future.result()
                    if result is not None:
                        tiff_info.append(result)
                        crs_stats[result['src_crs']] += 1
                    pbar.update(1)
        
        print(f"\nCreated {len(tiff_info)} multi-band TIFF files")
        print("CRS distribution of source files:")
        for crs, count in crs_stats.items():
            print(f"  {crs}: {count} files")
        
        if not tiff_info:
            print("No valid TIFF files created")
            return
        
        # Extract number of bands
        num_bands = tiff_info[0]['num_bands']
        print(f"Number of bands: {num_bands}")
        
        # Step 2: Create mosaic using optimal projection
        print(f"\nStep 2: Creating {target_crs} mosaic...")
        
        # Output filename
        bands_suffix = "rgb" if rgb_only else f"{num_bands}bands"
        
        # Check if we need dateline filtering
        all_bounds_wgs84 = []
        for info in tiff_info:
            with rasterio.open(info['path']) as src:
                if str(src.crs) != 'EPSG:4326':
                    transformer = pyproj.Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
                    west, south = transformer.transform(src.bounds.left, src.bounds.bottom)
                    east, north = transformer.transform(src.bounds.right, src.bounds.top)
                else:
                    west, south, east, north = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
                all_bounds_wgs84.append((west, south, east, north))
        
        crosses_dateline, _, _ = analyze_dateline_crossing(all_bounds_wgs84)
        filter_suffix = "_filtered" if (filter_cross_dateline_tiles and crosses_dateline) else ""
        
        # Determine output extension
        output_ext = '.tiff' if output_repr_format == 'tiff' else '.npy'
        output_path = os.path.join(output_dir, f"{location_name}_map_{resolution}m_{crs_name}_{bands_suffix}{filter_suffix}{output_ext}")
        scales_output_path = os.path.join(output_dir, f"{location_name}_map_{resolution}m_{crs_name}_scales{filter_suffix}.npy")
        
        # Use optimized merge
        success = merge_multiband_optimized(tiff_info, resolution, output_path, scales_output_path, target_crs,
                                          rgb_only=rgb_only, filter_cross_dateline_tiles=filter_cross_dateline_tiles,
                                          output_repr_format=output_repr_format)
        
        if not success:
            print("Failed to create mosaic")
            return
        
        print(f"Multi-band {output_repr_format.upper()} saved to: {output_path}")
        
        # Step 3: Create RGB visualization
        print("\nStep 3: Creating RGB visualization...")
        
        # Read data for visualization
        if output_repr_format == 'tiff':
            with rasterio.open(output_path) as src:
                rgb_array = np.zeros((src.height, src.width, 3), dtype=np.uint8)
                for i in range(min(3, num_bands)):
                    band_data = src.read(i + 1)
                    # Convert from int8 to uint8 for visualization
                    rgb_array[:, :, i] = ((band_data.astype(np.float32) + 127.0) * (255.0 / 254.0)).astype(np.uint8)
                
                # Get transform and metadata
                transform = src.transform
                width = src.width
                height = src.height
                crs = src.crs
        else:  # npy format
            data = np.load(output_path, mmap_mode='r')
            height, width, _ = data.shape
            rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(min(3, num_bands)):
                # Convert from int8 to uint8 for visualization
                rgb_array[:, :, i] = ((data[:, :, i].astype(np.float32) + 127.0) * (255.0 / 254.0)).astype(np.uint8)
            
            # Calculate transform from known information
            # This is approximate since we don't have the exact georeferencing from NPY
            # You might want to save georeferencing info separately
            west = -180  # Default values, you should calculate these properly
            north = 90
            east = 180
            south = -90
            crs = target_crs
        
        # Calculate extent based on CRS type
        if output_repr_format == 'tiff':
            west = transform.c
            north = transform.f
            east = west + transform.a * width
            south = north + transform.e * height
        
        # Create figure
        fig, ax = plt.subplots(figsize=(20, 15))
        
        # Display image
        im = ax.imshow(rgb_array, extent=[west, east, south, north], 
                       interpolation='nearest', origin='upper')
        
        # Set title and labels based on CRS
        title = f'{location_name.title()} Map Visualization ({resolution}m resolution, {target_crs})'
        if filter_suffix:
            title += ' - Dateline Filtered'
        ax.set_title(title, fontsize=16)
        
        if target_crs == 'EPSG:4326':
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
        else:
            ax.set_xlabel('Easting (m)', fontsize=12)
            ax.set_ylabel('Northing (m)', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Set coordinate format
        ax.ticklabel_format(useOffset=False, style='plain')
        
        plt.tight_layout()
        
        # Save image
        output_png = os.path.join(output_dir, f"{location_name}_map_{resolution}m_{crs_name}_{bands_suffix}{filter_suffix}.png")
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        print(f"RGB visualization saved to: {output_png}")
        
        # Display statistics
        print(f"\nFinal image statistics:")
        print(f"  - Image size: {width} x {height}")
        print(f"  - Number of bands: {num_bands}")
        print(f"  - Resolution: {resolution}m")
        print(f"  - CRS: {target_crs}")
        print(f"  - Output format: {output_repr_format}")
        
        if output_repr_format == 'tiff' and target_crs == 'EPSG:4326':
            print(f"  - Geographic bounds: {west:.6f}°, {south:.6f}°, {east:.6f}°, {north:.6f}°")
            print(f"  - Longitude span: {east - west:.2f}°")
            print(f"  - Latitude span: {north - south:.2f}°")
        elif output_repr_format == 'tiff':
            print(f"  - Projected bounds: {west:.2f}m, {south:.2f}m, {east:.2f}m, {north:.2f}m")
            print(f"  - Width: {(east - west)/1000:.2f} km")
            print(f"  - Height: {(north - south)/1000:.2f} km")
        
        # RGB statistics
        print(f"  - RGB Min values: {np.min(rgb_array, axis=(0,1))}")
        print(f"  - RGB Max values: {np.max(rgb_array, axis=(0,1))}")
        
        mean_rgb = np.mean(rgb_array, axis=(0,1))
        print(f"  - RGB Mean values: [{mean_rgb[0]:.2f}, {mean_rgb[1]:.2f}, {mean_rgb[2]:.2f}]")
        
        coverage = np.sum(np.any(rgb_array > 0, axis=2)) / (height * width) * 100
        print(f"  - Coverage: {coverage:.2f}%")
        
        # Check if scales were processed
        if os.path.exists(scales_output_path):
            scales_data = np.load(scales_output_path, mmap_mode='r')
            print(f"\nScales statistics:")
            print(f"  - Shape: {scales_data.shape}")
            print(f"  - Dtype: {scales_data.dtype}")
            print(f"  - Min: {np.min(scales_data)}")
            print(f"  - Max: {np.max(scales_data)}")
            print(f"  - Mean: {np.mean(scales_data):.6f}")
        
        plt.show()
        
    finally:
        # Clean up temporary files
        print("\nCleaning up temporary files...")
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # The code will automatically determine the best projection
    main(
        base_dir="/scratch/zf281/btfm_representation/senegal/representation/2021", 
        output_dir="/scratch/zf281/btfm_representation/senegal/representation",
        downsample_factor=1, # 1 means no downsampling (10m resolution)
        rgb_only=False, # For quick testing, set to True to only use RGB bands. If False, it will use all 128 bands.
        filter_cross_dateline_tiles=True,  # This will filter out those tiles which cross the dateline
        force_projected=True,  # This ensures we use UTM instead of WGS84
        output_repr_format='tiff',  # Can be 'tiff' or 'npy'. Usually 'npy' mode is much faster...
    )