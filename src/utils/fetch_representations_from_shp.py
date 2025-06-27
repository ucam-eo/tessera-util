#!/usr/bin/env python3
"""
Grid Data Downloader with Shapefile Intersection - Optimized Version
Downloads grid data from remote server based on shapefile intersection using rsync
"""

import os
import sys
import re
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
from datetime import datetime
import time
import subprocess
import tempfile
import shutil

# Setup logging with beautiful formatting
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    grey = "\x1b[38;21m"
    blue = "\x1b[34m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "[%(asctime)s] %(levelname)-8s" + reset + " %(message)s",
        logging.INFO: blue + "[%(asctime)s] %(levelname)-8s" + reset + " %(message)s",
        logging.WARNING: yellow + "[%(asctime)s] %(levelname)-8s" + reset + " %(message)s",
        logging.ERROR: red + "[%(asctime)s] %(levelname)-8s" + reset + " %(message)s",
        logging.CRITICAL: bold_red + "[%(asctime)s] %(levelname)-8s" + reset + " %(message)s"
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter())
logger.addHandler(handler)

class OptimizedGridDownloader:
    """Optimized class for downloading grid data based on shapefile intersection"""
    
    def __init__(self, hostname, username):
        """
        Initialize the downloader
        
        Args:
            hostname: Remote server address
            username: SSH username
        """
        self.hostname = hostname
        self.username = username
        self.remote_tiff_path = "/tank/zf281/global_0.1_degree_tiff_all"
        self.remote_data_base = "/tank/zf281/global_0.1_degree_representation"
        self.grid_size = 0.1  # Grid size in degrees
        
    def load_shapefile(self, shapefile_path):
        """
        Load shapefile and ensure it's in WGS84
        
        Args:
            shapefile_path: Path to shapefile
            
        Returns:
            tuple: (GeoDataFrame, bounds)
        """
        logger.info(f"Loading shapefile: {shapefile_path}")
        gdf = gpd.read_file(shapefile_path)
        
        if gdf.crs != "EPSG:4326":
            logger.info(f"Converting CRS from {gdf.crs} to EPSG:4326")
            gdf = gdf.to_crs("EPSG:4326")
        
        bounds = gdf.total_bounds
        logger.info(f"Shapefile bounds: Lon [{bounds[0]:.4f}, {bounds[2]:.4f}], Lat [{bounds[1]:.4f}, {bounds[3]:.4f}]")
        
        return gdf, bounds
        
    def generate_potential_grids(self, bounds, buffer_degrees=0.1):
        """
        Generate potential grid coordinates based on bounds
        Grid centers are at x.x5 pattern (e.g., -179.95, -179.85, ..., -0.05, 0.05, 0.15, ...)
        
        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
            buffer_degrees: Buffer to add around bounds
            
        Returns:
            list: List of (lon, lat) tuples
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Add buffer
        min_lon -= buffer_degrees
        max_lon += buffer_degrees
        min_lat -= buffer_degrees
        max_lat += buffer_degrees
        
        logger.info(f"Search bounds (with buffer): Lon [{min_lon:.4f}, {max_lon:.4f}], Lat [{min_lat:.4f}, {max_lat:.4f}]")
        
        # Calculate grid cell centers within bounds
        # Grid cells have their lower-left corners at multiples of 0.1
        # Centers are at those values + 0.05
        potential_grids = []
        
        # Find the starting grid cell that contains or is just before min_lon/min_lat
        # Grid cells start at multiples of 0.1, so we floor to nearest 0.1
        lon_start = np.floor(min_lon * 10) / 10
        lon_end = np.ceil(max_lon * 10) / 10
        lat_start = np.floor(min_lat * 10) / 10
        lat_end = np.ceil(max_lat * 10) / 10
        
        # Generate grid centers (add 0.05 to get center from lower-left corner)
        for lon_corner in np.arange(lon_start, lon_end + 0.01, self.grid_size):
            for lat_corner in np.arange(lat_start, lat_end + 0.01, self.grid_size):
                # Center is at corner + 0.05
                center_lon = round(lon_corner + 0.05, 2)
                center_lat = round(lat_corner + 0.05, 2)
                
                # Only include if center is within search bounds
                if (min_lon <= center_lon <= max_lon and 
                    min_lat <= center_lat <= max_lat):
                    potential_grids.append((center_lon, center_lat))
                
        logger.info(f"Generated {len(potential_grids)} potential grid coordinates")
        
        # Debug: show first few grids
        if potential_grids:
            logger.debug(f"First few grid centers: {potential_grids[:5]}")
            
        return potential_grids
        
    def find_intersecting_grids(self, shapefile_gdf, bounds):
        """
        Find all grid cells that intersect with the shapefile
        
        Args:
            shapefile_gdf: Shapefile GeoDataFrame
            bounds: Shapefile bounds
            
        Returns:
            list: List of (lon, lat) tuples for intersecting grids
        """
        logger.info("Finding intersecting grid cells...")
        
        # Generate potential grids based on bounds
        potential_grids = self.generate_potential_grids(bounds)
        
        intersecting_grids = []
        
        # Check each potential grid for intersection
        for center_lon, center_lat in tqdm(potential_grids, desc="Checking grid intersections"):
            # Create grid cell polygon (0.1 x 0.1 degrees centered at the grid center)
            grid_box = box(
                center_lon - 0.05,
                center_lat - 0.05,
                center_lon + 0.05,
                center_lat + 0.05
            )
            
            # Check intersection with shapefile
            if shapefile_gdf.intersects(grid_box).any():
                # Format coordinates to match file naming convention
                lon_str = f"{center_lon:.2f}"
                lat_str = f"{center_lat:.2f}"
                intersecting_grids.append((lon_str, lat_str))
                
        logger.info(f"Found {len(intersecting_grids)} grids intersecting with shapefile")
        
        # Debug: show first few intersecting grids
        if intersecting_grids:
            logger.debug(f"First few intersecting grids: {intersecting_grids[:5]}")
            
        return intersecting_grids
        
    def check_remote_grids_exist(self, grids, years):
        """
        Check which grid/year combinations exist on remote server
        
        Args:
            grids: List of (lon, lat) tuples
            years: List of years
            
        Returns:
            dict: Mapping of year to list of existing grids
        """
        logger.info("Checking which grids exist on remote server...")
        
        existing_grids = {}
        total_checks = len(years) * len(grids)
        
        with tqdm(total=total_checks, desc="Checking remote grids") as pbar:
            for year in years:
                existing_grids[year] = []
                
                # Build SSH command to check multiple grids at once
                grid_names = [f"grid_{lon}_{lat}" for lon, lat in grids]
                
                # Create a command that checks all grids for this year
                remote_year_path = f"{self.remote_data_base}/{year}"
                
                # Split grid names into smaller batches to avoid command line length limits
                batch_size = 50
                for i in range(0, len(grid_names), batch_size):
                    batch = grid_names[i:i+batch_size]
                    check_cmd = f"cd {remote_year_path} 2>/dev/null && ls -d {' '.join(batch)} 2>/dev/null || true"
                    
                    ssh_cmd = [
                        'ssh',
                        f'{self.username}@{self.hostname}',
                        check_cmd
                    ]
                    
                    try:
                        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
                        if result.returncode == 0 and result.stdout:
                            # Parse existing grids from output
                            for line in result.stdout.strip().split('\n'):
                                if line.startswith('grid_'):
                                    # Extract coordinates from grid name
                                    match = re.match(r'grid_(-?\d+\.?\d*)_(-?\d+\.?\d*)', line)
                                    if match:
                                        lon, lat = match.groups()
                                        existing_grids[year].append((lon, lat))
                                        
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Timeout checking grids for year {year}")
                    except Exception as e:
                        logger.warning(f"Error checking grids for year {year}: {e}")
                    
                pbar.update(len(grids))
                
        # Summary
        total_existing = sum(len(grids) for grids in existing_grids.values())
        logger.info(f"Found {total_existing} existing grid/year combinations")
        
        # Debug: show some existing grids
        for year, grids in existing_grids.items():
            if grids:
                logger.debug(f"Year {year}: Found {len(grids)} grids (first few: {grids[:3]})")
        
        return existing_grids
        
    def download_with_rsync(self, existing_grids, output_base):
        """
        Download grids using rsync for better performance
        
        Args:
            existing_grids: Dict mapping year to list of (lon, lat) tuples
            output_base: Base output directory
        """
        logger.info("Starting rsync download process...")
        logger.info("Note: Using --no-group flag to avoid permission errors")
        
        # Create output directory
        os.makedirs(output_base, exist_ok=True)
        
        # Count total grids to download
        total_grids = sum(len(grids) for grids in existing_grids.values())
        
        if total_grids == 0:
            logger.warning("No grids to download!")
            return
            
        # Create temporary directory for file lists
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download each year separately
            downloaded_count = 0
            with tqdm(total=total_grids, desc="Downloading grids") as pbar:
                for year, grids in existing_grids.items():
                    if not grids:
                        continue
                        
                    logger.info(f"Downloading {len(grids)} grids for year {year}")
                    
                    # Create file list for this year
                    file_list_path = os.path.join(temp_dir, f"files_{year}.txt")
                    with open(file_list_path, 'w') as f:
                        for lon, lat in grids:
                            # Write relative path from year directory
                            f.write(f"grid_{lon}_{lat}/\n")
                    
                    # Construct rsync command
                    source = f"{self.username}@{self.hostname}:{self.remote_data_base}/{year}/"
                    target = os.path.join(output_base, str(year))
                    
                    rsync_cmd = [
                        'rsync',
                        '-avz',  # archive, verbose, compress
                        '--no-group',  # don't preserve group ownership
                        '--files-from', file_list_path,
                        source,
                        target
                    ]
                    
                    # Create target directory
                    os.makedirs(target, exist_ok=True)
                    
                    # Execute rsync
                    try:
                        logger.debug(f"Running rsync (without group preservation): {' '.join(rsync_cmd)}")
                        
                        # Run rsync and capture output
                        process = subprocess.Popen(
                            rsync_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )
                        
                        # Process output in real-time
                        grid_count_this_year = 0
                        while True:
                            output = process.stdout.readline()
                            if output == '' and process.poll() is not None:
                                break
                            if output and 'grid_' in output:
                                # Extract grid name from output - look for directory transfers
                                match = re.search(r'(grid_-?\d+\.?\d*_-?\d+\.?\d*)/?', output)
                                if match:
                                    grid_count_this_year += 1
                                    if grid_count_this_year <= 5:  # Show first 5 for each year
                                        tqdm.write(f"  ✓ Downloaded: {year}/{match.group(1)}")
                                    pbar.update(1)
                                    downloaded_count += 1
                        
                        # Wait for process to complete
                        process.wait()
                        
                        # Check for errors
                        if process.returncode != 0:
                            stderr = process.stderr.read()
                            # Check if it's just permission errors (code 23) with successful transfers
                            if process.returncode == 23 and "some files/attrs were not transferred" in stderr:
                                logger.warning(f"Files transferred for year {year} but with permission warnings (this is usually harmless)")
                                if grid_count_this_year > 5:
                                    tqdm.write(f"  ... and {grid_count_this_year - 5} more grids for year {year}")
                            else:
                                logger.error(f"rsync failed for year {year}: {stderr}")
                        else:
                            if grid_count_this_year > 5:
                                tqdm.write(f"  ... and {grid_count_this_year - 5} more grids for year {year}")
                            
                    except Exception as e:
                        logger.error(f"Error downloading year {year}: {e}")
                        
        logger.info(f"Download process completed! Total grids downloaded: {downloaded_count}")
        
def main():
    parser = argparse.ArgumentParser(
        description='Download grid data based on shapefile intersection (Optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python grid_downloader.py input.shp -o /path/to/output -y 2023 2024
    python grid_downloader.py input.shp -o /path/to/output -y 2019 2020 2021 2022 2023 2024
    
    /maps/zf281/miniconda3/envs/detectree-env/bin/python \
    /maps/zf281/btfm4rs/src/utils/fetch_representations_from_shp.py \
    /maps/zf281/btfm4rs/david_cci_workshop_and_maddy_roi/centralsenegal_jcam_bbox.shp \
    -y 2021 \
    -o /maps/zf281/btfm4rs/data/tmp
    """
    )
    
    parser.add_argument('shapefile', help='Input shapefile path')
    parser.add_argument('-o', '--output', required=True, help='Output directory path')
    parser.add_argument('-y', '--years', nargs='+', type=int, required=True,
                        help='Years to download (e.g., 2023 2024)')
    parser.add_argument('--hostname', default='antiope.cl.cam.ac.uk',
                        help='Remote server hostname (default: antiope.cl.cam.ac.uk)')
    parser.add_argument('--username', default='zf281',
                        help='SSH username (default: zf281)')
    parser.add_argument('--buffer', type=float, default=0.1,
                        help='Buffer in degrees around shapefile bounds (default: 0.1)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Print header
    logger.info("="*60)
    logger.info("GRID DATA DOWNLOADER (OPTIMIZED)")
    logger.info("="*60)
    logger.info(f"Shapefile: {args.shapefile}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Years to download: {args.years}")
    logger.info(f"Server: {args.username}@{args.hostname}")
    logger.info("="*60)
    
    # Start timer
    start_time = time.time()
    
    # Create downloader instance
    downloader = OptimizedGridDownloader(
        hostname=args.hostname,
        username=args.username
    )
    
    try:
        # Load shapefile
        shapefile_gdf, bounds = downloader.load_shapefile(args.shapefile)
        
        # Find intersecting grids (no remote connection needed!)
        intersecting_grids = downloader.find_intersecting_grids(
            shapefile_gdf, bounds
        )
        
        if not intersecting_grids:
            logger.warning("No intersecting grids found!")
            return
            
        # Save grid list for reference
        os.makedirs(args.output, exist_ok=True)
        grid_list_file = os.path.join(args.output, "grid_list.txt")
        with open(grid_list_file, 'w') as f:
            for lon, lat in intersecting_grids:
                f.write(f"grid_{lon}_{lat}\n")
        logger.info(f"Saved grid list to: {grid_list_file}")
        
        # Check which grids exist on remote server
        existing_grids = downloader.check_remote_grids_exist(
            intersecting_grids, args.years
        )
        
        # Download using rsync
        downloader.download_with_rsync(existing_grids, args.output)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Final summary
        logger.info("="*60)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("="*60)
        total_downloaded = sum(len(grids) for grids in existing_grids.values())
        logger.info(f"Total grids downloaded: {total_downloaded}")
        logger.info(f"Output directory: {args.output}")
        logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Program failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()