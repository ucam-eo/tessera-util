#!/usr/bin/env python3
"""
Grid Data Downloader with Shapefile Intersection - Optimized Version
Downloads grid data from remote server based on shapefile intersection using rsync
Now includes TIFF file downloads alongside NPY files
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
    
    def __init__(self, hostnames=None, username=None):
        """
        Initialize the downloader
        
        Args:
            hostnames: List of remote server addresses, defaults to ['myrina.cl.cam.ac.uk', 'antiope.cl.cam.ac.uk']
            username: SSH username
        """
        if hostnames is None:
            hostnames = ['myrina.cl.cam.ac.uk', 'antiope.cl.cam.ac.uk']
        self.hostnames = hostnames
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
        Check which grid/year combinations exist on remote servers
        
        Args:
            grids: List of (lon, lat) tuples
            years: List of years
            
        Returns:
            dict: Mapping of server -> year -> list of existing grids
        """
        logger.info("Checking which grids exist on remote servers...")
        
        server_grids = {}
        total_checks = len(years) * len(grids) * len(self.hostnames)
        
        with tqdm(total=total_checks, desc="Checking remote grids") as pbar:
            for hostname in self.hostnames:
                server_grids[hostname] = {}
                logger.info(f"Checking server: {hostname}")
                
                for year in years:
                    server_grids[hostname][year] = []
                    
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
                            f'{self.username}@{hostname}',
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
                                            server_grids[hostname][year].append((lon, lat))
                                            
                        except subprocess.TimeoutExpired:
                            logger.warning(f"Timeout checking grids for year {year} on {hostname}")
                        except Exception as e:
                            logger.warning(f"Error checking grids for year {year} on {hostname}: {e}")
                        
                    pbar.update(len(grids))
                    
        # Summary
        for hostname in self.hostnames:
            total_existing = sum(len(grids) for grids in server_grids[hostname].values())
            logger.info(f"Server {hostname}: Found {total_existing} existing grid/year combinations")
    def get_file_mtime_ssh(self, remote_path, hostname):
        """Get modification time of remote file via SSH"""
        try:
            ssh_cmd = [
                'ssh',
                f'{self.username}@{hostname}',
                f'stat -c %Y "{remote_path}" 2>/dev/null || echo "0"'
            ]
            
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
            return 0
        except:
            return 0
            
    def find_best_server_for_grid(self, year, lon, lat, server_grids):
        """
        Find the best server for a specific grid by comparing modification times
        
        Args:
            year: Year to check
            lon, lat: Grid coordinates
            server_grids: Dict mapping server -> year -> list of grids
            
        Returns:
            str: Best hostname or None if not found on any server
        """
        best_server = None
        latest_mtime = 0
        
        grid_tuple = (str(lon), str(lat))
        remote_grid_path = f"{self.remote_data_base}/{year}/grid_{lon}_{lat}"
        
        for hostname in self.hostnames:
            # Check if this server has data and the specific year/grid exists
            if (hostname in server_grids and 
                server_grids[hostname] and 
                year in server_grids[hostname] and 
                grid_tuple in server_grids[hostname][year]):
                
                mtime = self.get_file_mtime_ssh(remote_grid_path, hostname)
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    best_server = hostname
                    
        return best_server
        
    def consolidate_grids_by_best_server(self, server_grids):
        """
        Consolidate grids by selecting the best server for each grid/year combination
        
        Args:
            server_grids: Dict mapping server -> year -> list of grids
            
        Returns:
            dict: Mapping of server -> year -> list of grids (optimized)
        """
        logger.info("Consolidating grids by selecting best servers...")
        
        # Create consolidated structure
        consolidated = {hostname: {} for hostname in self.hostnames}
        
        # Get all unique year/grid combinations
        all_combinations = set()
        for hostname in self.hostnames:
            if hostname in server_grids and server_grids[hostname]:
                for year, grids in server_grids[hostname].items():
                    if grids:  # Only add if grids list is not empty
                        for lon, lat in grids:
                            all_combinations.add((year, lon, lat))
        
        # For each combination, find the best server
        for year, lon, lat in tqdm(all_combinations, desc="Finding best servers for grids"):
            best_server = self.find_best_server_for_grid(year, lon, lat, server_grids)
            if best_server:
                if year not in consolidated[best_server]:
                    consolidated[best_server][year] = []
                consolidated[best_server][year].append((lon, lat))
                
        # Remove empty entries
        final_consolidated = {}
        for hostname in self.hostnames:
            if consolidated[hostname] and any(consolidated[hostname].values()):
                final_consolidated[hostname] = {year: grids for year, grids in consolidated[hostname].items() if grids}
                
        # Summary
        for hostname, years_data in final_consolidated.items():
            total_grids = sum(len(grids) for grids in years_data.values())
            logger.info(f"Server {hostname}: {total_grids} grids selected as best source")
            
        return final_consolidated
        
    def download_tiff_files(self, consolidated_grids, output_base):
        """
        Download TIFF files for all grids from their respective best servers
        
        Args:
            consolidated_grids: Dict mapping server -> year -> list of grids
            output_base: Base output directory
        """
        logger.info("Downloading TIFF files...")
        
        # Get unique grids across all servers and years
        unique_grids = set()
        for hostname_data in consolidated_grids.values():
            for grids in hostname_data.values():
                unique_grids.update(grids)
        
        if not unique_grids:
            logger.warning("No grids to download TIFF files for!")
            return
            
        logger.info(f"Downloading TIFF files for {len(unique_grids)} unique grids")
        
        # Group grids by server for TIFF downloads
        server_tiff_grids = {}
        for hostname, years_data in consolidated_grids.items():
            grids_for_server = set()
            for grids in years_data.values():
                grids_for_server.update(grids)
            if grids_for_server:
                server_tiff_grids[hostname] = list(grids_for_server)
        
        # Download TIFF files from each server
        for hostname, grids in server_tiff_grids.items():
            logger.info(f"Downloading {len(grids)} TIFF files from {hostname}")
            
            # Create temporary directory for file lists
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create file list for TIFF files
                file_list_path = os.path.join(temp_dir, "tiff_files.txt")
                
                # Write TIFF file names to list
                with open(file_list_path, 'w') as f:
                    for lon, lat in grids:
                        f.write(f"grid_{lon}_{lat}.tiff\n")
                
                # Download all TIFF files to a temporary location first
                temp_tiff_dir = os.path.join(temp_dir, "tiffs")
                os.makedirs(temp_tiff_dir, exist_ok=True)
                
                # Construct rsync command
                source = f"{self.username}@{hostname}:{self.remote_tiff_path}/"
                
                rsync_cmd = [
                    'rsync',
                    '-avz',  # archive, verbose, compress
                    '--no-group',  # don't preserve group ownership
                    '--files-from', file_list_path,
                    source,
                    temp_tiff_dir
                ]
                
                # Execute rsync
                try:
                    logger.info(f"Running rsync for TIFF files from {hostname}...")
                    
                    process = subprocess.Popen(
                        rsync_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    # Process output in real-time
                    tiff_count = 0
                    while True:
                        output = process.stdout.readline()
                        if output == '' and process.poll() is not None:
                            break
                        if output and '.tiff' in output:
                            match = re.search(r'grid_-?\d+\.?\d*_-?\d+\.?\d*\.tiff', output)
                            if match:
                                tiff_count += 1
                                if tiff_count <= 3:
                                    tqdm.write(f"  ✓ Downloaded TIFF from {hostname}: {match.group()}")
                    
                    # Wait for process to complete
                    process.wait()
                    
                    if process.returncode != 0:
                        stderr = process.stderr.read()
                        if process.returncode == 23 and "some files/attrs were not transferred" in stderr:
                            logger.warning(f"TIFF files transferred from {hostname} but with permission warnings")
                        else:
                            logger.error(f"rsync failed for TIFF files from {hostname}: {stderr}")
                            continue
                    
                    if tiff_count > 3:
                        tqdm.write(f"  ... and {tiff_count - 3} more TIFF files from {hostname}")
                        
                    # Now copy TIFF files to their respective grid folders
                    logger.info(f"Copying TIFF files from {hostname} to grid folders...")
                    
                    # Map grids to their years for this server
                    grid_to_years = {}
                    for year, year_grids in consolidated_grids[hostname].items():
                        for lon, lat in year_grids:
                            if (lon, lat) not in grid_to_years:
                                grid_to_years[(lon, lat)] = []
                            grid_to_years[(lon, lat)].append(year)
                    
                    with tqdm(total=sum(len(years) for years in grid_to_years.values()), 
                             desc=f"Copying TIFF files from {hostname}") as pbar:
                        for (lon, lat), years in grid_to_years.items():
                            tiff_filename = f"grid_{lon}_{lat}.tiff"
                            source_tiff = os.path.join(temp_tiff_dir, tiff_filename)
                            
                            if os.path.exists(source_tiff):
                                # Copy to each year directory where this grid exists
                                for year in years:
                                    dest_dir = os.path.join(output_base, str(year), f"grid_{lon}_{lat}")
                                    if os.path.exists(dest_dir):
                                        dest_tiff = os.path.join(dest_dir, tiff_filename)
                                        shutil.copy2(source_tiff, dest_tiff)
                                        pbar.update(1)
                            else:
                                logger.warning(f"TIFF file not found: {tiff_filename}")
                                pbar.update(len(years))
                                
                except Exception as e:
                    logger.error(f"Error downloading TIFF files from {hostname}: {e}")
                
    def download_with_rsync(self, consolidated_grids, output_base):
        """
        Download grids using rsync from multiple servers for better performance
        
        Args:
            consolidated_grids: Dict mapping server -> year -> list of grids
            output_base: Base output directory
        """
        logger.info("Starting multi-server rsync download process...")
        logger.info("Note: Using --no-group flag to avoid permission errors")
        
        # Create output directory
        os.makedirs(output_base, exist_ok=True)
        
        # Count total grids to download
        total_grids = sum(sum(len(grids) for grids in server_data.values()) 
                         for server_data in consolidated_grids.values())
        
        if total_grids == 0:
            logger.warning("No grids to download!")
            return
            
        logger.info(f"Downloading {total_grids} grids from {len(consolidated_grids)} servers")
        
        # Create temporary directory for file lists
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download from each server separately
            downloaded_count = 0
            with tqdm(total=total_grids, desc="Downloading grids") as pbar:
                for hostname, years_data in consolidated_grids.items():
                    if not years_data:
                        continue
                        
                    server_total = sum(len(grids) for grids in years_data.values())
                    logger.info(f"Downloading {server_total} grids from {hostname}")
                    
                    # Download each year separately for this server
                    for year, grids in years_data.items():
                        if not grids:
                            continue
                            
                        logger.info(f"Downloading {len(grids)} grids for year {year} from {hostname}")
                        
                        # Create file list for this year
                        file_list_path = os.path.join(temp_dir, f"files_{hostname}_{year}.txt")
                        with open(file_list_path, 'w') as f:
                            for lon, lat in grids:
                                # Write relative path from year directory
                                f.write(f"grid_{lon}_{lat}/\n")
                        
                        # Construct rsync command
                        source = f"{self.username}@{hostname}:{self.remote_data_base}/{year}/"
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
                            logger.debug(f"Running rsync from {hostname}: {' '.join(rsync_cmd)}")
                            
                            # Run rsync and capture output
                            process = subprocess.Popen(
                                rsync_cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True
                            )
                            
                            # Process output in real-time
                            grid_count_this_batch = 0
                            while True:
                                output = process.stdout.readline()
                                if output == '' and process.poll() is not None:
                                    break
                                if output and 'grid_' in output:
                                    # Extract grid name from output - look for directory transfers
                                    match = re.search(r'(grid_-?\d+\.?\d*_-?\d+\.?\d*)/?', output)
                                    if match:
                                        grid_count_this_batch += 1
                                        if grid_count_this_batch <= 3:  # Show first 3 for each batch
                                            tqdm.write(f"  ✓ Downloaded from {hostname}: {year}/{match.group(1)}")
                                        pbar.update(1)
                                        downloaded_count += 1
                            
                            # Wait for process to complete
                            process.wait()
                            
                            # Check for errors
                            if process.returncode != 0:
                                stderr = process.stderr.read()
                                # Check if it's just permission errors (code 23) with successful transfers
                                if process.returncode == 23 and "some files/attrs were not transferred" in stderr:
                                    logger.warning(f"Files transferred for year {year} from {hostname} but with permission warnings")
                                    if grid_count_this_batch > 3:
                                        tqdm.write(f"  ... and {grid_count_this_batch - 3} more grids for year {year} from {hostname}")
                                else:
                                    logger.error(f"rsync failed for year {year} from {hostname}: {stderr}")
                            else:
                                if grid_count_this_batch > 3:
                                    tqdm.write(f"  ... and {grid_count_this_batch - 3} more grids for year {year} from {hostname}")
                                
                        except Exception as e:
                            logger.error(f"Error downloading year {year} from {hostname}: {e}")
        
        # Now download TIFF files
        self.download_tiff_files(consolidated_grids, output_base)
                        
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
    /maps/zf281/btfm4rs/cci_workshop_roi_shp/Rudiyanto_Malaysia.shp \
    -o /maps/zf281/btfm4rs/data/external-request/Malaysia \
    -y 2024
    """
    )
    
    parser.add_argument('shapefile', help='Input shapefile path')
    parser.add_argument('-o', '--output', required=True, help='Output directory path')
    parser.add_argument('-y', '--years', nargs='+', type=int, required=True,
                        help='Years to download (e.g., 2023 2024)')
    parser.add_argument('--hostname', default=None,
                        help='Remote server hostname (default: use both myrina and antiope)')
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
    logger.info("GRID DATA DOWNLOADER (OPTIMIZED WITH TIFF)")
    logger.info("="*60)
    logger.info(f"Shapefile: {args.shapefile}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Years to download: {args.years}")
    logger.info(f"Server: {args.username}@{args.hostname}")
    logger.info("="*60)
    
    # Start timer
    start_time = time.time()
    
    # Create downloader instance
    hostnames = [args.hostname] if args.hostname else None
    downloader = OptimizedGridDownloader(
        hostnames=hostnames,
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
        
        # Check which grids exist on remote servers
        server_grids = downloader.check_remote_grids_exist(
            intersecting_grids, args.years
        )
        
        # Consolidate grids by best server
        consolidated_grids = downloader.consolidate_grids_by_best_server(server_grids)
        
        # Download using rsync
        downloader.download_with_rsync(consolidated_grids, args.output)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Final summary
        logger.info("="*60)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("="*60)
        total_downloaded = sum(sum(len(grids) for grids in server_data.values()) 
                              for server_data in consolidated_grids.values())
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