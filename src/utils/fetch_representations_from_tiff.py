#!/usr/bin/env python3
"""
Grid Data Downloader with TIFF Intersection - Optimized Version
Downloads grid data from remote server based on TIFF bounds intersection using rsync
Now includes TIFF file downloads alongside NPY files
"""

import os
import sys
import re
import numpy as np
import rasterio
import rasterio.warp
from rasterio.features import bounds as feature_bounds
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
import paramiko
import threading

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

# 线程锁，用于并发下载时的日志输出
print_lock = threading.Lock()

class OptimizedGridDownloader:
    """Optimized class for downloading grid data based on TIFF intersection"""
    
    def __init__(self, hostnames=None, username=None, password=None, key_filename=None):
        """
        Initialize the downloader
        
        Args:
            hostnames: 远程服务器地址列表，默认为['myrina.cl.cam.ac.uk', 'antiope.cl.cam.ac.uk']
            username: SSH username
            password: SSH密码（可选）
            key_filename: SSH密钥文件路径（可选）
        """
        if hostnames is None:
            hostnames = ['myrina.cl.cam.ac.uk', 'antiope.cl.cam.ac.uk']
        self.hostnames = hostnames
        self.username = username
        self.password = password
        self.key_filename = key_filename
        self.connections = {}  # 存储多个连接
        self.remote_tiff_path = "/tank/zf281/global_0.1_degree_tiff_all"
        self.remote_data_base = "/tank/zf281/global_0.1_degree_representation"
        self.grid_size = 0.1  # Grid size in degrees
        
    def load_tiff(self, tiff_path):
        """
        Load TIFF file and get its bounds in WGS84
        
        Args:
            tiff_path: Path to TIFF file
            
        Returns:
            tuple: (dataset info dict, bounds in WGS84)
        """
        logger.info(f"Loading TIFF: {tiff_path}")
        
        with rasterio.open(tiff_path) as src:
            # Get original CRS and bounds
            original_crs = src.crs
            original_bounds = src.bounds
            
            logger.info(f"TIFF CRS: {original_crs}")
            logger.info(f"TIFF shape: {src.shape} (height, width)")
            logger.info(f"TIFF data type: {src.dtypes[0]}")
            
            # Transform bounds to WGS84 if necessary
            if original_crs != rasterio.crs.CRS.from_epsg(4326):
                logger.info(f"Converting bounds from {original_crs} to WGS84")
                wgs84_bounds = rasterio.warp.transform_bounds(
                    original_crs, 
                    rasterio.crs.CRS.from_epsg(4326), 
                    *original_bounds
                )
            else:
                wgs84_bounds = original_bounds
            
            # Extract bounds (left, bottom, right, top)
            min_lon, min_lat, max_lon, max_lat = wgs84_bounds
            
            logger.info(f"TIFF bounds (WGS84): Lon [{min_lon:.6f}, {max_lon:.6f}], Lat [{min_lat:.6f}, {max_lat:.6f}]")
            
            # Create dataset info dictionary
            dataset_info = {
                'path': tiff_path,
                'crs': original_crs,
                'bounds_original': original_bounds,
                'bounds_wgs84': wgs84_bounds,
                'shape': src.shape,
                'dtype': src.dtypes[0],
                'nodata': src.nodata,
                'count': src.count
            }
        
        return dataset_info, wgs84_bounds
        
    def connect(self, hostname):
        """建立到指定服务器的SSH连接"""
        if hostname in self.connections:
            return self.connections[hostname]
            
        try:
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # 连接参数
            connect_params = {
                'hostname': hostname,
                'username': self.username,
                'timeout': 30
            }
            
            if self.key_filename:
                connect_params['key_filename'] = self.key_filename
            elif self.password:
                connect_params['password'] = self.password
            else:
                # 尝试使用默认SSH密钥
                connect_params['look_for_keys'] = True
                
            ssh_client.connect(**connect_params)
            sftp_client = ssh_client.open_sftp()
            
            self.connections[hostname] = {
                'ssh': ssh_client,
                'sftp': sftp_client
            }
            
            logger.info(f"成功连接到 {hostname}")
            return self.connections[hostname]
            
        except Exception as e:
            logger.error(f"连接到 {hostname} 失败: {e}")
            return None
            
    def disconnect(self):
        """断开所有SSH连接"""
        for hostname, connection in self.connections.items():
            if connection['sftp']:
                connection['sftp'].close()
            if connection['ssh']:
                connection['ssh'].close()
            logger.info(f"已断开与 {hostname} 的连接")
        self.connections.clear()
        
    def get_remote_file_size(self, remote_path, hostname):
        """获取远程文件的大小"""
        connection = self.connect(hostname)
        if not connection:
            return None
            
        try:
            stat = connection['sftp'].stat(remote_path)
            return stat.st_size
        except:
            return None
    
    def get_file_mtime(self, remote_path, hostname):
        """获取远程文件的修改时间"""
        connection = self.connect(hostname)
        if not connection:
            return None
            
        try:
            stat = connection['sftp'].stat(remote_path)
            return stat.st_mtime
        except:
            return None
    
    def is_local_file_complete(self, local_path, remote_path, hostname):
        """
        检查本地文件是否完整（通过比较文件大小）
        
        Args:
            local_path: 本地文件路径
            remote_path: 远程文件路径
            hostname: 远程服务器主机名
            
        Returns:
            bool: True表示本地文件完整，False表示不完整或不存在
        """
        # 检查本地文件是否存在
        if not os.path.exists(local_path):
            return False
            
        try:
            # 获取本地文件大小
            local_size = os.path.getsize(local_path)
            
            # 获取远程文件大小
            remote_size = self.get_remote_file_size(remote_path, hostname)
            
            if remote_size is None:
                logger.warning(f"无法获取远程文件大小: {remote_path}")
                return False
                
            # 比较文件大小
            if local_size == remote_size:
                return True
            else:
                logger.info(f"文件大小不匹配 {local_path}: 本地={local_size}, 远程={remote_size}")
                return False
                
        except Exception as e:
            logger.error(f"检查文件完整性时出错 {local_path}: {e}")
            return False
            
    def find_best_server_for_file(self, remote_path):
        """
        在所有服务器上查找文件，返回最新修改时间的服务器和时间戳
        
        Args:
            remote_path: 远程文件路径
            
        Returns:
            tuple: (最优服务器主机名, 最新修改时间 in Unix timestamp), or (None, 0)
        """
        best_server = None
        latest_mtime = 0
        
        for hostname in self.hostnames:
            mtime = self.get_file_mtime(remote_path, hostname)
            if mtime and mtime > latest_mtime:
                latest_mtime = mtime
                best_server = hostname
                
        return best_server, latest_mtime
        
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
        
    def find_intersecting_grids(self, tiff_bounds, buffer_degrees=0.1, include_edge_grids=True):
        """
        Find all grid cells that intersect with the TIFF bounds
        
        Args:
            tiff_bounds: TIFF bounds in WGS84 (min_lon, min_lat, max_lon, max_lat)
            buffer_degrees: Buffer to add around bounds  
            include_edge_grids: Whether to include grids that only touch the edge
            
        Returns:
            list: List of (lon, lat) tuples for intersecting grids
        """
        logger.info("Finding intersecting grid cells based on TIFF bounds...")
        
        # Generate potential grids based on bounds
        potential_grids = self.generate_potential_grids(tiff_bounds, buffer_degrees)
        
        intersecting_grids = []
        min_lon, min_lat, max_lon, max_lat = tiff_bounds
        
        # Create TIFF bounding box
        tiff_box = box(min_lon, min_lat, max_lon, max_lat)
        
        # Check each potential grid for intersection
        for center_lon, center_lat in tqdm(potential_grids, desc="Checking grid intersections"):
            # Create grid cell polygon (0.1 x 0.1 degrees centered at the grid center)
            grid_box = box(
                center_lon - 0.05,
                center_lat - 0.05,
                center_lon + 0.05,
                center_lat + 0.05
            )
            
            # Check intersection with TIFF bounds
            if include_edge_grids:
                # Include grids that intersect or touch the TIFF bounds
                intersects = tiff_box.intersects(grid_box)
            else:
                # Only include grids that have substantial overlap
                intersects = tiff_box.intersects(grid_box) and not tiff_box.touches(grid_box)
            
            if intersects:
                # Format coordinates to match file naming convention
                lon_str = f"{center_lon:.2f}"
                lat_str = f"{center_lat:.2f}"
                intersecting_grids.append((lon_str, lat_str))
                
        logger.info(f"Found {len(intersecting_grids)} grids intersecting with TIFF bounds")
        
        # Debug: show first few intersecting grids
        if intersecting_grids:
            logger.debug(f"First few intersecting grids: {intersecting_grids[:5]}")
            
        return intersecting_grids
        
    def check_remote_grids_exist(self, grids, years):
        """
        Check which grid/year combinations exist on remote server with multi-server support
        
        Args:
            grids: List of (lon, lat) tuples
            years: List of years
            
        Returns:
            dict: Mapping of year to list of existing grids
        """
        logger.info("Checking which grids exist on remote servers...")
        
        existing_grids = {}
        total_checks = len(years) * len(grids)
        
        with tqdm(total=total_checks, desc="Checking remote grids") as pbar:
            for year in years:
                existing_grids[year] = []
                
                # Try each server to find grids
                for hostname in self.hostnames:
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
                                            # Avoid duplicates from multiple servers
                                            if (lon, lat) not in existing_grids[year]:
                                                existing_grids[year].append((lon, lat))
                                                
                        except subprocess.TimeoutExpired:
                            logger.warning(f"Timeout checking grids for year {year} on {hostname}")
                        except Exception as e:
                            logger.warning(f"Error checking grids for year {year} on {hostname}: {e}")
                    
                    # If we found grids on this server, we can break (assuming all servers have the same data)
                    if existing_grids[year]:
                        logger.debug(f"Found grids for year {year} on {hostname}")
                        break
                
                pbar.update(len(grids))
                
        # Summary
        total_existing = sum(len(grids) for grids in existing_grids.values())
        logger.info(f"Found {total_existing} existing grid/year combinations")
        
        # Debug: show some existing grids
        for year, grids in existing_grids.items():
            if grids:
                logger.debug(f"Year {year}: Found {len(grids)} grids (first few: {grids[:3]})")
        
        return existing_grids
        
    def download_tiff_files(self, existing_grids, output_base):
        """
        Download TIFF files for all grids with integrity checks and multi-server support
        
        Args:
            existing_grids: Dict mapping year to list of (lon, lat) tuples
            output_base: Base output directory
        """
        logger.info("Downloading TIFF files with integrity verification...")
        
        # Get unique grids across all years
        unique_grids = set()
        for grids in existing_grids.values():
            unique_grids.update(grids)
        
        if not unique_grids:
            logger.warning("No grids to download TIFF files for!")
            return
            
        logger.info(f"Downloading TIFF files for {len(unique_grids)} unique grids")
        
        # Map to store which years each grid appears in
        grid_to_years = {}
        for year, grids in existing_grids.items():
            for lon, lat in grids:
                if (lon, lat) not in grid_to_years:
                    grid_to_years[(lon, lat)] = []
                grid_to_years[(lon, lat)].append(year)
        
        # Group grids by best server for efficient downloading
        server_grids = {}
        for lon, lat in tqdm(unique_grids, desc="Finding best servers for TIFF files"):
            tiff_filename = f"grid_{lon}_{lat}.tiff"
            remote_path = f"{self.remote_tiff_path}/{tiff_filename}"
            
            best_server, _ = self.find_best_server_for_file(remote_path)
            if best_server:
                if best_server not in server_grids:
                    server_grids[best_server] = []
                server_grids[best_server].append((lon, lat))
            else:
                logger.warning(f"TIFF file not found on any server: {tiff_filename}")
        
        if not server_grids:
            logger.error("No TIFF files found on any server!")
            return
        
        # Check which TIFF files already exist and are complete
        files_to_download = {}
        for server, grids in server_grids.items():
            files_to_download[server] = []
            
            for lon, lat in grids:
                tiff_filename = f"grid_{lon}_{lat}.tiff"
                remote_path = f"{self.remote_tiff_path}/{tiff_filename}"
                
                # Check all destination paths for this grid
                needs_download = False
                for year in grid_to_years[(lon, lat)]:
                    dest_dir = os.path.join(output_base, str(year), f"grid_{lon}_{lat}")
                    dest_tiff = os.path.join(dest_dir, tiff_filename)
                    
                    if not self.is_local_file_complete(dest_tiff, remote_path, server):
                        needs_download = True
                        break
                
                if needs_download:
                    files_to_download[server].append((lon, lat))
        
        # Download files from each server
        for server, grids in files_to_download.items():
            if not grids:
                continue
                
            logger.info(f"Downloading {len(grids)} TIFF files from {server}")
            
            # Create temporary directory for this server's downloads
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
                source = f"{self.username}@{server}:{self.remote_tiff_path}/"
                
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
                    logger.info(f"Running rsync for TIFF files from {server}...")
                    
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
                                if tiff_count <= 5:
                                    with print_lock:
                                        tqdm.write(f"  ✓ Downloaded TIFF: {match.group()}")
                    
                    # Wait for process to complete
                    process.wait()
                    
                    if process.returncode != 0:
                        stderr = process.stderr.read()
                        if process.returncode == 23 and "some files/attrs were not transferred" in stderr:
                            logger.warning("TIFF files transferred but with permission warnings (this is usually harmless)")
                        else:
                            logger.error(f"rsync failed for TIFF files from {server}: {stderr}")
                            continue
                    
                    if tiff_count > 5:
                        with print_lock:
                            tqdm.write(f"  ... and {tiff_count - 5} more TIFF files")
                        
                    # Now copy TIFF files to their respective grid folders with verification
                    logger.info(f"Copying and verifying TIFF files from {server}...")
                    
                    copy_count = 0
                    for lon, lat in grids:
                        tiff_filename = f"grid_{lon}_{lat}.tiff"
                        source_tiff = os.path.join(temp_tiff_dir, tiff_filename)
                        remote_path = f"{self.remote_tiff_path}/{tiff_filename}"
                        
                        if os.path.exists(source_tiff):
                            # Verify downloaded file integrity
                            if not self.is_local_file_complete(source_tiff, remote_path, server):
                                logger.error(f"Downloaded TIFF file is incomplete: {tiff_filename}")
                                continue
                            
                            # Copy to each year directory where this grid exists
                            for year in grid_to_years[(lon, lat)]:
                                dest_dir = os.path.join(output_base, str(year), f"grid_{lon}_{lat}")
                                if os.path.exists(dest_dir):
                                    dest_tiff = os.path.join(dest_dir, tiff_filename)
                                    
                                    # Remove incomplete local file if exists
                                    if os.path.exists(dest_tiff) and not self.is_local_file_complete(dest_tiff, remote_path, server):
                                        logger.info(f"Removing incomplete TIFF file: {dest_tiff}")
                                        os.remove(dest_tiff)
                                    
                                    # Copy if not already complete
                                    if not os.path.exists(dest_tiff):
                                        shutil.copy2(source_tiff, dest_tiff)
                                        copy_count += 1
                                        
                                        # Verify copied file
                                        if not self.is_local_file_complete(dest_tiff, remote_path, server):
                                            logger.error(f"Failed to copy TIFF file correctly: {dest_tiff}")
                                            os.remove(dest_tiff)
                        else:
                            logger.warning(f"TIFF file not downloaded: {tiff_filename}")
                    
                    logger.info(f"Successfully copied {copy_count} TIFF files from {server}")
                            
                except Exception as e:
                    logger.error(f"Error downloading TIFF files from {server}: {e}")
        
        # Final verification
        logger.info("Performing final verification of all TIFF files...")
        verification_failed = 0
        verification_passed = 0
        
        for (lon, lat), years in grid_to_years.items():
            tiff_filename = f"grid_{lon}_{lat}.tiff"
            remote_path = f"{self.remote_tiff_path}/{tiff_filename}"
            
            # Find the best server for this file
            best_server, _ = self.find_best_server_for_file(remote_path)
            if not best_server:
                continue
            
            for year in years:
                dest_dir = os.path.join(output_base, str(year), f"grid_{lon}_{lat}")
                dest_tiff = os.path.join(dest_dir, tiff_filename)
                
                if os.path.exists(dest_tiff):
                    if self.is_local_file_complete(dest_tiff, remote_path, best_server):
                        verification_passed += 1
                    else:
                        verification_failed += 1
                        logger.error(f"TIFF file verification failed: {dest_tiff}")
                else:
                    verification_failed += 1
                    logger.error(f"TIFF file missing: {dest_tiff}")
        
        logger.info(f"TIFF verification complete: {verification_passed} passed, {verification_failed} failed")
                
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
        
        # Now download TIFF files
        self.download_tiff_files(existing_grids, output_base)
                        
        logger.info(f"Download process completed! Total grids downloaded: {downloaded_count}")
        
def main():
    parser = argparse.ArgumentParser(
        description='Download grid data based on TIFF bounds intersection (Optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python fetch_representations_from_tiff.py input.tiff -o /path/to/output -y 2023 2024
    python fetch_representations_from_tiff.py input.tiff -o /path/to/output -y 2019 2020 2021 2022 2023 2024
    
    /maps/zf281/miniconda3/envs/detectree-env/bin/python \
    /maps/zf281/btfm4rs/src/utils/fetch_representations_from_tiff.py \
    /maps/zf281/btfm4rs/uluguru_Tanzania_Neil_bounding_box.tif \
    -y 2024 \
    -o /scratch/zf281/btfm_representation/uluguru_Tanzania
    """
    )
    
    parser.add_argument('tiff', help='Input TIFF file path')
    parser.add_argument('-o', '--output', required=True, help='Output directory path')
    parser.add_argument('-y', '--years', nargs='+', type=int, required=True,
                        help='Years to download (e.g., 2023 2024)')
    parser.add_argument('--hostname', default='antiope.cl.cam.ac.uk',
                        help='Remote server hostname (default: antiope.cl.cam.ac.uk)')
    parser.add_argument('--username', default='zf281',
                        help='SSH username (default: zf281)')
    parser.add_argument('--buffer', type=float, default=0.1,
                        help='Buffer in degrees around TIFF bounds (default: 0.1)')
    parser.add_argument('--no-edge-grids', action='store_true',
                        help='Exclude grids that only touch the edge of TIFF bounds')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Print header
    logger.info("="*60)
    logger.info("GRID DATA DOWNLOADER FROM TIFF (OPTIMIZED WITH TIFF)")
    logger.info("="*60)
    logger.info(f"TIFF file: {args.tiff}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Years to download: {args.years}")
    logger.info(f"Server: {args.username}@{args.hostname}")
    logger.info(f"Buffer: {args.buffer} degrees")
    logger.info(f"Include edge grids: {not args.no_edge_grids}")
    logger.info("="*60)
    
    # Start timer
    start_time = time.time()
    
    # Create downloader instance
    downloader = OptimizedGridDownloader(
        hostname=args.hostname,
        username=args.username
    )
    
    try:
        # Load TIFF
        tiff_info, bounds = downloader.load_tiff(args.tiff)
        
        # Find intersecting grids (no remote connection needed!)
        intersecting_grids = downloader.find_intersecting_grids(
            bounds, 
            buffer_degrees=args.buffer,
            include_edge_grids=not args.no_edge_grids
        )
        
        if not intersecting_grids:
            logger.warning("No intersecting grids found!")
            return
            
        # Save grid list and TIFF info for reference
        os.makedirs(args.output, exist_ok=True)
        
        # Save grid list
        grid_list_file = os.path.join(args.output, "grid_list.txt")
        with open(grid_list_file, 'w') as f:
            for lon, lat in intersecting_grids:
                f.write(f"grid_{lon}_{lat}\n")
        logger.info(f"Saved grid list to: {grid_list_file}")
        
        # Save TIFF info
        tiff_info_file = os.path.join(args.output, "tiff_info.txt")
        with open(tiff_info_file, 'w') as f:
            f.write(f"Source TIFF: {tiff_info['path']}\n")
            f.write(f"Original CRS: {tiff_info['crs']}\n")
            f.write(f"Shape (H, W): {tiff_info['shape']}\n")
            f.write(f"Data type: {tiff_info['dtype']}\n")
            f.write(f"Bands: {tiff_info['count']}\n")
            f.write(f"NoData value: {tiff_info['nodata']}\n")
            f.write(f"Bounds (original): {tiff_info['bounds_original']}\n")
            f.write(f"Bounds (WGS84): {tiff_info['bounds_wgs84']}\n")
            f.write(f"Buffer used: {args.buffer} degrees\n")
            f.write(f"Include edge grids: {not args.no_edge_grids}\n")
            f.write(f"Total intersecting grids: {len(intersecting_grids)}\n")
        logger.info(f"Saved TIFF info to: {tiff_info_file}")
        
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