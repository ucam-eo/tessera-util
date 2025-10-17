#!/usr/bin/env python3
"""
Script to download grid data from remote server based on a txt file listing
"""

import os
import sys
import re
import paramiko
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import subprocess
import tempfile
import shutil
from datetime import datetime # 导入 datetime 模块

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 线程锁，用于并发下载时的日志输出
print_lock = threading.Lock()

class GridDataDownloader:
    def __init__(self, hostnames=None, username=None, password=None, key_filename=None):
        """
        初始化下载器
        
        Args:
            hostnames: 远程服务器地址列表，默认为['myrina.cl.cam.ac.uk', 'antiope.cl.cam.ac.uk']
            username: SSH用户名
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
        
    def parse_grid_coordinates(self, txt_path):
        """
        从txt文件中解析grid坐标
        
        Args:
            txt_path: txt文件路径
            
        Returns:
            list: grid坐标列表，格式为 [(lon, lat), ...]
        """
        coordinates = []
        
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and line.endswith('.tiff'):
                        # 使用正则表达式提取坐标
                        match = re.match(r'grid_(-?\d+\.?\d*)_(-?\d+\.?\d*)\.tiff', line)
                        if match:
                            lon, lat = match.groups()
                            coordinates.append((lon, lat))
                        else:
                            logger.warning(f"无法解析行: {line}")
                            
            logger.info(f"从txt文件中解析出 {len(coordinates)} 个坐标")
            return coordinates
            
        except Exception as e:
            logger.error(f"读取txt文件失败: {e}")
            raise
            
    def check_remote_directory_exists(self, remote_path, hostname):
        """检查指定服务器上的远程目录是否存在"""
        connection = self.connect(hostname)
        if not connection:
            return False
            
        try:
            connection['sftp'].stat(remote_path)
            return True
        except:
            return False
            
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
            
    def download_file(self, remote_path, local_path, hostname):
        """从指定服务器下载单个文件"""
        # 首先检查本地文件是否已经完整存在
        if self.is_local_file_complete(local_path, remote_path, hostname):
            logger.info(f"跳过已存在的完整文件: {local_path}")
            return True
            
        connection = self.connect(hostname)
        if not connection:
            return False
            
        try:
            # 确保本地目录存在
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # 如果本地文件存在但不完整，先删除它
            if os.path.exists(local_path):
                logger.info(f"删除不完整的本地文件: {local_path}")
                os.remove(local_path)
            
            # 下载文件
            connection['sftp'].get(remote_path, local_path)
            
            # 下载后再次验证文件完整性
            if self.is_local_file_complete(local_path, remote_path, hostname):
                return True
            else:
                logger.error(f"下载后文件验证失败: {local_path}")
                # 删除损坏的文件
                if os.path.exists(local_path):
                    os.remove(local_path)
                return False
            
        except Exception as e:
            logger.error(f"从 {hostname} 下载文件失败 {remote_path}: {e}")
            # 如果下载失败，删除可能的部分文件
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except:
                    pass
            return False
            
    def download_grid_folder(self, year, lon, lat, local_base_path, best_server):
        """
        从预先确定的最佳服务器下载指定年份和坐标的grid文件夹
        
        Args:
            year: 年份
            lon: 经度
            lat: 纬度
            local_base_path: 本地基础路径
            best_server: 预先确定好的最佳下载服务器
            
        Returns:
            bool: 是否成功下载
        """
        if not best_server:
            return False

        grid_name = f"grid_{lon}_{lat}"
        remote_grid_path = f"/tank/zf281/global_0.1_degree_representation/{year}/{grid_name}"
        
        # 创建本地目录结构
        local_year_path = os.path.join(local_base_path, str(year))
        local_grid_path = os.path.join(local_year_path, grid_name)
        os.makedirs(local_grid_path, exist_ok=True)
        
        # 下载npy文件
        npy_files = [
            f"{grid_name}.npy",
            f"{grid_name}_scales.npy"
        ]
        
        success = True
        for npy_file in npy_files:
            remote_file_path = f"{remote_grid_path}/{npy_file}"
            local_file_path = os.path.join(local_grid_path, npy_file)
            
            # 我们只需要检查文件是否存在，然后下载
            if not self.download_file(remote_file_path, local_file_path, best_server):
                # 如果主要npy文件下载失败，则认为整个grid下载失败
                if npy_file == f"{grid_name}.npy":
                    success = False
                    break # 停止下载该grid的其他文件
                
        return success
        
    def download_tiff_files(self, downloaded_grids, local_base_path):
        """
        Download TIFF files for all successfully downloaded grids from their respective best servers
        
        Args:
            downloaded_grids: Dict mapping (year, lon, lat) -> hostname for successfully downloaded grids
            local_base_path: 本地基础路径
        """
        logger.info("开始下载TIFF文件...")
        
        if not downloaded_grids:
            logger.warning("没有成功下载的grid，跳过TIFF文件下载")
            return
            
        # Group grids by server for TIFF downloads
        server_tiff_grids = {}
        for (year, lon, lat), hostname in downloaded_grids.items():
            if hostname not in server_tiff_grids:
                server_tiff_grids[hostname] = set()
            server_tiff_grids[hostname].add((lon, lat))
        
        logger.info(f"需要从 {len(server_tiff_grids)} 个服务器下载TIFF文件")
        
        # Download TIFF files from each server
        for hostname, grids in server_tiff_grids.items():
            logger.info(f"从 {hostname} 下载 {len(grids)} 个TIFF文件")
            
            # Map grids to their years for this server
            grid_to_years = {}
            for (year, lon, lat), server in downloaded_grids.items():
                if server == hostname:
                    if (lon, lat) not in grid_to_years:
                        grid_to_years[(lon, lat)] = []
                    grid_to_years[(lon, lat)].append(year)
            
            # 首先检查哪些TIFF文件需要下载（跳过已完整存在的文件）
            files_to_download = []
            skipped_count = 0
            
            for (lon, lat), years in grid_to_years.items():
                tiff_filename = f"grid_{lon}_{lat}.tiff"
                remote_tiff_path = f"{self.remote_tiff_path}/{tiff_filename}"
                
                # 检查所有年份目录中是否都有完整的TIFF文件
                need_download = False
                for year in years:
                    dest_dir = os.path.join(local_base_path, str(year), f"grid_{lon}_{lat}")
                    dest_tiff = os.path.join(dest_dir, tiff_filename)
                    
                    if not self.is_local_file_complete(dest_tiff, remote_tiff_path, hostname):
                        need_download = True
                        break
                
                if need_download:
                    files_to_download.append(tiff_filename)
                else:
                    skipped_count += 1
            
            if skipped_count > 0:
                logger.info(f"跳过 {skipped_count} 个已存在的完整TIFF文件")
            
            if not files_to_download:
                logger.info(f"从 {hostname} 无需下载任何TIFF文件，全部已存在且完整")
                continue
            
            logger.info(f"需要从 {hostname} 下载 {len(files_to_download)} 个TIFF文件")
            
            # Create temporary directory for file lists
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create file list for TIFF files that need to be downloaded
                file_list_path = os.path.join(temp_dir, "tiff_files.txt")
                
                # Write TIFF file names to list
                with open(file_list_path, 'w') as f:
                    for tiff_filename in files_to_download:
                        f.write(f"{tiff_filename}\n")
                
                # Download all TIFF files to a temporary location first
                temp_tiff_dir = os.path.join(temp_dir, "tiffs")
                os.makedirs(temp_tiff_dir, exist_ok=True)
                
                # Construct rsync command with skip existing files option
                source = f"{self.username}@{hostname}:{self.remote_tiff_path}/"
                
                rsync_cmd = [
                    'rsync',
                    '-avz',  # archive, verbose, compress
                    '--no-group',  # don't preserve group ownership
                    '--ignore-existing',  # skip files that exist on receiver
                    '--files-from', file_list_path,
                    source,
                    temp_tiff_dir
                ]
                
                # Execute rsync
                try:
                    logger.info(f"正在从 {hostname} 运行rsync下载TIFF文件...")
                    
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
                                    with print_lock:
                                        tqdm.write(f"  ✓ 从 {hostname} 下载TIFF: {match.group()}")
                    
                    # Wait for process to complete
                    process.wait()
                    
                    if process.returncode != 0:
                        stderr = process.stderr.read()
                        if process.returncode == 23 and "some files/attrs were not transferred" in stderr:
                            logger.warning(f"从 {hostname} 下载TIFF文件时有权限警告（通常无害）")
                        else:
                            logger.error(f"从 {hostname} 下载TIFF文件失败: {stderr}")
                            continue
                    
                    if tiff_count > 3:
                        with print_lock:
                            tqdm.write(f"  ... 以及从 {hostname} 下载的其他 {tiff_count - 3} 个TIFF文件")
                        
                    # Now copy TIFF files to their respective grid folders with integrity check
                    logger.info(f"正在将从 {hostname} 下载的TIFF文件复制到grid文件夹...")
                    
                    copied_count = 0
                    for (lon, lat), years in grid_to_years.items():
                        tiff_filename = f"grid_{lon}_{lat}.tiff"
                        source_tiff = os.path.join(temp_tiff_dir, tiff_filename)
                        remote_tiff_path = f"{self.remote_tiff_path}/{tiff_filename}"
                        
                        if os.path.exists(source_tiff):
                            # 验证下载的TIFF文件完整性
                            if not self.is_local_file_complete(source_tiff, remote_tiff_path, hostname):
                                logger.warning(f"下载的TIFF文件不完整，跳过: {tiff_filename}")
                                continue
                                
                            # Copy to each year directory where this grid exists
                            for year in years:
                                dest_dir = os.path.join(local_base_path, str(year), f"grid_{lon}_{lat}")
                                if os.path.exists(dest_dir):
                                    dest_tiff = os.path.join(dest_dir, tiff_filename)
                                    
                                    # 如果目标文件已存在且完整，跳过复制
                                    if self.is_local_file_complete(dest_tiff, remote_tiff_path, hostname):
                                        continue
                                        
                                    # 删除不完整的目标文件（如果存在）
                                    if os.path.exists(dest_tiff):
                                        os.remove(dest_tiff)
                                    
                                    shutil.copy2(source_tiff, dest_tiff)
                                    copied_count += 1
                        else:
                            logger.warning(f"TIFF文件未找到: {tiff_filename}")
                    
                    logger.info(f"从 {hostname} 复制了 {copied_count} 个TIFF文件到grid文件夹")
                            
                except Exception as e:
                    logger.error(f"从 {hostname} 下载TIFF文件时出错: {e}")
    
        logger.info("TIFF文件下载完成！")
        
    def download_all_grids(self, coordinates, local_base_path, years=None, max_workers=5):
        """
        下载所有grid数据
        
        Args:
            coordinates: 坐标列表
            local_base_path: 本地基础路径
            years: 要下载的年份列表，默认为2019-2024
            max_workers: 最大并发数
        """
        if years is None:
            years = list(range(2019, 2025))  # 2019-2024
            
        # 创建本地基础目录
        os.makedirs(local_base_path, exist_ok=True)
        
        # 生成所有下载任务
        tasks = []
        for year in years:
            for lon, lat in coordinates:
                tasks.append((year, lon, lat))
                
        logger.info(f"总共需要下载 {len(tasks)} 个grid文件夹")
        
        # 进度条
        downloaded = 0
        failed = 0
        downloaded_grids = {}  # Track successfully downloaded grids: (year, lon, lat) -> hostname
        
        with tqdm(total=len(tasks), desc="下载进度") as pbar:
            for task in tasks:
                year, lon, lat = task
                
                try:
                    # **修改点 1**: 获取最佳服务器和它的修改时间
                    remote_path = f"/tank/zf281/global_0.1_degree_representation/{year}/grid_{lon}_{lat}"
                    best_server, latest_mtime = self.find_best_server_for_file(remote_path)
                    
                    # **修改点 2**: 将 best_server 传递给 download_grid_folder
                    if best_server and self.download_grid_folder(year, lon, lat, local_base_path, best_server):
                        downloaded += 1
                        downloaded_grids[(year, lon, lat)] = best_server
                        
                        # **修改点 3**: 格式化时间戳并更新日志信息
                        readable_time = datetime.fromtimestamp(latest_mtime).strftime('%Y-%m-%d %H:%M:%S')
                        
                        with print_lock:
                            tqdm.write(f"✓ 已下载: {year}/grid_{lon}_{lat} (从 {best_server}, 修改时间: {readable_time})")
                    else:
                        failed += 1
                        with print_lock:
                            tqdm.write(f"✗ 不存在: {year}/grid_{lon}_{lat}")
                            
                except Exception as e:
                    failed += 1
                    with print_lock:
                        tqdm.write(f"✗ 下载失败: {year}/grid_{lon}_{lat} - {e}")
                        
                pbar.update(1)
                
        logger.info(f"NPY文件下载完成！成功: {downloaded}, 失败/不存在: {failed}")
        
        # 下载TIFF文件
        if downloaded_grids:
            self.download_tiff_files(downloaded_grids, local_base_path)
        else:
            logger.warning("没有成功下载的grid，跳过TIFF文件下载")
        
        logger.info(f"下载完成！成功: {downloaded}, 失败/不存在: {failed}")


def main():
    parser = argparse.ArgumentParser(description='从远程服务器下载grid数据')
    parser.add_argument('txt_path', help='包含grid列表的txt文件路径')
    parser.add_argument('--output', '-o', default='/scratch/zf281/pangaea-bench/data/treesatai/raw_tessera_tif',
                        help='本地输出路径')
    parser.add_argument('--hostname', default=None,
                        help='远程服务器地址（默认使用myrina和antiope两个服务器）')
    parser.add_argument('--username', default='zf281',
                        help='SSH用户名（默认: zf281）')
    parser.add_argument('--password', help='SSH密码（可选）')
    parser.add_argument('--key', help='SSH密钥文件路径（可选）')
    parser.add_argument('--years', nargs='+', type=int,
                        help='要下载的年份（默认: 2019-2024）')
    parser.add_argument('--workers', type=int, default=10,
                        help='并发下载数（默认: 10）')
    
    args = parser.parse_args()
    
    # 创建下载器实例
    hostnames = [args.hostname] if args.hostname else None
    downloader = GridDataDownloader(
        hostnames=hostnames,
        username=args.username,
        password=args.password,
        key_filename=args.key
    )
    
    try:
        # 连接到服务器
        logger.info("正在初始化连接...")
        
        # 解析坐标
        logger.info(f"正在解析txt文件: {args.txt_path}")
        coordinates = downloader.parse_grid_coordinates(args.txt_path)
        
        if not coordinates:
            logger.error("未找到有效的坐标")
            return
            
        # 开始下载
        logger.info(f"开始下载到: {args.output}")
        downloader.download_all_grids(
            coordinates=coordinates,
            local_base_path=args.output,
            years=args.years,
            max_workers=args.workers
        )
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)
        
    finally:
        # 断开连接
        downloader.disconnect()


if __name__ == "__main__":
    main()