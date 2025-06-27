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

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 线程锁，用于并发下载时的日志输出
print_lock = threading.Lock()

class GridDataDownloader:
    def __init__(self, hostname, username, password=None, key_filename=None):
        """
        初始化下载器
        
        Args:
            hostname: 远程服务器地址
            username: SSH用户名
            password: SSH密码（可选）
            key_filename: SSH密钥文件路径（可选）
        """
        self.hostname = hostname
        self.username = username
        self.password = password
        self.key_filename = key_filename
        self.ssh_client = None
        self.sftp_client = None
        
    def connect(self):
        """建立SSH连接"""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # 连接参数
            connect_params = {
                'hostname': self.hostname,
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
                
            self.ssh_client.connect(**connect_params)
            self.sftp_client = self.ssh_client.open_sftp()
            logger.info(f"成功连接到 {self.hostname}")
            
        except Exception as e:
            logger.error(f"连接失败: {e}")
            raise
            
    def disconnect(self):
        """断开SSH连接"""
        if self.sftp_client:
            self.sftp_client.close()
        if self.ssh_client:
            self.ssh_client.close()
        logger.info("已断开连接")
        
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
            
    def check_remote_directory_exists(self, remote_path):
        """检查远程目录是否存在"""
        try:
            self.sftp_client.stat(remote_path)
            return True
        except:
            return False
            
    def download_file(self, remote_path, local_path):
        """下载单个文件"""
        try:
            # 确保本地目录存在
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # 下载文件
            self.sftp_client.get(remote_path, local_path)
            return True
            
        except Exception as e:
            logger.error(f"下载文件失败 {remote_path}: {e}")
            return False
            
    def download_grid_folder(self, year, lon, lat, local_base_path):
        """
        下载指定年份和坐标的grid文件夹
        
        Args:
            year: 年份
            lon: 经度
            lat: 纬度
            local_base_path: 本地基础路径
            
        Returns:
            bool: 是否成功下载
        """
        grid_name = f"grid_{lon}_{lat}"
        remote_grid_path = f"/tank/zf281/global_0.1_degree_representation/{year}/{grid_name}"
        
        # 检查远程文件夹是否存在
        if not self.check_remote_directory_exists(remote_grid_path):
            return False
            
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
            
            if self.check_remote_directory_exists(remote_file_path):
                if not self.download_file(remote_file_path, local_file_path):
                    success = False
                    
        return success
        
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
        
        with tqdm(total=len(tasks), desc="下载进度") as pbar:
            for task in tasks:
                year, lon, lat = task
                
                try:
                    if self.download_grid_folder(year, lon, lat, local_base_path):
                        downloaded += 1
                        with print_lock:
                            tqdm.write(f"✓ 已下载: {year}/grid_{lon}_{lat}")
                    else:
                        failed += 1
                        with print_lock:
                            tqdm.write(f"✗ 不存在: {year}/grid_{lon}_{lat}")
                            
                except Exception as e:
                    failed += 1
                    with print_lock:
                        tqdm.write(f"✗ 下载失败: {year}/grid_{lon}_{lat} - {e}")
                        
                pbar.update(1)
                
        logger.info(f"下载完成！成功: {downloaded}, 失败/不存在: {failed}")


def main():
    parser = argparse.ArgumentParser(description='从远程服务器下载grid数据')
    parser.add_argument('txt_path', help='包含grid列表的txt文件路径')
    parser.add_argument('--output', '-o', default='/scratch/zf281/btfm_representation/new_zealand',
                        help='本地输出路径（默认: /scratch/zf281/btfm_representation/new_zealand）')
    parser.add_argument('--hostname', default='antiope.cl.cam.ac.uk',
                        help='远程服务器地址（默认: antiope.cl.cam.ac.uk）')
    parser.add_argument('--username', default='zf281',
                        help='SSH用户名（默认: zf281）')
    parser.add_argument('--password', help='SSH密码（可选）')
    parser.add_argument('--key', help='SSH密钥文件路径（可选）')
    parser.add_argument('--years', nargs='+', type=int,
                        help='要下载的年份（默认: 2019-2024）')
    parser.add_argument('--workers', type=int, default=3,
                        help='并发下载数（默认: 1）')
    
    args = parser.parse_args()
    
    # 创建下载器实例
    downloader = GridDataDownloader(
        hostname=args.hostname,
        username=args.username,
        password=args.password,
        key_filename=args.key
    )
    
    try:
        # 连接到服务器
        logger.info("正在连接到远程服务器...")
        downloader.connect()
        
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