#!/usr/bin/env python3
"""
检查远程服务器上的grid处理状态
"""

import paramiko
import os
from pathlib import Path

def extract_grid_name(filename):
    """从tiff文件名中提取grid名称"""
    # 移除.tiff扩展名，得到grid名称
    if filename.endswith('.tiff'):
        return filename[:-5]  # 去掉'.tiff'
    return filename

def check_remote_grid_processing(local_txt_path, remote_host, remote_path, username, password=None, key_filename=None):
    """
    检查远程服务器上的grid处理状态
    
    参数:
    - local_txt_path: 本地txt文件路径
    - remote_host: 远程服务器地址
    - remote_path: 远程服务器上的基础路径
    - username: SSH用户名
    - password: SSH密码（可选）
    - key_filename: SSH密钥文件路径（可选）
    """
    
    # 读取本地txt文件获取所有grid
    print(f"读取本地文件: {local_txt_path}")
    grid_names = []
    
    try:
        with open(local_txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and line.endswith('.tiff'):
                    grid_name = extract_grid_name(line)
                    grid_names.append(grid_name)
        
        print(f"共找到 {len(grid_names)} 个grid需要检查\n")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {local_txt_path}")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 建立SSH连接
    print(f"连接到远程服务器: {remote_host}")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # 尝试使用密钥或密码连接
        if key_filename:
            ssh.connect(remote_host, username=username, key_filename=key_filename)
        else:
            ssh.connect(remote_host, username=username, password=password)
        
        print("SSH连接成功\n")
        
        # 检查每个grid的处理状态
        unprocessed_grids = []
        processed_count = 0
        
        print("开始检查grid处理状态...")
        
        for i, grid_name in enumerate(grid_names, 1):
            # 构建远程文件夹路径
            remote_folder = os.path.join(remote_path, grid_name)
            
            # 检查文件夹是否存在以及npy文件数量
            check_cmd = f"if [ -d '{remote_folder}' ]; then ls '{remote_folder}'/*.npy 2>/dev/null | wc -l; else echo 0; fi"
            
            stdin, stdout, stderr = ssh.exec_command(check_cmd)
            npy_count = stdout.read().decode().strip()
            
            try:
                npy_count = int(npy_count)
            except ValueError:
                npy_count = 0
            
            # 判断是否处理成功（需要2个npy文件）
            if npy_count >= 2:
                processed_count += 1
            else:
                unprocessed_grids.append(grid_name)
            
            # 每处理100个打印一次进度
            if i % 100 == 0:
                print(f"  已检查: {i}/{len(grid_names)}")
        
        print(f"\n检查完成!")
        print(f"成功处理的grid数量: {processed_count}")
        print(f"未成功处理的grid数量: {len(unprocessed_grids)}")
        
        # 打印未处理好的grid
        if unprocessed_grids:
            print(f"\n以下 {len(unprocessed_grids)} 个grid未处理好:")
            print("=" * 50)
            for grid in unprocessed_grids:
                print(grid)
        else:
            print("\n所有grid都已成功处理!")
        
    except paramiko.AuthenticationException:
        print("SSH认证失败，请检查用户名和密码/密钥")
    except paramiko.SSHException as e:
        print(f"SSH连接错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        ssh.close()
        print("\nSSH连接已关闭")

def main():
    # 配置参数
    LOCAL_TXT_PATH = "/maps/zf281/btfm4rs/grid_analysis_output/tiffs_to_process.txt"
    REMOTE_HOST = "antiope.cl.cam.ac.uk"
    REMOTE_PATH = "/tank/zf281/global_0.1_degree_representation/2024"
    USERNAME = "zf281"
    
    # 选择认证方式
    # 方式1: 使用密码（不推荐在代码中硬编码密码）
    # PASSWORD = "your_password"  # 请替换为你的密码
    # check_remote_grid_processing(LOCAL_TXT_PATH, REMOTE_HOST, REMOTE_PATH, USERNAME, password=PASSWORD)
    
    # 方式2: 使用SSH密钥（推荐）
    KEY_FILE = os.path.expanduser("~/.ssh/id_rsa")  # 请确保密钥路径正确
    
    # 如果需要输入密码，可以使用getpass
    import getpass
    password = None
    
    # 先尝试使用密钥
    if os.path.exists(KEY_FILE):
        print("使用SSH密钥进行认证")
        check_remote_grid_processing(LOCAL_TXT_PATH, REMOTE_HOST, REMOTE_PATH, USERNAME, key_filename=KEY_FILE)
    else:
        # 如果没有密钥，使用密码
        print("未找到SSH密钥，将使用密码认证")
        password = getpass.getpass(f"请输入 {USERNAME}@{REMOTE_HOST} 的密码: ")
        check_remote_grid_processing(LOCAL_TXT_PATH, REMOTE_HOST, REMOTE_PATH, USERNAME, password=password)

if __name__ == "__main__":
    main()