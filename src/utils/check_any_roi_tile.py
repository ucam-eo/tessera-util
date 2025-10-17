import paramiko
import datetime
import os
import logging
import threading
import concurrent.futures
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import time

# --- 配置部分 (与您提供的一致) ---
SERVERS = ['zf281@myrina.cl.cam.ac.uk', 'zf281@antiope.cl.cam.ac.uk']
DPIXEL_SERVER = 'zf281@otrera.cl.cam.ac.uk'
GRID_FILE_PATH = '/maps/zf281/btfm4rs/external_request_#68_1_tiff_files.txt'

# YEARS = range(2017, 2025)  # 2017 到 2024
# YEARS = range(2019, 2025)
# YEARS = range(2020, 2025)
# YEARS = [2017, 2020, 2023, 2024]
# YEARS = [2023, 2022]
# YEARS = [2020]
YEARS = [2024]

BASE_DIR_TEMPLATE = '/tank/zf281/global_0.1_degree_representation/{year}/{grid_id}'
DPIXEL_DIR_TEMPLATE = '/tank/zf281/global_0.1_degree_tiff_d_pixel/{year}/{grid_id}'

CUTOFF_DATETIME = datetime.datetime(2025, 8, 20, 23, 59, 0)
# CUTOFF_DATETIME = datetime.datetime(2025, 5, 20, 23, 59, 0)

# 性能配置
MAX_WORKERS = 4  # 并行线程数
BATCH_SIZE = 50  # 批次大小 - 减小以避免SSH命令过长

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def batch_check_remote_dirs(ssh_client, grid_year_pairs: List[Tuple[str, int]], server_name: str) -> Dict[Tuple[str, int], Optional[float]]:
    """
    批量检查远程目录是否存在并返回修改时间
    使用脚本文件方式避免命令行过长问题
    """
    if not grid_year_pairs:
        return {}

    # 创建远程脚本内容
    script_lines = ['#!/bin/bash']
    for grid_id, year in grid_year_pairs:
        remote_path = BASE_DIR_TEMPLATE.format(year=year, grid_id=grid_id)
        script_lines.append(f'if [ -d "{remote_path}" ]; then')
        script_lines.append(f'  echo "{grid_id}|{year}|$(stat -c %Y "{remote_path}")"')
        script_lines.append(f'else')
        script_lines.append(f'  echo "{grid_id}|{year}|NOTFOUND"')
        script_lines.append(f'fi')

    script_content = '\n'.join(script_lines)

    try:
        start_time = time.time()

        # 创建临时脚本文件
        script_name = f'/tmp/check_batch_{int(time.time())}.sh'
        stdin, stdout, stderr = ssh_client.exec_command(f'cat > {script_name}', timeout=60)
        stdin.write(script_content)
        stdin.close()
        stdout.read()  # 等待完成

        # 执行脚本
        stdin, stdout, stderr = ssh_client.exec_command(f'chmod +x {script_name} && {script_name} && rm {script_name}', timeout=300)
        exit_status = stdout.channel.recv_exit_status()
        output = stdout.read().decode('utf-8').strip()

        elapsed = time.time() - start_time
        logging.info(f"批量检查 {len(grid_year_pairs)} 个目录在 {server_name} 用时: {elapsed:.2f}秒")

        # 解析结果
        results = {}
        if output:
            lines = output.split('\n')
            for line in lines:
                line = line.strip()
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) == 3:
                        grid_id, year_str, timestamp_str = parts
                        key = (grid_id, int(year_str))
                        if timestamp_str != 'NOTFOUND':
                            try:
                                results[key] = float(timestamp_str)
                            except ValueError:
                                results[key] = None
                        else:
                            results[key] = None

        return results

    except Exception as e:
        logging.error(f"批量检查远程目录时出错 (服务器: {server_name}): {e}")
        return {key: None for key in grid_year_pairs}


def batch_check_dpixel_files(ssh_client, grid_year_pairs: List[Tuple[str, int]]) -> Dict[Tuple[str, int], Tuple[bool, bool]]:
    """
    批量检查dpixel服务器上的文件
    返回字典: {(grid_id, year): (exists, is_after_cutoff)}
    """
    if not grid_year_pairs:
        return {}

    required_files = [
        'band_mean.npy', 'band_std.npy', 'bands.npy', 'doys.npy', 'masks.npy',
        'sar_ascending.npy', 'sar_ascending_doy.npy', 'sar_descending.npy', 'sar_descending_doy.npy'
    ]

    # 创建远程脚本内容
    script_lines = ['#!/bin/bash']

    for grid_id, year in grid_year_pairs:
        remote_path = DPIXEL_DIR_TEMPLATE.format(year=year, grid_id=grid_id)

        # 检查目录是否存在并获取修改时间
        script_lines.append(f'if [ -d "{remote_path}" ]; then')
        script_lines.append(f'  echo "{grid_id}|{year}|DIR|$(stat -c %Y "{remote_path}")"')
        script_lines.append(f'else')
        script_lines.append(f'  echo "{grid_id}|{year}|DIR|NOTFOUND"')
        script_lines.append(f'fi')

        # 检查所有必需文件
        for filename in required_files:
            file_path = f"{remote_path}/{filename}"
            script_lines.append(f'if [ -f "{file_path}" ]; then')
            script_lines.append(f'  echo "{grid_id}|{year}|FILE|{filename}|EXISTS"')
            script_lines.append(f'else')
            script_lines.append(f'  echo "{grid_id}|{year}|FILE|{filename}|MISSING"')
            script_lines.append(f'fi')

    script_content = '\n'.join(script_lines)

    try:
        start_time = time.time()

        # 创建临时脚本文件
        script_name = f'/tmp/check_dpixel_{int(time.time())}.sh'
        stdin, stdout, stderr = ssh_client.exec_command(f'cat > {script_name}', timeout=60)
        stdin.write(script_content)
        stdin.close()
        stdout.read()  # 等待完成

        # 执行脚本
        stdin, stdout, stderr = ssh_client.exec_command(f'chmod +x {script_name} && {script_name} && rm {script_name}', timeout=300)
        exit_status = stdout.channel.recv_exit_status()
        output = stdout.read().decode('utf-8').strip()

        elapsed = time.time() - start_time
        logging.info(f"批量检查 {len(grid_year_pairs)} 个dpixel目录用时: {elapsed:.2f}秒")

        # 解析结果
        dir_timestamps = {}
        file_status = defaultdict(set)  # {(grid_id, year): {existing_files}}

        if output:
            lines = output.split('\n')
            for line in lines:
                line = line.strip()
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        grid_id, year_str, check_type = parts[0], parts[1], parts[2]
                        key = (grid_id, int(year_str))

                        if check_type == 'DIR' and len(parts) == 4:
                            timestamp_str = parts[3]
                            if timestamp_str != 'NOTFOUND':
                                try:
                                    dir_timestamps[key] = float(timestamp_str)
                                except ValueError:
                                    pass
                        elif check_type == 'FILE' and len(parts) == 5:
                            filename, status = parts[3], parts[4]
                            if status == 'EXISTS':
                                file_status[key].add(filename)

        # 生成最终结果
        results = {}
        required_files_set = set(required_files)

        for grid_id, year in grid_year_pairs:
            key = (grid_id, year)

            # 检查目录是否存在
            if key not in dir_timestamps:
                results[key] = (False, False)
                continue

            # 检查所有文件是否存在
            existing_files = file_status.get(key, set())
            if not required_files_set.issubset(existing_files):
                results[key] = (False, False)
                continue

            # 检查时间条件
            mod_datetime = datetime.datetime.fromtimestamp(dir_timestamps[key])
            is_after_cutoff = mod_datetime >= CUTOFF_DATETIME
            results[key] = (True, is_after_cutoff)

        return results

    except Exception as e:
        logging.error(f"批量检查dpixel文件时出错: {e}")
        return {key: (False, False) for key in grid_year_pairs}

def check_server_batch(server_address: str, grid_year_pairs: List[Tuple[str, int]]) -> Dict[Tuple[str, int], Dict]:
    """
    并行检查单个服务器上的所有grid
    """
    hostname = server_address.split('@')[1]
    username = server_address.split('@')[0]

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        logging.info(f"正在连接到 {hostname}...")
        ssh.connect(hostname, username=username, timeout=30)

        # 分批处理以避免命令过长
        all_results = {}
        for i in range(0, len(grid_year_pairs), BATCH_SIZE):
            batch = grid_year_pairs[i:i + BATCH_SIZE]
            batch_results = batch_check_remote_dirs(ssh, batch, hostname)

            # 转换结果格式
            for (grid_id, year), timestamp in batch_results.items():
                if timestamp is not None:
                    mod_datetime = datetime.datetime.fromtimestamp(timestamp)
                    all_results[(grid_id, year)] = {
                        'server': hostname,
                        'mod_time': mod_datetime
                    }

        logging.info(f"服务器 {hostname} 检查完成，找到 {len(all_results)} 个有效目录")
        return all_results

    except Exception as e:
        logging.error(f"检查服务器 {server_address} 时出错: {e}")
        return {}
    finally:
        ssh.close()


def check_dpixel_batch(grid_year_pairs: List[Tuple[str, int]]) -> Dict[Tuple[str, int], Tuple[bool, bool]]:
    """
    批量检查dpixel服务器
    """
    if not grid_year_pairs:
        return {}

    hostname = DPIXEL_SERVER.split('@')[1]
    username = DPIXEL_SERVER.split('@')[0]

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        logging.info(f"正在连接到dpixel服务器 {hostname}...")
        ssh.connect(hostname, username=username, timeout=30)

        # 分批处理
        all_results = {}
        for i in range(0, len(grid_year_pairs), BATCH_SIZE):
            batch = grid_year_pairs[i:i + BATCH_SIZE]
            batch_results = batch_check_dpixel_files(ssh, batch)
            all_results.update(batch_results)

        logging.info(f"dpixel服务器检查完成，检查了 {len(grid_year_pairs)} 个目录")
        return all_results

    except Exception as e:
        logging.error(f"检查dpixel服务器时出错: {e}")
        return {key: (False, False) for key in grid_year_pairs}
    finally:
        ssh.close()


def main():
    """
    优化后的主执行函数 - 使用并行处理和批量检查
    """
    start_time = time.time()

    try:
        with open(GRID_FILE_PATH, 'r') as f:
            grid_ids = [line.strip().replace('.tiff', '') for line in f if line.strip()]
        logging.info(f"成功从 {GRID_FILE_PATH} 读取 {len(grid_ids)} 个 grid_id.")
    except FileNotFoundError:
        logging.error(f"错误：找不到 grid 文件路径 '{GRID_FILE_PATH}'。请检查路径是否正确。")
        return

    # 创建所有 (grid_id, year) 组合
    all_grid_year_pairs = [(grid_id, year) for grid_id in grid_ids for year in YEARS]
    logging.info(f"总共需要检查 {len(all_grid_year_pairs)} 个 grid-year 组合")

    # --- 阶段 1: 并行检查所有服务器 ---
    logging.info("="*50)
    logging.info("开始并行检查representation服务器...")
    logging.info("="*50)

    results = defaultdict(list)

    # 使用线程池并行检查所有服务器
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SERVERS)) as executor:
        future_to_server = {
            executor.submit(check_server_batch, server, all_grid_year_pairs): server
            for server in SERVERS
        }

        for future in concurrent.futures.as_completed(future_to_server):
            server = future_to_server[future]
            try:
                server_results = future.result()
                for key, result_info in server_results.items():
                    results[key].append(result_info)
                logging.info(f"服务器 {server} 完成检查")
            except Exception as e:
                logging.error(f"服务器 {server} 检查失败: {e}")

    logging.info(f"所有representation服务器检查完毕，找到 {len(results)} 个有效目录")

    # --- 阶段 2: 处理结果并确定需要dpixel检查的项目 ---
    logging.info("="*50)
    logging.info("处理representation检查结果...")
    logging.info("="*50)

    need_dpixel_check = []
    need_inference = []

    all_possible_keys = set(all_grid_year_pairs)
    found_keys = set(results.keys())
    not_found_keys = all_possible_keys - found_keys

    # 处理找到的目录
    for (grid_id, year), found_locations in results.items():
        if len(found_locations) > 1:
            logging.debug(f"目录 {grid_id}_{year} 在 {len(found_locations)} 个服务器上都存在")

        # 选择最新的修改时间
        latest_location = max(found_locations, key=lambda x: x['mod_time'])
        latest_mod_time = latest_location['mod_time']

        if latest_mod_time < CUTOFF_DATETIME:
            need_dpixel_check.append((grid_id, year))
            need_inference.append((grid_id, year))
            logging.debug(f"{grid_id}_{year}: representation过旧，需要检查dpixel和inference")
        else:
            logging.debug(f"{grid_id}_{year}: representation满足条件，跳过")

    # 处理未找到的目录
    for grid_id, year in not_found_keys:
        need_dpixel_check.append((grid_id, year))
        need_inference.append((grid_id, year))
        logging.debug(f"{grid_id}_{year}: 未找到representation，需要检查dpixel和inference")

    logging.info(f"需要检查dpixel的项目: {len(need_dpixel_check)} 个")

    # --- 阶段 3: 批量检查dpixel服务器 ---
    dpixel_results = {}
    if need_dpixel_check:
        logging.info("="*50)
        logging.info("开始批量检查dpixel服务器...")
        logging.info("="*50)

        dpixel_results = check_dpixel_batch(need_dpixel_check)

    # --- 阶段 4: 生成任务列表 ---
    logging.info("="*50)
    logging.info("生成最终任务列表...")
    logging.info("="*50)

    dpixel_tasks = []
    inference_tasks = []

    for grid_id, year in need_dpixel_check:
        key = (grid_id, year)
        dpixel_exists, dpixel_after_cutoff = dpixel_results.get(key, (False, False))

        if not dpixel_exists or not dpixel_after_cutoff:
            dpixel_tasks.append(f"touch {grid_id}_{year}.task")

    for grid_id, year in need_inference:
        inference_tasks.append(f"touch {grid_id}_{year}.task")

    # --- 阶段 5: 输出结果 ---
    total_time = time.time() - start_time

    print("\n" + "="*60)
    print(f"处理完成! 总用时: {total_time:.2f}秒")
    print(f"检查了 {len(grid_ids)} 个grid, {len(YEARS)} 个年份")
    print("="*60)

    print(f"\n需要创建的 dpixel 任务: {len(dpixel_tasks)} 个")
    if dpixel_tasks and len(dpixel_tasks) <= 20:  # 只在任务数量较少时显示
        for task in sorted(dpixel_tasks)[:10]:
            print(f"  {task}")
        if len(dpixel_tasks) > 10:
            print(f"  ... 还有 {len(dpixel_tasks) - 10} 个任务")

    print(f"\n需要创建的 inference 任务: {len(inference_tasks)} 个")
    if inference_tasks and len(inference_tasks) <= 20:
        for task in sorted(inference_tasks)[:10]:
            print(f"  {task}")
        if len(inference_tasks) > 10:
            print(f"  ... 还有 {len(inference_tasks) - 10} 个任务")

    # 写入文件
    dpixel_file_path = '/maps/zf281/btfm4rs/task_dpixel.txt'
    inference_file_path = '/maps/zf281/btfm4rs/task_inference.txt'

    try:
        with open(dpixel_file_path, 'w') as f:
            for task in sorted(dpixel_tasks):
                f.write(task + '\n')
        logging.info(f"dpixel 任务列表已写入: {dpixel_file_path}")
    except Exception as e:
        logging.error(f"写入 dpixel 任务文件时出错: {e}")

    try:
        with open(inference_file_path, 'w') as f:
            for task in sorted(inference_tasks):
                f.write(task + '\n')
        logging.info(f"inference 任务列表已写入: {inference_file_path}")
    except Exception as e:
        logging.error(f"写入 inference 任务文件时出错: {e}")

    print(f"\n任务文件已保存:")
    print(f"  {dpixel_file_path}")
    print(f"  {inference_file_path}")
    print("="*60)


if __name__ == '__main__':
    main()