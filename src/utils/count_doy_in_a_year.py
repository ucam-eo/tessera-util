#!/usr/bin/env python3
"""
doy_statistics.py - 统计TIFF文件对应位置的哨兵1/2数据DOY数量
创建日期：2025-05-24
功能：并行化处理多个TIFF文件，统计2022年哨兵1/2数据的唯一DOY数量，支持断点续传
优化：添加请求限流、改进错误处理、减少API压力
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import concurrent.futures
import threading
import random
import queue
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
import pystac_client
import planetary_computer

# 设置日志
def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('doy_statistics.log', 'a', encoding='utf-8')
        ]
    )

# 全局常量
TIFF_DIR = "/scratch/zf281/global_map_1_degree_tiff"
OUTPUT_EXCEL = "/maps/zf281/btfm4rs/src/utils/doy_statistics_2022_1_degree_grid.xlsx"
YEAR = 2022
START_DATE = f"{YEAR}-01-01"
END_DATE = f"{YEAR}-12-31"

# 重试配置 - 更保守的设置
MAX_RETRIES = 8
RETRY_BACKOFF_FACTOR = 2.0
BASE_RETRY_DELAY = 3
MAX_RETRY_DELAY = 120

# 并行配置 - 更保守的设置以避免API超时
DEFAULT_MAX_WORKERS = 2  # 减少默认并发数
DEFAULT_BATCH_SIZE = 4   # 减少批处理大小

# 请求限流配置
REQUEST_DELAY = 1.0  # 每个请求间隔1秒
API_QUOTA_DELAY = 0.5  # API配额延迟

# 全局锁和限流器
excel_lock = threading.Lock()
api_limiter = threading.Semaphore(2)  # 最多同时2个API请求
request_times = queue.Queue()  # 记录请求时间用于限流

def rate_limit_request():
    """请求限流器 - 确保不会过快请求API"""
    with api_limiter:
        current_time = time.time()
        
        # 清理旧的请求时间记录（超过60秒的）
        while not request_times.empty():
            try:
                old_time = request_times.get_nowait()
                if current_time - old_time > 60:
                    continue
                else:
                    request_times.put(old_time)
                    break
            except queue.Empty:
                break
        
        # 记录当前请求时间
        request_times.put(current_time)
        
        # 添加请求间隔
        time.sleep(REQUEST_DELAY)

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="统计TIFF文件对应位置的哨兵1/2数据DOY数量")
    parser.add_argument("--tiff_dir", default=TIFF_DIR, help="TIFF文件目录路径")
    parser.add_argument("--output", default=OUTPUT_EXCEL, help="输出Excel文件名")
    parser.add_argument("--year", type=int, default=YEAR, help="统计年份")
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS, help="最大并行数")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="批处理大小")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--request_delay", type=float, default=REQUEST_DELAY, help="请求间隔(秒)")
    return parser.parse_args()

def validate_bbox(bbox_ll, tile_id):
    """
    验证边界框是否有效，检测跨越180度经线的情况
    bbox_ll格式: [min_lon, min_lat, max_lon, max_lat]
    """
    min_lon, min_lat, max_lon, max_lat = bbox_ll
    
    # 检查基本的经纬度范围
    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
        logging.warning(f"{tile_id}: 经度超出有效范围 [-180, 180]: min_lon={min_lon}, max_lon={max_lon}")
        return False
    
    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        logging.warning(f"{tile_id}: 纬度超出有效范围 [-90, 90]: min_lat={min_lat}, max_lat={max_lat}")
        return False
    
    # 检查是否跨越180度经线（国际日期变更线）
    if min_lon > max_lon:
        logging.warning(f"{tile_id}: ⚠️ 边界框跨越180度经线（国际日期变更线），STAC API无法处理")
        logging.warning(f"{tile_id}: bbox=[{min_lon:.3f}, {min_lat:.3f}, {max_lon:.3f}, {max_lat:.3f}]")
        logging.warning(f"{tile_id}: 这通常发生在接近太平洋日期变更线的区域，将跳过此tile")
        return False
    
    # 检查纬度是否正确
    if min_lat > max_lat:
        logging.warning(f"{tile_id}: 最小纬度大于最大纬度: min_lat={min_lat}, max_lat={max_lat}")
        return False
    
    return True
def load_tiff_bounds(tiff_path):
    """
    加载TIFF文件的边界框信息
    返回：(bbox_proj, bbox_ll, crs)
    """
    try:
        with rasterio.open(tiff_path) as src:
            bbox_proj = src.bounds
            crs = src.crs
            # 转换为WGS84经纬度坐标
            bbox_ll = transform_bounds(crs, "EPSG:4326", *bbox_proj, densify_pts=21)
            return bbox_proj, bbox_ll, crs
    except Exception as e:
        logging.error(f"读取TIFF文件 {tiff_path} 失败: {e}")
        return None, None, None

def search_sentinel_items_with_fallback(bbox_ll, date_range, collection, query_params=None, max_retries=MAX_RETRIES, tile_id="unknown"):
    """
    搜索哨兵数据，带重试机制和fallback策略
    """
    retry_count = 0
    retry_delay = BASE_RETRY_DELAY
    
    while retry_count <= max_retries:
        try:
            # 应用请求限流
            rate_limit_request()
            
            # 创建STAC客户端
            cat = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=planetary_computer.sign_inplace
            )
            
            # 构建查询参数
            search_params = {
                "collections": [collection],
                "bbox": bbox_ll,
                "datetime": date_range,
                "limit": 999  # 设置为999避免超过1000限制
            }
            
            # 添加额外查询参数
            if query_params:
                search_params.update(query_params)
            
            logging.debug(f"{tile_id}: 搜索参数 {collection} - bbox: {bbox_ll}, datetime: {date_range}")
            if query_params and 'query' in query_params:
                logging.debug(f"{tile_id}: 查询过滤器: {query_params['query']}")
            
            # 执行搜索
            q = cat.search(**search_params)
            
            # 使用items()而不是get_items()来避免FutureWarning
            try:
                items = list(q.items())
            except AttributeError:
                # 如果items()方法不存在，使用get_items()
                items = list(q.get_items())
            
            logging.info(f"{tile_id}: 原始搜索到 {len(items)} 个items ({collection})")
            
            return items
            
        except Exception as e:
            error_msg = str(e).lower()
            retry_count += 1
            
            # 检查是否是超时错误
            is_timeout_error = any(keyword in error_msg for keyword in [
                'timeout', 'exceeded', 'maximum allowed time', 'timed out'
            ])
            
            # 检查是否是网络错误
            is_network_error = any(keyword in error_msg for keyword in [
                'connection', 'network', 'unreachable', 'refused'
            ])
            
            if retry_count > max_retries:
                logging.error(f"搜索 {collection} 数据失败，已重试 {max_retries} 次: {e}")
                return []
            
            # 根据错误类型调整重试策略
            if is_timeout_error:
                # 超时错误 - 使用更长的延迟
                retry_delay = min(MAX_RETRY_DELAY, retry_delay * RETRY_BACKOFF_FACTOR * 1.5)
                logging.warning(f"搜索 {collection} 超时 (尝试 {retry_count}/{max_retries+1}): {e}")
            elif is_network_error:
                # 网络错误 - 中等延迟
                retry_delay = min(MAX_RETRY_DELAY, retry_delay * RETRY_BACKOFF_FACTOR)
                logging.warning(f"搜索 {collection} 网络错误 (尝试 {retry_count}/{max_retries+1}): {e}")
            else:
                # 其他错误 - 标准延迟
                retry_delay = min(MAX_RETRY_DELAY, retry_delay * RETRY_BACKOFF_FACTOR)
                logging.warning(f"搜索 {collection} 失败 (尝试 {retry_count}/{max_retries+1}): {e}")
            
            # 添加随机抖动
            jitter = random.uniform(0.8, 1.2)
            actual_delay = retry_delay * jitter
            
            logging.info(f"{actual_delay:.1f}秒后重试...")
            time.sleep(actual_delay)
    
    return []

def extract_unique_doys(items, tile_id="unknown", collection="unknown", target_year=None):
    """
    从items中提取唯一的DOY（Day of Year），添加详细日志和年份验证
    """
    doys = set()
    valid_items = 0
    invalid_date_count = 0
    wrong_year_count = 0
    date_examples = []
    
    logging.info(f"{tile_id}: 开始处理 {len(items)} 个 {collection} items，目标年份: {target_year}")
    
    for i, item in enumerate(items):
        try:
            # 获取日期时间字符串
            datetime_str = item.properties.get("datetime", "")
            if not datetime_str:
                invalid_date_count += 1
                continue
                
            # 解析日期
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            
            # 检查年份是否匹配目标年份
            if target_year and dt.year != target_year:
                wrong_year_count += 1
                if wrong_year_count <= 3:  # 只记录前3个错误年份的例子
                    logging.warning(f"{tile_id}: 发现非目标年份数据 - 期望{target_year}，实际{dt.year} ({datetime_str})")
                continue
            
            # 计算DOY
            doy = dt.timetuple().tm_yday
            doys.add(doy)
            valid_items += 1
            
            # 收集前5个日期作为例子
            if len(date_examples) < 5:
                date_examples.append(f"{dt.strftime('%Y-%m-%d')}(DOY:{doy})")
                
        except Exception as e:
            invalid_date_count += 1
            if invalid_date_count <= 3:  # 只记录前3个解析错误
                logging.warning(f"{tile_id}: 解析日期失败: {datetime_str}, 错误: {e}")
            continue
    
    unique_doy_count = len(doys)
    
    # 详细日志
    logging.info(f"{tile_id}: {collection} 处理结果:")
    logging.info(f"{tile_id}:   - 总items: {len(items)}")
    logging.info(f"{tile_id}:   - 有效items: {valid_items}")
    logging.info(f"{tile_id}:   - 日期解析失败: {invalid_date_count}")
    logging.info(f"{tile_id}:   - 年份不匹配: {wrong_year_count}")
    logging.info(f"{tile_id}:   - 唯一DOY数量: {unique_doy_count}")
    logging.info(f"{tile_id}:   - 日期示例: {', '.join(date_examples)}")
    
    # 如果DOY数量异常（>366），输出更详细信息
    if unique_doy_count > 366:
        logging.error(f"{tile_id}: ⚠️ 异常！DOY数量 {unique_doy_count} 超过一年最大天数！")
        logging.error(f"{tile_id}: DOY列表（前20个）: {sorted(list(doys))[:20]}")
        
        # 按年份分组显示统计
        year_stats = defaultdict(int)
        for item in items[:50]:  # 只检查前50个避免太多日志
            try:
                datetime_str = item.properties.get("datetime", "")
                if datetime_str:
                    dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                    year_stats[dt.year] += 1
            except:
                continue
        logging.error(f"{tile_id}: 年份分布统计: {dict(year_stats)}")
    
    return unique_doy_count

def process_single_tiff(tiff_path, year, request_delay=REQUEST_DELAY):
    """
    处理单个TIFF文件，返回统计结果
    """
    tiff_path = Path(tiff_path)
    tile_id = tiff_path.stem
    
    logging.info(f"开始处理 TIFF: {tile_id}")
    
    # 读取TIFF边界
    bbox_proj, bbox_ll, crs = load_tiff_bounds(tiff_path)
    if bbox_ll is None:
        logging.error(f"无法读取 {tile_id} 的边界信息")
        return tile_id, 0, 0, 0
    
    # 验证边界框是否有效（检查跨越180度经线等问题）
    if not validate_bbox(bbox_ll, tile_id):
        logging.warning(f"⚠️ {tile_id}: 边界框无效，跳过处理")
        return tile_id, -1, -1, -1  # 使用-1标记为跳过的tile
    
    # 确保日期范围严格限制在指定年份
    date_range = f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z"
    logging.info(f"{tile_id}: 使用严格日期范围: {date_range}")
    logging.info(f"{tile_id}: 有效边界框: [{bbox_ll[0]:.3f}, {bbox_ll[1]:.3f}, {bbox_ll[2]:.3f}, {bbox_ll[3]:.3f}]")
    
    # 初始化结果
    s2_doy_count = 0
    s1_asc_doy_count = 0
    s1_desc_doy_count = 0
    
    try:
        # 搜索Sentinel-2数据（按照官方文档的正确方式）
        logging.info(f"{tile_id}: 搜索Sentinel-2数据...")
        s2_query_params = {
            "query": {
                "eo:cloud_cover": {"lt": 90}  # 云量限制改为90%
            }
        }
        s2_items_raw = search_sentinel_items_with_fallback(
            bbox_ll, 
            date_range, 
            "sentinel-2-l2a",
            query_params=s2_query_params,
            tile_id=tile_id
        )
        
        # 过滤包含MSI波段的items（例如B04红色波段，所有S2 L2A都应该有）
        s2_items = [item for item in s2_items_raw if 'B04' in item.assets]
        logging.info(f"{tile_id}: S2过滤后保留 {len(s2_items)}/{len(s2_items_raw)} 个包含B04(MSI)波段的items")
        
        s2_doy_count = extract_unique_doys(s2_items, tile_id, "Sentinel-2", year)
        logging.info(f"{tile_id}: ✓ S2 最终DOY数量 = {s2_doy_count}")
        
        # 添加额外延迟
        time.sleep(request_delay)
        
        # 搜索Sentinel-1升轨数据
        logging.info(f"{tile_id}: 搜索Sentinel-1升轨数据...")
        s1_asc_items = search_sentinel_items_with_fallback(
            bbox_ll,
            date_range,
            "sentinel-1-rtc",
            query_params={"query": {"sat:orbit_state": {"eq": "ascending"}}},
            tile_id=tile_id
        )
        s1_asc_doy_count = extract_unique_doys(s1_asc_items, tile_id, "S1-ASC", year)
        logging.info(f"{tile_id}: ✓ S1升轨 最终DOY数量 = {s1_asc_doy_count}")
        
        # 添加额外延迟
        time.sleep(request_delay)
        
        # 搜索Sentinel-1降轨数据
        logging.info(f"{tile_id}: 搜索Sentinel-1降轨数据...")
        s1_desc_items = search_sentinel_items_with_fallback(
            bbox_ll,
            date_range,
            "sentinel-1-rtc",
            query_params={"query": {"sat:orbit_state": {"eq": "descending"}}},
            tile_id=tile_id
        )
        s1_desc_doy_count = extract_unique_doys(s1_desc_items, tile_id, "S1-DESC", year)
        logging.info(f"{tile_id}: ✓ S1降轨 最终DOY数量 = {s1_desc_doy_count}")
        
    except Exception as e:
        logging.error(f"处理 {tile_id} 时发生错误: {e}")
    
    logging.info(f"🎯 完成 {tile_id}: S2={s2_doy_count}, S1_ASC={s1_asc_doy_count}, S1_DESC={s1_desc_doy_count}")
    return tile_id, s2_doy_count, s1_asc_doy_count, s1_desc_doy_count

def load_existing_results(excel_path):
    """
    加载现有的Excel结果，用于断点续传
    """
    # 确保输出路径是绝对路径
    excel_path = Path(excel_path).resolve()
    
    if not excel_path.exists():
        # 创建新的DataFrame
        df = pd.DataFrame(columns=['Tile_id', 's2_doy', 's1_ascending_doy', 's1_descending_doy'])
        logging.info(f"创建新的结果文件: {excel_path}")
        return df, excel_path
    
    try:
        df = pd.read_excel(excel_path)
        logging.info(f"加载了现有结果文件: {excel_path}，包含 {len(df)} 条记录")
        return df, excel_path
    except Exception as e:
        logging.error(f"读取现有结果文件失败: {e}")
        # 返回空DataFrame
        return pd.DataFrame(columns=['Tile_id', 's2_doy', 's1_ascending_doy', 's1_descending_doy']), excel_path

def save_results_to_excel(df, excel_path):
    """
    保存结果到Excel文件
    """
    try:
        with excel_lock:
            # 确保目录存在
            excel_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_excel(excel_path, index=False)
        logging.info(f"结果已保存到 {excel_path} ({len(df)} 条记录)")
    except Exception as e:
        logging.error(f"保存Excel文件失败: {e}")

def is_record_complete(row):
    """
    检查记录是否完整（所有列都有有效值）
    -1表示跳过的tile（例如边界框跨越180度经线）
    """
    return (
        pd.notna(row.get('s2_doy', np.nan)) and 
        pd.notna(row.get('s1_ascending_doy', np.nan)) and 
        pd.notna(row.get('s1_descending_doy', np.nan)) and
        (
            # 正常情况：所有值都>=0
            (row.get('s2_doy', -2) >= 0 and
             row.get('s1_ascending_doy', -2) >= 0 and
             row.get('s1_descending_doy', -2) >= 0) or
            # 跳过情况：所有值都是-1
            (row.get('s2_doy', -2) == -1 and
             row.get('s1_ascending_doy', -2) == -1 and
             row.get('s1_descending_doy', -2) == -1)
        )
    )

def get_tiff_files(tiff_dir):
    """
    获取目录下所有TIFF文件
    """
    tiff_dir = Path(tiff_dir)
    if not tiff_dir.exists():
        logging.error(f"TIFF目录不存在: {tiff_dir}")
        return []
    
    tiff_files = list(tiff_dir.glob("*.tiff")) + list(tiff_dir.glob("*.tif"))
    logging.info(f"找到 {len(tiff_files)} 个TIFF文件")
    return tiff_files

def update_dataframe_with_result(df, tile_id, s2_doy, s1_asc_doy, s1_desc_doy):
    """
    更新DataFrame中的结果
    """
    # 查找是否已存在该tile_id的记录
    mask = df['Tile_id'] == tile_id
    if mask.any():
        # 更新现有记录
        df.loc[mask, 's2_doy'] = s2_doy
        df.loc[mask, 's1_ascending_doy'] = s1_asc_doy
        df.loc[mask, 's1_descending_doy'] = s1_desc_doy
    else:
        # 添加新记录
        new_row = {
            'Tile_id': tile_id,
            's2_doy': s2_doy,
            's1_ascending_doy': s1_asc_doy,
            's1_descending_doy': s1_desc_doy
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    return df

def main():
    """主函数"""
    args = get_args()
    setup_logging(args.debug)
    
    # 更新全局请求延迟
    global REQUEST_DELAY
    REQUEST_DELAY = args.request_delay
    
    logging.info("="*60)
    logging.info("开始DOY统计任务")
    logging.info(f"TIFF目录: {args.tiff_dir}")
    logging.info(f"输出文件: {args.output}")
    logging.info(f"统计年份: {args.year}")
    logging.info(f"最大并行数: {args.max_workers}")
    logging.info(f"批处理大小: {args.batch_size}")
    logging.info(f"请求间隔: {args.request_delay}秒")
    logging.info("="*60)
    
    # 获取所有TIFF文件
    tiff_files = get_tiff_files(args.tiff_dir)
    if not tiff_files:
        logging.error("未找到TIFF文件，退出")
        return
    
    # 加载现有结果
    df, excel_path = load_existing_results(args.output)
    
    # 筛选出需要处理的文件（跳过已完成的）
    pending_files = []
    for tiff_file in tiff_files:
        tile_id = tiff_file.stem
        # 检查是否已存在完整记录
        existing_mask = df['Tile_id'] == tile_id
        if existing_mask.any():
            existing_row = df[existing_mask].iloc[0]
            if is_record_complete(existing_row):
                if existing_row.get('s2_doy', 0) == -1:
                    logging.debug(f"跳过已标记为无效的tile: {tile_id}")
                else:
                    logging.debug(f"跳过已完成的tile: {tile_id}")
                continue
        
        pending_files.append(tiff_file)
    
    logging.info(f"需要处理的文件数量: {len(pending_files)} / {len(tiff_files)}")
    
    if not pending_files:
        logging.info("所有文件都已处理完成")
        return
    
    # 开始处理
    start_time = time.time()
    processed_count = 0
    
    # 使用更保守的线程池设置
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 分批提交任务
        for batch_start in range(0, len(pending_files), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(pending_files))
            batch_files = pending_files[batch_start:batch_end]
            
            logging.info(f"处理批次 {batch_start//args.batch_size + 1}: {len(batch_files)} 个文件")
            
            # 提交当前批次的任务
            future_to_file = {
                executor.submit(process_single_tiff, tiff_file, args.year, args.request_delay): tiff_file 
                for tiff_file in batch_files
            }
            
            # 处理完成的任务
            for future in concurrent.futures.as_completed(future_to_file):
                tiff_file = future_to_file[future]
                try:
                    tile_id, s2_doy, s1_asc_doy, s1_desc_doy = future.result()
                    
                    # 更新DataFrame
                    df = update_dataframe_with_result(df, tile_id, s2_doy, s1_asc_doy, s1_desc_doy)
                    
                    processed_count += 1
                    
                    # 记录处理结果类型
                    if s2_doy == -1 and s1_asc_doy == -1 and s1_desc_doy == -1:
                        logging.info(f"📋 {tile_id}: 已跳过（边界框问题）")
                    else:
                        logging.info(f"✅ {tile_id}: 处理完成 (S2={s2_doy}, S1_ASC={s1_asc_doy}, S1_DESC={s1_desc_doy})")
                    
                    # 每处理完3个文件就保存一次（减少保存频率）
                    if processed_count % 3 == 0:
                        save_results_to_excel(df, excel_path)
                        elapsed_time = time.time() - start_time
                        avg_time = elapsed_time / processed_count
                        remaining_files = len(pending_files) - processed_count
                        eta = remaining_files * avg_time
                        
                        logging.info(f"进度: {processed_count}/{len(pending_files)} "
                                   f"({processed_count/len(pending_files)*100:.1f}%), "
                                   f"平均用时: {avg_time:.1f}s/文件, "
                                   f"预计剩余时间: {eta/60:.1f}分钟")
                    
                except Exception as e:
                    logging.error(f"处理文件 {tiff_file} 时发生异常: {e}")
            
            # 批次完成后休息更长时间，减少API压力
            if batch_end < len(pending_files):
                batch_delay = min(30, args.batch_size * 2)  # 根据批次大小调整延迟
                logging.info(f"批次完成，休息 {batch_delay} 秒...")
                time.sleep(batch_delay)
    
    # 最终保存
    save_results_to_excel(df, excel_path)
    
    # 统计各种处理结果
    skipped_count = len(df[df['s2_doy'] == -1])  # 跳过的tile数量
    valid_count = len(df[df['s2_doy'] >= 0])     # 正常处理的tile数量
    
    # 统计总结
    total_time = time.time() - start_time
    logging.info("="*60)
    logging.info("DOY统计任务完成")
    logging.info(f"处理文件数: {processed_count}")
    logging.info(f"  - 正常处理: {valid_count}")
    logging.info(f"  - 跳过处理: {skipped_count} (边界框问题)")
    logging.info(f"总用时: {total_time/60:.1f} 分钟")
    if processed_count > 0:
        logging.info(f"平均用时: {total_time/processed_count:.1f} 秒/文件")
    logging.info(f"结果已保存到: {excel_path}")
    logging.info("="*60)

if __name__ == "__main__":
    main()