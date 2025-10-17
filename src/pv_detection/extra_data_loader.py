import os
import numpy as np
import rasterio
from typing import Dict, List, Tuple, Optional
import logging
import re
from pathlib import Path
from skimage.transform import resize

logger = logging.getLogger(__name__)

class ExtraDataLoader:
    """
    加载额外训练数据的类，处理不同的标注规则和时间范围
    """
    
    def __init__(self, extra_data_dir: str, uk_data_dir: str):
        """
        初始化额外数据加载器
        
        Args:
            extra_data_dir: 额外训练数据目录路径
            uk_data_dir: UK数据目录路径
        """
        self.extra_data_dir = Path(extra_data_dir)
        self.uk_data_dir = Path(uk_data_dir)
        self.years = list(range(2017, 2025))  # 2017-2024
        
    def parse_label_filename(self, filename: str) -> Dict:
        """
        解析标签文件名，提取相关信息
        
        Args:
            filename: 标签文件名
            
        Returns:
            解析后的信息字典
        """
        filename = filename.replace('_label.tif', '')
        
        # 检查是否是复杂命名格式 (包含时间范围)
        complex_pattern = r'(\d{4})_(\d{4})_(-?\d+)_(\d{4})_(\d{4})_(-?\d+)_(grid_-?\d+\.\d+_\d+\.\d+)'
        complex_match = re.match(complex_pattern, filename)
        
        if complex_match:
            start_year1, end_year1, label_type1, start_year2, end_year2, label_type2, grid_id = complex_match.groups()
            return {
                'type': 'complex',
                'grid_id': grid_id,
                'time_ranges': [
                    {
                        'start_year': int(start_year1),
                        'end_year': int(end_year1),
                        'label_type': int(label_type1)
                    },
                    {
                        'start_year': int(start_year2),
                        'end_year': int(end_year2),
                        'label_type': int(label_type2)
                    }
                ]
            }
        
        # 检查是否是年份特定格式 {year}_{grid_id}
        yearly_pattern = r'(\d{4})_(grid_-?\d+\.\d+_\d+\.\d+)'
        yearly_match = re.match(yearly_pattern, filename)
        
        if yearly_match:
            year, grid_id = yearly_match.groups()
            return {
                'type': 'yearly',
                'grid_id': grid_id,
                'year': int(year)
            }
        
        # 简单命名格式
        simple_pattern = r'(grid_-?\d+\.\d+_\d+\.\d+)'
        simple_match = re.match(simple_pattern, filename)
        if simple_match:
            grid_id = simple_match.group(1)
            return {
                'type': 'simple',
                'grid_id': grid_id
            }
        else:
            raise ValueError(f"无法解析文件名: {filename}")
    
    def load_label_tif(self, label_path: str) -> np.ndarray:
        """
        加载标签TIFF文件
        
        Args:
            label_path: 标签文件路径
            
        Returns:
            标签数组
        """
        with rasterio.open(label_path) as src:
            label_data = src.read(1)  # 读取第一个波段
            return label_data
    
    def load_grid_data(self, grid_id: str, year: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        加载指定grid和年份的数据
        
        Args:
            grid_id: grid标识符
            year: 年份
            
        Returns:
            (embeddings, scales) 元组，如果文件不存在则返回 (None, None)
        """
        grid_dir = self.uk_data_dir / str(year) / grid_id
        
        if not grid_dir.exists():
            return None, None
            
        npy_path = grid_dir / f"{grid_id}.npy"
        scales_path = grid_dir / f"{grid_id}_scales.npy"
        
        if not npy_path.exists() or not scales_path.exists():
            return None, None
            
        try:
            embeddings = np.load(npy_path)
            scales = np.load(scales_path)
            return embeddings, scales
        except Exception as e:
            logger.warning(f"加载数据失败 {grid_id} {year}: {e}")
            return None, None
    
    def resize_label_if_needed(self, label: np.ndarray, target_shape: Tuple[int, int], 
                              label_path: str) -> np.ndarray:
        """
        如果需要，调整标签大小以匹配目标形状
        
        Args:
            label: 标签数组
            target_shape: 目标形状 (H, W)
            label_path: 标签文件路径（用于日志）
            
        Returns:
            调整后的标签数组
        """
        if label.shape != target_shape:
            logger.info(f"调整标签大小: {label_path} 从 {label.shape} 到 {target_shape}")
            # 使用最近邻插值保持标签的离散性
            label_resized = resize(label, target_shape, order=0, preserve_range=True, anti_aliasing=False)
            return label_resized.astype(label.dtype)
        return label
    
    def extract_pixel_features_from_coordinates(self, embeddings: np.ndarray, scales: np.ndarray,
                                              coordinates: List[Tuple[int, int]]) -> List[Dict]:
        """
        从坐标提取单个像素的特征（用于像素级学习）
        
        Args:
            embeddings: 嵌入数据 (H, W, C)
            scales: 缩放数据 (H, W)
            coordinates: 坐标列表 [(row, col), ...]
            
        Returns:
            包含pixel_feature和coord的字典列表 [{'pixel_feature': feature, 'coord': (row, col)}, ...]
        """
        pixel_data = []
        
        for row, col in coordinates:
            # 检查边界
            if (row >= 0 and row < embeddings.shape[0] and
                col >= 0 and col < embeddings.shape[1]):
                
                # 提取单个像素的特征
                pixel_embeddings = embeddings[row, col, :]  # Shape: (C,)
                pixel_scale = scales[row, col]  # Shape: ()
                
                # 手动反量化
                # 将int8转换为float32
                pixel_float = pixel_embeddings.astype(np.float32)
                
                # 反量化
                pixel_feature = pixel_float * pixel_scale
                
                pixel_data.append({
                    'pixel_feature': pixel_feature,
                    'coord': (row, col)
                })
        
        return pixel_data

    def extract_patches_from_coordinates(self, embeddings: np.ndarray, scales: np.ndarray,
                                       coordinates: List[Tuple[int, int]], 
                                       patch_size: int = 3) -> List[Dict]:
        """
        从坐标提取patches（用于CNN训练）
        
        Args:
            embeddings: 嵌入数据 (H, W, C)
            scales: 缩放数据 (H, W)
            coordinates: 坐标列表 [(row, col), ...]
            patch_size: patch大小
            
        Returns:
            包含patch和coord的字典列表 [{'patch': patch, 'coord': (row, col)}, ...]
        """
        patches_data = []
        half_size = patch_size // 2
        
        for row, col in coordinates:
            # 检查边界
            if (row >= half_size and row < embeddings.shape[0] - half_size and
                col >= half_size and col < embeddings.shape[1] - half_size):
                
                # 提取patch
                patch_embeddings = embeddings[row-half_size:row+half_size+1, 
                                            col-half_size:col+half_size+1, :]
                patch_scales = scales[row-half_size:row+half_size+1, 
                                    col-half_size:col+half_size+1]
                
                # 手动反量化 (不调用load_and_dequantize_representation，因为它需要文件路径)
                # 将int8转换为float32
                patch_float = patch_embeddings.astype(np.float32)
                
                # 扩展scales维度以匹配patch维度: (H, W) -> (H, W, 1)
                scales_expanded = patch_scales[..., np.newaxis]
                
                # 反量化
                patch = patch_float * scales_expanded
                
                # 转换为PyTorch格式: (C, H, W)
                patch = patch.transpose(2, 0, 1)
                
                patches_data.append({
                    'patch': patch,
                    'coord': (row, col)
                })
        
        return patches_data

    def extract_coordinates_from_label(self, label: np.ndarray, 
                                     target_values: List[int]) -> List[Tuple[int, int]]:
        """
        从标签中提取指定值的坐标
        
        Args:
            label: 标签数组
            target_values: 目标值列表
            
        Returns:
            坐标列表 [(row, col), ...]
        """
        coordinates = []
        for value in target_values:
            rows, cols = np.where(label == value)
            coordinates.extend(list(zip(rows, cols)))
        return coordinates
    
    def process_simple_label(self, label_path: str, grid_id: str) -> Dict[int, Dict[str, List]]:
        """
        处理简单格式的标签文件
        
        Args:
            label_path: 标签文件路径
            grid_id: grid标识符
            
        Returns:
            按年份组织的坐标字典 {year: {'positive': [...], 'negative': [...]}}
        """
        label = self.load_label_tif(label_path)
        unique_values = np.unique(label)
        
        logger.info(f"处理简单标签 {label_path}, 唯一值: {unique_values}")
        
        result = {}
        
        for year in self.years:
            embeddings, scales = self.load_grid_data(grid_id, year)
            if embeddings is None:
                continue
                
            # 调整标签大小以匹配embeddings
            target_shape = embeddings.shape[:2]  # (H, W)
            label_resized = self.resize_label_if_needed(label, target_shape, label_path)
            
            year_data = {'positive': [], 'negative': []}
            
            if 1 in unique_values and 0 in unique_values:
                # 1和0的情况：1是正样本，0是背景
                positive_coords = self.extract_coordinates_from_label(label_resized, [1])
                if positive_coords:
                    year_data['positive'] = self.extract_pixel_features_from_coordinates(
                        embeddings, scales, positive_coords
                    )
                
            elif -1 in unique_values and 0 in unique_values:
                # -1和0的情况：-1是强制负样本，0是背景
                negative_coords = self.extract_coordinates_from_label(label_resized, [-1])
                if negative_coords:
                    year_data['negative'] = self.extract_pixel_features_from_coordinates(
                        embeddings, scales, negative_coords
                    )
                
            result[year] = year_data
            
        return result
    
    def process_simple_label_for_patches(self, label_path: str, grid_id: str) -> Dict[int, Dict[str, List]]:
        """
        处理简单格式的标签文件并提取patch数据（用于CNN训练）
        
        Args:
            label_path: 标签文件路径
            grid_id: 网格ID
            
        Returns:
            按年份组织的patch数据字典
        """
        result = {year: {'positive': [], 'negative': []} for year in self.years}
        
        # 加载标签
        label = self.load_label_tif(label_path)
        
        # 加载对应的网格数据
        grid_data = self.load_grid_data(grid_id)
        if grid_data is None:
            return result
        
        # 调整标签大小以匹配嵌入
        label_resized = cv2.resize(label, (grid_data['embeddings'].shape[1], grid_data['embeddings'].shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # 获取唯一值
        unique_values = np.unique(label_resized)
        
        # 处理每个年份
        for year in self.years:
            # 正样本（标签值为1）
            if 1 in unique_values:
                pos_coords = np.column_stack(np.where(label_resized == 1))
                if len(pos_coords) > 0:
                    pos_patches = self.extract_patches_from_coordinates(
                        grid_data['embeddings'], pos_coords, grid_data['scale']
                    )
                    result[year]['positive'].extend(pos_patches)
            
            # 负样本（标签值为0）
            if 0 in unique_values:
                neg_coords = np.column_stack(np.where(label_resized == 0))
                if len(neg_coords) > 0:
                    neg_patches = self.extract_patches_from_coordinates(
                        grid_data['embeddings'], neg_coords, grid_data['scale']
                    )
                    result[year]['negative'].extend(neg_patches)
        
        return result
    
    def process_yearly_label_for_patches(self, label_path: str, grid_id: str, year: int) -> Dict[int, Dict[str, List]]:
        """
        处理年份特定格式的标签文件并提取patch数据（用于CNN训练）
        
        Args:
            label_path: 标签文件路径
            grid_id: 网格ID
            year: 年份
            
        Returns:
            按年份组织的patch数据字典
        """
        result = {year: {'positive': [], 'negative': []} for year in self.years}
        
        # 只处理指定年份
        if year not in self.years:
            return result
        
        # 加载标签
        label = self.load_label_tif(label_path)
        
        # 加载对应的网格数据
        grid_data = self.load_grid_data(grid_id)
        if grid_data is None:
            return result
        
        # 调整标签大小以匹配嵌入
        label_resized = cv2.resize(label, (grid_data['embeddings'].shape[1], grid_data['embeddings'].shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # 正样本（标签值为1）
        pos_coords = np.column_stack(np.where(label_resized == 1))
        if len(pos_coords) > 0:
            pos_patches = self.extract_patches_from_coordinates(
                grid_data['embeddings'], pos_coords, grid_data['scale']
            )
            result[year]['positive'].extend(pos_patches)
        
        # 负样本（标签值为-1）
        neg_coords = np.column_stack(np.where(label_resized == -1))
        if len(neg_coords) > 0:
            neg_patches = self.extract_patches_from_coordinates(
                grid_data['embeddings'], neg_coords, grid_data['scale']
            )
            result[year]['negative'].extend(neg_patches)
        
        return result
    
    def process_complex_label_for_patches(self, label_path: str, grid_id: str, time_ranges: List[Dict]) -> Dict[int, Dict[str, List]]:
        """
        处理复杂格式的标签文件并提取patch数据（用于CNN训练）
        
        Args:
            label_path: 标签文件路径
            grid_id: 网格ID
            time_ranges: 时间范围列表
            
        Returns:
            按年份组织的patch数据字典
        """
        result = {year: {'positive': [], 'negative': []} for year in self.years}
        
        # 加载标签
        label = self.load_label_tif(label_path)
        
        # 加载对应的网格数据
        grid_data = self.load_grid_data(grid_id)
        if grid_data is None:
            return result
        
        # 调整标签大小以匹配嵌入
        label_resized = cv2.resize(label, (grid_data['embeddings'].shape[1], grid_data['embeddings'].shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # 处理每个时间范围
        for time_range in time_ranges:
            start_year = time_range['start_year']
            end_year = time_range['end_year']
            label_value = time_range['label']
            
            # 确定这个时间范围内的年份
            years_in_range = [year for year in self.years if start_year <= year <= end_year]
            
            if label_value == 1:
                # 正样本
                pos_coords = np.column_stack(np.where(label_resized == 1))
                if len(pos_coords) > 0:
                    pos_patches = self.extract_patches_from_coordinates(
                        grid_data['embeddings'], pos_coords, grid_data['scale']
                    )
                    for year in years_in_range:
                        result[year]['positive'].extend(pos_patches)
            elif label_value == -1:
                # 负样本
                neg_coords = np.column_stack(np.where(label_resized == -1))
                if len(neg_coords) > 0:
                    neg_patches = self.extract_patches_from_coordinates(
                        grid_data['embeddings'], neg_coords, grid_data['scale']
                    )
                    for year in years_in_range:
                        result[year]['negative'].extend(neg_patches)
        
        return result
    
    def load_all_extra_patches(self, years: List[int]) -> Dict[str, np.ndarray]:
        """
        加载所有额外训练数据并转换为patch格式（用于CNN训练）
        
        Args:
            years: 需要加载的年份列表
            
        Returns:
            包含patches、labels、coords和years的字典
        """
        self.years = years
        logger.info("开始加载额外训练数据（patch格式）...")
        
        all_data = {year: {'positive': [], 'negative': []} for year in self.years}
        
        # 遍历所有标签文件
        for label_file in self.extra_data_dir.glob("*_label.tif"):
            try:
                # 解析文件名
                file_info = self.parse_label_filename(label_file.name)
                grid_id = file_info['grid_id']
                
                logger.info(f"处理文件: {label_file.name}, grid_id: {grid_id}")
                
                if file_info['type'] == 'simple':
                    # 处理简单格式
                    year_data = self.process_simple_label_for_patches(str(label_file), grid_id)
                elif file_info['type'] == 'yearly':
                    # 处理年份特定格式
                    year_data = self.process_yearly_label_for_patches(str(label_file), grid_id, file_info['year'])
                else:
                    # 处理复杂格式
                    year_data = self.process_complex_label_for_patches(
                        str(label_file), grid_id, file_info['time_ranges']
                    )
                
                # 合并数据
                for year, data in year_data.items():
                    all_data[year]['positive'].extend(data['positive'])
                    all_data[year]['negative'].extend(data['negative'])
                    
            except Exception as e:
                logger.error(f"处理文件 {label_file.name} 时出错: {e}")
                continue
        
        # 转换为patch格式
        all_patches = []
        all_labels = []
        all_coords = []
        all_years = []
        
        for year in self.years:
            pos_coords = all_data[year]['positive']
            neg_coords = all_data[year]['negative']
            
            # 处理正样本
            for coord_data in pos_coords:
                patch = coord_data['patch']  # Shape: (C, H, W)
                coord = coord_data['coord']
                all_patches.append(patch)
                all_labels.append(1)
                all_coords.append(coord)
                all_years.append(year)
            
            # 处理负样本
            for coord_data in neg_coords:
                patch = coord_data['patch']  # Shape: (C, H, W)
                coord = coord_data['coord']
                all_patches.append(patch)
                all_labels.append(0)
                all_coords.append(coord)
                all_years.append(year)
        
        # 转换为numpy数组
        if all_patches:
            patches = np.array(all_patches)  # Shape: (N, C, H, W)
            labels = np.array(all_labels)
            coords = np.array(all_coords)
            years = np.array(all_years)
        else:
            # 如果没有数据，返回空数组
            patches = np.empty((0, 128, 3, 3), dtype=np.float32)  # 假设128通道，3x3 patch
            labels = np.empty((0,), dtype=np.int64)
            coords = np.empty((0, 2), dtype=np.int32)
            years = np.empty((0,), dtype=np.int32)
        
        # 打印统计信息
        for year in self.years:
            year_mask = years == year if len(years) > 0 else np.array([])
            if len(year_mask) > 0:
                year_labels = labels[year_mask]
                pos_count = np.sum(year_labels == 1)
                neg_count = np.sum(year_labels == 0)
                logger.info(f"年份 {year}: 额外正样本 {pos_count}, 额外负样本 {neg_count}")
        
        logger.info(f"额外训练数据加载完成，总计 {len(patches)} 个patch")
        
        return {
            'patches': patches,
            'labels': labels,
            'coords': coords,
            'years': years
        }
    
    def process_yearly_label(self, label_path: str, grid_id: str, year: int) -> Dict[int, Dict[str, List]]:
        """
        处理年份特定格式的标签文件 {year}_{grid_id}_label.tif
        
        Args:
            label_path: 标签文件路径
            grid_id: grid标识符
            year: 标签对应的年份
            
        Returns:
            按年份组织的坐标字典 {year: {'positive': [...], 'negative': [...]}}
        """
        label = self.load_label_tif(label_path)
        unique_values = np.unique(label)
        
        logger.info(f"处理年份特定标签 {label_path}, 年份: {year}, 唯一值: {unique_values}")
        
        result = {}
        
        # 只处理标签对应的年份
        embeddings, scales = self.load_grid_data(grid_id, year)
        if embeddings is None:
            logger.warning(f"无法加载 grid {grid_id} 年份 {year} 的数据")
            return result
            
        # 调整标签大小以匹配embeddings
        target_shape = embeddings.shape[:2]  # (H, W)
        label_resized = self.resize_label_if_needed(label, target_shape, label_path)
        
        year_data = {'positive': [], 'negative': []}
        
        # 处理正样本 (值为1)
        if 1 in unique_values:
            positive_coords = self.extract_coordinates_from_label(label_resized, [1])
            if positive_coords:
                year_data['positive'] = self.extract_pixel_features_from_coordinates(
                    embeddings, scales, positive_coords
                )
        
        # 处理强制负样本 (值为-1)
        if -1 in unique_values:
            negative_coords = self.extract_coordinates_from_label(label_resized, [-1])
            if negative_coords:
                year_data['negative'] = self.extract_pixel_features_from_coordinates(
                    embeddings, scales, negative_coords
                )
        
        result[year] = year_data
        
        return result
    
    def process_complex_label(self, label_path: str, grid_id: str, 
                            time_ranges: List[Dict]) -> Dict[int, Dict[str, List]]:
        """
        处理复杂格式的标签文件
        
        Args:
            label_path: 标签文件路径
            grid_id: grid标识符
            time_ranges: 时间范围列表
            
        Returns:
            按年份组织的坐标字典 {year: {'positive': [...], 'negative': [...]}}
        """
        label = self.load_label_tif(label_path)
        
        logger.info(f"处理复杂标签 {label_path}, 时间范围: {time_ranges}")
        
        result = {}
        
        for year in self.years:
            embeddings, scales = self.load_grid_data(grid_id, year)
            if embeddings is None:
                continue
                
            # 调整标签大小以匹配embeddings
            target_shape = embeddings.shape[:2]  # (H, W)
            label_resized = self.resize_label_if_needed(label, target_shape, label_path)
            
            year_data = {'positive': [], 'negative': []}
            
            # 确定当前年份应该使用哪个标签类型
            for time_range in time_ranges:
                if time_range['start_year'] <= year <= time_range['end_year']:
                    label_type = time_range['label_type']
                    
                    # 提取值为1的坐标
                    coords_with_value_1 = self.extract_coordinates_from_label(label_resized, [1])
                    
                    if coords_with_value_1:
                        if label_type == 1:
                            # 标签类型为1：值为1的地方是正样本
                            year_data['positive'] = self.extract_pixel_features_from_coordinates(
                                embeddings, scales, coords_with_value_1
                            )
                        elif label_type == -1:
                            # 标签类型为-1：值为1的地方是强制负样本
                            year_data['negative'] = self.extract_pixel_features_from_coordinates(
                                embeddings, scales, coords_with_value_1
                            )
                    
                    break
                    
            result[year] = year_data
            
        return result
    
    def load_all_extra_data(self, years: List[int]) -> Dict[str, np.ndarray]:
        """
        加载所有额外训练数据并转换为像素级特征格式
        
        Args:
            years: 需要加载的年份列表
            
        Returns:
            包含features、labels、coords和years的字典
        """
        self.years = years
        logger.info("开始加载额外训练数据...")
        
        all_data = {year: {'positive': [], 'negative': []} for year in self.years}
        
        # 遍历所有标签文件
        for label_file in self.extra_data_dir.glob("*_label.tif"):
            try:
                # 解析文件名
                file_info = self.parse_label_filename(label_file.name)
                grid_id = file_info['grid_id']
                
                logger.info(f"处理文件: {label_file.name}, grid_id: {grid_id}")
                
                if file_info['type'] == 'simple':
                    # 处理简单格式
                    year_data = self.process_simple_label(str(label_file), grid_id)
                elif file_info['type'] == 'yearly':
                    # 处理年份特定格式
                    year_data = self.process_yearly_label(str(label_file), grid_id, file_info['year'])
                else:
                    # 处理复杂格式
                    year_data = self.process_complex_label(
                        str(label_file), grid_id, file_info['time_ranges']
                    )
                
                # 合并数据
                for year, data in year_data.items():
                    all_data[year]['positive'].extend(data['positive'])
                    all_data[year]['negative'].extend(data['negative'])
                    
            except Exception as e:
                logger.error(f"处理文件 {label_file.name} 时出错: {e}")
                continue
        
        # 转换为像素级特征格式
        all_features = []
        all_labels = []
        all_coords = []
        all_years = []
        
        for year in self.years:
            pos_coords = all_data[year]['positive']
            neg_coords = all_data[year]['negative']
            
            # 处理正样本
            for coord_data in pos_coords:
                pixel_feature = coord_data['pixel_feature']  # Shape: (128,)
                coord = coord_data['coord']
                all_features.append(pixel_feature)
                all_labels.append(1)
                all_coords.append(coord)
                all_years.append(year)
            
            # 处理负样本
            for coord_data in neg_coords:
                pixel_feature = coord_data['pixel_feature']  # Shape: (128,)
                coord = coord_data['coord']
                all_features.append(pixel_feature)
                all_labels.append(0)
                all_coords.append(coord)
                all_years.append(year)
        
        # 转换为numpy数组
        if all_features:
            features = np.array(all_features)  # Shape: (N, 128)
            labels = np.array(all_labels)
            coords = np.array(all_coords)
            years = np.array(all_years)
        else:
            # 如果没有数据，返回空数组
            features = np.empty((0, 128), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int64)
            coords = np.empty((0, 2), dtype=np.int32)
            years = np.empty((0,), dtype=np.int32)
        
        # 打印统计信息
        for year in self.years:
            year_mask = years == year if len(years) > 0 else np.array([])
            if len(year_mask) > 0:
                year_labels = labels[year_mask]
                pos_count = np.sum(year_labels == 1)
                neg_count = np.sum(year_labels == 0)
                logger.info(f"年份 {year}: 额外正样本 {pos_count}, 额外负样本 {neg_count}")
        
        logger.info(f"额外训练数据加载完成，总计 {len(features)} 个像素级特征")
        
        return {
            'features': features,
            'labels': labels,
            'coords': coords,
            'years': years
        }

def test_extra_data_loader():
    """测试额外数据加载器"""
    extra_data_dir = "/maps/zf281/btfm4rs/data/downstream/pv_detection/extra_training_data"
    uk_data_dir = "/maps/zf281/btfm4rs/data/downstream/pv_detection/uk"
    
    loader = ExtraDataLoader(extra_data_dir, uk_data_dir)
    
    # 测试文件名解析
    test_files = [
        "grid_-0.65_52.75_label.tif",
        "2017_2019_-1_2020_2024_1_grid_-3.15_51.45_label.tif"
    ]
    
    for filename in test_files:
        try:
            info = loader.parse_label_filename(filename)
            print(f"文件: {filename}")
            print(f"解析结果: {info}")
            print()
        except Exception as e:
            print(f"解析失败 {filename}: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_extra_data_loader()