#!/usr/bin/env python3
"""
优化的PCA可视化脚本 - 专注于类别分离和性能
使用监督降维方法提高太阳能板和others的分离度
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import time
from typing import Tuple, Dict, Any, List, Optional
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import matplotlib.patches as mpatches

# 本地导入
from data_preprocessing import load_and_dequantize_representation, identify_valid_pixels, extract_valid_data

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pca_visualization_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OptimizedPCAVisualizer:
    """
    优化的PCA可视化器，专注于类别分离和性能
    """
    
    def __init__(self, data_dir: str, output_dir: str, year: int = 2024, random_seed: int = 42):
        """
        初始化可视化器
        
        Args:
            data_dir: 数据目录路径
            output_dir: 输出目录路径
            year: 年份
            random_seed: 随机种子
        """
        self.data_dir = data_dir  # 保持为字符串类型，与增强版本一致
        self.output_dir = Path(output_dir)
        self.year = year
        self.random_seed = random_seed
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据存储
        self.embeddings = None
        self.labels = None
        self.coords = None  # 添加坐标存储
        self.sampled_embeddings = None
        self.sampled_labels = None
        self.preprocessed_embeddings = None
        self.pca_embeddings = None
        
        # 模型存储
        self.scaler = None
        self.feature_selector = None
        self.pca_model = None
        self.lda_model = None
        
        logger.info(f"初始化OptimizedPCAVisualizer - 年份: {year}, 随机种子: {random_seed}")

    def load_data(self) -> None:
        """加载数据"""
        logger.info("加载数据...")
        
        try:
            # 文件路径 - 与增强版本保持一致
            representation_path = os.path.join(self.data_dir, f"{self.year}_change_detection_test_tile_map_10m_utm31n_128bands.npy")
            scales_path = os.path.join(self.data_dir, f"{self.year}_change_detection_test_tile_map_10m_utm31n_scales.npy")
            labels_path = os.path.join(self.data_dir, "2024_change_detection_test_tile_labels.npy")
            
            # 检查文件是否存在
            for path, name in [(representation_path, "representation"), (scales_path, "scales"), (labels_path, "labels")]:
                if not os.path.exists(path):
                    logger.error(f"{name.capitalize()} 文件不存在: {path}")
                    raise FileNotFoundError(f"{name.capitalize()} 文件不存在")
            
            logger.info("加载表示数据...")
            self.embeddings = load_and_dequantize_representation(representation_path, scales_path)
            logger.info(f"加载的嵌入向量形状: {self.embeddings.shape}")
            
            logger.info("加载标签...")
            self.labels = np.load(labels_path)
            logger.info(f"加载的标签形状: {self.labels.shape}")
            
            # 识别有效像素
            logger.info("识别有效像素...")
            valid_mask = identify_valid_pixels(self.embeddings)
            logger.info(f"有效像素: {np.sum(valid_mask):,} / {len(valid_mask):,}")
            
            # 提取有效数据
            self.embeddings, self.labels, self.coords = extract_valid_data(
                self.embeddings, self.labels, valid_mask
            )
            
            logger.info(f"最终数据形状:")
            logger.info(f"  嵌入向量: {self.embeddings.shape}")
            logger.info(f"  标签: {self.labels.shape}")
            logger.info(f"  坐标: {self.coords.shape}")
            
            # 数据统计
            solar_count = np.sum(self.labels == 1)
            others_count = np.sum(self.labels == 0)
            logger.info(f"数据分布:")
            logger.info(f"  太阳能板: {solar_count:,} ({solar_count/len(self.labels)*100:.2f}%)")
            logger.info(f"  其他类别: {others_count:,} ({others_count/len(self.labels)*100:.2f}%)")
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise

    def sample_data(self, others_multiplier: int = 5, max_solar_samples: int = 50000) -> None:
        """
        平衡采样数据以提高可视化效果和性能
        
        Args:
            others_multiplier: others类别相对于太阳能板的倍数
            max_solar_samples: 太阳能板最大采样数量
        """
        logger.info("开始数据采样...")
        
        # 分离不同类别
        solar_mask = self.labels == 1
        others_mask = self.labels == 0
        
        solar_embeddings = self.embeddings[solar_mask]
        others_embeddings = self.embeddings[others_mask]
        
        # 采样太阳能板数据
        n_solar = len(solar_embeddings)
        if max_solar_samples and n_solar > max_solar_samples:
            np.random.seed(self.random_seed)
            solar_indices = np.random.choice(n_solar, max_solar_samples, replace=False)
            solar_embeddings = solar_embeddings[solar_indices]
            n_solar_sampled = max_solar_samples
        else:
            n_solar_sampled = n_solar
        
        # 采样others数据
        n_others_target = n_solar_sampled * others_multiplier
        n_others = len(others_embeddings)
        
        if n_others > n_others_target:
            np.random.seed(self.random_seed + 1)
            others_indices = np.random.choice(n_others, n_others_target, replace=False)
            others_embeddings = others_embeddings[others_indices]
            n_others_sampled = n_others_target
        else:
            n_others_sampled = n_others
        
        # 合并采样数据
        self.sampled_embeddings = np.vstack([solar_embeddings, others_embeddings])
        self.sampled_labels = np.hstack([
            np.ones(n_solar_sampled, dtype=int),
            np.zeros(n_others_sampled, dtype=int)
        ])
        
        # 随机打乱
        np.random.seed(self.random_seed + 2)
        shuffle_indices = np.random.permutation(len(self.sampled_embeddings))
        self.sampled_embeddings = self.sampled_embeddings[shuffle_indices]
        self.sampled_labels = self.sampled_labels[shuffle_indices]
        
        logger.info(f"数据采样完成:")
        logger.info(f"  采样后形状: {self.sampled_embeddings.shape}")
        logger.info(f"  太阳能板: {np.sum(self.sampled_labels == 1):,}")
        logger.info(f"  其他类别: {np.sum(self.sampled_labels == 0):,}")

    def preprocess_data_for_separation(self, use_feature_selection: bool = True, 
                                     n_features: int = 100) -> None:
        """
        针对类别分离优化的数据预处理
        
        Args:
            use_feature_selection: 是否使用特征选择
            n_features: 选择的特征数量
        """
        logger.info("开始针对类别分离的数据预处理...")
        
        # 1. 特征缩放
        self.scaler = RobustScaler()
        scaled_embeddings = self.scaler.fit_transform(self.sampled_embeddings)
        
        # 2. 特征选择 - 选择对分类最有用的特征
        if use_feature_selection:
            logger.info(f"使用特征选择，选择前{n_features}个最重要的特征...")
            self.feature_selector = SelectKBest(score_func=f_classif, k=min(n_features, scaled_embeddings.shape[1]))
            self.preprocessed_embeddings = self.feature_selector.fit_transform(scaled_embeddings, self.sampled_labels)
            
            # 获取特征重要性分数
            feature_scores = self.feature_selector.scores_
            selected_features = self.feature_selector.get_support()
            logger.info(f"选择了 {np.sum(selected_features)} 个特征")
            logger.info(f"平均特征分数: {np.mean(feature_scores[selected_features]):.2f}")
        else:
            self.preprocessed_embeddings = scaled_embeddings
        
        logger.info(f"预处理后数据形状: {self.preprocessed_embeddings.shape}")

    def apply_supervised_dimensionality_reduction(self) -> None:
        """
        应用监督降维方法以最大化类别分离
        """
        logger.info("应用监督降维方法...")
        
        # 1. 首先使用LDA进行监督降维（最多1维，因为只有2个类别）
        try:
            self.lda_model = LinearDiscriminantAnalysis(n_components=1)
            lda_embeddings = self.lda_model.fit_transform(self.preprocessed_embeddings, self.sampled_labels)
            
            # 计算LDA分离度
            solar_lda = lda_embeddings[self.sampled_labels == 1]
            others_lda = lda_embeddings[self.sampled_labels == 0]
            separation_score = abs(np.mean(solar_lda) - np.mean(others_lda)) / (np.std(solar_lda) + np.std(others_lda))
            
            logger.info(f"LDA分离度分数: {separation_score:.4f}")
            
        except Exception as e:
            logger.warning(f"LDA失败: {e}, 使用标准PCA")
            lda_embeddings = None
        
        # 2. 使用PCA获得第二个维度
        self.pca_model = PCA(n_components=2, random_state=self.random_seed)
        pca_embeddings = self.pca_model.fit_transform(self.preprocessed_embeddings)
        
        # 3. 组合LDA和PCA结果
        if lda_embeddings is not None:
            # 使用LDA作为第一维，PCA第二主成分作为第二维
            self.pca_embeddings = np.column_stack([
                lda_embeddings.flatten(),
                pca_embeddings[:, 1]  # 使用PCA的第二主成分
            ])
            logger.info("使用LDA+PCA组合降维")
        else:
            # 仅使用PCA
            self.pca_embeddings = pca_embeddings
            logger.info("使用标准PCA降维")
        
        # 计算解释方差
        explained_variance = self.pca_model.explained_variance_ratio_
        logger.info(f"PCA解释方差比: {explained_variance}")
        logger.info(f"总解释方差: {np.sum(explained_variance):.4f}")
        
        # 计算类别分离统计
        self._calculate_separation_metrics()

    def _calculate_separation_metrics(self) -> None:
        """计算类别分离指标"""
        solar_points = self.pca_embeddings[self.sampled_labels == 1]
        others_points = self.pca_embeddings[self.sampled_labels == 0]
        
        # 计算中心点距离
        solar_center = np.mean(solar_points, axis=0)
        others_center = np.mean(others_points, axis=0)
        center_distance = np.linalg.norm(solar_center - others_center)
        
        # 计算类内方差
        solar_var = np.mean(np.var(solar_points, axis=0))
        others_var = np.mean(np.var(others_points, axis=0))
        avg_intra_var = (solar_var + others_var) / 2
        
        # 分离度指标
        separation_ratio = center_distance / np.sqrt(avg_intra_var)
        
        logger.info(f"类别分离指标:")
        logger.info(f"  中心点距离: {center_distance:.4f}")
        logger.info(f"  平均类内方差: {avg_intra_var:.4f}")
        logger.info(f"  分离度比率: {separation_ratio:.4f}")

    def create_optimized_visualization(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        创建优化的2D PCA可视化
        
        Args:
            figsize: 图像大小
        """
        logger.info("创建优化的2D PCA可视化...")
        
        # 设置绘图样式
        plt.style.use('default')
        
        # 创建图像
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # 分离不同类别的点
        solar_mask = self.sampled_labels == 1
        others_mask = self.sampled_labels == 0
        
        solar_points = self.pca_embeddings[solar_mask]
        others_points = self.pca_embeddings[others_mask]
        
        # 绘制散点图
        ax.scatter(others_points[:, 0], others_points[:, 1], 
                  c='lightblue', alpha=0.6, s=20, label=f'Others ({len(others_points):,})', 
                  edgecolors='none')
        
        ax.scatter(solar_points[:, 0], solar_points[:, 1], 
                  c='red', alpha=0.8, s=25, label=f'Solar Panels ({len(solar_points):,})', 
                  edgecolors='darkred', linewidths=0.5)
        
        # 添加中心点
        solar_center = np.mean(solar_points, axis=0)
        others_center = np.mean(others_points, axis=0)
        
        ax.scatter(solar_center[0], solar_center[1], 
                  c='darkred', s=200, marker='x', linewidths=3, 
                  label='Solar Center')
        
        ax.scatter(others_center[0], others_center[1], 
                  c='darkblue', s=200, marker='x', linewidths=3, 
                  label='Others Center')
        
        # 设置标签和标题
        ax.set_xlabel('First Discriminant Component', fontsize=12)
        ax.set_ylabel('Second Principal Component', fontsize=12)
        ax.set_title(f'Optimized PCA Visualization - Solar Panel Detection ({self.year})', 
                    fontsize=14, fontweight='bold')
        
        # 添加图例
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 设置坐标轴比例相等以保持真实的距离关系
        ax.set_aspect('equal', adjustable='box')
        
        # 保存图像
        output_path = self.output_dir / f'pca_visualization_optimized_{self.year}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"优化可视化已保存: {output_path}")
        
        plt.show()

    def run_optimized_pipeline(self, others_multiplier: int = 5, 
                             max_solar_samples: int = 50000,
                             use_feature_selection: bool = True,
                             n_features: int = 100) -> Dict:
        """
        运行完整的优化管道
        
        Args:
            others_multiplier: others类别倍数
            max_solar_samples: 最大太阳能板采样数
            use_feature_selection: 是否使用特征选择
            n_features: 选择的特征数量
            
        Returns:
            包含结果的字典
        """
        start_time = time.time()
        logger.info("开始运行优化的PCA可视化管道...")
        
        try:
            # 1. 加载数据
            self.load_data()
            
            # 2. 采样数据
            self.sample_data(others_multiplier=others_multiplier, 
                           max_solar_samples=max_solar_samples)
            
            # 3. 预处理数据
            self.preprocess_data_for_separation(use_feature_selection=use_feature_selection,
                                              n_features=n_features)
            
            # 4. 应用监督降维
            self.apply_supervised_dimensionality_reduction()
            
            # 5. 创建可视化
            self.create_optimized_visualization()
            
            total_time = time.time() - start_time
            
            # 编译结果
            results = {
                'execution_time': total_time,
                'data_shape': self.sampled_embeddings.shape,
                'solar_panels': int(np.sum(self.sampled_labels == 1)),
                'others': int(np.sum(self.sampled_labels == 0)),
                'pca_shape': self.pca_embeddings.shape,
                'explained_variance': self.pca_model.explained_variance_ratio_.tolist() if self.pca_model else None,
                'feature_selection_used': use_feature_selection,
                'selected_features': int(self.preprocessed_embeddings.shape[1])
            }
            
            logger.info(f"优化管道完成! 总耗时: {total_time:.2f}秒")
            self._print_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"管道执行失败: {e}")
            raise

    def _print_summary(self, results: Dict) -> None:
        """打印结果摘要"""
        logger.info("=" * 60)
        logger.info("优化PCA可视化结果摘要")
        logger.info("=" * 60)
        logger.info(f"执行时间: {results['execution_time']:.2f}秒")
        logger.info(f"数据形状: {results['data_shape']}")
        logger.info(f"太阳能板: {results['solar_panels']:,}")
        logger.info(f"其他类别: {results['others']:,}")
        logger.info(f"PCA形状: {results['pca_shape']}")
        if results['explained_variance']:
            logger.info(f"解释方差: {results['explained_variance']}")
        logger.info(f"特征选择: {'是' if results['feature_selection_used'] else '否'}")
        logger.info(f"选择特征数: {results['selected_features']}")
        logger.info("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='优化的PCA可视化 - 专注于类别分离')
    parser.add_argument('--data_dir', type=str,
                       default='/maps/zf281/btfm4rs/data/downstream/pv_detection/change_detection_test_tile',
                       help='Directory containing the data files')
    parser.add_argument('--output_dir', type=str,
                       default='/maps/zf281/btfm4rs/src/pv_detection',
                       help='Directory to save visualization outputs')
    parser.add_argument('--year', type=int, default=2024, help='年份 (默认: 2024)')
    parser.add_argument('--others_multiplier', type=int, default=5, 
                       help='others类别相对于太阳能板的倍数 (默认: 5)')
    parser.add_argument('--max_solar_samples', type=int, default=5000000,
                       help='太阳能板最大采样数 (默认: 5000000)')
    parser.add_argument('--use_feature_selection', action='store_true', default=True,
                       help='使用特征选择 (默认: True)')
    parser.add_argument('--n_features', type=int, default=100,
                       help='选择的特征数量 (默认: 100)')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = OptimizedPCAVisualizer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        year=args.year,
        random_seed=args.random_seed
    )
    
    # 运行管道
    results = visualizer.run_optimized_pipeline(
        others_multiplier=args.others_multiplier,
        max_solar_samples=args.max_solar_samples,
        use_feature_selection=args.use_feature_selection,
        n_features=args.n_features
    )
    
    logger.info("优化PCA可视化完成!")


if __name__ == "__main__":
    main()