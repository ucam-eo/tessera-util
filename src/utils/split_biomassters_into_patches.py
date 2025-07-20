#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生物质掩膜分块器
(Patch Generator)

将大的mask TIFF分割成多个矩形patch，满足以下约束：
1. 至多N个patch
2. 包围所有值为1的部分
3. patch之间没有重叠 (Note: current algorithm aims to minimize overlap through component-based
                      generation and merging, but strict non-overlap of final N patches is very hard
                      to guarantee simultaneously with full coverage under N-patch constraint if regions are complex.)
4. 最大patch面积不超过总面积的指定百分比
5. patch中值为1的部分占patch总面积的比值尽可能大
"""

import matplotlib
matplotlib.use("Agg") # Use Agg backend for non-interactive environments

import numpy as np
import rasterio
from rasterio.transform import Affine
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.measure import label, regionprops
import logging
from typing import List, Tuple 
from dataclasses import dataclass
import sys
import os
import pathlib

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
MAX_PATCHES = 40
MAX_PATCH_RATIO = 0.15
MASK_PATH = "/mnt/e/Codes/btfm4rs/agbm_footprint_mask.tif" 
MAX_REFINEMENT_ITERATIONS = 3 # 新增：迭代优化的最大次数

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Patch:
    minx: int
    miny: int
    maxx: int
    maxy: int
    @property
    def width(self) -> int: return self.maxx - self.minx
    @property
    def height(self) -> int: return self.maxy - self.miny
    @property
    def area(self) -> int: return self.width * self.height
    def get_mask_coverage(self, mask: np.ndarray) -> float:
        if self.minx >= self.maxx or self.miny >= self.maxy: return 0.0
        h, w = mask.shape
        eff_miny, eff_maxy = max(0, self.miny), min(h, self.maxy)
        eff_minx, eff_maxx = max(0, self.minx), min(w, self.maxx)
        if eff_miny >= eff_maxy or eff_minx >= eff_maxx: return 0.0
        patch_mask_slice = mask[eff_miny:eff_maxy, eff_minx:eff_maxx]
        if patch_mask_slice.size == 0: return 0.0
        valid_pixels = np.sum(patch_mask_slice == 1)
        total_pixels = patch_mask_slice.size
        return valid_pixels / total_pixels if total_pixels > 0 else 0.0
    def __repr__(self):
        return f"Patch({self.minx}, {self.miny}, {self.maxx}, {self.maxy}) area={self.area}"

class MaskPatcher:
    def __init__(self, mask: np.ndarray, max_patches: int, max_patch_ratio: float):
        self.mask = mask
        self.max_patches = max_patches
        self.max_patch_area = int(mask.size * max_patch_ratio)
        self.height, self.width = mask.shape
        logger.info(f"初始化分块器:")
        logger.info(f"  掩膜尺寸: {self.width} x {self.height}")
        logger.info(f"  最大patches: {max_patches}")
        logger.info(f"  最大patch面积: {self.max_patch_area} ({max_patch_ratio*100:.1f}%)")
        if self.max_patch_area == 0 and mask.size > 0 and mask.size * max_patch_ratio > 0:
             self.max_patch_area = 1 

    def find_connected_components(self) -> List[Tuple[int, int, int, int]]:
        logger.info("寻找连通分量...")
        binary_mask = self.mask == 1
        labeled_mask = label(binary_mask, connectivity=2, background=0)
        props = regionprops(labeled_mask)
        bboxes = []
        for prop in props:
            minr, minc, maxr, maxc = prop.bbox
            bboxes.append((minc, minr, maxc, maxr)) 
        logger.info(f"找到 {len(bboxes)} 个连通分量")
        return bboxes

    def split_large_bbox(self, bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        minx, miny, maxx, maxy = bbox
        area = (maxx - minx) * (maxy - miny)
        if area == 0: return []
        if area <= self.max_patch_area: return [bbox]

        logger.debug(f"分割大bbox: {bbox}, area {area} > max_area {self.max_patch_area}")
        n_splits_w = 1
        current_width, current_height = maxx - minx, maxy - miny
        while (current_width / n_splits_w) * current_height > self.max_patch_area : n_splits_w +=1
        n_splits_h = 1
        while (current_width / n_splits_w) * (current_height / n_splits_h) > self.max_patch_area: n_splits_h +=1
        split_factor_w, split_factor_h = max(1, n_splits_w), max(1, n_splits_h)
        logger.debug(f"  分割因子 (w,h): ({split_factor_w}, {split_factor_h})")

        width_bbox, height_bbox = maxx - minx, maxy - miny
        sub_bboxes = []
        base_split_width = max(1, width_bbox // split_factor_w)
        base_split_height = max(1, height_bbox // split_factor_h)

        for i in range(split_factor_w):
            for j in range(split_factor_h):
                sub_minx = minx + i * base_split_width
                sub_miny = miny + j * base_split_height
                sub_maxx = minx + (i + 1) * base_split_width if i < split_factor_w - 1 else maxx
                sub_maxy = miny + (j + 1) * base_split_height if j < split_factor_h - 1 else maxy
                sub_maxx, sub_maxy = max(sub_minx + 1, sub_maxx), max(sub_miny + 1, sub_maxy)
                clipped_sub_miny, clipped_sub_maxy = max(0, sub_miny), min(self.height, sub_maxy)
                clipped_sub_minx, clipped_sub_maxx = max(0, sub_minx), min(self.width, sub_maxx)
                if clipped_sub_miny < clipped_sub_maxy and clipped_sub_minx < clipped_sub_maxx:
                    if np.any(self.mask[clipped_sub_miny:clipped_sub_maxy, clipped_sub_minx:clipped_sub_maxx] == 1):
                        sub_bboxes.append((sub_minx, sub_miny, sub_maxx, sub_maxy)) 
                elif (sub_maxx - sub_minx) > 0 and (sub_maxy - sub_miny) > 0 :
                    logger.warning(f"Sub-bbox invalid after clip: orig {(sub_minx, sub_miny, sub_maxx, sub_maxy)}, clip {(clipped_sub_minx, clipped_sub_miny, clipped_sub_maxx, clipped_sub_maxy)}")
        if not sub_bboxes and np.any(self.mask[max(0,miny):min(self.height,maxy), max(0,minx):min(self.width,maxx)] == 1):
            logger.warning(f"Splitting failed for {bbox}, returning original.")
            return [bbox] 
        logger.debug(f"  分割产生 {len(sub_bboxes)} sub-bboxes")
        return sub_bboxes

    def merge_nearby_patches(self, patches: List[Patch], merge_threshold: int = 50) -> List[Patch]:
        if len(patches) <= 1: return patches
        current_patches = list(patches) 
        while True: 
            if len(current_patches) <= 1: break
            merged_in_this_pass = False
            next_pass_patches = []
            patch_was_merged_this_pass = [False] * len(current_patches)
            current_patches.sort(key=lambda p: (p.miny, p.minx)) 
            for i in range(len(current_patches)):
                if patch_was_merged_this_pass[i]: continue
                patch1 = current_patches[i]
                accumulated_patch = patch1 
                for j in range(i + 1, len(current_patches)):
                    if patch_was_merged_this_pass[j]: continue
                    patch2 = current_patches[j]
                    if self._can_merge_patches(accumulated_patch, patch2, merge_threshold):
                        potential_new_patch = Patch(
                            min(accumulated_patch.minx, patch2.minx), min(accumulated_patch.miny, patch2.miny),
                            max(accumulated_patch.maxx, patch2.maxx), max(accumulated_patch.maxy, patch2.maxy)
                        )
                        if potential_new_patch.area <= self.max_patch_area:
                            accumulated_patch = potential_new_patch
                            patch_was_merged_this_pass[j] = True 
                            merged_in_this_pass = True
                next_pass_patches.append(accumulated_patch)
                patch_was_merged_this_pass[i] = True 
            current_patches = next_pass_patches 
            if not merged_in_this_pass: break
        logger.info(f"合并后patches数量: {len(current_patches)}")
        return current_patches

    def _can_merge_patches(self, patch1: Patch, patch2: Patch, threshold: int) -> bool:
        if patch1.maxx < patch2.minx: dx = patch2.minx - patch1.maxx
        elif patch2.maxx < patch1.minx: dx = patch1.minx - patch2.maxx
        else: dx = 0
        if patch1.maxy < patch2.miny: dy = patch2.miny - patch1.maxy
        elif patch2.maxy < patch1.miny: dy = patch1.miny - patch2.maxy
        else: dy = 0
        if dx == 0 and dy == 0: return True 
        return np.sqrt(dx*dx + dy*dy) <= threshold

    def _shrink_patch_to_content(self, patch: Patch) -> Patch:
        if patch.minx >= patch.maxx or patch.miny >= patch.maxy: 
            logger.debug(f"Shrink: Patch {patch} invalid.")
            return Patch(patch.minx, patch.miny, patch.minx, patch.miny) 
        eff_miny, eff_maxy = max(0, patch.miny), min(self.height, patch.maxy)
        eff_minx, eff_maxx = max(0, patch.minx), min(self.width, patch.maxx)
        if eff_miny >= eff_maxy or eff_minx >= eff_maxx: 
            logger.debug(f"Shrink: Patch {patch} outside or invalid after clip.")
            return Patch(patch.minx, patch.miny, patch.minx, patch.miny) 
        patch_mask_slice = self.mask[eff_miny:eff_maxy, eff_minx:eff_maxx]
        if patch_mask_slice.size == 0 or np.sum(patch_mask_slice == 1) == 0:
            logger.debug(f"Shrink: Patch {patch} slice no content.")
            return Patch(patch.minx, patch.miny, patch.minx, patch.miny) 
        rows_with_content = np.any(patch_mask_slice == 1, axis=1)
        cols_with_content = np.any(patch_mask_slice == 1, axis=0)
        if not np.any(rows_with_content) or not np.any(cols_with_content):
            logger.debug(f"Shrink: Patch {patch} no projectable content.")
            return Patch(patch.minx, patch.miny, patch.minx, patch.miny) 
        first_row, last_row = np.argmax(rows_with_content), patch_mask_slice.shape[0] - 1 - np.argmax(rows_with_content[::-1])
        first_col, last_col = np.argmax(cols_with_content), patch_mask_slice.shape[1] - 1 - np.argmax(cols_with_content[::-1])
        new_minx, new_miny = eff_minx + first_col, eff_miny + first_row
        new_maxx, new_maxy = eff_minx + last_col + 1, eff_miny + last_row + 1
        shrunk_patch = Patch(new_minx, new_miny, new_maxx, new_maxy)
        logger.debug(f"Shrunk {patch} to {shrunk_patch}")
        return shrunk_patch

    def _truncate_to_max_coverage_greedy(self, patches: List[Patch], target_count: int) -> List[Patch]:
        if len(patches) <= target_count: return patches
        logger.info(f"使用贪心算法从 {len(patches)} 个补丁中选择 {target_count} 个以最大化覆盖...")
        selected_patches = []
        pixels_to_be_covered = (self.mask == 1).astype(bool) 
        candidate_patches = list(patches) 
        for i_selection in range(target_count):
            if not candidate_patches: break
            best_patch_idx, max_newly_covered_pixels = -1, -1
            for i, patch in enumerate(candidate_patches):
                if patch.area == 0: continue
                eff_miny, eff_maxy = max(0, patch.miny), min(self.height, patch.maxy)
                eff_minx, eff_maxx = max(0, patch.minx), min(self.width, patch.maxx)
                num_covered_by_this_patch = 0
                if not (eff_miny >= eff_maxy or eff_minx >= eff_maxx):
                    patch_slice_of_target_mask = pixels_to_be_covered[eff_miny:eff_maxy, eff_minx:eff_maxx]
                    num_covered_by_this_patch = np.sum(patch_slice_of_target_mask)
                if num_covered_by_this_patch > max_newly_covered_pixels:
                    max_newly_covered_pixels, best_patch_idx = num_covered_by_this_patch, i
            if best_patch_idx != -1:
                best_patch = candidate_patches.pop(best_patch_idx)
                selected_patches.append(best_patch)
                logger.debug(f"  贪心选择 #{i_selection+1}: {best_patch}, 新覆盖: {max_newly_covered_pixels} px")
                eff_miny, eff_maxy = max(0, best_patch.miny), min(self.height, best_patch.maxy)
                eff_minx, eff_maxx = max(0, best_patch.minx), min(self.width, best_patch.maxx)
                if eff_miny < eff_maxy and eff_minx < eff_maxx:
                     pixels_to_be_covered[eff_miny:eff_maxy, eff_minx:eff_maxx] = False
            else: logger.info("  贪心选择提前结束"); break 
        logger.info(f"贪心选择完成，选择了 {len(selected_patches)} 个补丁。")
        return selected_patches

    def optimize_patches_and_truncate(self, patches: List[Patch]) -> List[Patch]:
        logger.info("优化并（可能）截断patches...") # Renamed log for clarity
        if not patches: return []
        optimized = [self._shrink_patch_to_content(p) for p in patches]
        optimized = [p for p in optimized if p.area > 0]
        if not optimized: logger.warning("所有 patches 收缩后均为空。"); return []

        if len(optimized) > self.max_patches:
            optimized = self._truncate_to_max_coverage_greedy(optimized, self.max_patches)
        else:
            optimized.sort(key=lambda p: p.get_mask_coverage(self.mask), reverse=True)
        logger.info(f"优化并（可能）截断后， patches 数量: {len(optimized)}")
        return optimized

    def _get_patches_for_uncovered_areas(self, current_selected_patches: List[Patch]) -> List[Patch]:
        logger.debug(f"为当前 {len(current_selected_patches)} 个补丁未覆盖的区域生成新补丁...")
        temp_coverage_mask = np.zeros_like(self.mask, dtype=bool)
        for patch in current_selected_patches:
            eff_miny, eff_maxy = max(0, patch.miny), min(self.height, patch.maxy)
            eff_minx, eff_maxx = max(0, patch.minx), min(self.width, patch.maxx)
            if eff_miny < eff_maxy and eff_minx < eff_maxx:
                temp_coverage_mask[eff_miny:eff_maxy, eff_minx:eff_maxx] = True
        
        currently_uncovered_pixels = (self.mask == 1) & (~temp_coverage_mask)
        newly_generated_patches = []
        if np.any(currently_uncovered_pixels):
            logger.debug("发现当前未覆盖像素，为其生成补丁...")
            labeled_uncovered = label(currently_uncovered_pixels, connectivity=2, background=0)
            props_uncovered = regionprops(labeled_uncovered)
            bboxes_for_uncovered = []
            for prop in props_uncovered:
                minr, minc, maxr, maxc = prop.bbox
                bboxes_for_uncovered.append((minc, minr, maxc, maxr))
            for bbox in bboxes_for_uncovered:
                sub_bboxes = self.split_large_bbox(bbox)
                for sub_bbox in sub_bboxes:
                    p = Patch(*sub_bbox)
                    if p.area > 0: newly_generated_patches.append(p)
            logger.debug(f"为未覆盖区域生成了 {len(newly_generated_patches)} 个新补丁。")
        else:
            logger.debug("未发现当前未覆盖像素。")
        return newly_generated_patches

    def _calculate_uncovered_count(self, patches_to_check: List[Patch]) -> int:
        if not patches_to_check: return np.sum(self.mask == 1)
        check_mask = np.zeros_like(self.mask, dtype=bool)
        for p in patches_to_check:
            eff_miny, eff_maxy = max(0, p.miny), min(self.height, p.maxy)
            eff_minx, eff_maxx = max(0, p.minx), min(self.width, p.maxx)
            if eff_miny < eff_maxy and eff_minx < eff_maxx:
                check_mask[eff_miny:eff_maxy, eff_minx:eff_maxx] = True
        return np.sum((self.mask == 1) & (~check_mask))

    def generate_patches(self) -> List[Patch]:
        logger.info("开始生成patches...")
        bboxes = self.find_connected_components()
        if not bboxes: logger.warning("未找到任何有效区域"); return []

        initial_patches_from_components: List[Patch] = []
        for bbox in bboxes:
            sub_bboxes = self.split_large_bbox(bbox)
            for sub_bbox in sub_bboxes: initial_patches_from_components.append(Patch(*sub_bbox))
        initial_patches_from_components = [p for p in initial_patches_from_components if p.area > 0]
        if not initial_patches_from_components: logger.warning("分割后未产生有效patches。"); return []
        logger.info(f"初始patches数量 (分割和过滤后): {len(initial_patches_from_components)}")

        logger.info(f"进入主合并阶段，当前 patches: {len(initial_patches_from_components)}")
        merged_patches = self.merge_nearby_patches(initial_patches_from_components) 
        logger.info(f"主合并阶段后 patches: {len(merged_patches)}")
        
        # This set of patches should ideally cover everything, but might be > max_patches
        candidate_patches_for_selection = [self._shrink_patch_to_content(p) for p in merged_patches]
        candidate_patches_for_selection = [p for p in candidate_patches_for_selection if p.area > 0] 
        logger.info(f"收缩后，准备进行选择的候选patches数量: {len(candidate_patches_for_selection)}")
        if not candidate_patches_for_selection: logger.warning("所有 patches 在收缩后均为空。"); return []

        # Initial selection of N patches
        current_best_N_patches = self.optimize_patches_and_truncate(list(candidate_patches_for_selection)) # Pass a copy
        
        uncovered_count = self._calculate_uncovered_count(current_best_N_patches)
        logger.info(f"初步选择 {len(current_best_N_patches)} 个补丁后，未覆盖像素: {uncovered_count:,}")

        refinement_iteration = 0
        while uncovered_count > 0 and refinement_iteration < MAX_REFINEMENT_ITERATIONS:
            refinement_iteration += 1
            logger.info(f"覆盖优化迭代 #{refinement_iteration}: {uncovered_count:,} 像素未覆盖。尝试改进...")
            
            patches_for_gaps = self._get_patches_for_uncovered_areas(current_best_N_patches)
            if not patches_for_gaps:
                logger.warning("无法为剩余未覆盖区域生成新补丁。优化迭代停止。")
                break
            
            # Combine current best N with new gap-filling patches.
            # The pool for re-selection should ensure all original '1's are potentially coverable.
            # Using candidate_patches_for_selection (e.g. the 36 patches) + patches_for_gaps
            # ensures we re-evaluate from a state that was known to cover everything, plus new detail.
            # However, if patches_for_gaps are already well represented by candidate_patches_for_selection,
            # this might not add much new.
            # A more direct pool: current N patches + patches for their gaps.
            pool_for_reselection = current_best_N_patches + patches_for_gaps
            logger.info(f"迭代 #{refinement_iteration}: 新候选池大小: {len(pool_for_reselection)}")
            
            merged_pool = self.merge_nearby_patches(pool_for_reselection)
            shrunk_merged_pool = [self._shrink_patch_to_content(p) for p in merged_pool]
            shrunk_merged_pool = [p for p in shrunk_merged_pool if p.area > 0]
            logger.info(f"迭代 #{refinement_iteration}: 合并和收缩后候选池大小: {len(shrunk_merged_pool)}")

            if not shrunk_merged_pool:
                logger.warning("迭代 #{refinement_iteration}: 合并后的候选池为空。优化迭代停止。")
                break
            
            current_best_N_patches = self.optimize_patches_and_truncate(shrunk_merged_pool)
            uncovered_count = self._calculate_uncovered_count(current_best_N_patches)

            if uncovered_count == 0:
                logger.info(f"迭代 #{refinement_iteration}: 已用 {len(current_best_N_patches)} 个补丁实现完全覆盖！")
                break
        else: # After while loop
            if uncovered_count > 0:
                logger.warning(f"在 {refinement_iteration} 次迭代优化后，仍有 {uncovered_count:,} 像素未被选中的 {len(current_best_N_patches)} 个补丁覆盖。")
        
        final_patches = current_best_N_patches
        if not final_patches: logger.error("最终未能生成任何patches。"); return []
        
        # Final statistics calculation
        total_patch_area = sum(p.area for p in final_patches)
        total_mask_area = np.sum(self.mask == 1)
        average_coverage = 0.0
        if final_patches: 
            coverages = [p.get_mask_coverage(self.mask) for p in final_patches if p.area > 0]
            if coverages: average_coverage = np.mean(coverages)
        
        logger.info(f"最终结果:")
        logger.info(f"  Patches数量: {len(final_patches)}")
        logger.info(f"  平均覆盖率: {average_coverage:.3f}")
        logger.info(f"  Patches总面积: {total_patch_area:,}")
        logger.info(f"  有效mask面积: {total_mask_area:,}")
        
        final_uncovered_count = self._calculate_uncovered_count(final_patches) # Use the helper
        if final_uncovered_count > 0:
            logger.error(f"警告: 最终仍有 {final_uncovered_count:,} 个有效像素未被覆盖!")
            logger.error(f"这可能发生在迭代优化后（如果 {self.max_patches} 个补丁不足以在约束下覆盖所有区域）。")
        else:
            logger.info("确认所有有效像素均被最终patches覆盖。")
        return final_patches

def visualize_patches(mask_array: np.ndarray, patch_list: List[Patch], save_path: str = "mask_patches_visualization.png"):
    """可视化mask和patches"""
    logger.info("创建可视化...")
    
    if not patch_list:
        logger.warning("Patch list is empty. Skipping visualization.")
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
        ax1.imshow(mask_array, cmap='GnBu', alpha=0.8)
        ax1.set_title(f'Original Mask\n(Total valid pixels: {np.sum(mask_array == 1):,})\nNo Patches Generated',
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"可视化（仅原图）保存至: {save_path}")
        plt.close(fig) 
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.imshow(mask_array, cmap='GnBu', alpha=0.8)
    ax1.set_title(f'Original Mask\n(Total valid pixels: {np.sum(mask_array == 1):,})',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')

    ax2.imshow(mask_array, cmap='GnBu', alpha=0.8)

    num_patches = len(patch_list)
    if num_patches == 0 : 
        colors = []
    elif num_patches <= 12: 
        colors = plt.cm.Set3(np.linspace(0, 1, num_patches))
    elif num_patches <= 20: 
        colors = plt.cm.tab20(np.linspace(0, 1, num_patches))
    else: 
        colors = plt.cm.viridis(np.linspace(0, 1, num_patches))

    for i, (patch, color) in enumerate(zip(patch_list, colors)):
        if patch.area == 0: continue 
        rect = mpatches.Rectangle(
            (patch.minx, patch.miny), patch.width, patch.height,
            linewidth=2, edgecolor=color, facecolor='none', alpha=0.8
        )
        ax2.add_patch(rect)
        center_x, center_y = patch.minx + patch.width // 2, patch.miny + patch.height // 2
        coverage = patch.get_mask_coverage(mask_array)
        ax2.text(center_x, center_y, f'P{i+1}\n{coverage:.2f}',
                 ha='center', va='center', fontsize=8, fontweight='bold', 
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)) 

    avg_coverage_val = 0.0
    if patch_list: 
        patch_coverages = [p.get_mask_coverage(mask_array) for p in patch_list if p.area > 0]
        if patch_coverages: avg_coverage_val = np.mean(patch_coverages)

    ax2.set_title(f'Mask with {len(patch_list)} Patches\nAvg Coverage: {avg_coverage_val:.3f}',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (pixels)'); ax2.set_ylabel('Y (pixels)')

    if num_patches > 0 and num_patches <= 20 : 
        legend_elements = [mpatches.Patch(facecolor=color, edgecolor=color, alpha=0.7,
                                        label=f'P{i+1} (Area: {patch.area:,})')
                        for i, (patch, color) in enumerate(zip(patch_list, colors)) if patch.area > 0]
        if legend_elements: 
            ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small')

    plt.tight_layout(rect=[0, 0, 0.9 if num_patches > 0 and num_patches <=20 else 1, 1]) 
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"可视化保存至: {save_path}")
    plt.close(fig) 

def save_patches_as_tiffs(mask_array: np.ndarray, patch_list: List[Patch], original_tiff_path: str):
    """将每个patch保存为单独的TIFF文件，保持原始坐标系和分辨率(10m)，值为0/1"""
    if not patch_list:
        logger.warning("Patch list is empty. No TIFF patches to save.")
        return
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(original_tiff_path), "patches")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"创建补丁保存目录: {output_dir}")
    
    # 获取原始TIFF的元数据
    with rasterio.open(original_tiff_path) as src:
        original_transform = src.transform
        original_crs = src.crs
        original_nodata = src.nodata
        original_dtype = src.dtypes[0]
        original_data = src.read(1)  # 读取原始数据，用于确保值的一致性
        
        # 确认分辨率
        x_res = abs(original_transform.a)
        y_res = abs(original_transform.e)
        logger.info(f"原始分辨率: X={x_res}m, Y={y_res}m")
        logger.info(f"原始坐标系: {original_crs}")
        
        for i, patch in enumerate(patch_list):
            if patch.area == 0:
                logger.warning(f"Patch {i+1} 面积为0，跳过保存。")
                continue
            
            # 提取patch对应的区域边界
            eff_miny, eff_maxy = max(0, patch.miny), min(mask_array.shape[0], patch.maxy)
            eff_minx, eff_maxx = max(0, patch.minx), min(mask_array.shape[1], patch.maxx)
            
            if eff_miny >= eff_maxy or eff_minx >= eff_maxx:
                logger.warning(f"Patch {i+1} 边界无效，跳过保存。")
                continue
            
            # 创建与原始TIFF相同值的patch
            # 注意：这里直接从原始数据中提取，确保0和1的值与原始一致
            patch_data = np.zeros((eff_maxy - eff_miny, eff_maxx - eff_minx), dtype=original_dtype)
            
            try:
                # 从原始数据中提取相应区域
                patch_region = original_data[eff_miny:eff_maxy, eff_minx:eff_maxx]
                if patch_region.shape == patch_data.shape:
                    patch_data = patch_region
                else:
                    logger.warning(f"Patch {i+1} 形状不匹配，使用掩膜数据。")
                    patch_data = mask_array[eff_miny:eff_maxy, eff_minx:eff_maxx]
            except Exception as e:
                logger.warning(f"Patch {i+1} 从原始数据提取失败 ({str(e)})，使用掩膜数据。")
                patch_data = mask_array[eff_miny:eff_maxy, eff_minx:eff_maxx]
            
            # 确保数据是二值的（0和1）
            patch_data = (patch_data > 0).astype(original_dtype)
            
            # 计算patch的新transform，保持原始分辨率
            new_transform = Affine(
                original_transform.a,  # 保持x方向分辨率
                original_transform.b,
                original_transform.c + original_transform.a * eff_minx,  # 调整原点x坐标
                original_transform.d,
                original_transform.e,  # 保持y方向分辨率
                original_transform.f + original_transform.e * eff_miny   # 调整原点y坐标
            )
            
            # 设置输出文件路径
            output_path = os.path.join(output_dir, f"patch{i+1}.tif")
            
            # 保存patch为TIFF文件
            patch_profile = {
                'driver': 'GTiff',
                'height': patch_data.shape[0],
                'width': patch_data.shape[1],
                'count': 1,
                'dtype': original_dtype,
                'crs': original_crs,
                'transform': new_transform,
                'compress': 'lzw',
                'nodata': original_nodata if original_nodata is not None else 0
            }
            
            with rasterio.open(output_path, 'w', **patch_profile) as dst:
                dst.write(patch_data, 1)
            
            coverage = patch.get_mask_coverage(mask_array)
            x_size_m = patch_data.shape[1] * x_res
            y_size_m = patch_data.shape[0] * y_res
            logger.info(f"已保存 Patch {i+1} (覆盖率: {coverage:.3f}, 尺寸: {patch_data.shape[1]}x{patch_data.shape[0]}像素, "
                        f"{x_size_m:.1f}mx{y_size_m:.1f}m) 到: {output_path}")
    
    logger.info(f"所有patch TIFF文件已保存至: {output_dir}")

def main():
    logger.info("=" * 60); logger.info("生物质掩膜分块器"); logger.info("=" * 60)
    try:
        logger.info(f"读取mask文件: {MASK_PATH}")
        with rasterio.open(MASK_PATH) as src:
            mask_data = src.read(1)
            mask_data = (mask_data > 0).astype(np.uint8) 
            logger.info(f"Mask尺寸: {mask_data.shape}")
            logger.info(f"有效像素数量 (值为1): {np.sum(mask_data == 1):,}")
            logger.info(f"总像素数量: {mask_data.size:,}")
            if mask_data.size > 0: logger.info(f"有效像素比例: {np.sum(mask_data == 1) / mask_data.size * 100:.2f}%")
            else: logger.error("Mask 文件为空或读取失败。"); return

        patcher = MaskPatcher(mask_data, MAX_PATCHES, MAX_PATCH_RATIO)
        patches = patcher.generate_patches()

        if not patches:
            logger.error("未能生成任何patches。可视化将只显示原图。")
            visualize_patches(mask_data, []) 
            return
        
        # 可视化patches
        vis_path = os.path.join(os.path.dirname(MASK_PATH), "mask_patches_visualization.png")
        visualize_patches(mask_data, patches, vis_path)
        
        # 保存patches为TIFF文件
        save_patches_as_tiffs(mask_data, patches, MASK_PATH)
        
        logger.info("\n详细统计信息:"); logger.info("="*40)
        for i, patch in enumerate(patches):
            coverage = patch.get_mask_coverage(mask_data)
            area_percentage = 0.0
            if mask_data.size > 0: area_percentage = patch.area / mask_data.size * 100
            logger.info(f"Patch {i+1}: {patch}")
            logger.info(f"  覆盖率: {coverage:.3f}")
            logger.info(f"  面积占比: {area_percentage:.2f}%")
            if patch.area > patcher.max_patch_area:
                logger.error(f"  警告: Patch {i+1} 面积 {patch.area} 超过最大允许面积 {patcher.max_patch_area}!")
        logger.info("=" * 60); logger.info("分块完成！"); logger.info("=" * 60)
    except FileNotFoundError: logger.error(f"错误: Mask文件未找到于路径 {MASK_PATH}"); sys.exit(1)
    except rasterio.errors.RasterioIOError as e: logger.error(f"错误: 读取mask文件时发生Rasterio错误: {e}"); sys.exit(1)
    except Exception as e: logger.error(f"程序执行失败: {e}"); logger.exception("详细错误信息:"); sys.exit(1)

if __name__ == "__main__":
    main()