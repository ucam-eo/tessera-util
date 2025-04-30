use std::fs;
use std::path::Path;

use clap::Parser;
use env_logger;
use indicatif::{ProgressBar, ProgressStyle};
use log::{info, warn, error, debug};
use rayon::prelude::*;
use rand::Rng;
use rand::seq::SliceRandom;
use anyhow::{Result, Context};

use ndarray::{Array3, Array4, Array5, Array1, Axis, s};
use ndarray_npy::{read_npy, write_npy};

#[derive(Parser, Debug)]
#[command(name = "training-data-preprocessing-patch-doy", version = "0.1.0")]
struct Args {
    /// Path to root directory containing tile subfolders
    #[arg(long, default_value = "/mnt/e/Codes/btfm/data/global")]
    data_root: String,

    /// Minimal valid timesteps for S2 (patch-scale)
    #[arg(long, default_value = "20")]
    s2_min_valid_timesteps: usize,

    /// Minimal valid timesteps for S1 (patch-scale)
    #[arg(long, default_value = "20")]
    s1_min_valid_timesteps: usize,

    /// Number of timesteps to sample (time_steps)
    #[arg(long, default_value = "20")]
    time_steps: usize,

    /// How many tiles to process in one batch (control memory usage)
    #[arg(long, default_value = "100")]
    tile_batch: usize,

    /// Output directory for augmented data
    #[arg(long, default_value = "./aug_patch_output")]
    output_dir: String,

    /// Save chunk size, i.e. how many patches in one .npy file
    #[arg(long, default_value = "1000")]
    chunk_size: usize,

    /// Patch size (must be odd)
    #[arg(long, default_value = "5")]
    patch_size: usize,

    /// Valid ratio threshold for S2 (e.g. 0.8 => 80%)
    #[arg(long, default_value = "0.8")]
    s2_valid_ratio: f32,

    /// Valid ratio threshold for S1 (e.g. 0.8 => 80%)
    #[arg(long, default_value = "0.8")]
    s1_valid_ratio: f32,

    /// Maximum number of patches to generate per (1000x1000) sub-tile
    #[arg(long, default_value = "10000")]
    max_patch_per_1k_1k_tile: usize,

    /// Stride between consecutive pixel samples in one patch dimension.
    /// e.g., stride=0 => consecutive pixels, stride=1 =>跳过1个像素才采一个 =>更大覆盖范围
    #[arg(long, default_value = "0")]
    stride: usize,
}

// S2 band mean / std
static S2_BAND_MEAN: [f32; 10] = [
    1711.0938, 1308.8511, 1546.4543, 3010.1293, 3106.5083,
    2068.3044, 2685.0845, 2931.5889, 2514.6928, 1899.4922
];
static S2_BAND_STD: [f32; 10] = [
    1926.1026, 1862.9751, 1803.1792, 1741.7837, 1677.4543,
    1888.7862, 1736.3090, 1715.8104, 1514.5199, 1398.4779
];
// S1 band mean / std
static S1_BAND_MEAN: [f32; 2] = [5484.0407, 3003.7812];
static S1_BAND_STD: [f32; 2] = [1871.2334, 1726.0670];

/// 每个 patch 返回两份增广 (s2_a1, s2_a2, s1_a1, s1_a2)；
/// 注意这里用的是 `Array4<f32>`，shape=(patch_size, patch_size, time_steps, band_num)
/// 最终会在 stack 时再拼成 5D
struct PatchOut {
    s2_a1: Array4<f32>,
    s2_a2: Array4<f32>,
    s1_a1: Array4<f32>,
    s1_a2: Array4<f32>,
}

fn main() -> Result<()> {
    // 初始化日志
    env_logger::Builder::from_default_env()
        .format_timestamp_secs()
        .init();
    info!("=== training-data-preprocessing-patch-doy (with stride) start ===");

    let args = Args::parse();
    info!("Parsed command-line args: {:?}", args);

    let data_root = Path::new(&args.data_root);
    let out_dir = Path::new(&args.output_dir);

    // **自动创建输出目录及其子目录**
    info!("Creating output directories under {:?}", out_dir);
    fs::create_dir_all(&out_dir)
        .with_context(|| format!("Failed to create output directory {:?}", out_dir))?;
    fs::create_dir_all(out_dir.join("aug1/s2"))?;
    fs::create_dir_all(out_dir.join("aug1/s1"))?;
    fs::create_dir_all(out_dir.join("aug2/s2"))?;
    fs::create_dir_all(out_dir.join("aug2/s1"))?;

    // 1) 查找 tile 文件夹
    info!("Scanning tiles under data_root={:?}", data_root);
    let mut tile_paths = Vec::new();
    let rd = match fs::read_dir(data_root) {
        Ok(r) => r,
        Err(e) => {
            error!("Cannot read data_root={:?}, err={}", data_root, e);
            return Ok(());
        }
    };

    for entry_r in rd {
        let entry = match entry_r {
            Ok(en) => en,
            Err(e) => {
                warn!("Skipping an entry due to read_dir error: {:?}", e);
                continue;
            }
        };
        let tile_path = entry.path();
        if !entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
            debug!("Skipping non-dir file={:?}", tile_path);
            continue;
        }
        // 检查所需文件是否存在
        let needed = [
            "bands.npy", "masks.npy", "doys.npy",
            "sar_ascending.npy", "sar_ascending_doy.npy",
            "sar_descending.npy", "sar_descending_doy.npy"
        ];
        let all_ok = needed.iter().all(|f| tile_path.join(f).exists());
        if all_ok {
            tile_paths.push(tile_path);
        } else {
            debug!("Skipping dir={:?}, not all needed files exist", tile_path);
        }
    }

    // 排序后进行全局打乱
    tile_paths.sort();
    tile_paths.shuffle(&mut rand::thread_rng());

    let total_tiles = tile_paths.len();
    info!("Found {} tile subfolders in {:?}", total_tiles, data_root);
    if total_tiles == 0 {
        error!("No valid tile subfolders found => nothing to do!");
        return Ok(());
    }

    // 进度条
    let pb = ProgressBar::new(total_tiles as u64);
    pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>3}/{len:3} {msg}")
            .unwrap()
            .progress_chars("##-")
    );

    let mut start_tile_idx = 0;
    let mut batch_index = 0;

    while start_tile_idx < total_tiles {
        let end_tile_idx = (start_tile_idx + args.tile_batch).min(total_tiles);
        let this_batch = &tile_paths[start_tile_idx..end_tile_idx];
        batch_index += 1;

        info!("==== [Batch #{}] => tile indices in range [{}, {}) ====", batch_index, start_tile_idx, end_tile_idx);

        // 并行处理每个 tile
        let results: Vec<_> = this_batch.par_iter()
            .map(|tile_dir| process_tile(tile_dir, &args))
            .collect();

        let mut s2a1_vec = Vec::new();
        let mut s2a2_vec = Vec::new();
        let mut s1a1_vec = Vec::new();
        let mut s1a2_vec = Vec::new();

        let mut tile_counter = 0;
        for r in results {
            pb.inc(1);
            tile_counter += 1;
            match r {
                Ok(patch_out_vec) => {
                    let npatch = patch_out_vec.len();
                    info!("Tile #{} => got {} valid patches => accumulate..", tile_counter, npatch);
                    if npatch == 0 {
                        continue;
                    }
                    for po in patch_out_vec {
                        s2a1_vec.push(po.s2_a1);
                        s2a2_vec.push(po.s2_a2);
                        s1a1_vec.push(po.s1_a1);
                        s1a2_vec.push(po.s1_a2);
                    }
                }
                Err(e) => {
                    warn!("Tile error: {:?}", e);
                }
            }
        }

        let total_patches_in_batch = s2a1_vec.len();
        if total_patches_in_batch == 0 {
            info!("[Batch #{}] => No valid patch => skip this batch", batch_index);
            start_tile_idx = end_tile_idx;
            continue;
        }

        // 合并数据：从 Vec<Array4<f32>> 拼成 Array5<f32>
        let s2a1_all = stack_4d(&s2a1_vec)?; // shape => (N, patch_size, patch_size, time_steps, 11)
        let s2a2_all = stack_4d(&s2a2_vec)?;
        let s1a1_all = stack_4d(&s1a1_vec)?; // shape => (N, patch_size, patch_size, time_steps, 3)
        let s1a2_all = stack_4d(&s1a2_vec)?;

        info!("[Batch #{}] => done merging => total valid patches={}, shape s2_a1=({}, {}, {}, {}, {})",
              batch_index,
              total_patches_in_batch,
              s2a1_all.len_of(Axis(0)), // N
              s2a1_all.len_of(Axis(1)), // patch_size
              s2a1_all.len_of(Axis(2)), // patch_size
              s2a1_all.len_of(Axis(3)), // T
              s2a1_all.len_of(Axis(4))); // 11

        // 全局 Shuffle：先生成排列索引，再用并行复制
        let n_patches = s2a1_all.len_of(Axis(0));
        let mut perm: Vec<usize> = (0..n_patches).collect();
        perm.shuffle(&mut rand::thread_rng());

        let s2a1_all = parallel_permute_5d(&s2a1_all, &perm);
        let s2a2_all = parallel_permute_5d(&s2a2_all, &perm);
        let s1a1_all = parallel_permute_5d(&s1a1_all, &perm);
        let s1a2_all = parallel_permute_5d(&s1a2_all, &perm);

        info!("[Batch #{}] => shuffled all patches", batch_index);

        // 只写入完整的 chunk 文件，舍弃最后不足 chunk_size 的边角料
        let n_chunks = n_patches / args.chunk_size;
        info!("[Batch #{}] => starting parallel write of {} chunks", batch_index, n_chunks);
        (0..n_chunks).into_par_iter().for_each(|chunk_idx| {
            let offset = chunk_idx * args.chunk_size;
            let end_offset = offset + args.chunk_size;

            let s2a1_slice = s2a1_all.slice(s![offset..end_offset, .., .., .., ..]).to_owned();
            let s2a2_slice = s2a2_all.slice(s![offset..end_offset, .., .., .., ..]).to_owned();
            let s1a1_slice = s1a1_all.slice(s![offset..end_offset, .., .., .., ..]).to_owned();
            let s1a2_slice = s1a2_all.slice(s![offset..end_offset, .., .., .., ..]).to_owned();

            let file_counter = chunk_idx + 1;
            let s2a1_path = out_dir.join(format!("aug1/s2/data_B{}_F{}.npy", batch_index, file_counter));
            let s2a2_path = out_dir.join(format!("aug2/s2/data_B{}_F{}.npy", batch_index, file_counter));
            let s1a1_path = out_dir.join(format!("aug1/s1/data_B{}_F{}.npy", batch_index, file_counter));
            let s1a2_path = out_dir.join(format!("aug2/s1/data_B{}_F{}.npy", batch_index, file_counter));

            if let Err(e) = write_npy(&s2a1_path, &s2a1_slice) {
                error!("Failed to write file {:?}: {:?}", s2a1_path, e);
            }
            if let Err(e) = write_npy(&s2a2_path, &s2a2_slice) {
                error!("Failed to write file {:?}: {:?}", s2a2_path, e);
            }
            if let Err(e) = write_npy(&s1a1_path, &s1a1_slice) {
                error!("Failed to write file {:?}: {:?}", s1a1_path, e);
            }
            if let Err(e) = write_npy(&s1a2_path, &s1a2_slice) {
                error!("Failed to write file {:?}: {:?}", s1a2_path, e);
            }
            info!("[Batch #{}] => wrote chunk file #{} with {} patches", batch_index, file_counter, args.chunk_size);
        });

        start_tile_idx = end_tile_idx;
    }
    pb.finish_with_message("All done.");
    info!("=== Done. Check your output_dir for files. ===");

    Ok(())
}

/// 处理单个 tile：读取数据、分割成多个 1000×1000 小块（不足 1000×1000 则跳过），并在每个小块里随机选取 patch
fn process_tile(tile_dir: &Path, args: &Args) -> Result<Vec<PatchOut>> {
    let tile_name = tile_dir.file_name()
        .map(|os| os.to_string_lossy().to_string())
        .unwrap_or_else(|| "UnknownTile".to_string());
    info!("process_tile() start => tile={}", tile_name);

    // 读取 Sentinel-2 数据（bands 为 u16）
    let s2_bands: Array4<u16> = read_npy(tile_dir.join("bands.npy"))
        .with_context(|| format!("reading s2 bands in tile={}", tile_name))?;
    // masks.npy 存储的是 u8
    let s2_masks: Array3<u8> = read_npy(tile_dir.join("masks.npy"))
        .with_context(|| format!("reading s2 masks in tile={}", tile_name))?;
    let s2_doys: Array1<u16> = read_npy(tile_dir.join("doys.npy"))
        .with_context(|| format!("reading s2 doys in tile={}", tile_name))?;

    // 读取 SAR 数据
    let s1_asc_bands: Array4<i16> = read_npy(tile_dir.join("sar_ascending.npy"))
        .with_context(|| format!("reading s1 asc bands in tile={}", tile_name))?;
    let s1_asc_doys: Array1<i32> = read_npy(tile_dir.join("sar_ascending_doy.npy"))
        .with_context(|| format!("reading s1 asc doys in tile={}", tile_name))?;
    let s1_desc_bands: Array4<i16> = read_npy(tile_dir.join("sar_descending.npy"))
        .with_context(|| format!("reading s1 desc bands in tile={}", tile_name))?;
    let s1_desc_doys: Array1<i32> = read_npy(tile_dir.join("sar_descending_doy.npy"))
        .with_context(|| format!("reading s1 desc doys in tile={}", tile_name))?;

    let shape_s2 = s2_bands.shape(); // [t_s2, H, W, 10]
    let t_s2 = shape_s2[0];
    let h = shape_s2[1];
    let w = shape_s2[2];
    info!("Tile = {} => s2_bands shape = ({},{},{},{})", tile_name, t_s2, h, w, shape_s2[3]);

    let patch_size = args.patch_size;
    // coverage_radius用来避免越界；stride越大，覆盖越大 => coverage_radius也越大
    // coverage_radius = (patch_size//2) * (stride+1)
    let coverage_radius = (patch_size / 2) * (args.stride + 1);

    // 将大 tile 拆分成若干 (1000×1000) 子块
    let sub_tile_size = 1000;
    let n_subtiles_y = (h + sub_tile_size - 1) / sub_tile_size; // 向上取整
    let n_subtiles_x = (w + sub_tile_size - 1) / sub_tile_size;

    let mut rng = rand::thread_rng();
    let mut results = Vec::new();

    info!("Tile {} => subdividing into {} x {} subtiles of size ~1000x1000 (except edges).", 
          tile_name, n_subtiles_y, n_subtiles_x);

    for sy in 0..n_subtiles_y {
        for sx in 0..n_subtiles_x {
            let y0 = sy * sub_tile_size;
            let x0 = sx * sub_tile_size;
            let y1 = (y0 + sub_tile_size).min(h);
            let x1 = (x0 + sub_tile_size).min(w);

            let block_h = y1 - y0;
            let block_w = x1 - x0;

            // 如果这个子块小于 1000×1000，则直接跳过
            if block_h < sub_tile_size || block_w < sub_tile_size {
                info!("Tile {} => skip sub-tile at (y0={}, x0={}) because it's {}x{} (< 1000x1000).",
                      tile_name, y0, x0, block_h, block_w);
                continue;
            }

            // 如果子块再加上 stride 覆盖也放不下
            // (center ± coverage_radius)都要在子块内
            if block_h < 2 * coverage_radius + 1 || block_w < 2 * coverage_radius + 1 {
                info!("Tile {} => skip sub-tile at (y0={}, x0={}) because block < coverage radius({}).",
                      tile_name, y0, x0, coverage_radius);
                continue;
            }

            info!("Tile {} => process sub-tile at (y0={}, x0={}), shape {}x{}", 
                  tile_name, y0, x0, block_h, block_w);

            // 准备在该子块内最多生成多少个 patch
            let max_needed = args.max_patch_per_1k_1k_tile;
            // 为了避免无限循环，这里设置一个抽样上限
            let max_tries = max_needed * 5;

            let mut sub_tile_patches = 0usize;
            for _ in 0..max_tries {
                if sub_tile_patches >= max_needed {
                    break;
                }
                // 随机选一个中心 (必须保证 coverage 不越界)
                let center_y = rng.gen_range(y0 + coverage_radius .. y1 - coverage_radius);
                let center_x = rng.gen_range(x0 + coverage_radius .. x1 - coverage_radius);

                // 收集 S2 有效时间步 (只针对采样到的点)
                let s2_valid_times = collect_s2_valid_times(
                    &s2_masks,
                    center_y,
                    center_x,
                    patch_size,
                    args.s2_valid_ratio,
                    args.stride
                );
                // 收集 S1 有效时间步 (只针对采样到的点)
                let s1_valid_times = collect_s1_valid_times(
                    &s1_asc_bands,
                    &s1_asc_doys,
                    &s1_desc_bands,
                    &s1_desc_doys,
                    center_y,
                    center_x,
                    patch_size,
                    args.s1_valid_ratio,
                    args.stride
                );

                // 如果有效帧数不足 => 跳过
                if s2_valid_times.len() < args.s2_min_valid_timesteps {
                    continue;
                }
                if s1_valid_times.len() < args.s1_min_valid_timesteps {
                    continue;
                }

                // 生成增广结果
                let s2_a1 = sample_s2_patch(
                    &s2_bands,
                    &s2_doys,
                    center_y,
                    center_x,
                    patch_size,
                    &s2_valid_times,
                    args.time_steps,
                    args.stride
                );
                let s2_a2 = sample_s2_patch(
                    &s2_bands,
                    &s2_doys,
                    center_y,
                    center_x,
                    patch_size,
                    &s2_valid_times,
                    args.time_steps,
                    args.stride
                );
                let (s1_a1, s1_a2) = sample_s1_patch(
                    &s1_asc_bands, &s1_asc_doys,
                    &s1_desc_bands, &s1_desc_doys,
                    center_y, center_x,
                    patch_size,
                    &s1_valid_times,
                    args.time_steps,
                    args.stride
                );
                // 如果都生成了，则加入结果
                if let (Some(s2_a1), Some(s2_a2), Some(s1_a1), Some(s1_a2)) = (s2_a1, s2_a2, s1_a1, s1_a2) {
                    results.push(PatchOut {
                        s2_a1, s2_a2, s1_a1, s1_a2
                    });
                    sub_tile_patches += 1;
                }
            }

            info!("Tile {} => sub-tile (y0={}, x0={}) => generated {} patches.", 
                  tile_name, y0, x0, sub_tile_patches);
        }
    }

    info!("Tile {} => total patches generated: {}", tile_name, results.len());
    Ok(results)
}

/// 根据 patch_size×patch_size 实际采样到的像素(考虑 stride)中，统计 mask=1 的个数，
/// 若 >= ratio×(patch_size^2)，则此时间步计为 valid。
fn collect_s2_valid_times(
    s2_masks: &Array3<u8>,
    center_y: usize,
    center_x: usize,
    patch_size: usize,
    ratio: f32,
    stride: usize
) -> Vec<usize> {
    let t_s2 = s2_masks.len_of(Axis(0));
    let half_idx = patch_size / 2;
    let area_f32 = patch_size as f32 * patch_size as f32;
    let threshold = ratio * area_f32;

    let mut valid_times = Vec::new();
    for t in 0..t_s2 {
        let mut count_valid = 0usize;
        // 按 stride 跳步
        for i in 0..patch_size {
            let row_offset = (i as isize - half_idx as isize) * (stride as isize + 1);
            for j in 0..patch_size {
                let col_offset = (j as isize - half_idx as isize) * (stride as isize + 1);
                let yy = (center_y as isize + row_offset) as usize;
                let xx = (center_x as isize + col_offset) as usize;
                if s2_masks[(t, yy, xx)] == 1 {
                    count_valid += 1;
                }
            }
        }
        if (count_valid as f32) >= threshold {
            valid_times.push(t);
        }
    }
    valid_times
}

/// 收集给定 patch 中(考虑 stride)在各个时间步上的有效性 (S1: asc + desc)。
/// 若采样到的像素里 (VV!=0 or VH!=0) 的个数占比 >= ratio 则视为 valid
fn collect_s1_valid_times(
    asc_bands: &Array4<i16>,
    asc_doys: &Array1<i32>,
    desc_bands: &Array4<i16>,
    desc_doys: &Array1<i32>,
    center_y: usize,
    center_x: usize,
    patch_size: usize,
    ratio: f32,
    stride: usize
) -> Vec<(bool, usize)> {
    let mut merged = Vec::new();
    let half_idx = patch_size / 2;
    let area_f32 = patch_size as f32 * patch_size as f32;
    let threshold = ratio * area_f32;

    // asc
    let t_asc = asc_bands.len_of(Axis(0));
    for t in 0..t_asc {
        let mut count_valid = 0usize;
        for i in 0..patch_size {
            let row_offset = (i as isize - half_idx as isize) * (stride as isize + 1);
            for j in 0..patch_size {
                let col_offset = (j as isize - half_idx as isize) * (stride as isize + 1);
                let yy = (center_y as isize + row_offset) as usize;
                let xx = (center_x as isize + col_offset) as usize;
                let vv = asc_bands[(t, yy, xx, 0)];
                let vh = asc_bands[(t, yy, xx, 1)];
                if vv != 0 || vh != 0 {
                    count_valid += 1;
                }
            }
        }
        if (count_valid as f32) >= threshold {
            merged.push((true, t));
        }
    }

    // desc
    let t_desc = desc_bands.len_of(Axis(0));
    for t in 0..t_desc {
        let mut count_valid = 0usize;
        for i in 0..patch_size {
            let row_offset = (i as isize - half_idx as isize) * (stride as isize + 1);
            for j in 0..patch_size {
                let col_offset = (j as isize - half_idx as isize) * (stride as isize + 1);
                let yy = (center_y as isize + row_offset) as usize;
                let xx = (center_x as isize + col_offset) as usize;
                let vv = desc_bands[(t, yy, xx, 0)];
                let vh = desc_bands[(t, yy, xx, 1)];
                if vv != 0 || vh != 0 {
                    count_valid += 1;
                }
            }
        }
        if (count_valid as f32) >= threshold {
            merged.push((false, t));
        }
    }
    merged
}

/// 在已知 patch 有效时间步里随机采样 time_steps 帧，S2 shape=(patch_size, patch_size, time_steps, 11)
/// 同样按 stride 跳读像素
fn sample_s2_patch(
    s2_bands: &Array4<u16>,
    s2_doys: &Array1<u16>,
    center_y: usize,
    center_x: usize,
    patch_size: usize,
    valid_idx: &Vec<usize>,
    time_steps: usize,
    stride: usize
) -> Option<Array4<f32>> {
    if valid_idx.is_empty() {
        return None;
    }
    let mut rng = rand::thread_rng();
    let half_idx = patch_size / 2;

    let mut out = Array4::<f32>::zeros((patch_size, patch_size, time_steps, 11));

    for t_out in 0..time_steps {
        let pick = valid_idx[rng.gen_range(0..valid_idx.len())];
        let doy_f32 = s2_doys[pick] as f32;
        // 按 stride 跳取
        for i in 0..patch_size {
            let row_offset = (i as isize - half_idx as isize) * (stride as isize + 1);
            for j in 0..patch_size {
                let col_offset = (j as isize - half_idx as isize) * (stride as isize + 1);
                let yy = (center_y as isize + row_offset) as usize;
                let xx = (center_x as isize + col_offset) as usize;
                // 10个 band
                for b in 0..10 {
                    let rawv = s2_bands[(pick, yy, xx, b)] as f32;
                    let normed = (rawv - S2_BAND_MEAN[b]) / (S2_BAND_STD[b] + 1e-9);
                    out[(i, j, t_out, b)] = normed;
                }
                // 第11个存 doy
                out[(i, j, t_out, 10)] = doy_f32;
            }
        }
    }
    Some(out)
}

/// 在已知 patch 有效时间步里随机采样 time_steps 帧，S1 shape=(patch_size, patch_size, time_steps, 3)
/// 同样按 stride 跳读像素
fn sample_s1_patch(
    asc_bands: &Array4<i16>,
    asc_doys: &Array1<i32>,
    desc_bands: &Array4<i16>,
    desc_doys: &Array1<i32>,
    center_y: usize,
    center_x: usize,
    patch_size: usize,
    valid_times: &Vec<(bool, usize)>,
    time_steps: usize,
    stride: usize
) -> (Option<Array4<f32>>, Option<Array4<f32>>) {
    if valid_times.is_empty() {
        return (None, None);
    }
    let mut rng = rand::thread_rng();
    let half_idx = patch_size / 2;

    let mut arr1 = Array4::<f32>::zeros((patch_size, patch_size, time_steps, 3));
    let mut arr2 = Array4::<f32>::zeros((patch_size, patch_size, time_steps, 3));

    // 填充 arr1
    for t_out in 0..time_steps {
        let (is_asc, idx) = valid_times[rng.gen_range(0..valid_times.len())];
        let doy_val = if is_asc {
            asc_doys[idx] as f32
        } else {
            desc_doys[idx] as f32
        };
        for i in 0..patch_size {
            let row_offset = (i as isize - half_idx as isize) * (stride as isize + 1);
            for j in 0..patch_size {
                let col_offset = (j as isize - half_idx as isize) * (stride as isize + 1);
                let yy = (center_y as isize + row_offset) as usize;
                let xx = (center_x as isize + col_offset) as usize;
                let (raw_vv, raw_vh) = if is_asc {
                    (
                        asc_bands[(idx, yy, xx, 0)] as f32,
                        asc_bands[(idx, yy, xx, 1)] as f32
                    )
                } else {
                    (
                        desc_bands[(idx, yy, xx, 0)] as f32,
                        desc_bands[(idx, yy, xx, 1)] as f32
                    )
                };
                let vv_norm = (raw_vv - S1_BAND_MEAN[0]) / (S1_BAND_STD[0] + 1e-9);
                let vh_norm = (raw_vh - S1_BAND_MEAN[1]) / (S1_BAND_STD[1] + 1e-9);

                arr1[(i, j, t_out, 0)] = vv_norm;
                arr1[(i, j, t_out, 1)] = vh_norm;
                arr1[(i, j, t_out, 2)] = doy_val;
            }
        }
    }

    // 填充 arr2
    for t_out in 0..time_steps {
        let (is_asc, idx) = valid_times[rng.gen_range(0..valid_times.len())];
        let doy_val = if is_asc {
            asc_doys[idx] as f32
        } else {
            desc_doys[idx] as f32
        };
        for i in 0..patch_size {
            let row_offset = (i as isize - half_idx as isize) * (stride as isize + 1);
            for j in 0..patch_size {
                let col_offset = (j as isize - half_idx as isize) * (stride as isize + 1);
                let yy = (center_y as isize + row_offset) as usize;
                let xx = (center_x as isize + col_offset) as usize;
                let (raw_vv, raw_vh) = if is_asc {
                    (
                        asc_bands[(idx, yy, xx, 0)] as f32,
                        asc_bands[(idx, yy, xx, 1)] as f32
                    )
                } else {
                    (
                        desc_bands[(idx, yy, xx, 0)] as f32,
                        desc_bands[(idx, yy, xx, 1)] as f32
                    )
                };
                let vv_norm = (raw_vv - S1_BAND_MEAN[0]) / (S1_BAND_STD[0] + 1e-9);
                let vh_norm = (raw_vh - S1_BAND_MEAN[1]) / (S1_BAND_STD[1] + 1e-9);

                arr2[(i, j, t_out, 0)] = vv_norm;
                arr2[(i, j, t_out, 1)] = vh_norm;
                arr2[(i, j, t_out, 2)] = doy_val;
            }
        }
    }

    (Some(arr1), Some(arr2))
}

/// 将多个 shape=(p, p, t, c) 的 Array4<f32> 拼合成 (n, p, p, t, c) 的 Array5<f32>
fn stack_4d(v: &Vec<Array4<f32>>) -> Result<Array5<f32>> {
    if v.is_empty() {
        return Ok(Array5::<f32>::zeros((0, 0, 0, 0, 0)));
    }
    let n = v.len();
    let p_s = v[0].len_of(Axis(0));
    let p_s2 = v[0].len_of(Axis(1));
    let t_s = v[0].len_of(Axis(2));
    let c = v[0].len_of(Axis(3));

    // shape=(n, p_s, p_s2, t_s, c)
    let mut out = Array5::<f32>::zeros((n, p_s, p_s2, t_s, c));
    let row_len = p_s * p_s2 * t_s * c;

    // 并行复制
    let out_slice = out.as_slice_mut().expect("Array5 not contiguous");
    out_slice
        .par_chunks_exact_mut(row_len)
        .enumerate()
        .for_each(|(i, chunk)| {
            let src = v[i].as_slice().expect("v[i] not contiguous");
            chunk.copy_from_slice(src);
        });

    Ok(out)
}

/// 并行排列第0维 (n维) => 返回新的 5D
fn parallel_permute_5d(arr: &Array5<f32>, perm: &[usize]) -> Array5<f32> {
    let (n, p_s, p_s2, t_s, c) = arr.dim();
    let row_len = p_s * p_s2 * t_s * c;
    let mut new_arr = Array5::<f32>::zeros((n, p_s, p_s2, t_s, c));

    let arr_slice = arr.as_slice().expect("arr not contiguous");
    new_arr
        .as_slice_mut()
        .expect("new_arr not contiguous")
        .par_chunks_mut(row_len)
        .enumerate()
        .for_each(|(i, chunk)| {
            let src_start = perm[i] * row_len;
            let src_end = src_start + row_len;
            chunk.copy_from_slice(&arr_slice[src_start..src_end]);
        });
    new_arr
}
