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

use ndarray::{Array2, Array3, Array4, Array1, Axis, s};
use ndarray_npy::{read_npy, write_npy};

#[derive(Parser, Debug)]
#[command(name = "training-data-preprocessing-single-doy", version = "0.1.0")]
struct Args {
    /// Path to root directory containing tile subfolders
    #[arg(long, default_value = "/mnt/e/Codes/btfm/data/global")]
    data_root: String,

    /// Minimal valid timesteps for S2
    #[arg(long, default_value = "20")]
    s2_min_valid_timesteps: usize,

    /// Minimal valid timesteps for S1
    #[arg(long, default_value = "20")]
    s1_min_valid_timesteps: usize,

    /// Number of timesteps to sample (time_steps)
    #[arg(long, default_value = "20")]
    time_steps: usize,

    /// How many tiles to process in one batch (control memory usage)
    #[arg(long, default_value = "100")]
    tile_batch: usize,

    /// Output directory for augmented data
    #[arg(long, default_value = "./aug_output")]
    output_dir: String,

    /// Save chunk size, i.e. how many pixels in one .npy file
    #[arg(long, default_value = "10000")]
    chunk_size: usize,
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

/// 每个像素会生成四个 Array2<f32> 表示 (s2_a1, s2_a2, s1_a1, s1_a2)
struct SampleOut {
    s2_a1: Array2<f32>, // shape=(time_steps, 11)
    s2_a2: Array2<f32>,
    s1_a1: Array2<f32>, // shape=(time_steps, 3)
    s1_a2: Array2<f32>,
}

fn main() -> Result<()> {
    env_logger::Builder::from_default_env()
        .format_timestamp_secs()
        .init();
    info!("=== training-data-preprocessing-single-doy start ===");

    let args = Args::parse();
    info!("Parsed command-line args: {:?}", args);

    let data_root = Path::new(&args.data_root);
    let out_dir = Path::new(&args.output_dir);

    // 创建输出目录(含 aug1/aug2/s2/s1)
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
                Ok(sample_out_vec) => {
                    let npix = sample_out_vec.len();
                    info!("Tile #{} => got {} valid px => accumulate..", tile_counter, npix);
                    if npix == 0 {
                        continue;
                    }
                    for so in sample_out_vec {
                        s2a1_vec.push(so.s2_a1);
                        s2a2_vec.push(so.s2_a2);
                        s1a1_vec.push(so.s1_a1);
                        s1a2_vec.push(so.s1_a2);
                    }
                }
                Err(e) => {
                    warn!("Tile error: {:?}", e);
                }
            }
        }

        let total_pixels_in_batch = s2a1_vec.len();
        if total_pixels_in_batch == 0 {
            info!("[Batch #{}] => No valid pixel => skip this batch", batch_index);
            start_tile_idx = end_tile_idx;
            continue;
        }

        // 并行合并像素数据
        let s2a1_all = stack_3d(&s2a1_vec)?;
        let s2a2_all = stack_3d(&s2a2_vec)?;
        let s1a1_all = stack_3d(&s1a1_vec)?;
        let s1a2_all = stack_3d(&s1a2_vec)?;

        info!("[Batch #{}] => done merging => total valid px={}, shape s2_a1=({}, {}, {})",
              batch_index,
              total_pixels_in_batch,
              s2a1_all.len_of(Axis(0)),
              s2a1_all.len_of(Axis(1)),
              s2a1_all.len_of(Axis(2)));

        // 全局 Shuffle：先生成排列索引，再用并行复制每一行数据
        let n_pix = s2a1_all.len_of(Axis(0));
        let mut perm: Vec<usize> = (0..n_pix).collect();
        perm.shuffle(&mut rand::thread_rng());

        let s2a1_all = parallel_permute(&s2a1_all, &perm);
        let s2a2_all = parallel_permute(&s2a2_all, &perm);
        let s1a1_all = parallel_permute(&s1a1_all, &perm);
        let s1a2_all = parallel_permute(&s1a2_all, &perm);

        info!("[Batch #{}] => shuffled all pixels", batch_index);

        // 只写入完整的 chunk 文件，舍弃最后不足 chunk_size 的边角料
        let n_chunks = n_pix / args.chunk_size;
        info!("[Batch #{}] => starting parallel write of {} chunks", batch_index, n_chunks);
        (0..n_chunks).into_par_iter().for_each(|chunk_idx| {
            let offset = chunk_idx * args.chunk_size;
            let end_offset = offset + args.chunk_size;
            let block_len = args.chunk_size;

            let s2a1_slice = s2a1_all.slice(s![offset..end_offset, .., ..]).to_owned();
            let s2a2_slice = s2a2_all.slice(s![offset..end_offset, .., ..]).to_owned();
            let s1a1_slice = s1a1_all.slice(s![offset..end_offset, .., ..]).to_owned();
            let s1a2_slice = s1a2_all.slice(s![offset..end_offset, .., ..]).to_owned();

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
            info!("[Batch #{}] => wrote chunk file #{} with {} pixels", batch_index, file_counter, block_len);
        });

        start_tile_idx = end_tile_idx;
    }
    pb.finish_with_message("All done.");
    info!("=== Done. Check your output_dir for files. ===");

    Ok(())
}

/// 处理单个 tile：读取数据、转换并增广
fn process_tile(tile_dir: &Path, args: &Args) -> Result<Vec<SampleOut>> {
    let tile_name = tile_dir.file_name()
        .map(|os| os.to_string_lossy().to_string())
        .unwrap_or_else(|| "UnknownTile".to_string());
    debug!("process_tile() start => tile={}", tile_name);

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
    debug!("tile={}, s2_bands shape=({},{},{},{})", tile_name, t_s2, h, w, shape_s2[3]);

    // 将 masks 转为 0/1 数组
    let mut s2_mask_u8 = ndarray::Array3::<u8>::zeros((t_s2, h, w));
    for ((tt, yy, xx), &val) in s2_masks.indexed_iter() {
        if val != 0 {
            s2_mask_u8[(tt, yy, xx)] = 1;
        }
    }

    let mut valid_pixels = Vec::new();
    for yy in 0..h {
        for xx in 0..w {
            // 根据 s2_min_valid_timesteps 判断是否有效
            let s2_valid = s2_mask_u8.slice(s![.., yy, xx]).sum() as usize;
            if s2_valid < args.s2_min_valid_timesteps {
                continue;
            }
            // 根据 s1_min_valid_timesteps 判断是否有效
            let asc_valid = count_sar_valid(&s1_asc_bands, yy, xx);
            let desc_valid = count_sar_valid(&s1_desc_bands, yy, xx);
            if (asc_valid + desc_valid) < args.s1_min_valid_timesteps as u16 {
                continue;
            }
            valid_pixels.push((yy, xx));
        }
    }
    let valid_count = valid_pixels.len();
    debug!("tile={}, found {} valid pixels", tile_name, valid_count);
    if valid_count == 0 {
        return Ok(Vec::new());
    }

    // 对每个有效像素并行生成增广数据
    let result_vec: Vec<SampleOut> = (0..valid_count).into_par_iter()
        .map(|i| {
            let (yy, xx) = valid_pixels[i];
            // 收集 Sentinel-2 中有效的时间索引
            let mut s2_v_idx = Vec::new();
            for t in 0..t_s2 {
                if s2_mask_u8[(t, yy, xx)] == 1 {
                    s2_v_idx.push(t);
                }
            }

            let s2a1 = sample_s2_pixel(&s2_bands, &s2_doys, &s2_v_idx, yy, xx, args.time_steps);
            let s2a2 = sample_s2_pixel(&s2_bands, &s2_doys, &s2_v_idx, yy, xx, args.time_steps);

            let (s1a1, s1a2) = sample_s1_pixel(
                &s1_asc_bands, &s1_asc_doys,
                &s1_desc_bands, &s1_desc_doys,
                yy, xx, args.time_steps
            );
            SampleOut { s2_a1: s2a1, s2_a2: s2a2, s1_a1: s1a1, s1_a2: s1a2 }
        })
        .collect();

    Ok(result_vec)
}

/// 统计给定 SAR band 中某像素非零帧数
fn count_sar_valid(sar: &Array4<i16>, yy: usize, xx: usize) -> u16 {
    let t_sar = sar.len_of(Axis(0));
    let mut cnt = 0u16;
    for t in 0..t_sar {
        let vv = sar[(t, yy, xx, 0)];
        let vh = sar[(t, yy, xx, 1)];
        if vv != 0 || vh != 0 {
            cnt += 1;
        }
    }
    cnt
}

/// 对 S2：10 个 band 归一化并添加 doy，结果 shape=(time_steps, 11)
fn sample_s2_pixel(
    s2_bands: &Array4<u16>,
    s2_doys: &Array1<u16>,
    valid_idx: &Vec<usize>,
    yy: usize, xx: usize,
    time_steps: usize
) -> Array2<f32> {
    let n_valid = valid_idx.len();
    let mut rng = rand::thread_rng();
    let mut out = Array2::<f32>::zeros((time_steps, 11));  // Changed from 12 to 11

    for t in 0..time_steps {
        if n_valid == 0 { break; }
        let pick = valid_idx[rng.gen_range(0..n_valid)];
        // 读取 10 个 band 并归一化
        let mut tmp = [0f32; 10];
        for b in 0..10 {
            let rawv = s2_bands[(pick, yy, xx, b)] as f32;
            let normed = (rawv - S2_BAND_MEAN[b]) / (S2_BAND_STD[b] + 1e-9);
            tmp[b] = normed;
        }
        let doy_f32 = s2_doys[pick] as f32;  // Convert DOY to float

        for b in 0..10 {
            out[(t, b)] = tmp[b];
        }
        out[(t, 10)] = doy_f32;  // Directly add the DOY as float
    }
    out
}

/// 对 S1：合并 asc 与 desc 数据，生成两个增广结果，结果 shape=(time_steps, 3)
fn sample_s1_pixel(
    asc_bands: &Array4<i16>,
    asc_doys: &Array1<i32>,
    desc_bands: &Array4<i16>,
    desc_doys: &Array1<i32>,
    yy: usize, xx: usize,
    time_steps: usize
) -> (Array2<f32>, Array2<f32>) {
    let t_asc = asc_bands.len_of(Axis(0));
    let mut merged_idx = Vec::new();
    // asc 数据
    for t in 0..t_asc {
        let vv = asc_bands[(t, yy, xx, 0)];
        let vh = asc_bands[(t, yy, xx, 1)];
        if vv != 0 || vh != 0 {
            merged_idx.push((true, t));
        }
    }
    // desc 数据
    let t_desc = desc_bands.len_of(Axis(0));
    for t in 0..t_desc {
        let vv = desc_bands[(t, yy, xx, 0)];
        let vh = desc_bands[(t, yy, xx, 1)];
        if vv != 0 || vh != 0 {
            merged_idx.push((false, t));
        }
    }
    let n_valid = merged_idx.len();
    let mut rng = rand::thread_rng();

    let mut arr1 = Array2::<f32>::zeros((time_steps, 3));  // Changed from 4 to 3
    let mut arr2 = Array2::<f32>::zeros((time_steps, 3));  // Changed from 4 to 3

    for arr in [&mut arr1, &mut arr2] {
        for t in 0..time_steps {
            if n_valid == 0 { break; }
            let pick = merged_idx[rng.gen_range(0..n_valid)];
            let (is_asc, idx) = pick;
            let (raw_vv, raw_vh, doy_val) = if is_asc {
                (asc_bands[(idx, yy, xx, 0)] as f32,
                 asc_bands[(idx, yy, xx, 1)] as f32,
                 asc_doys[idx] as f32)  // Convert to float
            } else {
                (desc_bands[(idx, yy, xx, 0)] as f32,
                 desc_bands[(idx, yy, xx, 1)] as f32,
                 desc_doys[idx] as f32)  // Convert to float
            };
            let vv_norm = (raw_vv - S1_BAND_MEAN[0]) / (S1_BAND_STD[0] + 1e-9);
            let vh_norm = (raw_vh - S1_BAND_MEAN[1]) / (S1_BAND_STD[1] + 1e-9);

            arr[(t, 0)] = vv_norm;
            arr[(t, 1)] = vh_norm;
            arr[(t, 2)] = doy_val;  // Directly add the DOY as float
        }
    }
    (arr1, arr2)
}

/// 将多个 (time_steps, c) 的 Array2<f32> 拼接成 (n_pix, time_steps, c) 的 3D 数组
fn stack_3d(v: &Vec<Array2<f32>>) -> Result<Array3<f32>> {
    if v.is_empty() {
        return Ok(Array3::<f32>::zeros((0, 0, 0)));
    }
    let total_n = v.len();
    let t_s = v[0].len_of(Axis(0));
    let c = v[0].len_of(Axis(1));
    let mut out = Array3::<f32>::zeros((total_n, t_s, c));
    // 由于 Array3 内存连续，我们获取可变切片，并按行（连续 row_len 个元素）进行并行复制
    let row_len = t_s * c;
    {
        let out_slice = out.as_slice_mut().expect("Array3 not contiguous");
        out_slice.par_chunks_exact_mut(row_len)
            .enumerate()
            .for_each(|(i, chunk)| {
                let src = v[i].as_slice().expect("v[i] not contiguous");
                chunk.copy_from_slice(src);
            });
    }
    Ok(out)
}

/// 并行排列：根据排列索引 perm，对 Array3<f32> 的第一维行重排，返回新的 Array3<f32>
fn parallel_permute(arr: &Array3<f32>, perm: &[usize]) -> Array3<f32> {
    let (n, t, c) = arr.dim();
    let row_len = t * c;
    let mut new_arr = Array3::<f32>::zeros((n, t, c));
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