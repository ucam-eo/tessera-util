use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

use clap::Parser;
use env_logger;
use indicatif::{ProgressBar, ProgressStyle};
use log::{info, warn, error, debug};
use rayon::prelude::*;
use rand::Rng;
use rand::seq::SliceRandom;
use anyhow::{Result, Context};

use ndarray::{Array1, Array3, Array4, Array5, Axis, s};
use ndarray_npy::{read_npy, write_npy};

#[derive(Parser, Debug)]
#[command(name = "training-data-preprocessing-patch", version = "0.1.0")]
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
    /// e.g., stride=0 => consecutive pixels, stride=1 =>skip 1 pixel =>larger coverage
    #[arg(long, default_value = "0")]
    stride: usize,
    
    /// Sub-tile size in pixels
    #[arg(long, default_value = "1000")]
    subtile_size: usize,
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

/// Struct to hold patch data for S1 and S2 (two augmentations)
struct SampleOut {
    s2_a1: Array4<f32>, // shape=(patch_size, patch_size, time_steps, 11)
    s2_a2: Array4<f32>, // shape=(patch_size, patch_size, time_steps, 11)
    s1_a1: Array4<f32>, // shape=(patch_size, patch_size, time_steps, 3)
    s1_a2: Array4<f32>, // shape=(patch_size, patch_size, time_steps, 3)
}

fn main() -> Result<()> {
    env_logger::Builder::from_default_env()
        .format_timestamp_secs()
        .init();
    info!("=== training-data-preprocessing-patch start ===");

    let args = Args::parse();
    info!("Parsed command-line args: {:?}", args);

    // Validate patch_size is odd
    if args.patch_size % 2 == 0 {
        error!("Patch size must be odd, got {}", args.patch_size);
        return Ok(());
    }

    let data_root = Path::new(&args.data_root);
    let out_dir = Path::new(&args.output_dir);

    // Create output directories
    info!("Creating output directories under {:?}", out_dir);
    fs::create_dir_all(&out_dir)
        .with_context(|| format!("Failed to create output directory {:?}", out_dir))?;
    fs::create_dir_all(out_dir.join("aug1/s2"))?;
    fs::create_dir_all(out_dir.join("aug1/s1"))?;
    fs::create_dir_all(out_dir.join("aug2/s2"))?;
    fs::create_dir_all(out_dir.join("aug2/s1"))?;

    // 1) Scan for tile folders
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
        // Check if required files exist
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

    // Sort and shuffle tiles
    tile_paths.sort();
    tile_paths.shuffle(&mut rand::thread_rng());

    let total_tiles = tile_paths.len();
    info!("Found {} tile subfolders in {:?}", total_tiles, data_root);
    if total_tiles == 0 {
        error!("No valid tile subfolders found => nothing to do!");
        return Ok(());
    }

    // Progress bar
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

        info!("==== [Batch #{}] => tile indices in range [{}, {}) ====", 
              batch_index, start_tile_idx, end_tile_idx);

        // Process each tile in parallel
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
                    let npatches = sample_out_vec.len();
                    info!("Tile #{} => got {} valid patches => accumulate..", 
                          tile_counter, npatches);
                    if npatches == 0 {
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

        let total_patches_in_batch = s2a1_vec.len();
        if total_patches_in_batch == 0 {
            info!("[Batch #{}] => No valid patches => skip this batch", batch_index);
            start_tile_idx = end_tile_idx;
            continue;
        }

        // Merge patch data
        let s2a1_all = stack_5d(&s2a1_vec, args.patch_size, args.patch_size, args.time_steps, 11)?;
        let s2a2_all = stack_5d(&s2a2_vec, args.patch_size, args.patch_size, args.time_steps, 11)?;
        let s1a1_all = stack_5d(&s1a1_vec, args.patch_size, args.patch_size, args.time_steps, 3)?;
        let s1a2_all = stack_5d(&s1a2_vec, args.patch_size, args.patch_size, args.time_steps, 3)?;

        info!("[Batch #{}] => done merging => total valid patches={}, shape s2_a1=({}, {}, {}, {}, {})",
              batch_index,
              total_patches_in_batch,
              s2a1_all.len_of(Axis(0)),
              s2a1_all.len_of(Axis(1)),
              s2a1_all.len_of(Axis(2)),
              s2a1_all.len_of(Axis(3)),
              s2a1_all.len_of(Axis(4)));

        // Global shuffle of patches
        let n_patches = s2a1_all.len_of(Axis(0));
        let mut perm: Vec<usize> = (0..n_patches).collect();
        perm.shuffle(&mut rand::thread_rng());

        let s2a1_all = parallel_permute_5d(&s2a1_all, &perm);
        let s2a2_all = parallel_permute_5d(&s2a2_all, &perm);
        let s1a1_all = parallel_permute_5d(&s1a1_all, &perm);
        let s1a2_all = parallel_permute_5d(&s1a2_all, &perm);

        info!("[Batch #{}] => shuffled all patches", batch_index);

        // Only write complete chunks, discard any remaining incomplete chunk
        let n_chunks = n_patches / args.chunk_size;
        info!("[Batch #{}] => starting parallel write of {} chunks", batch_index, n_chunks);
        
        (0..n_chunks).into_par_iter().for_each(|chunk_idx| {
            let offset = chunk_idx * args.chunk_size;
            let end_offset = offset + args.chunk_size;
            let block_len = args.chunk_size;

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
            info!("[Batch #{}] => wrote chunk file #{} with {} patches", 
                  batch_index, file_counter, block_len);
        });

        start_tile_idx = end_tile_idx;
    }
    
    pb.finish_with_message("All done.");
    info!("=== Done. Check your output_dir for files. ===");

    Ok(())
}

/// Process a single tile: split into sub-tiles and generate patches
fn process_tile(tile_dir: &Path, args: &Args) -> Result<Vec<SampleOut>> {
    let tile_name = tile_dir.file_name()
        .map(|os| os.to_string_lossy().to_string())
        .unwrap_or_else(|| "UnknownTile".to_string());
    
    info!("process_tile() start => tile={}", tile_name);

    // Read Sentinel-2 data
    let s2_bands: Array4<u16> = read_npy(tile_dir.join("bands.npy"))
        .with_context(|| format!("reading s2 bands in tile={}", tile_name))?;
    let s2_masks: Array3<u8> = read_npy(tile_dir.join("masks.npy"))
        .with_context(|| format!("reading s2 masks in tile={}", tile_name))?;
    let s2_doys: Array1<u16> = read_npy(tile_dir.join("doys.npy"))
        .with_context(|| format!("reading s2 doys in tile={}", tile_name))?;

    // Read SAR data
    let s1_asc_bands: Array4<i16> = read_npy(tile_dir.join("sar_ascending.npy"))
        .with_context(|| format!("reading s1 asc bands in tile={}", tile_name))?;
    let s1_asc_doys: Array1<i32> = read_npy(tile_dir.join("sar_ascending_doy.npy"))
        .with_context(|| format!("reading s1 asc doys in tile={}", tile_name))?;
    let s1_desc_bands: Array4<i16> = read_npy(tile_dir.join("sar_descending.npy"))
        .with_context(|| format!("reading s1 desc bands in tile={}", tile_name))?;
    let s1_desc_doys: Array1<i32> = read_npy(tile_dir.join("sar_descending_doy.npy"))
        .with_context(|| format!("reading s1 desc doys in tile={}", tile_name))?;

    let shape_s2 = s2_bands.shape();
    let t_s2 = shape_s2[0];
    let h = shape_s2[1];
    let w = shape_s2[2];
    
    info!("tile={}, s2_bands shape=({},{},{},{})", tile_name, t_s2, h, w, shape_s2[3]);

    // Convert masks to 0/1 array
    let mut s2_mask_u8 = ndarray::Array3::<u8>::zeros((t_s2, h, w));
    for ((tt, yy, xx), &val) in s2_masks.indexed_iter() {
        if val != 0 {
            s2_mask_u8[(tt, yy, xx)] = 1;
        }
    }

    // Determine valid S1 timesteps for each pixel
    let t_s1_asc = s1_asc_bands.len_of(Axis(0));
    let t_s1_desc = s1_desc_bands.len_of(Axis(0));
    
    // Create S1 masks for both ascending and descending orbits
    let mut s1_asc_mask = ndarray::Array3::<u8>::zeros((t_s1_asc, h, w));
    let mut s1_desc_mask = ndarray::Array3::<u8>::zeros((t_s1_desc, h, w));
    
    // Fill S1 ascending mask
    for t in 0..t_s1_asc {
        for y in 0..h {
            for x in 0..w {
                let vv = s1_asc_bands[(t, y, x, 0)];
                let vh = s1_asc_bands[(t, y, x, 1)];
                if vv != 0 || vh != 0 {
                    s1_asc_mask[(t, y, x)] = 1;
                }
            }
        }
    }
    
    // Fill S1 descending mask
    for t in 0..t_s1_desc {
        for y in 0..h {
            for x in 0..w {
                let vv = s1_desc_bands[(t, y, x, 0)];
                let vh = s1_desc_bands[(t, y, x, 1)];
                if vv != 0 || vh != 0 {
                    s1_desc_mask[(t, y, x)] = 1;
                }
            }
        }
    }

    info!("tile={}, analyzing sub-tiles", tile_name);
    
    // Split into sub-tiles of size subtile_size x subtile_size, processing partial sub-tiles at edges
    let subtile_size = args.subtile_size;
    let mut all_samples = Vec::new();
    
    // Calculate number of complete sub-tiles and handle remainders
    let num_subtiles_h = (h + subtile_size - 1) / subtile_size; // Ceiling division
    let num_subtiles_w = (w + subtile_size - 1) / subtile_size; // Ceiling division
    
    info!("tile={}, will process {}x{} sub-tiles (including partial edge tiles)", 
          tile_name, num_subtiles_h, num_subtiles_w);
    
    for sh in 0..num_subtiles_h {
        for sw in 0..num_subtiles_w {
            let start_h = sh * subtile_size;
            let start_w = sw * subtile_size;
            // Handle the edge cases - make sure we don't go beyond image boundaries
            let end_h = (start_h + subtile_size).min(h);
            let end_w = (start_w + subtile_size).min(w);
            
            let subtile_h = end_h - start_h;
            let subtile_w = end_w - start_w;
            
            // Calculate area ratio compared to full sub-tile
            let area_ratio = (subtile_h * subtile_w) as f32 / (subtile_size * subtile_size) as f32;
            let scaled_max_patches = (args.max_patch_per_1k_1k_tile as f32 * area_ratio).round() as usize;
            
            if scaled_max_patches < 1 {
                info!("tile={}, skipping tiny sub-tile ({},{}) with area ratio {:.4}, too small", 
                      tile_name, sh, sw, area_ratio);
                continue;
            }
            
            info!("tile={}, processing sub-tile ({},{}) => region [{}:{}, {}:{}], area ratio={:.4}, max_patches={}", 
                  tile_name, sh, sw, start_h, end_h, start_w, end_w, area_ratio, scaled_max_patches);
            
            let mut subtile_samples = process_subtile(
                &s2_bands, &s2_mask_u8, &s2_doys,
                &s1_asc_bands, &s1_asc_mask, &s1_asc_doys,
                &s1_desc_bands, &s1_desc_mask, &s1_desc_doys,
                start_h, end_h, start_w, end_w, scaled_max_patches, args, &tile_name
            )?;
            
            info!("tile={}, sub-tile ({},{}) generated {} patches", 
                  tile_name, sh, sw, subtile_samples.len());
            
            all_samples.append(&mut subtile_samples);
        }
    }
    
    info!("tile={}, total patches generated: {}", tile_name, all_samples.len());
    Ok(all_samples)
}

/// Process a sub-tile to generate patches
fn process_subtile(
    s2_bands: &Array4<u16>,
    s2_mask: &Array3<u8>,
    s2_doys: &Array1<u16>,
    s1_asc_bands: &Array4<i16>,
    s1_asc_mask: &Array3<u8>,
    s1_asc_doys: &Array1<i32>,
    s1_desc_bands: &Array4<i16>,
    s1_desc_mask: &Array3<u8>,
    s1_desc_doys: &Array1<i32>,
    start_h: usize,
    end_h: usize,
    start_w: usize,
    end_w: usize,
    max_patches: usize,
    args: &Args,
    tile_name: &str
) -> Result<Vec<SampleOut>> {
    let mut rng = rand::thread_rng();
    let subtile_h = end_h - start_h;
    let subtile_w = end_w - start_w;
    
    info!("tile={}, subtile [{}:{}, {}:{}] processing", 
          tile_name, start_h, end_h, start_w, end_w);
    
    // Calculate half patch size for later use
    let half_patch = args.patch_size / 2;
    
    // Calculate valid region where patches can be centered
    // (considering patch size and stride)
    let effective_stride = args.stride + 1;
    // Removing the unused variable warning by adding an underscore
    let _stride_patch_size = args.patch_size * effective_stride;
    
    // Make sure we don't go beyond the image boundaries
    let valid_start_h = start_h + half_patch * effective_stride;
    let valid_end_h = end_h.saturating_sub(half_patch * effective_stride);
    let valid_start_w = start_w + half_patch * effective_stride;
    let valid_end_w = end_w.saturating_sub(half_patch * effective_stride);
    
    // If valid region is empty, return
    if valid_start_h >= valid_end_h || valid_start_w >= valid_end_w {
        info!("tile={}, subtile [{}:{}, {}:{}] has no valid region for patches with size={} and stride={}", 
              tile_name, start_h, end_h, start_w, end_w, args.patch_size, args.stride);
        return Ok(Vec::new());
    }
    
    // Find valid S2 timesteps for the subtile
    let t_s2 = s2_mask.len_of(Axis(0));
    let mut valid_s2_timesteps_map = Vec::new();
    
    for t in 0..t_s2 {
        let mut valid_count = 0;
        let total_pixels = subtile_h * subtile_w;
        
        for y in start_h..end_h {
            for x in start_w..end_w {
                if s2_mask[(t, y, x)] == 1 {
                    valid_count += 1;
                }
            }
        }
        
        let valid_ratio = valid_count as f32 / total_pixels as f32;
        if valid_ratio >= args.s2_valid_ratio {
            valid_s2_timesteps_map.push(t);
        }
    }
    
    info!("tile={}, subtile [{}:{}, {}:{}] has {} valid S2 timesteps", 
          tile_name, start_h, end_h, start_w, end_w, valid_s2_timesteps_map.len());
    
    if valid_s2_timesteps_map.len() < args.s2_min_valid_timesteps {
        info!("tile={}, subtile [{}:{}, {}:{}] has insufficient valid S2 timesteps: {} < {}", 
              tile_name, start_h, end_h, start_w, end_w, 
              valid_s2_timesteps_map.len(), args.s2_min_valid_timesteps);
        return Ok(Vec::new());
    }
    
    // Find valid S1 timesteps for the subtile
    let t_s1_asc = s1_asc_mask.len_of(Axis(0));
    let t_s1_desc = s1_desc_mask.len_of(Axis(0));
    
    let mut valid_s1_asc_timesteps = Vec::new();
    let mut valid_s1_desc_timesteps = Vec::new();
    
    // Check S1 ascending timesteps
    for t in 0..t_s1_asc {
        let mut valid_count = 0;
        let total_pixels = subtile_h * subtile_w;
        
        for y in start_h..end_h {
            for x in start_w..end_w {
                if s1_asc_mask[(t, y, x)] == 1 {
                    valid_count += 1;
                }
            }
        }
        
        let valid_ratio = valid_count as f32 / total_pixels as f32;
        if valid_ratio >= args.s1_valid_ratio {
            valid_s1_asc_timesteps.push(t);
        }
    }
    
    // Check S1 descending timesteps
    for t in 0..t_s1_desc {
        let mut valid_count = 0;
        let total_pixels = subtile_h * subtile_w;
        
        for y in start_h..end_h {
            for x in start_w..end_w {
                if s1_desc_mask[(t, y, x)] == 1 {
                    valid_count += 1;
                }
            }
        }
        
        let valid_ratio = valid_count as f32 / total_pixels as f32;
        if valid_ratio >= args.s1_valid_ratio {
            valid_s1_desc_timesteps.push(t);
        }
    }
    
    let total_valid_s1_timesteps = valid_s1_asc_timesteps.len() + valid_s1_desc_timesteps.len();
    
    info!("tile={}, subtile [{}:{}, {}:{}] has {} valid S1 timesteps ({} asc, {} desc)", 
          tile_name, start_h, end_h, start_w, end_w, 
          total_valid_s1_timesteps, valid_s1_asc_timesteps.len(), valid_s1_desc_timesteps.len());
    
    if total_valid_s1_timesteps < args.s1_min_valid_timesteps {
        info!("tile={}, subtile [{}:{}, {}:{}] has insufficient valid S1 timesteps: {} < {}", 
              tile_name, start_h, end_h, start_w, end_w, 
              total_valid_s1_timesteps, args.s1_min_valid_timesteps);
        return Ok(Vec::new());
    }
    
    // Generate random patch centers
    let max_attempts = max_patches * 2; // Allow some failures
    
    let mut samples = Vec::new();
    let mut valid_patches = 0;
    let mut attempts = 0;
    
    info!("tile={}, subtile [{}:{}, {}:{}] generating up to {} patches with max {} attempts", 
          tile_name, start_h, end_h, start_w, end_w, max_patches, max_attempts);
    
    while valid_patches < max_patches && attempts < max_attempts {
        attempts += 1;
        
        // Generate random center point within valid region
        let center_y = rng.gen_range(valid_start_h..valid_end_h);
        let center_x = rng.gen_range(valid_start_w..valid_end_w);
        
        // Check if patch is valid (uses validation functions we'll implement below)
        let is_valid_s2 = is_valid_s2_patch(
            s2_mask, &valid_s2_timesteps_map, 
            center_y, center_x, half_patch, args.stride, args.s2_valid_ratio
        );
        
        let is_valid_s1 = is_valid_s1_patch(
            s1_asc_mask, &valid_s1_asc_timesteps,
            s1_desc_mask, &valid_s1_desc_timesteps,
            center_y, center_x, half_patch, args.stride, args.s1_valid_ratio
        );
        
        if !is_valid_s2 || !is_valid_s1 {
            continue; // Skip this patch center
        }
        
        // Both S2 and S1 are valid, extract patch data
        let patch_result = extract_patch(
            s2_bands, s2_mask, s2_doys,
            s1_asc_bands, s1_asc_mask, s1_asc_doys,
            s1_desc_bands, s1_desc_mask, s1_desc_doys,
            center_y, center_x, &valid_s2_timesteps_map,
            &valid_s1_asc_timesteps, &valid_s1_desc_timesteps,
            args
        );
        
        match patch_result {
            Ok(patch) => {
                samples.push(patch);
                valid_patches += 1;
                
                if valid_patches % 1000 == 0 || valid_patches == max_patches {
                    info!("tile={}, subtile [{}:{}, {}:{}] generated {} patches so far (attempts: {})", 
                          tile_name, start_h, end_h, start_w, end_w, valid_patches, attempts);
                }
            }
            Err(e) => {
                debug!("Error extracting patch at ({}, {}): {:?}", center_y, center_x, e);
            }
        }
    }
    
    info!("tile={}, subtile [{}:{}, {}:{}] completed with {} valid patches after {} attempts", 
          tile_name, start_h, end_h, start_w, end_w, valid_patches, attempts);
    
    Ok(samples)
}

/// Check if S2 patch at given center is valid
fn is_valid_s2_patch(
    s2_mask: &Array3<u8>,
    valid_timesteps: &Vec<usize>,
    center_y: usize,
    center_x: usize,
    half_patch: usize,
    stride: usize,
    valid_ratio_threshold: f32
) -> bool {
    if valid_timesteps.is_empty() {
        return false;
    }
    
    // For each timestep, check if the patch has enough valid pixels
    let patch_size = half_patch * 2 + 1;
    let total_pixels = patch_size * patch_size;
    let effective_stride = stride + 1;
    
    // For each valid timestep, all pixels in the patch should be checked
    for &t in valid_timesteps {
        let mut valid_pixels = 0;
        
        // Iterate over the patch pixels with stride
        for py in 0..patch_size {
            // Calculate absolute y coordinate with stride
            let y_offset = py as isize - half_patch as isize;
            let y = center_y as isize + y_offset * effective_stride as isize;
            
            for px in 0..patch_size {
                // Calculate absolute x coordinate with stride
                let x_offset = px as isize - half_patch as isize;
                let x = center_x as isize + x_offset * effective_stride as isize;
                
                // Check boundaries and get mask value if valid
                if y >= 0 && x >= 0 && 
                   y < s2_mask.len_of(Axis(1)) as isize && 
                   x < s2_mask.len_of(Axis(2)) as isize {
                    // Safe to convert to usize here as we've checked they're positive
                    if s2_mask[(t, y as usize, x as usize)] == 1 {
                        valid_pixels += 1;
                    }
                }
            }
        }
        
        let patch_valid_ratio = valid_pixels as f32 / total_pixels as f32;
        if patch_valid_ratio < valid_ratio_threshold {
            return false;
        }
    }
    
    true
}

/// Check if S1 patch at given center is valid
fn is_valid_s1_patch(
    s1_asc_mask: &Array3<u8>,
    valid_asc_timesteps: &Vec<usize>,
    s1_desc_mask: &Array3<u8>,
    valid_desc_timesteps: &Vec<usize>,
    center_y: usize,
    center_x: usize,
    half_patch: usize,
    stride: usize,
    valid_ratio_threshold: f32
) -> bool {
    let total_valid_timesteps = valid_asc_timesteps.len() + valid_desc_timesteps.len();
    if total_valid_timesteps == 0 {
        return false;
    }
    
    // For each timestep, check if the patch has enough valid pixels
    let patch_size = half_patch * 2 + 1;
    let total_pixels = patch_size * patch_size;
    let effective_stride = stride + 1;
    
    // We need at least min_valid_timesteps across both ascending and descending
    let mut total_valid_timesteps_for_patch = 0;
    
    // Check ascending orbits
    for &t in valid_asc_timesteps {
        let mut valid_pixels = 0;
        
        // Iterate over the patch pixels with stride
        for py in 0..patch_size {
            // Calculate absolute y coordinate with stride
            let y_offset = py as isize - half_patch as isize;
            let y = center_y as isize + y_offset * effective_stride as isize;
            
            for px in 0..patch_size {
                // Calculate absolute x coordinate with stride
                let x_offset = px as isize - half_patch as isize;
                let x = center_x as isize + x_offset * effective_stride as isize;
                
                // Check boundaries and get mask value if valid
                if y >= 0 && x >= 0 && 
                   y < s1_asc_mask.len_of(Axis(1)) as isize && 
                   x < s1_asc_mask.len_of(Axis(2)) as isize {
                    // Safe to convert to usize here as we've checked they're positive
                    if s1_asc_mask[(t, y as usize, x as usize)] == 1 {
                        valid_pixels += 1;
                    }
                }
            }
        }
        
        let patch_valid_ratio = valid_pixels as f32 / total_pixels as f32;
        if patch_valid_ratio >= valid_ratio_threshold {
            total_valid_timesteps_for_patch += 1;
        }
    }
    
    // Check descending orbits
    for &t in valid_desc_timesteps {
        let mut valid_pixels = 0;
        
        // Iterate over the patch pixels with stride
        for py in 0..patch_size {
            // Calculate absolute y coordinate with stride
            let y_offset = py as isize - half_patch as isize;
            let y = center_y as isize + y_offset * effective_stride as isize;
            
            for px in 0..patch_size {
                // Calculate absolute x coordinate with stride
                let x_offset = px as isize - half_patch as isize;
                let x = center_x as isize + x_offset * effective_stride as isize;
                
                // Check boundaries and get mask value if valid
                if y >= 0 && x >= 0 && 
                   y < s1_desc_mask.len_of(Axis(1)) as isize && 
                   x < s1_desc_mask.len_of(Axis(2)) as isize {
                    // Safe to convert to usize here as we've checked they're positive
                    if s1_desc_mask[(t, y as usize, x as usize)] == 1 {
                        valid_pixels += 1;
                    }
                }
            }
        }
        
        let patch_valid_ratio = valid_pixels as f32 / total_pixels as f32;
        if patch_valid_ratio >= valid_ratio_threshold {
            total_valid_timesteps_for_patch += 1;
        }
    }
    
    total_valid_timesteps_for_patch >= 1 // At least one timestep should be valid for the patch
}

/// Extract a patch from both S1 and S2 data, creating two augmentations for each
fn extract_patch(
    s2_bands: &Array4<u16>,
    _s2_mask: &Array3<u8>,
    s2_doys: &Array1<u16>,
    s1_asc_bands: &Array4<i16>,
    _s1_asc_mask: &Array3<u8>,
    s1_asc_doys: &Array1<i32>,
    s1_desc_bands: &Array4<i16>,
    _s1_desc_mask: &Array3<u8>,
    s1_desc_doys: &Array1<i32>,
    center_y: usize,
    center_x: usize,
    valid_s2_timesteps: &Vec<usize>,
    valid_s1_asc_timesteps: &Vec<usize>,
    valid_s1_desc_timesteps: &Vec<usize>,
    args: &Args
) -> Result<SampleOut> {
    let mut rng = rand::thread_rng();
    let patch_size = args.patch_size;
    let half_patch = patch_size / 2;
    let effective_stride = args.stride + 1;
    
    // Create output arrays
    let mut s2_a1 = Array4::<f32>::zeros((patch_size, patch_size, args.time_steps, 11));
    let mut s2_a2 = Array4::<f32>::zeros((patch_size, patch_size, args.time_steps, 11));
    let mut s1_a1 = Array4::<f32>::zeros((patch_size, patch_size, args.time_steps, 3));
    let mut s1_a2 = Array4::<f32>::zeros((patch_size, patch_size, args.time_steps, 3));
    
    // Sample S2 timesteps (we'll use the same timesteps for both augmentations but sample differently)
    let n_valid_s2 = valid_s2_timesteps.len();
    if n_valid_s2 == 0 {
        return Err(anyhow::anyhow!("No valid S2 timesteps for patch"));
    }
    
    let mut selected_s2_timesteps = Vec::new();
    for _ in 0..args.time_steps {
        let idx = rng.gen_range(0..n_valid_s2);
        selected_s2_timesteps.push(valid_s2_timesteps[idx]);
    }
    
    // Process S2 data for both augmentations
    for (_aug_idx, aug_output) in [&mut s2_a1, &mut s2_a2].iter_mut().enumerate() {
        // For each selected timestep
        for (t_out, &t_in) in selected_s2_timesteps.iter().enumerate() {
            // Extract patch for this timestep
            for py in 0..patch_size {
                let y_offset = py as isize - half_patch as isize;
                let y = center_y as isize + y_offset * effective_stride as isize;
                
                for px in 0..patch_size {
                    let x_offset = px as isize - half_patch as isize;
                    let x = center_x as isize + x_offset * effective_stride as isize;
                    
                    if y >= 0 && x >= 0 && 
                       y < s2_bands.len_of(Axis(1)) as isize && 
                       x < s2_bands.len_of(Axis(2)) as isize {
                        
                        let y = y as usize;
                        let x = x as usize;
                        
                        // Get and normalize S2 bands
                        for b in 0..10 {
                            let raw_val = s2_bands[(t_in, y, x, b)] as f32;
                            let normalized = (raw_val - S2_BAND_MEAN[b]) / (S2_BAND_STD[b] + 1e-9);
                            aug_output[(py, px, t_out, b)] = normalized;
                        }
                        
                        // Add DOY as the 11th band
                        aug_output[(py, px, t_out, 10)] = s2_doys[t_in] as f32;
                    }
                }
            }
        }
    }
    
    // Combine S1 ascending and descending timesteps
    let mut combined_s1_timesteps = Vec::new();
    for &t in valid_s1_asc_timesteps {
        combined_s1_timesteps.push((true, t)); // true = ascending
    }
    for &t in valid_s1_desc_timesteps {
        combined_s1_timesteps.push((false, t)); // false = descending
    }
    
    let n_valid_s1 = combined_s1_timesteps.len();
    if n_valid_s1 == 0 {
        return Err(anyhow::anyhow!("No valid S1 timesteps for patch"));
    }
    
    // Process S1 data for both augmentations separately
    for (_aug_idx, aug_output) in [&mut s1_a1, &mut s1_a2].iter_mut().enumerate() {
        // Sample timesteps for this augmentation
        let mut selected_s1_timesteps = Vec::new();
        for _ in 0..args.time_steps {
            let idx = rng.gen_range(0..n_valid_s1);
            selected_s1_timesteps.push(combined_s1_timesteps[idx]);
        }
        
        // For each selected timestep
        for (t_out, &(is_asc, t_in)) in selected_s1_timesteps.iter().enumerate() {
            // Extract patch for this timestep
            for py in 0..patch_size {
                let y_offset = py as isize - half_patch as isize;
                let y = center_y as isize + y_offset * effective_stride as isize;
                
                for px in 0..patch_size {
                    let x_offset = px as isize - half_patch as isize;
                    let x = center_x as isize + x_offset * effective_stride as isize;
                    
                    if y >= 0 && x >= 0 {
                        // Convert to usize as we've confirmed the values are non-negative
                        let y = y as usize;
                        let x = x as usize;
                        
                        let (bands, doys) = if is_asc {
                            // Check bounds for ascending data
                            if y >= s1_asc_bands.len_of(Axis(1)) || x >= s1_asc_bands.len_of(Axis(2)) {
                                continue;
                            }
                            (&s1_asc_bands, &s1_asc_doys)
                        } else {
                            // Check bounds for descending data
                            if y >= s1_desc_bands.len_of(Axis(1)) || x >= s1_desc_bands.len_of(Axis(2)) {
                                continue;
                            }
                            (&s1_desc_bands, &s1_desc_doys)
                        };
                        
                        // Get and normalize S1 bands (VV, VH)
                        for b in 0..2 {
                            let raw_val = bands[(t_in, y, x, b)] as f32;
                            let normalized = (raw_val - S1_BAND_MEAN[b]) / (S1_BAND_STD[b] + 1e-9);
                            aug_output[(py, px, t_out, b)] = normalized;
                        }
                        
                        // Add DOY as the 3rd band
                        aug_output[(py, px, t_out, 2)] = doys[t_in] as f32;
                    }
                }
            }
        }
    }
    
    Ok(SampleOut { s2_a1, s2_a2, s1_a1, s1_a2 })
}

/// Stack multiple 4D arrays (patch_size, patch_size, time_steps, channels) into a 5D array
/// (n_patches, patch_size, patch_size, time_steps, channels)
fn stack_5d(
    arrays: &Vec<Array4<f32>>, 
    patch_size: usize, 
    patch_size2: usize, 
    time_steps: usize, 
    channels: usize
) -> Result<Array5<f32>> {
    if arrays.is_empty() {
        return Ok(Array5::<f32>::zeros((0, 0, 0, 0, 0)));
    }
    
    let n_patches = arrays.len();
    let mut output = Array5::<f32>::zeros((n_patches, patch_size, patch_size2, time_steps, channels));
    
    // Copy each patch into the output array
    for (i, patch) in arrays.iter().enumerate() {
        for y in 0..patch_size {
            for x in 0..patch_size2 {
                for t in 0..time_steps {
                    for c in 0..channels {
                        output[(i, y, x, t, c)] = patch[(y, x, t, c)];
                    }
                }
            }
        }
    }
    
    Ok(output)
}

/// Permute the first dimension of a 5D array in parallel, using Arc + Mutex for thread safety
fn parallel_permute_5d(input: &Array5<f32>, perm: &[usize]) -> Array5<f32> {
    let shape = input.shape();
    let n_patches = shape[0];
    let patch_size = shape[1];
    let patch_size2 = shape[2];
    let time_steps = shape[3];
    let channels = shape[4];
    
    // First create output
    let output = Array5::<f32>::zeros((n_patches, patch_size, patch_size2, time_steps, channels));
    
    // Wrap in Arc+Mutex for thread-safe access
    let output = Arc::new(Mutex::new(output));
    
    // Process in parallel chunks for better performance
    let chunk_size = (n_patches + 15) / 16; // Dividing into ~16 chunks
    
    (0..n_patches).into_par_iter().for_each(|i| {
        let src_idx = perm[i];
        
        // Create a buffer for each pixel's data - this avoids multiple lock acquisitions
        let mut pixel_data = Vec::with_capacity(patch_size * patch_size2 * time_steps * channels);
        
        // First collect all data for this patch
        for y in 0..patch_size {
            for x in 0..patch_size2 {
                for t in 0..time_steps {
                    for c in 0..channels {
                        pixel_data.push(input[(src_idx, y, x, t, c)]);
                    }
                }
            }
        }
        
        // Now lock the output once and update all values
        let mut output_guard = output.lock().unwrap(); 
        let mut idx = 0;
        
        for y in 0..patch_size {
            for x in 0..patch_size2 {
                for t in 0..time_steps {
                    for c in 0..channels {
                        output_guard[(i, y, x, t, c)] = pixel_data[idx];
                        idx += 1;
                    }
                }
            }
        }
    });
    
    // Unwrap the Arc+Mutex to get the result
    Arc::try_unwrap(output).unwrap().into_inner().unwrap()
}