#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_pastis_patch_d-pixel.py

A script to extract data patches from large MGRS tile-based NumPy arrays (bands, masks, SAR ascending, SAR descending)
based on patch metadata given in a GeoJSON (EPSG:2154) format. The extracted patches will be aligned
with the polygons from the metadata, assuming a 10m resolution. Each patch will have a final shape of
(128, 128) in the spatial dimensions, and will be saved along with its time dimension(s).

Additionally:
  - We copy over the DOY (day-of-year) arrays (doys.npy, sar_ascending_doy.npy, sar_descending_doy.npy)
    directly from the tile folder into each patch folder (no change in content).
  - We skip processing (extraction + copying) if the patch folder already has all 7 required files:
    [bands.npy, masks.npy, doys.npy, sar_ascending_doy.npy, sar_ascending.npy, sar_descending_doy.npy, sar_descending.npy].
  - We do not prefix patch_{patch_id}_ for each file, since the patch_id folder itself suffices to distinguish them.

Default paths are:
    --metadata       /maps/zf281/pangaea-bench/data/PASTIS-HD/metadata.geojson
    --data_root      /scratch/zf281/pastis/2019-2020_d_pixel
    --raw_tiff_root  /scratch/zf281/pastis/2019-2020_d_pixel_raw/data_raw
    --out_dir        /maps/zf281/btfm4rs/data/pastis_patch_d-pixel

Usage example:
    /maps/zf281/miniconda3/envs/detectree-env/bin/python generate_pastis_patch_d-pixel.py \
        --n_jobs 4 \
        --use_memmap \
        --verbose

Dependencies:
    - numpy
    - rasterio
    - shapely
    - pyproj
    - json (standard library)
    - logging (standard library)
    - argparse (standard library)
    - multiprocessing (standard library)
    - shutil (to copy files)
"""

import os
import json
import logging
import argparse
import multiprocessing
import shutil

import numpy as np
import rasterio
from rasterio import transform as rtransform
from shapely.geometry import shape
from pyproj import CRS, Transformer

# Global cache (in each process) for tile data to avoid reloading multiple times in the same process.
TILE_ARRAYS_CACHE = {}

def get_args():
    parser = argparse.ArgumentParser(description="Extract 128x128 patches from large tile .npy arrays.")
    parser.add_argument(
        "--metadata",
        type=str,
        default="/maps/zf281/pangaea-bench/data/PASTIS-HD/metadata.geojson",
        help="Path to the metadata GeoJSON file (EPSG:2154 polygons).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/scratch/zf281/pastis/2019-2020_d_pixel",
        help="Path to the folder containing MGRS tile subfolders (e.g., 30UXV, 31TFJ, etc.).",
    )
    parser.add_argument(
        "--raw_tiff_root",
        type=str,
        default="/scratch/zf281/pastis/2019-2020_d_pixel_raw/data_raw",
        help="Path to the folder containing the reference GeoTIFFs for each tile.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/scratch/zf281/pastis/pastis_patch_d-pixel",
        help="Output directory where subfolders for each patch_id will be created.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=128,
        help="Patch size in pixels (height and width). Default: 128."
    )
    parser.add_argument(
        "--use_memmap",
        action="store_true",
        default=True,
        help="If set, load .npy arrays with mmap_mode='r' to save memory. Default: True."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of processes for parallel processing. Default: 1 (no parallel)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set logging level to DEBUG for more detailed output."
    )
    return parser.parse_args()


def load_tile_arrays(tile_path, use_memmap=True):
    """
    Load the necessary arrays from the tile directory:
    - bands.npy (T_s2, 10980, 10980, 10)
    - masks.npy (T_s2, 10980, 10980)
    - sar_ascending.npy (T_s1_asc, 10980, 10980, 2)
    - sar_descending.npy (T_s1_desc, 10980, 10980, 2)

    Also present in tile_path (but we do not read them into memory here):
    - doys.npy
    - sar_ascending_doy.npy
    - sar_descending_doy.npy

    If use_memmap=True, load with mmap_mode='r'.
    Returns a dict with these loaded as numpy arrays or memory-mapped arrays.
    """
    logging.info(f"Loading arrays from: {tile_path}")
    mmap_mode = "r" if use_memmap else None

    arrays = {}
    arrays['bands'] = np.load(os.path.join(tile_path, "bands.npy"), mmap_mode=mmap_mode)
    arrays['masks'] = np.load(os.path.join(tile_path, "masks.npy"), mmap_mode=mmap_mode)
    arrays['sar_asc'] = np.load(os.path.join(tile_path, "sar_ascending.npy"), mmap_mode=mmap_mode)
    arrays['sar_desc'] = np.load(os.path.join(tile_path, "sar_descending.npy"), mmap_mode=mmap_mode)

    logging.info("Arrays loaded successfully (memmap={}).".format(use_memmap))
    return arrays


def geometry_to_pixel_bounds(geometry_2154, tile_tiff_path):
    """
    Convert a (Multi)Polygon geometry in EPSG:2154 to a bounding box in pixel coordinates
    of the tile specified by tile_tiff_path.

    We do this by:
    1) Getting the bounding box (minx, miny, maxx, maxy) from the geometry (shapely).
    2) Transform these corners to the tile's CRS.
    3) Convert the transformed corners to pixel row/col with rasterio.transform.rowcol().
    4) Return the sorted bounding box in pixel coords: (row_min, row_max, col_min, col_max, tile_transform).
    """
    with rasterio.open(tile_tiff_path) as src:
        tile_crs = src.crs
        tile_transform = src.transform  # affine transform for pixel <-> map coords

    # Setup coordinate transformation: EPSG:2154 -> tile CRS
    epsg_2154 = CRS.from_epsg(2154)
    transformer = Transformer.from_crs(epsg_2154, tile_crs, always_xy=True)

    # bounding box in EPSG:2154
    minx, miny, maxx, maxy = geometry_2154.bounds

    # We'll transform the corner points:
    # top-left corner in map coords: (minx, maxy)
    # bottom-right corner: (maxx, miny)
    tl_x, tl_y = transformer.transform(minx, maxy)
    br_x, br_y = transformer.transform(maxx, miny)

    # rowcol expects (x, y)
    row_min, col_min = rtransform.rowcol(tile_transform, tl_x, tl_y)
    row_max, col_max = rtransform.rowcol(tile_transform, br_x, br_y)

    # ensure ordering
    row_min, row_max = sorted([row_min, row_max])
    col_min, col_max = sorted([col_min, col_max])

    return row_min, row_max, col_min, col_max, tile_transform


def get_centered_bounds_on_patchsize(r0, r1, c0, c1, patch_size, max_height, max_width):
    """
    We want a patch of size patch_size x patch_size.
    We'll center the patch on the bounding box center if possible,
    or at least ensure we don't go out of bounds of the full tile array.

    (r0, r1, c0, c1) = bounding box in pixel coords for the polygon
    patch_size: the desired patch dimension (e.g. 128)
    max_height, max_width: the tile's dimension in pixels (e.g. 10980)

    Return (row_start, row_end, col_start, col_end) for slicing the big tile arrays.
    """
    row_center = (r0 + r1) // 2
    col_center = (c0 + c1) // 2

    half_ps = patch_size // 2

    row_start = row_center - half_ps
    row_end = row_start + patch_size
    col_start = col_center - half_ps
    col_end = col_start + patch_size

    # Clip to boundaries
    if row_start < 0:
        row_start = 0
        row_end = patch_size
    if col_start < 0:
        col_start = 0
        col_end = patch_size
    if row_end > max_height:
        row_end = max_height
        row_start = row_end - patch_size
    if col_end > max_width:
        col_end = max_width
        col_start = col_end - patch_size

    # Final check if the patch is still valid (non-negative sizes)
    if row_start < 0 or col_start < 0:
        raise ValueError("Patch cannot be extracted because bounding box is out of tile range.")
    if (row_end - row_start) != patch_size or (col_end - col_start) != patch_size:
        raise ValueError("Patch extraction problem: patch size mismatch after clipping.")

    return row_start, row_end, col_start, col_end


def extract_patch(arrays, row_bounds, col_bounds):
    """
    Given the dictionary of tile arrays (bands, masks, sar_asc, sar_desc)
    and the row/col bounds, extract the sub-arrays.

    row_bounds: (row_start, row_end)
    col_bounds: (col_start, col_end)

    Return a dict of the extracted patches:
      - 'bands': shape (T_s2, patch_size, patch_size, 10)
      - 'masks': shape (T_s2, patch_size, patch_size)
      - 'sar_asc': shape (T_s1_asc, patch_size, patch_size, 2)
      - 'sar_desc': shape (T_s1_desc, patch_size, patch_size, 2)
    """
    rs, re = row_bounds
    cs, ce = col_bounds

    logging.debug(f"Extracting sub-arrays with row slice [{rs}:{re}], col slice [{cs}:{ce}]")

    patch_dict = {}
    patch_dict['bands'] = arrays['bands'][:, rs:re, cs:ce, :]     # (T_s2, 128, 128, 10)
    patch_dict['masks'] = arrays['masks'][:, rs:re, cs:ce]        # (T_s2, 128, 128)
    patch_dict['sar_asc'] = arrays['sar_asc'][:, rs:re, cs:ce, :] # (T_s1_asc, 128, 128, 2)
    patch_dict['sar_desc'] = arrays['sar_desc'][:, rs:re, cs:ce, :]  # (T_s1_desc, 128, 128, 2)

    return patch_dict


def get_tile_arrays_for_tile(tile_str_upper, data_root, use_memmap):
    """
    Retrieve the tile arrays from global cache (TILE_ARRAYS_CACHE).
    If they don't exist, load them (with or without memmap) and store in cache.
    """
    global TILE_ARRAYS_CACHE
    if tile_str_upper in TILE_ARRAYS_CACHE:
        return TILE_ARRAYS_CACHE[tile_str_upper]
    else:
        tile_npy_path = os.path.join(data_root, tile_str_upper)
        if not os.path.isdir(tile_npy_path):
            msg = f"Tile directory {tile_npy_path} does not exist."
            raise FileNotFoundError(msg)

        arrays = load_tile_arrays(tile_npy_path, use_memmap=use_memmap)
        TILE_ARRAYS_CACHE[tile_str_upper] = arrays
        return arrays


def copy_doy_files_if_needed(tile_str_upper, data_root, patch_out_dir):
    """
    Copy the doys.npy, sar_ascending_doy.npy, sar_descending_doy.npy
    from the tile folder (e.g. /scratch/zf281/pastis/2019-2020_d_pixel/30UXV)
    to the patch_out_dir, if they exist.

    No changes in content, just a direct copy.
    """
    tile_path = os.path.join(data_root, tile_str_upper)
    files_to_copy = ["doys.npy", "sar_ascending_doy.npy", "sar_descending_doy.npy"]

    for fname in files_to_copy:
        src = os.path.join(tile_path, fname)
        dst = os.path.join(patch_out_dir, fname)
        if os.path.isfile(src):
            shutil.copyfile(src, dst)
        else:
            logging.warning(f"Tile {tile_str_upper} is missing {fname}. Not copying.")


def all_outputs_exist(patch_out_dir):
    """
    Check if patch_out_dir has all 7 required files:
      - bands.npy
      - masks.npy
      - doys.npy
      - sar_ascending_doy.npy
      - sar_ascending.npy
      - sar_descending_doy.npy
      - sar_descending.npy
    If all exist, return True, else False.
    """
    required_files = [
        "bands.npy",
        "masks.npy",
        "doys.npy",
        "sar_ascending_doy.npy",
        "sar_ascending.npy",
        "sar_descending_doy.npy",
        "sar_descending.npy"
    ]
    for fname in required_files:
        fpath = os.path.join(patch_out_dir, fname)
        if not os.path.isfile(fpath):
            return False
    return True


def process_one_feature(
    idx,
    feat,
    tile_ref_tiff_fallback,
    data_root,
    raw_tiff_root,
    out_dir,
    patch_size,
    use_memmap,
    tile_height=10980,
    tile_width=10980
):
    """
    Process a single feature from the GeoJSON:
    - check if output folder already has all needed files (7 .npy). If yes, skip
    - parse geometry
    - find bounding box in tile pixel coords
    - center a patch_size window
    - load tile arrays
    - extract subarrays
    - save to disk (bands.npy, masks.npy, sar_asc*.npy, etc.)
    - copy doys.npy, sar_ascending_doy.npy, sar_descending_doy.npy from tile folder
    """
    if "properties" not in feat or "geometry" not in feat:
        logging.warning(f"Feature index {idx} missing 'properties' or 'geometry'. Skipping.")
        return

    props = feat["properties"]
    geom = feat["geometry"]

    patch_id = props.get("ID_PATCH", f"no_id_{idx}")
    tile_str = props.get("TILE", None)
    if tile_str is None:
        logging.warning(f"Patch {patch_id} has no TILE info. Skipping.")
        return

    # e.g., 't30uxv' -> '30UXV'
    tile_str_upper = tile_str.upper().lstrip('T')

    patch_out_dir = os.path.join(out_dir, str(patch_id))
    os.makedirs(patch_out_dir, exist_ok=True)

    # --- 1. Check if patch data is already present ---
    if all_outputs_exist(patch_out_dir):
        logging.info(f"Patch {patch_id} already has all outputs. Skipping.")
        return

    # --- 2. Proceed with extraction ---
    logging.info(f"Processing patch {patch_id} (feature index {idx}), tile = {tile_str_upper}")

    # Build shapely geometry
    try:
        patch_geometry_2154 = shape(geom)
    except Exception as ex:
        logging.error(f"Failed to parse geometry for patch {patch_id}: {ex}")
        return

    # Reference TIFF path
    if tile_str_upper not in tile_ref_tiff_fallback:
        logging.error(f"No fallback reference TIFF known for tile {tile_str_upper}. Skipping patch {patch_id}.")
        return

    tiff_path = os.path.join(
        raw_tiff_root,
        tile_str_upper,
        "red",
        tile_ref_tiff_fallback[tile_str_upper]
    )
    if not os.path.exists(tiff_path):
        logging.error(f"Reference TIFF {tiff_path} not found. Skipping patch {patch_id}.")
        return

    # Convert geometry to bounding box in pixel coords
    try:
        r0, r1, c0, c1, tile_transform = geometry_to_pixel_bounds(patch_geometry_2154, tiff_path)
    except Exception as ex:
        logging.error(f"Failed to transform geometry for patch {patch_id}: {ex}")
        return

    logging.debug(
        f"Patch {patch_id} pixel bounding box: row[{r0}:{r1}], col[{c0}:{c1}]"
    )

    # Decide how to extract a patch_size region from that bounding box
    try:
        row_start, row_end, col_start, col_end = get_centered_bounds_on_patchsize(
            r0, r1, c0, c1,
            patch_size,
            tile_height,
            tile_width
        )
    except ValueError as ve:
        logging.warning(f"Skipping patch {patch_id} due to bounding box issue: {ve}")
        return

    # Load tile arrays from global cache or from disk
    try:
        tile_arrays = get_tile_arrays_for_tile(tile_str_upper, data_root, use_memmap)
    except Exception as ex:
        logging.error(f"Failed to load tile arrays for tile {tile_str_upper}, patch {patch_id}: {ex}")
        return

    # Extract the sub-arrays
    try:
        patch_data = extract_patch(tile_arrays, (row_start, row_end), (col_start, col_end))
    except Exception as ex:
        logging.error(f"Failed to extract patch data for patch {patch_id}: {ex}")
        return

    # Save the sub-arrays to patch_out_dir
    # We do NOT prefix with patch_id, as each folder is unique for that patch
    np.save(os.path.join(patch_out_dir, "bands.npy"), patch_data['bands'])
    np.save(os.path.join(patch_out_dir, "masks.npy"), patch_data['masks'])
    np.save(os.path.join(patch_out_dir, "sar_ascending.npy"), patch_data['sar_asc'])
    np.save(os.path.join(patch_out_dir, "sar_descending.npy"), patch_data['sar_desc'])

    # Copy the DOY files from tile folder to patch folder
    try:
        copy_doy_files_if_needed(tile_str_upper, data_root, patch_out_dir)
    except Exception as ex:
        logging.error(f"Failed to copy DOY files for patch {patch_id}: {ex}")
        return

    logging.info(f"Saved patch data (7 files) for patch {patch_id} to {patch_out_dir}")


def main():
    args = get_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    # 1. Read the metadata (GeoJSON)
    logging.info(f"Loading metadata from {args.metadata}")
    with open(args.metadata, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # 2. Prepare to loop over patches
    features = metadata["features"]
    logging.info(f"Found {len(features)} features (patches) in the metadata.")

    # Because all tiles have the same dimension and resolution:
    TILE_HEIGHT = 10980
    TILE_WIDTH = 10980

    # For reading georeference info, define a fallback dictionary for each tile's reference TIFF.
    tile_ref_tiff_fallback = {
        "30UXV": "S2A_30UXV_20190102_0_L2A.tiff",
        "31TFJ": "S2A_31TFJ_20190103_0_L2A.tiff",
        "31TFM": "S2A_31TFM_20190103_0_L2A.tiff",
        "32ULU": "S2A_32ULU_20190103_0_L2A.tiff",
    }

    # Build a list of tasks for each feature
    tasks = []
    for idx, feat in enumerate(features):
        tasks.append((
            idx,
            feat,
            tile_ref_tiff_fallback,
            args.data_root,
            args.raw_tiff_root,
            args.out_dir,
            args.patch_size,
            args.use_memmap,
            TILE_HEIGHT,
            TILE_WIDTH
        ))

    n_jobs = args.n_jobs
    if n_jobs < 1:
        n_jobs = 1

    if n_jobs == 1:
        # Single-process execution
        logging.info("Running single-process (n_jobs=1).")
        for t in tasks:
            process_one_feature(*t)
    else:
        # Multi-process execution
        logging.info(f"Running with multiprocessing (n_jobs={n_jobs}).")
        with multiprocessing.Pool(processes=n_jobs) as pool:
            pool.starmap(process_one_feature, tasks)

    logging.info("All patches processed. Script finished.")


if __name__ == "__main__":
    main()
