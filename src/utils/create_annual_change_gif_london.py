#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
London Region Change Detection with ROI Mask - Optimized Version

This script creates a change detection GIF animation based on yearly representation data,
masking out regions outside the specified ROI boundary.

Optimized for high-performance computing with parallel processing and memory caching.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import imageio.v2 as imageio
from tqdm import tqdm
import rasterio
import shutil
import multiprocessing as mp
from functools import partial

# Global variables for sharing data between processes
base_normalized = None


def load_roi_mask(tiff_path):
    """Load and return the ROI mask from a TIFF file."""
    print(f"Loading ROI mask from {tiff_path}...")
    with rasterio.open(tiff_path) as src:
        mask = src.read(1)  # Read the first band
        # Ensure the mask is binary (1 for valid areas, 0 for invalid)
        mask = mask > 0
    return mask


# Define helper functions at the module level for multiprocessing compatibility
def normalize_chunk(chunk):
    """Normalize a chunk of embedding vectors to unit length."""
    magnitudes = np.sqrt(np.sum(chunk**2, axis=1, keepdims=True))
    return np.divide(chunk, magnitudes, out=np.zeros_like(chunk), where=magnitudes!=0)


def compute_chunk_dot_product(chunk_pair):
    """Compute dot product for a pair of chunks."""
    chunk1, chunk2 = chunk_pair
    return np.sum(chunk1 * chunk2, axis=1)


def normalize_embeddings_parallel(embeddings, n_jobs=-1):
    """Normalize embedding vectors to unit length using parallel processing."""
    h, w, c = embeddings.shape
    
    # Determine number of processes to use (default: use all available)
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    # Limit to a reasonable number to avoid overhead
    n_jobs = min(n_jobs, 64, mp.cpu_count())
    
    # Reshape for easier parallel processing
    reshaped = embeddings.reshape(-1, c)
    
    # Split the data into chunks for parallel processing
    chunk_size = max(1, reshaped.shape[0] // n_jobs)
    chunks = [reshaped[i:i+chunk_size] for i in range(0, reshaped.shape[0], chunk_size)]
    
    # Process chunks in parallel
    with mp.Pool(n_jobs) as pool:
        normalized_chunks = pool.map(normalize_chunk, chunks)
    
    # Recombine normalized chunks
    normalized = np.vstack(normalized_chunks)
    
    # Reshape back to original dimensions
    return normalized.reshape(h, w, c)


def compute_dot_product_parallel(norm1, norm2, n_jobs=-1):
    """Compute dot product between two normalized embedding arrays in parallel."""
    h, w, c = norm1.shape
    
    # Determine number of processes to use
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    # Limit to a reasonable number to avoid overhead
    n_jobs = min(n_jobs, 64, mp.cpu_count())
    
    # Reshape for easier parallel processing
    reshaped1 = norm1.reshape(-1, c)
    reshaped2 = norm2.reshape(-1, c)
    
    # Split the data into chunks for parallel processing
    chunk_size = max(1, reshaped1.shape[0] // n_jobs)
    chunks1 = [reshaped1[i:i+chunk_size] for i in range(0, reshaped1.shape[0], chunk_size)]
    chunks2 = [reshaped2[i:i+chunk_size] for i in range(0, reshaped2.shape[0], chunk_size)]
    
    # Process chunks in parallel
    with mp.Pool(n_jobs) as pool:
        dot_chunks = pool.map(compute_chunk_dot_product, zip(chunks1, chunks2))
    
    # Recombine dot product chunks
    dot_product = np.concatenate(dot_chunks)
    
    # Reshape back to original spatial dimensions
    return dot_product.reshape(h, w)


def compute_change(curr_rep_path, roi_mask, n_jobs=-1):
    """Compute change between base embedding and current year embedding."""
    global base_normalized
    
    print(f"Loading embeddings from {curr_rep_path}...")
    start_time = time.time()
    
    # Load current year embedding
    curr_embedding = np.load(curr_rep_path)
    print(f"Embedding loaded in {time.time() - start_time:.2f} seconds")
    
    # Only process ROI area to save memory and computation
    if roi_mask is not None:
        # Ensure mask has correct shape
        if roi_mask.shape[:2] != curr_embedding.shape[:2]:
            print("Warning: ROI mask shape doesn't match embedding shape. Adjusting...")
            if roi_mask.shape[0] >= curr_embedding.shape[0] and roi_mask.shape[1] >= curr_embedding.shape[1]:
                roi_mask = roi_mask[:curr_embedding.shape[0], :curr_embedding.shape[1]]
            else:
                new_mask = np.zeros(curr_embedding.shape[:2], dtype=bool)
                h, w = min(roi_mask.shape[0], curr_embedding.shape[0]), min(roi_mask.shape[1], curr_embedding.shape[1])
                new_mask[:h, :w] = roi_mask[:h, :w]
                roi_mask = new_mask
    
    # Normalize embeddings (only once for the current year)
    print("Normalizing embeddings...")
    start_time = time.time()
    
    # Current year embedding normalization (base_embedding is already normalized)
    curr_normalized = normalize_embeddings_parallel(curr_embedding, n_jobs=n_jobs)
    print(f"Normalization completed in {time.time() - start_time:.2f} seconds")
    
    # Free up memory
    del curr_embedding
    
    # Compute dot product
    print("Computing dot product...")
    start_time = time.time()
    dot_product = np.sum(base_normalized * curr_normalized, axis=2)  # Simplified dot product
    print(f"Dot product computed in {time.time() - start_time:.2f} seconds")
    
    # Free up memory
    del curr_normalized
    
    # Rescale dot product: multiply by -1 and add 1
    # This converts from range [-1,1] to [0,2] where 0=no change, 2=maximum change
    rescaled = (dot_product * -1) + 1
    
    return rescaled


def generate_change_map(curr_rep_path, roi_mask, output_path, n_jobs=-1, change_threshold=0.5):
    """Generate and save a change map between base embedding and current year."""
    # Extract year from the path
    curr_year = os.path.basename(os.path.dirname(curr_rep_path))
    base_year = os.environ.get('BASE_YEAR', '2017')  # Get from environment or default to 2017
    
    # Compute change
    rescaled_change = compute_change(curr_rep_path, roi_mask, n_jobs=n_jobs)
    
    # Create change mask
    change_mask = rescaled_change > change_threshold
    
    # Combine change mask with ROI mask (only show changes within ROI)
    combined_mask = change_mask
    if roi_mask is not None:
        combined_mask = change_mask & roi_mask
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Create custom colormap: transparent for no change, yellow for change
    colors = [(1, 1, 1, 0), (1, 1, 0, 1)]  # From transparent to yellow
    change_cmap = LinearSegmentedColormap.from_list('change_cmap', colors, N=256)
    
    # Create a colormap for rescaled_change with transparency outside ROI
    viridis = plt.cm.viridis
    viridis_colors = viridis(np.linspace(0, 1, 256))
    viridis_colors[:, 3] = np.linspace(0, 1, 256)  # Set alpha channel
    masked_viridis = LinearSegmentedColormap.from_list('masked_viridis', viridis_colors)
    
    # Create a masked version of rescaled_change, setting non-ROI areas to NaN (will render as transparent)
    masked_change = np.copy(rescaled_change)
    if roi_mask is not None:
        masked_change[~roi_mask] = np.nan
    
    # Plot masked change map
    plt.imshow(masked_change, cmap=masked_viridis, vmin=0, vmax=2)
    
    # Overlay change highlights (yellow) only within valid areas
    plt.imshow(np.ma.masked_where(~combined_mask, combined_mask), cmap=change_cmap, vmin=0, vmax=1)
    
    # Add title and format
    plt.title(f'{base_year}-{curr_year}', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save with transparency
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    # Free up memory
    del rescaled_change, masked_change
    
    return output_path


def process_year(args):
    """Process a single year for parallel execution."""
    year, base_dir, temp_dir, roi_mask, n_jobs = args
    
    # Current year representation path
    curr_rep_path = os.path.join(base_dir, year, "stitched_representation.npy")
    
    # Output frame path
    base_year = os.environ.get('BASE_YEAR', '2017')
    frame_path = os.path.join(temp_dir, f'change_{base_year}_{year}.png')
    
    # Generate change map
    try:
        generate_change_map(curr_rep_path, roi_mask, frame_path, n_jobs=n_jobs)
        return (year, frame_path)
    except Exception as e:
        print(f"Error processing year {year}: {e}")
        return (year, None)


def generate_change_detection_gif(base_dir, output_dir, roi_mask_path, base_year="2017", frame_duration=2, n_jobs=-1):
    """Generate change detection GIF comparing base_year to all other years."""
    global base_normalized  # We'll use a global variable to share the normalized base embedding
    
    # Set base year in environment for child processes
    os.environ['BASE_YEAR'] = base_year
    
    # Create temporary directory for frames
    temp_dir = os.path.join(output_dir, 'change_detection_frames')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Load ROI mask
    start_time = time.time()
    roi_mask = load_roi_mask(roi_mask_path)
    print(f"ROI mask loaded with shape: {roi_mask.shape} in {time.time() - start_time:.2f} seconds")
    
    # Get all year directories (excluding the base year)
    all_years = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) 
                 and d.isdigit() and d != base_year]
    all_years.sort()
    
    # Base year representation path
    base_rep_path = os.path.join(base_dir, base_year, "stitched_representation.npy")
    
    # Load and normalize base year embeddings (cached for all comparisons)
    print(f"Loading base year ({base_year}) embeddings...")
    start_time = time.time()
    base_embedding = np.load(base_rep_path)
    print(f"Base embedding loaded in {time.time() - start_time:.2f} seconds, shape: {base_embedding.shape}")
    
    start_time = time.time()
    print(f"Normalizing base year embeddings...")
    
    # Single-threaded normalization for the base embedding (to avoid multiprocessing issues)
    h, w, c = base_embedding.shape
    reshaped = base_embedding.reshape(-1, c)
    magnitudes = np.sqrt(np.sum(reshaped**2, axis=1, keepdims=True))
    normalized = np.divide(reshaped, magnitudes, out=np.zeros_like(reshaped), where=magnitudes!=0)
    base_normalized = normalized.reshape(h, w, c)
    
    print(f"Base normalization completed in {time.time() - start_time:.2f} seconds")
    
    # Free up memory
    del base_embedding
    
    # Output GIF path
    output_gif = os.path.join(output_dir, f'london_change_detection_{base_year}_to_all.gif')
    
    # Prepare arguments for parallel processing
    process_args = [(year, base_dir, temp_dir, roi_mask, n_jobs//len(all_years)) for year in all_years]
    
    # Generate frames in parallel
    print(f"Generating change detection frames using {base_year} as base year...")
    print(f"Using up to {n_jobs} CPU cores for processing")
    
    # Process each year sequentially to avoid memory issues
    frame_paths = []
    for year_args in tqdm(process_args, desc="Processing years"):
        result = process_year(year_args)
        if result[1] is not None:
            frame_paths.append(result[1])
    
    # Sort frame paths by year
    frame_paths.sort()
    
    # Create GIF animation
    print(f"Creating GIF animation: {output_gif}...")
    
    # Load all images
    images = []
    for frame_path in frame_paths:
        images.append(imageio.imread(frame_path))
    
    # Create GIF with specified frame duration and infinite looping
    imageio.mimsave(output_gif, images, duration=frame_duration*1000, loop=0)
    
    print(f"GIF animation created: {output_gif}")
    print(f"Total GIF duration: {frame_duration * len(images)} seconds")
    
    # Clean up temporary frames
    shutil.rmtree(temp_dir)
    print(f"Temporary frame directory removed: {temp_dir}")
    
    return output_gif


def main():
    # Record start time
    start_time = time.time()
    
    # Define paths
    base_dir = "/scratch/zf281/btfm_representation/london"
    roi_mask_path = os.path.join(base_dir, "London_GLA_Boundary.tiff")
    
    # Set output directory to be the same as the ROI mask
    output_dir = os.path.dirname(roi_mask_path)
    
    # Determine optimal number of jobs based on system
    n_jobs = min(mp.cpu_count(), 128)  # Use up to 128 cores
    print(f"System has {mp.cpu_count()} CPU cores available, using {n_jobs}")
    
    try:
        # Generate change detection GIF
        gif_path = generate_change_detection_gif(base_dir, output_dir, roi_mask_path, n_jobs=n_jobs)
        
        # Print file size information
        gif_size_mb = os.path.getsize(gif_path) / (1024 * 1024)
        print(f"GIF file size: {gif_size_mb:.2f} MB")
        
        # Report total runtime
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
    except Exception as e:
        print(f"Error: {e}")
        
        # Provide guidance if rasterio is missing
        if "No module named 'rasterio'" in str(e):
            print("\nYou need to install rasterio to read TIFF files:")
            print("Run: pip install rasterio")
            print("\nAlternative method using PIL:")
            print("from PIL import Image")
            print("roi_mask = np.array(Image.open('/path/to/mask.tiff')) > 0")


if __name__ == "__main__":
    main()