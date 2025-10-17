#!/usr/bin/env python3
"""
Test script to verify memmap efficiency in patch data loading.
"""

import sys
import os
import time
import psutil
import numpy as np
from pathlib import Path

# Add the current directory to Python path
sys.path.append('/maps/zf281/btfm4rs/src/pv_detection')

from patch_data_preprocessing import load_patch_data

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_memmap_efficiency():
    """Test the efficiency of memmap-based patch loading."""
    print("=== Testing Memmap Efficiency ===")
    
    # Test configuration
    base_dir = "/maps/zf281/btfm4rs/data/downstream/pv_detection/roi_1"
    cache_dir = "/tmp/test_patch_cache"
    years = [2018, 2019]  # Test with 2 years
    patch_size = 3
    
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Base directory: {base_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"Years: {years}")
    print(f"Patch size: {patch_size}")
    
    # Test 1: Memory usage during cache creation
    print("\n--- Test 1: Cache Creation ---")
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    start_time = time.time()
    
    try:
        # Load with cache (will create cache if not exists)
        data = load_patch_data(
            base_dir=base_dir,
            patch_size=patch_size,
            years=years,
            use_cache=True,
            cache_dir=cache_dir,
            pos_neg_ratio=0.33
        )
        
        cache_creation_time = time.time() - start_time
        cache_memory = get_memory_usage()
        
        print(f"Cache creation completed in {cache_creation_time:.2f} seconds")
        print(f"Memory usage after cache creation: {cache_memory:.2f} MB")
        print(f"Memory increase: {cache_memory - initial_memory:.2f} MB")
        
        print(f"Loaded data shapes:")
        print(f"  - Patches: {data['patches'].shape}")
        print(f"  - Labels: {data['labels'].shape}")
        print(f"  - Coordinates: {data['coords'].shape}")
        print(f"  - Year indices: {data['year_indices'].shape}")
        
        # Test 2: Memory usage during cached loading
        print("\n--- Test 2: Cached Loading ---")
        
        # Clear data to reset memory
        del data
        
        pre_load_memory = get_memory_usage()
        print(f"Memory before cached load: {pre_load_memory:.2f} MB")
        
        start_time = time.time()
        
        # Load with existing cache
        data = load_patch_data(
            base_dir=base_dir,
            patch_size=patch_size,
            years=years,
            use_cache=True,
            cache_dir=cache_dir,
            pos_neg_ratio=0.33
        )
        
        cached_load_time = time.time() - start_time
        cached_memory = get_memory_usage()
        
        print(f"Cached loading completed in {cached_load_time:.2f} seconds")
        print(f"Memory usage after cached load: {cached_memory:.2f} MB")
        print(f"Memory increase: {cached_memory - pre_load_memory:.2f} MB")
        
        # Test 3: Compare with non-cached loading
        print("\n--- Test 3: Non-cached Loading Comparison ---")
        
        # Clear data and cache
        del data
        
        # Remove cache files for comparison
        import shutil
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        
        pre_nocache_memory = get_memory_usage()
        print(f"Memory before non-cached load: {pre_nocache_memory:.2f} MB")
        
        start_time = time.time()
        
        # Load without cache
        data_nocache = load_patch_data(
            base_dir=base_dir,
            patch_size=patch_size,
            years=years,
            use_cache=False,
            pos_neg_ratio=0.33
        )
        
        nocache_load_time = time.time() - start_time
        nocache_memory = get_memory_usage()
        
        print(f"Non-cached loading completed in {nocache_load_time:.2f} seconds")
        print(f"Memory usage after non-cached load: {nocache_memory:.2f} MB")
        print(f"Memory increase: {nocache_memory - pre_nocache_memory:.2f} MB")
        
        # Summary
        print("\n=== Performance Summary ===")
        print(f"Cache creation time: {cache_creation_time:.2f}s")
        print(f"Cached loading time: {cached_load_time:.2f}s")
        print(f"Non-cached loading time: {nocache_load_time:.2f}s")
        print(f"Speed improvement: {nocache_load_time / cached_load_time:.2f}x")
        
        print(f"\nMemory efficiency:")
        print(f"Cached loading memory: {cached_memory - pre_load_memory:.2f} MB")
        print(f"Non-cached loading memory: {nocache_memory - pre_nocache_memory:.2f} MB")
        
        # Verify data consistency
        print(f"\nData consistency check:")
        print(f"Cached patches shape: {data['patches'].shape}")
        print(f"Non-cached patches shape: {data_nocache['patches'].shape}")
        print(f"Shapes match: {data['patches'].shape == data_nocache['patches'].shape}")
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_memmap_efficiency()
    if success:
        print("\n✅ Memmap efficiency test completed successfully!")
    else:
        print("\n❌ Memmap efficiency test failed!")
        sys.exit(1)