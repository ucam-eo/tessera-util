#!/usr/bin/env python3
"""
Test script to verify the fixed extra data loading functionality.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the current directory to Python path
sys.path.append('/maps/zf281/btfm4rs/src/pv_detection')

from extra_data_loader import ExtraDataLoader

def test_extra_data_loading():
    """Test the fixed extra data loading functionality."""
    print("=== Testing Fixed Extra Data Loading ===")
    
    # Test configuration
    extra_data_dir = "/maps/zf281/btfm4rs/data/downstream/pv_detection/extra_training_data"
    uk_data_dir = "/maps/zf281/btfm4rs/data/downstream/pv_detection/uk_data"
    years = [2017, 2018, 2019]  # Test with 3 years
    
    print(f"Extra data directory: {extra_data_dir}")
    print(f"UK data directory: {uk_data_dir}")
    print(f"Years: {years}")
    
    try:
        # Initialize the loader
        loader = ExtraDataLoader(extra_data_dir, uk_data_dir)
        print("✅ ExtraDataLoader initialized successfully")
        
        # Test loading all extra data
        print("\n--- Testing load_all_extra_data ---")
        
        extra_data = loader.load_all_extra_data(years)
        
        print(f"Extra data loaded successfully!")
        print(f"Keys in extra_data: {list(extra_data.keys())}")
        
        if 'patches' in extra_data:
            print(f"Patches shape: {extra_data['patches'].shape}")
            print(f"Patches dtype: {extra_data['patches'].dtype}")
        
        if 'labels' in extra_data:
            print(f"Labels shape: {extra_data['labels'].shape}")
            print(f"Labels dtype: {extra_data['labels'].dtype}")
            print(f"Unique labels: {np.unique(extra_data['labels'])}")
        
        if 'coords' in extra_data:
            print(f"Coordinates shape: {extra_data['coords'].shape}")
            print(f"Coordinates dtype: {extra_data['coords'].dtype}")
        
        if 'year_indices' in extra_data:
            print(f"Year indices shape: {extra_data['year_indices'].shape}")
            print(f"Year indices dtype: {extra_data['year_indices'].dtype}")
            print(f"Unique year indices: {np.unique(extra_data['year_indices'])}")
        
        # Test individual file processing
        print("\n--- Testing individual file processing ---")
        
        # List available label files
        label_files = list(Path(extra_data_dir).glob("*_label.tif"))
        print(f"Found {len(label_files)} label files")
        
        if label_files:
            # Test parsing filename
            test_file = label_files[0]
            filename = test_file.name
            print(f"Testing file: {filename}")
            
            parsed_info = loader.parse_label_filename(filename)
            print(f"Parsed info: {parsed_info}")
            
            # Test loading label
            try:
                label = loader.load_label_tif(str(test_file))
                print(f"Label loaded successfully: shape={label.shape}, dtype={label.dtype}")
                print(f"Label unique values: {np.unique(label)}")
            except Exception as e:
                print(f"Error loading label: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_patch_extraction():
    """Test patch extraction functionality."""
    print("\n=== Testing Patch Extraction ===")
    
    try:
        # Create dummy data for testing
        H, W, C = 100, 100, 128
        embeddings = np.random.randint(-128, 127, size=(H, W, C), dtype=np.int8)
        scales = np.random.uniform(0.1, 2.0, size=(H, W)).astype(np.float32)
        
        # Create some test coordinates
        coordinates = [(50, 50), (25, 25), (75, 75)]
        
        print(f"Test embeddings shape: {embeddings.shape}")
        print(f"Test scales shape: {scales.shape}")
        print(f"Test coordinates: {coordinates}")
        
        # Initialize loader
        extra_data_dir = "/maps/zf281/btfm4rs/data/downstream/pv_detection/extra_training_data"
        uk_data_dir = "/maps/zf281/btfm4rs/data/downstream/pv_detection/uk_data"
        loader = ExtraDataLoader(extra_data_dir, uk_data_dir)
        
        # Test patch extraction
        patches_data = loader.extract_patches_from_coordinates(
            embeddings, scales, coordinates, patch_size=3
        )
        
        print(f"Extracted {len(patches_data)} patches")
        
        for i, patch_data in enumerate(patches_data):
            patch = patch_data['patch']
            coord = patch_data['coord']
            print(f"Patch {i}: shape={patch.shape}, coord={coord}, dtype={patch.dtype}")
            print(f"  Min value: {patch.min():.4f}, Max value: {patch.max():.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error during patch extraction testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting extra data loading tests...\n")
    
    success1 = test_patch_extraction()
    success2 = test_extra_data_loading()
    
    if success1 and success2:
        print("\n✅ All extra data loading tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)