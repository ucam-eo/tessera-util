import os
import numpy as np
import rasterio
from rasterio.transform import Affine

def load_and_dequantize_representation(representation_file_path, scales_file_path):
    """
    Load and dequantize int8 representations back to float32.
    
    Args:
        representation_file_path: Path to the int8 representation file (H,W,C)
        scales_file_path: Path to the float32 scales file (H,W)
    
    Returns:
        representation_f32: float32 ndarray of shape (H,W,C)
    """
    # Load the files
    representation_int8 = np.load(representation_file_path)  # (H, W, C), dtype=int8
    scales = np.load(scales_file_path)  # (H, W), dtype=float32
    
    # Convert int8 to float32 for computation
    representation_f32 = representation_int8.astype(np.float32)
    
    # Expand scales to match representation shape
    # scales shape: (H, W) -> (H, W, 1)
    scales_expanded = scales[..., np.newaxis]
    
    # Dequantize by multiplying with scales
    representation_f32 = representation_f32 * scales_expanded
    
    return representation_f32

def convert_npy_to_tiff(npy_path, scales_path, ref_tiff_path, out_dir, downsample_rate=1):
    # Load and dequantize the int8 data to float32
    data = load_and_dequantize_representation(npy_path, scales_path)
    H, W, C = data.shape
    
    print(f"Loaded data shape: {data.shape}, dtype: {data.dtype}")

    # If downsampling is needed
    if downsample_rate > 1:
        # Calculate the dimensions after downsampling
        new_H = H // downsample_rate
        new_W = W // downsample_rate

        # Create the downsampled array
        downsampled_data = np.zeros((new_H, new_W, C), dtype=np.float32)

        # Perform downsampling using the area averaging method
        for i in range(new_H):
            for j in range(new_W):
                # Calculate the index range of the current block
                i_start = i * downsample_rate
                i_end = min((i + 1) * downsample_rate, H)
                j_start = j * downsample_rate
                j_end = min((j + 1) * downsample_rate, W)

                # Calculate the average value of the current block
                block = data[i_start:i_end, j_start:j_end, :]
                downsampled_data[i, j, :] = np.mean(block, axis=(0, 1))

        # Replace the original data with the downsampled data
        data = downsampled_data
        H, W = new_H, new_W

    # Open the reference tiff file to get spatial reference, affine transform, width, height, etc.
    with rasterio.open(ref_tiff_path) as ref:
        ref_meta = ref.meta.copy()
        # Keep the coordinate system and affine transform information from the reference tiff
        transform = ref.transform
        crs = ref.crs

        # If downsampling is needed, modify the pixel size in the affine transform
        if downsample_rate > 1:
            # Create a new affine transform, increasing the pixel size
            transform = Affine(
                transform.a * downsample_rate, transform.b, transform.c,
                transform.d, transform.e * downsample_rate, transform.f
            )

    # Update metadata, the number of bands for the new tiff is C, data type is float32
    new_meta = ref_meta.copy()
    new_meta.update({
        'driver': 'GTiff',
        'height': H,
        'width': W,
        'count': C,
        'dtype': 'float32',  # Explicitly set to float32
        'transform': transform  # Update affine transform information
    })

    # Construct the output filename, same name as the npy file but with .tif extension
    base_name = os.path.splitext(os.path.basename(npy_path))[0]
    out_path = os.path.join(out_dir, f"{base_name}.tif")

    # Write the new tiff file
    with rasterio.open(out_path, 'w', **new_meta) as dst:
        # Assume the 3rd dimension of the npy data is the band, write data for each band
        for i in range(C):
            dst.write(data[:, :, i], i + 1)
            print(f"Band {i + 1} writing complete")

    print(f"Output file saved as: {out_path}")
    print(f"Resolution: Original 10m, after downsampling {10 * downsample_rate}m")
    print(f"Output data type: float32")

if __name__ == "__main__":
    npy_path = "/scratch/zf281/btfm_representation/borneo/2020/grid_117.85_4.95/grid_117.85_4.95.npy"  # int8 npy file path
    scales_path = "/scratch/zf281/btfm_representation/borneo/2020/grid_117.85_4.95/grid_117.85_4.95_scales.npy"  # scales file path
    ref_tiff_path = "/scratch/zf281/btfm_representation/borneo/2020/grid_117.85_4.95/grid_117.85_4.95.tiff"  # reference tiff file path
    out_dir = "/scratch/zf281/btfm_representation/borneo"  # output directory
    downsample_rate = 1  # Default is no downsampling, modify as needed
    
    convert_npy_to_tiff(npy_path, scales_path, ref_tiff_path, out_dir, downsample_rate)