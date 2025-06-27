import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
import tifffile
import argparse
from matplotlib import cm
from skimage import exposure

def create_timelapse_gif(base_dir, output_gif, downsample_rate=1, dpi=600, 
                         color_enhance=1.0, contrast_enhance=1.0, 
                         use_colormap=None, saturation_boost=1.0):
    """
    Create a timelapse GIF from a series of npy files with enhanced colors.
    
    Args:
        base_dir: Directory containing the year folders
        output_gif: Path to save the output GIF
        downsample_rate: Skip every N pixels (default: 1, no downsampling)
        dpi: DPI for the output images (default: 600)
        color_enhance: Factor to enhance color intensity (default: 1.0)
        contrast_enhance: Factor to enhance contrast (default: 1.0)
        use_colormap: Optional colormap name (e.g., 'viridis', 'plasma', 'jet')
        saturation_boost: Factor to boost color saturation (default: 1.0)
    """
    # Define the ROI mask path
    tiff_path = os.path.join(base_dir, "London_GLA_Boundary.tiff")
    
    # Optimize output resolution settings
    figsize = (8, 6)  # Base size, adjusted based on DPI
    
    # Font size adjustment parameters
    title_fontsize_base = 24
    title_fontsize_scale = 3  # Magnification factor
    
    # Load the ROI mask
    print("Loading ROI mask...")
    roi_mask = tifffile.imread(tiff_path).astype(bool)
    
    # Find all year directories (only those that are digits)
    year_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()])
    
    # Load all npy files and extract the first 3 channels, using memmap mode
    data_list = []
    years = []
    
    for year_dir in year_dirs:
        year = year_dir  # The folder name is the year
        npy_file = os.path.join(base_dir, year_dir, "stitched_representation.npy")
        
        # Check if file exists
        if not os.path.exists(npy_file):
            print(f"Warning: File not found for year {year}: {npy_file}")
            continue
        
        years.append(year)
        
        # Load npy file using memmap mode
        print(f"Loading file for year {year}...")
        data = np.load(npy_file, mmap_mode='r')
        
        # Apply downsampling if needed
        if downsample_rate > 1:
            # Only read the downsampled data
            data_first_3_channels = data[::downsample_rate, ::downsample_rate, :3].copy()
        else:
            # Read the first three channels without downsampling
            data_first_3_channels = data[:, :, :3].copy()
        
        data_list.append(data_first_3_channels)
        
        # Close the memmap connection to free resources
        del data
    
    print(f"Found {len(years)} npy files, years: {', '.join(years)}")
    
    # Apply downsampling to the mask if needed
    if downsample_rate > 1:
        roi_mask = roi_mask[::downsample_rate, ::downsample_rate]
        print(f"Downsampled ROI mask shape: {roi_mask.shape}")
    
    # Check if the ROI mask matches the data dimensions
    if data_list and roi_mask.shape != data_list[0].shape[:2]:
        print(f"Warning: ROI mask shape {roi_mask.shape} does not match data shape {data_list[0].shape[:2]}")
        print("Resizing ROI mask to match data dimensions...")
        
        # Create a resized mask
        h, w = data_list[0].shape[:2]
        resized_mask = np.zeros((h, w), dtype=bool)
        
        # Copy the values that fit within the bounds
        h_min = min(h, roi_mask.shape[0])
        w_min = min(w, roi_mask.shape[1])
        resized_mask[:h_min, :w_min] = roi_mask[:h_min, :w_min]
        
        roi_mask = resized_mask
    
    # Create frames for the GIF
    frames = []
    
    for i, (data, year) in enumerate(zip(data_list, years)):
        print(f"Processing year {year}...")
        
        # Create normalized RGB image
        normalized_data = np.zeros_like(data, dtype=np.float32)
        
        # Calculate min and max values for each channel in the current time step
        time_step_min = np.zeros(3)
        time_step_max = np.zeros(3)
        
        for c in range(3):
            channel_data = data[:, :, c]
            roi_values = channel_data[roi_mask]
            
            if len(roi_values) == 0:
                print(f"  Warning: No data points within ROI for channel {c}")
                time_step_min[c] = 0
                time_step_max[c] = 1
            else:
                # Calculate min and max values within ROI for current time step
                time_step_min[c] = np.min(roi_values)
                time_step_max[c] = np.max(roi_values)
            
            print(f"  Channel {c} - Min: {time_step_min[c]:.4f}, Max: {time_step_max[c]:.4f}")
            
            # Normalize for this time step only
            norm_channel = np.zeros_like(channel_data, dtype=np.float32)
            
            channel_range = time_step_max[c] - time_step_min[c]
            if channel_range > 0:  # Avoid division by zero
                norm_channel[roi_mask] = (channel_data[roi_mask] - time_step_min[c]) / channel_range
            
            normalized_data[:, :, c] = norm_channel
        
        # Apply colormap if specified
        if use_colormap:
            # Create a single channel from the normalized data
            # Using a weighted average for grayscale conversion
            gray_data = 0.299 * normalized_data[:, :, 0] + 0.587 * normalized_data[:, :, 1] + 0.114 * normalized_data[:, :, 2]
            
            # Apply colormap
            cmap = plt.get_cmap(use_colormap)
            colored_data = cmap(gray_data)[:, :, :3]  # Remove alpha channel
            
            # Replace normalized data with colormap data
            normalized_data = colored_data
        else:
            # Apply color enhancement
            if color_enhance != 1.0:
                # Enhance colors by scaling the normalized values and clipping
                enhanced_data = normalized_data * color_enhance
                normalized_data = np.clip(enhanced_data, 0, 1)
            
            # Apply contrast enhancement
            if contrast_enhance != 1.0:
                # Apply contrast stretching to each channel
                for c in range(3):
                    if contrast_enhance > 1.0:
                        # Increase contrast by centering around 0.5 and scaling
                        channel = normalized_data[:, :, c]
                        channel = 0.5 + (channel - 0.5) * contrast_enhance
                        normalized_data[:, :, c] = np.clip(channel, 0, 1)
                    
                # Optional: Apply adaptive histogram equalization for even more contrast
                if contrast_enhance > 1.5:
                    # Convert to LAB color space for better contrast enhancement
                    for c in range(3):
                        channel = normalized_data[:, :, c]
                        # Only apply to non-zero areas (within ROI)
                        mask = channel > 0
                        if np.any(mask):
                            # Use exposure from skimage for better contrast
                            channel[mask] = exposure.equalize_adapthist(
                                channel[mask].reshape(-1, 1), 
                                clip_limit=0.03
                            ).flatten()
                        normalized_data[:, :, c] = channel
            
            # Apply saturation boost if specified
            if saturation_boost > 1.0:
                # Convert RGB to HSV for saturation adjustment
                hsv_data = np.zeros_like(normalized_data)
                
                # Simple RGB to HSV conversion for saturation boosting
                # Calculate value (max of RGB)
                v = np.max(normalized_data, axis=2)
                
                # Calculate saturation
                min_val = np.min(normalized_data, axis=2)
                s = np.zeros_like(v)
                non_zero_v = v > 0
                s[non_zero_v] = (v[non_zero_v] - min_val[non_zero_v]) / v[non_zero_v]
                
                # Boost saturation
                s = np.clip(s * saturation_boost, 0, 1)
                
                # Convert back to RGB (simplified approach)
                for c in range(3):
                    normalized_data[:, :, c] = np.clip(
                        v * (1 + s * (normalized_data[:, :, c] / (v + 1e-10) - 1)), 
                        0, 1
                    )
        
        # Create RGBA image, transparent outside ROI
        rgba = np.zeros((*data.shape[:2], 4))
        rgba[:, :, :3] = normalized_data  # RGB channels
        rgba[:, :, 3] = roi_mask.astype(float)  # Alpha channel (1 inside ROI, 0 outside)
        
        # Create figure with title, high DPI
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(rgba)
        
        # Adjust title font size based on DPI and user-requested scaling
        title_fontsize = max(15, int(title_fontsize_base * title_fontsize_scale * (100/dpi)))
        print(f"  Title font size: {title_fontsize}")
        
        ax.set_title(f"Year: {year}", fontsize=title_fontsize)
        ax.axis('off')
        
        # Save as temporary PNG (with transparency)
        temp_file = os.path.join(base_dir, f"temp_{year}.png")
        plt.savefig(temp_file, transparent=True, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close(fig)
        
        # Load saved image for GIF
        frames.append(Image.open(temp_file))
    
    if not frames:
        print("Error: No frames created. Check file paths and data.")
        return
    
    # Adjust all frames to the same size (based on the first frame)
    first_frame_size = frames[0].size
    print(f"Output image size: {first_frame_size[0]}x{first_frame_size[1]} pixels")
    resized_frames = []
    for frame in frames:
        if frame.size != first_frame_size:
            frame = frame.resize(first_frame_size, Image.LANCZOS)
        resized_frames.append(frame)
    
    # Save as GIF
    print(f"Creating GIF with {len(resized_frames)} frames...")
    resized_frames[0].save(
        output_gif,
        save_all=True,
        append_images=resized_frames[1:],
        duration=1000,  # 1 second per frame
        loop=0,  # Infinite loop
        disposal=2,  # Replace previous frame with each new frame
        optimize=False  # Disable optimization to maintain high quality
    )
    
    # Clean up temporary files
    for year in years:
        temp_file = os.path.join(base_dir, f"temp_{year}.png")
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print(f"GIF created successfully: {output_gif}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a timelapse GIF from npy files with enhanced colors')
    parser.add_argument('--base_dir', type=str, default='london', help='Directory containing the year folders')
    parser.add_argument('--output_gif', type=str, default='london_timelapse.gif', help='Path to save the output GIF')
    parser.add_argument('--downsample_rate', type=int, default=1, help='Skip every N pixels (default: 1, no downsampling)')
    parser.add_argument('--dpi', type=int, default=600, help='DPI for the output images (default: 600)')
    parser.add_argument('--color_enhance', type=float, default=1, help='Color enhancement factor (default: 1.2)')
    parser.add_argument('--contrast_enhance', type=float, default=1, help='Contrast enhancement factor (default: 1.3)')
    parser.add_argument('--saturation_boost', type=float, default=1, help='Saturation boost factor (default: 1.5)')
    parser.add_argument('--colormap', type=str, default=None, 
                        help='Optional colormap (e.g., viridis, plasma, jet). If specified, replaces RGB visualization')
    
    args = parser.parse_args()
    
    create_timelapse_gif(
        args.base_dir, 
        args.output_gif, 
        args.downsample_rate, 
        args.dpi,
        args.color_enhance,
        args.contrast_enhance,
        args.colormap,
        args.saturation_boost
    )