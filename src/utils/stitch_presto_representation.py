import rasterio
from rasterio.merge import merge
from pathlib import Path
import sys

def merge_tiff_files(input_dir: Path, output_file: Path):
    """
    Merges all TIFF files from an input directory into a single output TIFF.

    Args:
        input_dir (Path): The directory containing the source .tif files.
        output_file (Path): The path for the created output .tif file.
    """
    # Step 1: Find all .tif files in the input directory
    print(f"Searching for TIFF files in: {input_dir}")
    
    try:
        # Use .glob() to find files, converting the generator to a list
        tif_files = list(input_dir.glob('*.tif'))
        if not tif_files:
            # If .tif yields nothing, try .tiff as a fallback
            tif_files = list(input_dir.glob('*.tiff'))
    except Exception as e:
        print(f"Error accessing the input directory: {e}", file=sys.stderr)
        sys.exit(1)

    if not tif_files:
        print(f"Error: No .tif or .tiff files found in directory '{input_dir}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(tif_files)} TIFF files to merge.")
    print("First few files found:")
    for f in tif_files[:5]:
        print(f"  - {f.name}")

    # Step 2: Merge the rasters
    # Open all found files to pass to the merge function
    src_files_to_mosaic = []
    for tif in tif_files:
        try:
            src = rasterio.open(tif)
            src_files_to_mosaic.append(src)
        except rasterio.errors.RasterioIOError as e:
            print(f"Warning: Could not open file {tif.name}, skipping. Error: {e}", file=sys.stderr)

    if not src_files_to_mosaic:
        print("Error: Could not open any of the found TIFF files.", file=sys.stderr)
        sys.exit(1)
        
    print(f"\nMerging {len(src_files_to_mosaic)} files... (this may take a while depending on size)")
    
    # The merge function handles the core mosaicking logic
    mosaic, out_trans = merge(src_files_to_mosaic)

    print("Merge complete. Preparing output file...")

    # Step 3: Copy metadata and update for the output file
    # Use the metadata from the first file as a base
    out_meta = src_files_to_mosaic[0].meta.copy()

    # Update the metadata with the new properties of the mosaic
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "compress": "lzw"  # Use LZW compression for a more efficient file
    })

    # Close all the source files we opened
    for src in src_files_to_mosaic:
        src.close()

    # Step 4: Write the mosaic to the output file
    print(f"Writing mosaic to: {output_file}")
    
    try:
        # Enable BigTIFF for files that are likely to be > 4GB
        # This is the key fix for the "Write failed" error
        with rasterio.open(output_file, "w", **out_meta, BIGTIFF='YES') as dest:
            dest.write(mosaic)
    except Exception as e:
        print(f"Error writing the output file: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nProcess completed successfully! ✨")
    print(f"Merged file saved to: {output_file}")


if __name__ == '__main__':
    # Define the path to the input directory
    input_directory = Path("/mnt/e/Codes/btfm4rs/data/representation/austrian_crop_Presto")
    
    # Define the output file path in the parent directory
    output_directory = input_directory.parent
    output_filename = "austria_Presto_embeddings.tif"
    output_path = output_directory / output_filename
    
    # Check if the input directory exists
    if not input_directory.is_dir():
        print(f"Error: Input directory does not exist: {input_directory}", file=sys.stderr)
        sys.exit(1)
        
    # Run the main function
    merge_tiff_files(input_directory, output_path)