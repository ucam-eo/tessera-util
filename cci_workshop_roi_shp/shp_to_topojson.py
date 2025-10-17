#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "geopandas",
#   "topojson",
#   "tqdm",
# ]
# ///
"""
Convert shapefiles to TopoJSON format.

Usage:
    uv run shp_to_topojson.py
"""

import json
import glob
from pathlib import Path
import geopandas as gpd
import topojson as tp
from tqdm import tqdm


def convert_shapefiles_to_topojson(input_dir='.', output_dir='topojson_output'):
    """Convert all shapefiles in a directory to TopoJSON format."""
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all .shp files
    shapefiles = glob.glob(f"{input_dir}/*.shp")
    
    if not shapefiles:
        print("No shapefiles found in the current directory.")
        return
    
    print(f"Found {len(shapefiles)} shapefiles to convert")
    
    # Process each shapefile
    for shp_file in tqdm(shapefiles, desc="Converting shapefiles"):
        try:
            # Read shapefile
            gdf = gpd.read_file(shp_file)
            
            # Get filename without extension
            base_name = Path(shp_file).stem
            
            # Convert to TopoJSON
            # Using quantization to reduce file size while maintaining precision
            topo = tp.Topology(gdf, prequantize=True)
            
            # Output file path
            output_file = output_path / f"{base_name}.topojson"
            
            # Save TopoJSON
            with open(output_file, 'w') as f:
                f.write(topo.to_json())
            
            print(f"✓ Converted: {base_name}")
            
            # Print some basic info about the conversion
            print(f"  - Features: {len(gdf)}")
            print(f"  - CRS: {gdf.crs}")
            
        except Exception as e:
            print(f"✗ Error converting {shp_file}: {str(e)}")
    
    print(f"\nConversion complete! TopoJSON files saved to '{output_dir}' directory.")


def main():
    """Main function."""
    convert_shapefiles_to_topojson()


if __name__ == "__main__":
    main()