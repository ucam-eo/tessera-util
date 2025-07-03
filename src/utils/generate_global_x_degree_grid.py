import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
import matplotlib.pyplot as plt
from shapely.geometry import box
import pyproj
from tqdm import tqdm
import multiprocessing as mp
import concurrent.futures
import glob
import pandas as pd
import random
from functools import partial
import time

def load_countries_shapefile(shapefile_path, exclude_antarctica=True, buffer_distance=3000):
    """
    Load the country boundary shapefile, optionally exclude Antarctica, and add a 3 km buffer.
    """
    countries = gpd.read_file(shapefile_path)
    
    # Filter out continental regions (exclude Antarctica)
    if exclude_antarctica:
        # countries = countries[countries['SOVEREIGNT'] != 'Antarctica']
        countries = countries[countries['NAME'] != 'Antarctica']
    
    # Add buffer - first convert to projected coordinate system for buffering
    # Use Web Mercator (EPSG:3857) for buffering
    countries_proj = countries.to_crs("EPSG:3857")
    countries_buffered = countries_proj.buffer(buffer_distance)
    
    # Create a new GeoDataFrame with the buffer
    countries_with_buffer = gpd.GeoDataFrame(
        countries.drop(columns='geometry'), 
        geometry=countries_buffered,
        crs="EPSG:3857"
    )
    
    # Convert back to WGS84 (EPSG:4326)
    countries_with_buffer = countries_with_buffer.to_crs("EPSG:4326")
    
    print(f"Added {buffer_distance} meters buffer to country boundaries")
    
    return countries_with_buffer

def generate_global_grid(grid_size=2.0, x_min=-180, x_max=180, y_min=-90, y_max=90):
    """
    Generate a global grid of specified size (in degrees).
    """
    # Create grid cells
    grid_cells = []
    
    for x in np.arange(x_min, x_max, grid_size):
        for y in np.arange(y_min, y_max, grid_size):
            # Calculate center coordinates
            center_lon = x + grid_size/2
            center_lat = y + grid_size/2
            
            # Create a rectangular box for the grid cell
            cell = box(x, y, x + grid_size, y + grid_size)
            grid_cells.append({
                'geometry': cell,
                'lon_min': x,
                'lon_max': x + grid_size,
                'lat_min': y,
                'lat_max': y + grid_size,
                'center_lon': center_lon,
                'center_lat': center_lat
            })
    
    # Create GeoDataFrame from grid cells
    grid = gpd.GeoDataFrame(grid_cells, crs="EPSG:4326")
    return grid

def get_utm_zone(longitude, latitude):
    """
    Get the UTM zone for the given longitude and latitude.
    """
    # Calculate UTM zone number
    if longitude >= 180:
        longitude = longitude - 360
    
    zone_number = int((longitude + 180) / 6) + 1
    
    # Determine whether it's in the northern or southern hemisphere
    if latitude >= 0:
        # Northern hemisphere
        epsg = 32600 + zone_number
    else:
        # Southern hemisphere
        epsg = 32700 + zone_number
    
    return f"EPSG:{epsg}"

def create_grid_raster(grid_cell, countries, output_path, resolution=1000):
    """
    Create a raster for the grid cell, where areas intersecting with land = 1, other areas = 0.
    Use custom resolution, default is 1000 meters, all TIFFs have the same dimensions.
    """
    # Extract grid cell attributes
    lon_min = grid_cell['lon_min']
    lon_max = grid_cell['lon_max']
    lat_min = grid_cell['lat_min']
    lat_max = grid_cell['lat_max']
    center_lon = grid_cell['center_lon']
    center_lat = grid_cell['center_lat']
    
    # Create output filename based on grid center coordinates
    filename = f"grid_{center_lon:.2f}_{center_lat:.2f}.tiff"
    output_file = os.path.join(output_path, filename)
    
    # Skip if file already exists
    if os.path.exists(output_file):
        return output_file
    
    # Get appropriate UTM projection
    utm_epsg = get_utm_zone(center_lon, center_lat)
    
    # Create GeoDataFrame for this grid cell
    grid_gdf = gpd.GeoDataFrame([{
        'geometry': box(lon_min, lat_min, lon_max, lat_max),
    }], crs="EPSG:4326")
    
    # Clip country boundaries to the extent of this grid cell
    grid_geom = grid_gdf.geometry.iloc[0]
    countries_in_grid = countries[countries.intersects(grid_geom)]
    
    # Project to UTM
    grid_utm = grid_gdf.to_crs(utm_epsg)
    
    # Calculate grid boundaries in UTM
    bounds_utm = grid_utm.total_bounds
    xmin, ymin, xmax, ymax = bounds_utm
    
    # Calculate width and height using custom resolution
    width = int(round((xmax - xmin) / resolution))
    height = int(round((ymax - ymin) / resolution))
    
    # Print actual grid and resolution information
    print(f"Grid {filename}: Resolution={resolution}m, Size={width}x{height} pixels")
    
    # Create transformation matrix
    transform = from_origin(xmin, ymax, resolution, resolution)
    
    # Rasterize land areas if there is land in this grid cell
    if len(countries_in_grid) > 0:
        countries_utm = countries_in_grid.to_crs(utm_epsg)
        shapes = [(geom, 1) for geom in countries_utm.geometry]
        raster = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=True,  # Ensure all intersecting pixels are included
            dtype=np.uint8
        )
    else:
        # Create an all-zero array if there is no land
        raster = np.zeros((height, width), dtype=np.uint8)
    
    # Write compressed GeoTIFF
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.uint8,
        crs=utm_epsg,
        transform=transform,
        compress='lzw',      # Use LZW compression
        predictor=2,         # Horizontal differencing predictor for better compression
        tiled=True,          # Use tiled storage
        blockxsize=256,      # Optimize block size
        blockysize=256,
        zlevel=9,            # Maximum compression level
        nodata=0             # Set no-data value
    ) as dst:
        dst.write(raster, 1)
    
    return output_file

def process_grid_cell(grid_cell, countries, output_path, resolution=1000):
    """Function to process a single grid cell, used for parallel processing."""
    return create_grid_raster(grid_cell, countries, output_path, resolution)

def main():
    # Define paths
    shapefile_path = "/maps/zf281/btfm4rs/data/global_map_shp/detailed_world_map.shp"
    output_path = "/scratch/zf281/global_map_0.1_degree_tiff"
    
    # Custom resolution setting (unit: meters)
    # Modify this value as needed
    resolution = 10  # Default resolution is 1000 meters
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Load country boundary shapefile and add a 1 km buffer
    print("Loading country boundary shapefile and adding a 1 km buffer...")
    countries = load_countries_shapefile(shapefile_path, buffer_distance=1000)
    
    # Generate global grid
    print("Generating global grid...")
    grid = generate_global_grid(grid_size=0.1)  # 0.1-degree grid is approximately 11 km
    
    # Filter grid cells that intersect with land
    print("Filtering grid cells that intersect with land...")
    land_grid = []
    for _, grid_row in tqdm(grid.iterrows(), total=len(grid), desc="Filtering grid"):
        grid_geom = grid_row.geometry
        if countries.intersects(grid_geom).any():
            land_grid.append({
                'lon_min': grid_row.lon_min,
                'lon_max': grid_row.lon_max,
                'lat_min': grid_row.lat_min,
                'lat_max': grid_row.lat_max,
                'center_lon': grid_row.center_lon,
                'center_lat': grid_row.center_lat
            })
    
    print(f"Found {len(land_grid)} grid cells intersecting with land")
    print(f"Using resolution: {resolution} meters")
    
    # Use multiprocessing for parallel processing
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    # Create raster for each grid cell
    print("Creating raster for grid cells...")
    
    # Create partial function to pass fixed arguments
    process_func = partial(
        process_grid_cell, 
        countries=countries, 
        output_path=output_path, 
        resolution=resolution
    )
    
    # Use process pool for parallel grid processing
    with mp.Pool(processes=num_cores) as pool:
        # Use tqdm to display progress
        list(tqdm(pool.imap(process_func, land_grid), total=len(land_grid), desc="Creating grid TIFFs"))
    
    print("Done!")

if __name__ == "__main__":
    main()