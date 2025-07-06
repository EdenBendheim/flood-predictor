import os
import re
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom

def find_dem_files(root_dir):
    """Recursively finds all .flt files in a directory."""
    dem_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.flt'):
                dem_files.append(os.path.join(root, file))
    return dem_files

def parse_filename(filename):
    """Parses latitude and longitude from standard DEM filenames."""
    base = os.path.basename(filename)
    match = re.search(r'n(\d+)w(\d+)', base)
    if match:
        lat = int(match.group(1))
        lon = -int(match.group(2))
        return lat, lon
    raise ValueError(f"Could not parse filename for lat/lon: {base}")

def process_dem_data():
    """
    Finds all DEM tiles, stitches them into a single large grid,
    aggregates them into a lower-resolution grid with multiple channels (mean, min, max),
    and returns the aggregated grid along with the geographic boundaries.
    """
    base_dem_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Scanning for DEM directories in: {base_dem_dir}")
    
    all_dem_files = []
    for item in os.listdir(base_dem_dir):
        full_path = os.path.join(base_dem_dir, item)
        if os.path.isdir(full_path) and 'dem' in item.lower():
            all_dem_files.extend(find_dem_files(full_path))
    
    if not all_dem_files:
        raise FileNotFoundError(f"No DEM .flt files found in subdirectories of {base_dem_dir}")

    print(f"Found {len(all_dem_files)} DEM tiles.")
    
    lats, lons = zip(*[parse_filename(f) for f in all_dem_files])
    unique_lats = sorted(list(set(lats)), reverse=True)
    unique_lons = sorted(list(set(lons)))
    
    lat_map = {lat: i for i, lat in enumerate(unique_lats)}
    lon_map = {lon: i for i, lon in enumerate(unique_lons)}
    
    tile_height, tile_width = 6000, 6000
    n_lat_tiles, n_lon_tiles = len(unique_lats), len(unique_lons)
    full_dem_height = n_lat_tiles * tile_height
    full_dem_width = n_lon_tiles * tile_width
    
    full_dem = np.full((full_dem_height, full_dem_width), np.nan, dtype=np.float32)

    for filepath in tqdm(all_dem_files, desc="Stitching DEM tiles"):
        lat, lon = parse_filename(filepath)
        row_idx, col_idx = lat_map[lat], lon_map[lon]
        y_start, x_start = row_idx * tile_height, col_idx * tile_width
        tile_data = np.fromfile(filepath, dtype=np.float32).reshape((tile_height, tile_width))
        full_dem[y_start:y_start+tile_height, x_start:x_start+tile_width] = tile_data

    full_dem[full_dem == -9999] = np.nan

    block_size = 20
    agg_n_rows = full_dem_height // block_size
    agg_n_cols = full_dem_width // block_size
    trimmed_dem = full_dem[:agg_n_rows * block_size, :agg_n_cols * block_size]
    
    dem_blocks = trimmed_dem.reshape(agg_n_rows, block_size, agg_n_cols, block_size).transpose(0, 2, 1, 3)

    print("Aggregating DEM data...")
    with np.errstate(invalid='ignore', all='ignore'):
        mean_grid = np.nanmean(dem_blocks, axis=(2, 3))
    
    mean_grid = np.nan_to_num(mean_grid, nan=0.0)
    
    # Define geographic boundaries, assuming 10m resolution and tile naming convention
    # This might need adjustment depending on the exact DEM standard.
    lat_max_us = unique_lats[0] + 5 
    lon_min_us = unique_lons[0]
    lat_min_us = unique_lats[-1]
    lon_max_us = unique_lons[-1] + 5

    return mean_grid, lat_min_us, lat_max_us, lon_min_us, lon_max_us

if __name__ == "__main__":
    agg_grid, lat_min, lat_max, lon_min, lon_max = process_dem_data()
    
    n_rows, n_cols, _ = agg_grid.shape
    
    # Example usage of coord_to_agg_grid_index
    lat, lon = 35.0, -120.0  # Example coordinate
    row, col = coord_to_agg_grid_index(lat, lon, lat_min, lat_max, lon_min, lon_max, n_rows, n_cols)
    print(f"Grid index for ({lat}, {lon}): ({row}, {col})")

    # Plot heatmap with increased downsampling for better visualization
    # plot_elevation_heatmap(full_dem, no_data_value=-9999, downsample=2, cmap='terrain', dpi=3000, out_file='us_elevation_heatmap_highres.png')
    
    