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
    
    # --- Hardcoded WLDAS Boundaries ---
    wldas_lat_max = 40.0
    wldas_lon_min = -110.0
    wldas_lat_min = 25.065
    wldas_lon_max = -89.025

    # --- Filter DEM files to ensure they are within the WLDAS bounding box ---
    filtered_dem_files = []
    for f in all_dem_files:
        try:
            dem_lat, dem_lon = parse_filename(f)
            # DEM filename (e.g., n40w120) marks the NW corner of a 1x1 degree tile.
            # So it covers lat [dem_lat-1, dem_lat] and lon [dem_lon, dem_lon+1]
            tile_lat_max, tile_lon_min = dem_lat, dem_lon
            tile_lat_min, tile_lon_max = dem_lat - 1, dem_lon + 1
            
            # Basic overlap check
            if (tile_lon_min < wldas_lon_max and tile_lon_max > wldas_lon_min and
                tile_lat_min < wldas_lat_max and tile_lat_max > wldas_lat_min):
                filtered_dem_files.append(f)
        except ValueError:
            continue
    
    if not filtered_dem_files:
        raise FileNotFoundError("No DEM tiles found that overlap with the WLDAS coordinate range.")
    all_dem_files = filtered_dem_files
    print(f"Filtered to {len(all_dem_files)} tiles overlapping with WLDAS area.")

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

    # --- Crop the stitched DEM to WLDAS boundaries BEFORE aggregation ---
    stitched_lat_max = unique_lats[0]
    stitched_lon_min = unique_lons[0]

    # Calculate pixel indices for cropping
    y_start_crop = int(round((stitched_lat_max - wldas_lat_max) * tile_height))
    y_end_crop = int(round((stitched_lat_max - wldas_lat_min) * tile_height))
    x_start_crop = int(round((wldas_lon_min - stitched_lon_min) * tile_width))
    x_end_crop = int(round((wldas_lon_max - stitched_lon_min) * tile_width))

    # Clamp to the dimensions of the stitched array
    y_start_crop = max(0, y_start_crop)
    y_end_crop = min(full_dem_height, y_end_crop)
    x_start_crop = max(0, x_start_crop)
    x_end_crop = min(full_dem_width, x_end_crop)
    
    full_dem = full_dem[y_start_crop:y_end_crop, x_start_crop:x_end_crop]
    print(f"Cropped full DEM grid to shape {full_dem.shape} based on WLDAS coordinates.")

    block_size = 11
    agg_n_rows = full_dem.shape[0] // block_size
    agg_n_cols = full_dem.shape[1] // block_size
    trimmed_dem = full_dem[:agg_n_rows * block_size, :agg_n_cols * block_size]
    
    dem_blocks = trimmed_dem.reshape(agg_n_rows, block_size, agg_n_cols, block_size).transpose(0, 2, 1, 3)

    print("Aggregating DEM data...")
    with np.errstate(invalid='ignore', all='ignore'):
        mean_grid = np.nanmean(dem_blocks, axis=(2, 3))
    
    mean_grid = np.nan_to_num(mean_grid, nan=0.0)
    
    # Define geographic boundaries, which now correspond to the WLDAS coordinates used for the crop
    lat_max_us = wldas_lat_max
    lon_min_us = wldas_lon_min
    lat_min_us = wldas_lat_min
    lon_max_us = wldas_lon_max
    print(f"Final grid boundaries set to WLDAS (Lat, Lon): ({lat_min_us}, {lat_max_us}), ({lon_min_us}, {lon_max_us})")
    print(f"mean_grid.shape: {mean_grid.shape}")    
    return mean_grid, lat_min_us, lat_max_us, lon_min_us, lon_max_us

if __name__ == "__main__":
    agg_grid, lat_min, lat_max, lon_min, lon_max = process_dem_data()
    
    n_rows, n_cols = agg_grid.shape
    
    # This function is not defined in the file, so it will raise an error.
    # It seems to be a leftover from a previous version. I have commented it out.
    # lat, lon = 35.0, -120.0  # Example coordinate
    # row, col = coord_to_agg_grid_index(lat, lon, lat_min, lat_max, lon_min, lon_max, n_rows, n_cols)
    # print(f"Grid index for ({lat}, {lon}): ({row}, {col})")

    # Plot heatmap with increased downsampling for better visualization
    # plot_elevation_heatmap(full_dem, no_data_value=-9999, downsample=2, cmap='terrain', dpi=3000, out_file='us_elevation_heatmap_highres.png')
    
    