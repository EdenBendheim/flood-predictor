import numpy as np
import os
import re
from tqdm import tqdm
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

def read_dem_flt(file_path, rows=6000, cols=6000):
    """
    Read the entire DEM FLT file into a numpy array.
    FLT files are typically 32-bit floating point binary files.
    """
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    return data.reshape(rows, cols)

def get_coordinates_from_filename(filename):
    """
    Extract latitude and longitude from the filename.
    Filename format: n{lat}w{lon}_dem.flt
    """
    match = re.match(r'n(\d+)w(\d+)_dem\.flt', filename)
    if match:
        lat = int(match.group(1))
        lon = int(match.group(2))
        return lat, lon
    return None, None

def plot_elevation_heatmap(elevation_array, no_data_value=-9999, downsample=100, cmap='terrain', dpi=300, out_file='us_elevation_heatmap.png'):
    """
    Plot a heatmap of the elevation data using proper map projection.
    """
    # Mask no-data values
    masked = np.ma.masked_where(elevation_array == no_data_value, elevation_array)
    # Downsample for visualization
    masked_ds = masked[::downsample, ::downsample]

    # Calculate a suitable figsize (in inches) for the array shape and dpi
    height, width = masked_ds.shape
    figsize = (width / dpi, height / dpi)

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())

    im = ax.imshow(masked_ds, cmap=cmap, origin='upper',
                   extent=[-124.848974, -66.885444, 24.396308, 49.384358],
                   transform=ccrs.PlateCarree())

    # ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ax.add_feature(cfeature.STATES, linestyle=':')

    plt.colorbar(im, label='Elevation (m)')
    plt.title(f'US Elevation Map (downsampled by {downsample}x)')

    ax.set_extent([-125, -65, 24, 50], crs=ccrs.PlateCarree())

    plt.tight_layout()
    plt.savefig(out_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'Elevation heatmap saved as {out_file}')

def aggregate_dem_blocks(dem, block_size=11, no_data_value=-9999):
    """
    Aggregate DEM into larger blocks (e.g., 1km x 1km) by computing mean, min, and max for each block,
    ignoring no_data_value.
    Returns three arrays: mean_grid, min_grid, max_grid.
    """
    # Replace no_data_value with np.nan for nan-aware operations
    dem = np.where(dem == no_data_value, np.nan, dem)
    rows, cols = dem.shape
    # Trim to a multiple of block_size
    trimmed_rows = (rows // block_size) * block_size
    trimmed_cols = (cols // block_size) * block_size
    dem = dem[:trimmed_rows, :trimmed_cols]
    # Reshape to (n_blocks_row, block_size, n_blocks_col, block_size)
    dem_blocks = dem.reshape(
        trimmed_rows // block_size, block_size,
        trimmed_cols // block_size, block_size
    )
    # Move block axes together for easier aggregation
    dem_blocks = dem_blocks.transpose(0, 2, 1, 3)  # (n_blocks_row, n_blocks_col, block_size, block_size)
    # Now compute the aggregated values (mean, min, max) for each block
    # This might produce NaNs for blocks that had no data (e.g., ocean)
    print("Aggregating DEM blocks...")
    with np.errstate(invalid='ignore', all='ignore'):
        mean_grid = np.nanmean(dem_blocks, axis=(2, 3))
        min_grid = np.nanmin(dem_blocks, axis=(2, 3))
        max_grid = np.nanmax(dem_blocks, axis=(2, 3))

    # Replace any remaining NaNs (from all-NaN blocks) with a fill value, e.g., 0.
    mean_grid = np.nan_to_num(mean_grid, nan=0.0)
    min_grid = np.nan_to_num(min_grid, nan=0.0)
    max_grid = np.nan_to_num(max_grid, nan=0.0)

    # Stack the aggregated grids to form the final multi-channel grid
    agg_grid = np.stack([mean_grid, min_grid, max_grid], axis=-1)
    return mean_grid, min_grid, max_grid

def coord_to_agg_grid_index(lat, lon, lat_min, lat_max, lon_min, lon_max, n_rows, n_cols):
    """
    Convert a (lat, lon) coordinate to (row, col) index in agg_grid.
    Returns (row, col) as integers.
    """
    # Fractional position in the grid
    row_frac = (lat_max - lat) / (lat_max - lat_min)
    col_frac = (lon - lon_min) / (lon_max - lon_min)
    # Convert to indices
    row = int(row_frac * n_rows)
    col = int(col_frac * n_cols)
    # Clamp to grid bounds
    row = min(max(row, 0), n_rows - 1)
    col = min(max(col, 0), n_cols - 1)
    return row, col

def find_dem_files(directory):
    """Find all .flt DEM files in the given directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.flt')]

def parse_filename(filepath):
    """
    Parses the filename to extract latitude and longitude.
    Example: 'n45w110_dem.flt' -> lat=45, lon=-110
    """
    basename = os.path.basename(filepath)
    parts = basename.split('_')[0]
    lat_part = parts[0]
    lon_part = parts[3]
    lat = int(parts[1:3])
    lon = int(parts[4:7])
    if lat_part == 's':
        lat = -lat
    if lon_part == 'w':
        lon = -lon
    return lat, lon

def process_dem_data():
    base_dem_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Scanning DEM directories in {base_dem_dir}...")
    all_dem_files = []
    for item in os.listdir(base_dem_dir):
        full_path = os.path.join(base_dem_dir, item)
        if os.path.isdir(full_path) and 'dem' in item.lower():
            all_dem_files.extend(find_dem_files(full_path))
    
    if not all_dem_files:
        raise FileNotFoundError(f"No DEM .flt files found in subdirectories of {base_dem_dir}")

    # Parse all filenames to find geographic bounds
    lats, lons = zip(*[parse_filename(f) for f in all_dem_files])
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)

    # Assuming each DEM tile is 6000x6000 pixels
    tile_height, tile_width = 6000, 6000
    
    # Get unique sorted latitudes (high to low) and longitudes (low to high)
    unique_lats = sorted(list(set(lats)), reverse=True)
    unique_lons = sorted(list(set(lons)))
    
    print(f"Sorted latitudes (north to south): {unique_lats}")
    print(f"Sorted longitudes (west to east): {unique_lons}")

    # Create a mapping from lat/lon to grid position
    lat_map = {lat: i for i, lat in enumerate(unique_lats)}
    lon_map = {lon: i for i, lon in enumerate(unique_lons)}
    
    n_lat_tiles = len(unique_lats)
    n_lon_tiles = len(unique_lons)
    
    print(f"Grid: {n_lat_tiles} tiles north-south, {n_lon_tiles} tiles east-west")

    # Pre-allocate a large NumPy array to hold the full DEM
    full_dem_height = n_lat_tiles * tile_height
    full_dem_width = n_lon_tiles * tile_width
    
    print(f"Pre-allocated array shape: ({full_dem_height}, {full_dem_width})")
    full_dem = np.full((full_dem_height, full_dem_width), np.nan, dtype=np.float32)
    print(f"Pre-allocated array size: {full_dem.nbytes / 1e9:.2f} GB")


    # Load each DEM tile into the correct position in the large array
    for filepath in tqdm(all_dem_files, desc="Loading DEM tiles"):
        lat, lon = parse_filename(filepath)
        row_idx = lat_map[lat]
        col_idx = lon_map[lon]
        
        y_start = row_idx * tile_height
        y_end = y_start + tile_height
        x_start = col_idx * tile_width
        x_end = x_start + tile_width
        
        # Read the binary float data
        tile_data = np.fromfile(filepath, dtype=np.float32).reshape((tile_height, tile_width))
        full_dem[y_start:y_end, x_start:x_end] = tile_data

    print(f"\nFull DEM shape: {full_dem.shape}")
    
    # Replace -9999 (common no-data value) with NaN
    full_dem[full_dem == -9999] = np.nan
    
    print(f"Min elevation: {np.nanmin(full_dem):.1f} m")
    print(f"Max elevation: {np.nanmax(full_dem):.3f} m")
    print(f"Mean elevation: {np.nanmean(full_dem):.4f} m\n")

    # --- Aggregation Step ---
    block_size = 20  # Aggregate into 20x20 pixel blocks
    n_rows, n_cols = full_dem.shape
    
    # Calculate the new dimensions
    agg_n_rows = n_rows // block_size
    agg_n_cols = n_cols // block_size
    
    # Trim the full_dem to be divisible by block_size
    trimmed_dem = full_dem[:agg_n_rows * block_size, :agg_n_cols * block_size]
    
    # Reshape into blocks
    dem_blocks = trimmed_dem.reshape(agg_n_rows, block_size, agg_n_cols, block_size)
    dem_blocks = dem_blocks.transpose(0, 2, 1, 3) # Shape: (agg_n_rows, agg_n_cols, block_size, block_size)

    # Compute stats, ignoring nan values. This is where warnings can occur for all-NaN blocks (ocean).
    print("Aggregating DEM blocks...")
    with np.errstate(invalid='ignore', all='ignore'): # Suppress expected warnings
        mean_grid = np.nanmean(dem_blocks, axis=(2, 3))
        min_grid = np.nanmin(dem_blocks, axis=(2, 3))
        max_grid = np.nanmax(dem_blocks, axis=(2, 3))

    # Replace any resulting NaNs with 0
    mean_grid = np.nan_to_num(mean_grid, nan=0.0)
    min_grid = np.nan_to_num(min_grid, nan=0.0)
    max_grid = np.nan_to_num(max_grid, nan=0.0)

    # Stack the aggregated grids to form the final multi-channel grid
    agg_grid = np.stack([mean_grid, min_grid, max_grid], axis=-1)

    print(f"Aggregated DEM shape (agg_grid): {agg_grid.shape}")
    print(f"Mean of means: {agg_grid[:,:,0].mean():.5f}")
    print(f"Min of mins: {agg_grid[:,:,1].min():.5f}")
    print(f"Max of maxs: {agg_grid[:,:,2].max():.5f}")
    
    # Define the geographic boundaries of the entire DEM grid
    # Top-left corner of the top-left tile
    lat_max_us = unique_lats[0] + 5 
    lon_min_us = unique_lons[0]

    # Bottom-right corner of the bottom-right tile
    lat_min_us = unique_lats[-1]
    lon_max_us = unique_lons[-1] + 5

    return agg_grid, lat_min_us, lat_max_us, lon_min_us, lon_max_us

if __name__ == "__main__":
    agg_grid, lat_min, lat_max, lon_min, lon_max = process_dem_data()
    
    n_rows, n_cols, _ = agg_grid.shape
    
    # Example usage of coord_to_agg_grid_index
    lat, lon = 35.0, -120.0  # Example coordinate
    row, col = coord_to_agg_grid_index(lat, lon, lat_min, lat_max, lon_min, lon_max, n_rows, n_cols)
    print(f"Grid index for ({lat}, {lon}): ({row}, {col})")

    # Plot heatmap with increased downsampling for better visualization
    # plot_elevation_heatmap(full_dem, no_data_value=-9999, downsample=2, cmap='terrain', dpi=3000, out_file='us_elevation_heatmap_highres.png')
    
    