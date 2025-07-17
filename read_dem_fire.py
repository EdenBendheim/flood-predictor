import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
from osgeo import gdal
from tqdm import tqdm

def process_fire_dem_data(manifest_path, resolution=0.01):
    """
    Creates a GDAL Virtual Raster (VRT) from raw DEM tiles and then uses
    a custom target grid derived from the fire manifest to create a
    DEM perfectly aligned with the fire data's spatial extent.

    Args:
        manifest_path (str): Path to the fire_manifest.csv file.
        resolution (float): The target resolution in degrees for the output grid.
    
    Returns:
        The aligned DEM grid as a numpy array and its boundaries.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- 1. Find all DEM .flt files (remains the same) ---
    print("--- 1. Finding DEM tiles ---")
    dem_files = []
    for item in os.listdir(script_dir):
        full_path = os.path.join(script_dir, item)
        if os.path.isdir(full_path) and 'dem' in item.lower():
            for root, _, files in os.walk(full_path):
                for file in files:
                    if file.endswith('.flt'):
                        dem_files.append(os.path.join(root, file))
    
    if not dem_files:
        raise FileNotFoundError(f"No DEM .flt files found in subdirectories of {script_dir}")
    print(f"Found {len(dem_files)} DEM tiles.")

    # --- 2. Create a Virtual Raster (VRT) from the DEM tiles (remains the same) ---
    print("\n--- 2. Creating virtual raster mosaic (VRT) ---")
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    vrt_path = os.path.join(data_dir, 'dem_mosaic.vrt')
    gdal.BuildVRT(vrt_path, dem_files, resolution='highest')
    print(f"VRT mosaic created at: {vrt_path}")

    # --- 3. Load VRT and Create Target Grid from Fire Manifest ---
    print("\n--- 3. Creating target grid from fire manifest ---")
    try:
        # Load the VRT with rioxarray. This will load the full dataset into memory.
        dem_da = rioxarray.open_rasterio(vrt_path)
        dem_da = dem_da.rio.write_crs("EPSG:4326") # MERIT DEM is in WGS84

        # Load the fire manifest
        manifest_df = pd.read_csv(manifest_path)
        # Convert WKT strings to Shapely geometries
        geometries = gpd.GeoSeries.from_wkt(manifest_df['final_geometry_wkt'])
        all_fires_gdf = gpd.GeoDataFrame(manifest_df, geometry=geometries, crs="EPSG:4326")
        
        # Get the total bounds of all fires
        total_bounds = all_fires_gdf.total_bounds
        lon_min, lat_min, lon_max, lat_max = total_bounds
        
        # Add a small buffer to the bounds
        buffer = resolution * 5
        lat_min, lat_max = lat_min - buffer, lat_max + buffer
        lon_min, lon_max = lon_min - buffer, lon_max + buffer

        print(f"Target grid bounds derived from fire data: Lat({lat_min:.3f}, {lat_max:.3f}), Lon({lon_min:.3f}, {lon_max:.3f})")

        # Create the target grid as an empty xarray.DataArray
        target_grid_lats = np.arange(lat_max, lat_min, -resolution)
        target_grid_lons = np.arange(lon_min, lon_max, resolution)
        target_grid = xr.DataArray(
            np.zeros((len(target_grid_lats), len(target_grid_lons))),
            dims=('y', 'x'),
            coords={'y': target_grid_lats, 'x': target_grid_lons}
        )
        target_grid.rio.write_crs("EPSG:4326", inplace=True)
        print(f"Created target grid with shape {target_grid.shape}")

    except Exception as e:
        raise IOError(f"Error loading VRT or creating target grid: {e}")
    
    # --- 4. Align DEM to the new Fire-based Grid ---
    print("\n--- 4. Aligning DEM to fire data grid using reproject_match ---")
    dem_aligned_da = dem_da.rio.reproject_match(
        target_grid, 
        resampling=Resampling.bilinear
    )
    print("Alignment complete.")

    # Squeeze the 'band' dimension which is 1
    final_elevation_grid = dem_aligned_da.squeeze().values
    final_elevation_grid = np.nan_to_num(final_elevation_grid, nan=0.0)

    # Extract final boundaries from the aligned grid for metadata
    final_lat_min = dem_aligned_da.y.min().item()
    final_lat_max = dem_aligned_da.y.max().item()
    final_lon_min = dem_aligned_da.x.min().item()
    final_lon_max = dem_aligned_da.x.max().item()
    
    print("\n--- Verification ---")
    print(f"Final aligned DEM grid shape: {final_elevation_grid.shape}")
    print(f"Final bounds: Lat ({final_lat_min:.3f}, {final_lat_max:.3f}), Lon ({final_lon_min:.3f}, {final_lon_max:.3f})")
    
    # The vertical flip is no longer needed as we control the grid definition.

    return final_elevation_grid, final_lat_min, final_lat_max, final_lon_min, final_lon_max

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    manifest_path = os.path.join(script_dir, 'data', 'fire_manifest.csv')
    
    if os.path.exists(manifest_path):
        print("\n--- Running standalone test for process_fire_dem_data ---")
        process_fire_dem_data(manifest_path, resolution=0.01)
    else:
        print(f"Fire manifest not found at {manifest_path}. Cannot run standalone test.")
        print("Please run process_fire_data.py first.")
    
    