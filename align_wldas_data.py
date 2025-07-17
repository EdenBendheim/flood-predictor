import os
import xarray as xr
import rioxarray
import numpy as np
from tqdm import tqdm
from read_dem_fire import process_fire_dem_data # We need the grid from the DEM script

def align_wldas_to_fire_grid(wldas_dir, manifest_path, output_dir):
    """
    Aligns all WLDAS NetCDF files to the common fire grid.

    The fire grid is derived from the DEM, which is aligned to the total
    extent of all fires in the manifest. This script ensures that WLDAS
    variables are on the exact same grid for use as model features.

    Args:
        wldas_dir (str): Directory containing the raw WLDAS .nc files.
        manifest_path (str): Path to the fire_manifest.csv file.
        output_dir (str): Directory to save the aligned NetCDF files.
    """
    # --- 1. Get the Target Grid from the DEM processing script ---
    print("--- 1. Deriving target grid from fire DEM data ---")
    # We call process_fire_dem_data to get the grid parameters.
    # We don't need the elevation data itself, just the grid definition.
    _, lat_min, lat_max, lon_min, lon_max = process_fire_dem_data(manifest_path)
    
    # Define the target grid using the boundaries from the DEM script.
    # The resolution should match the one used in the DEM script (e.g., 0.01 degrees).
    resolution = 0.01 
    target_grid_lats = np.arange(lat_max, lat_min, -resolution)
    target_grid_lons = np.arange(lon_min, lon_max, resolution)
    
    target_grid = xr.DataArray(
        dims=('y', 'x'),
        coords={'y': target_grid_lats, 'x': target_grid_lons}
    )
    target_grid.rio.write_crs("EPSG:4326", inplace=True)
    
    print(f"Target grid created with shape {target_grid.shape} and bounds:")
    print(f"Lat ({lat_min:.3f}, {lat_max:.3f}), Lon ({lon_min:.3f}, {lon_max:.3f})")

    # --- 2. Find, Reproject, and Save WLDAS data ---
    os.makedirs(output_dir, exist_ok=True)
    wldas_files = [os.path.join(wldas_dir, f) for f in os.listdir(wldas_dir) if f.endswith('.nc')]
    
    if not wldas_files:
        raise FileNotFoundError(f"No .nc files found in {wldas_dir}")
        
    print(f"\n--- 2. Found {len(wldas_files)} WLDAS files to align ---")

    # --- TEMPORARY: Process only the first file for testing ---
    print("\n*** RUNNING IN TEST MODE: PROCESSING ONLY ONE FILE ***")
    wldas_files = wldas_files[:1]
    # ---------------------------------------------------------

    for wldas_path in tqdm(wldas_files, desc="Aligning WLDAS files"):
        try:
            # Open the WLDAS dataset
            wldas_ds = xr.open_dataset(wldas_path, engine="netcdf4")
            
            # Keep only variables with both 'lat' and 'lon' dimensions
            spatial_vars = [v for v in wldas_ds.data_vars if set(['lat', 'lon']).issubset(wldas_ds[v].dims)]
            wldas_ds = wldas_ds[spatial_vars]

            # Set spatial dims for rioxarray
            wldas_ds = wldas_ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')

            # Set the CRS if it's not present. WLDAS is typically WGS84.
            if wldas_ds.rio.crs is None:
                wldas_ds.rio.write_crs("EPSG:4326", inplace=True)

            # Reproject to match the target grid
            wldas_aligned = wldas_ds.rio.reproject_match(target_grid)
            
            # --- 3. Save the aligned data ---
            output_filename = os.path.basename(wldas_path).replace('.nc', '_aligned.nc')
            output_save_path = os.path.join(output_dir, output_filename)
            wldas_aligned.to_netcdf(output_save_path)

        except Exception as e:
            print(f"Could not process file {wldas_path}: {e}")
            continue
            
    print(f"\n--- Alignment complete ---")
    print(f"Aligned WLDAS data saved to: {output_dir}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the raw WLDAS data
    wldas_data_dir = os.path.join(script_dir, 'WLDAS')
    
    # Path to the fire manifest (needed to define the grid)
    manifest_path = os.path.join(script_dir, 'data', 'fire_manifest.csv')
    
    # Directory to save the output files
    output_aligned_dir = os.path.join(script_dir, 'data', 'WLDAS_aligned')
    
    # Check if the required files and directories exist
    if not os.path.exists(wldas_data_dir):
        print(f"Error: WLDAS data directory not found at '{wldas_data_dir}'")
    elif not os.path.exists(manifest_path):
        print(f"Error: Fire manifest not found at '{manifest_path}'.")
        print("This is required to define the target grid. Please run 'process_fire_data.py' first.")
    else:
        align_wldas_to_fire_grid(wldas_data_dir, manifest_path, output_aligned_dir)


        # On the supercomputer
