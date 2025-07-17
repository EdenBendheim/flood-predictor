import os
import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
import rioxarray
from tqdm import tqdm

def create_target_grid(manifest_path, resolution=0.01):
    """
    Creates a target xarray.DataArray grid based on the fire manifest.
    This function is a lightweight version of the one in read_dem_fire.py,
    used to ensure the grid is identical.
    """
    manifest_df = pd.read_csv(manifest_path)
    geometries = gpd.GeoSeries.from_wkt(manifest_df['final_geometry_wkt'])
    all_fires_gdf = gpd.GeoDataFrame(manifest_df, geometry=geometries, crs="EPSG:4326")
    
    total_bounds = all_fires_gdf.total_bounds
    lon_min, lat_min, lon_max, lat_max = total_bounds
    
    buffer = resolution * 5
    lat_min, lat_max = lat_min - buffer, lat_max + buffer
    lon_min, lon_max = lon_min - buffer, lon_max + buffer

    target_grid_lats = np.arange(lat_max, lat_min, -resolution)
    target_grid_lons = np.arange(lon_min, lon_max, resolution)
    target_grid = xr.DataArray(
        np.zeros((len(target_grid_lats), len(target_grid_lons))),
        dims=('y', 'x'),
        coords={'y': target_grid_lats, 'x': target_grid_lons}
    )
    target_grid.rio.write_crs("EPSG:4326", inplace=True)
    return target_grid

def get_era5_data(manifest_path, output_dir, start_year, end_year):
    """
    Downloads, processes, and aligns ERA5 weather data to the fire-specific grid.

    Args:
        manifest_path (str): Path to the fire_manifest.csv file.
        output_dir (str): Directory to save the processed NetCDF files.
        start_year (int): The first year of data to process.
        end_year (int): The last year of data to process.
    """
    print("--- 1. Creating target grid from fire manifest ---")
    target_grid = create_target_grid(manifest_path)
    print(f"Target grid created with shape: {target_grid.shape}")

    print("\n--- 2. Accessing ERA5 Zarr store ---")
    # Path to the full ERA5 dataset on Google Cloud
    gcs_path = 'gs://weatherbench2/datasets/era5/1959-2022-full_37-6h-0p25deg_derived.zarr'
    
    try:
        # Open the remote Zarr store directly with lazy loading
        era5_ds = xr.open_zarr(gcs_path, consolidated=True)
        print("Successfully opened ERA5 Zarr store.")
    except Exception as e:
        raise IOError(f"Could not open Zarr store. Ensure gcsfs is installed and you have access. Error: {e}")

    # --- Rename coordinates to match rioxarray expectations if needed ---
    if 'latitude' in era5_ds.coords and 'longitude' in era5_ds.coords:
        era5_ds = era5_ds.rename({'latitude': 'y', 'longitude': 'x'})

    # --- Select variables and set spatial dimensions ---
    variables = ['u_component_of_wind', 'v_component_of_wind', '2m_temperature', '2m_dewpoint_temperature']
    era5_subset = era5_ds[variables]
    era5_subset = era5_subset.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
    era5_subset = era5_subset.rio.write_crs("EPSG:4326", inplace=True)
    
    os.makedirs(output_dir, exist_ok=True)

    # --- 3. Process data year by year ---
    for year in tqdm(range(start_year, end_year + 1), desc="Processing yearly ERA5 data"):
        output_path = os.path.join(output_dir, f'era5_aligned_{year}.nc')
        if os.path.exists(output_path):
            print(f"Year {year} already processed. Skipping.")
            continue

        print(f"\n--- Processing year: {year} ---")
        
        # Select the data for the current year
        yearly_data = era5_subset.sel(time=str(year))
        
        print("Aligning yearly data to target grid...")
        # Use reproject_match to align the data. This handles resampling and cropping.
        # This is a memory-intensive step.
        aligned_data = yearly_data.rio.reproject_match(target_grid)
        
        # Calculate Relative Humidity
        # Formula uses temperature and dewpoint in Celsius
        T = aligned_data['2m_temperature'] - 273.15
        Td = aligned_data['2m_dewpoint_temperature'] - 273.15
        rh = 100 * (np.exp((17.625 * Td) / (243.04 + Td)) / np.exp((17.625 * T) / (243.04 + T)))
        
        # Add RH to the dataset as a new variable
        aligned_data['relative_humidity'] = rh
        
        # Drop dewpoint temp as it's no longer needed
        aligned_data = aligned_data.drop_vars(['2m_dewpoint_temperature'])

        print(f"Saving aligned data for {year} to {output_path}...")
        # Save to NetCDF
        aligned_data.to_netcdf(output_path)
        print("Save complete.")

    print("\n--- All years processed successfully. ---")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    manifest = os.path.join(script_dir, 'data', 'fire_manifest.csv')
    output = os.path.join(script_dir, 'data', 'era5_aligned')
    
    # We only need data for the years covered by our fire manifest
    # In a real scenario, you'd parse this from the manifest.
    # For now, we'll hardcode the known range of the FEDS data.
    START_YEAR = 2012
    END_YEAR = 2023

    if not os.path.exists(manifest):
        print(f"Error: Fire manifest not found at {manifest}")
    else:
        get_era5_data(manifest, output, START_YEAR, END_YEAR) 