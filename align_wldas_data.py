import os
import xarray as xr
import rioxarray
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads
from tqdm import tqdm
import re
from datetime import datetime, timedelta

def align_wldas_per_fire(wldas_dir, manifest_path, output_base_dir, buffer_km=20):
    """
    For each fire in the manifest, aligns daily WLDAS data to a buffered
    bounding box of the fire's total perimeter.

    Args:
        wldas_dir (str): Directory with raw WLDAS .nc files.
        manifest_path (str): Path to the fire_manifest.csv file.
        output_base_dir (str): Base directory to save the aligned data.
        buffer_km (int): Buffer to add around the fire perimeter in kilometers.
    """
    # --- 1. Load Fire Manifest ---
    print("--- 1. Loading fire manifest ---")
    try:
        manifest_df = pd.read_csv(manifest_path)
        # Convert WKT strings to Shapely geometries
        manifest_df['geometry'] = manifest_df['union_geometry_wkt'].apply(loads)
        fires_gdf = gpd.GeoDataFrame(manifest_df, geometry='geometry', crs="EPSG:4326")
        print(f"Loaded {len(fires_gdf)} fire events.")
    except FileNotFoundError:
        print(f"Error: Manifest file not found at {manifest_path}")
        return

    # --- 2. Scan WLDAS files and create a date-to-path map ---
    print("\n--- 2. Scanning WLDAS directory ---")
    date_to_wldas_path = {}
    for filename in os.listdir(wldas_dir):
        if filename.endswith('.nc'):
            # Extract date from a filename like 'WLDAS_NOAHMP001_DA1_D.A20130601.001.nc'
            match = re.search(r'\.A(\d{8})\.', filename)
            if match:
                date_str = match.group(1)
                file_date = datetime.strptime(date_str, '%Y%m%d').date()
                date_to_wldas_path[file_date] = os.path.join(wldas_dir, filename)
    
    if not date_to_wldas_path:
        print(f"Error: No WLDAS .nc files with parsable dates found in {wldas_dir}")
        return
    print(f"Found {len(date_to_wldas_path)} WLDAS files.")

    # --- 3. Process Each Fire ---
    print("\n--- 3. Processing each fire event ---")
    print("\n*** RUNNING IN SINGLE-FIRE-TEST MODE ***")
    buffer_m = buffer_km * 1000  # Convert buffer from km to meters

    for index, fire in tqdm(fires_gdf.iterrows(), total=fires_gdf.shape[0], desc="Processing fires"):
        fire_id = fire['fire_id']
        fire_output_dir = os.path.join(output_base_dir, str(fire_id))
        os.makedirs(fire_output_dir, exist_ok=True)

        # Project geometry to an equal-area projection (e.g., California Albers) to buffer in meters
        fire_geom_proj = gpd.GeoSeries([fire['geometry']], crs=fires_gdf.crs).to_crs('EPSG:3310')
        buffered_geom_proj = fire_geom_proj.buffer(buffer_m)
        
        # Get the bounding box of the buffered geometry for clipping
        clip_box = buffered_geom_proj.total_bounds

        # Iterate through the duration of the fire
        start_date = datetime.fromordinal(fire['start_t']).date()
        end_date = datetime.fromordinal(fire['end_t']).date()
        current_date = start_date

        while current_date <= end_date:
            if current_date in date_to_wldas_path:
                wldas_path = date_to_wldas_path[current_date]
                output_filename = f"{fire_id}_{current_date.strftime('%Y%m%d')}_aligned.nc"
                output_save_path = os.path.join(fire_output_dir, output_filename)

                try:
                    # Open WLDAS data
                    wldas_ds = xr.open_dataset(wldas_path, engine="netcdf4")
                    wldas_ds = wldas_ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat').rio.write_crs("EPSG:4326")

                    # Clip the data to the fire's buffered bounding box
                    # We must project the dataset to the same CRS as the clipping box
                    wldas_clipped = wldas_ds.rio.reproject('EPSG:3310').rio.clip_box(*clip_box)
                    
                    # Save the clipped data
                    wldas_clipped.to_netcdf(output_save_path)

                except Exception as e:
                    print(f"\nError processing {wldas_path} for fire {fire_id}: {e}")
            
            current_date += timedelta(days=1)
        break # Stop after the first fire

    print(f"\n--- Alignment complete ---")
    print(f"Per-fire aligned WLDAS data saved in: {output_base_dir}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    wldas_data_dir = os.path.join(script_dir, 'WLDAS')
    manifest_path = os.path.join(script_dir, 'data', 'fire_manifest.csv')
    output_aligned_dir = os.path.join(script_dir, 'data', 'WLDAS_aligned_per_fire')
    
    if not os.path.exists(wldas_data_dir):
        print(f"Error: WLDAS data directory not found at '{wldas_data_dir}'")
    elif not os.path.exists(manifest_path):
        print(f"Error: Fire manifest not found at '{manifest_path}'. Please run 'process_fire_data.py' first.")
    else:
        align_wldas_per_fire(wldas_data_dir, manifest_path, output_aligned_dir, buffer_km=20)