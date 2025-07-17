import os
import numpy as np
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
from osgeo import gdal
from tqdm import tqdm

def process_dem_data(wldas_dir, subset_bounds=None):
    """
    Creates a GDAL Virtual Raster (VRT) from raw DEM tiles and then uses 
    rioxarray.reproject_match to align the DEM to the WLDAS grid, ensuring 
    perfect geospatial alignment. This function returns the full-resolution
    aligned grid.

    Args:
        wldas_dir (str): Path to the directory containing WLDAS NetCDF files.
        subset_bounds (dict, optional): This parameter is unused, but kept for
            API consistency. Clipping is handled post-alignment.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- 1. Find all DEM .flt files ---
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

    # --- 2. Create a Virtual Raster (VRT) from the DEM tiles ---
    print("\n--- 2. Creating virtual raster mosaic (VRT) ---")
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    vrt_path = os.path.join(data_dir, 'dem_mosaic.vrt')
    # The VRT will act as a single, georeferenced file representing all tiles.
    gdal.BuildVRT(vrt_path, dem_files, resolution='highest')
    print(f"VRT mosaic created at: {vrt_path}")

    # --- 3. Load the VRT and the WLDAS target grid ---
    print("\n--- 3. Loading VRT and WLDAS target grid ---")
    try:
        # Load the VRT directly with rioxarray.
        dem_da = rioxarray.open_rasterio(vrt_path)
        # The VRT from raw .flt files doesn't have an embedded CRS, so we must set it.
        # MERIT DEM is in WGS84 (EPSG:4326).
        dem_da = dem_da.rio.write_crs("EPSG:4326")

        # Load the WLDAS target grid
        wldas_files = sorted([f for f in os.listdir(wldas_dir) if f.endswith('.nc')])
        sample_nc_path = os.path.join(wldas_dir, wldas_files[0])
        wldas_target_grid = xr.open_dataset(sample_nc_path)
        # Ensure the target grid has its CRS defined, using write_crs() as recommended
        if wldas_target_grid.rio.crs is None:
            wldas_target_grid = wldas_target_grid.rio.write_crs("EPSG:4326")
            
    except Exception as e:
        raise IOError(f"Error loading VRT or WLDAS file: {e}")
    print(f"Loaded target grid from: {os.path.basename(sample_nc_path)}")
    
    # --- 4. Align DEM to WLDAS Grid using reproject_match ---
    print("\n--- 4. Aligning DEM to WLDAS grid using reproject_match ---")
    # This single command performs cropping, reprojection, and resampling.
    dem_aligned_da = dem_da.rio.reproject_match(
        wldas_target_grid, 
        resampling=Resampling.bilinear
    )
    print("Alignment complete.")

    # Squeeze the 'band' dimension which is 1
    final_elevation_grid = dem_aligned_da.squeeze().values
    final_elevation_grid = np.nan_to_num(final_elevation_grid, nan=0.0)

    # Extract boundaries from the aligned grid for metadata
    lat_min = dem_aligned_da.y.min().item()
    lat_max = dem_aligned_da.y.max().item()
    lon_min = dem_aligned_da.x.min().item()
    lon_max = dem_aligned_da.x.max().item()
    
    print("\n--- Verification ---")
    print(f"Full-resolution aligned DEM grid shape: {final_elevation_grid.shape}")
    print(f"Final bounds: Lat ({lat_min:.3f}, {lat_max:.3f}), Lon ({lon_min:.3f}, {lon_max:.3f})")
    
    # Per user request, apply a final vertical flip to the DEM grid to ensure
    # its orientation matches the raw WLDAS data's south-up orientation.
    final_elevation_grid = final_elevation_grid[::-1, :].copy()
    print("Applying final vertical flip to DEM grid.")

    return final_elevation_grid, lat_min, lat_max, lon_min, lon_max

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wldas_path = os.path.join(script_dir, 'WLDAS')
    if os.path.exists(wldas_path):
        # Example with subsetting enabled
        subset_config = {
            'enabled': True,
            'lat_min': 30,
            'lat_max': 40,
            'lon_min': -100,
            'lon_max': -80
        }
        print("\n--- Running standalone test with SUBSETTING ---")
        process_dem_data(wldas_path, subset_bounds=subset_config)

        print("\n\n--- Running standalone test with FULL EXTENT ---")
        process_dem_data(wldas_path, subset_bounds={'enabled': False})
    else:
        print(f"WLDAS directory not found at {wldas_path}. Cannot run standalone test.")
    
    