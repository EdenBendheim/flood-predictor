import xarray as xr
import gcsfs

# Path to the ERA5 data in the WeatherBench2 bucket - updated to the new dataset
gcs_path = 'gs://weatherbench2/datasets/era5/1959-2022-full_37-6h-0p25deg_derived.zarr'

# Try opening the zarr store directly with xarray
try:
    # Open the remote Zarr store directly
    ds = xr.open_zarr(gcs_path, consolidated=True)
except Exception as e:
    print(f"Direct zarr opening failed: {e}")
    # Fallback: Create a filesystem object and try alternative approach
    gcs = gcsfs.GCSFileSystem(token='anon')
    mapper = gcs.get_mapper(gcs_path)
    ds = xr.open_dataset(mapper, engine='zarr', consolidated=True)

# Select data for the US for January 1st, 2015 at the 500 hPa level
us_data_slice = ds['u_component_of_wind'].sel(
    time='2015-01-01T00:00:00',
    level=500,
    latitude=slice(50, 24),  # North to South (approx. US)
    longitude=slice(235, 295) # West to East (approx. US)
)

# Load the data into memory
us_wind_data = us_data_slice.load()

print(us_wind_data)