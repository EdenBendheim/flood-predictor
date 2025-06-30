import os
import numpy as np
import xarray as xr

# Try to use cupy for GPU acceleration if available
try:
    import cupy as cp
    xp = cp
    use_gpu = True
except ImportError:
    xp = np
    use_gpu = False

# Path to the first WLDAS file for 2012
# Update to use the correct local WLDAS folder
# Relative to this script: '../WLDAS/WLDAS_NOAHMP001_DA1_20120101.D10.nc'
data_path = os.path.join(
    os.path.dirname(__file__),
    '../WLDAS/WLDAS_NOAHMP001_DA1_20120101.D10.nc'
)

ds = xr.open_dataset(data_path)

# Get lat, lon, and variable names
lat = ds['lat'].values
lon = ds['lon'].values
lat_count = lat.shape[0]
lon_count = lon.shape[0]

# List all variables with shape (time, lat, lon)
var_names = [
    v for v in ds.data_vars
    if ds[v].dims == ('time', 'lat', 'lon')
]
var_count = len(var_names)

# Initialize the array with NaNs (use cupy if available)
if use_gpu:
    arr = xp.full((lat_count, lon_count, var_count), xp.nan, dtype=xp.float32)
else:
    arr = xp.full((lat_count, lon_count, var_count), xp.nan, dtype=np.float32)

# Fill the array for the first day (time=0)
for i, v in enumerate(var_names):
    data = ds[v].isel(time=0).values.astype(np.float32)
    # Replace fill values with NaN
    fill_value = ds[v].attrs.get('_FillValue', None)
    if fill_value is not None:
        data = np.where(data == fill_value, np.nan, data)
    if use_gpu:
        data = cp.asarray(data)
    arr[:, :, i] = data

# Optionally, you can keep track of variable names for later use
print(f"Array shape: {arr.shape} (lat, lon, variables)")
print(f"Variables: {var_names}")
print(f"Latitude range: {lat[0]} to {lat[-1]}")
print(f"Longitude range: {lon[0]} to {lon[-1]}")

# The array 'arr' now contains all variables for the first day in 2012, with NaNs for missing data.
