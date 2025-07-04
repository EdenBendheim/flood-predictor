#!/usr/bin/env python3
"""
Merge flood data with WLDAS by adding an 'active_flood' column.
Optimized for A100 GPU and EPYC CPU.
"""

import os
import glob
import pandas as pd
import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy.spatial import cKDTree
import gc

# Try to import cupy for GPU acceleration
try:
    import cupy as cp
    HAS_GPU = True
    print("CuPy available - using GPU acceleration")
except ImportError:
    HAS_GPU = False
    print("CuPy not available, using CPU-only computation")

def load_flood_data(usfd_path):
    """Load and prepare flood data with date ranges."""
    print("Loading USFD flood data...")
    df = pd.read_csv(usfd_path)
    
    # Convert date strings to datetime
    df['DATE_BEGIN'] = pd.to_datetime(df['DATE_BEGIN'], format='%Y%m%d%H%M', errors='coerce')
    df['DATE_END'] = pd.to_datetime(df['DATE_END'], format='%Y%m%d%H%M', errors='coerce')
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['DATE_BEGIN', 'DATE_END', 'LON', 'LAT'])
    
    return df[['DATE_BEGIN', 'DATE_END', 'LON', 'LAT']].copy()

def create_flood_date_hash(flood_df):
    """Create hash table of active floods for each date."""
    print("Creating flood date hash...")
    flood_hash = defaultdict(list)
    
    for _, flood in tqdm(flood_df.iterrows(), total=len(flood_df), desc="Processing flood events"):
        # Generate all dates between begin and end
        current_date = flood['DATE_BEGIN'].date()
        end_date = flood['DATE_END'].date()
        
        while current_date <= end_date:
            flood_hash[current_date].append([flood['LAT'], flood['LON']])  # lat, lon for KDTree
            current_date += timedelta(days=1)
    
    # Convert to numpy arrays and create KDTrees for fast spatial queries
    flood_trees = {}
    for date_key in flood_hash:
        if flood_hash[date_key]:
            coords = np.array(flood_hash[date_key], dtype=np.float32)
            flood_trees[date_key] = cKDTree(coords)
        else:
            flood_trees[date_key] = None
    
    return flood_trees

def get_flood_mask_gpu(flood_coords, wldas_coords, radius_km=10.0):
    """Create flood mask using GPU acceleration."""
    if len(flood_coords) == 0:
        return np.zeros(len(wldas_coords), dtype=bool)
    
    if HAS_GPU:
        try:
            # Convert to GPU arrays
            flood_coords_gpu = cp.array(flood_coords, dtype=cp.float32)
            wldas_coords_gpu = cp.array(wldas_coords, dtype=cp.float32)
            
            # Vectorized distance calculation using broadcasting
            flood_lat = flood_coords_gpu[:, 0][:, None]  # Shape: (n_floods, 1)
            flood_lon = flood_coords_gpu[:, 1][:, None]
            wldas_lat = wldas_coords_gpu[:, 0][None, :]   # Shape: (1, n_wldas)
            wldas_lon = wldas_coords_gpu[:, 1][None, :]
            
            # Haversine distance calculation
            dlat = cp.radians(wldas_lat - flood_lat)
            dlon = cp.radians(wldas_lon - flood_lon)
            
            a = (cp.sin(dlat/2)**2 + 
                 cp.cos(cp.radians(flood_lat)) * cp.cos(cp.radians(wldas_lat)) * 
                 cp.sin(dlon/2)**2)
            
            distances = 2 * 6371.0 * cp.arcsin(cp.sqrt(a))  # Earth radius = 6371 km
            
            # Check if any flood point is within radius
            within_radius = cp.any(distances <= radius_km, axis=0)
            
            return within_radius.get()  # Transfer back to CPU
            
        except Exception as e:
            print(f"GPU processing failed: {e}, falling back to CPU")
    
    # CPU fallback using KDTree (still fast)
    if len(flood_coords) > 0:
        tree = cKDTree(flood_coords)
        # Convert km to degrees (approximate)
        radius_deg = radius_km / 111.0
        indices = tree.query_ball_point(wldas_coords, r=radius_deg)
        return np.array([len(idx) > 0 for idx in indices], dtype=bool)
    else:
        return np.zeros(len(wldas_coords), dtype=bool)

def process_wldas_file(args):
    """Process a single WLDAS file and add flood indicators."""
    wldas_file, flood_trees, output_dir = args
    
    try:
        # Extract date from filename
        filename = os.path.basename(wldas_file)
        date_part = filename.split('_')[3].split('.')[0]
        file_date = datetime.strptime(date_part, '%Y%m%d').date()
        
        # Check if output already exists
        output_file = os.path.join(output_dir, f"WLDAS_{date_part}_with_floods.csv")
        if os.path.exists(output_file):
            return f"Skipped {filename} (already exists)"
        
        # Load WLDAS data using netCDF4
        ds = nc.Dataset(wldas_file, 'r')
        
        # Get coordinate arrays
        lat = ds.variables["lat"][:].astype(np.float32)
        lon = ds.variables["lon"][:].astype(np.float32)
        
        # Create coordinate meshgrids
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Flatten coordinates for processing
        lat_flat = lat_grid.flatten()
        lon_flat = lon_grid.flatten()
        wldas_coords = np.column_stack([lat_flat, lon_flat])  # lat, lon order
        
        # Get floods for this date
        flood_tree = flood_trees.get(file_date, None)
        
        # Create flood mask
        if flood_tree is not None:
            # Get flood coordinates
            flood_coords = flood_tree.data
            flood_mask = get_flood_mask_gpu(flood_coords, wldas_coords)
        else:
            flood_mask = np.zeros(len(wldas_coords), dtype=bool)
        
        # Load WLDAS variables
        wldas_vars = {}
        for var_name in ds.variables:
            if var_name in ['lat', 'lon', 'time', 'time_bnds']:
                continue
            
            try:
                var_data = ds.variables[var_name][:]
                if len(var_data.shape) == 3:  # time, lat, lon
                    wldas_vars[var_name] = var_data[0].flatten().astype(np.float32)
                elif len(var_data.shape) == 2:  # lat, lon
                    wldas_vars[var_name] = var_data.flatten().astype(np.float32)
            except Exception:
                continue
        
        # Create output DataFrame
        output_data = {
            'date': np.full(len(wldas_coords), str(file_date)),
            'lat': lat_flat,
            'lon': lon_flat,
            'active_flood': flood_mask.astype(np.uint8)
        }
        
        # Add WLDAS variables
        output_data.update(wldas_vars)
        
        # Create DataFrame and filter invalid data
        df = pd.DataFrame(output_data)
        
        # Filter out invalid data points (assuming first WLDAS variable for filtering)
        if wldas_vars:
            first_var = list(wldas_vars.keys())[0]
            df = df[df[first_var] != -9999.0]
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        ds.close()
        del output_data, df, wldas_vars
        gc.collect()
        
        flood_count = np.sum(flood_mask)
        return f"Processed {filename} - {flood_count} flood-affected points"
        
    except Exception as e:
        return f"Error processing {filename}: {str(e)}"

def main():
    """Main function to process all WLDAS files."""
    
    # Configuration
    base_dir = "../"
    usfd_path = os.path.join(base_dir, "FloodPredictor", "USFD_v1.0.csv")
    wldas_dir = os.path.join(base_dir, "WLDAS")
    output_dir = os.path.join(base_dir, "FloodPredictor", "WLDAS_with_active_floods")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load flood data
    flood_df = load_flood_data(usfd_path)
    print(f"Loaded {len(flood_df)} flood events")
    
    # Create flood trees for fast spatial queries
    flood_trees = create_flood_date_hash(flood_df)
    print(f"Created spatial indices for {len(flood_trees)} dates")
    
    # Get all WLDAS files
    wldas_files = sorted(glob.glob(os.path.join(wldas_dir, "*.nc")))
    print(f"Found {len(wldas_files)} WLDAS files")
    
    if not wldas_files:
        print("No WLDAS files found! Check the path.")
        return
    
    # Prepare arguments for multiprocessing
    args_list = [(wldas_file, flood_trees, output_dir) for wldas_file in wldas_files]
    
    # Use maximum CPU cores for EPYC processor
    num_processes = min(cpu_count(), 32)
    print(f"Processing with {num_processes} processes...")
    
    # Process files in parallel
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_wldas_file, args_list),
            total=len(args_list),
            desc="Processing WLDAS files",
            unit="file"
        ))
    
    # Print summary
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    
    successful = 0
    errors = 0
    skipped = 0
    
    for result in results:
        if "Error" in result:
            print(f"❌ {result}")
            errors += 1
        elif "Skipped" in result:
            skipped += 1
        else:
            successful += 1
    
    print(f"\n📊 SUMMARY:")
    print(f"✅ Successfully processed: {successful} files")
    print(f"⏭️  Skipped (already exist): {skipped} files") 
    print(f"❌ Errors: {errors} files")
    print(f"📁 Output directory: {output_dir}")
    
    if successful > 0:
        print(f"\n🎉 Success! Created {successful} CSV files with active flood indicators.")

if __name__ == "__main__":
    main()