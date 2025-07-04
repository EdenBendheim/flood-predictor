import os
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
import xarray as xr
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom
from multiprocessing import Pool, cpu_count
from functools import partial

# Assuming read_dem.py is in the same directory or accessible
from read_dem import process_dem_data

# Helper function for parallel processing
# It must be defined at the top level to be pickleable by multiprocessing
def process_single_day(idx, processed_dir, wldas_files, flood_data_path, agg_grid, lat_min_us, lat_max_us, lon_min_us, lon_max_us, n_rows, n_cols, num_nodes, edge_index, edge_weight):
    try:
        # Recreate the necessary parts of the dataset object for one sample
        # This avoids pickling large objects
        
        # --- Load WLDAS data for the window ---
        wldas_window_files = wldas_files[idx:idx+14]
        feature_vars = [
            'Rainf_tavg', 'SoilMoi00_10cm_tavg', 'Snowf_tavg', 'Tair_f_tavg',
            'Evap_tavg', 'SWE_tavg', 'Qs_tavg', 'Qsb_tavg'
        ]
        all_features = []
        for f in wldas_window_files:
            with xr.open_dataset(f) as ds:
                features = ds[feature_vars].to_array(dim="variable").values
                features = features.squeeze().transpose(1, 2, 0)
                reshaped_features = features.reshape(num_nodes, len(feature_vars))
                all_features.append(reshaped_features)
        wldas_tensor = torch.tensor(np.stack(all_features, axis=1), dtype=torch.float)

        # --- Get static elevation features ---
        elevation_features = torch.tensor(agg_grid.reshape(num_nodes, -1), dtype=torch.float)

        # --- Sanitize data ---
        wldas_tensor = torch.nan_to_num(wldas_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        elevation_features = torch.nan_to_num(elevation_features, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Create target label ---
        target_day_idx = idx + 14
        target_date_str = os.path.basename(wldas_files[target_day_idx]).split('_')[-1].split('.')[0]
        
        # Load flood data inside the worker
        flood_df = pd.read_csv(
            flood_data_path, 
            usecols=['DATE_BEGIN', 'DATE_END', 'LAT', 'LON'], 
            dtype=str,
            encoding='latin1'
        )
        
        y = torch.zeros(num_nodes, dtype=torch.float)
        date_part = target_date_str.split('_')[-1]
        target_date = pd.to_datetime(date_part, format='%Y%m%d')

        flood_df['DATE_BEGIN'] = flood_df['DATE_BEGIN'].astype(str).str.replace(r'\.0$', '', regex=True)
        flood_df['DATE_END'] = flood_df['DATE_END'].astype(str).str.replace(r'\.0$', '', regex=True)
        begin_dates = pd.to_datetime(flood_df['DATE_BEGIN'], format='%Y%m%d%H%M%S', errors='coerce')
        end_dates = pd.to_datetime(flood_df['DATE_END'], format='%Y%m%d%H%M%S', errors='coerce')
        valid_dates_mask = begin_dates.notna() & end_dates.notna()
        daily_floods = flood_df[valid_dates_mask & (begin_dates.dt.date <= target_date.date()) & (end_dates.dt.date >= target_date.date())]

        for _, row in daily_floods.iterrows():
            lat = pd.to_numeric(row['LAT'], errors='coerce')
            lon = pd.to_numeric(row['LON'], errors='coerce')
            if pd.notna(lat) and pd.notna(lon):
                row_frac = (lat_max_us - lat) / (lat_max_us - lat_min_us)
                col_frac = (lon - lon_min_us) / (lon_max_us - lon_min_us)
                grid_row = int(row_frac * n_rows)
                grid_col = int(col_frac * n_cols)
                grid_row = min(max(grid_row, 0), n_rows - 1)
                grid_col = min(max(grid_col, 0), n_cols - 1)
                
                node_idx = grid_row * n_cols + grid_col
                if 0 <= node_idx < num_nodes:
                    y[node_idx] = 1

        # --- Create and save Data object ---
        data = Data(
            x_static=elevation_features,
            x_dynamic=wldas_tensor,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            num_nodes=num_nodes
        )
        
        torch.save(data, os.path.join(processed_dir, f'data_{idx}.pt'))
        return None
    except Exception as e:
        # Return error to be handled in the main process
        return f"Error processing day {idx}: {e}"


class FloodDataset(Dataset):
    def __init__(self, root, wldas_dir, flood_csv, transform=None, pre_transform=None, mode='train', train_test_split=0.8):
        self.wldas_dir = wldas_dir
        self.flood_csv_path = flood_csv
        self.mode = mode
        
        all_wldas_files = sorted([os.path.join(self.wldas_dir, f) for f in os.listdir(self.wldas_dir) if f.endswith('.nc')])
        
        split_idx = int(len(all_wldas_files) * train_test_split)
        if self.mode == 'train':
            self.wldas_files = all_wldas_files[:split_idx]
        else:
            self.wldas_files = all_wldas_files[split_idx:]

        super(FloodDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        # These are the files the dataset depends on.
        return [os.path.basename(self.flood_csv_path)] + [os.path.basename(f) for f in self.wldas_files]

    @property
    def processed_file_names(self):
        # The names of the files that will be generated in self.processed_dir
        return [f'data_{i}.pt' for i in range(len(self.wldas_files) - 14)]

    def len(self):
        return len(self.wldas_files) - 14

    def process(self):
        print("Starting one-time pre-processing...")
        
        # --- Perform one-time setup for elevation and graph structure ---
        print("Preparing elevation grid and graph structure (this happens once)...")
        agg_grid, lat_min_us, lat_max_us, lon_min_us, lon_max_us = process_dem_data()
        
        sample_wldas_file = self.wldas_files[0]
        with xr.open_dataset(sample_wldas_file) as ds:
            target_rows, target_cols = ds.sizes['lat'], ds.sizes['lon']

        # Ensure agg_grid is 3D for zoom compatibility
        if agg_grid.ndim == 2:
            agg_grid = agg_grid[:, :, np.newaxis]

        zoom_factors = (target_rows / agg_grid.shape[0], target_cols / agg_grid.shape[1], 1)
        agg_grid = zoom(agg_grid, zoom_factors, order=1)
        
        n_rows, n_cols, _ = agg_grid.shape
        num_nodes = n_rows * n_cols
        
        # Create graph structure
        edges, weights = [], []
        mean_elevation = agg_grid[:, :, 0]
        
        # Normalize elevation to be used in weight calculation
        min_elev, max_elev = np.nanmin(mean_elevation), np.nanmax(mean_elevation)
        if max_elev > min_elev:
            normalized_elevation = (mean_elevation - min_elev) / (max_elev - min_elev)
        else:
            normalized_elevation = np.zeros_like(mean_elevation)

        for r in range(n_rows):
            for c in range(n_cols):
                node_idx = r * n_cols + c
                node_elev = normalized_elevation[r, c]
                
                if np.isnan(node_elev): continue

                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        
                        if 0 <= nr < n_rows and 0 <= nc < n_cols:
                            neighbor_idx = nr * n_cols + nc
                            neighbor_elev = normalized_elevation[nr, nc]

                            if np.isnan(neighbor_elev): continue
                            
                            # Create a directed edge from higher to lower elevation
                            if node_elev > neighbor_elev:
                                edges.append([node_idx, neighbor_idx])
                                # Weight is the elevation difference
                                weights.append(node_elev - neighbor_elev)

        if not edges:
            # Handle case with no edges to avoid errors
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(weights, dtype=torch.float)

        print("Elevation grid and graph structure are ready.")

        # --- Parallel processing of daily graph data ---
        num_cores = min(cpu_count(), 16) # Cap at 16 cores to be reasonable
        print(f"Processing {self.len()} days of data using {num_cores} cores...")

        # Use functools.partial to pre-fill arguments for the worker function
        worker_func = partial(process_single_day, 
                              processed_dir=self.processed_dir,
                              wldas_files=self.wldas_files,
                              flood_data_path=self.flood_csv_path,
                              agg_grid=agg_grid,
                              lat_min_us=lat_min_us, lat_max_us=lat_max_us,
                              lon_min_us=lon_min_us, lon_max_us=lon_max_us,
                              n_rows=n_rows, n_cols=n_cols,
                              num_nodes=num_nodes,
                              edge_index=edge_index,
                              edge_weight=edge_weight)

        with Pool(processes=num_cores) as pool:
            # Create a tqdm progress bar
            results = list(tqdm(pool.imap(worker_func, range(self.len())), total=self.len(), desc="Generating Graph Files"))

        # Check for errors returned by workers
        errors = [res for res in results if res is not None]
        if errors:
            for error in errors:
                print(error)
            raise RuntimeError("Errors occurred during parallel processing.")

        print("Pre-processing finished.")

    def get(self, idx):
        # Load the pre-processed data file
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)
        return data

if __name__ == '__main__':
    # Example of how to use the dataset
    # This will trigger the `process` method on the first run, which will take time
    # but will be fast on subsequent runs.
    print("Initializing dataset...")
    dataset = FloodDataset(
        root='./data/flood_dataset_test_parallel', # Use a new root to trigger processing
        wldas_dir='WLDAS_2012',
        flood_csv='USFD_v1.0.csv',
        mode='test' # Use a smaller dataset for testing
    )

    print(f"Dataset size: {len(dataset)}")
    
    # Get a single sample - this should be very fast now
    print("Loading a sample...")
    sample = dataset[0]
    print(f"Sample 0: {sample}")
    print(f"Static features shape: {sample.x_static.shape}")
    print(f"Dynamic features shape: {sample.x_dynamic.shape}")
    print(f"Target shape: {sample.y.shape}")
    print(f"Number of flooded cells in sample 0: {sample.y.sum()}")
    print("Sample loaded successfully.")
