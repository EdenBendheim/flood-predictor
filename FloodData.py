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

# --- Worker Initialization for Multiprocessing ---
# This avoids passing the large flood grid dictionary repeatedly.
worker_flood_grids = None

def init_worker(flood_grids_data):
    """Initializes each worker process with the flood grid data."""
    global worker_flood_grids
    worker_flood_grids = flood_grids_data

# Helper function for parallel processing of daily data files.
# It must be defined at the top level to be pickleable by multiprocessing.
def process_day_features(file_path, feature_vars, downsample_factor):
    """
    Processes a single day's WLDAS data and combines it with flood data.
    Instead of saving to a file, it returns the processed tensor and its date.
    """
    global worker_flood_grids # Access the data initialized for this worker
    try:
        # Extract date from filename to fetch the correct flood grid
        date_str = os.path.basename(file_path).split('_')[-1].split('.')[0]
        current_date = pd.to_datetime(date_str, format='%Y%m%d').date()

        # Load WLDAS data for the day
        with xr.open_dataset(file_path) as ds:
            # Ensure all requested variables are present, filling with zeros if not
            features_list = []
            for var in feature_vars:
                if var in ds:
                    features_list.append(ds[var].values)
                else:
                    # If a variable is missing, create a zero-filled array of the correct shape.
                    # This is more robust than failing the entire pre-processing run.
                    print(f"Warning: Variable '{var}' not found in {os.path.basename(file_path)}. Using zeros.")
                    any_present_var = next(iter(ds.data_vars))
                    shape = ds[any_present_var].shape
                    features_list.append(np.zeros(shape))
            
            features = np.stack(features_list)
            # Squeeze the time dimension if it exists and has a size of 1
            if features.ndim == 4 and features.shape[1] == 1:
                features = features.squeeze(1)
                        # --- CRITICAL FIX ---
            # The raw WLDAS NetCDF data has its latitude axis flipped (south-up).
            # We must flip it vertically here to align with our north-up coordinate system.
            features = features[:, ::-1, :]


            zoom_factors = (1, 1 / downsample_factor, 1 / downsample_factor)
            downsampled_features = zoom(features, zoom_factors, order=1, prefilter=False)
            reshaped_features = downsampled_features.transpose(1, 2, 0).reshape(-1, len(feature_vars))

        # Get the corresponding historical flood grid for the day from the worker's global data
        flood_grid_for_day = worker_flood_grids.get(current_date)
        if flood_grid_for_day is None:
            # This case should be rare if the date ranges are aligned, but it's good practice to handle it.
            print(f"Warning: No flood grid found for {current_date}. Using zeros.")
            n_nodes = reshaped_features.shape[0]
            historical_flood_feature = np.zeros((n_nodes, 1))
        else:
            historical_flood_feature = flood_grid_for_day.flatten().reshape(-1, 1)

        # Combine WLDAS features with the historical flood feature
        combined_features = np.concatenate([reshaped_features, historical_flood_feature], axis=1)

        # Create and sanitize the final tensor for the day
        day_array = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Return the processed numpy array and its date for sorting
        return (current_date, day_array)
    except Exception as e:
        # Return error to be handled in the main process
        print(f"CRITICAL ERROR processing file {os.path.basename(file_path)}: {e}")
        return None


class FloodDataset(Dataset):
    def __init__(self, root, wldas_dir, flood_csv, config, transform=None, pre_transform=None):
        self.wldas_dir = wldas_dir
        self.flood_csv_path = flood_csv
        self.config = config
        self.sequence_length = self.config['model']['sequence_length']
        self.aggregation_strategy = self.config['model'].get('aggregation_strategy')
        if self.aggregation_strategy:
            print(f"INFO: Using aggregation strategy: {self.aggregation_strategy}")

        print(f"Scanning for all WLDAS data in: {self.wldas_dir}")
        self.wldas_files = sorted([
            os.path.join(self.wldas_dir, f) for f in os.listdir(self.wldas_dir) if f.endswith('.nc')
        ])

        if not self.wldas_files:
            raise FileNotFoundError(f"No WLDAS .nc files found in {self.wldas_dir}.")
        
        print(f"Found {len(self.wldas_files)} total WLDAS files.")
            
        super(FloodDataset, self).__init__(root, transform, pre_transform)
        
        # --- Load all processed data into memory ---
        
        # 1. Load the static graph data
        graph_data_path = os.path.join(self.processed_dir, 'graph_data_10x.pt')
        if not os.path.exists(graph_data_path):
            raise FileNotFoundError(f"graph_data_10x.pt not found in {self.processed_dir}. Please run pre-processing.")
        self.graph_data = torch.load(graph_data_path)
        
        # 2. Load the single file containing all daily feature tensors
        features_path = os.path.join(self.processed_dir, 'all_daily_features.pt')
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"all_daily_features.pt not found in {self.processed_dir}. Please run pre-processing.")
        
        dates_path = os.path.join(self.processed_dir, 'daily_dates.pt')
        if not os.path.exists(dates_path):
            raise FileNotFoundError(f"daily_dates.pt not found in {self.processed_dir}. Please run pre-processing.")

        print("Loading all daily features and dates into memory...")
        self.all_daily_features = torch.load(features_path)
        self.all_daily_dates = torch.load(dates_path)
        print("All features and dates loaded.")

        # --- In-Memory Filtering by Year Range ---
        year_range_config = self.config['data'].get('year_range')
        if year_range_config and year_range_config.get('enabled', False):
            try:
                start_year = year_range_config['start']
                end_year = year_range_config['end']
                print(f"--- Filtering dataset to years {start_year}-{end_year} ---")

                indices_to_keep = [
                    i for i, date in enumerate(self.all_daily_dates) 
                    if start_year <= date.year <= end_year
                ]

                if not indices_to_keep:
                    raise ValueError(f"No data found for the year range {start_year}-{end_year}. Check your data and config.")

                self.all_daily_features = self.all_daily_features[indices_to_keep]
                self.all_daily_dates = [self.all_daily_dates[i] for i in indices_to_keep]
                print(f"Dataset filtered. New size: {len(self.all_daily_features)} days.")

            except KeyError as e:
                print(f"Warning: `year_range` is enabled but missing a key: {e}. Skipping filtering.")
            except Exception as e:
                print(f"Error during year filtering: {e}")


    @property
    def raw_file_names(self):
        # These are the source files the dataset depends on.
        return [os.path.basename(self.flood_csv_path)] + [os.path.basename(f) for f in self.wldas_files]

    @property
    def processed_file_names(self):
        # The entire processed dataset now consists of three files.
        return ['graph_data_10x.pt', 'all_daily_features.pt', 'daily_dates.pt']

    def len(self):
        # The length is the number of possible windows.
        # We need `sequence_length` days for features and 1 day for the target label.
        return len(self.all_daily_features) - self.sequence_length

    def process(self):
        print("--- Starting New One-Time Pre-processing ---")
        
        # --- Define downsampling factor for WLDAS data (used for WLDAS files only) ---
        DOWNSAMPLE_FACTOR = self.config['preprocessing']['downsample_factor']
        print(f"Using WLDAS downsampling factor: {DOWNSAMPLE_FACTOR}")

        # --- Perform one-time setup for elevation and graph structure ---
        print("Preparing elevation grid and graph structure (this happens once)...")
        # 1. Get the full-resolution, aligned DEM grid
        # Pass the subsetting configuration to the DEM processing function
        full_res_grid, lat_min_us, lat_max_us, lon_min_us, lon_max_us = process_dem_data(
            self.wldas_dir,
            subset_bounds=self.config['data'].get('subset_bounds')
        )

        # 2. Deterministically resample the DEM to the final target grid size
        target_shape = tuple(self.config['preprocessing']['target_shape'])
        print(f"Resampling aligned DEM from {full_res_grid.shape} to target shape {target_shape}...")
        zoom_factors = (
            target_shape[0] / full_res_grid.shape[0],
            target_shape[1] / full_res_grid.shape[1]
        )
        agg_grid = zoom(full_res_grid, zoom_factors, order=1, prefilter=False)
        print(f"Final downsampled DEM grid shape: {agg_grid.shape}")
        
        # --- Store boundaries and grid dimensions for later use ---
        self.lat_min, self.lat_max = lat_min_us, lat_max_us
        self.lon_min, self.lon_max = lon_min_us, lon_max_us
        
        n_rows, n_cols = agg_grid.shape
        self.n_rows, self.n_cols = n_rows, n_cols # Store grid shape
        num_nodes = n_rows * n_cols
        print(f"Final grid shape is {n_rows}x{n_cols} ({num_nodes} nodes).")
        
        # --- Vectorized Graph Structure Creation ---
        print("Creating graph structure with vectorized operations...")
        mean_elevation = agg_grid
        
        # Normalize elevation
        min_elev, max_elev = np.nanmin(mean_elevation), np.nanmax(mean_elevation)
        if max_elev > min_elev:
            normalized_elevation = (mean_elevation - min_elev) / (max_elev - min_elev)
        else:
            normalized_elevation = np.zeros_like(mean_elevation)

        # Create a grid of node indices
        node_indices = np.arange(num_nodes).reshape(n_rows, n_cols)

        # Pad the elevation and index grids to handle boundaries easily
        padded_elev = np.pad(normalized_elevation, pad_width=1, mode='constant', constant_values=np.nan)
        
        all_edges = []
        all_weights = []

        # Iterate over the 8 neighbor directions
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                # Create "shifted" views of the elevation grid
                neighbor_elev = padded_elev[1+dr : 1+dr+n_rows, 1+dc : 1+dc+n_cols]

                # Find all valid edges where source elevation > neighbor
                mask = (normalized_elevation > neighbor_elev) & ~np.isnan(normalized_elevation) & ~np.isnan(neighbor_elev)
                
                if np.any(mask):
                    source_nodes = node_indices[mask]
                    padded_indices = np.pad(node_indices, pad_width=1, mode='constant', constant_values=-1)
                    neighbor_indices = padded_indices[1+dr : 1+dr+n_rows, 1+dc : 1+dc+n_cols]
                    dest_nodes = neighbor_indices[mask]
                    
                    weights = normalized_elevation[mask] - neighbor_elev[mask]
                    
                    # Edges are shape [2, num_edges]
                    edges = np.stack([source_nodes, dest_nodes])
                    all_edges.append(edges)
                    all_weights.append(weights)

        # Handle the case where the elevation grid is flat and no edges can be created.
        if not all_edges:
            min_el, max_el = np.nanmin(agg_grid), np.nanmax(agg_grid)
            error_msg = (
                "CRITICAL ERROR: No downward-sloping edges were found in the elevation grid.\n"
                f"Elevation grid stats: min={min_el}, max={max_el}, shape={agg_grid.shape}\n"
                "This usually means the elevation data is completely flat or invalid after "
                "the reprojection process, which prevents graph construction."
            )
            print(error_msg)
            raise ValueError(error_msg)

        # Concatenate edges and weights from all directions
        edge_index_np = np.concatenate(all_edges, axis=1)
        edge_weight_np = np.concatenate(all_weights)
        
        print(f"Graph created with {edge_index_np.shape[1]} edges.")
        
        # --- Save the Static Graph Data ---
        static_data = Data(
            x_static=torch.tensor(normalized_elevation.flatten(), dtype=torch.float).unsqueeze(1),
            edge_index=torch.tensor(edge_index_np, dtype=torch.long),
            edge_weight=torch.tensor(edge_weight_np, dtype=torch.float),
            num_nodes=num_nodes,
            n_rows=torch.tensor(n_rows),
            n_cols=torch.tensor(n_cols),
            lat_min=torch.tensor(self.lat_min),
            lat_max=torch.tensor(self.lat_max),
            lon_min=torch.tensor(self.lon_min),
            lon_max=torch.tensor(self.lon_max),
        )
        graph_data_path = os.path.join(self.processed_dir, 'graph_data_10x.pt')
        torch.save(static_data, graph_data_path)
        print(f"Static graph data saved to {graph_data_path}")

        # --- Pre-process flood data once ---
        print("Pre-calculating flood grids (this may take a moment)...")
        # Load and filter flood events once
        flood_df = pd.read_csv(
            self.flood_csv_path,
            usecols=['DATE_BEGIN', 'DATE_END', 'LAT', 'LON'],
            dtype={'LAT': str, 'LON': str},
            encoding='latin1'
        )
        # Date parsing...
        flood_df['DATE_BEGIN'] = flood_df['DATE_BEGIN'].astype(str).str.replace(r'\.0$', '', regex=True)
        flood_df['DATE_END'] = flood_df['DATE_END'].astype(str).str.replace(r'\.0$', '', regex=True)
        flood_df['begin'] = pd.to_datetime(flood_df['DATE_BEGIN'], format='%Y%m%d%H%M%S', errors='coerce')
        flood_df['end'] = pd.to_datetime(flood_df['DATE_END'], format='%Y%m%d%H%M%S', errors='coerce')
        flood_df.dropna(subset=['begin', 'end'], inplace=True)

        # Lat/Lon parsing...
        flood_df['lat'] = pd.to_numeric(flood_df['LAT'], errors='coerce')
        flood_df['lon'] = pd.to_numeric(flood_df['LON'], errors='coerce')
        flood_df.dropna(subset=['lat', 'lon'], inplace=True)
        
        # --- Pre-calculate flood grids (Optimized Method) ---
        print("Pre-calculating daily flood grids with optimized method...")
        
        # Get the date range of our dataset
        start_date = pd.to_datetime(os.path.basename(self.wldas_files[0]).split('_')[-1].split('.')[0], format='%Y%m%d')
        end_date = pd.to_datetime(os.path.basename(self.wldas_files[-1]).split('_')[-1].split('.')[0], format='%Y%m%d')
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # --- Vectorized Filtering and Grid Index Calculation ---
        # 1. Filter events to those overlapping with our dataset's time range
        flood_df = flood_df[(flood_df['begin'] <= end_date) & (flood_df['end'] >= start_date)].copy()
        print(f"Filtered to {len(flood_df)} flood events within the dataset's date range.")

        # 2. Filter events to those within our geographic bounding box
        # The bounds used for filtering now come from the authoritative source: the processed DEM grid
        flood_df = flood_df[
            (flood_df['lat'] >= lat_min_us) & (flood_df['lat'] <= lat_max_us) &
            (flood_df['lon'] >= lon_min_us) & (flood_df['lon'] <= lon_max_us)
        ]
        print(f"Filtered to {len(flood_df)} events within the geographic bounding box.")
        
        if not flood_df.empty:
            # 3. Calculate grid indices for all events at once (vectorized)
            lat_frac = (lat_max_us - flood_df['lat']) / (lat_max_us - lat_min_us)
            lon_frac = (flood_df['lon'] - lon_min_us) / (lon_max_us - lon_min_us)
            flood_df['grid_row'] = (lat_frac * n_rows).astype(int)
            flood_df['grid_col'] = (lon_frac * n_cols).astype(int)
    
            # Clamp values to be within grid bounds
            flood_df['grid_row'] = flood_df['grid_row'].clip(0, n_rows - 1)
            flood_df['grid_col'] = flood_df['grid_col'].clip(0, n_cols - 1)
    
            # Convert 'begin' and 'end' to date objects once for faster comparison in the loop
            flood_df['begin_date'] = flood_df['begin'].dt.date
            flood_df['end_date'] = flood_df['end'].dt.date

        # --- Rasterize Events Day by Day (Optimized Loop) ---
        # Initialize a dictionary of empty grids
        daily_flood_grids = {d.date(): np.zeros((n_rows, n_cols), dtype=np.float32) for d in all_dates}

        if not flood_df.empty:
            for current_date in tqdm(all_dates, desc="Projecting flood events onto grid"):
                d = current_date.date()
                
                # Find all flood events active on this specific day (vectorized)
                active_events = flood_df[(flood_df['begin_date'] <= d) & (flood_df['end_date'] >= d)]
                
                if not active_events.empty:
                    # Get the grid coordinates for all active events
                    active_rows = active_events['grid_row'].values
                    active_cols = active_events['grid_col'].values
                    
                    # Use advanced numpy indexing to set all flooded cells to 1.0 at once
                    daily_flood_grids[d][active_rows, active_cols] = 1.0

        print("Daily flood grids are ready.")

        # --- Define the expanded list of WLDAS features ---
        feature_vars = sorted(list(set(self.config['preprocessing']['feature_vars'])))
        print(f"Using {len(feature_vars)} WLDAS variables.")
        
        # --- Parallel processing of daily data ---
        # The worker function now returns the data instead of saving it.
        num_cores = min(cpu_count(), 25) 
        print(f"Processing {len(self.wldas_files)} daily feature files using {num_cores} cores...")

        # Initialize workers with the large flood grid data once.
        # This is a major performance optimization.
        with Pool(processes=num_cores, initializer=init_worker, initargs=(daily_flood_grids,)) as pool:
            # The worker function no longer needs the large dictionary passed to it.
            worker_func = partial(process_day_features,
                                  feature_vars=feature_vars,
                                  downsample_factor=DOWNSAMPLE_FACTOR)
            
            # pool.imap preserves order, which is crucial.
            results = list(tqdm(pool.imap(worker_func, self.wldas_files), total=len(self.wldas_files), desc="Processing Daily Data"))

        # --- Filter out None results from errors and sort by date ---
        valid_results = [res for res in results if res is not None]
        if not valid_results:
            raise RuntimeError("No daily data could be processed. Check for errors above.")
        
        valid_results.sort(key=lambda x: x[0]) # Sort by date
        
        # --- Stack all tensors and save to a single file ---
        # This is the key change to optimize loading speed.
        all_dates = [res[0] for res in valid_results]
        all_arrays = [res[1] for res in valid_results]
        # First, stack the numpy arrays, which is fast.
        master_array = np.stack(all_arrays, axis=0)
        # Then, convert the entire block to a torch tensor at once.
        master_tensor = torch.from_numpy(master_array).float()
        
        print(f"Master tensor created with shape: {master_tensor.shape}")
        
        features_save_path = os.path.join(self.processed_dir, 'all_daily_features.pt')
        torch.save(master_tensor, features_save_path)
        print(f"All daily features saved to a single file: {features_save_path}")

        # --- Save the corresponding dates ---
        dates_save_path = os.path.join(self.processed_dir, 'daily_dates.pt')
        torch.save(all_dates, dates_save_path)
        print(f"Daily dates saved to: {dates_save_path}")

        print("--- Pre-processing finished. ---")

    def get(self, idx):
        # With all data pre-loaded into memory, this method becomes extremely fast.
        # It's just slicing tensors that are already in RAM.

        # 1. Get the target data (flood status from the day after the sequence)
        target_day_tensor = self.all_daily_features[idx + self.sequence_length]
        y = target_day_tensor[:, -1].float()

        # 2. Prepare the dynamic features based on the aggregation strategy
        full_window = self.all_daily_features[idx : idx + self.sequence_length]
        
        if self.aggregation_strategy:
            # --- Aggregation Logic ---
            aggregated_steps = []
            for group in self.aggregation_strategy:
                # Robust validation of the group structure
                if not isinstance(group, list):
                    raise ValueError(f"Invalid group in aggregation_strategy: Expected a list of lists, but found an element of type {type(group)}.")

                for i in group:
                    if not isinstance(i, int) or not (0 <= i < self.sequence_length):
                        raise ValueError(
                            f"Invalid index '{i}' in aggregation group {group}. "
                            f"Each index must be an integer between 0 and {self.sequence_length - 1}."
                        )

                # Select the tensors for the days in the current group and average them
                days_in_group = full_window[group]
                mean_of_group = torch.mean(days_in_group, dim=0)
                aggregated_steps.append(mean_of_group)

            # If we have successfully created aggregated steps, stack them.
            if aggregated_steps:
                # Stack along dim=1 to create a [nodes, new_seq_len, features] tensor.
                x_dynamic = torch.stack(aggregated_steps, dim=1)
            else:
                # Fallback to standard logic if the strategy was empty or invalid
                print("Warning: Aggregation strategy resulted in no steps. Using standard sequence.")
                x_dynamic = full_window.permute(1, 0, 2)
        else:
            # --- Standard Logic ---
            # The master tensor is [days, nodes, features], so we permute to [nodes, days, features].
            x_dynamic = full_window.permute(1, 0, 2)


        # 3. Combine into a single Data object
        data = Data(
            x_static=self.graph_data.x_static,
            x_dynamic=x_dynamic,
            edge_index=self.graph_data.edge_index,
            edge_weight=self.graph_data.edge_weight,
            y=y,
            num_nodes=self.graph_data.num_nodes
        )
        return data

if __name__ == '__main__':
    # Example of how to use the dataset
    # This will trigger the `process` method on the first run, which will take time
    # but will be fast on subsequent runs.
    
    # --- Dummy Config for standalone execution ---
    # In a real run, this would be loaded from config.yaml
    dummy_config = {
        'data': {
            'root_dir': './data/flood_dataset_daily_features',
            'wldas_dir': 'WLDAS',
            'flood_csv': 'USFD_v1.0.csv',
            'subset_bounds': {'enabled': False},
            'year_range': {'enabled': False, 'start': 2003, 'end': 2013}
        },
        'preprocessing': {
            'downsample_factor': 10,
            'target_shape': [279, 359],
            'feature_vars': [
                'Rainf_tavg', 'SoilMoi00_10cm_tavg', 'Snowf_tavg', 'Tair_f_tavg', 'Evap_tavg',
                'SWE_tavg', 'Qs_tavg', 'Qsb_tavg', 'GWS_tavg', 'Qle_tavg', 'Qsm_tavg',
                'Rainf_f_tavg', 'Snowcover_tavg', 'SnowDepth_tavg', 'SoilMoi10_40cm_tavg',
                'SoilMoi40_100cm_tavg', 'SoilMoi100_200cm_tavg', 'WaterTableD_tavg', 'WT_tavg'
            ]
        },
        'model': {
            'sequence_length': 8,
            'aggregation_strategy': []
        }
    }

    print("Initializing dataset...")
    dataset = FloodDataset(
        root=dummy_config['data']['root_dir'], # Use a new root to trigger processing
        wldas_dir=dummy_config['data']['wldas_dir'], # This should be the single directory with all .nc files
        flood_csv=dummy_config['data']['flood_csv'],
        config=dummy_config
    )

    print(f"\nDataset size: {len(dataset)}")
    
    # Get a single sample - this should be very fast now
    print("Loading a sample...")
    # The 'get' method now returns a full Data object for a training window
    sample = dataset[0]
    print(f"Sample 0: {sample}")
    print(f"  > Static features shape: {sample.x_static.shape}")
    print(f"  > Dynamic features shape: {sample.x_dynamic.shape}")
    print(f"  > Target 'y' shape: {sample.y.shape}")
    print(f"  > Number of flooded cells in target: {sample.y.sum():.0f}")

    print("\nSample loaded successfully.")
