import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import math

def get_grid_dims(processed_dir, script_dir):
    """
    Calculates grid dimensions by using the number of nodes from the processed
    data and the aspect ratio from a raw WLDAS file. This avoids needing to
    re-run pre-processing.
    """
    try:
        # 1. Get the number of nodes from the processed data
        features_path = os.path.join(processed_dir, 'all_daily_features.pt')
        if not os.path.exists(features_path):
            print(f"Error: Could not find '{features_path}'.")
            return None, None
            
        # Use torch.load with map_location='cpu' to inspect the tensor size
        # without loading the whole thing into GPU memory.
        features_tensor = torch.load(features_path, map_location='cpu')
        num_nodes = features_tensor.shape[1]

        # 2. Get the aspect ratio from a raw NetCDF file
        wldas_dir = os.path.join(script_dir, 'WLDAS')
        import xarray as xr
        wldas_files = sorted([f for f in os.listdir(wldas_dir) if f.endswith('.nc')])
        if not wldas_files:
            print(f"Error: No .nc files found in {wldas_dir} to determine aspect ratio.")
            return None, None
            
        sample_nc_path = os.path.join(wldas_dir, wldas_files[0])
        with xr.open_dataset(sample_nc_path) as ds:
            original_rows = ds.sizes['lat']
            original_cols = ds.sizes['lon']
        
        aspect_ratio = original_rows / original_cols

        # 3. Solve for n_rows and n_cols
        # num_nodes = n_rows * n_cols
        # aspect_ratio = n_rows / n_cols  => n_rows = aspect_ratio * n_cols
        # num_nodes = (aspect_ratio * n_cols) * n_cols
        n_cols = math.sqrt(num_nodes / aspect_ratio)
        n_rows = num_nodes / n_cols
        
        # Return integer dimensions
        return round(n_rows), round(n_cols)

    except ImportError:
        print("Error: Could not import xarray. Please install it (`pip install xarray netcdf4`).")
        return None, None
    except Exception as e:
        print(f"An error occurred while calculating grid dimensions: {e}")
        return None, None


def inspect_data():
    """
    Loads the consolidated processed data, inspects it, and generates a
    visual validation image of the flood map for a random day.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, 'data', 'processed')
    
    print('--- Inspecting Consolidated Processed Data ---')

    graph_data_path = os.path.join(processed_dir, 'graph_data_10x.pt')
    features_path = os.path.join(processed_dir, 'all_daily_features.pt')

    if not os.path.exists(graph_data_path) or not os.path.exists(features_path):
        print(f"Error: Data not found in '{processed_dir}'")
        print("Please ensure pre-processing has run successfully by running train.py.")
        return

    # --- Dynamically Calculate Grid Dimensions ---
    print('\n--- 1. Calculating Grid Dimensions ---')
    n_rows, n_cols = get_grid_dims(processed_dir, script_dir)
    if n_rows is None:
        return # Error message was already printed
    print(f'Successfully calculated grid dimensions: {n_rows} rows x {n_cols} cols')

    # --- Inspect Consolidated Daily Features ---
    print('\n--- 2. Consolidated Daily Features ---')
    print(f"Loading master feature file: {features_path}")
    all_features = torch.load(features_path, map_location='cpu')
    print(f'Shape of master tensor (Days, Nodes, Features): {all_features.shape}')
    
    num_days = all_features.shape[0]

    # --- Select a Random Day and Create Validation Image ---
    print('\n--- 3. Visual Validation ---')
    
    # --- Define Features and Find Their Indices ---
    overlay_feature_name = 'Tair_f_tavg'
    wldas_feature_list = sorted(list(set([
        'Rainf_tavg', 'SoilMoi00_10cm_tavg', 'Snowf_tavg', 'Tair_f_tavg', 'Evap_tavg', 'SWE_tavg', 'Qs_tavg', 'Qsb_tavg',
        'GWS_tavg', 'Qle_tavg', 'Qsm_tavg', 'Rainf_f_tavg', 'Snowcover_tavg', 'SnowDepth_tavg', 
        'SoilMoi10_40cm_tavg', 'SoilMoi40_100cm_tavg', 'SoilMoi100_200cm_tavg', 'WaterTableD_tavg', 'WT_tavg'
    ])))
    
    try:
        overlay_feature_index = wldas_feature_list.index(overlay_feature_name)
        print(f"Found overlay feature '{overlay_feature_name}' at index {overlay_feature_index}.")
    except ValueError:
        print(f"Error: Feature '{overlay_feature_name}' not found. Please check the name.")
        return

    sample_day_idx = random.randint(0, num_days - 1)
    print(f'Selecting random day at index {sample_day_idx} for validation...')

    # --- Extract and Reshape Data ---
    graph_data = torch.load(graph_data_path, map_location='cpu')
    day_data = all_features[sample_day_idx]
    
    # Get the background elevation data and reshape it
    elevation_vector = graph_data.x_static
    elevation_grid = elevation_vector.numpy().reshape(n_rows, n_cols)
    
    # Mask out values equal to the grid's minimum so they become transparent.
    # This is more robust than assuming the minimum/no-data value is always 0.
    min_elevation = elevation_grid[259,0]
    masked_elevation = np.ma.masked_where(elevation_grid <= min_elevation, elevation_grid)
    
    # Get the overlay temperature data
    overlay_vector = day_data[:, overlay_feature_index]
    overlay_grid = overlay_vector.numpy().reshape(n_rows, n_cols)
    
    # Mask out zero values in the overlay so they become transparent
    masked_overlay = np.ma.masked_where(overlay_grid == 0, overlay_grid)

    # --- Generate and Save the Comparison Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    
    # Use a standard terrain colormap
    terrain_cmap = plt.get_cmap('terrain').copy()
    
    # --- Plot 1: Just the Elevation Map ---
    axes[0].imshow(masked_elevation, cmap=terrain_cmap, interpolation='nearest')
    axes[0].set_title(f'Base Elevation Map')
    axes[0].set_xlabel('Grid Column')
    axes[0].set_ylabel('Grid Row')
    
    # --- Plot 2: Temperature Overlay on Elevation ---
    # Plot the background elevation map (zeros will be transparent)
    axes[1].imshow(masked_elevation, cmap=terrain_cmap, interpolation='nearest')
    # Plot the semi-transparent temperature overlay
    im = axes[1].imshow(masked_overlay, cmap='hot', alpha=0.8, interpolation='nearest')
    fig.colorbar(im, ax=axes[1], label=f'Value of {overlay_feature_name} (Kelvin)', shrink=0.7)

    axes[1].set_title(f'Overlay of {overlay_feature_name} (Day {sample_day_idx})')
    axes[1].set_xlabel('Grid Column')
    
    fig.suptitle('Visual Comparison of Elevation and WLDAS Temperature Data', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle

    plot_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, f'data_validation_comparison_map.png')
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f'\nSUCCESS: Validation image saved to {save_path}')
    

if __name__ == '__main__':
    inspect_data() 