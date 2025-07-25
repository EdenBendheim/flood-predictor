import os
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

def process_fire_events(fire_data_dir, output_path):
    """
    Processes raw FEDS2.5 GeoPackage files to identify unique fire events
    and create a master manifest file.

    Args:
        fire_data_dir (str): Path to the directory containing .gpkg files.
        output_path (str): Path to save the output manifest CSV file.
    """
    gpkg_files = [os.path.join(fire_data_dir, f) for f in os.listdir(fire_data_dir) if f.endswith('.gpkg')]
    if not gpkg_files:
        raise FileNotFoundError(f"No .gpkg files found in {fire_data_dir}")

    print(f"Found {len(gpkg_files)} GeoPackage files to process.")

    # --- Load and combine all yearly data ---
    # We specify the 'perimeter' layer, as the .gpkg files contain multiple layers.
    all_fires_gdf = pd.concat(
        [gpd.read_file(f, layer='perimeter') for f in tqdm(gpkg_files, desc="Loading .gpkg files")],
        ignore_index=True
    )

    print(f"Loaded a total of {len(all_fires_gdf)} fire records.")
    
    # Let's inspect columns to find the best identifier.
    print("Available columns:", all_fires_gdf.columns.tolist())
    print("\n--- Data Head ---")
    print(all_fires_gdf.head())
    print("-----------------\n")

    # --- Data Cleaning and Type Conversion ---
    # The 't' column appears to be an integer representing the day.
    # The 'FireID' column will be used as the unique identifier.
    all_fires_gdf['t'] = pd.to_numeric(all_fires_gdf['t'])
    all_fires_gdf['FireID'] = all_fires_gdf['FireID'].astype(str)

    # --- Create the Manifest by Aggregating Daily Records ---
    from shapely.ops import unary_union
    manifest_data = []
    
    # Check CRS for area calculation
    if all_fires_gdf.crs:
        print(f"\nCRS found: {all_fires_gdf.crs}\n")
    else:
        print("\nWarning: No CRS found. Area calculations may be incorrect.\n")

    # Group by the unique fire ID
    for fire_id, group in tqdm(all_fires_gdf.groupby('FireID'), desc="Aggregating fire events"):
        # Sort the fire's history by the time column 't'
        group = group.sort_values(by='t')
        
        start_t = group['t'].min()
        end_t = group['t'].max()
        
        # Calculate the union of all daily perimeters for the fire
        union_geom = unary_union(group['geometry'])

        # Project to an equal-area projection (California Albers - EPSG:3310) to get area in meters
        # Create a GeoSeries from the union geometry to perform the projection
        geom_gs = gpd.GeoSeries([union_geom], crs=all_fires_gdf.crs)
        geom_proj = geom_gs.to_crs('EPSG:3310')
        
        # Calculate area in square kilometers
        area_sq_km = geom_proj.area.iloc[0] / 1e6

        manifest_data.append({
            'fire_id': fire_id,
            'start_t': start_t,
            'end_t': end_t,
            'duration_days': end_t - start_t + 1,
            'union_geometry_wkt': union_geom.wkt,
            'union_area_sq_km': area_sq_km
        })

    manifest_df = pd.DataFrame(manifest_data)
    
    # Save the manifest to a CSV file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    manifest_df.to_csv(output_path, index=False)
    
    print(f"\nSuccessfully created fire manifest with {len(manifest_df)} unique fire events.")
    print(f"Manifest saved to: {output_path}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fire_data_dir = os.path.join(script_dir, 'FEDS2.5')
    output_csv_path = os.path.join(script_dir, 'data', 'fire_manifest.csv')
    
    if not os.path.exists(fire_data_dir):
        print(f"Error: Fire data directory not found at '{fire_data_dir}'")
    else:
        process_fire_events(fire_data_dir, output_csv_path) 