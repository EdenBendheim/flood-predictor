import os
from FloodData import FloodDataset

def main():
    """
    This script is a simple utility to regenerate the 'graph_data.pt' file.
    It works by instantiating the FloodDataset, which triggers the one-time
    pre-processing step that creates the elevation grid and graph structure.
    By calling our new, corrected `read_dem.py` logic, this will save a new,
    correct version of the static graph data.
    """
    print("--- Starting Regeneration of graph_data.pt ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the paths required by the FloodDataset constructor
    data_root = os.path.join(script_dir, 'data')
    wldas_dir = os.path.join(script_dir, 'WLDAS')
    flood_csv = os.path.join(script_dir, 'USFD_v1.0.csv')
    
    # The 'processed' sub-directory is where the files are stored.
    processed_dir = os.path.join(data_root, 'processed')
    graph_data_path = os.path.join(processed_dir, 'graph_data.pt')
    
    # For safety, remove the old file if it exists.
    # This ensures the pre-processing logic is definitely triggered.
    if os.path.exists(graph_data_path):
        print(f"Removing old graph data file: {graph_data_path}")
        os.remove(graph_data_path)
    
    # Instantiate the dataset. This is all that's needed.
    # The __init__ method will see that graph_data.pt is missing and will
    # call the self.process() method to regenerate it.
    print("Instantiating FloodDataset to trigger pre-processing...")
    FloodDataset(root=data_root, wldas_dir=wldas_dir, flood_csv=flood_csv)
    
    print("\n--- Regeneration Complete ---")
    if os.path.exists(graph_data_path):
        print(f"Successfully created new graph data file at:")
        print(graph_data_path)
    else:
        print("ERROR: For some reason, the graph data file was not created.")

if __name__ == '__main__':
    main() 