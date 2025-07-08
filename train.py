# Idea: See if adding gencasts latents to the model improves performance
# Idea: see if we can run on grace hopper for emergent technology.
# Idea: Change model to emulate gencast more. See what the retrained gencast did
# Idea: Test with more data, see if maybe this data isnt enough.


import os
import torch
import sys
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from torch.cuda.amp import GradScaler
import threading
import queue
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import gc
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

from FloodData import FloodDataset
from gcn_model import SpatioTemporalGCN

# --- Performance and Debugging ---
# Enables cuDNN for GPU acceleration, which is crucial for performance.
# The benchmark mode finds the best algorithm for the specific input sizes.
torch.backends.cudnn.benchmark = True
# Enable anomaly detection to get a traceback when NaNs/Infs are created.
torch.autograd.set_detect_anomaly(True)

def setup(rank, world_size):
    """Initializes the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Cleans up the distributed environment."""
    dist.destroy_process_group()

def train_one_day(model, day_data, optimizer, device, scaler):
    model.train()
    day_data = day_data.to(device)
    optimizer.zero_grad(set_to_none=True)
    
    with torch.amp.autocast(device_type=device.type):
        out = model(day_data)
        
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"WARNING: NaN or Inf detected in model output!")
        if torch.isnan(day_data.y).any() or torch.isinf(day_data.y).any():
            print(f"WARNING: NaN or Inf detected in targets!")
        
        num_positives = day_data.y.sum()
        num_negatives = len(day_data.y) - num_positives
        pos_weight = num_negatives / num_positives if num_positives > 0 else torch.tensor(1.0, device=device)
        
        if torch.isnan(pos_weight) or torch.isinf(pos_weight):
            print(f"WARNING: Invalid pos_weight! num_positives={num_positives}, num_negatives={num_negatives}")
            pos_weight = torch.tensor(1.0, device=device)
        
        pos_weight = torch.clamp(pos_weight, min=1.0, max=10000.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = criterion(out, day_data.y)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"CRITICAL: NaN or Inf loss detected! Skipping day.")
            return 0
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()

@torch.no_grad()
def test_one_day(model, day_data, device):
    model.eval()
    day_data = day_data.to(device)
    
    with torch.amp.autocast(device_type=device.type):
        out = model(day_data)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(out, day_data.y)
    
    probs = out.sigmoid()
    preds = (probs > 0.5).long()
    
    return loss.item(), preds.cpu(), day_data.y.cpu(), probs.cpu()

def plot_loss_curve(epochs, train_losses, val_losses, save_path):
    """Plots the training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")

def plot_spatial_predictions(day_labels, day_preds, dataset, save_path):
    """Generates and saves a spatial plot of ground truth vs. predictions."""
    try:
        # --- Download and load US states shapefile ---
        # Using a publicly available shapefile from the US Census Bureau.
        shapefile_url = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip"
        us_states = gpd.read_file(shapefile_url)
    except Exception as e:
        print(f"Could not download or read shapefile, skipping map plot: {e}")
        return

    # --- Hardcoded Geospatial Info (to avoid data regeneration for plotting) ---
    # These values must match the ones used during the last `process()` run.
    lat_min, lat_max = 25.065, 40.0
    lon_min, lon_max = -110.0, -89.025
    n_rows, n_cols = 279, 359

    # --- Reshape data to 2D grid ---
    truth_grid = day_labels.numpy().reshape(n_rows, n_cols)
    pred_grid = day_preds.numpy().reshape(n_rows, n_cols)

    # --- Create coordinate grids ---
    lons = np.linspace(lon_min, lon_max, n_cols)
    lats = np.linspace(lat_max, lat_min, n_rows)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # --- Plotting ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Plot state borders, cropped to our data's extent for focus
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    us_states.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)

    # --- Overlay heatmaps for truth and predictions ---
    # Use different colors and slight transparency
    # Plotting non-zero values only for clarity
    
    # Ground Truth (Blue)
    true_lats = lat_grid[truth_grid > 0]
    true_lons = lon_grid[truth_grid > 0]
    ax.scatter(true_lons, true_lats, color='blue', alpha=0.7, label='Actual Floods', marker='s', s=10)

    # Predictions (Red)
    pred_lats = lat_grid[pred_grid > 0]
    pred_lons = lon_grid[pred_grid > 0]
    ax.scatter(pred_lons, pred_lats, color='red', alpha=0.7, label='Predicted Floods', marker='s', s=10)

    ax.set_title('Spatial Comparison of Floods: Actual vs. Predicted (First Test Day)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(True)
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Spatial prediction map saved to {save_path}")

def main(rank, world_size):
    setup(rank, world_size)
    
    try:
        # --- Hyperparameters ---
        EPOCHS = 10
        LEARNING_RATE = 0.001 * world_size # Scale learning rate
        HIDDEN_DIM = 256
        GCN_LAYERS = 8
        
        # --- Setup ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        device = torch.device(f'cuda:{rank}')
        
        # --- Dataset ---
        wldas_base_dir = os.path.join(script_dir, 'WLDAS')
        
        # Designate rank 0 as the sole data processor to prevent race conditions.
        if rank == 0:
            print("--- Rank 0: Starting Data Pre-processing check ---")
            # Instantiating the datasets will trigger the `process` method if data is not cached.
            # This will be done sequentially for train, val, and test.
            print("Processing train data...")
            FloodDataset(
                root=os.path.join(script_dir, 'data/flood_dataset_train_multiyear'),
                wldas_dir=wldas_base_dir,
                flood_csv=os.path.join(script_dir, 'USFD_v1.0.csv'),
                mode='train'
            )
            print("Processing val data...")
            FloodDataset(
                root=os.path.join(script_dir, 'data/flood_dataset_val_multiyear'),
                wldas_dir=wldas_base_dir,
                flood_csv=os.path.join(script_dir, 'USFD_v1.0.csv'),
                mode='val'
            )
            print("Processing test data...")
            FloodDataset(
                root=os.path.join(script_dir, 'data/flood_dataset_test_multiyear'),
                wldas_dir=wldas_base_dir,
                flood_csv=os.path.join(script_dir, 'USFD_v1.0.csv'),
                mode='test'
            )
            print("--- Rank 0: Data Pre-processing Finished ---")

        # Use a barrier to synchronize all processes.
        # This ensures that no process continues until rank 0 has finished pre-processing.
        dist.barrier()

        # Now, all processes can safely instantiate the datasets.
        # The data will be loaded from the cache, not re-processed.
        print(f"Rank {rank}: Loading datasets from cache...")
        train_dataset = FloodDataset(
            root=os.path.join(script_dir, 'data/flood_dataset_train_multiyear'), 
            wldas_dir=wldas_base_dir,
            flood_csv=os.path.join(script_dir, 'USFD_v1.0.csv'),
            mode='train'
        )
        val_dataset = FloodDataset(
            root=os.path.join(script_dir, 'data/flood_dataset_val_multiyear'),
            wldas_dir=wldas_base_dir,
            flood_csv=os.path.join(script_dir, 'USFD_v1.0.csv'),
            mode='val'
        )
        test_dataset = FloodDataset(
            root=os.path.join(script_dir, 'data/flood_dataset_test_multiyear'),
            wldas_dir=wldas_base_dir,
            flood_csv=os.path.join(script_dir, 'USFD_v1.0.csv'),
            mode='test'
        )
        
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        # Use a standard DataLoader to iterate over days, which is more robust
        # than the custom prefetcher for managing worker processes.
        train_day_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=None,  # Important: yields one day_data at a time
            num_workers=0, # Use 0 to avoid multiprocessing deadlocks in distributed setup
            prefetch_factor=2,
            persistent_workers=False,
        )
        val_day_loader = DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=None,
            num_workers=0, # Use 0 to avoid multiprocessing deadlocks in distributed setup
            prefetch_factor=2,
            persistent_workers=False,
        )
        test_day_loader = DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=None,
            num_workers=0, # Use 0 to avoid multiprocessing deadlocks in distributed setup
            prefetch_factor=2,
            persistent_workers=False,
        )

        # --- Model ---
        # Load one sample to infer model dimensions
        sample_data = next(iter(train_day_loader))
        model = SpatioTemporalGCN(
            static_feature_dim=sample_data.x_static.shape[1],
            dynamic_feature_dim=sample_data.x_dynamic.shape[2],
            sequence_length=sample_data.x_dynamic.shape[1],
            hidden_dim=HIDDEN_DIM,
            gcn_layers=GCN_LAYERS,
        ).to(device)
        
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scaler = torch.amp.GradScaler(enabled=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

        # --- Best model tracking ---
        best_val_f1_score = 0.0
        history = {'train_loss': [], 'val_loss': [], 'epochs': []}

        # --- Training Loop ---
        for epoch in range(1, EPOCHS + 1):
            train_sampler.set_epoch(epoch)
            epoch_total_loss = 0
            
            # The loader now yields one full day's data graph at a time
            progress_bar = tqdm(train_day_loader, desc=f"Epoch {epoch:02d}", unit="day", total=len(train_sampler), disable=(rank != 0 or not sys.stdout.isatty()))
            for day_data in progress_bar:
                day_loss = train_one_day(model, day_data, optimizer, device, scaler)
                epoch_total_loss += day_loss
                progress_bar.set_postfix(loss=f'{day_loss:.4f}')
            
            avg_epoch_loss = epoch_total_loss / len(train_sampler) if len(train_sampler) > 0 else 0
            
            # --- Evaluation Step ---
            val_total_loss = 0
            epoch_val_preds, epoch_val_labels = [], []
            
            val_progress_bar = tqdm(val_day_loader, desc=f"Epoch {epoch:02d} (Val)", unit="day", total=len(val_sampler), disable=(rank != 0 or not sys.stdout.isatty()))
            for day_data in val_progress_bar:
                day_loss, day_preds, day_labels, _ = test_one_day(model.module, day_data, device)
                val_total_loss += day_loss
                if day_preds.numel() > 0:
                    epoch_val_preds.append(day_preds)
                    epoch_val_labels.append(day_labels)
            
            # --- Gather results from all processes ---
            if rank == 0:
                # Rank 0 is the master, it will receive data from workers
                gathered_preds = [epoch_val_preds]
                gathered_labels = [epoch_val_labels]
                for i in range(1, world_size):
                    worker_preds = [None] # Placeholder for receiving object
                    worker_labels = [None]
                    dist.recv_object(worker_preds, src=i)
                    dist.recv_object(worker_labels, src=i)
                    gathered_preds.append(worker_preds[0])
                    gathered_labels.append(worker_labels[0])
            else:
                # Worker ranks send their data to rank 0
                dist.send_object([epoch_val_preds], dst=0)
                dist.send_object([epoch_val_labels], dst=0)
            
            # Synchronize all processes before continuing
            dist.barrier()

            # --- Evaluation Metrics (on rank 0) ---
            if rank == 0:
                # Flatten the lists of lists into a single list of tensors
                all_val_preds_list = [item for sublist in gathered_preds for item in sublist]
                all_val_labels_list = [item for sublist in gathered_labels for item in sublist]

                # Calculate average loss across all processes
                # Note: This is an approximation as batch sizes might vary slightly.
                # A more precise way would be to gather the loss from each process too.
                avg_val_loss = val_total_loss / len(val_sampler) if len(val_sampler) > 0 else 0
                
                # --- Store history for plotting ---
                history['train_loss'].append(avg_epoch_loss)
                history['val_loss'].append(avg_val_loss)
                history['epochs'].append(epoch)

                val_f1_score = 0
                if all_val_preds_list and all_val_labels_list:
                    all_val_preds = torch.cat(all_val_preds_list)
                    all_val_labels = torch.cat(all_val_labels_list).long()
                    
                    tp = ((all_val_preds == 1) & (all_val_labels == 1)).sum().item()
                    fp = ((all_val_preds == 1) & (all_val_labels == 0)).sum().item()
                    fn = ((all_val_preds == 0) & (all_val_labels == 1)).sum().item()
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    val_f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    print(f'Epoch: {epoch:02d}, Avg Train Loss: {avg_epoch_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1_score:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
                    total_predicted_floods = tp + fp
                    print(f'    > Val Metrics: FN: {fn} | FP: {fp} | Total Predictions: {total_predicted_floods} | Total Floods: {all_val_labels.sum().item()}')
                else:
                    print(f'Epoch: {epoch:02d}, Avg Train Loss: {avg_epoch_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1_score:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

                # --- Save Model Checkpoints based on Validation F1 ---
                save_dir = os.path.join(script_dir, 'saved_models')
                os.makedirs(save_dir, exist_ok=True)
                
                # Save the latest model
                latest_save_path = os.path.join(save_dir, f'flood_predictor_epoch_{epoch}.pth')
                torch.save(model.module.state_dict(), latest_save_path)
                
                # Save the best model if validation F1 score has improved
                if val_f1_score > best_val_f1_score:
                    best_val_f1_score = val_f1_score
                    best_save_path = os.path.join(save_dir, 'best_flood_predictor.pth')
                    torch.save(model.module.state_dict(), best_save_path)
                    print(f'New best model saved to {best_save_path} (Validation F1-Score: {best_val_f1_score:.4f})')

            scheduler.step()
            
        if rank == 0:
            print("Training finished.")
            print("--- Running Final Evaluation on Test Set ---")
            # Load the best model for final testing
            best_model_path = os.path.join(script_dir, 'saved_models', 'best_flood_predictor.pth')
            model_to_test = model.module
            if os.path.exists(best_model_path):
                # Load state dict on CPU first to avoid device mismatches
                model_to_test.load_state_dict(torch.load(best_model_path, map_location='cpu'))
                model_to_test.to(device) # Move model to the correct device for this rank
                if rank == 0:
                    print("Loaded best model for final evaluation.")

            test_total_loss = 0
            epoch_test_preds, epoch_test_labels, epoch_test_probs = [], [], []
            test_progress_bar = tqdm(test_day_loader, desc="Final Test", unit="day", total=len(test_sampler), disable=(rank != 0 or not sys.stdout.isatty()))
            for day_data in test_progress_bar:
                day_loss, day_preds, day_labels, day_probs = test_one_day(model_to_test, day_data, device)
                test_total_loss += day_loss
                if day_preds.numel() > 0:
                    epoch_test_preds.append(day_preds)
                    epoch_test_labels.append(day_labels)
                    epoch_test_probs.append(day_probs)

            # --- Gather test results from all processes ---
            if rank == 0:
                gathered_test_preds = [epoch_test_preds]
                gathered_test_labels = [epoch_test_labels]
                gathered_test_probs = [epoch_test_probs]
                for i in range(1, world_size):
                    worker_preds = [None]
                    worker_labels = [None]
                    worker_probs = [None]
                    dist.recv_object(worker_preds, src=i)
                    dist.recv_object(worker_labels, src=i)
                    dist.recv_object(worker_probs, src=i)
                    gathered_test_preds.append(worker_preds[0])
                    gathered_test_labels.append(worker_labels[0])
                    gathered_test_probs.append(worker_probs[0])
            else:
                dist.send_object([epoch_test_preds], dst=0)
                dist.send_object([epoch_test_labels], dst=0)
                dist.send_object([epoch_test_probs], dst=0)
            
            dist.barrier()

            if rank == 0:
                all_preds_list = [item for sublist in gathered_test_preds for item in sublist]
                all_labels_list = [item for sublist in gathered_test_labels for item in sublist]
                all_probs_list = [item for sublist in gathered_test_probs for item in sublist]

                # Note: This loss is also an approximation based on rank 0's data
                avg_test_loss = test_total_loss / len(test_sampler)
                
                if all_preds_list and all_labels_list:
                    all_preds = torch.cat(all_preds_list)
                    all_labels_float = torch.cat(all_labels_list)
                    all_probs = torch.cat(all_probs_list)
                    
                    smudge_mae = torch.nn.functional.l1_loss(all_probs, all_labels_float).item()
                    all_labels_binary = all_labels_float.long()
                    
                    tp = ((all_preds == 1) & (all_labels_binary == 1)).sum().item()
                    fp = ((all_preds == 1) & (all_labels_binary == 0)).sum().item()
                    fn = ((all_preds == 0) & (all_labels_binary == 1)).sum().item()
                    tn = ((all_preds == 0) & (all_labels_binary == 0)).sum().item()
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    accuracy = (tp + tn) / (tp + tn + fp + fn)

                    print(f'\nFinal Test Set Metrics:')
                    print(f'  Avg Test Loss: {avg_test_loss:.4f}')
                    print(f'  Test Metrics: Accuracy: {accuracy:.4f} | F1-Score: {f1_score:.4f} | Smudge MAE: {smudge_mae:.4f}')
                    print(f'    > Precision: {precision:.4f} | Recall: {recall:.4f}')
                    total_predicted_floods = tp + fp
                    print(f'    > Missed Floods (FN): {fn} | False Alarms (FP): {fp} | Total Predicted Floods: {total_predicted_floods}')

            # --- Plotting ---
            plot_loss_curve(
                history['epochs'],
                history['train_loss'],
                history['val_loss'],
                save_path=os.path.join(script_dir, 'loss_curve.png')
            )
            if epoch_test_labels and epoch_test_preds:
                plot_spatial_predictions(
                    epoch_test_labels[0],
                    epoch_test_preds[0],
                    test_dataset,
                    save_path=os.path.join(script_dir, 'spatial_prediction_map.png')
                )

    finally:
        cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
 
