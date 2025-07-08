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
            num_workers=2, # A few persistent workers to load day files
            prefetch_factor=2,
            persistent_workers=True,
        )
        val_day_loader = DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=None,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
        )
        test_day_loader = DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=None,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
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
            val_tp = 0
            val_fp = 0
            val_fn = 0
            
            val_progress_bar = tqdm(val_day_loader, desc=f"Epoch {epoch:02d} (Val)", unit="day", total=len(val_sampler), disable=(rank != 0 or not sys.stdout.isatty()))
            for day_data in val_progress_bar:
                day_loss, day_preds, day_labels, _ = test_one_day(model.module, day_data, device)
                val_total_loss += day_loss
                
                # Calculate local metrics for the batch
                val_tp += ((day_preds == 1) & (day_labels == 1)).sum().item()
                val_fp += ((day_preds == 1) & (day_labels == 0)).sum().item()
                val_fn += ((day_preds == 0) & (day_labels == 1)).sum().item()
            
            # --- Reduce (sum) results from all processes to rank 0 ---
            # This is more robust than gathering large lists of tensors.
            metrics_tensor = torch.tensor([val_tp, val_fp, val_fn], dtype=torch.float32).to(device)
            dist.reduce(metrics_tensor, dst=0, op=dist.ReduceOp.SUM)

            # --- Evaluation Metrics (on rank 0) ---
            if rank == 0:
                # Unpack the reduced tensor
                total_tp, total_fp, total_fn = metrics_tensor.cpu().numpy()

                # --- Store history for plotting ---
                avg_val_loss = val_total_loss / len(val_sampler) if len(val_sampler) > 0 else 0
                history['train_loss'].append(avg_epoch_loss)
                history['val_loss'].append(avg_val_loss) 
                history['epochs'].append(epoch)

                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                val_f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                print(f'Epoch: {epoch:02d}, Avg Train Loss: {avg_epoch_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1_score:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
                total_predicted_floods = total_tp + total_fp
                total_actual_floods = total_tp + total_fn
                print(f'    > Val Metrics: FN: {int(total_fn)} | FP: {int(total_fp)} | Total Predictions: {int(total_predicted_floods)} | Total Floods: {int(total_actual_floods)}')

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

            # --- Final Test Evaluation ---
            test_tp, test_fp, test_fn, test_tn = 0, 0, 0, 0
            l1_sum, num_elements = 0.0, 0
            # Data for plotting is only needed on rank 0
            plot_labels, plot_preds = None, None

            test_progress_bar = tqdm(test_day_loader, desc="Final Test", unit="day", total=len(test_sampler), disable=(rank != 0 or not sys.stdout.isatty()))
            for i, day_data in enumerate(test_progress_bar):
                day_loss, day_preds, day_labels, day_probs = test_one_day(model_to_test, day_data, device)
                
                # Calculate local metrics
                test_tp += ((day_preds == 1) & (day_labels == 1)).sum().item()
                test_fp += ((day_preds == 1) & (day_labels == 0)).sum().item()
                test_fn += ((day_preds == 0) & (day_labels == 1)).sum().item()
                test_tn += ((day_preds == 0) & (day_labels == 0)).sum().item()
                
                # Calculate local components for MAE
                l1_sum += torch.nn.functional.l1_loss(day_probs, day_labels, reduction='sum').item()
                num_elements += day_labels.numel()
                
                # Save the first day's data from rank 0 for plotting
                if rank == 0 and i == 0:
                    plot_labels = day_labels
                    plot_preds = day_preds

            # Reduce all metrics from all processes
            metrics_tensor = torch.tensor([test_tp, test_fp, test_fn, test_tn, l1_sum, num_elements], dtype=torch.float64).to(device)
            dist.reduce(metrics_tensor, dst=0, op=dist.ReduceOp.SUM)

            if rank == 0:
                # Unpack reduced metrics
                total_tp, total_fp, total_fn, total_tn, total_l1_sum, total_elements = metrics_tensor.cpu().numpy()

                # Calculate final global metrics
                smudge_mae = total_l1_sum / total_elements if total_elements > 0 else 0
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0
                
                # Note: This loss is an approximation based on rank 0's data
                avg_test_loss = 0 # This metric is not easily calculated globally

                print(f'\nFinal Test Set Metrics:')
                print(f'  Avg Test Loss (approx): {avg_test_loss:.4f}')
                print(f'  Test Metrics: Accuracy: {accuracy:.4f} | F1-Score: {f1_score:.4f} | Smudge MAE: {smudge_mae:.4f}')
                print(f'    > Precision: {precision:.4f} | Recall: {recall:.4f}')
                total_predicted_floods = total_tp + total_fp
                total_actual_floods = total_tp + total_fn
                print(f'    > Missed Floods (FN): {int(total_fn)} | False Alarms (FP): {int(total_fp)} | Total Predicted Floods: {int(total_predicted_floods)} | Total Floods: {int(total_actual_floods)}')

            # --- Plotting (only on rank 0) ---
            if rank == 0:
                plot_loss_curve(
                    history['epochs'],
                    history['train_loss'],
                    history['val_loss'],
                    save_path=os.path.join(script_dir, 'loss_curve.png')
                )
                if plot_labels is not None and plot_preds is not None:
                    # For the plot, we only need one day's worth of data
                    plot_spatial_predictions(
                        plot_labels,
                        plot_preds,
                        test_dataset,
                        save_path=os.path.join(script_dir, 'spatial_prediction_map.png')
                    )

    finally:
        cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
 
