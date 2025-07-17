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
from torch_geometric.loader import DataLoader
from torch.cuda.amp import GradScaler
import threading
import queue
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
import gc
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import argparse
import yaml

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

def train_one_batch(model, batch_data, optimizer, device, scaler):
    model.train()
    batch_data = batch_data.to(device)
    optimizer.zero_grad(set_to_none=True)
    
    with torch.amp.autocast(device_type=device.type):
        out = model(batch_data)
        
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"WARNING: NaN or Inf detected in model output!")
        if torch.isnan(batch_data.y).any() or torch.isinf(batch_data.y).any():
            print(f"WARNING: NaN or Inf detected in targets!")
        
        # --- Weighted Loss for Backpropagation ---
        num_positives = batch_data.y.sum()
        num_negatives = len(batch_data.y) - num_positives
        pos_weight = num_negatives / num_positives if num_positives > 0 else torch.tensor(1.0, device=device)
        
        if torch.isnan(pos_weight) or torch.isinf(pos_weight):
            print(f"WARNING: Invalid pos_weight! num_positives={num_positives}, num_negatives={num_negatives}")
            pos_weight = torch.tensor(1.0, device=device)
        
        pos_weight = torch.clamp(pos_weight, min=1.0, max=10000.0)
        criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_weighted = criterion_weighted(out, batch_data.y)
        
        if torch.isnan(loss_weighted) or torch.isinf(loss_weighted):
            print(f"CRITICAL: NaN or Inf weighted loss detected! Skipping batch.")
            return 0

        # --- Unweighted Loss for Logging ---
        # We calculate this separately for plotting, so we can compare apples to apples.
        criterion_unweighted = nn.BCEWithLogitsLoss()
        loss_unweighted = criterion_unweighted(out, batch_data.y)
    
    # Use the weighted loss for the backward pass, as it's better for training
    scaler.scale(loss_weighted).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    
    # Return the unweighted loss for plotting and logging
    return loss_unweighted.item()

@torch.no_grad()
def test_one_batch(model, batch_data, device):
    model.eval()
    batch_data = batch_data.to(device)
    
    with torch.amp.autocast(device_type=device.type):
        out = model(batch_data)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(out, batch_data.y)
    
    probs = out.sigmoid()
    preds = (probs > 0.5).long()
    
    return loss.item(), preds.cpu(), batch_data.y.cpu(), probs.cpu()

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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")

def plot_spatial_predictions(day_labels, day_preds, dataset, shapefile_path, save_path):
    """Generates and saves a spatial plot of ground truth vs. predictions."""
    try:
        us_states = gpd.read_file(shapefile_path)
    except Exception as e:
        print(f"Could not read shapefile from {shapefile_path}, skipping map plot: {e}")
        return

    # --- Get Geospatial Info from the dataset object's graph_data attribute ---
    # The dataset object now correctly stores its own boundaries within its graph_data.
    lat_min, lat_max = dataset.graph_data.lat_min.item(), dataset.graph_data.lat_max.item()
    lon_min, lon_max = dataset.graph_data.lon_min.item(), dataset.graph_data.lon_max.item()
    n_rows, n_cols = dataset.graph_data.n_rows.item(), dataset.graph_data.n_cols.item()

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

    # --- Overlay heatmaps for different categories ---
    # True Positives (Green): Correctly predicted floods
    tp_mask = (truth_grid > 0) & (pred_grid > 0)
    tp_lats = lat_grid[tp_mask]
    tp_lons = lon_grid[tp_mask]
    if tp_lats.size > 0:
        ax.scatter(tp_lons, tp_lats, color='green', alpha=0.7, label='Correctly Predicted', marker='s', s=10)

    # False Negatives (Blue): Actual floods that were missed
    fn_mask = (truth_grid > 0) & (pred_grid == 0)
    fn_lats = lat_grid[fn_mask]
    fn_lons = lon_grid[fn_mask]
    if fn_lats.size > 0:
        ax.scatter(fn_lons, fn_lats, color='blue', alpha=0.7, label='Missed Flood', marker='s', s=10)

    # False Positives (Red): Predicted floods that were not actual floods
    fp_mask = (truth_grid == 0) & (pred_grid > 0)
    fp_lats = lat_grid[fp_mask]
    fp_lons = lon_grid[fp_mask]
    if fp_lats.size > 0:
        ax.scatter(fp_lons, fp_lats, color='red', alpha=0.7, label='False Alarm', marker='s', s=10)

    ax.set_title('Spatial Comparison of Flood Prediction Accuracy (51st Test Day)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Spatial prediction map saved to {save_path}")

def main(rank, world_size, args, config):
    setup(rank, world_size)
    
    try:
        # --- Hyperparameters ---
        # All hyperparameters are now loaded from the config file.
        EPOCHS = config['training']['epochs']
        if args.finetune:
            EPOCHS = config['training']['finetune_epochs']
        LEARNING_RATE = config['training']['learning_rate'] * world_size # Scale learning rate
        
        # --- Setup ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        device = torch.device(f'cuda:{rank}')
        
        # --- Dataset ---
        wldas_base_dir = os.path.join(script_dir, config['data']['wldas_dir'])
        
        # With the new data pipeline, we instantiate one dataset and then split it.
        # The pre-processing check is now handled automatically within the FloodDataset class.
        # All ranks will either wait for rank 0 to process, or load from cache.
        if rank == 0:
            print("--- Loading and preparing dataset... ---")
        
        full_dataset = FloodDataset(
            root=os.path.join(script_dir, config['data']['root_dir']),
            wldas_dir=wldas_base_dir,
            flood_csv=os.path.join(script_dir, config['data']['flood_csv']),
            config=config # Pass the whole config dict
        )
        
        # --- Split the dataset into train, validation, and test sets ---
        # We perform a chronological split to prevent data leakage.
        total_size = len(full_dataset)
        train_ratio = config['training']['split_ratios']['train']
        val_ratio = config['training']['split_ratios']['val']
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size

        indices = list(range(total_size))
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size:]

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)
        
        if rank == 0:
            print(f"Chronological split: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples.")
        
        # The sampler for training should shuffle the order of samples within the training period each epoch.
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        # For validation and testing, we do not shuffle to ensure consistent evaluation.
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        # Use the PyG DataLoader which understands how to batch graph data.
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            sampler=train_sampler,
            num_workers=config['training']['dataloader_workers'],
            prefetch_factor=2,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            sampler=val_sampler,
            num_workers=config['training']['dataloader_workers'],
            prefetch_factor=2,
            persistent_workers=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            sampler=test_sampler,
            num_workers=config['training']['dataloader_workers'],
            prefetch_factor=2,
            persistent_workers=True,
        )

        # --- Model ---
        # Load one sample to infer model dimensions. We get it from the full dataset.
        sample_data = full_dataset[0]
        model = SpatioTemporalGCN(
            static_feature_dim=sample_data.x_static.shape[1],
            dynamic_feature_dim=sample_data.x_dynamic.shape[2],
            sequence_length=sample_data.x_dynamic.shape[1],
            hidden_dim=config['model']['hidden_dim'],
            gcn_layers=config['model']['gcn_layers'],
            dropout=config['model']['dropout']
        ).to(device)
        
        if args.finetune:
            best_model_path = os.path.join(script_dir, config['training']['save_dir'], config['training']['best_model_name'])
            if os.path.exists(best_model_path):
                # Load state dict on CPU first to avoid device mismatches, then move model to correct device
                model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
                model.to(device)
                if rank == 0:
                    print(f"Loaded best model from {best_model_path} for fine-tuning.")
            elif rank == 0:
                print(f"WARNING: --finetune flag was set, but best model not found at {best_model_path}. Starting from scratch.")
        
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
            
            # The loader now yields batches of day graphs at a time
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d}", unit="batch", total=len(train_loader), disable=(rank != 0 or not sys.stdout.isatty()))
            for batch_data in progress_bar:
                batch_loss = train_one_batch(model, batch_data, optimizer, device, scaler)
                epoch_total_loss += batch_loss
                progress_bar.set_postfix(loss=f'{batch_loss:.4f}')
            
            avg_epoch_loss = epoch_total_loss / len(train_loader) if len(train_loader) > 0 else 0
            
            # --- Evaluation Step ---
            val_total_loss = 0
            val_tp = 0
            val_fp = 0
            val_fn = 0
            
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch:02d} (Val)", unit="batch", total=len(val_loader), disable=(rank != 0 or not sys.stdout.isatty()))
            for batch_data in val_progress_bar:
                day_loss, day_preds, day_labels, _ = test_one_batch(model.module, batch_data, device)
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
                avg_val_loss = val_total_loss / len(val_loader) if len(val_loader) > 0 else 0
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
                save_dir = os.path.join(script_dir, config['training']['save_dir'])
                os.makedirs(save_dir, exist_ok=True)
                
                # Save the latest model
                latest_save_path = os.path.join(save_dir, f'flood_predictor_epoch_{epoch}.pth')
                torch.save(model.module.state_dict(), latest_save_path)
                
                # Save the best model if validation F1 score has improved
                if val_f1_score > best_val_f1_score:
                    best_val_f1_score = val_f1_score
                    best_save_path = os.path.join(save_dir, config['training']['best_model_name'])
                    torch.save(model.module.state_dict(), best_save_path)
                    print(f'New best model saved to {best_save_path} (Validation F1-Score: {best_val_f1_score:.4f})')

            scheduler.step()
            
        if rank == 0:
            print("Training finished.")

        # --- Final Evaluation on the Test Set (runs on all ranks) ---
        if rank == 0:
            print("--- Running Final Evaluation on Test Set ---")
        
        # Load the best model for final testing
        best_model_path = os.path.join(script_dir, config['training']['save_dir'], config['training']['best_model_name'])
        model_to_test = model.module
        if os.path.exists(best_model_path):
            # Load state dict on CPU first to avoid device mismatches
            model_to_test.load_state_dict(torch.load(best_model_path, map_location='cpu'))
            model_to_test.to(device) # Move model to the correct device for this rank
            if rank == 0:
                print("Loaded best model for final evaluation.")

        # The hardcoded metadata is no longer needed, it's in the dataset object.
        underlying_dataset = test_dataset.dataset
        if rank == 0 and not all(hasattr(underlying_dataset, attr) for attr in ['lat_min', 'lat_max', 'lon_min', 'lon_max', 'n_rows', 'n_cols']):
             print("Warning: Geospatial metadata not found in dataset object. Map plotting may fail.")

        # --- Final Test Evaluation Loop ---
        test_tp, test_fp, test_fn, test_tn = 0, 0, 0, 0
        l1_sum, num_elements = 0.0, 0
        plot_labels, plot_preds = None, None

        test_progress_bar = tqdm(test_loader, desc="Final Test", unit="batch", total=len(test_loader), disable=(rank != 0 or not sys.stdout.isatty()))
        for i, batch_data in enumerate(test_progress_bar):
            day_loss, day_preds, day_labels, day_probs = test_one_batch(model_to_test, batch_data, device)
            
            test_tp += ((day_preds == 1) & (day_labels == 1)).sum().item()
            test_fp += ((day_preds == 1) & (day_labels == 0)).sum().item()
            test_fn += ((day_preds == 0) & (day_labels == 1)).sum().item()
            test_tn += ((day_preds == 0) & (day_labels == 0)).sum().item()
            
            l1_sum += torch.nn.functional.l1_loss(day_probs, day_labels, reduction='sum').item()
            num_elements += day_labels.numel()
            
            if rank == 0 and i == 50:
                # To plot a spatial map, we need to "unbatch" the first batch.
                # The 'ptr' attribute tells us where each graph's nodes start and end.
                node_slice_for_first_graph = slice(batch_data.ptr[0], batch_data.ptr[1])
                
                # Slice both the predictions and the labels from the batch tensors
                # to ensure they are consistent.
                plot_preds = day_preds[node_slice_for_first_graph]
                plot_labels = day_labels[node_slice_for_first_graph]


        # Reduce all metrics from all processes
        metrics_tensor = torch.tensor([test_tp, test_fp, test_fn, test_tn, l1_sum, num_elements], dtype=torch.float64).to(device)
        dist.reduce(metrics_tensor, dst=0, op=dist.ReduceOp.SUM)

        # --- Final Metrics and Plotting (on rank 0) ---
        if rank == 0:
            total_tp, total_fp, total_fn, total_tn, total_l1_sum, total_elements = metrics_tensor.cpu().numpy()

            smudge_mae = total_l1_sum / total_elements if total_elements > 0 else 0
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0
            
            avg_test_loss = 0

            print(f'\nFinal Test Set Metrics:')
            print(f'  Avg Test Loss (approx): {avg_test_loss:.4f}')
            print(f'  Test Metrics: Accuracy: {accuracy:.4f} | F1-Score: {f1_score:.4f} | Smudge MAE: {smudge_mae:.4f}')
            print(f'    > Precision: {precision:.4f} | Recall: {recall:.4f}')
            total_predicted_floods = total_tp + total_fp
            total_actual_floods = total_tp + total_fn
            print(f'    > Missed Floods (FN): {int(total_fn)} | False Alarms (FP): {int(total_fp)} | Total Predicted Floods: {int(total_predicted_floods)} | Total Floods: {int(total_actual_floods)}')

            plot_dir = os.path.join(script_dir, 'plots')
            plot_loss_curve(
                history['epochs'],
                history['train_loss'],
                history['val_loss'],
                save_path=os.path.join(plot_dir, 'loss_curve.png')
            )
            if plot_labels is not None and plot_preds is not None:
                shapefile_path = os.path.join(script_dir, config['data']['shapefile_path'])
                plot_spatial_predictions(
                    plot_labels,
                    plot_preds,
                    test_dataset.dataset, # Pass the underlying full dataset for geo-metadata
                    shapefile_path,
                    save_path=os.path.join(plot_dir, 'spatial_prediction_map.png')
                )

    finally:
        cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flood Predictor Training Script')
    parser.add_argument('--finetune', action='store_true', help='Load the best model and run for one epoch before testing.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    # --- Load Configuration ---
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)

    # --- Single-threaded Data Pre-processing Check ---
    # By instantiating the dataset once in the main process *before* spawning,
    # we ensure that the potentially long pre-processing step is done only once.
    # All spawned processes will then find the cached data and load it instantly.
    print("--- Initializing Dataset: Pre-processing will run if needed. ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    FloodDataset(
        root=os.path.join(script_dir, config['data']['root_dir']),
        wldas_dir=os.path.join(script_dir, config['data']['wldas_dir']),
        flood_csv=os.path.join(script_dir, config['data']['flood_csv']),
        config=config
    )
    print("--- Dataset is ready. Starting training... ---")

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size, args, config), nprocs=world_size, join=True)
 
