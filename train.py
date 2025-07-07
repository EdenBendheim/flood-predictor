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

def train_one_day(model, day_data, optimizer, device, batch_size, num_neighbors, scaler, num_workers):
    model.train()
    loader = NeighborLoader(
        day_data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=None,
        shuffle=False,  # Sampler handles shuffling
        num_workers=num_workers,
        pin_memory=True,
    )

    total_loss = 0
    num_batches = 0
    progress_bar = tqdm(loader, desc="Day's Batches", leave=False, disable=not sys.stdout.isatty())
    for batch in progress_bar:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type=device.type):
            out = model(batch)
            
            # Debug: Check for NaN/Inf in model output
            if torch.isnan(out).any() or torch.isinf(out).any():
                print(f"WARNING: NaN or Inf detected in model output!")
                
            # Debug: Check for NaN/Inf in targets
            if torch.isnan(batch.y).any() or torch.isinf(batch.y).any():
                print(f"WARNING: NaN or Inf detected in targets!")
            
            num_positives = batch.y.sum()
            num_negatives = len(batch.y) - num_positives
            pos_weight = num_negatives / num_positives if num_positives > 0 else torch.tensor(1.0, device=device)
            
            # Debug: Check pos_weight calculation
            if torch.isnan(pos_weight) or torch.isinf(pos_weight):
                print(f"WARNING: Invalid pos_weight! num_positives={num_positives}, num_negatives={num_negatives}")
                pos_weight = torch.tensor(1.0, device=device)
            
            # Clamp the pos_weight to a reasonable value to prevent loss from becoming nan/inf
            pos_weight = torch.clamp(pos_weight, min=1.0, max=10000.0)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = criterion(out, batch.y)
            
            # Debug: Check loss value
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"CRITICAL: NaN or Inf loss detected! Skipping batch.")
                continue
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix(loss=f'{loss.item():.4f}')
        
    # Explicitly clean up to free resources from the NeighborLoader
    del loader
    gc.collect()
    return total_loss / num_batches if num_batches > 0 else 0

@torch.no_grad()
def test_one_day(model, day_data, device, batch_size, num_neighbors, num_workers):
    model.eval()
    loader = NeighborLoader(
        day_data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    total_loss = 0
    num_batches = 0
    day_preds, day_labels, day_probs = [], [], []
    for batch in tqdm(loader, desc="Day's Batches (Test)", leave=False, disable=not sys.stdout.isatty()):
        batch = batch.to(device)
        with torch.amp.autocast(device_type=device.type):
            out = model(batch)
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(out, batch.y)
        
        total_loss += loss.item()
        num_batches += 1
        
        probs = out.sigmoid()
        preds = (probs > 0.5).long()
        day_preds.append(preds.cpu())
        day_labels.append(batch.y.cpu())
        day_probs.append(probs.cpu())
        
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    if day_preds and day_labels:
        day_preds = torch.cat(day_preds, dim=0)
        day_labels = torch.cat(day_labels, dim=0)
        day_probs = torch.cat(day_probs, dim=0)
    
    # Explicitly clean up to free resources from the NeighborLoader
    del loader
    gc.collect()
    return avg_loss, day_preds, day_labels, day_probs

def main(rank, world_size):
    setup(rank, world_size)
    
    try:
        # --- Hyperparameters ---
        EPOCHS = 10
        LEARNING_RATE = 0.001 * world_size # Scale learning rate
        HIDDEN_DIM = 256 # Increased model capacity
        GCN_LAYERS = 8
        BATCH_SIZE = 256000 # Reduced batch size to prevent CUDA errors with a larger model
        NEIGHBOR_SAMPLES = [15, 10, 5, 5, 5, 5, 5, 5] # Deeper neighborhood sampling for 4 GCN layers
        
        # --- Setup ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        device = torch.device(f'cuda:{rank}')
        # num_workers for the inner NeighborLoader, per GPU.
        num_workers_neighbor = 4
        
        # --- Dataset ---
        train_dataset = FloodDataset(
            root=os.path.join(script_dir, 'data/flood_dataset_train_parallel'), 
            wldas_dir=os.path.join(script_dir, 'WLDAS_2012'),
            flood_csv=os.path.join(script_dir, 'USFD_v1.0.csv'),
            mode='train'
        )
        test_dataset = FloodDataset(
            root=os.path.join(script_dir, 'data/flood_dataset_test_parallel'),
            wldas_dir=os.path.join(script_dir, 'WLDAS_2012'),
            flood_csv=os.path.join(script_dir, 'USFD_v1.0.csv'),
            mode='test'
        )
        
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
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
        best_f1_score = 0.0

        # --- Training Loop ---
        for epoch in range(1, EPOCHS + 1):
            train_sampler.set_epoch(epoch)
            epoch_total_loss = 0
            
            # Iterate over the robust day loader
            for day_data in tqdm(train_day_loader, desc=f"Epoch {epoch:02d}", unit="day", total=len(train_sampler), disable=not sys.stdout.isatty()):
                day_loss = train_one_day(model, day_data, optimizer, device, BATCH_SIZE, NEIGHBOR_SAMPLES, scaler, num_workers_neighbor)
                epoch_total_loss += day_loss
            
            avg_epoch_loss = epoch_total_loss / len(train_sampler)
            
            # --- Evaluation Step (on rank 0) ---
            if rank == 0:
                test_total_loss = 0
                epoch_preds, epoch_labels, epoch_probs = [], [], []
                
                for day_data in tqdm(test_day_loader, desc=f"Epoch {epoch:02d} (Test)", unit="day", total=len(test_sampler), disable=not sys.stdout.isatty()):
                    day_loss, day_preds, day_labels, day_probs = test_one_day(model.module, day_data, device, BATCH_SIZE, NEIGHBOR_SAMPLES, num_workers_neighbor)
                    test_total_loss += day_loss
                    if day_preds.numel() > 0:
                        epoch_preds.append(day_preds)
                        epoch_labels.append(day_labels)
                        epoch_probs.append(day_probs)
                
                avg_test_loss = test_total_loss / len(test_sampler)
                print(f'Epoch: {epoch:02d}, Avg Train Loss: {avg_epoch_loss:.4f}, Avg Test Loss: {avg_test_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
                
                if epoch_preds and epoch_labels:
                    all_preds = torch.cat(epoch_preds)
                    all_labels_float = torch.cat(epoch_labels)
                    all_probs = torch.cat(epoch_probs)
                    
                    # --- New Metric: Mean Absolute Error on smudged labels ---
                    smudge_mae = torch.nn.functional.l1_loss(all_probs, all_labels_float).item()
                    
                    # --- Original Binary Metrics ---
                    all_labels_binary = all_labels_float.long()
                    
                    # Calculate metrics
                    tp = ((all_preds == 1) & (all_labels_binary == 1)).sum().item()
                    fp = ((all_preds == 1) & (all_labels_binary == 0)).sum().item()
                    fn = ((all_preds == 0) & (all_labels_binary == 1)).sum().item()
                    tn = ((all_preds == 0) & (all_labels_binary == 0)).sum().item()
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    accuracy = (tp + tn) / (tp + tn + fp + fn)

                    print(f'  Test Metrics: Accuracy: {accuracy:.4f} | F1-Score: {f1_score:.4f} | Smudge MAE: {smudge_mae:.4f}')
                    print(f'    > Precision: {precision:.4f} | Recall: {recall:.4f}')
                    total_predicted_floods = tp + fp
                    print(f'    > Missed Floods (FN): {fn} | False Alarms (FP): {fp} | Total Predicted Floods: {total_predicted_floods}')

                # --- Save Model Checkpoints ---
                save_dir = os.path.join(script_dir, 'saved_models')
                os.makedirs(save_dir, exist_ok=True)
                
                # Save the latest model
                latest_save_path = os.path.join(save_dir, f'flood_predictor_epoch_{epoch}.pth')
                torch.save(model.module.state_dict(), latest_save_path)
                
                # Save the best model if F1 score has improved
                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    best_save_path = os.path.join(save_dir, 'best_flood_predictor.pth')
                    torch.save(model.module.state_dict(), best_save_path)
                    print(f'New best model saved to {best_save_path} (F1-Score: {best_f1_score:.4f})')

            scheduler.step()
            
        if rank == 0:
            print("Training finished.")
            
    finally:
        cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
 
