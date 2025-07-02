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
from torch.utils.data.distributed import DistributedSampler

from FloodData import FloodDataset
from gcn_model import SpatioTemporalGCN

# --- Performance Tuning: Enable cuDNN ---
# Enables cuDNN for GPU acceleration, which is crucial for performance.
# The benchmark mode finds the best algorithm for the specific input sizes.
torch.backends.cudnn.benchmark = True

# --- Prefetcher for Data Loading ---
class Prefetcher:
    def __init__(self, dataset, sampler, num_prefetch=1):
        self.dataset = dataset
        self.sampler = sampler
        self.num_prefetch = num_prefetch
        self.queue = queue.Queue(maxsize=self.num_prefetch)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.data_iter = iter(self.dataset)
        self.thread.start()

    def _run(self):
        try:
            for index in self.sampler:
                item = self.dataset[index]
                self.queue.put(item)
            self.queue.put(None)  # Sentinel to signal end
        except Exception as e:
            print(f"Error in prefetcher thread: {e}")
            self.queue.put(e)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item

    def __len__(self):
        return len(self.sampler)

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
            num_positives = batch.y.sum()
            num_negatives = len(batch.y) - num_positives
            pos_weight = num_negatives / num_positives if num_positives > 0 else torch.tensor(1.0, device=device)
            pos_weight = torch.clamp(pos_weight, min=1.0, max=100.0)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = criterion(out, batch.y)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix(loss=f'{loss.item():.4f}')
        
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
    day_preds, day_labels = [], []
    for batch in tqdm(loader, desc="Day's Batches (Test)", leave=False, disable=not sys.stdout.isatty()):
        batch = batch.to(device)
        with torch.amp.autocast(device_type=device.type):
            out = model(batch)
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(out, batch.y)
        
        total_loss += loss.item()
        num_batches += 1
        preds = (out.sigmoid() > 0.5).long()
        day_preds.append(preds.cpu())
        day_labels.append(batch.y.cpu())
        
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    if day_preds and day_labels:
        day_preds = torch.cat(day_preds, dim=0)
        day_labels = torch.cat(day_labels, dim=0)
    
    return avg_loss, day_preds, day_labels

def main(rank, world_size):
    setup(rank, world_size)
    
    # --- Hyperparameters ---
    EPOCHS = 10
    LEARNING_RATE = 0.001 * world_size # Scale learning rate
    HIDDEN_DIM = 124 # Increased model capacity
    LSTM_LAYERS = 1
    GCN_LAYERS = 2
    BATCH_SIZE = 50000 # Increased batch size per GPU
    NEIGHBOR_SAMPLES = [10, 5] # Deeper neighborhood sampling
    
    # --- Setup ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device(f'cuda:{rank}')
    num_workers = 12 // world_size # Allocate CPU cores per GPU
    
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

    # --- Model ---
    sample_data = train_dataset[0]
    model = SpatioTemporalGCN(
        static_feature_dim=sample_data.x_static.shape[1],
        dynamic_feature_dim=sample_data.x_dynamic.shape[2],
        hidden_dim=HIDDEN_DIM,
        lstm_layers=LSTM_LAYERS,
        gcn_layers=GCN_LAYERS,
    ).to(device)
    
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(enabled=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # --- Best model tracking ---
    best_val_loss = float('inf')

    # --- Training Loop ---
    for epoch in range(1, EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        epoch_total_loss = 0
        
        train_prefetcher = Prefetcher(train_dataset, train_sampler)
        for day_data in tqdm(train_prefetcher, desc=f"Epoch {epoch:02d}", unit="day", total=len(train_prefetcher), disable=not sys.stdout.isatty()):
            day_loss = train_one_day(model, day_data, optimizer, device, BATCH_SIZE, NEIGHBOR_SAMPLES, scaler, num_workers)
            epoch_total_loss += day_loss
        
        avg_epoch_loss = epoch_total_loss / len(train_prefetcher)
        
        # --- Evaluation Step (on rank 0) ---
        if rank == 0:
            test_total_loss = 0
            epoch_preds, epoch_labels = [], []
            
            test_prefetcher = Prefetcher(test_dataset, test_sampler)
            for day_data in tqdm(test_prefetcher, desc=f"Epoch {epoch:02d} (Test)", unit="day", total=len(test_prefetcher), disable=not sys.stdout.isatty()):
                day_loss, day_preds, day_labels = test_one_day(model.module, day_data, device, BATCH_SIZE, NEIGHBOR_SAMPLES, num_workers)
                test_total_loss += day_loss
                if day_preds.numel() > 0:
                    epoch_preds.append(day_preds)
                    epoch_labels.append(day_labels)
            
            avg_test_loss = test_total_loss / len(test_sampler)
            print(f'Epoch: {epoch:02d}, Avg Train Loss: {avg_epoch_loss:.4f}, Avg Test Loss: {avg_test_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
            
            if epoch_preds and epoch_labels:
                all_preds = torch.cat(epoch_preds)
                all_labels = torch.cat(epoch_labels).long()
                
                # Calculate metrics
                tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
                fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
                fn = ((all_preds == 0) & (all_labels == 1)).sum().item()
                tn = ((all_preds == 0) & (all_labels == 0)).sum().item()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (tp + tn) / (tp + tn + fp + fn)

                print(f'  Test Metrics: Accuracy: {accuracy:.4f} | F1-Score: {f1_score:.4f}')
                print(f'    > Precision: {precision:.4f} | Recall: {recall:.4f}')
                print(f'    > Missed Floods (FN): {fn} | False Alarms (FP): {fp}')

            # --- Save Model Checkpoints ---
            save_dir = os.path.join(script_dir, 'saved_models')
            os.makedirs(save_dir, exist_ok=True)
            
            # Save the latest model
            latest_save_path = os.path.join(save_dir, f'flood_predictor_epoch_{epoch}.pth')
            torch.save(model.module.state_dict(), latest_save_path)
            print(f'Epoch {epoch} model saved to {latest_save_path}')

            # Save the best model if validation loss has improved
            if avg_test_loss < best_val_loss:
                best_val_loss = avg_test_loss
                best_save_path = os.path.join(save_dir, 'best_flood_predictor.pth')
                torch.save(model.module.state_dict(), best_save_path)
                print(f'New best model saved to {best_save_path} (Val Loss: {best_val_loss:.4f})')

        scheduler.step()
        
    if rank == 0:
        print("Training finished.")
        
    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
 
