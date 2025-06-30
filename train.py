import os
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from torch.cuda.amp import GradScaler
import threading
import queue

from FloodData import FloodDataset
from gcn_model import SpatioTemporalGCN

# --- Debugging: Disable cuDNN ---
# The cuDNN LSTM implementation can be sensitive to very large inputs.
# Disabling it can help diagnose if the issue is with cuDNN itself.
torch.backends.cudnn.enabled = False
# torch.backends.cuda.matmul.allow_tf32 = True

# --- Prefetcher for Data Loading ---
class Prefetcher:
    """
    A simple prefetcher to load data in a background thread.
    This helps to hide I/O latency by overlapping data loading with model computation.
    """
    def __init__(self, dataset, num_prefetch=1):
        self.dataset = dataset
        self.num_prefetch = num_prefetch
        self.queue = queue.Queue(maxsize=self.num_prefetch)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.data_iter = iter(self.dataset)
        self.thread.start()

    def _run(self):
        try:
            while True:
                # Get the next item from the dataset iterator
                try:
                    item = next(self.data_iter)
                except StopIteration:
                    # Dataset is exhausted
                    self.queue.put(None)  # Sentinel to signal end
                    return
                
                # Put the loaded item into the queue for the main thread
                self.queue.put(item)
        except Exception as e:
            print(f"Error in prefetcher thread: {e}")
            self.queue.put(e)

    def __iter__(self):
        return self

    def __next__(self):
        # Get an item from the queue
        item = self.queue.get()

        if item is None:
            # End of iteration
            raise StopIteration
        
        if isinstance(item, Exception):
            # Propagate exception from the worker thread
            raise item
            
        return item
    
    def __len__(self):
        return len(self.dataset)

def train_one_day(model, day_data, optimizer, device, batch_size, num_neighbors, scaler, num_workers):
    """
    Trains the model for one day's graph using mini-batching with NeighborLoader.
    """
    model.train()
    
    # Create a NeighborLoader for the current day's large graph.
    # This will create small subgraphs (mini-batches) to process.
    loader = NeighborLoader(
        day_data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=None,  # Use all nodes as training targets
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    total_loss = 0
    num_batches = 0
    # Process the full day's graph in mini-batches
    progress_bar = tqdm(loader, desc="Day's Batches", leave=False)
    for batch in progress_bar:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        # The model now operates on the smaller subgraph from the loader
        with torch.amp.autocast(device_type=device.type):
            out = model(batch)
            
            # --- Handle Class Imbalance on a per-batch basis ---
            num_positives = batch.y.sum()
            num_negatives = len(batch.y) - num_positives
            
            if num_positives > 0:
                pos_weight = num_negatives / num_positives
            else:
                pos_weight = torch.tensor(1.0, device=device)
            
            # --- Sanity Cap on pos_weight ---
            # Cap the weight to prevent excessively large loss values from a few samples
            pos_weight = torch.clamp(pos_weight, min=1.0, max=100.0)

            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            # The output 'out' and target 'batch.y' are correctly sized by the loader
            loss = criterion(out, batch.y)
        
        scaler.scale(loss).backward()
        
        # --- Gradient Clipping ---
        # Unscale the gradients before clipping
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
    """
    Evaluates the model for one day's graph using mini-batching with NeighborLoader.
    Returns loss, and the day's predictions and ground truth labels.
    """
    model.eval()
    
    loader = NeighborLoader(
        day_data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=None,
        shuffle=False,  # No need to shuffle for evaluation
        num_workers=num_workers,
        pin_memory=True,
    )

    total_loss = 0
    num_batches = 0
    
    day_preds, day_labels = [], []
    for batch in tqdm(loader, desc="Day's Batches (Test)", leave=False):
        batch = batch.to(device)
        
        with torch.amp.autocast(device_type=device.type):
            out = model(batch)
            
            # Use a simple, unweighted loss for evaluation
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(out, batch.y)
        
        total_loss += loss.item()
        num_batches += 1
        
        # Store predictions and labels for metric calculation
        # Convert logits to binary predictions (0 or 1)
        preds = (out.sigmoid() > 0.5).long()
        day_preds.append(preds.cpu())
        day_labels.append(batch.y.cpu())
        
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    # Concatenate all batch results for the day into single tensors
    if day_preds and day_labels:
        day_preds = torch.cat(day_preds, dim=0)
        day_labels = torch.cat(day_labels, dim=0)
    
    return avg_loss, day_preds, day_labels

def main():
    # --- Hyperparameters ---
    EPOCHS = 10
    LEARNING_RATE = 0.001
    HIDDEN_DIM = 128
    LSTM_LAYERS = 2
    GCN_LAYERS = 2
    BATCH_SIZE = 20000
    NEIGHBOR_SAMPLES = [10, 5]
    
    # --- Setup ---
    # Get the absolute path of the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use a reasonable number of workers
    # Reducing from 12 to 4 to lower system RAM usage from parallel data loading.
    num_workers = min(os.cpu_count(), 8)
    print(f"Using {num_workers} workers for data loading.")

    # --- Dataset ---
    # Create separate training and testing datasets
    # Using new root directories to trigger the parallelized pre-processing
    print("Loading training dataset...")
    train_dataset = FloodDataset(
        root=os.path.join(script_dir, 'data/flood_dataset_train_parallel'), 
        wldas_dir=os.path.join(script_dir, 'WLDAS_2012'),
        flood_csv=os.path.join(script_dir, 'USFD_v1.0.csv'),
        mode='train'
    )
    print("Training dataset loaded successfully.")

    print("Loading testing dataset...")
    test_dataset = FloodDataset(
        root=os.path.join(script_dir, 'data/flood_dataset_test_parallel'),
        wldas_dir=os.path.join(script_dir, 'WLDAS_2012'),
        flood_csv=os.path.join(script_dir, 'USFD_v1.0.csv'),
        mode='test'
    )
    print("Testing dataset loaded successfully.")
    
    # --- Model ---
    sample_data = train_dataset[0]
    static_feature_dim = sample_data.x_static.shape[1]
    dynamic_feature_dim = sample_data.x_dynamic.shape[2]

    model = SpatioTemporalGCN(
        static_feature_dim=static_feature_dim,
        dynamic_feature_dim=dynamic_feature_dim,
        hidden_dim=HIDDEN_DIM,
        lstm_layers=LSTM_LAYERS,
        gcn_layers=GCN_LAYERS,
    ).to(device)

    # --- Compile the model for a significant speedup ---
    print("Compiling the model (this may take a moment)...")
    model = torch.compile(model)
    print("Model compiled successfully.")

    print("Model architecture:")
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    
    # --- Learning Rate Scheduler ---
    # Helps in fine-tuning the model by gradually reducing the learning rate.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        epoch_total_loss = 0
        num_days = 0
        # Iterate over each day in the dataset
        train_prefetcher = Prefetcher(train_dataset, num_prefetch=2)
        for i, day_data in enumerate(tqdm(train_prefetcher, desc=f"Epoch {epoch:02d}", unit="day", total=len(train_dataset))):
            day_loss = train_one_day(model, day_data, optimizer, device, BATCH_SIZE, NEIGHBOR_SAMPLES, scaler, num_workers)
            epoch_total_loss += day_loss
            num_days += 1
        
        avg_epoch_loss = epoch_total_loss / num_days if num_days > 0 else 0
        
        # --- Evaluation Step ---
        print("Starting evaluation...")
        test_total_loss = 0
        num_test_days = 0
        
        epoch_preds, epoch_labels = [], []
        
        test_prefetcher = Prefetcher(test_dataset, num_prefetch=2)
        for day_data in tqdm(test_prefetcher, desc=f"Epoch {epoch:02d} (Test)", unit="day", total=len(test_dataset)):
            day_loss, day_preds, day_labels = test_one_day(model, day_data, device, BATCH_SIZE, NEIGHBOR_SAMPLES, num_workers)
            test_total_loss += day_loss
            
            if day_preds.numel() > 0:
                epoch_preds.append(day_preds)
                epoch_labels.append(day_labels)
                
            num_test_days += 1
            
        avg_test_loss = test_total_loss / num_test_days if num_test_days > 0 else 0

        print(f'Epoch: {epoch:02d}, Avg Train Loss: {avg_epoch_loss:.4f}, Avg Test Loss: {avg_test_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # --- Calculate and Print Detailed Metrics ---
        if epoch_preds and epoch_labels:
            all_preds = torch.cat(epoch_preds)
            all_labels = torch.cat(epoch_labels).long()

            true_positives = ((all_preds == 1) & (all_labels == 1)).sum().item()
            false_positives = ((all_preds == 1) & (all_labels == 0)).sum().item()
            false_negatives = ((all_preds == 0) & (all_labels == 1)).sum().item()
            true_negatives = ((all_preds == 0) & (all_labels == 0)).sum().item()
            
            total_nodes = len(all_labels)
            total_floods = true_positives + false_negatives

            accuracy = (true_positives + true_negatives) / total_nodes if total_nodes > 0 else 0

            print(f'  Test Metrics: Accuracy: {accuracy:.4f} | Total Actual Floods: {total_floods}')
            print(f'    > Missed Floods (FN): {false_negatives} | False Alarms (FP): {false_positives}')

        # Step the scheduler after each epoch
        scheduler.step()
        
    print("Training finished.")

    # --- Save the final model ---
    print("Saving the final model...")
    save_dir = os.path.join(script_dir, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'final_flood_predictor.pth')
    # We save the state_dict of the compiled model's original module
    torch.save(model._orig_mod.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main() 