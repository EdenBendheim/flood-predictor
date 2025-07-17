import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

class SpatioTemporalGCN(nn.Module):
    def __init__(self, static_feature_dim, dynamic_feature_dim, sequence_length, hidden_dim, gcn_layers, dropout):
        super(SpatioTemporalGCN, self).__init__()
        self.dropout = dropout

        # Calculate the size of the flattened dynamic features
        flattened_dynamic_dim = dynamic_feature_dim * sequence_length
        combined_feature_dim = flattened_dynamic_dim + static_feature_dim

        # Batch norm for the initial combined input features
        self.input_bn = BatchNorm(combined_feature_dim)

        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        if gcn_layers > 0:
            # First GCN layer and its corresponding BatchNorm
            self.gcn_layers.append(GCNConv(combined_feature_dim, hidden_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))
            
            # Subsequent GCN layers and their BatchNorms
            for _ in range(gcn_layers - 1):
                self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
                self.batch_norms.append(BatchNorm(hidden_dim))

        # This layer is now only needed if there are no GCN layers at all.
        self.feature_combiner = nn.Linear(combined_feature_dim, hidden_dim)

        # Output layer
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        # In PyG, the 'data' object might not have a batch dimension if you're processing a single graph
        # The LSTM expects a batch dimension, so we unsqueeze.
        x_static, x_dynamic, edge_index, edge_weight = data.x_static, data.x_dynamic, data.edge_index, data.edge_weight
        
        # --- Input Sanitization ---
        x_static = torch.nan_to_num(x_static, nan=0.0, posinf=1e6, neginf=-1e6)
        x_dynamic = torch.nan_to_num(x_dynamic, nan=0.0, posinf=1e6, neginf=-1e6)
        if edge_weight is not None:
            edge_weight = torch.nan_to_num(edge_weight, nan=0.0, posinf=1e6, neginf=-1e6)
            edge_weight = torch.clamp(edge_weight, min=0.0, max=1e6)
        
        # 1. Flatten the temporal dimension of dynamic features
        # x_dynamic shape: (num_nodes, seq_len, dynamic_features)
        # We flatten it to (num_nodes, seq_len * dynamic_features)
        dynamic_features_flattened = x_dynamic.reshape(x_dynamic.size(0), -1)

        # 2. Combine static and flattened dynamic features
        # Shape: (num_nodes, static_dim + flattened_dynamic_dim)
        combined_features = torch.cat([x_static, dynamic_features_flattened], dim=1)
        
        # --- Feature Sanitization ---
        combined_features = torch.nan_to_num(combined_features, nan=0.0, posinf=1e6, neginf=-1e6)
        combined_features = torch.clamp(combined_features, min=-1e6, max=1e6)

        # 3. Normalize the combined input features before feeding them to the first layer
        x = self.input_bn(combined_features)
        
        # --- Post-BatchNorm Sanitization ---
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = torch.clamp(x, min=-10.0, max=10.0)

        # 3. Process combined features with GCN
        # Note: GCNConv expects edge_weight as an argument in the forward pass
        
        # --- Safeguard for empty edge_index ---
        # If a batch has no edges, GCN layers will fail. Skip them.
        if edge_index.numel() > 0 and len(self.gcn_layers) > 0:
            for i, (conv, bn) in enumerate(zip(self.gcn_layers, self.batch_norms)):
                # Pass edge_weight to the GCN layer. If it's None, GCNConv handles it.
                x = conv(x, edge_index, edge_weight=edge_weight)
                
                # --- Layer-wise Sanitization ---
                x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
                x = torch.clamp(x, min=-10.0, max=10.0)
                
                x = bn(x) # Apply Batch Normalization
                
                # --- Post-BatchNorm Sanitization ---
                x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
                x = torch.clamp(x, min=-10.0, max=10.0)
                
                x = F.relu(x)
                # It's common to apply dropout after activation and normalization
                x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            # If there are no edges or no GCN layers, we can't use GCN.
            # Instead, we'll use a linear layer to process the combined features
            # to ensure the output shape is correct for the final layer.
            x = self.feature_combiner(x)
            x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
            x = torch.clamp(x, min=-10.0, max=10.0)
            x = F.relu(x)
            
        # 4. Final prediction layer
        # Output shape: (num_nodes, 1)
        output = self.linear(x)
        
        # --- Final Output Sanitization ---
        output = torch.nan_to_num(output, nan=0.0, posinf=10.0, neginf=-10.0)
        output = torch.clamp(output, min=-10.0, max=10.0)

        # We squeeze to get a (num_nodes,) shape. The loss function (BCEWithLogitsLoss) will handle the sigmoid.
        return output.squeeze(1) 