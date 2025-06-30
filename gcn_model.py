import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SpatioTemporalGCN(nn.Module):
    def __init__(self, static_feature_dim, dynamic_feature_dim, hidden_dim, lstm_layers, gcn_layers):
        super(SpatioTemporalGCN, self).__init__()

        # Temporal feature extractor (LSTM)
        self.lstm = nn.LSTM(
            input_size=dynamic_feature_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True  # Important: input shape is (batch, seq, feature)
        )

        # This layer will be used to project the combined features to the hidden dimension
        # when GCN layers are skipped (e.g., for a batch with no edges).
        self.feature_combiner = nn.Linear(hidden_dim + static_feature_dim, hidden_dim)

        # GCN layers for spatial feature extraction
        self.gcn_layers = nn.ModuleList()
        if gcn_layers > 0:
            # First GCN layer
            self.gcn_layers.append(GCNConv(hidden_dim + static_feature_dim, hidden_dim))
            
            # Subsequent GCN layers
            for _ in range(gcn_layers - 1):
                self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        # Output layer
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        # In PyG, the 'data' object might not have a batch dimension if you're processing a single graph
        # The LSTM expects a batch dimension, so we unsqueeze.
        x_static, x_dynamic, edge_index, edge_weight = data.x_static, data.x_dynamic, data.edge_index, data.edge_weight
        
        # 1. Process dynamic features with LSTM
        # Input to LSTM: (num_nodes, seq_len, dynamic_features)
        # We get hidden states for all time steps, we only need the last one.
        _, (h_n, _) = self.lstm(x_dynamic.contiguous())
        
        # h_n shape is (num_layers, num_nodes, hidden_dim). We take the last layer's output.
        dynamic_features_encoded = h_n[-1]

        # 2. Combine static and encoded dynamic features
        # Shape: (num_nodes, hidden_dim + static_feature_dim)
        combined_features = torch.cat([x_static, dynamic_features_encoded], dim=1)

        # 3. Process combined features with GCN
        # Note: GCNConv expects edge_weight as an argument in the forward pass
        
        # --- Safeguard for empty edge_index ---
        # If a batch has no edges, GCN layers will fail. Skip them.
        if edge_index.numel() > 0 and len(self.gcn_layers) > 0:
            x = combined_features
            for i, conv in enumerate(self.gcn_layers):
                x = conv(x, edge_index, edge_weight=edge_weight)
                x = F.relu(x)
                if i < len(self.gcn_layers) - 1: # No dropout on the last GCN layer's output
                    x = F.dropout(x, p=0.5, training=self.training)
        else:
            # If there are no edges or no GCN layers, we can't use GCN.
            # Instead, we'll use a linear layer to process the combined features
            # to ensure the output shape is correct for the final layer.
            x = self.feature_combiner(combined_features)
            x = F.relu(x)
            
        # 4. Final prediction layer
        # Output shape: (num_nodes, 1)
        output = self.linear(x)

        # We squeeze to get a (num_nodes,) shape. The loss function (BCEWithLogitsLoss) will handle the sigmoid.
        return output.squeeze(1) 