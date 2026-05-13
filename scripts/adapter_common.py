import torch.nn as nn

class TargetLayerAwareKVAdapter(nn.Module):
    def __init__(self, num_target_layers=32, input_dim=64, output_dim=128, hidden_dim=256):
        super().__init__()
        self.num_target_layers = num_target_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.key_mappers = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))
            for _ in range(num_target_layers)
        ])
        self.value_mappers = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))
            for _ in range(num_target_layers)
        ])

    def forward_target_layer(self, target_layer_idx, k, v):
        return self.key_mappers[target_layer_idx](k), self.value_mappers[target_layer_idx](v)
