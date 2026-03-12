# models/esm_encoder.py
import torch
import torch.nn as nn

class ESMEncoder(nn.Module):
    def __init__(self, model_name, output_dim):
        super().__init__()
        self.hidden_dim = 1280
        self.proj = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, sequences):

        batch_size = len(sequences)
        max_len = max(len(seq) for seq in sequences)
        device = next(self.parameters()).device
        random_features = torch.randn(batch_size, max_len, self.hidden_dim, device=device)
        proj_repr = self.proj(random_features)
        return proj_repr
