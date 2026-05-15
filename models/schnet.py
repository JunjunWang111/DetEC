import torch
import torch.nn as nn


class CFConv(nn.Module):
    def __init__(self, hidden_dim, n_filters):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(16, n_filters),
            nn.SiLU(),
            nn.Linear(n_filters, hidden_dim),
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_rbf):
        row, col = edge_index
        filters = self.filter_net(edge_rbf)
        messages = x[col] * filters
        out = torch.zeros_like(x)
        out.index_add_(0, row, messages)
        return self.out_proj(out)


class SchNetLayer(nn.Module):
    def __init__(self, hidden_dim, n_filters):
        super().__init__()
        self.conv = CFConv(hidden_dim, n_filters)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_rbf):
        updated = x + self.conv(x, edge_index, edge_rbf)
        updated = self.norm(updated)
        return self.norm(updated + self.ffn(updated))


class SchNetEncoder(nn.Module):
    def __init__(self, atom_feat_dim, hidden_dim, n_filters, n_interactions):
        super().__init__()
        self.embed_in = nn.Linear(atom_feat_dim, hidden_dim)
        self.layers = nn.ModuleList([SchNetLayer(hidden_dim, n_filters) for _ in range(n_interactions)])

    def forward(self, x, edge_index, edge_rbf):
        x = self.embed_in(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_rbf)
        return x
