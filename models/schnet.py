# models/schnet.py
import torch
import torch.nn as nn

class CFConv(nn.Module):
    def __init__(self, in_dim, out_dim, n_filters):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, n_filters)
        self.lin2 = nn.Linear(n_filters, out_dim)
        self.filter_net = nn.Sequential(
            nn.Linear(16, n_filters),
            nn.SiLU(),
            nn.Linear(n_filters, n_filters)
        )

    def forward(self, x, edge_index, edge_rbf):
        return self.propagate(x, edge_index, edge_rbf)

    def propagate(self, x, edge_index, edge_rbf):
        row, col = edge_index
        x_j = x[col]
        filter_weight = self.filter_net(edge_rbf)
        x_j = self.lin1(x_j)
        x_j = x_j * filter_weight
        out = torch.zeros_like(x)
        out.index_add_(0, row, x_j)
        return self.lin2(out)

class SchNetLayer(nn.Module):
    def __init__(self, hidden_dim, n_filters):
        super().__init__()
        self.conv = CFConv(hidden_dim, hidden_dim, n_filters)
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x, edge_index, edge_rbf):
        x = x + self.conv(x, edge_index, edge_rbf)
        x = self.act(self.lin(x))
        return x

class SchNetEncoder(nn.Module):
    def __init__(self, atom_feat_dim, hidden_dim, n_filters, n_interactions):
        super().__init__()
        self.embed_in = nn.Linear(atom_feat_dim, hidden_dim)
        self.layers = nn.ModuleList([
            SchNetLayer(hidden_dim, n_filters) for _ in range(n_interactions)
        ])

    def forward(self, x, edge_index, edge_rbf):
        x = self.embed_in(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_rbf)
        return x
