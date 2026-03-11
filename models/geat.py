# models/geat.py
import torch
import torch.nn as nn

class GeometricAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)

        self.geo_mlp = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, num_heads)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_dist):
        N = x.size(0)

        q = self.q_proj(x).view(N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(N, self.num_heads, self.head_dim)

        rbf = self.rbf(edge_dist)
        geo_bias = self.geo_mlp(rbf)

        out = self.propagate(q, k, v, edge_index, geo_bias, N)
        out = out.view(N, self.out_dim)
        out = self.out_proj(out)
        return out

    def propagate(self, q, k, v, edge_index, geo_bias, N):
        row, col = edge_index
        q_i = q[row]
        k_j = k[col]
        v_j = v[col]

        attn = (q_i * k_j).sum(dim=-1) / (self.head_dim ** 0.5)
        attn = attn + geo_bias

        attn = self.softmax(attn, row, N)
        attn = self.dropout(attn)

        out = torch.zeros(N, self.num_heads, self.head_dim, device=q.device)
        out.index_add_(0, row, v_j * attn.unsqueeze(-1))
        return out

    def softmax(self, attn, edge_index_i, N):
        max_attn = torch.zeros(N, self.num_heads, device=attn.device)
        for i in range(N):
            mask = edge_index_i == i
            if mask.any():
                max_attn[i] = attn[mask].max(dim=0)[0]
        max_attn = max_attn[edge_index_i]
        exp_attn = torch.exp(attn - max_attn)
        sum_attn = torch.zeros(N, self.num_heads, device=attn.device)
        for i in range(N):
            mask = edge_index_i == i
            if mask.any():
                sum_attn[i] = exp_attn[mask].sum(dim=0)
        sum_attn = sum_attn[edge_index_i]
        return exp_attn / (sum_attn + 1e-12)

    def rbf(self, dist):
        centers = torch.linspace(0, 6, 16, device=dist.device)
        gamma = 1.0
        return torch.exp(-gamma * (dist.unsqueeze(-1) - centers)**2)

class GEAT(nn.Module):
    def __init__(self, in_dim, d_model, num_heads, scales, scale_weights, dropout):
        super().__init__()
        self.scales = scales
        self.scale_weights = scale_weights
        self.layers = nn.ModuleList()
        for _ in scales:
            self.layers.append(GeometricAttentionLayer(in_dim, d_model, num_heads, dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, edge_indices, edge_dists):
        out = 0
        for i, layer in enumerate(self.layers):
            out_i = layer(x, edge_indices[i], edge_dists[i])
            out = out + self.scale_weights[i] * out_i
        out = self.norm(out)
        return out
