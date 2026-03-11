# models/decoder.py
import torch
import torch.nn as nn

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, q, memory):
        q2 = self.norm1(q + self.self_attn(q, q, q)[0])
        q3 = self.norm2(q2 + self.cross_attn(q2, memory, memory)[0])
        out = self.norm3(q3 + self.ffn(q3))
        return out

class FunctionQueryDecoder(nn.Module):
    def __init__(self, num_queries, d_model, num_layers, num_heads, dropout):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, d_model))
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, memory):
        B = memory.size(0)
        q = self.query_embed.expand(B, -1, -1)
        for layer in self.layers:
            q = layer(q, memory)
        return q
