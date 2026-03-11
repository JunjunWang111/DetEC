# models/fusion.py
import torch
import torch.nn as nn

class BiDirectionalCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.struct_to_seq = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.seq_to_struct = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, H_struct, H_seq):
        struct2seq, _ = self.struct_to_seq(H_struct, H_seq, H_seq)
        seq2struct, _ = self.seq_to_struct(H_seq, H_struct, H_struct)
        H_global = self.norm(struct2seq + seq2struct)
        return H_global

class QueryGuidedFusion(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, H_global, H_local):
        fused, _ = self.cross_attn(H_global, H_local, H_local)
        fused = self.norm(H_global + fused)
        return fused
