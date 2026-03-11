# models/detec.py
import torch
import torch.nn as nn
from .geat import GEAT
from .esm_encoder import ESMEncoder
from .schnet import SchNetEncoder
from .fusion import BiDirectionalCrossAttention, QueryGuidedFusion
from .decoder import FunctionQueryDecoder
from .heads import HierarchicalHead
from utils.geometry import compute_local_density, compute_curvature, compute_local_frames

class DetEC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 序列编码
        self.esm_encoder = ESMEncoder(config.esm_model_name, config.d_model)

        # 全局结构编码 (GEAT)
        self.geat = GEAT(
            in_dim=config.d_model,
            d_model=config.d_model,
            num_heads=config.num_heads,
            scales=config.geat_scales,
            scale_weights=config.geat_weights,
            dropout=config.dropout
        )
        self.struct_embed = nn.Linear(3 + 1 + 1 + 9, config.d_model)

        # 局部化学编码 (SchNet)
        self.schnet = SchNetEncoder(
            atom_feat_dim=4,
            hidden_dim=config.d_model,
            n_filters=config.schnet_n_filters,
            n_interactions=config.schnet_n_interactions
        )

        # 融合
        self.bi_attn = BiDirectionalCrossAttention(config.d_model, config.num_heads, config.fusion_dropout)
        self.query_fusion = QueryGuidedFusion(config.d_model, config.num_heads, config.fusion_dropout)

        # 解码器
        self.decoder = FunctionQueryDecoder(
            num_queries=config.num_queries,
            d_model=config.d_model,
            num_layers=config.decoder_layers,
            num_heads=config.decoder_heads,
            dropout=config.dropout
        )

        # 层次预测头
        self.hier_head = HierarchicalHead(config.d_model, config.ec_levels)

    def forward(self, batch):
        # 1. 序列编码
        seq_feat = self.esm_encoder([batch['seq']])  # (1, L, d)

        # 2. 结构编码（简化处理）
        coords_np = batch['coords']
        coords = torch.tensor(coords_np, device=next(self.parameters()).device, dtype=torch.float32)
        
        # 生成占位符几何特征
        seq_length = len(batch['seq'])
        rho = [0.0 for _ in range(seq_length)]
        kappa = [0.0 for _ in range(seq_length)]
        frames = [[0.0 for _ in range(9)] for _ in range(seq_length)]
        
        geo_input = torch.cat([
            coords,
            torch.tensor(rho, device=coords.device, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(kappa, device=coords.device, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(frames, device=coords.device, dtype=torch.float32)
        ], dim=-1)
        h0 = self.struct_embed(geo_input)
        
        # 生成与scales长度匹配的边索引和距离列表
        edge_indices = []
        edge_dists = []
        for scale in self.config.geat_scales:
            # 为每个尺度生成边索引和距离
            if seq_length > 1:
                # 生成相邻残基的边
                edges = []
                dists = []
                for i in range(seq_length - 1):
                    edges.append([i, i+1])
                    edges.append([i+1, i])
                    dists.append(scale)
                    dists.append(scale)
                edge_index = torch.tensor(edges, device=coords.device).t()
                edge_dist = torch.tensor(dists, device=coords.device)
            else:
                # 单个残基的情况
                edge_index = torch.empty((2, 0), device=coords.device, dtype=torch.long)
                edge_dist = torch.empty(0, device=coords.device)
            edge_indices.append(edge_index)
            edge_dists.append(edge_dist)
        
        struct_feat = self.geat(h0, edge_indices, edge_dists)

        # 3. 局部化学编码（简化处理）
        local_feat = torch.zeros_like(struct_feat)

        # 4. 融合
        # 处理长度不匹配的情况
        struct_feat = struct_feat.unsqueeze(0)
        seq_feat = seq_feat
        if struct_feat.size(1) != seq_feat.size(1):
            # 使用较短的长度
            min_len = min(struct_feat.size(1), seq_feat.size(1))
            struct_feat = struct_feat[:, :min_len, :]
            seq_feat = seq_feat[:, :min_len, :]
        H_global = self.bi_attn(struct_feat, seq_feat)
        H_global = H_global.squeeze(0)
        local_feat = local_feat[:H_global.size(0), :]
        H_fused = self.query_fusion(H_global.unsqueeze(0), local_feat.unsqueeze(0)).squeeze(0)

        # 5. 解码
        memory = H_fused.unsqueeze(0)
        queries = self.decoder(memory)

        # 6. 层次预测
        probs = self.hier_head(queries)

        return probs
