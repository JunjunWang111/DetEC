import torch.nn as nn

from models.decoder import FunctionQueryDecoder
from models.esm_encoder import ESMEncoder
from models.fusion import BiDirectionalCrossAttention, QueryGuidedFusion
from models.geat import GEAT
from models.heads import HierarchicalHead
from models.schnet import SchNetEncoder
from utils.protein import build_local_atom_edges, build_scale_edges


class DetEC(nn.Module):
    def __init__(self, config, taxonomy):
        super().__init__()
        self.config = config
        self.taxonomy = taxonomy

        self.esm_encoder = ESMEncoder(
            config.esm_model_name,
            config.d_model,
            max_length=config.max_seq_len,
            num_layers=config.seq_encoder_layers,
            num_heads=config.seq_encoder_heads,
            ffn_dim=config.seq_encoder_ffn,
            dropout=config.dropout,
            use_pretrained=config.use_pretrained_esm,
            freeze_pretrained=config.freeze_pretrained_esm,
            cache_dir=config.esm_cache_dir,
        )

        self.struct_embed = nn.Linear(14, config.d_model)
        self.geat = GEAT(
            in_dim=config.d_model,
            d_model=config.d_model,
            num_heads=config.num_heads,
            scales=config.geat_scales,
            scale_weights=config.geat_weights,
            dropout=config.dropout,
        )

        local_input_dim = config.atom_feature_dim
        self.schnet = SchNetEncoder(
            atom_feat_dim=local_input_dim,
            hidden_dim=config.d_model,
            n_filters=config.schnet_n_filters,
            n_interactions=config.schnet_n_interactions,
        )

        self.bi_attn = BiDirectionalCrossAttention(config.d_model, config.num_heads, config.fusion_dropout)
        self.query_fusion = QueryGuidedFusion(config.d_model, config.num_heads, config.fusion_dropout)
        self.decoder = FunctionQueryDecoder(
            num_queries=config.num_queries,
            d_model=config.d_model,
            num_layers=config.decoder_layers,
            num_heads=config.decoder_heads,
            dropout=config.dropout,
        )
        self.hier_head = HierarchicalHead(config.d_model, taxonomy.level_sizes, taxonomy.num_full_ecs)

    def _encode_structure(self, coords, geometry):
        h0 = self.struct_embed(geometry)
        edge_indices, edge_dists = build_scale_edges(
            coords,
            scales=self.config.geat_scales,
            knn=self.config.geat_knn,
            geometry=geometry,
        )
        return self.geat(h0, edge_indices, edge_dists)

    def _encode_local_environment(self, atom_features, atom_coords, atom_residue_ids, sequence_length):
        edge_index, edge_rbf = build_local_atom_edges(
            coords=atom_coords,
            cutoff=self.config.schnet_cutoff,
            n_rbf=self.config.schnet_rbf_centers,
        )
        atom_embeddings = self.schnet(atom_features, edge_index, edge_rbf)
        residue_embeddings = atom_embeddings.new_zeros(sequence_length, atom_embeddings.size(-1))
        counts = atom_embeddings.new_zeros(sequence_length, 1)
        residue_embeddings.index_add_(0, atom_residue_ids, atom_embeddings)
        counts.index_add_(0, atom_residue_ids, torch.ones_like(atom_embeddings[:, :1]))
        return residue_embeddings / counts.clamp_min(1.0)

    def forward(self, batch):
        seq_feat = self.esm_encoder(batch["token_ids"], batch["padding_mask"], sequences=batch["sequences"])
        outputs = []

        for batch_index, length in enumerate(batch["lengths"]):
            coords = batch["coords"][batch_index, :length]
            geometry = batch["geometry"][batch_index, :length]
            atom_features = batch["local_atom_features"][batch_index].to(coords.device)
            atom_coords = batch["local_atom_coords"][batch_index].to(coords.device)
            atom_residue_ids = batch["local_atom_residue_ids"][batch_index].to(coords.device)

            struct_feat = self._encode_structure(coords, geometry)
            local_feat = self._encode_local_environment(atom_features, atom_coords, atom_residue_ids, length)
            seq_sample = seq_feat[batch_index, :length]

            global_feat = self.bi_attn(struct_feat.unsqueeze(0), seq_sample.unsqueeze(0)).squeeze(0)
            fused_feat = self.query_fusion(global_feat.unsqueeze(0), local_feat.unsqueeze(0))
            queries = self.decoder(fused_feat)
            outputs.append(self.hier_head(queries))

        return {key: torch.cat([item[key] for item in outputs], dim=0) for key in outputs[0]}

