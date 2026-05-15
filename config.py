import os
import torch


class Config:
    def __init__(self):
        # ---------- Paths ----------
        self.data_root = "./data"
        self.pdb_dir = "structures"
        self.active_site_cache = os.path.join(self.data_root, "active_sites.json")
        self.accession_mapping_cache = os.path.join(self.data_root, "accession_mapping.json")
        self.p2rank_root = os.path.join(self.data_root, "tools", "p2rank")
        self.p2rank_output_dir = os.path.join(self.data_root, "p2rank_outputs")

        # ---------- Model ----------
        self.d_model = 256
        self.num_heads = 8
        self.dropout = 0.1

        # Sequence encoder
        self.max_seq_len = 1024
        self.seq_encoder_layers = 2
        self.seq_encoder_heads = 8
        self.seq_encoder_ffn = 512
        self.use_pretrained_esm = False
        self.freeze_pretrained_esm = True
        self.esm_model_name = "esm2_t33_650M_UR50D"
        self.esm_output_dim = 1280
        self.esm_cache_dir = os.path.join(self.data_root, "esm_cache")

        # GEAT
        self.geat_scales = [6.0, 10.0, 15.0]
        self.geat_weights = [0.5, 0.3, 0.2]
        self.geat_knn = 20
        self.geat_sigma = 2.0

        # Local chemistry encoder
        self.local_atom_radius = 6.0
        self.max_local_atoms = 512
        self.atom_feature_dim = 10
        self.schnet_cutoff = 6.0
        self.schnet_n_filters = 64
        self.schnet_n_interactions = 4
        self.schnet_rbf_centers = 16

        # Fusion and decoder
        self.fusion_dropout = 0.1
        self.num_queries = 10
        self.decoder_layers = 4
        self.decoder_heads = 8

        # ---------- Training ----------
        self.seed = 42
        self.batch_size = 8
        self.lr = 1e-4
        self.weight_decay = 1e-5
        self.epochs = 100
        self.patience = 10
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = 0
        self.save_dir = "./checkpoints"
        self.objectness_threshold = 0.35
        self.log_interval = 20

        # Optional dataset limits for quick smoke runs.
        self.max_train_samples = None
        self.max_val_samples = None
        self.max_test_samples = None

        # ---------- Auxiliary data ----------
        self.min_plddt = 70.0
        self.allow_download = False
        self.download_timeout = 45
        self.use_p2rank_active_sites = False
        self.java_path = None
        self.p2rank_profile = "alphafold"
        self.p2rank_threads = 1
        self.p2rank_probability_threshold = 0.25
        self.p2rank_top_pockets = 1
        self.p2rank_visualizations = False

    def apply_overrides(self, args):
        for key, value in vars(args).items():
            if value is None or not hasattr(self, key):
                continue
            setattr(self, key, value)
