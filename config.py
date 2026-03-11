# config.py
import torch

class Config:
    # ---------- 数据路径 ----------
    data_root = "./data"
    protein_list_file = "protein_list.csv"
    pdb_dir = "pdbs"
    active_site_file = "active_sites.csv"

    # ---------- 模型超参数 ----------
    d_model = 256
    num_heads = 8
    dropout = 0.1

    # GEAT
    geat_scales = [6.0, 10.0, 15.0]
    geat_weights = [0.5, 0.3, 0.2]
    geat_knn = 20
    geat_sigma = 2.0

    # SchNet
    schnet_cutoff = 6.0
    schnet_n_filters = 64
    schnet_n_interactions = 4
    schnet_rbf_centers = 16

    # ESM-2
    esm_model_name = "esm2_t33_650M_UR50D"
    esm_output_dim = 1280

    # Fusion
    fusion_dropout = 0.1

    # Decoder
    num_queries = 10
    decoder_layers = 6
    decoder_heads = 8

    # Hierarchy
    ec_levels = [7, 7, 7, 7]

    # ---------- 训练参数 ----------
    batch_size = 8
    lr = 1e-4
    weight_decay = 1e-5
    epochs = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4
    save_dir = "./checkpoints"
    log_interval = 10
