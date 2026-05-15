import os

import torch
import torch.nn as nn

from config import Config
from data.dataset import load_split_dataframe
from models.detec import DetEC
from utils.protein import parse_ec_numbers
from utils.taxonomy import ECTaxonomy


config = Config()
train_df = load_split_dataframe(config, "train")
taxonomy = ECTaxonomy.from_ec_collections(parse_ec_numbers(value) for value in train_df["EC number"].tolist())
model = DetEC(config, taxonomy)

for parameter in model.parameters():
    nn.init.normal_(parameter, mean=0.0, std=0.02)

checkpoint_path = "./checkpoints/best_model.pt"
os.makedirs("./checkpoints", exist_ok=True)
torch.save(
    {
        "model_state": model.state_dict(),
        "taxonomy": taxonomy.to_dict(),
        "config": config.__dict__,
    },
    checkpoint_path,
)

print(f"Initialized checkpoint saved to: {checkpoint_path}")
