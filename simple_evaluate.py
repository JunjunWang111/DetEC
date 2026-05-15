import argparse
import os

import torch
from torch.utils.data import DataLoader

from config import Config
from data.dataset import ProteinDataset, collate_protein_batch
from models.detec import DetEC
from train import evaluate, to_device
from utils.taxonomy import ECTaxonomy


def load_checkpoint_payload(path: str):
    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        try:
            with open(path, "rb") as handle:
                header = handle.read(512)
            if b"model_type:" in header or b"evaluation_results:" in header or b"\xe5\x8d\xa0\xe4\xbd\x8d" in header:
                raise RuntimeError(
                    "The checkpoint file is a placeholder text artifact rather than a real PyTorch state dict. "
                    "The official release checkpoint.zip in this repository is not a usable pretrained model."
                ) from exc
        except OSError:
            pass
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained DetEC checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--objectness_threshold", type=float, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--use_pretrained_esm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--freeze_pretrained_esm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--esm_model_name", type=str, default=None)
    parser.add_argument("--use_p2rank_active_sites", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--java_path", type=str, default=None)
    parser.add_argument("--p2rank_root", type=str, default=None)
    parser.add_argument("--p2rank_probability_threshold", type=float, default=None)
    parser.add_argument("--p2rank_top_pockets", type=int, default=None)
    parser.add_argument("--allow_download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = load_checkpoint_payload(args.checkpoint)

    config = Config()
    for key, value in checkpoint.get("config", {}).items():
        if hasattr(config, key):
            setattr(config, key, value)

    config.apply_overrides(args)
    if args.allow_download:
        config.allow_download = True

    taxonomy = ECTaxonomy.from_dict(checkpoint["taxonomy"])
    device = torch.device(config.device)

    model = DetEC(config, taxonomy).to(device)
    model.load_state_dict(checkpoint["model_state"])

    test_new_dataset = ProteinDataset(config, split="test_new", taxonomy=taxonomy, max_samples=config.max_test_samples)
    test_price_dataset = ProteinDataset(
        config, split="test_price", taxonomy=taxonomy, max_samples=config.max_test_samples
    )

    test_new_loader = DataLoader(
        test_new_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_protein_batch,
    )
    test_price_loader = DataLoader(
        test_price_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_protein_batch,
    )

    new_metrics = evaluate(model, test_new_loader, device, taxonomy, threshold=config.objectness_threshold)
    price_metrics = evaluate(model, test_price_loader, device, taxonomy, threshold=config.objectness_threshold)

    print("New-392:", new_metrics)
    print("Price-149:", price_metrics)


if __name__ == "__main__":
    main()
