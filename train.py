import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data.dataset import ProteinDataset, collate_protein_batch, load_split_dataframe
from models.detec import DetEC
from utils.losses import compute_set_prediction_loss
from utils.metrics import compute_multilabel_metrics, decode_prediction_sets
from utils.protein import parse_ec_numbers
from utils.taxonomy import ECTaxonomy


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_taxonomy(config):
    train_df = load_split_dataframe(config, "train")
    ec_collections = [parse_ec_numbers(value) for value in train_df["EC number"].tolist()]
    return ECTaxonomy.from_ec_collections(ec_collections)


def to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def build_scheduler(optimizer, steps_per_epoch, config):
    total_steps = max(steps_per_epoch * config.epochs, 1)
    warmup_steps = min(5000, max(1, total_steps // 10))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, loader, device, taxonomy, threshold):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_truths = []

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            outputs = model(batch)
            loss = compute_set_prediction_loss(outputs, batch["known_targets"], taxonomy)
            total_loss += float(loss.item())

            prediction_sets = decode_prediction_sets(outputs, taxonomy, threshold=threshold)
            all_predictions.extend(prediction_sets)
            all_truths.extend(batch["true_ecs"])

    metrics = compute_multilabel_metrics(all_predictions, all_truths, label_space=taxonomy.full_labels)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def build_dataloaders(config, taxonomy):
    train_dataset = ProteinDataset(config, split="train", taxonomy=taxonomy, max_samples=config.max_train_samples)
    val_dataset = ProteinDataset(config, split="val", taxonomy=taxonomy, max_samples=config.max_val_samples)
    test_new_dataset = ProteinDataset(config, split="test_new", taxonomy=taxonomy, max_samples=config.max_test_samples)
    test_price_dataset = ProteinDataset(
        config, split="test_price", taxonomy=taxonomy, max_samples=config.max_test_samples
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_protein_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_protein_batch,
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
    return train_loader, val_loader, test_new_loader, test_price_loader


def save_checkpoint(model, taxonomy, config, path):
    payload = {
        "model_state": model.state_dict(),
        "taxonomy": taxonomy.to_dict(),
        "config": config.__dict__,
    }
    torch.save(payload, path)


def print_metrics(prefix, metrics):
    print(f"{prefix} loss: {metrics['loss']:.4f}")
    print(f"{prefix} precision: {metrics['precision']:.4f}")
    print(f"{prefix} recall: {metrics['recall']:.4f}")
    print(f"{prefix} f1: {metrics['f1']:.4f}")
    print(f"{prefix} accuracy: {metrics['accuracy']:.4f}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Train DetEC with a runnable approximation of the paper pipeline.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--objectness_threshold", type=float, default=None)
    parser.add_argument("--use_pretrained_esm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--freeze_pretrained_esm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--esm_model_name", type=str, default=None)
    parser.add_argument("--local_atom_radius", type=float, default=None)
    parser.add_argument("--use_p2rank_active_sites", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--java_path", type=str, default=None)
    parser.add_argument("--p2rank_root", type=str, default=None)
    parser.add_argument("--p2rank_probability_threshold", type=float, default=None)
    parser.add_argument("--p2rank_top_pockets", type=int, default=None)
    parser.add_argument("--allow_download", action="store_true")
    return parser.parse_args()


def train():
    args = parse_args()
    config = Config()
    config.apply_overrides(args)
    if args.allow_download:
        config.allow_download = True

    set_seed(config.seed)
    device = torch.device(config.device)
    os.makedirs(config.save_dir, exist_ok=True)

    taxonomy = build_taxonomy(config)
    train_loader, val_loader, test_new_loader, test_price_loader = build_dataloaders(config, taxonomy)

    model = DetEC(config, taxonomy).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = build_scheduler(optimizer, steps_per_epoch=max(len(train_loader), 1), config=config)

    best_f1 = -1.0
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")
        for batch in progress_bar:
            batch = to_device(batch, device)
            outputs = model(batch)
            loss = compute_set_prediction_loss(outputs, batch["known_targets"], taxonomy)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch} train loss: {avg_train_loss:.4f}")

        val_metrics = evaluate(model, val_loader, device, taxonomy, threshold=config.objectness_threshold)
        print_metrics("Val", val_metrics)

        latest_path = os.path.join(config.save_dir, "latest_model.pt")
        save_checkpoint(model, taxonomy, config, latest_path)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            save_checkpoint(model, taxonomy, config, os.path.join(config.save_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    checkpoint = torch.load(os.path.join(config.save_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    print("=" * 60)
    print("Testing best model")
    print("=" * 60)
    new_metrics = evaluate(model, test_new_loader, device, taxonomy, threshold=config.objectness_threshold)
    price_metrics = evaluate(model, test_price_loader, device, taxonomy, threshold=config.objectness_threshold)

    print_metrics("New-392", new_metrics)
    print_metrics("Price-149", price_metrics)

    report = (
        "==================================================\n"
        "Final Evaluation Report\n"
        "==================================================\n"
        "Dataset    Precision  Recall     F1 Score   Accuracy\n"
        f"New-392    {new_metrics['precision']:.4f}      {new_metrics['recall']:.4f}      {new_metrics['f1']:.4f}      {new_metrics['accuracy']:.4f}\n"
        f"Price-149  {price_metrics['precision']:.4f}      {price_metrics['recall']:.4f}      {price_metrics['f1']:.4f}      {price_metrics['accuracy']:.4f}\n"
        "==================================================\n"
    )

    report_path = os.path.join(os.getcwd(), "evaluation_results.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)


if __name__ == "__main__":
    train()
