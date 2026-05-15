import argparse
import json
import os
from collections import Counter
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data.dataset import ProteinDataset, collate_protein_batch
from models.detec import DetEC
from simple_evaluate import load_checkpoint_payload
from train import to_device
from utils.losses import compute_set_prediction_loss
from utils.metrics import compute_multilabel_metrics, decode_prediction_sets
from utils.taxonomy import ECTaxonomy


def parse_args():
    parser = argparse.ArgumentParser(description="Scan objectness thresholds using cached logits from a single model pass.")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
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
    parser.add_argument("--thresholds", type=str, default="0.05:0.95:0.05")
    parser.add_argument("--output_json", type=str, default="./threshold_scan_results.json")
    parser.add_argument("--output_txt", type=str, default="./threshold_scan_results.txt")
    return parser.parse_args()


def parse_thresholds(spec: str) -> List[float]:
    text = (spec or "").strip()
    if not text:
        return [0.35]
    if ":" in text:
        start_text, stop_text, step_text = text.split(":")
        start = float(start_text)
        stop = float(stop_text)
        step = float(step_text)
        thresholds = []
        current = start
        while current <= stop + 1e-8:
            thresholds.append(round(current, 4))
            current += step
        return thresholds
    return [round(float(part.strip()), 4) for part in text.split(",") if part.strip()]


def build_loader(config, taxonomy, split: str, max_samples: int | None):
    dataset = ProteinDataset(config, split=split, taxonomy=taxonomy, max_samples=max_samples)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_protein_batch,
    )
    return dataset, loader


def collect_cached_outputs(model, loader, device, taxonomy, split_name: str):
    cached_outputs: Dict[str, List[torch.Tensor]] = {}
    truths = []
    active_site_sources = Counter()
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Collect {split_name}", leave=False):
            batch = to_device(batch, device)
            outputs = model(batch)
            loss = compute_set_prediction_loss(outputs, batch["known_targets"], taxonomy)
            total_loss += float(loss.item())

            for key, value in outputs.items():
                cached_outputs.setdefault(key, []).append(value.detach().cpu())
            truths.extend(batch["true_ecs"])
            active_site_sources.update(batch["active_site_sources"])

    return {
        "outputs": {key: torch.cat(chunks, dim=0) for key, chunks in cached_outputs.items()},
        "truths": truths,
        "loss": total_loss / max(len(loader), 1),
        "active_site_sources": dict(active_site_sources),
    }


def evaluate_cached_outputs(cached_payload, taxonomy, threshold: float):
    predictions = decode_prediction_sets(cached_payload["outputs"], taxonomy, threshold=threshold)
    metrics = compute_multilabel_metrics(predictions, cached_payload["truths"], label_space=taxonomy.full_labels)
    metrics["loss"] = cached_payload["loss"]
    metrics["num_samples"] = len(cached_payload["truths"])
    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"loss={metrics['loss']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"accuracy={metrics['accuracy']:.4f}"
    )


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
    thresholds = parse_thresholds(args.thresholds)

    model = DetEC(config, taxonomy).to(device)
    model.load_state_dict(checkpoint["model_state"])

    _, val_loader = build_loader(config, taxonomy, "val", config.max_val_samples)
    _, new_loader = build_loader(config, taxonomy, "test_new", config.max_test_samples)
    _, price_loader = build_loader(config, taxonomy, "test_price", config.max_test_samples)

    cached_val = collect_cached_outputs(model, val_loader, device, taxonomy, "val")
    cached_new = collect_cached_outputs(model, new_loader, device, taxonomy, "test_new")
    cached_price = collect_cached_outputs(model, price_loader, device, taxonomy, "test_price")

    rows = []
    for threshold in thresholds:
        val_metrics = evaluate_cached_outputs(cached_val, taxonomy, threshold)
        new_metrics = evaluate_cached_outputs(cached_new, taxonomy, threshold)
        price_metrics = evaluate_cached_outputs(cached_price, taxonomy, threshold)
        row = {
            "threshold": threshold,
            "val": val_metrics,
            "test_new": new_metrics,
            "test_price": price_metrics,
        }
        rows.append(row)
        print(
            f"threshold={threshold:.2f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"new_f1={new_metrics['f1']:.4f} "
            f"price_f1={price_metrics['f1']:.4f}"
        )

    best_row = max(
        rows,
        key=lambda row: (
            row["val"]["f1"],
            row["val"]["precision"],
            row["val"]["recall"],
            -abs(row["threshold"] - config.objectness_threshold),
        ),
    )

    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "device": str(device),
        "thresholds": rows,
        "best_threshold": best_row["threshold"],
        "best_val": best_row["val"],
        "best_test_new": best_row["test_new"],
        "best_test_price": best_row["test_price"],
        "active_site_sources": {
            "val": cached_val["active_site_sources"],
            "test_new": cached_new["active_site_sources"],
            "test_price": cached_price["active_site_sources"],
        },
    }

    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    lines = [
        "==================================================",
        "Threshold Scan Report",
        "==================================================",
        f"checkpoint: {os.path.abspath(args.checkpoint)}",
        f"device: {device}",
        f"best_threshold: {best_row['threshold']:.2f}",
        f"best_val: {format_metrics(best_row['val'])}",
        f"best_new: {format_metrics(best_row['test_new'])}",
        f"best_price: {format_metrics(best_row['test_price'])}",
        f"active_site_sources_val: {json.dumps(cached_val['active_site_sources'], ensure_ascii=False)}",
        f"active_site_sources_test_new: {json.dumps(cached_new['active_site_sources'], ensure_ascii=False)}",
        f"active_site_sources_test_price: {json.dumps(cached_price['active_site_sources'], ensure_ascii=False)}",
        "",
        "Per-threshold summary:",
    ]
    for row in rows:
        lines.append(
            f"threshold={row['threshold']:.2f} "
            f"val_f1={row['val']['f1']:.4f} "
            f"new_f1={row['test_new']['f1']:.4f} "
            f"price_f1={row['test_price']['f1']:.4f}"
        )

    with open(args.output_txt, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
