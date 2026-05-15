import argparse
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, Iterable, List, Tuple

from config import Config
from data.dataset import SPLIT_TO_FILE, load_split_dataframe
from utils.protein import (
    UNIPROT_PATTERN,
    active_site_record_to_indices,
    ensure_active_site_record,
    ensure_structure_file,
    sanitize_sequence,
    write_pseudo_pdb,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Bulk prefetch structures and active-site annotations for DetEC.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test_new", "test_price"],
        choices=list(SPLIT_TO_FILE.keys()),
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum unique entries to process after de-duplication.")
    parser.add_argument("--start", type=int, default=0, help="Start offset within the de-duplicated queue.")
    parser.add_argument("--status-file", type=str, default="./data/cache_manifest.tsv")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--skip-active-sites", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fill-missing-pseudo", action="store_true")
    parser.add_argument("--use_p2rank_active_sites", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--java_path", type=str, default=None)
    parser.add_argument("--p2rank_root", type=str, default=None)
    parser.add_argument("--p2rank_probability_threshold", type=float, default=None)
    parser.add_argument("--p2rank_top_pockets", type=int, default=None)
    return parser.parse_args()


def load_existing_status(path: str) -> Dict[str, Dict[str, str]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {row["entry"]: row for row in reader}


def write_status_rows(path: str, rows: Iterable[Dict[str, str]]):
    rows = list(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "entry",
            "splits",
            "status",
            "has_structure",
            "has_active_sites",
            "active_site_source",
            "message",
            "updated_at",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def collect_queue(config: Config, splits: List[str]) -> List[Tuple[str, str, List[str]]]:
    queue: Dict[str, Dict[str, object]] = {}
    for split in splits:
        df = load_split_dataframe(config, split)
        for _, row in df.iterrows():
            entry = str(row["Entry"]).strip()
            sequence = sanitize_sequence(str(row["Sequence"]), config.max_seq_len)
            record = queue.setdefault(entry, {"sequence": sequence, "splits": []})
            if not record["sequence"]:
                record["sequence"] = sequence
            record["splits"].append(split)
    return [(entry, meta["sequence"], sorted(set(meta["splits"]))) for entry, meta in queue.items()]


def summarize(status_rows: Dict[str, Dict[str, str]]):
    total = len(status_rows)
    completed = sum(row["status"] == "done" for row in status_rows.values())
    pseudo = sum(row["status"] == "pseudo" for row in status_rows.values())
    partial = sum(row["status"] == "partial" for row in status_rows.values())
    failed = sum(row["status"] == "failed" for row in status_rows.values())
    print(f"manifest_total={total}")
    print(f"done={completed}")
    print(f"pseudo={pseudo}")
    print(f"partial={partial}")
    print(f"failed={failed}")


def main():
    args = parse_args()
    config = Config()
    config.allow_download = True
    config.apply_overrides(args)

    queue = collect_queue(config, args.splits)
    queue = queue[args.start :]
    if args.limit is not None:
        queue = queue[: args.limit]

    status_rows = load_existing_status(args.status_file)
    if args.summary_only:
        summarize(status_rows)
        return

    print(f"queue_size={len(queue)}")
    print(f"splits={','.join(args.splits)}")
    print(f"status_file={args.status_file}")

    active_site_lock = Lock()

    def process_one(entry: str, sequence: str, splits: List[str]):
        existing = status_rows.get(entry)
        needs_active_backfill = (
            bool(existing)
            and not args.skip_active_sites
            and existing.get("has_structure") == "1"
            and existing.get("has_active_sites") != "1"
            and (UNIPROT_PATTERN.match(entry) or config.use_p2rank_active_sites)
        )
        if existing and existing["status"] == "done" and not needs_active_backfill:
            return entry, {
                "entry": entry,
                "splits": ",".join(splits),
                "status": existing["status"],
                "has_structure": existing["has_structure"],
                "has_active_sites": existing["has_active_sites"],
                "active_site_source": existing.get("active_site_source", ""),
                "message": existing["message"],
                "updated_at": existing["updated_at"],
            }, "skip done"
        if existing and existing["status"] == "failed" and not args.retry_failed:
            return entry, {
                "entry": entry,
                "splits": ",".join(splits),
                "status": existing["status"],
                "has_structure": existing["has_structure"],
                "has_active_sites": existing["has_active_sites"],
                "active_site_source": existing.get("active_site_source", ""),
                "message": existing["message"],
                "updated_at": existing["updated_at"],
            }, "skip failed"

        structure_path = None
        active_sites_ok = False
        active_site_source = ""
        message = ""
        try:
            structure_path = ensure_structure_file(
                entry=entry,
                sequence=sequence,
                structure_dir=os.path.join(config.data_root, config.pdb_dir),
                allow_download=True,
                mapping_cache_path=config.accession_mapping_cache,
                timeout=config.download_timeout,
            )
            has_structure = bool(structure_path and os.path.exists(structure_path))

            if args.skip_active_sites:
                active_sites_ok = False
                active_site_source = ""
            elif UNIPROT_PATTERN.match(entry) or config.use_p2rank_active_sites:
                with active_site_lock:
                    active_site_record = ensure_active_site_record(
                        entry=entry,
                        sequence=sequence,
                        structure_path=structure_path,
                        cache_path=config.active_site_cache,
                        timeout=config.download_timeout,
                        use_p2rank=config.use_p2rank_active_sites,
                        p2rank_root=config.p2rank_root,
                        p2rank_output_dir=config.p2rank_output_dir,
                        java_path=config.java_path,
                        p2rank_profile=config.p2rank_profile,
                        p2rank_threads=config.p2rank_threads,
                        p2rank_probability_threshold=config.p2rank_probability_threshold,
                        p2rank_top_pockets=config.p2rank_top_pockets,
                        p2rank_visualizations=config.p2rank_visualizations,
                    )
                active_sites_ok = len(active_site_record_to_indices(active_site_record)) > 0
                active_site_source = str(active_site_record.get("source", ""))
            else:
                active_sites_ok = False
                active_site_source = ""

            needs_active_sites = not args.skip_active_sites and (
                UNIPROT_PATTERN.match(entry) or config.use_p2rank_active_sites
            )
            if has_structure and (active_sites_ok or not needs_active_sites):
                status = "done"
            elif has_structure:
                status = "partial"
            else:
                if args.fill_missing_pseudo:
                    pseudo_path = write_pseudo_pdb(
                        entry=entry,
                        sequence=sequence,
                        target_path=os.path.join(config.data_root, config.pdb_dir, f"{entry}.pdb"),
                    )
                    has_structure = os.path.exists(pseudo_path)
                    status = "pseudo" if has_structure else "failed"
                    message = "pseudo_structure_generated" if has_structure else "structure_unavailable"
                else:
                    status = "failed"
                    message = "structure_unavailable"
        except Exception as exc:
            has_structure = bool(structure_path and os.path.exists(structure_path))
            status = "failed"
            message = str(exc)[:300]

        row = {
            "entry": entry,
            "splits": ",".join(splits),
            "status": status,
            "has_structure": "1" if has_structure else "0",
            "has_active_sites": "1" if active_sites_ok else "0",
            "active_site_source": active_site_source,
            "message": message,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return entry, row, (
            f"status={status} structure={has_structure} active_sites={active_sites_ok} "
            f"source={active_site_source or '-'} splits={','.join(splits)}"
        )

    updated = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_meta = {
            executor.submit(process_one, entry, sequence, splits): (index, entry)
            for index, (entry, sequence, splits) in enumerate(queue, start=1)
        }
        for future in as_completed(future_to_meta):
            index, entry = future_to_meta[future]
            returned_entry, row, message = future.result()
            status_rows[returned_entry] = row
            updated += 1
            print(f"[{index}/{len(queue)}] {entry} {message}")
            if updated % 10 == 0:
                write_status_rows(args.status_file, status_rows.values())

    write_status_rows(args.status_file, status_rows.values())
    summarize(status_rows)


if __name__ == "__main__":
    main()
