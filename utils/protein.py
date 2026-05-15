import csv
import json
import math
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_INDEX = {aa: idx + 1 for idx, aa in enumerate(AA_ALPHABET)}

AA_PROPERTIES = {
    "A": [1.8, 0, 0, 0, 0, 0, 0, 0],
    "C": [2.5, 0, 0, 0, 0, 1, 1, 0],
    "D": [-3.5, 0, 1, 1, 0, 0, 1, 0],
    "E": [-3.5, 0, 1, 1, 0, 0, 1, 0],
    "F": [2.8, 0, 0, 0, 1, 0, 0, 0],
    "G": [-0.4, 0, 0, 0, 0, 0, 0, 1],
    "H": [-3.2, 1, 0, 1, 1, 1, 1, 0],
    "I": [4.5, 0, 0, 0, 0, 0, 0, 0],
    "K": [-3.9, 1, 0, 1, 0, 1, 0, 0],
    "L": [3.8, 0, 0, 0, 0, 0, 0, 0],
    "M": [1.9, 0, 0, 0, 0, 0, 1, 0],
    "N": [-3.5, 0, 0, 1, 0, 1, 1, 0],
    "P": [-1.6, 0, 0, 0, 0, 0, 0, 1],
    "Q": [-3.5, 0, 0, 1, 0, 1, 1, 0],
    "R": [-4.5, 1, 0, 1, 0, 1, 0, 0],
    "S": [-0.8, 0, 0, 1, 0, 1, 1, 0],
    "T": [-0.7, 0, 0, 1, 0, 1, 1, 0],
    "V": [4.2, 0, 0, 0, 0, 0, 0, 0],
    "W": [-0.9, 0, 0, 1, 1, 1, 0, 0],
    "Y": [-1.3, 0, 0, 1, 1, 1, 1, 0],
    "X": [0.0, 0, 0, 0, 0, 0, 0, 0],
}

ATOM_TYPES = ("C", "N", "O", "S")
ATOM_ELECTRONEGATIVITY = {"C": 2.55, "N": 3.04, "O": 3.44, "S": 2.58}
ATOM_COVALENT_RADIUS = {"C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05}
ATOM_PARTIAL_CHARGE = {"C": 0.10, "N": -0.30, "O": -0.50, "S": -0.10}

UNIPROT_PATTERN = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|"
    r"[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)
REFSEQ_PATTERN = re.compile(r"^(?:WP|NP|XP|YP|AP|ZP)_\d+(?:\.\d+)?$")


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _parse_float_like(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        return float(match.group(0))


def _parse_int_like(value) -> Optional[int]:
    parsed = _parse_float_like(value)
    if parsed is None:
        return None
    return int(parsed)


def _normalize_active_site_record(value: Any) -> Dict[str, Any]:
    if isinstance(value, list):
        residues = sorted(
            {
                parsed_index
                for index in value
                if (parsed_index := _parse_int_like(index)) is not None
            }
        )
        return {
            "residues": residues,
            "source": "legacy",
            "scores": {str(index): 1.0 for index in residues},
            "metadata": {},
        }

    if isinstance(value, dict):
        residues_value = None
        for key in ("residues", "active_sites", "positions", "indices"):
            if key in value:
                residues_value = value.get(key)
                break
        if residues_value is None and "scores" in value and isinstance(value["scores"], dict):
            residues_value = list(value["scores"].keys())

        residues = []
        if isinstance(residues_value, (list, tuple, set)):
            residues = sorted(
                {
                    parsed_index
                    for index in residues_value
                    if (parsed_index := _parse_int_like(index)) is not None
                }
            )

        scores_raw = value.get("scores", {})
        scores = {}
        if isinstance(scores_raw, dict):
            for key, score in scores_raw.items():
                residue_index = _parse_int_like(key)
                residue_score = _parse_float_like(score)
                if residue_index is None or residue_score is None:
                    continue
                scores[str(residue_index)] = residue_score

        if not scores and residues:
            scores = {str(index): 1.0 for index in residues}

        return {
            "residues": residues,
            "source": str(value.get("source", "unknown")),
            "scores": scores,
            "metadata": dict(value.get("metadata", {})) if isinstance(value.get("metadata"), dict) else {},
            "updated_at": str(value.get("updated_at", "")) if value.get("updated_at") else "",
        }

    return {
        "residues": [],
        "source": "unknown",
        "scores": {},
        "metadata": {},
        "updated_at": "",
    }


def load_active_site_cache(cache_path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return {}

    if not isinstance(payload, dict):
        return {}
    return {str(entry): _normalize_active_site_record(record) for entry, record in payload.items()}


def save_active_site_cache(cache_path: str, cache: Dict[str, Dict[str, Any]]):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    normalized = {str(entry): _normalize_active_site_record(record) for entry, record in cache.items()}
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(normalized, handle, ensure_ascii=False, indent=2)


def active_site_record_to_indices(record: Optional[Union[Dict[str, Any], Sequence[int]]]) -> List[int]:
    if record is None:
        return []
    if isinstance(record, dict):
        return list(_normalize_active_site_record(record)["residues"])
    return sorted(
        {
            parsed_index
            for index in record
            if (parsed_index := _parse_int_like(index)) is not None
        }
    )


def sanitize_sequence(sequence: str, max_length: Optional[int] = None) -> str:
    cleaned = re.sub(r"[^A-Za-z]", "", sequence or "").upper()
    cleaned = "".join(aa if aa in AA_TO_INDEX else "X" for aa in cleaned)
    if max_length:
        cleaned = cleaned[:max_length]
    return cleaned


def tokenize_sequence(sequence: str) -> List[int]:
    return [AA_TO_INDEX.get(aa, AA_TO_INDEX["X"]) for aa in sequence]


def parse_ec_numbers(value) -> List[str]:
    if value is None:
        return []
    ecs = []
    for raw in str(value).split(";"):
        ec = raw.strip()
        if ec:
            ecs.append(ec)
    return ecs


def build_residue_features(sequence: str) -> np.ndarray:
    features = [AA_PROPERTIES.get(aa, AA_PROPERTIES["X"]) for aa in sequence]
    features = np.asarray(features, dtype=np.float32)
    features[:, 0] = features[:, 0] / 4.5
    return features


def build_atom_feature(element: str, residue_aa: str, atom_name: str) -> np.ndarray:
    element = (element or "C").upper()
    residue_aa = (residue_aa or "X").upper()
    one_hot = [1.0 if element == atom_type else 0.0 for atom_type in ATOM_TYPES]
    electronegativity = ATOM_ELECTRONEGATIVITY.get(element, 2.50) / 4.0
    covalent_radius = ATOM_COVALENT_RADIUS.get(element, 0.85) / 1.5
    donor = 1.0 if element in {"N", "O", "S"} and not atom_name.startswith("C") else 0.0
    acceptor = 1.0 if element in {"N", "O", "S"} and atom_name not in {"NZ", "NH1", "NH2"} else 0.0
    partial_charge = (ATOM_PARTIAL_CHARGE.get(element, 0.0) + 1.0) / 2.0
    residue_index = AA_TO_INDEX.get(residue_aa, AA_TO_INDEX["X"]) / float(len(AA_TO_INDEX))
    return np.asarray(
        one_hot + [electronegativity, covalent_radius, donor, acceptor, partial_charge, residue_index],
        dtype=np.float32,
    )


def generate_pseudo_ca_coords(length: int) -> np.ndarray:
    coords = []
    for index in range(length):
        angle = index * 1.7
        x_coord = 2.3 * math.cos(angle)
        y_coord = 2.3 * math.sin(angle)
        z_coord = index * 1.5
        coords.append([x_coord, y_coord, z_coord])
    return np.asarray(coords, dtype=np.float32)


def write_pseudo_pdb(entry: str, sequence: str, target_path: str) -> str:
    coords = generate_pseudo_ca_coords(len(sequence))
    lines = [
        f"HEADER    PSEUDO STRUCTURE FOR {entry}",
        "REMARK    GENERATED BY DetEC BULK CACHE FALLBACK",
    ]
    for index, (x_coord, y_coord, z_coord) in enumerate(coords, start=1):
        lines.append(
            f"ATOM  {index:5d}  CA  GLY A{index:4d}    "
            f"{x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}"
            f"{1.00:6.2f}{0.00:6.2f}           C"
        )
    lines.append("END")
    with open(target_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return target_path


def is_pseudo_structure_file(path: str) -> bool:
    if not path or not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            head = "".join(handle.readline() for _ in range(3))
        return "GENERATED BY DetEC BULK CACHE FALLBACK" in head
    except OSError:
        return False


def load_ca_coords_from_pdb(path: str, min_plddt: float = 0.0) -> np.ndarray:
    coords = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            x_coord = float(line[30:38])
            y_coord = float(line[38:46])
            z_coord = float(line[46:54])
            b_factor = float(line[60:66]) if line[60:66].strip() else 100.0
            if b_factor < min_plddt:
                continue
            coords.append([x_coord, y_coord, z_coord])
    return np.asarray(coords, dtype=np.float32)


def prepare_coordinates(sequence: str, structure_path: Optional[str], min_plddt: float) -> np.ndarray:
    if structure_path and os.path.exists(structure_path):
        coords = load_ca_coords_from_pdb(structure_path, min_plddt=min_plddt)
        if len(coords) == len(sequence):
            return coords
        if len(coords) > 0:
            if len(coords) > len(sequence):
                return coords[: len(sequence)]
            padding = generate_pseudo_ca_coords(len(sequence) - len(coords))
            padding += coords[-1]
            return np.concatenate([coords, padding], axis=0).astype(np.float32)
    return generate_pseudo_ca_coords(len(sequence))


def load_atoms_from_pdb(path: str, min_plddt: float = 0.0) -> List[Dict[str, object]]:
    atoms: List[Dict[str, object]] = []
    residue_to_index: Dict[Tuple[str, str, str], int] = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            alt_loc = line[16].strip()
            if alt_loc and alt_loc != "A":
                continue

            x_coord = float(line[30:38])
            y_coord = float(line[38:46])
            z_coord = float(line[46:54])
            b_factor = float(line[60:66]) if line[60:66].strip() else 100.0
            if b_factor < min_plddt:
                continue

            element = line[76:78].strip()
            if not element:
                letters = re.sub(r"[^A-Za-z]", "", atom_name)
                element = letters[:1] if letters else "C"
            element = element.upper()
            if element.startswith("H"):
                continue

            chain_id = line[21].strip() or "A"
            residue_seq = line[22:26].strip()
            insertion_code = line[26].strip()
            residue_key = (chain_id, residue_seq, insertion_code)
            residue_index = residue_to_index.setdefault(residue_key, len(residue_to_index))

            atoms.append(
                {
                    "atom_name": atom_name,
                    "element": element,
                    "coord": np.asarray([x_coord, y_coord, z_coord], dtype=np.float32),
                    "residue_index": residue_index,
                }
            )
    return atoms


def select_focus_residues(active_mask: np.ndarray, max_centers: int = 8, threshold: float = 0.35) -> List[int]:
    active_mask = np.asarray(active_mask, dtype=np.float32)
    if active_mask.size == 0:
        return []

    candidate_indices = np.where(active_mask >= threshold)[0].tolist()
    if not candidate_indices:
        top_k = min(max_centers, active_mask.size)
        candidate_indices = np.argsort(active_mask)[-top_k:].tolist()
    if len(candidate_indices) > max_centers:
        ranked = sorted(candidate_indices, key=lambda index: float(active_mask[index]), reverse=True)
        candidate_indices = ranked[:max_centers]
    return sorted({int(index) for index in candidate_indices})


def extract_local_atom_environment(
    sequence: str,
    structure_path: Optional[str],
    residue_coords: np.ndarray,
    focus_residue_indices: Sequence[int],
    min_plddt: float,
    radius: float = 6.0,
    max_atoms: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_focus = [index for index in focus_residues if 0 <= index < len(sequence)] if (focus_residues := list(focus_residue_indices)) else []
    if not valid_focus:
        valid_focus = list(range(min(4, len(sequence))))

    center_coords = residue_coords[valid_focus] if len(residue_coords) else generate_pseudo_ca_coords(len(sequence))[valid_focus]
    atoms = []
    if structure_path and os.path.exists(structure_path):
        try:
            atoms = load_atoms_from_pdb(structure_path, min_plddt=min_plddt)
        except (OSError, ValueError):
            atoms = []

    selected_atoms = []
    if atoms:
        for atom in atoms:
            atom_coord = atom["coord"]
            min_distance = float(np.linalg.norm(center_coords - atom_coord[None, :], axis=1).min())
            if min_distance <= radius:
                atom["center_distance"] = min_distance
                selected_atoms.append(atom)
        if not selected_atoms:
            selected_atoms = atoms
        selected_atoms = sorted(selected_atoms, key=lambda atom: float(atom.get("center_distance", 0.0)))[:max_atoms]

    if not selected_atoms:
        fallback_indices = valid_focus[:max_atoms]
        fallback_features = []
        fallback_coords = []
        fallback_residue_ids = []
        for residue_index in fallback_indices:
            fallback_features.append(build_atom_feature("C", sequence[residue_index], "CA"))
            fallback_coords.append(residue_coords[residue_index])
            fallback_residue_ids.append(residue_index)
        return (
            np.asarray(fallback_features, dtype=np.float32),
            np.asarray(fallback_coords, dtype=np.float32),
            np.asarray(fallback_residue_ids, dtype=np.int64),
        )

    atom_features = []
    atom_coords = []
    atom_residue_ids = []
    for atom in selected_atoms:
        residue_index = min(int(atom["residue_index"]), len(sequence) - 1)
        atom_features.append(build_atom_feature(str(atom["element"]), sequence[residue_index], str(atom["atom_name"])))
        atom_coords.append(atom["coord"])
        atom_residue_ids.append(residue_index)

    return (
        np.asarray(atom_features, dtype=np.float32),
        np.asarray(atom_coords, dtype=np.float32),
        np.asarray(atom_residue_ids, dtype=np.int64),
    )


def _download_url(url: str, timeout: int) -> Optional[bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (DetEC)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _safe_json_loads(raw_bytes: bytes):
    return json.loads(raw_bytes.decode("utf-8"))


def _alphafold_metadata(entry: str, timeout: int) -> Optional[dict]:
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{urllib.parse.quote(entry)}"
    try:
        raw = _download_url(url, timeout=timeout)
    except (urllib.error.URLError, TimeoutError, ValueError, urllib.error.HTTPError):
        return None
    data = _safe_json_loads(raw)
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict):
        return data
    return None


def _extract_first_url(node) -> Optional[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str) and value.startswith("http") and ("pdb" in value or "cif" in value):
                return value
            nested = _extract_first_url(value)
            if nested:
                return nested
    elif isinstance(node, list):
        for item in node:
            nested = _extract_first_url(item)
            if nested:
                return nested
    return None


def _download_alphafold_pdb(entry: str, target_path: str, timeout: int) -> bool:
    metadata = _alphafold_metadata(entry, timeout=timeout)
    if metadata:
        download_url = metadata.get("pdbUrl") or metadata.get("bcifUrl") or _extract_first_url(metadata)
        if download_url:
            raw = _download_url(download_url, timeout=timeout)
            with open(target_path, "wb") as handle:
                handle.write(raw)
            return True

    for version in ("v6", "v5", "v4"):
        fallback_url = f"https://alphafold.ebi.ac.uk/files/AF-{entry}-F1-model_{version}.pdb"
        try:
            raw = _download_url(fallback_url, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, ValueError, urllib.error.HTTPError):
            continue
        with open(target_path, "wb") as handle:
            handle.write(raw)
        return True
    return False


def _download_esmfold_pdb(sequence: str, target_path: str, timeout: int) -> bool:
    request = urllib.request.Request(
        "https://api.esmatlas.com/foldSequence/v1/pdb/",
        data=sequence.encode("utf-8"),
        headers={
            "Content-Type": "text/plain",
            "User-Agent": "DetEC/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    with open(target_path, "wb") as handle:
        handle.write(raw)
    return True


def _load_mapping_cache(cache_path: str) -> Dict[str, Optional[str]]:
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {str(key): (str(value) if value is not None else None) for key, value in payload.items()}
    except (OSError, ValueError, TypeError):
        return {}


def _save_mapping_cache(cache_path: str, mapping: Dict[str, Optional[str]]):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(mapping, handle, ensure_ascii=False, indent=2)


def resolve_to_uniprot_accession(entry: str, cache_path: str, timeout: int = 45) -> Optional[str]:
    if UNIPROT_PATTERN.match(entry):
        return entry
    if not REFSEQ_PATTERN.match(entry):
        return None

    cache = _load_mapping_cache(cache_path)
    if entry in cache:
        return cache[entry]

    url = (
        "https://idmapping.uniprot.org/cgi-bin/idmapping_http_client3"
        f"?async=NO&from=P_REFSEQ_AC&to=ACC&ids={urllib.parse.quote(entry)}"
    )
    try:
        raw = _download_url(url, timeout=timeout)
        decoded = raw.decode("utf-8", errors="ignore")
    except (urllib.error.URLError, TimeoutError, ValueError, urllib.error.HTTPError):
        return None

    mapped = None
    for line in decoded.splitlines():
        parts = [part.strip() for part in line.split("\t") if part.strip()]
        if len(parts) >= 2 and parts[0] == entry:
            mapped = parts[1]
            break

    cache[entry] = mapped
    _save_mapping_cache(cache_path, cache)
    return mapped


def ensure_structure_file(
    entry: str,
    sequence: str,
    structure_dir: str,
    allow_download: bool,
    mapping_cache_path: Optional[str] = None,
    timeout: int = 45,
) -> Optional[str]:
    os.makedirs(structure_dir, exist_ok=True)
    file_name = re.sub(r"[^A-Za-z0-9_.-]", "_", entry) + ".pdb"
    target_path = os.path.join(structure_dir, file_name)
    had_existing_pseudo = os.path.exists(target_path) and is_pseudo_structure_file(target_path)
    if os.path.exists(target_path):
        if not allow_download or not had_existing_pseudo:
            return target_path
    if not allow_download:
        return None

    candidate_accessions = [entry]
    if mapping_cache_path:
        mapped_accession = resolve_to_uniprot_accession(entry, mapping_cache_path, timeout=timeout)
        if mapped_accession and mapped_accession not in candidate_accessions:
            candidate_accessions.append(mapped_accession)

    try:
        for accession in candidate_accessions:
            if UNIPROT_PATTERN.match(accession):
                if _download_alphafold_pdb(accession, target_path, timeout=timeout):
                    return target_path
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        pass

    try:
        if _download_esmfold_pdb(sequence, target_path, timeout=timeout):
            return target_path
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        if had_existing_pseudo and os.path.exists(target_path):
            return target_path
        return None
    if had_existing_pseudo and os.path.exists(target_path):
        return target_path
    return None


def _extract_residue_positions(node) -> List[int]:
    positions = []
    if isinstance(node, dict):
        for key, value in node.items():
            lowered = key.lower()
            if lowered in {"resid", "residue_number", "seq_num", "sequence_position", "resnum", "residue_id"}:
                try:
                    positions.append(int(value))
                except (TypeError, ValueError):
                    pass
            positions.extend(_extract_residue_positions(value))
    elif isinstance(node, list):
        for item in node:
            positions.extend(_extract_residue_positions(item))
    return positions


def _resolve_p2rank_launcher(p2rank_root: Optional[str]) -> Optional[str]:
    if not p2rank_root:
        return None
    candidates = [
        os.path.join(p2rank_root, "prank.bat"),
        os.path.join(p2rank_root, "prank"),
        os.path.join(p2rank_root, "distro", "prank.bat"),
        os.path.join(p2rank_root, "distro", "prank"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _resolve_java_executable(java_path: Optional[str]) -> Optional[str]:
    if java_path and os.path.exists(java_path):
        return java_path
    return shutil.which("java")


def is_p2rank_available(p2rank_root: Optional[str], java_path: Optional[str] = None) -> bool:
    return _resolve_p2rank_launcher(p2rank_root) is not None and _resolve_java_executable(java_path) is not None


def _choose_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered_to_original = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lowered_to_original:
            return lowered_to_original[candidate]
    for fieldname in fieldnames:
        lowered = fieldname.lower()
        if any(candidate in lowered for candidate in candidates):
            return fieldname
    return None


def _extract_residue_position_from_row(row: Dict[str, str], residue_column: Optional[str]) -> Optional[int]:
    if residue_column:
        residue_index = _parse_int_like(row.get(residue_column))
        if residue_index is not None:
            return residue_index

    for key, value in row.items():
        lowered = key.lower()
        if "residue" not in lowered and lowered not in {"resid", "resnum", "residue_number", "seq_num"}:
            continue
        residue_index = _parse_int_like(value)
        if residue_index is not None:
            return residue_index
        if value:
            match = re.search(r"(-?\d+)", str(value))
            if match:
                return int(match.group(1))
    return None


def _parse_p2rank_residue_predictions(
    residues_csv_path: str,
    sequence_length: int,
    probability_threshold: float,
    top_pockets: int,
) -> Dict[str, Any]:
    with open(residues_csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        residue_column = _choose_column(
            fieldnames,
            ("residue_number", "residue id", "residue_id", "resid", "resnum", "seq_num", "residue"),
        )
        probability_column = _choose_column(
            fieldnames,
            ("probability", "residue_probability", "prob", "ligandability_probability"),
        )
        score_column = _choose_column(
            fieldnames,
            ("score", "residue_score", "ligandability", "prediction_score"),
        )
        pocket_column = _choose_column(
            fieldnames,
            ("pocket", "pocket_rank", "pocket_id", "cluster"),
        )

        selected_scores: Dict[int, float] = {}
        ranked_candidates: List[Tuple[int, float]] = []
        fallback_used = False

        for row in reader:
            residue_index = _extract_residue_position_from_row(row, residue_column)
            if residue_index is None or residue_index < 1 or residue_index > sequence_length:
                continue

            pocket_rank = _parse_int_like(row.get(pocket_column)) if pocket_column else 1
            if pocket_rank is not None and pocket_rank <= 0:
                continue
            if top_pockets > 0 and pocket_rank is not None and pocket_rank > top_pockets:
                continue

            score = None
            if probability_column:
                score = _parse_float_like(row.get(probability_column))
            if score is None and score_column:
                score = _parse_float_like(row.get(score_column))
            if score is None:
                score = 0.0

            ranked_candidates.append((residue_index, float(score)))
            if score >= probability_threshold:
                selected_scores[residue_index] = max(selected_scores.get(residue_index, 0.0), float(score))

        if not selected_scores and ranked_candidates:
            fallback_used = True
            for residue_index, score in sorted(ranked_candidates, key=lambda item: item[1], reverse=True)[:8]:
                selected_scores[residue_index] = max(selected_scores.get(residue_index, 0.0), float(score))

    residues = sorted(selected_scores)
    return {
        "residues": residues,
        "source": "p2rank",
        "scores": {str(index): float(selected_scores[index]) for index in residues},
        "metadata": {
            "file": os.path.relpath(residues_csv_path, start=os.getcwd()),
            "fallback_used": fallback_used,
        },
        "updated_at": _timestamp(),
    }


def predict_p2rank_active_sites(
    entry: str,
    sequence: str,
    structure_path: Optional[str],
    p2rank_root: Optional[str],
    output_dir: str,
    java_path: Optional[str] = None,
    profile: str = "alphafold",
    threads: int = 1,
    probability_threshold: float = 0.25,
    top_pockets: int = 1,
    visualizations: bool = False,
    timeout: int = 600,
) -> Optional[Dict[str, Any]]:
    if not structure_path or not os.path.exists(structure_path):
        return None

    launcher = _resolve_p2rank_launcher(p2rank_root)
    java_executable = _resolve_java_executable(java_path)
    if launcher is None or java_executable is None:
        return None

    entry_output_dir = os.path.join(output_dir, re.sub(r"[^A-Za-z0-9_.-]", "_", entry))
    os.makedirs(entry_output_dir, exist_ok=True)

    command = [launcher, "predict", "-f", os.path.abspath(structure_path), "-o", os.path.abspath(entry_output_dir)]
    if profile:
        command.extend(["-c", profile])
    command.extend(["-threads", str(max(1, threads))])
    if not visualizations:
        command.extend(["-visualizations", "0"])

    env = os.environ.copy()
    env["JAVA_HOME"] = os.path.dirname(os.path.dirname(java_executable))
    env["PATH"] = os.path.dirname(java_executable) + os.pathsep + env.get("PATH", "")

    if launcher.lower().endswith(".bat"):
        command = ["cmd", "/c"] + command

    completed = subprocess.run(
        command,
        cwd=os.path.dirname(launcher),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        return None

    residue_files = []
    for root, _, file_names in os.walk(entry_output_dir):
        for file_name in file_names:
            if file_name.endswith("_residues.csv"):
                residue_files.append(os.path.join(root, file_name))
    if not residue_files:
        return None

    residue_files.sort(key=os.path.getmtime, reverse=True)
    record = _parse_p2rank_residue_predictions(
        residue_files[0],
        sequence_length=len(sequence),
        probability_threshold=probability_threshold,
        top_pockets=top_pockets,
    )
    record["metadata"].update(
        {
            "entry": entry,
            "profile": profile,
            "threads": max(1, threads),
        }
    )
    return record


def ensure_active_site_annotations(entry: str, sequence: str, cache_path: str, timeout: int = 45) -> List[int]:
    if not UNIPROT_PATTERN.match(entry):
        return []
    url = (
        "https://www.ebi.ac.uk/thornton-srv/m-csa/api/residues/"
        f"?format=json&entries.proteins.sequences.uniprot_ids={urllib.parse.quote(entry)}"
    )
    try:
        raw = _download_url(url, timeout=timeout)
        payload = _safe_json_loads(raw)
    except (urllib.error.URLError, TimeoutError, ValueError):
        return []

    residues = sorted({pos for pos in _extract_residue_positions(payload) if 1 <= pos <= len(sequence)})
    if not residues:
        return []

    existing = load_active_site_cache(cache_path)
    existing[entry] = {
        "residues": residues,
        "source": "mcsa",
        "scores": {str(pos): 1.0 for pos in residues},
        "metadata": {"entry": entry},
        "updated_at": _timestamp(),
    }
    save_active_site_cache(cache_path, existing)
    return residues


def ensure_active_site_record(
    entry: str,
    sequence: str,
    structure_path: Optional[str],
    cache_path: str,
    timeout: int = 45,
    use_p2rank: bool = False,
    p2rank_root: Optional[str] = None,
    p2rank_output_dir: Optional[str] = None,
    java_path: Optional[str] = None,
    p2rank_profile: str = "alphafold",
    p2rank_threads: int = 1,
    p2rank_probability_threshold: float = 0.25,
    p2rank_top_pockets: int = 1,
    p2rank_visualizations: bool = False,
) -> Dict[str, Any]:
    cache = load_active_site_cache(cache_path)
    cached_record = cache.get(entry)
    if cached_record and active_site_record_to_indices(cached_record):
        return cached_record

    if UNIPROT_PATTERN.match(entry):
        residues = ensure_active_site_annotations(entry, sequence, cache_path, timeout=timeout)
        if residues:
            return load_active_site_cache(cache_path).get(entry, _normalize_active_site_record(residues))

    if use_p2rank:
        predicted_record = predict_p2rank_active_sites(
            entry=entry,
            sequence=sequence,
            structure_path=structure_path,
            p2rank_root=p2rank_root,
            output_dir=p2rank_output_dir or os.path.join(os.path.dirname(cache_path), "p2rank_outputs"),
            java_path=java_path,
            profile=p2rank_profile,
            threads=p2rank_threads,
            probability_threshold=p2rank_probability_threshold,
            top_pockets=p2rank_top_pockets,
            visualizations=p2rank_visualizations,
            timeout=max(timeout, 600),
        )
        if predicted_record and active_site_record_to_indices(predicted_record):
            cache = load_active_site_cache(cache_path)
            cache[entry] = predicted_record
            save_active_site_cache(cache_path, cache)
            return predicted_record

    return cached_record or {
        "residues": [],
        "source": "heuristic",
        "scores": {},
        "metadata": {},
        "updated_at": "",
    }


def build_active_site_mask(
    sequence: str,
    coords: np.ndarray,
    active_site_record: Optional[Union[Dict[str, Any], Sequence[int]]],
) -> np.ndarray:
    length = len(sequence)
    normalized_record = _normalize_active_site_record(active_site_record or {})
    active_indices = active_site_record_to_indices(normalized_record)
    score_lookup = normalized_record.get("scores", {})

    if active_indices:
        zero_based = np.asarray([max(0, min(length - 1, int(index) - 1)) for index in active_indices], dtype=np.int64)
        weights = []
        for index in active_indices:
            score = _parse_float_like(score_lookup.get(str(index)))
            weights.append(score if score is not None else 1.0)
        weights = np.asarray(weights, dtype=np.float32)
        if weights.size:
            weights = 0.35 + 0.65 * (weights / max(float(weights.max()), 1e-6))

        mask = np.zeros(length, dtype=np.float32)
        positions = np.arange(length, dtype=np.float32)
        for center, weight in zip(zero_based, weights.tolist()):
            mask = np.maximum(mask, weight * np.exp(-((positions - center) ** 2) / (2 * 12.0)))
        return mask.astype(np.float32)

    catalytic_like = {"C", "D", "E", "H", "K", "R", "S", "T", "Y", "N", "Q"}
    residue_scores = np.array([1.0 if aa in catalytic_like else 0.2 for aa in sequence], dtype=np.float32)
    if len(coords) > 1:
        pairwise = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        with np.errstate(over="ignore"):
            density = np.exp(-(pairwise ** 2) / (2 * 4.0 ** 2)).sum(axis=1)
        residue_scores = residue_scores + density / max(density.max(), 1.0)

    top_k = min(6, max(1, length // 100 + 1))
    top_indices = np.argsort(residue_scores)[-top_k:]
    mask = np.zeros(length, dtype=np.float32)
    positions = np.arange(length, dtype=np.float32)
    for center in top_indices:
        mask = np.maximum(mask, np.exp(-((positions - center) ** 2) / (2 * 12.0)))
    return mask.astype(np.float32)


def _self_loop_graph(num_nodes: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    nodes = torch.arange(num_nodes, device=device, dtype=torch.long)
    edge_index = torch.stack([nodes, nodes], dim=0)
    distances = torch.ones(num_nodes, device=device, dtype=torch.float32)
    return edge_index, distances


def build_scale_edges(coords: torch.Tensor, scales: Sequence[float], knn: int = 20, geometry: Optional[torch.Tensor] = None):
    num_nodes = coords.size(0)
    if num_nodes <= 1:
        return [_self_loop_graph(num_nodes, coords.device)[0] for _ in scales], [
            _self_loop_graph(num_nodes, coords.device)[1] for _ in scales
        ]

    pairwise = torch.cdist(coords, coords)
    local_similarity = struct_similarity = functional_similarity = None
    if geometry is not None and geometry.size(0) == num_nodes and geometry.size(1) >= 14:
        geo_features = F.normalize(geometry[:, 3:], dim=-1)
        local_similarity = geo_features @ geo_features.transpose(0, 1)

        frame_features = F.normalize(geometry[:, 5:].reshape(num_nodes, -1), dim=-1)
        struct_similarity = frame_features @ frame_features.transpose(0, 1)

        density = geometry[:, 3:4]
        curvature = geometry[:, 4:5]
        functional_features = torch.cat([1.0 - density, curvature, density * curvature], dim=-1)
        functional_features = F.normalize(functional_features, dim=-1)
        functional_similarity = functional_features @ functional_features.transpose(0, 1)

    edge_indices = []
    edge_dists = []
    lower_bound = 0.0
    for scale_index, upper_bound in enumerate(scales):
        mask = (pairwise > lower_bound) & (pairwise <= upper_bound)
        if scale_index == 0 and local_similarity is not None:
            mask = mask & (local_similarity > 0.65)
        elif scale_index == 1 and struct_similarity is not None:
            mask = mask & (struct_similarity > 0.60)
        elif scale_index >= 2 and functional_similarity is not None:
            mask = mask & (functional_similarity > 0.55)

        edges = []
        dists = []
        for node_index in range(num_nodes):
            neighbors = torch.nonzero(mask[node_index], as_tuple=False).squeeze(-1)
            if neighbors.numel() == 0:
                continue
            distances = pairwise[node_index, neighbors]
            keep = min(knn, neighbors.numel())
            topk = torch.topk(distances, k=keep, largest=False)
            selected_neighbors = neighbors[topk.indices]
            selected_distances = distances[topk.indices]
            for neighbor, distance in zip(selected_neighbors.tolist(), selected_distances.tolist()):
                edges.append([node_index, neighbor])
                dists.append(distance)
        if not edges:
            edge_index, dist = _self_loop_graph(num_nodes, coords.device)
        else:
            edge_index = torch.tensor(edges, device=coords.device, dtype=torch.long).t().contiguous()
            dist = torch.tensor(dists, device=coords.device, dtype=torch.float32)
        edge_indices.append(edge_index)
        edge_dists.append(dist)
        lower_bound = upper_bound
    return edge_indices, edge_dists


def build_local_edges(coords: torch.Tensor, active_mask: torch.Tensor, cutoff: float, n_rbf: int):
    num_nodes = coords.size(0)
    if num_nodes <= 1:
        edge_index, edge_dist = _self_loop_graph(num_nodes, coords.device)
        return edge_index, radial_basis(edge_dist, n_rbf=n_rbf, cutoff=cutoff)

    pairwise = torch.cdist(coords, coords)
    focus = active_mask > 0.15
    mask = (pairwise > 0) & (pairwise <= cutoff)
    if focus.any():
        mask = mask & (focus.unsqueeze(0) | focus.unsqueeze(1))

    row, col = torch.nonzero(mask, as_tuple=True)
    if row.numel() == 0:
        edge_index, edge_dist = _self_loop_graph(num_nodes, coords.device)
    else:
        edge_index = torch.stack([row, col], dim=0)
        edge_dist = pairwise[row, col]
    return edge_index, radial_basis(edge_dist, n_rbf=n_rbf, cutoff=cutoff)


def build_local_atom_edges(coords: torch.Tensor, cutoff: float, n_rbf: int):
    num_nodes = coords.size(0)
    if num_nodes <= 1:
        edge_index, edge_dist = _self_loop_graph(num_nodes, coords.device)
        return edge_index, radial_basis(edge_dist, n_rbf=n_rbf, cutoff=cutoff)

    pairwise = torch.cdist(coords, coords)
    mask = (pairwise > 0) & (pairwise <= cutoff)
    row, col = torch.nonzero(mask, as_tuple=True)
    if row.numel() == 0:
        edge_index, edge_dist = _self_loop_graph(num_nodes, coords.device)
    else:
        edge_index = torch.stack([row, col], dim=0)
        edge_dist = pairwise[row, col]
    return edge_index, radial_basis(edge_dist, n_rbf=n_rbf, cutoff=cutoff)


def radial_basis(distances: torch.Tensor, n_rbf: int = 16, cutoff: float = 6.0) -> torch.Tensor:
    centers = torch.linspace(0, cutoff, n_rbf, device=distances.device)
    gamma = 1.0 / max((cutoff / n_rbf) ** 2, 1e-6)
    return torch.exp(-gamma * (distances.unsqueeze(-1) - centers) ** 2)
