import json
from typing import Dict, Iterable, List, Optional


class ECTaxonomy:
    def __init__(self, level1_labels, level2_labels, level3_labels, full_labels):
        self.level1_labels = list(level1_labels)
        self.level2_labels = list(level2_labels)
        self.level3_labels = list(level3_labels)
        self.full_labels = list(full_labels)

        self.level1_to_idx = {label: index for index, label in enumerate(self.level1_labels)}
        self.level2_to_idx = {label: index for index, label in enumerate(self.level2_labels)}
        self.level3_to_idx = {label: index for index, label in enumerate(self.level3_labels)}
        self.full_to_idx = {label: index for index, label in enumerate(self.full_labels)}

        self.full_id_to_levels: Dict[int, tuple[int, int, int]] = {}
        self.prefix_to_full_ids: Dict[tuple[int, int, int], List[int]] = {}

        for full_label, full_id in self.full_to_idx.items():
            level1, level2, level3 = self._levels_from_ec(full_label)
            ids = (
                self.level1_to_idx[level1],
                self.level2_to_idx[level2],
                self.level3_to_idx[level3],
            )
            self.full_id_to_levels[full_id] = ids
            self.prefix_to_full_ids.setdefault(ids, []).append(full_id)

    @property
    def level_sizes(self):
        return [len(self.level1_labels), len(self.level2_labels), len(self.level3_labels)]

    @property
    def num_full_ecs(self):
        return len(self.full_labels)

    @staticmethod
    def _levels_from_ec(ec_number: str):
        parts = ec_number.split(".")
        if len(parts) < 4:
            parts = parts + ["0"] * (4 - len(parts))
        level1 = parts[0]
        level2 = ".".join(parts[:2])
        level3 = ".".join(parts[:3])
        return level1, level2, level3

    @classmethod
    def from_ec_collections(cls, ec_collections: Iterable[Iterable[str]]):
        full_labels = sorted({ec for collection in ec_collections for ec in collection if ec})
        level1_labels = sorted({cls._levels_from_ec(ec)[0] for ec in full_labels})
        level2_labels = sorted({cls._levels_from_ec(ec)[1] for ec in full_labels})
        level3_labels = sorted({cls._levels_from_ec(ec)[2] for ec in full_labels})
        return cls(level1_labels, level2_labels, level3_labels, full_labels)

    def encode_label(self, ec_number: str):
        if ec_number not in self.full_to_idx:
            level1, level2, level3 = self._levels_from_ec(ec_number)
            return {
                "ec": ec_number,
                "level1_id": self.level1_to_idx.get(level1),
                "level2_id": self.level2_to_idx.get(level2),
                "level3_id": self.level3_to_idx.get(level3),
                "full_id": None,
            }
        full_id = self.full_to_idx[ec_number]
        level1_id, level2_id, level3_id = self.full_id_to_levels[full_id]
        return {
            "ec": ec_number,
            "level1_id": level1_id,
            "level2_id": level2_id,
            "level3_id": level3_id,
            "full_id": full_id,
        }

    def encode_labels(self, ec_numbers: Iterable[str]):
        return [self.encode_label(ec_number) for ec_number in ec_numbers]

    def candidate_full_ids(self, level1_id: int, level2_id: int, level3_id: int):
        return self.prefix_to_full_ids.get((level1_id, level2_id, level3_id), [])

    def decode_full_id(self, full_id: int) -> str:
        return self.full_labels[full_id]

    def to_dict(self):
        return {
            "level1_labels": self.level1_labels,
            "level2_labels": self.level2_labels,
            "level3_labels": self.level3_labels,
            "full_labels": self.full_labels,
        }

    @classmethod
    def from_dict(cls, payload: Dict):
        return cls(
            level1_labels=payload["level1_labels"],
            level2_labels=payload["level2_labels"],
            level3_labels=payload["level3_labels"],
            full_labels=payload["full_labels"],
        )

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)
