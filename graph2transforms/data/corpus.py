from __future__ import annotations

import os
import json
from typing import Any, Dict, Iterable, List, Tuple

import ujson
import torch
from torch.utils.data import Dataset


class StepExample:
    def __init__(self, example_id: int, reaction_id: str, mapped_rxn: str, record: Dict):
        self.example_id = example_id
        self.reaction_id = reaction_id
        self.mapped_rxn = mapped_rxn
        self.record = record


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield ujson.loads(line)


def list_shard_paths(root: str, split: str) -> List[str]:
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        return []
    out = []
    for name in sorted(os.listdir(split_dir)):
        if name.endswith(".jsonl"):
            out.append(os.path.join(split_dir, name))
    return out


class EditStepDataset(Dataset):
    """Flat per-step dataset for teacher-forced training.

    Each item exposes one step record with its reaction context.
    """

    def __init__(self, shards_root: str, split: str, action_vocab_path: str):
        super().__init__()
        self.examples: List[StepExample] = []
        self.action_to_id: Dict[str, int] = json.load(open(action_vocab_path, "r"))
        eid = 0
        for shard in list_shard_paths(shards_root, split):
            for rxn in _read_jsonl(shard):
                rid = rxn.get("reaction_id") or ""
                mapped_rxn = rxn.get("mapped_rxn") or ""
                for rec in rxn.get("records", []):
                    self.examples.append(StepExample(eid, rid, mapped_rxn, rec))
                    eid += 1

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        g = ex.record["graph"]
        nodes = g["nodes"]
        edges = g["edges"]
        # Build tensors
        node_feats = torch.tensor(
            [
                [
                    n.get("z", 0),
                    n.get("degree", 0),
                    n.get("total_valence", 0),
                    n.get("formal_charge", 0),
                    n.get("hybridization", 0),
                    n.get("aromatic", 0),
                    n.get("implicit_h", 0),
                    n.get("explicit_h", 0),
                    n.get("chiral_tag", 0),
                    n.get("map_num", 0),
                ]
                for n in nodes
            ],
            dtype=torch.long,
        )
        edge_index = torch.tensor([[e["src"], e["dst"]] for e in edges], dtype=torch.long).t() if edges else torch.zeros((2, 0), dtype=torch.long)
        target = ex.record["target_action"]
        return {
            "reaction_id": ex.reaction_id,
            "mapped_rxn": ex.mapped_rxn,
            "node_feats": node_feats,
            "edge_index": edge_index,
            "target": target,
        }


def collate_steps(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Simple list-based batch to keep graphs variable-sized
    return {
        "reaction_id": [b["reaction_id"] for b in batch],
        "mapped_rxn": [b["mapped_rxn"] for b in batch],
        "node_feats": [b["node_feats"] for b in batch],
        "edge_index": [b["edge_index"] for b in batch],
        "target": [b["target"] for b in batch],
    }


class ReactionSequenceDataset(Dataset):
    """Sequence dataset for autoregressive validation.

    Returns all step records for a reaction.
    """

    def __init__(self, shards_root: str, split: str):
        super().__init__()
        self.samples: List[Dict[str, Any]] = []
        for shard in list_shard_paths(shards_root, split):
            for rxn in _read_jsonl(shard):
                self.samples.append(rxn)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

