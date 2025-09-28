from __future__ import annotations

import os
import sys
import math
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from functools import partial
from typing import Dict, Iterable, List, Tuple

import click
import numpy as _np

# Fail fast on NumPy 2.x if RDKit wheels are built against NumPy 1.x
if int(_np.__version__.split(".")[0]) >= 2:
    raise RuntimeError(
        "Detected NumPy %s. RDKit wheels in your env may be built against NumPy 1.x and crash with '_ARRAY_API not found'.\n"
        "Please install numpy<2 or upgrade RDKit to wheels compiled for NumPy 2.x.\n"
        "Recommended: pip install 'numpy<2' OR conda install 'numpy<2' in this env." % _np.__version__
    )

from rdkit import Chem
import rdkit
import rxnmapper
from tqdm import tqdm

from graph2transforms.chem.standardize import reconstruct_smiles_from_tokenized, standardize_reaction
from graph2transforms.mapping.rxnmap import ensure_atom_mapping
from graph2transforms.core.diff import compute_gold_edits
from graph2transforms.core.rollout import apply_actions_with_records
from graph2transforms.io.serialization import JsonlShardWriter, write_json
from graph2transforms.edits.actions import ActionType


def _read_pairs(src_path: str, tgt_path: str) -> List[Tuple[str, str]]:
    with open(src_path, "r", encoding="utf-8") as fs, open(tgt_path, "r", encoding="utf-8") as ft:
        src_lines = fs.readlines()
        tgt_lines = ft.readlines()
    assert len(src_lines) == len(tgt_lines), "Mismatched lines in src/tgt"
    pairs = []
    for s, t in zip(src_lines, tgt_lines):
        rs = reconstruct_smiles_from_tokenized(s)
        ps = reconstruct_smiles_from_tokenized(t)
        pairs.append((rs, ps))
    return pairs


def _canonical_no_map_smiles(m: Chem.Mol) -> str:
    mol = Chem.Mol(m)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return Chem.MolToSmiles(mol, canonical=True)


def _process_pair(reactants: str, products: str, opts: Dict) -> Dict:
    try:
        sr = standardize_reaction(reactants, products, kekulize=opts.get("kekulize", False))
        mapped = ensure_atom_mapping(sr, threshold=opts.get("map_threshold", 0.85))
        diff = compute_gold_edits(mapped, include_change_atom=True, compress_lg=opts.get("compress_lg", False))
        records, flags, final_smiles = apply_actions_with_records(mapped.r_mol, diff.actions)
        target_smiles = _canonical_no_map_smiles(mapped.p_mol)
        if flags.get("sanitize_failed", 0) == 1:
            raise RuntimeError("sanitize_failed_during_rollout")
        final_matches_target = final_smiles == target_smiles
        return {
            "reaction_id": mapped.reaction_id,
            "mapped_rxn": mapped.mapped_rxn,
            "confidence": mapped.confidence,
            "records": records,
            "lg_fragments": diff.leaving_group_fragments,
            "final_smiles": final_smiles,
            "target_smiles": target_smiles,
            "final_matches_target": final_matches_target,
            "failure": None,
        }
    except Exception as e:
        return {
            "reaction_id": None,
            "mapped_rxn": None,
            "confidence": 0.0,
            "records": [],
            "lg_fragments": [],
            "final_smiles": None,
            "target_smiles": None,
            "failure": str(e),
        }


def _chunk(lst: List, n: int) -> Iterable[List]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


@click.command()
@click.option("--data-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--out-dir", required=True, type=click.Path(file_okay=False))
@click.option("--num-workers", default=4, show_default=True, type=int)
@click.option("--map-threshold", default=0.85, show_default=True, type=float)
@click.option("--shard-size", default=20000, show_default=True, type=int)
@click.option("--kekulize/--no-kekulize", default=False, show_default=True)
@click.option("--compress-lg/--no-compress-lg", default=False, show_default=True)
@click.option("--topk-lg", default=512, show_default=True, type=int)
def main(data_dir: str, out_dir: str, num_workers: int, map_threshold: float, shard_size: int, kekulize: bool, compress_lg: bool, topk_lg: int):
    os.makedirs(out_dir, exist_ok=True)
    splits = ["train", "val", "test"]

    run_meta = {
        "map_threshold": map_threshold,
        "kekulize": kekulize,
        "compress_lg": compress_lg,
        "topk_lg": topk_lg,
        "versions": {
            "python": sys.version,
            "rdkit": getattr(rdkit, "__version__", "unknown"),
            "rxnmapper": getattr(rxnmapper, "__version__", "unknown"),
            "numpy": _np.__version__,
        },
    }

    write_json(os.path.join(out_dir, "run.json"), run_meta)

    actions_vocab = {name: i for i, name in enumerate([a.name for a in ActionType])}
    os.makedirs(os.path.join(out_dir, "vocabs"), exist_ok=True)
    write_json(os.path.join(out_dir, "vocabs", "actions.json"), actions_vocab)

    lg_counter = Counter()

    for split in splits:
        src_path = os.path.join(data_dir, f"src-{split}.txt")
        tgt_path = os.path.join(data_dir, f"tgt-{split}.txt")
        pairs = _read_pairs(src_path, tgt_path)
        out_split_dir = os.path.join(out_dir, split)
        os.makedirs(out_split_dir, exist_ok=True)

        shard_id = 0
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            opts = {"map_threshold": map_threshold, "kekulize": kekulize, "compress_lg": compress_lg}
            for chunk in _chunk(pairs, shard_size):
                shard_id += 1
                shard_path = os.path.join(out_split_dir, f"shard-{shard_id:05d}.jsonl")
                writer = JsonlShardWriter(shard_path)
                futs = [ex.submit(_process_pair, r, p, opts) for (r, p) in chunk]
                for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{split} shard {shard_id}"):
                    res = fut.result()
                    if res["failure"] is None:
                        writer.write(res)
                        for frag in res.get("lg_fragments", []):
                            if split == "train":
                                lg_counter[frag] += 1
                writer.close()

        manifest = {
            "split": split,
            "num_shards": shard_id,
        }
        write_json(os.path.join(out_dir, f"manifest-{split}.json"), manifest)

    if compress_lg and lg_counter:
        top = lg_counter.most_common(topk_lg)
        lg_vocab = {frag: i for i, (frag, _) in enumerate(top)}
        write_json(os.path.join(out_dir, "vocabs", "leaving_groups.json"), lg_vocab)


if __name__ == "__main__":
    main()
