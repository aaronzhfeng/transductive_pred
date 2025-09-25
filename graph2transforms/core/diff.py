from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from rdkit import Chem

from graph2transforms.edits.actions import Action, ActionType, BondOrder, BondStereo, canonical_sort_actions
from graph2transforms.features.featurize import bond_order_from_rdkit, bond_stereo_from_rdkit
from graph2transforms.mapping.rxnmap import MappedReaction


@dataclass
class DiffResult:
    actions: List[Action]
    leaving_group_fragments: List[str]


def _bond_index_by_map_pair(mol: Chem.Mol, amap: Dict[int, int]) -> Dict[Tuple[int, int], Tuple[BondOrder, BondStereo]]:
    # reverse map: atom idx -> map id
    idx_to_map: Dict[int, int] = {v: k for k, v in amap.items()}
    out: Dict[Tuple[int, int], Tuple[BondOrder, BondStereo]] = {}
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        mi = idx_to_map.get(i)
        mj = idx_to_map.get(j)
        if not mi or not mj:
            continue
        key = (mi, mj) if mi < mj else (mj, mi)
        out[key] = (bond_order_from_rdkit(b), bond_stereo_from_rdkit(b))
    return out


def _hydrogen_count_by_map(mol: Chem.Mol, amap: Dict[int, int]) -> Dict[int, int]:
    idx_to_map: Dict[int, int] = {v: k for k, v in amap.items()}
    out: Dict[int, int] = {}
    for a in mol.GetAtoms():
        mid = idx_to_map.get(a.GetIdx())
        if mid is None:
            continue
        out[mid] = a.GetTotalNumHs()
    return out


def _lg_components(smiles_r: str, r_mol: Chem.Mol, rmap: Dict[int, int], pmap: Dict[int, int]) -> List[List[int]]:
    # atoms present in reactants but not mapped to products
    r_only_maps: Set[int] = set(rmap.keys()) - set(pmap.keys())
    if not r_only_maps:
        return []
    # convert to atom indices
    r_only_idx: Set[int] = {rmap[mid] for mid in r_only_maps}
    # Build adjacency for subgraph
    visited: Set[int] = set()
    comps: List[List[int]] = []
    adj: Dict[int, List[int]] = defaultdict(list)
    for b in r_mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        if i in r_only_idx and j in r_only_idx:
            adj[i].append(j)
            adj[j].append(i)
    for v in r_only_idx:
        if v in visited:
            continue
        comp: List[int] = []
        stack = [v]
        visited.add(v)
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in adj.get(u, []):
                if w not in visited:
                    visited.add(w)
                    stack.append(w)
        comps.append(sorted(comp))
    return comps


def compute_gold_edits(mapped: MappedReaction, include_change_atom: bool = True, compress_lg: bool = False) -> DiffResult:
    r_bonds = _bond_index_by_map_pair(mapped.r_mol, mapped.rmap)
    p_bonds = _bond_index_by_map_pair(mapped.p_mol, mapped.pmap)

    actions: List[Action] = []

    # Bonds present in reactants but not in products => delete
    for key, (order, stereo) in r_bonds.items():
        if key not in p_bonds:
            i, j = key
            actions.append(Action(type=ActionType.DEL_BOND, args={"i": i, "j": j, "order": order, "stereo": stereo}))

    # Bonds present in products but not in reactants => add
    for key, (order, stereo) in p_bonds.items():
        if key not in r_bonds:
            i, j = key
            actions.append(Action(type=ActionType.ADD_BOND, args={"i": i, "j": j, "order": order, "stereo": stereo}))

    # Bonds in both but changed order/stereo
    for key, (r_order, r_stereo) in r_bonds.items():
        if key in p_bonds:
            p_order, p_stereo = p_bonds[key]
            if (r_order != p_order) or (r_stereo != p_stereo):
                i, j = key
                actions.append(
                    Action(
                        type=ActionType.CHANGE_BOND,
                        args={
                            "i": i,
                            "j": j,
                            "old_order": r_order,
                            "new_order": p_order,
                            "old_stereo": r_stereo,
                            "new_stereo": p_stereo,
                        },
                    )
                )

    # Hydrogen adjustments
    r_h = _hydrogen_count_by_map(mapped.r_mol, mapped.rmap)
    p_h = _hydrogen_count_by_map(mapped.p_mol, mapped.pmap)
    for mid in set(r_h.keys()).intersection(p_h.keys()):
        dr = p_h[mid] - r_h[mid]
        if dr > 0:
            for _ in range(dr):
                actions.append(Action(type=ActionType.H_GAIN, args={"i": mid}))
        elif dr < 0:
            for _ in range(-dr):
                actions.append(Action(type=ActionType.H_LOSS, args={"i": mid}))

    # Atom-level property changes (optional)
    if include_change_atom:
        r_idx_to_map = {v: k for k, v in mapped.rmap.items()}
        p_idx_to_map = {v: k for k, v in mapped.pmap.items()}
        # compare charge and chiral tag
        for a in mapped.r_mol.GetAtoms():
            mid = r_idx_to_map.get(a.GetIdx())
            if mid is None or mid not in mapped.pmap:
                continue
            a2 = mapped.p_mol.GetAtomWithIdx(mapped.pmap[mid])
            if a.GetFormalCharge() != a2.GetFormalCharge():
                actions.append(
                    Action(
                        type=ActionType.CHANGE_ATOM,
                        args={
                            "i": mid,
                            "prop": "formal_charge",
                            "old": a.GetFormalCharge(),
                            "new": a2.GetFormalCharge(),
                        },
                    )
                )
            if int(a.GetChiralTag()) != int(a2.GetChiralTag()):
                actions.append(
                    Action(
                        type=ActionType.CHANGE_ATOM,
                        args={
                            "i": mid,
                            "prop": "chiral_tag",
                            "old": int(a.GetChiralTag()),
                            "new": int(a2.GetChiralTag()),
                        },
                    )
                )

    lg_frags: List[str] = []
    comps = _lg_components(Chem.MolToSmiles(mapped.r_mol), mapped.r_mol, mapped.rmap, mapped.pmap)
    if compress_lg:
        for comp in comps:
            frag_smi = Chem.MolFragmentToSmiles(mapped.r_mol, atomsToUse=comp, canonical=True)
            lg_frags.append(frag_smi)
            node_set = sorted([mapped.r_mol.GetAtomWithIdx(i).GetAtomMapNum() for i in comp])
            actions.append(Action(type=ActionType.REMOVE_LG, args={"fragment_smiles": frag_smi, "node_set": node_set}))
    else:
        for comp in comps:
            actions.append(Action(type=ActionType.DELETE_SUBGRAPH, args={"node_set": sorted([mapped.r_mol.GetAtomWithIdx(i).GetAtomMapNum() for i in comp])}))

    # Terminate sequence
    actions.append(Action(type=ActionType.STOP, args={}))

    actions = canonical_sort_actions(actions)
    return DiffResult(actions=actions, leaving_group_fragments=lg_frags)
