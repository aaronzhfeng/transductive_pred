from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from rdkit import Chem

from graph2transforms.edits.actions import Action, ActionType, BondOrder, BondStereo
from graph2transforms.features.featurize import serialize_graph
from graph2transforms.core.masks import compute_masks


BOND_ORDER_TO_RDKIT = {
    BondOrder.SINGLE.value: Chem.BondType.SINGLE,
    BondOrder.DOUBLE.value: Chem.BondType.DOUBLE,
    BondOrder.TRIPLE.value: Chem.BondType.TRIPLE,
    BondOrder.AROMATIC.value: Chem.BondType.AROMATIC,
}


def _sanitize(m: Chem.Mol) -> Chem.Mol:
    Chem.SanitizeMol(m)
    return m


def _apply_h_edit(rw: Chem.RWMol, atom_idx: int, delta: int) -> None:
    a = rw.GetAtomWithIdx(atom_idx)
    new_h = max(0, a.GetNumExplicitHs() + delta)
    a.SetNumExplicitHs(new_h)
    a.UpdatePropertyCache()


def _set_bond_stereo_double(rw: Chem.RWMol, i: int, j: int, stereo: str) -> None:
    b = rw.GetBondBetweenAtoms(i, j)
    if b is None:
        return
    if b.GetBondType() != Chem.BondType.DOUBLE:
        return
    ni = [n.GetIdx() for n in rw.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() != j]
    nj = [n.GetIdx() for n in rw.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != i]
    if not ni or not nj:
        return
    b.SetStereoAtoms(ni[0], nj[0])
    if stereo == BondStereo.E.value:
        b.SetStereo(Chem.BondStereo.STEREOE)
    elif stereo == BondStereo.Z.value:
        b.SetStereo(Chem.BondStereo.STEREOZ)
    else:
        b.SetStereo(Chem.BondStereo.STEREONONE)


def _canonical_no_map_smiles(m: Chem.Mol) -> str:
    mol = Chem.Mol(m)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return Chem.MolToSmiles(mol, canonical=True)


def apply_actions_with_records(mol: Chem.Mol, actions: Iterable[Action]) -> Tuple[List[Dict], Dict[str, int], str]:
    rw = Chem.RWMol(mol)
    records: List[Dict] = []

    step = 1
    final_flags = {"sanitize_failed": 0}
    for act in actions:
        # recompute mapping from map number -> current atom index at each step
        map_to_idx: Dict[int, int] = {a.GetAtomMapNum(): a.GetIdx() for a in rw.GetAtoms() if a.GetAtomMapNum() > 0}

        graph = serialize_graph(rw)
        masks = compute_masks(rw)
        record = {
            "step": step,
            "graph": graph,
            "target_action": act.to_dict(),
            "maps": {int(k): int(v) for k, v in map_to_idx.items()},
            "masks": masks,
            "sanity_flags": {},
        }
        records.append(record)

        if act.type == ActionType.STOP:
            break

        try:
            if act.type == ActionType.DEL_BOND:
                i_map = int(act.args["i"]) ; j_map = int(act.args["j"]) 
                i = map_to_idx.get(i_map) ; j = map_to_idx.get(j_map)
                if i is not None and j is not None:
                    rw.RemoveBond(i, j)

            elif act.type == ActionType.ADD_BOND:
                i_map = int(act.args["i"]) ; j_map = int(act.args["j"]) 
                i = map_to_idx.get(i_map) ; j = map_to_idx.get(j_map)
                order = str(act.args.get("order", BondOrder.SINGLE.value))
                stereo = str(act.args.get("stereo", BondStereo.NONE.value))
                if i is not None and j is not None:
                    rw.AddBond(i, j, BOND_ORDER_TO_RDKIT.get(order, Chem.BondType.SINGLE))
                    _set_bond_stereo_double(rw, i, j, stereo)

            elif act.type == ActionType.CHANGE_BOND:
                i_map = int(act.args["i"]) ; j_map = int(act.args["j"]) 
                i = map_to_idx.get(i_map) ; j = map_to_idx.get(j_map)
                new_order = str(act.args.get("new_order", BondOrder.SINGLE.value))
                new_stereo = str(act.args.get("new_stereo", BondStereo.NONE.value))
                if i is not None and j is not None:
                    b = rw.GetBondBetweenAtoms(i, j)
                    if b is not None:
                        b.SetBondType(BOND_ORDER_TO_RDKIT.get(new_order, Chem.BondType.SINGLE))
                        _set_bond_stereo_double(rw, i, j, new_stereo)

            elif act.type == ActionType.H_GAIN:
                i_map = int(act.args["i"]) ; i = map_to_idx.get(i_map)
                if i is not None:
                    _apply_h_edit(rw, i, +1)

            elif act.type == ActionType.H_LOSS:
                i_map = int(act.args["i"]) ; i = map_to_idx.get(i_map)
                if i is not None:
                    _apply_h_edit(rw, i, -1)

            elif act.type in (ActionType.DELETE_SUBGRAPH, ActionType.REMOVE_LG):
                node_set = act.args.get("node_set") or []
                # map numbers -> current indices; delete in reverse-sorted order
                idxs = [map_to_idx[mid] for mid in node_set if mid in map_to_idx]
                for idx in sorted(idxs, reverse=True):
                    rw.RemoveAtom(idx)

            Chem.SanitizeMol(rw)
        except Exception:
            final_flags["sanitize_failed"] = 1
            break

        step += 1

    final_smiles = _canonical_no_map_smiles(rw.GetMol())
    return records, final_flags, final_smiles


def get_map_to_idx(rw: Chem.RWMol) -> Dict[int, int]:
    return {a.GetAtomMapNum(): a.GetIdx() for a in rw.GetAtoms() if a.GetAtomMapNum() > 0}


def apply_single_action(rw: Chem.RWMol, act: Action) -> Tuple[Chem.RWMol, bool]:
    """Apply a single edit action in-place to the provided RWMol.

    Returns the (possibly) modified RWMol and a boolean flag indicating
    whether sanitization failed during the application.
    """
    failed = False
    map_to_idx: Dict[int, int] = get_map_to_idx(rw)
    try:
        if act.type == ActionType.DEL_BOND:
            i_map = int(act.args["i"]) ; j_map = int(act.args["j"]) 
            i = map_to_idx.get(i_map) ; j = map_to_idx.get(j_map)
            if i is not None and j is not None:
                rw.RemoveBond(i, j)

        elif act.type == ActionType.ADD_BOND:
            i_map = int(act.args["i"]) ; j_map = int(act.args["j"]) 
            i = map_to_idx.get(i_map) ; j = map_to_idx.get(j_map)
            order = str(act.args.get("order", BondOrder.SINGLE.value))
            stereo = str(act.args.get("stereo", BondStereo.NONE.value))
            if i is not None and j is not None:
                rw.AddBond(i, j, BOND_ORDER_TO_RDKIT.get(order, Chem.BondType.SINGLE))
                _set_bond_stereo_double(rw, i, j, stereo)

        elif act.type == ActionType.CHANGE_BOND:
            i_map = int(act.args["i"]) ; j_map = int(act.args["j"]) 
            i = map_to_idx.get(i_map) ; j = map_to_idx.get(j_map)
            new_order = str(act.args.get("new_order", BondOrder.SINGLE.value))
            new_stereo = str(act.args.get("new_stereo", BondStereo.NONE.value))
            if i is not None and j is not None:
                b = rw.GetBondBetweenAtoms(i, j)
                if b is not None:
                    b.SetBondType(BOND_ORDER_TO_RDKIT.get(new_order, Chem.BondType.SINGLE))
                    _set_bond_stereo_double(rw, i, j, new_stereo)

        elif act.type == ActionType.H_GAIN:
            i_map = int(act.args["i"]) ; i = map_to_idx.get(i_map)
            if i is not None:
                _apply_h_edit(rw, i, +1)

        elif act.type == ActionType.H_LOSS:
            i_map = int(act.args["i"]) ; i = map_to_idx.get(i_map)
            if i is not None:
                _apply_h_edit(rw, i, -1)

        elif act.type in (ActionType.DELETE_SUBGRAPH, ActionType.REMOVE_LG):
            node_set = act.args.get("node_set") or []
            idxs = [map_to_idx[mid] for mid in node_set if mid in map_to_idx]
            for idx in sorted(idxs, reverse=True):
                rw.RemoveAtom(idx)

        elif act.type == ActionType.CHANGE_ATOM:
            i_map = int(act.args["i"]) ; i = map_to_idx.get(i_map)
            if i is not None:
                a = rw.GetAtomWithIdx(i)
                prop = str(act.args.get("prop"))
                if prop == "formal_charge":
                    a.SetFormalCharge(int(act.args.get("new", a.GetFormalCharge())))
                elif prop == "chiral_tag":
                    try:
                        new_tag = int(act.args.get("new", int(a.GetChiralTag())))
                        a.SetChiralTag(Chem.ChiralType(new_tag))
                    except Exception:
                        pass
        Chem.SanitizeMol(rw)
    except Exception:
        failed = True
    return rw, failed


def rw_mol_from_mapped_reactants(mapped_rxn: str) -> Chem.RWMol:
    r_smi = mapped_rxn.split(">>")[0]
    m = Chem.MolFromSmiles(r_smi)
    return Chem.RWMol(m) if m is not None else Chem.RWMol()
