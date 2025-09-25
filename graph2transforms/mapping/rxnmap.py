from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from rdkit import Chem
from rxnmapper import RXNMapper

from graph2transforms.chem.standardize import StandardizedReaction


@dataclass
class MappedReaction:
    reaction_id: str
    mapped_rxn: str
    confidence: float
    r_mol: Chem.Mol
    p_mol: Chem.Mol
    rmap: Dict[int, int]  # map_id -> atom_idx
    pmap: Dict[int, int]


def _build_map_dict(mol: Chem.Mol) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for atom in mol.GetAtoms():
        mid = atom.GetAtomMapNum()
        if mid > 0:
            if mid in mapping:
                raise ValueError("Duplicate map id in molecule")
            mapping[mid] = atom.GetIdx()
    return mapping


def ensure_atom_mapping(sr: StandardizedReaction, threshold: float = 0.85) -> MappedReaction:
    rxn_str = f"{sr.reactants_canonical}>>{sr.products_canonical}"
    mapper = RXNMapper()
    results = mapper.get_attention_guided_atom_maps([rxn_str])
    if not results:
        raise ValueError("RXNMapper returned no result")
    result = results[0]
    mapped_rxn = result["mapped_rxn"]
    confidence = float(result.get("confidence", 0.0))
    if confidence < threshold:
        raise ValueError(f"Mapping confidence below threshold: {confidence:.3f} < {threshold}")

    try:
        r_smi, p_smi = mapped_rxn.split(">>")
    except Exception as e:
        raise ValueError(f"Bad mapped reaction string: {mapped_rxn}")

    r_mol = Chem.MolFromSmiles(r_smi)
    p_mol = Chem.MolFromSmiles(p_smi)
    if r_mol is None or p_mol is None:
        raise ValueError("MolFromSmiles failed for mapped reaction")

    rmap = _build_map_dict(r_mol)
    pmap = _build_map_dict(p_mol)

    # Validate: every product atom mapping must exist in reactants
    if set(pmap.keys()) - set(rmap.keys()):
        raise ValueError("Product has map ids not present in reactants")

    return MappedReaction(
        reaction_id=sr.reaction_id,
        mapped_rxn=mapped_rxn,
        confidence=confidence,
        r_mol=r_mol,
        p_mol=p_mol,
        rmap=rmap,
        pmap=pmap,
    )
