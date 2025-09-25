from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List

from rdkit import Chem


@dataclass
class StandardizedReaction:
    reaction_id: str
    reactants_smiles: str
    products_smiles: str
    reactants_canonical: str
    products_canonical: str


def reconstruct_smiles_from_tokenized(line: str) -> str:
    """Convert a tokenized SMILES line back to a standard SMILES string.

    The dataset uses space-separated tokens and a '.' token between components.
    Reconstruct by removing spaces.
    """
    return line.strip().replace(" ", "")


def sanitize_mol(mol: Chem.Mol, kekulize: bool = False) -> Chem.Mol:
    if mol is None:
        raise ValueError("MolFromSmiles failed")
    try:
        Chem.SanitizeMol(mol)
        if kekulize:
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Exception:
                # If kekulization fails, keep aromatic form
                pass
    except Exception as e:
        raise ValueError(f"SanitizeMol failed: {e}")
    return mol


def canonical_smiles(mol: Chem.Mol, kekule: bool = False) -> str:
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=kekule)


def drop_small_fragments(smiles: str, min_heavy_atoms: int = 2) -> str:
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return smiles
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    kept = []
    for frag in frags:
        try:
            Chem.SanitizeMol(frag)
        except Exception:
            pass
        if frag.GetNumHeavyAtoms() >= min_heavy_atoms:
            kept.append(Chem.MolToSmiles(frag, canonical=True))
    if not kept:
        return smiles
    kept.sort()
    return ".".join(kept)


def standardize_reaction(
    reactants_smiles: str,
    products_smiles: str,
    kekulize: bool = False,
    drop_small: bool = True,
    min_heavy_atoms: int = 2,
) -> StandardizedReaction:
    if drop_small:
        reactants_smiles = drop_small_fragments(reactants_smiles, min_heavy_atoms)
        products_smiles = drop_small_fragments(products_smiles, min_heavy_atoms)

    r_mol = Chem.MolFromSmiles(reactants_smiles)
    p_mol = Chem.MolFromSmiles(products_smiles)

    r_mol = sanitize_mol(r_mol, kekulize=kekulize)
    p_mol = sanitize_mol(p_mol, kekulize=kekulize)

    r_canon = canonical_smiles(r_mol, kekule=kekulize)
    p_canon = canonical_smiles(p_mol, kekule=kekulize)

    return StandardizedReaction(
        reaction_id=str(uuid.uuid4()),
        reactants_smiles=reactants_smiles,
        products_smiles=products_smiles,
        reactants_canonical=r_canon,
        products_canonical=p_canon,
    )
