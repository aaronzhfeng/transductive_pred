from __future__ import annotations

from typing import Dict, List, Tuple

from rdkit import Chem

from graph2transforms.edits.actions import BondOrder, BondStereo


HYBRIDIZATION_MAP = {
    int(Chem.rdchem.HybridizationType.UNSPECIFIED): 0,
    int(Chem.rdchem.HybridizationType.SP): 1,
    int(Chem.rdchem.HybridizationType.SP2): 2,
    int(Chem.rdchem.HybridizationType.SP3): 3,
    int(Chem.rdchem.HybridizationType.SP3D): 4,
    int(Chem.rdchem.HybridizationType.SP3D2): 5,
}

CHIRAL_TAG_MAP = {
    int(Chem.rdchem.ChiralType.CHI_UNSPECIFIED): 0,
    int(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW): 1,
    int(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW): 2,
    int(Chem.rdchem.ChiralType.CHI_OTHER): 3,
}


def bond_order_from_rdkit(b: Chem.Bond) -> BondOrder:
    bt = b.GetBondType()
    if b.GetIsAromatic() or bt == Chem.BondType.AROMATIC:
        return BondOrder.AROMATIC
    if bt == Chem.BondType.SINGLE:
        return BondOrder.SINGLE
    if bt == Chem.BondType.DOUBLE:
        return BondOrder.DOUBLE
    if bt == Chem.BondType.TRIPLE:
        return BondOrder.TRIPLE
    return BondOrder.SINGLE


def bond_stereo_from_rdkit(b: Chem.Bond) -> BondStereo:
    st = int(b.GetStereo())
    if st == int(Chem.BondStereo.STEREOZ):
        return BondStereo.Z
    if st == int(Chem.BondStereo.STEREOE):
        return BondStereo.E
    return BondStereo.NONE


def atom_features(mol: Chem.Mol) -> List[Dict]:
    feats: List[Dict] = []
    for a in mol.GetAtoms():
        feats.append(
            {
                "z": a.GetAtomicNum(),
                "degree": a.GetDegree(),
                "total_valence": a.GetTotalValence(),
                "formal_charge": a.GetFormalCharge(),
                "hybridization": HYBRIDIZATION_MAP.get(int(a.GetHybridization()), 0),
                "aromatic": 1 if a.GetIsAromatic() else 0,
                "implicit_h": a.GetImplicitValence(),
                "explicit_h": a.GetNumExplicitHs(),
                "chiral_tag": CHIRAL_TAG_MAP.get(int(a.GetChiralTag()), 0),
                "map_num": a.GetAtomMapNum(),
            }
        )
    return feats


def edge_list_with_features(mol: Chem.Mol) -> List[Dict]:
    edges: List[Dict] = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        order = bond_order_from_rdkit(b).value
        stereo = bond_stereo_from_rdkit(b).value
        ring = 1 if b.IsInRing() else 0
        conj = 1 if b.GetIsConjugated() else 0
        edges.append({"src": i, "dst": j, "order": order, "stereo": stereo, "ring": ring, "conj": conj})
        edges.append({"src": j, "dst": i, "order": order, "stereo": stereo, "ring": ring, "conj": conj})
    return edges


def serialize_graph(mol: Chem.Mol) -> Dict:
    return {
        "num_nodes": mol.GetNumAtoms(),
        "nodes": atom_features(mol),
        "edges": edge_list_with_features(mol),
    }
