from __future__ import annotations

from typing import Dict, List, Set, Tuple

from rdkit import Chem

from graph2transforms.edits.actions import ActionType


VALENCE_LIMITS = {1: 1, 5: 5, 6: 4, 7: 3, 8: 2, 9: 1, 15: 5, 16: 6, 17: 1, 35: 1, 53: 1}


def compute_masks(mol: Chem.Mol) -> Dict[str, List[int]]:
    n = mol.GetNumAtoms()
    # Action-type mask: all enabled by default; refined later
    action_mask = [1] * len(ActionType)
    # Disallow self-bonds through pointer mask would be handled in model code
    return {"action": action_mask}
