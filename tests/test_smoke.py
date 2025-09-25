from __future__ import annotations

from graph2transforms.chem.standardize import reconstruct_smiles_from_tokenized, standardize_reaction
from graph2transforms.edits.actions import Action, ActionType, canonical_sort_actions


def test_reconstruct_smiles():
    line = "C C . O"
    assert reconstruct_smiles_from_tokenized(line) == "CC.O"


def test_standardize_reaction():
    sr = standardize_reaction("CC.O", "CCO")
    assert sr.reactants_canonical
    assert sr.products_canonical


def test_sort_actions():
    a1 = Action(ActionType.ADD_BOND, {"i": 2, "j": 3})
    a2 = Action(ActionType.DEL_BOND, {"i": 1, "j": 2})
    a3 = Action(ActionType.STOP, {})
    out = canonical_sort_actions([a1, a3, a2])
    assert out[0].type == ActionType.DEL_BOND
    assert out[-1].type == ActionType.STOP
