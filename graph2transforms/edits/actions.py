from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Tuple


class ActionType(str, Enum):
    DEL_BOND = "DEL_BOND"
    CHANGE_BOND = "CHANGE_BOND"
    ADD_BOND = "ADD_BOND"
    H_LOSS = "H_LOSS"
    H_GAIN = "H_GAIN"
    CHANGE_ATOM = "CHANGE_ATOM"
    DELETE_SUBGRAPH = "DELETE_SUBGRAPH"
    REMOVE_LG = "REMOVE_LG"
    STOP = "STOP"


class BondOrder(str, Enum):
    SINGLE = "SINGLE"
    DOUBLE = "DOUBLE"
    TRIPLE = "TRIPLE"
    AROMATIC = "AROMATIC"


class BondStereo(str, Enum):
    NONE = "NONE"
    E = "E"
    Z = "Z"


@dataclass
class Action:
    type: ActionType
    args: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type.value
        args = {}
        for k, v in self.args.items():
            if isinstance(v, Enum):
                args[k] = v.value
            else:
                args[k] = v
        data["args"] = args
        return data


def canonical_sort_actions(actions: List[Action]) -> List[Action]:
    order_rank = {
        ActionType.DEL_BOND: 0,
        ActionType.CHANGE_BOND: 1,
        ActionType.ADD_BOND: 2,
        ActionType.H_LOSS: 3,
        ActionType.H_GAIN: 4,
        ActionType.CHANGE_ATOM: 5,
        ActionType.REMOVE_LG: 6,
        ActionType.DELETE_SUBGRAPH: 6,
        ActionType.STOP: 7,
    }

    def key_fn(a: Action) -> Tuple:
        rank = order_rank.get(a.type, 99)
        i = a.args.get("i", -1)
        j = a.args.get("j", -1)
        first = min(i, j) if (i != -1 and j != -1) else i
        second = max(i, j) if (i != -1 and j != -1) else j
        bond_order = a.args.get("order") or a.args.get("new_order") or ""
        stereo = a.args.get("stereo") or a.args.get("new_stereo") or ""
        return (rank, first, second, str(bond_order), str(stereo))

    return sorted(actions, key=key_fn)
