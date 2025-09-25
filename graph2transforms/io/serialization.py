from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional

import ujson


class JsonlShardWriter:
    def __init__(self, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.out_path = out_path
        self._f = open(out_path, "w", encoding="utf-8")
        self.count = 0

    def write(self, obj: Dict) -> None:
        self._f.write(ujson.dumps(obj) + "\n")
        self.count += 1

    def close(self) -> None:
        self._f.close()


def write_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
