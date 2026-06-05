from __future__ import annotations

import csv
import os
from dataclasses import asdict, is_dataclass
from typing import Any


class CSVLogger:
    def __init__(self, out_dir: str, filename: str = "metrics.csv"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, filename)
        self._fp = open(self.path, "a", newline="", encoding="utf-8")
        self._writer: csv.DictWriter | None = None
        self._fieldnames: list[str] | None = None

    def log(self, row: dict[str, Any]):
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._fp, fieldnames=self._fieldnames)
            if self._fp.tell() == 0:
                self._writer.writeheader()
        self._writer.writerow(row)
        self._fp.flush()

    def close(self):
        if self._fp is not None:
            self._fp.close()


def dump_config(out_dir: str, cfg: Any, filename: str = "config.txt"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    if is_dataclass(cfg):
        d = asdict(cfg)
    elif isinstance(cfg, dict):
        d = dict(cfg)
    else:
        d = {"value": str(cfg)}
    with open(path, "w", encoding="utf-8") as f:
        for k in sorted(d.keys()):
            f.write(f"{k}={d[k]}\n")

