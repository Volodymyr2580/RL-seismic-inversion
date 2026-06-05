from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _natural_key(path: Path):
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else path.stem


def ensure_shot_receiver_time(seismic: np.ndarray) -> np.ndarray:
    """Return seismic data as [shot, receiver, time].

    CurveVel_A files are stored as [shot, time, receiver] in this workspace.
    The Phase II CNN expects [shot, receiver, time], so we transpose once at
    the data boundary instead of making every caller remember this convention.
    """
    if seismic.ndim != 3:
        raise ValueError(f"seismic 必须是 3D，得到 shape={seismic.shape}")
    if seismic.shape[1] > seismic.shape[2]:
        seismic = np.transpose(seismic, (0, 2, 1))
    return np.ascontiguousarray(seismic)


class CurveVelADataset:
    def __init__(self, root: str):
        self.root = Path(root)
        self.data_dir = self.root / "data"
        self.model_dir = self.root / "model"
        self.data_files = sorted(self.data_dir.glob("data*.npy"), key=_natural_key)
        self.model_files = sorted(self.model_dir.glob("model*.npy"), key=_natural_key)
        if len(self.data_files) == 0 or len(self.model_files) == 0:
            raise FileNotFoundError(f"未找到 data/model npy 文件: {self.root}")
        if len(self.data_files) != len(self.model_files):
            raise ValueError("data/model 文件数量不一致")

        d0 = np.load(self.data_files[0], mmap_mode="r")
        m0 = np.load(self.model_files[0], mmap_mode="r")
        if d0.shape[0] != m0.shape[0]:
            raise ValueError("data/model 文件样本数不一致")
        self.samples_per_file = int(d0.shape[0])
        self.total_samples = self.samples_per_file * len(self.data_files)

    def get_split_file_ranges(self, val_files: int = 4):
        total_files = len(self.data_files)
        official_train_files = min(48, total_files)
        official_test_start = official_train_files

        if official_train_files <= 1:
            train_range = list(range(0, official_train_files))
            val_range = []
        else:
            val_files = max(1, min(val_files, official_train_files - 1))
            train_range = list(range(0, official_train_files - val_files))
            val_range = list(range(official_train_files - val_files, official_train_files))

        test_range = list(range(official_test_start, total_files))
        return {"train": train_range, "val": val_range, "test": test_range}

    def build_index_plan(
        self,
        split: str = "train",
        max_files: Optional[int] = None,
        max_samples_per_file: Optional[int] = None,
        val_files: int = 4,
    ):
        split_ranges = self.get_split_file_ranges(val_files=val_files)
        if split not in split_ranges:
            raise ValueError(f"不支持的 split: {split}")

        file_range = split_ranges[split]
        if max_files is not None:
            file_range = file_range[:max_files]

        samples_per_file = self.samples_per_file if max_samples_per_file is None else min(self.samples_per_file, max_samples_per_file)
        plan: list[tuple[int, int]] = []
        for file_idx in file_range:
            for sample_idx in range(samples_per_file):
                plan.append((file_idx, sample_idx))
        return plan

    def get_by_file_index(self, file_idx: int, sample_idx: int):
        data_path = self.data_files[file_idx]
        model_path = self.model_files[file_idx]
        d = np.load(data_path, mmap_mode="r")
        m = np.load(model_path, mmap_mode="r")
        seismic = np.asarray(d[sample_idx]).astype(np.float32, copy=False)
        seismic = ensure_shot_receiver_time(seismic)
        max_val = np.max(np.abs(seismic))
        if max_val > 0:
            seismic = seismic / max_val

        model = np.asarray(m[sample_idx]).astype(np.float32, copy=False)
        if model.ndim == 3 and model.shape[0] == 1:
            model = model[0]
        return {"seismic": seismic, "model": model, "meta": {"data_file": data_path.name, "model_file": model_path.name, "file_idx": int(file_idx), "sample_idx": int(sample_idx)}}


class TorchCurveVelAPlanDataset(Dataset):
    def __init__(self, base: CurveVelADataset, plan: list[tuple[int, int]]):
        self.base = base
        self.plan = plan

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, idx: int):
        file_idx, sample_idx = self.plan[idx]
        item = self.base.get_by_file_index(file_idx, sample_idx)
        seismic = torch.from_numpy(np.array(item["seismic"], copy=True)).float()
        model = torch.from_numpy(np.array(item["model"], copy=True)).float()
        return {"seismic": seismic, "model": model, "meta": item["meta"]}


def make_dataloader(
    root: str,
    split: str,
    batch_size: int,
    max_files: Optional[int] = None,
    max_samples_per_file: Optional[int] = None,
    fixed_index: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
):
    base = CurveVelADataset(root)
    plan = base.build_index_plan(split=split, max_files=max_files, max_samples_per_file=max_samples_per_file)
    if fixed_index is not None:
        fixed_index = int(fixed_index)
        plan = [plan[fixed_index]]
        shuffle = False
    ds = TorchCurveVelAPlanDataset(base, plan)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=shuffle, num_workers=int(num_workers), pin_memory=True)
    return dl

