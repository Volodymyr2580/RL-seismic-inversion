import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Pick:
    folder: str
    file: str
    index: int


def _key_numeric(p: str) -> Tuple[int, str]:
    stem = Path(p).stem
    digits = "".join([c for c in stem if c.isdigit()])
    return (int(digits) if digits else 0, p)


def _load_npy_2d(path: str, index: int) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[index]
    if arr.ndim == 4:
        return arr[index, 0]
    raise ValueError(f"Unexpected shape: {path} -> {arr.shape}")


def _pick_two(folder: str, indices: Sequence[int]) -> List[Pick]:
    files = [str(p) for p in Path(folder).glob("model*.npy")]
    if not files:
        raise RuntimeError(f"No model*.npy found in {folder}")
    files = sorted(files, key=_key_numeric)
    fp = files[0]
    picks: List[Pick] = []
    for idx in indices:
        picks.append(Pick(folder=folder, file=os.path.basename(fp), index=int(idx)))
    return picks


def _safe_makedirs(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _viz_two(
    title: str,
    picks: Sequence[Pick],
    out_dir: str,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mats = []
    labels = []
    for p in picks:
        fp = str(Path(p.folder) / p.file)
        m = _load_npy_2d(fp, p.index)
        mats.append(m)
        labels.append(f"{Path(p.folder).name}/{p.file}:{p.index}")

    if vmin is None:
        vmin = float(min(m.min() for m in mats))
    if vmax is None:
        vmax = float(max(m.max() for m in mats))

    fig, ax = plt.subplots(1, len(mats), figsize=(5 * len(mats), 4), constrained_layout=True)
    if len(mats) == 1:
        ax = [ax]

    for i, (m, lab) in enumerate(zip(mats, labels)):
        im = ax[i].imshow(m, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[i].set_title(lab)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)

        one_path = os.path.join(out_dir, f"{Path(picks[i].folder).name}_{Path(picks[i].file).stem}_idx{picks[i].index:04d}.png")
        plt.figure(figsize=(4, 4))
        plt.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        plt.title(lab)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(one_path, dpi=180)
        plt.close()

    out_path = os.path.join(out_dir, f"{title}.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cvb_dir", type=str, default=os.path.join("data", "CVB_model"))
    parser.add_argument("--fva_dir", type=str, default=os.path.join("data", "FVA_model"))
    parser.add_argument("--indices", type=str, default="0,250")
    parser.add_argument("--out_dir", type=str, default=os.path.join("outputs", "dataset_structure_preview"))
    args = parser.parse_args()

    indices = [int(s.strip()) for s in args.indices.split(",") if s.strip()]
    if len(indices) != 2:
        raise ValueError("--indices must have exactly 2 integers, e.g. 0,250")

    out_dir = _safe_makedirs(args.out_dir)

    cvb_picks = _pick_two(args.cvb_dir, indices)
    fva_picks = _pick_two(args.fva_dir, indices)

    all_mats = []
    for p in cvb_picks + fva_picks:
        fp = str(Path(p.folder) / p.file)
        all_mats.append(_load_npy_2d(fp, p.index))

    vmin = float(min(m.min() for m in all_mats))
    vmax = float(max(m.max() for m in all_mats))

    _viz_two("CVB_two_samples", cvb_picks, out_dir, vmin=vmin, vmax=vmax)
    _viz_two("FVA_two_samples", fva_picks, out_dir, vmin=vmin, vmax=vmax)

    print(f"Saved to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()

