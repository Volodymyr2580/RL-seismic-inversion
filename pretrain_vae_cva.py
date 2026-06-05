"""
Phase III: VAE pretraining on CVA velocity models.

Trains a VelocityVAE to learn a 64-dimensional latent manifold of velocity models.
The trained decoder replaces B-spline interpolation in the RL pipeline.

Usage:
    # Quick smoke test
    python pretrain_vae_cva.py --epochs 5 --device cuda:0

    # Full training
    python pretrain_vae_cva.py --epochs 200 --batch_size 64 --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.vae import VelocityVAE

# ---------------------------------------------------------------------------
# Dataset: velocity models only (no seismic needed for VAE)
# ---------------------------------------------------------------------------


class VelocityModelDataset(Dataset):
    """Load velocity models from one or more data directories."""

    def __init__(
        self,
        roots: list[str],
        file_range: tuple[int, int] = (0, 40),
        max_samples_per_file: int | None = None,
        v_min: float = 1500.0,
        v_max: float = 4500.0,
    ):
        import glob
        import re

        self.v_min = v_min
        self.v_max = v_max

        self.model_files = []
        for root in roots:
            # Try root/model first (CVA convention), then root directly (FVA convention)
            model_dir = os.path.join(root, "model")
            if os.path.isdir(model_dir):
                search_dir = model_dir
            else:
                search_dir = root
            files = sorted(
                glob.glob(os.path.join(search_dir, "model*.npy")),
                key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)),
            )
            n = min(len(files), file_range[1])
            self.model_files.extend(files[:n])
            print(f"  {root}: {n} files")

        if len(self.model_files) == 0:
            raise FileNotFoundError(f"No model*.npy found in {roots}")

        sample = np.load(self.model_files[0], mmap_mode="r")
        self.samples_per_file = sample.shape[0]
        if max_samples_per_file is not None:
            self.samples_per_file = min(self.samples_per_file, max_samples_per_file)

        self._plan = []
        for file_idx in range(len(self.model_files)):
            for sample_idx in range(self.samples_per_file):
                self._plan.append((file_idx, sample_idx))

        # Verify value range
        m0 = np.load(self.model_files[0], mmap_mode="r")
        actual_min, actual_max = float(m0.min()), float(m0.max())
        print(f"  Data range: [{actual_min:.0f}, {actual_max:.0f}] m/s "
              f"(normalizing to [{self.v_min:.0f}, {self.v_max:.0f}])")

    def __len__(self):
        return len(self._plan)

    def __getitem__(self, idx):
        file_idx, sample_idx = self._plan[idx]
        model = np.load(self.model_files[file_idx], mmap_mode="r")
        v = np.asarray(model[sample_idx], dtype=np.float32).copy()

        # Normalize to [0, 1] range — VAE works better on normalized data
        v_norm = (v - self.v_min) / (self.v_max - self.v_min)

        if v_norm.ndim == 3 and v_norm.shape[0] == 1:
            v_norm = v_norm[0]  # [70, 70]
        elif v_norm.ndim == 2:
            pass
        else:
            v_norm = v_norm.squeeze()

        return torch.from_numpy(v_norm).float().unsqueeze(0)  # [1, 70, 70]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class VAETrainConfig:
    latent_dim: int = 64
    native_size: int = 70
    internal_size: int = 64

    # Data
    data_root: str = "data/CVA/CurveVel_A"
    val_file_start: int = 40
    val_file_end: int = 44  # files 40-43 for validation

    # Training
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    beta: float = 1.0  # KL weight (β-VAE)
    beta_warmup_epochs: int = 10  # linear warmup from 0 to beta

    # Logging
    log_every: int = 5
    save_dir: str = "runs/vae_pretrain"
    device: str = "cuda:0"
    seed: int = 42


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Pretrain VelocityVAE on CVA")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="runs/vae_pretrain")
    parser.add_argument("--max_files", type=int, default=None, help="Limit train files for quick test")
    parser.add_argument("--extra_data", type=str, default=None, help="Additional data dirs (comma-separated)")
    parser.add_argument("--data_root", type=str, default=None, help="Override default data root")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = VAETrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        latent_dim=args.latent_dim,
        device=args.device,
        save_dir=args.save_dir,
        seed=args.seed,
    )

    set_seed(cfg.seed)

    # Resolve data roots
    if args.data_root:
        data_roots = [args.data_root]
    else:
        data_roots = [cfg.data_root]
    if not os.path.isdir(data_roots[0]):
        data_roots[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_roots[0])
    if not os.path.isdir(data_roots[0]):
        alt = "/data/shengwz/swz/RL-seismic-inversion/data/CVA/CurveVel_A"
        if os.path.isdir(alt):
            data_roots[0] = alt
    if args.extra_data:
        for extra in args.extra_data.split(","):
            extra = extra.strip()
            if not os.path.isdir(extra):
                extra = os.path.join(os.path.dirname(os.path.abspath(__file__)), extra)
            if os.path.isdir(extra):
                data_roots.append(extra)
                print(f"  Extra data: {extra}")
    print(f"Data roots: {data_roots}")

    # Datasets — train on all data, val only on primary root
    train_files = min(40, cfg.val_file_start) if args.max_files is None else args.max_files
    train_ds = VelocityModelDataset(
        data_roots,
        file_range=(0, train_files),
        v_min=1500.0, v_max=4500.0,
    )
    val_ds = VelocityModelDataset(
        [data_roots[0]],
        file_range=(cfg.val_file_start, cfg.val_file_end),
        v_min=1500.0, v_max=4500.0,
    )

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )

    # Model
    vae = VelocityVAE(
        latent_dim=cfg.latent_dim,
        native_size=cfg.native_size,
        internal_size=cfg.internal_size,
    ).to(cfg.device)

    total_params = sum(p.numel() for p in vae.parameters())
    trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"VAE params: {total_params:,} total, {trainable_params:,} trainable")

    optimizer = torch.optim.AdamW(
        vae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01,
    )

    # Save directory
    os.makedirs(cfg.save_dir, exist_ok=True)
    best_val_mae = float("inf")

    print(f"\n{'='*60}")
    print(f"Training VAE: latent_dim={cfg.latent_dim}, β={cfg.beta}")
    print(f"GPU: {cfg.device}, epochs={cfg.epochs}, batch={cfg.batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # Linear β warmup
        if epoch <= cfg.beta_warmup_epochs and cfg.beta_warmup_epochs > 0:
            current_beta = cfg.beta * (epoch / cfg.beta_warmup_epochs)
        else:
            current_beta = cfg.beta

        # ---- Train ----
        vae.train()
        train_loss_sum = 0.0
        train_mse_sum = 0.0
        train_kl_sum = 0.0
        train_mae_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch.to(cfg.device, non_blocking=True)  # [B, 1, 70, 70]
            optimizer.zero_grad()

            recon, mu, logvar = vae(x)
            loss, metrics = VelocityVAE.loss_function(recon, x, mu, logvar, beta=current_beta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += metrics['loss']
            train_mse_sum += metrics['mse']
            train_kl_sum += metrics['kl']
            train_mae_sum += metrics['mae']
            n_batches += 1

        scheduler.step()

        # ---- Validation ----
        vae.eval()
        val_mse_sum = 0.0
        val_mae_sum = 0.0
        val_kl_sum = 0.0
        val_n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(cfg.device, non_blocking=True)
                recon, mu, logvar = vae(x)
                _, metrics = VelocityVAE.loss_function(recon, x, mu, logvar, beta=current_beta)
                val_mse_sum += metrics['mse']
                val_mae_sum += metrics['mae']
                val_kl_sum += metrics['kl']
                val_n_batches += 1

        elapsed = time.time() - t0

        train_mae_unscaled = (train_mae_sum / n_batches) * (train_ds.v_max - train_ds.v_min)
        val_mae_unscaled = (val_mae_sum / val_n_batches) * (train_ds.v_max - train_ds.v_min)

        if epoch % cfg.log_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{cfg.epochs} | β={current_beta:.2f} | "
                f"Train MSE={train_mse_sum/n_batches:.4f} MAE={train_mae_unscaled:.1f} m/s KL={train_kl_sum/n_batches:.1f} | "
                f"Val MAE={val_mae_unscaled:.1f} m/s KL={val_kl_sum/val_n_batches:.1f} | "
                f"{elapsed:.1f}s"
            )

        # Save best model
        val_mae_current = val_mae_sum / val_n_batches
        if val_mae_current < best_val_mae:
            best_val_mae = val_mae_current
            checkpoint_path = os.path.join(cfg.save_dir, "vae_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": vae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mae_norm": best_val_mae,
                    "val_mae_physical": val_mae_unscaled,
                    "cfg": {
                        "latent_dim": cfg.latent_dim,
                        "native_size": cfg.native_size,
                        "internal_size": cfg.internal_size,
                        "v_min": train_ds.v_min,
                        "v_max": train_ds.v_max,
                    },
                },
                checkpoint_path,
            )
            print(f"  → Saved best model (val MAE={val_mae_unscaled:.1f} m/s)")

    # Final save
    final_path = os.path.join(cfg.save_dir, "vae_final.pt")
    torch.save(
        {
            "epoch": cfg.epochs,
            "model_state_dict": vae.state_dict(),
            "cfg": {
                "latent_dim": cfg.latent_dim,
                "native_size": cfg.native_size,
                "internal_size": cfg.internal_size,
                "v_min": train_ds.v_min,
                "v_max": train_ds.v_max,
            },
        },
        final_path,
    )

    print(f"\n{'='*60}")
    print(f"Training complete. Best val MAE: {best_val_mae * (train_ds.v_max - train_ds.v_min):.1f} m/s")
    print(f"Models saved to {cfg.save_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
