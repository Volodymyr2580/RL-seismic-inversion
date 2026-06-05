"""
CVA CNN pretraining for Phase II.

Trains a CNN to map seismic → B-spline control points → 70×70 velocity model.
Loss: MSE + (1 - SSIM) on the full velocity field.

The pretrained CNN provides μ initialization for LearnableBetaMeanPolicy in RL phase.

Pipeline:
    1. p_data → CNN → μ_raw → sigmoid → μ [0,1] → scale to [v_min, v_max]
    2. v_ctrl_pred → B-spline → v_pred [70, 70]
    3. Loss = MSE(v_pred, v_true) + λ * (1 - SSIM(v_pred, v_true))
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.cnn_encoder import SeismicCNNEncoder
from agents.Bspline import bspline2d_inverse, bspline2d_prolong
from agents.velocity_reconstructor import VelocityReconstructor


# ---------------------------------------------------------------------------
#  SSIM (Structural Similarity) for velocity models
# ---------------------------------------------------------------------------

def ssim_2d(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01,
    C2: float = 0.03,
) -> torch.Tensor:
    """Compute SSIM between two 2D images. Returns per-batch mean SSIM in [0, 1]."""
    if img1.ndim == 3:
        # [B, H, W] — compute per-sample
        ssim_vals = []
        for b in range(img1.shape[0]):
            ssim_vals.append(ssim_2d(img1[b], img2[b], window_size, C1, C2))
        return torch.stack(ssim_vals).mean()

    # Single image [H, W]
    img1 = img1.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    img2 = img2.unsqueeze(0).unsqueeze(0)

    # Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device)
    coords -= window_size // 2
    gauss = torch.exp(-coords**2 / (2 * sigma**2))
    gauss /= gauss.sum()
    window_1d = gauss.unsqueeze(0) * gauss.unsqueeze(1)  # [window, window]
    window = window_1d.unsqueeze(0).unsqueeze(0)  # [1, 1, window, window]

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


# ---------------------------------------------------------------------------
#  CNN prediction head: features → velocity control points
# ---------------------------------------------------------------------------

class CNNVelocityPredictor(nn.Module):
    """CNN encoder + prediction head → velocity control points.

    Architecture:
        SeismicCNNEncoder: p_data → features [B, embed_dim, nx_ctrl, nz_ctrl]
        PredHead: 1×1 Conv → 1 channel → sigmoid → scale to physical velocity
    """

    def __init__(
        self,
        in_channels: int = 5,
        embed_dim: int = 128,
        nx_ctrl: int = 4,
        nz_ctrl: int = 4,
        v_min: float = 1500.0,
        v_max: float = 4500.0,
    ):
        super().__init__()
        self.encoder = SeismicCNNEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            nx_ctrl=nx_ctrl,
            nz_ctrl=nz_ctrl,
        )
        self.pred_head = nn.Conv2d(embed_dim, 1, kernel_size=1)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict velocity control points from seismic data.

        Args:
            x: [B, N_s, N_r, N_t] seismic shot gathers.

        Returns:
            v_ctrl: [B, nx_ctrl, nz_ctrl] predicted velocity control points.
            v_model: [B, nx_model, nz_model] reconstructed velocity model.
        """
        features = self.encoder(x)               # [B, embed_dim, Hc, Wc]
        raw = self.pred_head(features).squeeze(1)  # [B, Hc, Wc]
        # Map to [0, 1] → scale to velocity
        mu = torch.sigmoid(raw)                   # [B, Hc, Wc] in (0, 1)
        v_ctrl = self.v_min + (self.v_max - self.v_min) * mu

        # B-spline reconstruct to full velocity model
        v_model = bspline2d_prolong(
            v_ctrl, (70, 70)
        )  # [B, 70, 70]

        return v_ctrl, v_model


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------

@dataclass
class PretrainConfig:
    cva_root: str = "data/CVA/CurveVel_A"
    train_files: int = 40
    val_files: int = 4
    samples_per_file: int = 64

    in_channels: int = 5
    embed_dim: int = 128
    nx_ctrl: int = 4
    nz_ctrl: int = 4
    v_min: float = 1500.0
    v_max: float = 4500.0

    batch_size: int = 8
    epochs: int = 200      # max epochs (early stopping will likely stop earlier)
    lr: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 20    # early stopping patience
    target_mode: str = "control"  # "control" uses bspline inverse labels; "field" keeps legacy loss
    inverse_lam: float = 1e-3
    inverse_maxit: int = 100
    ssim_weight: float = 0.0      # optional weight for (1-SSIM) term
    mse_weight: float = 1.0       # weight for MSE term

    out_dir: str = "runs/cva_pretrain_phase2"
    save_every: int = 10
    device: str = "cpu"
    seed: int = 42


def load_cva_pairs(
    root: str,
    file_indices: list[int],
    samples_per_file: int,
    *,
    compute_control_labels: bool = True,
    nx_ctrl: int = 4,
    nz_ctrl: int = 4,
    v_min: float = 1500.0,
    v_max: float = 4500.0,
    inverse_lam: float = 1e-3,
    inverse_maxit: int = 100,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    """Load (seismic, velocity_model) pairs from CVA.

    Returns:
        list of (seismic [N_s, N_r, N_t], velocity [70, 70], control [Hc, Wc]) triples.
    """
    from utils.data_loader import CurveVelADataset
    from tqdm import tqdm

    cva = CurveVelADataset(root)
    data = []
    total = len(file_indices) * samples_per_file
    pbar = tqdm(total=total, desc="  Loading", unit="samples")
    for file_idx in file_indices:
        max_s = min(samples_per_file, cva.samples_per_file)
        for sample_idx in range(max_s):
            item = cva.get_by_file_index(file_idx, sample_idx)
            seismic = item["seismic"].astype(np.float32)
            velocity = item["model"].astype(np.float32)

            if velocity.shape != (70, 70):
                pbar.update(1)
                continue

            # Normalize seismic
            max_val = np.max(np.abs(seismic))
            if max_val > 0:
                seismic = seismic / max_val

            control = None
            if compute_control_labels:
                with torch.no_grad():
                    control_t = bspline2d_inverse(
                        torch.from_numpy(velocity.copy()).float(),
                        (int(nx_ctrl), int(nz_ctrl)),
                        lam=float(inverse_lam),
                        maxit=int(inverse_maxit),
                    )
                control = control_t.cpu().numpy().astype(np.float32, copy=False)
                control = np.clip(control, float(v_min), float(v_max))

            data.append((seismic, velocity, control))
            pbar.update(1)
    pbar.close()
    return data


class SeismicVelocityDataset(torch.utils.data.Dataset):
    def __init__(self, data: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seis, vel, ctrl = self.data[idx]
        ctrl_t = torch.empty(0) if ctrl is None else torch.from_numpy(ctrl.copy()).float()
        return (
            torch.from_numpy(seis.copy()).float(),
            torch.from_numpy(vel.copy()).float(),
            ctrl_t,
        )


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def pretrain(config: PretrainConfig):
    from tqdm import tqdm
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    os.makedirs(config.out_dir, exist_ok=True)

    # ---- Data ----
    print(f"Loading CVA: files 0-{config.train_files-1} (train), "
          f"{config.train_files}-{config.train_files+config.val_files-1} (val)")
    train_pairs = load_cva_pairs(
        config.cva_root,
        list(range(config.train_files)),
        config.samples_per_file,
        compute_control_labels=(config.target_mode == "control"),
        nx_ctrl=config.nx_ctrl,
        nz_ctrl=config.nz_ctrl,
        v_min=config.v_min,
        v_max=config.v_max,
        inverse_lam=config.inverse_lam,
        inverse_maxit=config.inverse_maxit,
    )
    val_pairs = load_cva_pairs(
        config.cva_root,
        list(range(config.train_files, config.train_files + config.val_files)),
        config.samples_per_file,
        compute_control_labels=(config.target_mode == "control"),
        nx_ctrl=config.nx_ctrl,
        nz_ctrl=config.nz_ctrl,
        v_min=config.v_min,
        v_max=config.v_max,
        inverse_lam=config.inverse_lam,
        inverse_maxit=config.inverse_maxit,
    )
    print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    train_ds = SeismicVelocityDataset(train_pairs)
    val_ds = SeismicVelocityDataset(val_pairs)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, pin_memory=True,
    )

    # ---- Model ----
    model = CNNVelocityPredictor(
        in_channels=config.in_channels,
        embed_dim=config.embed_dim,
        nx_ctrl=config.nx_ctrl,
        nz_ctrl=config.nz_ctrl,
        v_min=config.v_min,
        v_max=config.v_max,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs,
    )

    # ---- Training loop ----
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    metrics_path = os.path.join(config.out_dir, "pretrain_metrics.csv")
    with open(metrics_path, "w") as f:
        f.write("epoch,train_loss,train_mse,train_ssim,val_loss,val_mse,val_ssim,lr\n")

    for epoch in range(1, config.epochs + 1):
        # Train
        model.train()
        train_total = train_mse_sum = train_ssim_sum = 0.0
        n_train = 0

        train_pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{config.epochs} [train]", unit="b", leave=False)
        for seismic_b, vel_true_b, ctrl_true_b in train_pbar:
            seismic_b = seismic_b.to(device)
            vel_true_b = vel_true_b.to(device)
            ctrl_true_b = ctrl_true_b.to(device)

            v_ctrl_pred, v_pred = model(seismic_b)

            # Normalize velocities to [0, 1] for stable training
            v_range = model.v_max - model.v_min
            if config.target_mode == "control":
                if ctrl_true_b.numel() == 0:
                    raise RuntimeError("target_mode='control' requires B-spline inverse labels")
                v_ctrl_pred_norm = (v_ctrl_pred - model.v_min) / v_range
                ctrl_true_norm = (ctrl_true_b - model.v_min) / v_range
                loss_mse = F.mse_loss(v_ctrl_pred_norm, ctrl_true_norm)
            else:
                v_pred_norm = (v_pred - model.v_min) / v_range
                vel_true_norm = (vel_true_b - model.v_min) / v_range
                loss_mse = F.mse_loss(v_pred_norm, vel_true_norm)

            ssim_val = torch.tensor(0.0, device=device)
            loss_ssim = torch.tensor(0.0, device=device)
            if config.ssim_weight != 0.0:
                ssim_val = ssim_2d(v_pred, vel_true_b)
                loss_ssim = 1.0 - ssim_val

            loss = config.mse_weight * loss_mse + config.ssim_weight * loss_ssim

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = seismic_b.size(0)
            train_total += float(loss.item() * bs)
            train_mse_sum += float(loss_mse.item() * bs)
            train_ssim_sum += float(ssim_val.item() * bs)
            n_train += bs

        train_total /= n_train
        train_mse_avg = train_mse_sum / n_train
        train_ssim_avg = train_ssim_sum / n_train

        # Validate
        model.eval()
        val_total = val_mse_sum = val_ssim_sum = 0.0
        n_val = 0
        val_pbar = tqdm(val_loader, desc=f"  Epoch {epoch}/{config.epochs} [val]", unit="b", leave=False)
        with torch.no_grad():
            for seismic_b, vel_true_b, ctrl_true_b in val_pbar:
                seismic_b = seismic_b.to(device)
                vel_true_b = vel_true_b.to(device)
                ctrl_true_b = ctrl_true_b.to(device)
                v_ctrl_pred, v_pred = model(seismic_b)

                v_range = model.v_max - model.v_min
                if config.target_mode == "control":
                    if ctrl_true_b.numel() == 0:
                        raise RuntimeError("target_mode='control' requires B-spline inverse labels")
                    v_ctrl_pred_norm = (v_ctrl_pred - model.v_min) / v_range
                    ctrl_true_norm = (ctrl_true_b - model.v_min) / v_range
                    loss_mse = F.mse_loss(v_ctrl_pred_norm, ctrl_true_norm)
                else:
                    v_pred_norm = (v_pred - model.v_min) / v_range
                    vel_true_norm = (vel_true_b - model.v_min) / v_range
                    loss_mse = F.mse_loss(v_pred_norm, vel_true_norm)
                ssim_val = torch.tensor(0.0, device=device)
                loss_ssim = torch.tensor(0.0, device=device)
                if config.ssim_weight != 0.0:
                    ssim_val = ssim_2d(v_pred, vel_true_b)
                    loss_ssim = 1.0 - ssim_val
                loss = config.mse_weight * loss_mse + config.ssim_weight * loss_ssim

                bs = seismic_b.size(0)
                val_total += float(loss.item() * bs)
                val_mse_sum += float(loss_mse.item() * bs)
                val_ssim_sum += float(ssim_val.item() * bs)
                n_val += bs

        val_total /= n_val
        val_mse_avg = val_mse_sum / n_val
        val_ssim_avg = val_ssim_sum / n_val

        scheduler.step()

        print(f"  epoch {epoch:3d}/{config.epochs} | "
              f"train loss={train_total:.4f} mse={train_mse_avg:.2f} ssim={train_ssim_avg:.4f} | "
              f"val loss={val_total:.4f} mse={val_mse_avg:.2f} ssim={val_ssim_avg:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{train_total:.6f},{train_mse_avg:.6f},{train_ssim_avg:.6f},"
                    f"{val_total:.6f},{val_mse_avg:.6f},{val_ssim_avg:.6f},"
                    f"{scheduler.get_last_lr()[0]:.2e}\n")

        if val_total < best_val_loss:
            best_val_loss = val_total
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config.out_dir, "model_best.pt"))
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch} (best val={best_val_loss:.4f} @ epoch {best_epoch})")
            break

        if epoch % config.save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(config.out_dir, f"model_epoch_{epoch:03d}.pt"))

    torch.save(model.state_dict(), os.path.join(config.out_dir, "model_final.pt"))
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {config.out_dir}/model_best.pt")


def main():
    parser = argparse.ArgumentParser(description="CVA CNN pretraining for Phase II")
    parser.add_argument("--cva_root", type=str, default="data/CVA/CurveVel_A")
    parser.add_argument("--train_files", type=int, default=40)
    parser.add_argument("--val_files", type=int, default=4)
    parser.add_argument("--samples_per_file", type=int, default=64)
    parser.add_argument("--in_channels", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--target_mode", choices=["control", "field"], default="control")
    parser.add_argument("--inverse_lam", type=float, default=1e-3)
    parser.add_argument("--inverse_maxit", type=int, default=100)
    parser.add_argument("--ssim_weight", type=float, default=0.0)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--out_dir", type=str, default="runs/cva_pretrain_phase2")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = PretrainConfig(**{k: v for k, v in vars(args).items()})
    pretrain(config)


if __name__ == "__main__":
    main()
