from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    data_root: str = r"e:\sci_research\RL-seismic-inversion\data\CVA\CurveVel_A"
    split: str = "train"
    max_files: int | None = 1
    max_samples_per_file: int | None = 1
    fixed_index: int | None = 0

    seed: int = 42
    device: str = "cuda"

    nx_model: int = 70
    nz_model: int = 70
    nx_ctrl: int = 4
    nz_ctrl: int = 4

    n_bins: int = 100
    v_min: float = 1500.0
    v_max: float = 4500.0

    group_size: int = 8
    batch_size: int = 1

    lr: float = 8e-4
    weight_decay: float = 0.0

    epsilon_low: float = 0.2
    epsilon_high: float = 0.27

    reward_mode: str = "fwi"
    lambda_mix: float = 0.5
    log_transform_k: float = 3.0
    log_transform_c: float = 0.0

    steps: int = 20
    warmup_steps: int = 0
    grad_clip_norm: float | None = 1.0

    save_dir: str = "runs/rl_inversion"
    save_every: int = 10

    deepwave_dx: float = 10.0
    deepwave_dt: float = 0.001
    deepwave_freq: float = 15.0
    deepwave_nbc: int = 120
    deepwave_nt: int = 1000
    deepwave_n_shots: int = 5
    deepwave_n_receivers: int = 70

    si_batch_shots: int = 5
    si_callback_freq: int = 1
    si_pml_width: tuple[int, int, int, int] = (20, 20, 20, 20)
    si_pml_freq: float = 15.0
    si_cal_pad: list[int] | None = None

