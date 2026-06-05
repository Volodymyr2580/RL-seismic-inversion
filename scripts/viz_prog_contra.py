"""Prog TT→Contrastive report + viz."""
import csv, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = '/data/shengwz/swz/RL-seismic-inversion/reports/prog_contra'
os.makedirs(out, exist_ok=True)
os.makedirs(f'{out}/progressions', exist_ok=True)
n_shots, n_receivers, nx, nz = 5, 70, 70, 70
src_x = np.linspace(0, nx-1, n_shots)

models = ['18','50','10','6','8','5']
tt_best = {'18':10.2,'50':36.9,'10':150.0,'6':58.4,'8':75.1,'5':106.0}
prog_l2 = {'18':4.1,'50':41.3,'10':200.1,'6':51.7,'8':69.1,'5':141.2}
prog_contra = {}
for idx in models:
    f = f'/data/shengwz/swz/RL-seismic-inversion/runs/ProgContra_cva{idx}/metrics.csv'
    with open(f) as fh:
        rows = list(csv.DictReader(fh))
    best = min(rows, key=lambda r: float(r['best_mae_global']))
    prog_contra[idx] = float(best['best_mae_global'])

model_ranges = {}
for idx in models:
    v = np.load(f'/data/shengwz/swz/RL-seismic-inversion/data/smooth_models_v2/smooth_cva{idx}.npy')
    model_ranges[idx] = f'[{v.min():.0f},{v.max():.0f}]'

# FIG 1: Comparison bar chart
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()
for ai, idx in enumerate(models):
    ax = axes[ai]
    vals = [tt_best[idx], prog_l2[idx], prog_contra[idx]]
    names = ['TT-only', 'Prog TT→L2', 'Prog TT→Contra']
    colors = ['#888', '#2ca02c', '#ff7f0e']
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+2, f'{v:.1f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_title(f'CVA{idx} {model_ranges[idx]}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best MAE (m/s)')
    ax.grid(axis='y', alpha=0.3)
fig.suptitle('Progressive TT → Contrastive vs Baselines', fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{out}/01_comparison.png', dpi=150)
plt.close()

# FIG 2-7: Progression per model
run_dir = '/data/shengwz/swz/RL-seismic-inversion/runs'
for idx in models:
    best_path = f'{run_dir}/ProgContra_cva{idx}/best_velocity.npy'
    init_path = f'{run_dir}/ProgContra_cva{idx}/init_velocity.npy'
    v_true = np.load(f'/data/shengwz/swz/RL-seismic-inversion/data/smooth_models_v2/smooth_cva{idx}.npy')
    v_best = np.load(best_path)
    v_init = np.load(init_path) if os.path.exists(init_path) else np.full_like(v_true, 3000)
    mae_best = np.abs(v_best - v_true).mean()
    mae_init = np.abs(v_init - v_true).mean()
    vmin, vmax = v_true.min(), v_true.max()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, v, label, mae, color in [
        (axes[0], v_init, 'Init (from TT best)', mae_init, '#d62728'),
        (axes[1], v_best, 'Best (Contrastive)', mae_best, '#ff7f0e'),
        (axes[2], v_true, 'True', 0.0, 'black')]:
        im = ax.imshow(v, origin='upper', cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='m/s')
        ax.scatter(src_x, np.full(n_shots, nz-1), marker='v', c='red', s=80, zorder=5, edgecolors='white', linewidths=0.8, label='Src')
        ax.scatter(np.arange(n_receivers), np.zeros(n_receivers), marker='^', c='cyan', s=15, zorder=5, edgecolors='white', linewidths=0.3, label='Rec', alpha=0.7)
        ax.set_xlabel('x (grid)'); ax.set_ylabel('z (grid)')
        t = label; 
        if mae > 0: t += f'  |  MAE = {mae:.1f} m/s'
        ax.set_title(t, fontsize=12, color=color, fontweight='bold')
        ax.legend(loc='lower right', fontsize=7, markerscale=0.8, framealpha=0.8)
    fig.suptitle(f'Prog TT→Contrastive — CVA{idx} [{vmin:.0f}, {vmax:.0f}] m/s', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{out}/progressions/CVA{idx}_mae{mae_best:.0f}.png', dpi=150)
    plt.close()

# REPORT
rpt = ['# Progressive TT → Contrastive\n\n']
rpt.append('## Strategy\n')
rpt.append('1. Train Gaussian policy with TT reward (log-scaled energy-ratio picker)\n')
rpt.append('2. Load best TT checkpoint\n')
rpt.append('3. Continue training with Contrastive reward (spectrum similarity + cross-correlation)\n')
rpt.append('Lower initial temperature (1.0→0.1, 500 steps) since policy is already near optimum.\n\n')
rpt.append('## Results\n\n')
rpt.append('| Model | Range | TT-only | Prog TT→L2 | Prog TT→Contra | vs TT | vs ProgL2 |\n')
rpt.append('|-------|-------|---------|------------|----------------|-------|----------|\n')
for idx in models:
    pc = prog_contra[idx]; tt = tt_best[idx]; pl = prog_l2[idx]
    rpt.append(f'| CVA{idx} | {model_ranges[idx]} | {tt:.1f} | {pl:.1f} | {pc:.1f} | {pc-tt:+.1f} | {pc-pl:+.1f} |\n')

improved_tt = sum(1 for idx in models if prog_contra[idx] < tt_best[idx])
improved_l2 = sum(1 for idx in models if prog_contra[idx] < prog_l2[idx])
rpt.append(f'\n## Key Findings\n\n')
rpt.append(f'- Beat TT-only on {improved_tt}/6 models\n')
rpt.append(f'- Beat Prog L2 on {improved_l2}/6 models\n')
rpt.append(f'- **Strong on hard models**: CVA10 (150→116, -23%), CVA5 (106→81, -24%)\n')
rpt.append(f'- **Weak on easy models**: CVA18 (10→12, worse) — Contrastive lacks L2''s precision for already-good models\n')
rpt.append(f'- **Recommendation**: Use Prog L2 for well-constrained models, Prog Contra for difficult ones\n')

with open(f'{out}/report.md', 'w') as f:
    f.writelines(rpt)
print(f'Done: {out}/')
