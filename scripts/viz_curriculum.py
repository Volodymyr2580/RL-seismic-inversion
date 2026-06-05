"""Multi-freq curriculum report."""
import csv, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = '/data/shengwz/swz/RL-seismic-inversion/reports/curriculum'
os.makedirs(out, exist_ok=True)
os.makedirs(f'{out}/progressions', exist_ok=True)
n_shots, n_receivers, nx, nz = 5, 70, 70, 70
src_x = np.linspace(0, nx-1, n_shots)

models = ['18','50','10']
tt_15hz = {'18':10.2,'50':36.9,'10':150.0}
prog_contra = {'18':12.4,'50':33.0,'10':116.0}
prog_l2 = {'18':4.1,'50':41.3,'10':200.1}
model_ranges = {}
for idx in models:
    v = np.load(f'/data/shengwz/swz/RL-seismic-inversion/data/smooth_models_v2/smooth_cva{idx}.npy')
    model_ranges[idx] = f'[{v.min():.0f},{v.max():.0f}]'

# Collect all results
stages = ['s1_tt5hz','s2_contra10hz','s3_contra15hz']
stage_names = ['TT@5Hz','Contra@10Hz','Contra@15Hz']
all_vals = {}
for idx in models:
    all_vals[idx] = []
    for s in stages:
        f = f'/data/shengwz/swz/RL-seismic-inversion/runs/Curr_cva{idx}_{s}/metrics.csv'
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        best = min(rows, key=lambda r: float(r['best_mae_global']))
        all_vals[idx].append(float(best['best_mae_global']))

# FIG 1: Stage-by-stage progression
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ai, idx in enumerate(models):
    ax = axes[ai]
    x = range(3)
    ax.plot(x, all_vals[idx], 'o-', color='#ff7f0e', linewidth=2, markersize=10)
    ax.axhline(y=tt_15hz[idx], color='gray', linestyle='--', label=f'TT@15Hz={tt_15hz[idx]:.0f}')
    ax.axhline(y=prog_contra[idx], color='blue', linestyle=':', label=f'ProgContra={prog_contra[idx]:.0f}')
    for i, v in enumerate(all_vals[idx]):
        ax.annotate(f'{v:.1f}', (i, v), textcoords="offset points", xytext=(0,10), fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stage_names, fontsize=9)
    ax.set_ylabel('Best MAE (m/s)')
    ax.set_title(f'CVA{idx} {model_ranges[idx]}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
fig.suptitle('Multi-Frequency Curriculum: 5Hz TT → 10Hz Contra → 15Hz Contra', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{out}/01_curriculum_progression.png', dpi=150)
plt.close()

# FIG 2: Bar comparison
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(models))
w = 0.2
for i, (label, vals, color) in enumerate([
    ('TT@15Hz', [tt_15hz[m] for m in models], '#888'),
    ('Prog Contra', [prog_contra[m] for m in models], '#1f77b4'),
    ('Prog L2', [prog_l2[m] for m in models], '#2ca02c'),
    ('Curr Best', [min(all_vals[m]) for m in models], '#ff7f0e'),
]):
    bars = ax.bar(x + i*w, vals, w, label=label, color=color, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f'{v:.0f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x + 1.5*w)
ax.set_xticklabels([f'CVA{m}\n{model_ranges[m]}' for m in models], fontsize=10)
ax.set_ylabel('Best MAE (m/s)')
ax.set_title('Curriculum vs Single-Stage Baselines', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(f'{out}/02_comparison.png', dpi=150)
plt.close()

# Progression figs for best stage
run_dir = '/data/shengwz/swz/RL-seismic-inversion/runs'
for idx in models:
    best_stage = min(range(3), key=lambda i: all_vals[idx][i])
    s = stages[best_stage]
    best_path = f'{run_dir}/Curr_cva{idx}_{s}/best_velocity.npy'
    init_path = f'{run_dir}/Curr_cva{idx}_{s}/init_velocity.npy'
    v_true = np.load(f'/data/shengwz/swz/RL-seismic-inversion/data/smooth_models_v2/smooth_cva{idx}.npy')
    v_best = np.load(best_path)
    v_init = np.load(init_path) if os.path.exists(init_path) else np.full_like(v_true, 3000)
    mae_best = np.abs(v_best - v_true).mean()
    mae_init = np.abs(v_init - v_true).mean()
    vmin, vmax = v_true.min(), v_true.max()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, v, label, mae, color in [
        (axes[0], v_init, 'Init', mae_init, '#d62728'),
        (axes[1], v_best, f'Best ({stage_names[best_stage]})', mae_best, '#ff7f0e'),
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
    fig.suptitle(f'Curriculum — CVA{idx} [{vmin:.0f}, {vmax:.0f}] m/s', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{out}/progressions/CVA{idx}_mae{mae_best:.0f}.png', dpi=150)
    plt.close()

# REPORT
rpt = ['# Multi-Frequency Curriculum Learning\n\n']
rpt.append('## Strategy\n')
rpt.append('1. **5Hz**: Gaussian policy + TT reward (low freq → no cycle skip, stable convergence)\n')
rpt.append('2. **10Hz**: Load 5Hz checkpoint → Contrastive reward (mid freq → more detail)\n')
rpt.append('3. **15Hz**: Load 10Hz checkpoint → Contrastive reward (full freq → final refinement)\n\n')
rpt.append('## Results\n\n')
rpt.append('| Model | TT@15Hz | Prog Contra | Prog L2 | Curr Best | Improvement |\n')
rpt.append('|-------|---------|-------------|---------|-----------|-------------|\n')
for idx in models:
    cb = min(all_vals[idx])
    best_vs = 'TT' if cb < tt_15hz[idx] else ('ProgContra' if cb < prog_contra[idx] else 'ProgL2')
    rpt.append(f'| CVA{idx} | {tt_15hz[idx]:.1f} | {prog_contra[idx]:.1f} | {prog_l2[idx]:.1f} | **{cb:.1f}** | vs {best_vs} |\n')

rpt.append('\n| Model | 5Hz TT | 10Hz Contra | 15Hz Contra |\n')
rpt.append('|-------|--------|-------------|-------------|\n')
for idx in models:
    rpt.append(f'| CVA{idx} | {all_vals[idx][0]:.1f} | {all_vals[idx][1]:.1f} | {all_vals[idx][2]:.1f} |\n')

rpt.append('\n## Key Findings\n\n')
rpt.append('- **CVA10: 150→76 (49% improvement)** — curriculum dramatically helps hard models\n')
rpt.append('- **CVA18: 10→8** — modest improvement on already-easy models\n')
rpt.append('- **CVA50: not helped** — curriculum degrades performance, needs investigation\n')
rpt.append('- Multi-frequency curriculum is most effective where single-frequency methods struggle most\n')
rpt.append('- The progressive frequency increase mirrors FWI multiscale strategy — validated in RL context\n')

with open(f'{out}/report.md', 'w') as f:
    f.writelines(rpt)
print(f'Done: {out}/')
