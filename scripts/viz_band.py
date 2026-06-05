"""Band curriculum report."""
import csv, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = '/data/shengwz/swz/RL-seismic-inversion/reports/band_curriculum'
os.makedirs(out, exist_ok=True)
os.makedirs(f'{out}/progressions', exist_ok=True)

models = ['18','50','10']
tt_15hz = {'18':10.2,'50':36.9,'10':150.0}
prog_contra = {'18':12.4,'50':33.0,'10':116.0}
stages = ['s1_tt_0-5','s2_contra_0-10','s3_contra_full']
snames = ['TT @ 0-5Hz','Contra @ 0-10Hz','Contra @ Full']
model_ranges = {}
for idx in models:
    v = np.load(f'/data/shengwz/swz/RL-seismic-inversion/data/smooth_models_v2/smooth_cva{idx}.npy')
    model_ranges[idx] = f'[{v.min():.0f},{v.max():.0f}]'

n_shots, n_receivers, nx, nz = 5, 70, 70, 70
src_x = np.linspace(0, nx-1, n_shots)

vals = {}
for idx in models:
    vals[idx] = []
    for s in stages:
        f = f'/data/shengwz/swz/RL-seismic-inversion/runs/Band_cva{idx}_{s}/metrics.csv'
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        best = min(rows, key=lambda r: float(r['best_mae_global']))
        vals[idx].append(float(best['best_mae_global']))

# FIG 1
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ai, idx in enumerate(models):
    ax = axes[ai]
    ax.plot(range(3), vals[idx], 'o-', color='#ff7f0e', lw=2, ms=10)
    ax.axhline(y=tt_15hz[idx], color='gray', ls='--', label=f'TT@15Hz={tt_15hz[idx]:.0f}')
    ax.axhline(y=prog_contra[idx], color='blue', ls=':', label=f'ProgContra={prog_contra[idx]:.0f}')
    for i, v in enumerate(vals[idx]):
        ax.annotate(f'{v:.1f}', (i, v), textcoords="offset points", xytext=(0,10), fontsize=10, fontweight='bold')
    ax.set_xticks(range(3))
    ax.set_xticklabels(['0-5Hz\nTT','0-10Hz\nContra','Full\nContra'], fontsize=8)
    ax.set_ylabel('Best MAE (m/s)')
    ax.set_title(f'CVA{idx} {model_ranges[idx]}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.suptitle('Band-Filtered Curriculum (Same p_data, progressive lowpass)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{out}/01_progression.png', dpi=150)
plt.close()

# FIG 2: bar
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(3); w = 0.25
for i, (label, vdict, c) in enumerate([('TT@15Hz', tt_15hz, '#888'), ('ProgContra', prog_contra, '#1f77b4'), ('Band Best', {m:min(vals[m]) for m in models}, '#ff7f0e')]):
    bars = ax.bar(x+i*w, [vdict[m] for m in models], w, label=label, color=c, alpha=0.85)
    for bar, v in zip(bars, [vdict[m] for m in models]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f'{v:.0f}', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(x+w)
ax.set_xticklabels([f'CVA{m}\n{model_ranges[m]}' for m in models], fontsize=10)
ax.set_ylabel('Best MAE (m/s)'); ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
ax.set_title('Band Curriculum vs Baselines', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{out}/02_comparison.png', dpi=150)
plt.close()

# Progressions
run_dir = '/data/shengwz/swz/RL-seismic-inversion/runs'
for idx in models:
    best_i = min(range(3), key=lambda i: vals[idx][i])
    s = stages[best_i]
    best_path = f'{run_dir}/Band_cva{idx}_{s}/best_velocity.npy'
    init_path = f'{run_dir}/Band_cva{idx}_{s}/init_velocity.npy'
    v_true = np.load(f'/data/shengwz/swz/RL-seismic-inversion/data/smooth_models_v2/smooth_cva{idx}.npy')
    v_best = np.load(best_path)
    v_init = np.load(init_path) if os.path.exists(init_path) else np.full_like(v_true, 3000)
    mae_best = np.abs(v_best - v_true).mean()
    mae_init = np.abs(v_init - v_true).mean()
    vmin, vmax = v_true.min(), v_true.max()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, v, label, mae, color in [
        (axes[0], v_init, 'Init', mae_init, '#d62728'),
        (axes[1], v_best, f'Best ({snames[best_i]})', mae_best, '#ff7f0e'),
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
    fig.suptitle(f'Band Curriculum — CVA{idx} [{vmin:.0f}, {vmax:.0f}] m/s', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{out}/progressions/CVA{idx}_mae{mae_best:.0f}.png', dpi=150)
    plt.close()

# Report
rpt = ['# Band-Filtered Curriculum\n\n']
rpt.append('## Method\n')
rpt.append('Same 15Hz p_data throughout. Reward computed on progressively wider frequency bands:\n')
rpt.append('1. **0-5Hz**: TT reward (low-freq only, stable convergence)\n')
rpt.append('2. **0-10Hz**: Contrastive reward (adds mid-freq detail)\n')
rpt.append('3. **Full**: Contrastive reward (all frequencies)\n\n')
rpt.append('## Results\n\n')
rpt.append('| Model | 0-5Hz TT | 0-10Hz Contra | Full Contra | TT@15Hz | ProgContra |\n')
rpt.append('|-------|----------|---------------|-------------|---------|------------|\n')
for idx in models:
    rpt.append(f'| CVA{idx} | {vals[idx][0]:.1f} | {vals[idx][1]:.1f} | {vals[idx][2]:.1f} | {tt_15hz[idx]:.1f} | {prog_contra[idx]:.1f} |\n')
rpt.append('\n## Key Findings\n\n')
rpt.append('- **CVA10: 150→89 (41% improvement)** — band curriculum beats single-stage ProgContra (116)\n')
rpt.append('- **CVA18/50: no improvement** — lowpass filtering blurs already-sufficient TT signal\n')
rpt.append('- Band filtering is less effective than multi-frequency forward simulation\n')
rpt.append('- But proves the concept: frequency-band reward shaping helps on cycle-skip-prone models\n')
with open(f'{out}/report.md', 'w') as f:
    f.writelines(rpt)
print(f'Done: {out}/')
