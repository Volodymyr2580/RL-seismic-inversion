"""Phase IV Final Report — COMPLETE (9 strategies × 6 models)."""
import csv, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

out_dir = '/data/shengwz/swz/RL-seismic-inversion/reports/phase4_final'
os.makedirs(out_dir, exist_ok=True)
os.makedirs(f'{out_dir}/progressions', exist_ok=True)

n_shots, n_receivers, nx, nz = 5, 70, 70, 70
src_x = np.linspace(0, nx-1, n_shots)

# ── Complete data ──
STRATEGIES = [
    ('TT-only',       {'18':'B_cva18','50':'phase4_gauss_logtt_s50','10':'B_cva10','6':'B_cva6','8':'M8_cva8','5':'M5_cva5'}),
    ('Prog_TT_to_L2', {'18':'Prog_cva18','50':'Prog_cva50','10':'Prog_cva10','6':'Prog_cva6','8':'Prog_cva8','5':'Prog_cva5'}),
    ('Multi_TT+L2',   {'18':'Multi2_cva18','50':'Multi2_cva50','10':'Multi2_cva10','6':'Multi2_cva6','8':'Multi2_cva8','5':'Multi2_cva5'}),
    ('L1+L2',         {'18':'L1L2_cva18','50':'L1L2_cva50','10':'L1L2_cva10','6':'L1L2_cva6','8':'L1L2_cva8','5':'L1L2_cva5'}),
    ('FWI_L2',        {'18':'L2_cva18','50':'L2_cva50','10':'L2_cva10','6':'L2_cva6','8':'L2_cva8','5':'L2_cva5'}),
    ('FWI_Envelope',  {'18':'FWI_envelope_cva18','50':'FWI_envelope_cva50','10':'FWI_envelope_cva10','6':'FWI_envelope_cva6','8':'FWI_envelope_cva8','5':'FWI_envelope_cva5'}),
    ('FWI_Windowed',  {'18':'FWI_windowed_l2_cva18','50':'FWI_windowed_l2_cva50','10':'FWI_windowed_l2_cva10','6':'FWI_windowed_l2_cva6','8':'FWI_windowed_l2_cva8','5':'FWI_windowed_l2_cva5'}),
    ('FWI_Wasserstein',{'18':'FWI_wasserstein_cva18','50':'FWI_wasserstein_cva50','10':'FWI_wasserstein_cva10','6':'FWI_wasserstein_cva6','8':'FWI_wasserstein_cva8','5':'FWI_wasserstein_cva5'}),
    ('FWI_Contrastive',{'18':'FWI_contrastive_cva18','50':'FWI_contrastive_cva50','10':'FWI_contrastive_cva10','6':'FWI_contrastive_cva6','8':'FWI_contrastive_cva8','5':'FWI_contrastive_cva5'}),
]

models = ['18','50','10','6','8','5']
model_ranges = {}
for idx in models:
    v = np.load(f'/data/shengwz/swz/RL-seismic-inversion/data/smooth_models_v2/smooth_cva{idx}.npy')
    model_ranges[idx] = f'[{v.min():.0f},{v.max():.0f}]'

run_dir = '/data/shengwz/swz/RL-seismic-inversion/runs'
data = {}
for name, run_map in STRATEGIES:
    data[name] = {}
    for idx, run_name in run_map.items():
        f = os.path.join(run_dir, run_name, 'metrics.csv')
        if os.path.exists(f):
            with open(f) as fh:
                rows = list(csv.DictReader(fh))
            best = min(rows, key=lambda r: float(r['best_mae_global']))
            data[name][idx] = float(best['best_mae_global'])
        else:
            data[name][idx] = float('nan')

# ═══════ FIG 1: Bar chart — all strategies on all 6 models ═══════
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
axes = axes.flatten()
strat_names = [s[0] for s in STRATEGIES]
colors = plt.cm.tab10(np.linspace(0, 1, len(strat_names)))

for ai, idx in enumerate(models):
    ax = axes[ai]
    vals = [data[n].get(idx, 0) for n in strat_names]
    bars = ax.barh(range(len(strat_names)), vals, color=colors)
    for i, (n, v) in enumerate(zip(strat_names, vals)):
        if not np.isnan(v):
            ax.text(v+2, i, f'{v:.0f}', va='center', fontsize=7, fontweight='bold')
    ax.set_yticks(range(len(strat_names)))
    ax.set_yticklabels([n.replace('_','\n') for n in strat_names], fontsize=6)
    ax.set_xlabel('Best MAE (m/s)')
    ax.set_title(f'CVA{idx} {model_ranges[idx]}', fontsize=11, fontweight='bold')
    ax.invert_yaxis(); ax.grid(axis='x', alpha=0.3)
fig.suptitle('Phase IV Final: 9 Strategies × 6 Models', fontsize=15, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{out_dir}/01_all_strategies.png', dpi=150)
plt.close()

# ═══════ FIG 2: Heatmap ═══════
fig, ax = plt.subplots(figsize=(14, 6))
mat = np.array([[data[n].get(m, np.nan) for m in models] for n in strat_names])
mask = np.isnan(mat)
im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=400)
for i in range(len(strat_names)):
    for j in range(len(models)):
        if not mask[i,j]:
            c = 'white' if mat[i,j] > 200 else 'black'
            ax.text(j, i, f'{mat[i,j]:.0f}', ha='center', va='center', fontsize=9, fontweight='bold', color=c)
ax.set_xticks(range(len(models)))
ax.set_xticklabels([f'CVA{m}\n{model_ranges[m]}' for m in models], fontsize=8)
ax.set_yticks(range(len(strat_names)))
ax.set_yticklabels(strat_names, fontsize=9)
ax.set_title('Best MAE Heatmap', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='MAE (m/s)', fraction=0.02)
plt.tight_layout()
fig.savefig(f'{out_dir}/02_heatmap.png', dpi=150)
plt.close()

# ═══════ FIG 3: Rank summary ═══════
fig, ax = plt.subplots(figsize=(14, 6))
best_per_model = {}
for idx in models:
    vals = [(n, data[n][idx]) for n in strat_names if not np.isnan(data[n].get(idx, float('nan')))]
    best_per_model[idx] = min(vals, key=lambda x: x[1])

x = np.arange(len(models))
width = 0.12
for si, sname in enumerate(strat_names):
    vals = [data[sname].get(m, 0) if not np.isnan(data[sname].get(m, float('nan'))) else 0 for m in models]
    bars = ax.bar(x + si*width, vals, width, label=sname, color=colors[si], alpha=0.85)
    for bi, (bar, v) in enumerate(zip(bars, vals)):
        if v > 0 and v < 400:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3, f'{v:.0f}',
                    ha='center', va='bottom', fontsize=5, rotation=90)

ax.set_xticks(x + width*(len(strat_names)-1)/2)
ax.set_xticklabels([f'CVA{m}\n{model_ranges[m]}' for m in models], fontsize=8)
ax.set_ylabel('Best MAE (m/s)')
ax.set_title('Grouped Bar: All Strategies per Model', fontsize=14, fontweight='bold')
ax.legend(fontsize=6, ncol=3, loc='upper left')
ax.set_ylim(0, 400); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(f'{out_dir}/03_grouped_bar.png', dpi=150)
plt.close()

# ═══════ Progression figures for all experiments ═══════
for sname, run_map in STRATEGIES:
    strat_dir = os.path.join(out_dir, 'progressions', sname)
    os.makedirs(strat_dir, exist_ok=True)
    for idx, run_name in run_map.items():
        best_path = os.path.join(run_dir, run_name, 'best_velocity.npy')
        init_path = os.path.join(run_dir, run_name, 'init_velocity.npy')
        v_true_path = f'/data/shengwz/swz/RL-seismic-inversion/data/smooth_models_v2/smooth_cva{idx}.npy'
        if not os.path.exists(best_path): continue
        v_true = np.load(v_true_path)
        v_best = np.load(best_path)
        v_init = np.load(init_path) if os.path.exists(init_path) else np.full_like(v_true, 3000)
        mae_best = np.abs(v_best - v_true).mean()
        mae_init = np.abs(v_init - v_true).mean()
        vmin, vmax = v_true.min(), v_true.max()
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        for ax, v, label, mae, color in [
            (axes[0], v_init, 'Init', mae_init, '#d62728'),
            (axes[1], v_best, 'Best', mae_best, '#2ca02c'),
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
        fig.suptitle(f'{sname} — CVA{idx} [{vmin:.0f}, {vmax:.0f}] m/s', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(strat_dir, f'CVA{idx}_mae{mae_best:.0f}.png'), dpi=150)
        plt.close()

# ═══════ REPORT ═══════
rpt = ['# Phase IV Final Report — Complete\n\n']
rpt.append('## Results (9 strategies × 6 models)\n\n')
rpt.append('| Strategy |' + '|'.join(f' CVA{m} ' for m in models) + '|\n')
rpt.append('|' + '---|'*7 + '\n')
for sname, _ in STRATEGIES:
    row = f'| {sname} |'
    for idx in models:
        v = data[sname].get(idx, float('nan'))
        row += f' {v:.1f} |' if not np.isnan(v) else ' — |'
    rpt.append(row + '\n')

rpt.append('\n## Best per Model\n\n')
for idx in models:
    vals = [(n, data[n][idx]) for n in strat_names if not np.isnan(data[n].get(idx, float('nan')))]
    if vals:
        best = min(vals, key=lambda x: x[1])
        rpt.append(f'- **CVA{idx}**: {best[1]:.1f} ({best[0]})\n')

rpt.append('\n## Key Findings\n\n')
rpt.append('1. **Progressive TT→L2 is best for easy models** (CVA18: 4.1)\n')
rpt.append('2. **Contrastive reward excels on hard models** (CVA10: 97.5, CVA5: 90.3)\n')
rpt.append('3. **L1+L2 is competitive** (CVA8: 22.8, CVA6: 33.5)\n')
rpt.append('4. **No single strategy dominates** — best approach depends on model complexity\n')
rpt.append('5. **Envelope/Wasserstein/Windowed fail from random init** — need good starting point\n')
rpt.append('6. **TT-only is the most robust baseline** — never catastrophic\n')

with open(f'{out_dir}/report.md', 'w') as f:
    f.writelines(rpt)

# Count progressions
total_prog = sum(len(list(os.walk(os.path.join(out_dir, 'progressions', s[0])))) for s in STRATEGIES)
import glob
total_prog = len(glob.glob(f'{out_dir}/progressions/*/*.png'))
print(f'Done: 3 summary figs + {total_prog} progression figs + report.md')
