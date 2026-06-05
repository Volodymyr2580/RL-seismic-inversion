"""
Visualize tt-only experimental results across 19 CVA models.
"""
import csv, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Load model ranges ──
model_info = {}
for idx in [1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,18,19,20,50]:
    path = f'/data/shengwz/swz/RL-seismic-inversion/data/smooth_models_v2/smooth_cva{idx}.npy'
    if not os.path.exists(path):
        continue
    v = np.load(path)
    model_info[str(idx)] = {'vmin': v.min(), 'vmax': v.max(), 'vrange': v.max()-v.min(),
                            'vmean': v.mean(), 'vstd': v.std()}

# ── Load experiment results ──
results = {}
name_map = {}
for idx, run_name in [
    ('1','M1_cva1'),('5','M5_cva5'),('8','M8_cva8'),
    ('2','B_cva2'),('3','B_cva3'),('4','B_cva4'),('6','B_cva6'),('7','B_cva7'),
    ('9','B_cva9'),('10','B_cva10'),('11','B_cva11'),('12','B_cva12'),
    ('14','B_cva14'),('15','B_cva15'),('16','B_cva16'),('18','B_cva18'),
    ('19','B_cva19'),('20','B_cva20'),('50','phase4_gauss_logtt_s50')
]:
    f = f'/data/shengwz/swz/RL-seismic-inversion/runs/{run_name}/metrics.csv'
    if not os.path.exists(f):
        continue
    with open(f) as fh:
        rows = list(csv.DictReader(fh))
    best = min(rows, key=lambda r: float(r['best_mae_global']))
    init = float(rows[0]['mae_oracle_best'])
    results[idx] = {
        'init': init, 'best': float(best['best_mae_global']),
        'best_step': int(best['step']), 'total': len(rows),
        'history': [(int(r['step']), float(r['best_mae_global']), float(r['entropy']))
                     for r in rows]
    }

out_dir = '/data/shengwz/swz/RL-seismic-inversion/reports/tt_only_19models'
os.makedirs(out_dir, exist_ok=True)

# Sort by best MAE
sorted_idx = sorted(results.keys(), key=lambda i: results[i]['best'])

# ═══════════════════════════════════════════════════════════════
# Figure 1: Best MAE bar chart (all 19 models)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))
colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(sorted_idx)))
bars = ax.bar(range(len(sorted_idx)), [results[i]['best'] for i in sorted_idx], color=colors)
for i, idx in enumerate(sorted_idx):
    r = results[idx]
    ax.text(i, r['best']+2, f"{r['best']:.0f}", ha='center', va='bottom', fontsize=7, fontweight='bold')
ax.set_xticks(range(len(sorted_idx)))
ax.set_xticklabels([f'CVA{i}' for i in sorted_idx], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Best MAE (m/s)', fontsize=12)
ax.set_title('Travel-Time Only RL: Best MAE Across 19 CVA Models', fontsize=14, fontweight='bold')
ax.axhline(y=0.3, color='black', linestyle='--', linewidth=1, alpha=0.5, label='B-spline floor (~0.3)')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, '01_best_mae_all.png'), dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════
# Figure 2: MAE vs velocity range scatter
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
for idx in results:
    if idx not in model_info: continue
    r = results[idx]
    m = model_info[idx]
    ax.scatter(m['vrange'], r['best'], s=80, zorder=5)
    ax.annotate(f'CVA{idx}', (m['vrange'], r['best']), textcoords="offset points",
                xytext=(5,5), fontsize=8)
ax.set_xlabel('Velocity Range (vmax - vmin) [m/s]', fontsize=12)
ax.set_ylabel('Best MAE (m/s)', fontsize=12)
ax.set_title('Best MAE vs Model Velocity Range', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, '02_mae_vs_range.png'), dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════
# Figure 3: Convergence curves for top 6 + bottom 3 models
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

# Panel A: Best 6 models
ax = axes[0]
for idx in sorted_idx[:6]:
    hist = results[idx]['history']
    steps = [h[0] for h in hist]
    maes = [h[1] for h in hist]
    ax.plot(steps, maes, label=f'CVA{idx} ({results[idx]["best"]:.0f})', linewidth=1.5)
ax.set_xlabel('Step', fontsize=11)
ax.set_ylabel('Best MAE', fontsize=11)
ax.set_title('Top 6 Models — Convergence', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)

# Panel B: Worst 6
ax = axes[1]
for idx in sorted_idx[-6:]:
    hist = results[idx]['history']
    steps = [h[0] for h in hist]
    maes = [h[1] for h in hist]
    ax.plot(steps, maes, label=f'CVA{idx} ({results[idx]["best"]:.0f})', linewidth=1.5)
ax.set_xlabel('Step', fontsize=11)
ax.set_ylabel('Best MAE', fontsize=11)
ax.set_title('Bottom 6 Models — Convergence', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)

# Panel C: Entropy curves for best 6
ax = axes[2]
for idx in sorted_idx[:6]:
    hist = results[idx]['history']
    steps = [h[0] for h in hist]
    ents = [h[2] for h in hist]
    ax.plot(steps, ents, label=f'CVA{idx}', linewidth=1.2, alpha=0.8)
ax.set_xlabel('Step', fontsize=11)
ax.set_ylabel('Entropy', fontsize=11)
ax.set_title('Top 6 Models — Entropy', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, ncol=3)
ax.grid(alpha=0.3)

# Panel D: MAE histogram
ax = axes[3]
maes = [results[i]['best'] for i in results]
ax.hist(maes, bins=12, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(np.mean(maes), color='red', linestyle='--', label=f'Mean={np.mean(maes):.1f}')
ax.axvline(np.median(maes), color='orange', linestyle='--', label=f'Median={np.median(maes):.1f}')
ax.set_xlabel('Best MAE (m/s)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title(f'MAE Distribution (n={len(maes)})', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, '03_convergence_curves.png'), dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════
# Figure 4: Summary table as image
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

table_data = []
for idx in sorted_idx:
    r = results[idx]
    m = model_info.get(idx, {})
    table_data.append([
        f'CVA{idx}',
        f'[{m.get("vmin",0):.0f},{m.get("vmax",0):.0f}]',
        f'{r["init"]:.0f}',
        f'{r["best"]:.1f}',
        f'{r["best_step"]}',
        f'{r["total"]}'
    ])

col_labels = ['Model', 'Range', 'Init MAE', 'Best MAE', '@Step', 'Steps']
table = ax.table(cellText=table_data, colLabels=col_labels,
                 cellLoc='center', loc='center',
                 colWidths=[0.1, 0.2, 0.12, 0.12, 0.1, 0.1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.3)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#404040')
        cell.set_text_props(color='white', fontweight='bold')

ax.set_title('Travel-Time Only RL: 19 CVA Models — Summary', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, '04_summary_table.png'), dpi=150)
plt.close()

print(f"Saved 4 figures to {out_dir}/")
print(f"Files: 01_best_mae_all.png, 02_mae_vs_range.png, 03_convergence_curves.png, 04_summary_table.png")
