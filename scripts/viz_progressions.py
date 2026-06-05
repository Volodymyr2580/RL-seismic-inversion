"""
Generate init→best→true progression figures for all 19 CVA models.
"""
import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out_dir = '/data/shengwz/swz/RL-seismic-inversion/reports/tt_only_19models'
os.makedirs(out_dir, exist_ok=True)

# ── Model list and run name mapping ──
model_list = [
    ('18','B_cva18'),('19','B_cva19'),('50','phase4_gauss_logtt_s50'),
    ('9','B_cva9'),('20','B_cva20'),('12','B_cva12'),
    ('11','B_cva11'),('6','B_cva6'),('14','B_cva14'),('15','B_cva15'),
    ('1','M1_cva1'),('8','M8_cva8'),('3','B_cva3'),('2','B_cva2'),
    ('16','B_cva16'),('7','B_cva7'),('5','M5_cva5'),('4','B_cva4'),
    ('10','B_cva10')
]

# Collect data
models = []
for idx, run_name in model_list:
    run_dir = f'/data/shengwz/swz/RL-seismic-inversion/runs/{run_name}'
    v_true_path = f'/data/shengwz/swz/RL-seismic-inversion/data/smooth_models_v2/smooth_cva{idx}.npy'
    
    if idx == '50':
        v_true_path = '/data/shengwz/swz/RL-seismic-inversion/data/smooth_models/smooth_cva50.npy'
    
    best_path = os.path.join(run_dir, 'best_velocity.npy')
    init_path = os.path.join(run_dir, 'init_velocity.npy')
    
    if not all(os.path.exists(p) for p in [v_true_path, best_path, init_path]):
        print(f"SKIP CVA{idx}: missing files")
        continue
    
    v_true = np.load(v_true_path)
    v_best = np.load(best_path)
    v_init = np.load(init_path)
    
    mae_best = np.abs(v_best - v_true).mean()
    mae_init = np.abs(v_init - v_true).mean()
    
    models.append({
        'idx': idx, 'v_true': v_true, 'v_best': v_best, 'v_init': v_init,
        'mae_best': mae_best, 'mae_init': mae_init,
        'vmin': v_true.min(), 'vmax': v_true.max()
    })

# Sort by best MAE
models.sort(key=lambda m: m['mae_best'])
n = len(models)
print(f"Loaded {n} models, best MAE range [{models[0]['mae_best']:.1f}, {models[-1]['mae_best']:.1f}]")

# ═══════════════════════════════════════════════════════════════
# Figure 5: Compact grid — all 19 models, init | best | true
# ═══════════════════════════════════════════════════════════════
ncols = 5
nrows = (n + ncols - 1) // ncols  # 4 rows for 19 models

fig, axes = plt.subplots(nrows * 3, ncols, figsize=(ncols * 3.2, nrows * 3.0))
axes = np.atleast_2d(axes)

# Global color scale
all_v = np.concatenate([m['v_true'].ravel() for m in models])
vmin_g, vmax_g = all_v.min(), all_v.max()

for i, m in enumerate(models):
    col = i % ncols
    row_base = (i // ncols) * 3
    
    for j, (v, label, mae) in enumerate([
        (m['v_init'], 'Init', m['mae_init']),
        (m['v_best'], 'Best', m['mae_best']),
        (m['v_true'], 'True', 0.0)
    ]):
        ax = axes[row_base + j, col]
        im = ax.imshow(v.T, origin='lower', cmap='viridis', aspect='auto',
                       vmin=vmin_g, vmax=vmax_g)
        ax.set_xticks([])
        ax.set_yticks([])
        
        color = 'red' if j == 0 else ('green' if j == 1 else 'black')
        title = f'{label}'
        if j < 2:
            title += f' ({mae:.0f})'
        ax.set_title(title, fontsize=7, color=color, fontweight='bold' if j == 2 else 'normal')
    
    # Model label on the left
    axes[row_base, col].set_ylabel(f'CVA{m["idx"]}\n[{m["vmin"]:.0f},{m["vmax"]:.0f}]',
                                    fontsize=6, rotation=0, ha='right', va='center',
                                    labelpad=25)

# Hide unused subplots
for i in range(n, nrows * ncols):
    col = i % ncols
    row_base = (i // ncols) * 3
    for j in range(3):
        axes[row_base + j, col].set_visible(False)

plt.suptitle('Travel-Time Only RL: Init → Best → True (sorted by Best MAE)', 
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, '05_all_progressions.png'), dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════
# Figure 6: Top 3 + Bottom 3 large detailed progression
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(6, 3, figsize=(14, 18))
selected = models[:3] + models[-3:]  # top 3 + bottom 3

for i, m in enumerate(selected):
    for j, (v, label, mae) in enumerate([
        (m['v_init'], 'Init', m['mae_init']),
        (m['v_best'], 'Best', m['mae_best']),
        (m['v_true'], 'True', 0.0)
    ]):
        ax = axes[i, j]
        im = ax.imshow(v.T, origin='lower', cmap='viridis', aspect='auto',
                       vmin=m['vmin'], vmax=m['vmax'])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if j == 0:
            ax.set_ylabel(f'z (grid)', fontsize=10)
        if i == 5:
            ax.set_xlabel('x (grid)', fontsize=10)
        
        color = 'red' if j == 0 else ('green' if j == 1 else 'black')
        title = f'CVA{m["idx"]} — {label}'
        if j < 2:
            title += f'\nMAE={mae:.1f} m/s'
        ax.set_title(title, fontsize=11, color=color, fontweight='bold' if j == 2 else 'normal')

plt.suptitle('Travel-Time Only RL: Detailed Progression (Top 3 + Bottom 3)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(out_dir, '06_detailed_top_bottom.png'), dpi=150)
plt.close()

print(f"Done. Saved 05_all_progressions.png and 06_detailed_top_bottom.png")
