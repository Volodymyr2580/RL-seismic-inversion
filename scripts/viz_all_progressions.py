"""Generate progression (init→best→true) for ALL experiments × models."""
import csv, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

out_base = '/data/shengwz/swz/RL-seismic-inversion/reports/phase4_final/progressions'
n_shots, n_receivers, nx, nz = 5, 70, 70, 70
src_x = np.linspace(0, nx-1, n_shots)

# Map: (strategy_name, run_prefix, model_indices)
EXPERIMENTS = [
    ('TT-only', {
        '18': 'B_cva18', '50': 'phase4_gauss_logtt_s50', '10': 'B_cva10',
        '6': 'B_cva6', '8': 'M8_cva8', '5': 'M5_cva5'
    }),
    ('Prog_TT_to_L2', {
        '18': 'Prog_cva18', '50': 'Prog_cva50', '10': 'Prog_cva10',
        '6': 'Prog_cva6', '8': 'Prog_cva8', '5': 'Prog_cva5'
    }),
    ('Multi_TT+L2', {
        '18': 'Multi2_cva18', '50': 'Multi2_cva50', '10': 'Multi2_cva10',
        '6': 'Multi2_cva6', '8': 'Multi2_cva8', '5': 'Multi2_cva5'
    }),
    ('L1+L2', {
        '18': 'L1L2_cva18', '50': 'L1L2_cva50', '10': 'L1L2_cva10',
        '6': 'L1L2_cva6', '8': 'L1L2_cva8', '5': 'L1L2_cva5'
    }),
    ('FWI_L2', {'18': 'FWI_l2_cva18', '50': 'FWI_l2_cva50', '10': 'FWI_l2_cva10'}),
    ('FWI_Envelope', {'18': 'FWI_envelope_cva18', '50': 'FWI_envelope_cva50', '10': 'FWI_envelope_cva10'}),
    ('FWI_Windowed', {'18': 'FWI_windowed_l2_cva18', '50': 'FWI_windowed_l2_cva50', '10': 'FWI_windowed_l2_cva10'}),
    ('FWI_Wasserstein', {'18': 'FWI_wasserstein_cva18', '50': 'FWI_wasserstein_cva50', '10': 'FWI_wasserstein_cva10'}),
    ('FWI_Contrastive', {'18': 'FWI_contrastive_cva18', '50': 'FWI_contrastive_cva50', '10': 'FWI_contrastive_cva10'}),
]

run_dir = '/data/shengwz/swz/RL-seismic-inversion/runs'
smooth_dir = '/data/shengwz/swz/RL-seismic-inversion/data/smooth_models_v2'

total = 0
for strategy, models in EXPERIMENTS:
    strat_dir = os.path.join(out_base, strategy)
    os.makedirs(strat_dir, exist_ok=True)
    
    for idx, run_name in models.items():
        run_path = os.path.join(run_dir, run_name)
        best_path = os.path.join(run_path, 'best_velocity.npy')
        init_path = os.path.join(run_path, 'init_velocity.npy')
        v_true_path = os.path.join(smooth_dir, f'smooth_cva{idx}.npy')
        
        if idx == '50' and 'FWI_' in strategy:
            v_true_path = os.path.join(smooth_dir, 'smooth_cva50.npy')
        
        if not os.path.exists(best_path):
            print(f"  SKIP {strategy}/CVA{idx}: no best_velocity.npy")
            continue
        
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
            (axes[2], v_true, 'True', 0.0, 'black')
        ]:
            im = ax.imshow(v, origin='upper', cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='m/s')
            ax.scatter(src_x, np.full(n_shots, nz-1), marker='v', c='red', s=80, zorder=5,
                       edgecolors='white', linewidths=0.8, label=f'Src')
            ax.scatter(np.arange(n_receivers), np.zeros(n_receivers), marker='^', c='cyan', s=15,
                       zorder=5, edgecolors='white', linewidths=0.3, label=f'Rec', alpha=0.7)
            ax.set_xlabel('x (grid)'); ax.set_ylabel('z (grid)')
            title = f'{label}'
            if mae > 0: title += f'  |  MAE = {mae:.1f} m/s'
            ax.set_title(title, fontsize=12, color=color, fontweight='bold')
            ax.legend(loc='lower right', fontsize=7, markerscale=0.8, framealpha=0.8)
        
        fig.suptitle(f'{strategy} — CVA{idx} [{vmin:.0f}, {vmax:.0f}] m/s',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(strat_dir, f'CVA{idx}_mae{mae_best:.0f}.png'), dpi=150)
        plt.close()
        total += 1

print(f"Done: {total} progression figures")
