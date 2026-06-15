"""Build an HTML report for the Phase7 reward suite."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ["marmousi", "salt_A_90x270", "synthetic"]
INIT_ORDER = ["far_uniform", "medium_gradient", "near_blur"]
REWARD_ORDER = [
    "tt_only",
    "l1l2",
    "wasserstein_w1",
    "wasserstein_w2",
    "ncc_zero",
    "ncc_maxlag",
    "envelope_ncc",
    "awi",
]

MODEL_INFO = {
    "marmousi": {
        "title": "Marmousi",
        "nx": 210,
        "nz": 70,
        "dx": 10.0,
        "dz": 10.0,
        "ctrl": "30x10",
        "receivers": 210,
    },
    "salt_A_90x270": {
        "title": "Salt body crop A",
        "nx": 270,
        "nz": 90,
        "dx": 10.0,
        "dz": 10.0,
        "ctrl": "36x12",
        "receivers": 270,
    },
    "synthetic": {
        "title": "Generated synthetic",
        "nx": 210,
        "nz": 70,
        "dx": 10.0,
        "dz": 10.0,
        "ctrl": "30x10",
        "receivers": 210,
    },
}

REWARD_TEXT = {
    "tt_only": "Travel-time only reward. Uses log-scaled first-arrival time-shift similarity; L1/L2 waveform terms are disabled.",
    "l1l2": "Sign-preserving log waveform misfit with both L1 and L2 components enabled.",
    "wasserstein_w1": "Wasserstein-1 style trace distribution distance, using absolute-amplitude normalization.",
    "wasserstein_w2": "Wasserstein-2 style trace distribution distance.",
    "ncc_zero": "Zero-lag normalized cross-correlation; sensitive to waveform similarity without allowing time shift.",
    "ncc_maxlag": "Maximum normalized cross-correlation within a finite lag window, with lag penalty.",
    "envelope_ncc": "Normalized cross-correlation on waveform envelopes, reducing polarity/phase sensitivity.",
    "awi": "Adaptive waveform inversion reward. The default script uses the lightweight L1 AWI variant.",
}


def read_metrics(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, val in row.items():
                if val == "" or val is None:
                    continue
                try:
                    parsed[key] = float(val)
                except ValueError:
                    pass
            rows.append(parsed)
    return rows


def read_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_runs(runs_root: Path) -> list[dict]:
    rows: list[dict] = []
    for model in MODEL_ORDER:
        for init in INIT_ORDER:
            for reward in REWARD_ORDER:
                run_dir = runs_root / model / init / reward
                cfg = read_config(run_dir / "config.json")
                metrics = read_metrics(run_dir / "metrics.csv")
                last = metrics[-1] if metrics else {}
                best = min((r.get("best_mae_global", np.inf) for r in metrics), default=np.inf)
                status = "complete" if (run_dir / "policy_final.pt").exists() and (run_dir / "final_velocity.npy").exists() else "missing"
                if metrics and status == "missing":
                    status = "partial"
                rows.append(
                    {
                        "model": model,
                        "init": init,
                        "reward": reward,
                        "run_dir": str(run_dir),
                        "status": status,
                        "steps_done": int(last.get("step", 0)),
                        "best_mae_global": float(best) if np.isfinite(best) else np.nan,
                        "last_mae_oracle": last.get("mae_oracle_best", np.nan),
                        "last_reward_l1": last.get("reward_l1_mean", np.nan),
                        "last_reward_l2": last.get("reward_l2_mean", np.nan),
                        "last_reward_tt": last.get("reward_tt_mean", np.nan),
                        "wall_time_s": last.get("wall_time", np.nan),
                        "fwi_type": cfg.get("fwi_type", ""),
                        "init_velocity_path": cfg.get("init_velocity_path", ""),
                    }
                )
    return rows


def rel(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve()).replace("\\", "/")


def write_summary(rows: list[dict], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with (report_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def plot_heatmaps(rows: list[dict], report_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_paths = []
    for model in MODEL_ORDER:
        arr = np.full((len(REWARD_ORDER), len(INIT_ORDER)), np.nan, dtype=np.float32)
        for r in rows:
            if r["model"] != model:
                continue
            i = REWARD_ORDER.index(r["reward"])
            j = INIT_ORDER.index(r["init"])
            arr[i, j] = r["best_mae_global"]
        fig, ax = plt.subplots(figsize=(7, 8))
        im = ax.imshow(arr, cmap="viridis_r", aspect="auto")
        ax.set_xticks(np.arange(len(INIT_ORDER)), INIT_ORDER, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(REWARD_ORDER)), REWARD_ORDER)
        ax.set_title(f"{MODEL_INFO[model]['title']} best MAE (lower is better)")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                txt = "-" if np.isnan(val) else f"{val:.0f}"
                ax.text(j, i, txt, ha="center", va="center", color="white" if not np.isnan(val) else "gray", fontsize=8)
        fig.colorbar(im, ax=ax, label="m/s")
        fig.tight_layout()
        out = report_dir / f"heatmap_{model}.png"
        fig.savefig(out, dpi=170)
        plt.close(fig)
        out_paths.append(out)
    return out_paths


def img_tag(path: Path, report_dir: Path, alt: str, cls: str = "") -> str:
    if not path.exists():
        return f"<div class='missing'>missing: {html.escape(str(path))}</div>"
    cls_attr = f" class='{cls}'" if cls else ""
    return f"<img{cls_attr} src='{html.escape(rel(path, report_dir))}' alt='{html.escape(alt)}'>"


def build_html(rows: list[dict], report_dir: Path, heatmaps: list[Path], runs_root: Path) -> None:
    css = """
    body{font-family:Inter,Segoe UI,Arial,sans-serif;margin:24px;color:#17202a;background:#fafafa}
    h1,h2,h3{margin:18px 0 10px} p{line-height:1.5}
    table{border-collapse:collapse;width:100%;font-size:13px;background:white}
    th,td{border:1px solid #d8dee8;padding:6px 8px;text-align:left}
    th{background:#eef2f7;position:sticky;top:0} tr.partial{background:#fff9e6} tr.missing{color:#777}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}
    .card{background:white;border:1px solid #dde3ee;border-radius:8px;padding:14px}
    img{max-width:100%;border:1px solid #d8dee8;border-radius:6px;background:white}
    .small{font-size:12px;color:#5d6d7e}.missing{padding:12px;background:#f4f4f4;color:#777;border-radius:6px}
    code{background:#eef2f7;padding:1px 4px;border-radius:4px}
    """

    rows_sorted = sorted(rows, key=lambda r: (MODEL_ORDER.index(r["model"]), INIT_ORDER.index(r["init"]), REWARD_ORDER.index(r["reward"])))

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Phase7 Reward Suite Report</title>",
        f"<style>{css}</style></head><body>",
        "<h1>Phase7 Reward Suite Report</h1>",
        "<p>Experiment grid: 3 models × 3 deterministic initial models × 8 reward settings. "
        "Acquisition is bottom sources and surface receivers for the Phase7 transmission setting.</p>",
        f"<p class='small'>Runs root: <code>{html.escape(str(runs_root))}</code></p>",
        "<h2>Model And Acquisition Settings</h2><div class='grid'>",
    ]

    for model in MODEL_ORDER:
        info = MODEL_INFO[model]
        panel_dir = ROOT / "outputs" / "phase7_sensitivity" / "compare_panels" / model
        parts.append("<div class='card'>")
        parts.append(f"<h3>{html.escape(info['title'])}</h3>")
        parts.append(
            f"<p>(nx,nz)=({info['nx']},{info['nz']}), (dx,dz)=({info['dx']},{info['dz']}) m, "
            f"B-spline control grid={info['ctrl']}. n_shots=7, n_receivers={info['receivers']}. "
            "Sources are placed at the bottom and receivers at the surface.</p>"
        )
        parts.append(img_tag(panel_dir / "model_annotated.png", report_dir, f"{model} model"))
        parts.append(img_tag(panel_dir / "global_gather_compare.png", report_dir, f"{model} gather compare"))
        parts.append("</div>")
    parts.append("</div>")

    parts.append("<h2>Initial Models</h2><p>The three starts are deterministic: "
                 "<code>far_uniform</code> is a single spatial mean velocity, "
                 "<code>medium_gradient</code> is a laterally uniform vertical gradient, and "
                 "<code>near_blur</code> is a strong Gaussian blur of the true model.</p><div class='grid'>")
    init_root = ROOT / "outputs" / "phase7_sensitivity" / "initial_models"
    for model in MODEL_ORDER:
        parts.append("<div class='card'>")
        parts.append(f"<h3>{html.escape(MODEL_INFO[model]['title'])}</h3>")
        parts.append(img_tag(init_root / model / "initial_models_preview.png", report_dir, f"{model} initial models"))
        parts.append("</div>")
    parts.append("</div>")

    parts.append("<h2>Reward Design</h2><table><tr><th>Reward</th><th>Design</th></tr>")
    for reward in REWARD_ORDER:
        parts.append(f"<tr><td><code>{html.escape(reward)}</code></td><td>{html.escape(REWARD_TEXT[reward])}</td></tr>")
    parts.append("</table>")

    parts.append("<h2>Best MAE Heatmaps</h2><div class='grid'>")
    for path in heatmaps:
        parts.append("<div class='card'>" + img_tag(path, report_dir, path.stem) + "</div>")
    parts.append("</div>")

    parts.append("<h2>Run Table</h2><table><tr>"
                 "<th>model</th><th>init</th><th>reward</th><th>status</th><th>steps</th>"
                 "<th>best MAE</th><th>last oracle MAE</th><th>R_L1</th><th>R_L2/FWI</th><th>R_TT</th>"
                 "<th>visuals</th><th>run dir</th></tr>")
    for r in rows_sorted:
        run_dir = Path(r["run_dir"])
        vis_dir = run_dir / "phase7_visuals"
        vis_link = rel(vis_dir / "models.png", report_dir) if (vis_dir / "models.png").exists() else ""
        vis_html = f"<a href='{html.escape(vis_link)}'>models</a>" if vis_link else "-"
        if (vis_dir / "wiggle_overlay.png").exists():
            vis_html += f" | <a href='{html.escape(rel(vis_dir / 'wiggle_overlay.png', report_dir))}'>wiggle</a>"
        if (vis_dir / "gather_images.png").exists():
            vis_html += f" | <a href='{html.escape(rel(vis_dir / 'gather_images.png', report_dir))}'>gather</a>"
        cls = html.escape(str(r["status"]))
        fmt = lambda x: "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3g}"
        parts.append(
            f"<tr class='{cls}'><td>{html.escape(r['model'])}</td><td>{html.escape(r['init'])}</td>"
            f"<td>{html.escape(r['reward'])}</td><td>{html.escape(r['status'])}</td><td>{r['steps_done']}</td>"
            f"<td>{fmt(r['best_mae_global'])}</td><td>{fmt(r['last_mae_oracle'])}</td>"
            f"<td>{fmt(r['last_reward_l1'])}</td><td>{fmt(r['last_reward_l2'])}</td><td>{fmt(r['last_reward_tt'])}</td>"
            f"<td>{vis_html}</td><td><code>{html.escape(str(run_dir))}</code></td></tr>"
        )
    parts.append("</table>")

    parts.append("<h2>Regeneration</h2>")
    parts.append("<p>After the suite finishes, regenerate this page with "
                 "<code>python scripts/phase7_build_report.py --runs_root runs/phase7 --report_dir reports/phase7_reward_suite</code>. "
                 "Per-run images are generated by <code>scripts/phase7_visualize_run.py</code>.</p>")
    parts.append("</body></html>")

    with (report_dir / "index.html").open("w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase7 reward-suite HTML report")
    parser.add_argument("--runs_root", default=str(ROOT / "runs" / "phase7"))
    parser.add_argument("--report_dir", default=str(ROOT / "reports" / "phase7_reward_suite"))
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    report_dir = Path(args.report_dir)
    rows = collect_runs(runs_root)
    write_summary(rows, report_dir)
    heatmaps = plot_heatmaps(rows, report_dir)
    build_html(rows, report_dir, heatmaps, runs_root)
    print(f"[ok] report: {report_dir / 'index.html'}")


if __name__ == "__main__":
    main()
