"""Build an HTML report for Phase7 optimizer/parameterization strategy experiments."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = ["marmousi", "salt_A_90x270", "synthetic"]
INITS = ["far_uniform", "medium_gradient", "near_blur"]
REWARDS = ["tt_only", "ncc_zero", "wasserstein_w2"]
STRATEGIES = [
    "full_low_joint",
    "full_token_high",
    "full_token_low",
    "full_token_tether",
    "full_token_tether_strong",
    "coarse_token_low",
    "mid_token_low",
]

STRATEGY_TEXT = {
    "full_low_joint": "Full control grid, joint action PPO ratio, low exploration. Tests whether exploration alone caused degradation.",
    "full_token_high": "Full control grid, per-control-point PPO ratio, original high exploration. Tests ratio normalization alone.",
    "full_token_low": "Full control grid, token ratio plus low exploration. Main high-dimensional PPO stabilization candidate.",
    "full_token_tether": "Full grid, token ratio, low exploration, weak init tether. Tests constrained local refinement.",
    "full_token_tether_strong": "Full grid, token ratio, very low exploration, strong init tether. Tests conservative refinement.",
    "coarse_token_low": "Coarse control grid with token ratio and low exploration. Tests whether low-dimensional B-spline space is safer.",
    "mid_token_low": "Mid control grid with token ratio and low exploration. Tests the tradeoff between expressiveness and stability.",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fnum(value, default=math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def collect_runs(runs_root: Path) -> list[dict]:
    rows: list[dict] = []
    for strategy in STRATEGIES:
        reward_list = REWARDS if not (strategy.startswith("coarse_") or strategy.startswith("mid_")) else ["tt_only", "ncc_zero"]
        for model in MODELS:
            for init in INITS:
                for reward in reward_list:
                    run_dir = runs_root / strategy / model / init / reward
                    cfg = read_json(run_dir / "config.json")
                    metrics = read_csv(run_dir / "metrics.csv")
                    last = metrics[-1] if metrics else {}
                    status = "complete" if (run_dir / "policy_final.pt").exists() and (run_dir / "final_velocity.npy").exists() else "missing"
                    if metrics and status == "missing":
                        status = "partial"

                    initial_mae = fnum(last.get("initial_model_mae"))
                    best_mae = fnum(last.get("best_mae_global"))
                    final_oracle = fnum(last.get("mae_oracle_best"))
                    delta_best = fnum(last.get("delta_best_vs_init"), best_mae - initial_mae)
                    rows.append(
                        {
                            "strategy": strategy,
                            "model": model,
                            "init": init,
                            "reward": reward,
                            "status": status,
                            "steps_done": int(fnum(last.get("step"), 0)),
                            "initial_mae": initial_mae,
                            "best_mae": best_mae,
                            "final_oracle_mae": final_oracle,
                            "delta_best_vs_init": delta_best,
                            "delta_final_vs_init": final_oracle - initial_mae if math.isfinite(final_oracle) and math.isfinite(initial_mae) else math.nan,
                            "last_reward_l2": fnum(last.get("reward_l2_mean")),
                            "last_reward_tt": fnum(last.get("reward_tt_mean")),
                            "last_tether": fnum(last.get("reward_tether_mean")),
                            "ratio_mean": fnum(last.get("ratio_mean")),
                            "ratio_std": fnum(last.get("ratio_std")),
                            "clip_frac": fnum(last.get("clip_frac")),
                            "wall_time_s": fnum(last.get("wall_time")),
                            "nx_ctrl": cfg.get("nx_ctrl", ""),
                            "nz_ctrl": cfg.get("nz_ctrl", ""),
                            "run_dir": str(run_dir),
                        }
                    )
    return rows


def rel(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve()).replace("\\", "/")


def save_tables(rows: list[dict], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with (report_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_strategy_heatmaps(rows: list[dict], report_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_paths: list[Path] = []
    for model in MODELS:
        for init in INITS:
            sub = [r for r in rows if r["model"] == model and r["init"] == init and r["status"] != "missing"]
            if not sub:
                continue
            rewards = sorted(set(r["reward"] for r in sub), key=lambda x: REWARDS.index(x))
            strategies = [s for s in STRATEGIES if any(r["strategy"] == s for r in sub)]
            arr = np.full((len(strategies), len(rewards)), np.nan, dtype=np.float32)
            for r in sub:
                i = strategies.index(r["strategy"])
                j = rewards.index(r["reward"])
                arr[i, j] = float(r["delta_best_vs_init"])

            fig, ax = plt.subplots(figsize=(7, 0.45 * len(strategies) + 2.4))
            lim = float(np.nanpercentile(np.abs(arr), 95)) if np.isfinite(arr).any() else 1.0
            lim = max(lim, 1.0)
            im = ax.imshow(arr, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            ax.set_xticks(np.arange(len(rewards)), rewards, rotation=25, ha="right")
            ax.set_yticks(np.arange(len(strategies)), strategies)
            ax.set_title(f"{model} / {init}: delta best MAE vs init (negative is improvement)")
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    val = arr[i, j]
                    txt = "-" if np.isnan(val) else f"{val:+.0f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, label="best MAE - initial MAE (m/s)")
            fig.tight_layout()
            out = report_dir / f"delta_{model}_{init}.png"
            fig.savefig(out, dpi=170)
            plt.close(fig)
            out_paths.append(out)
    return out_paths


def fmt(x: float, nd: int = 1) -> str:
    if x is None or not math.isfinite(float(x)):
        return "-"
    return f"{float(x):.{nd}f}"


def img(path: Path, report_dir: Path) -> str:
    if not path.exists():
        return f"<div class='missing'>missing: {html.escape(str(path))}</div>"
    return f"<img src='{html.escape(rel(path, report_dir))}' alt='{html.escape(path.stem)}'>"


def build_html(rows: list[dict], report_dir: Path, heatmaps: list[Path], runs_root: Path) -> None:
    css = """
    body{font-family:Inter,Segoe UI,Arial,sans-serif;margin:24px;color:#17202a;background:#fafafa}
    h1,h2,h3{margin:18px 0 10px} p{line-height:1.5}
    table{border-collapse:collapse;width:100%;font-size:13px;background:white}
    th,td{border:1px solid #d8dee8;padding:6px 8px;text-align:left}
    th{background:#eef2f7;position:sticky;top:0} tr.good{background:#edf8ef} tr.bad{background:#fff1f0}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}
    .card{background:white;border:1px solid #dde3ee;border-radius:8px;padding:14px}
    img{max-width:100%;border:1px solid #d8dee8;border-radius:6px;background:white}
    code{background:#eef2f7;padding:1px 4px;border-radius:4px}.small{font-size:12px;color:#5d6d7e}
    .missing{padding:12px;background:#f4f4f4;color:#777;border-radius:6px}
    """
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Phase7 Strategy Suite Report</title>",
        f"<style>{css}</style></head><body>",
        "<h1>Phase7 Strategy Suite Report</h1>",
        "<p>This suite tests whether high-dimensional B-spline RL degradation is caused by exploration, joint PPO ratio scaling, missing init tethering, or control-grid size.</p>",
        f"<p class='small'>Runs root: <code>{html.escape(str(runs_root))}</code></p>",
        "<h2>Strategies</h2><table><tr><th>strategy</th><th>description</th></tr>",
    ]
    for s in STRATEGIES:
        parts.append(f"<tr><td><code>{html.escape(s)}</code></td><td>{html.escape(STRATEGY_TEXT[s])}</td></tr>")
    parts.append("</table>")

    complete = sum(1 for r in rows if r["status"] == "complete")
    improved = [r for r in rows if r["status"] == "complete" and math.isfinite(float(r["delta_best_vs_init"])) and float(r["delta_best_vs_init"]) < 0]
    parts.append("<h2>Summary</h2>")
    parts.append(f"<p>Completed runs: <b>{complete}/{len(rows)}</b>. Runs improving over their deterministic initial model: <b>{len(improved)}</b>.</p>")

    parts.append("<h3>Best Per Model / Init</h3><table><tr><th>model</th><th>init</th><th>strategy</th><th>reward</th><th>initial MAE</th><th>best MAE</th><th>delta</th></tr>")
    for model in MODELS:
        for init in INITS:
            sub = [r for r in rows if r["model"] == model and r["init"] == init and r["status"] == "complete" and math.isfinite(float(r["best_mae"]))]
            if not sub:
                continue
            best = min(sub, key=lambda r: float(r["best_mae"]))
            cls = "good" if float(best["delta_best_vs_init"]) < 0 else "bad"
            parts.append(
                f"<tr class='{cls}'><td>{model}</td><td>{init}</td><td>{best['strategy']}</td><td>{best['reward']}</td>"
                f"<td>{fmt(best['initial_mae'])}</td><td>{fmt(best['best_mae'])}</td><td>{fmt(best['delta_best_vs_init'])}</td></tr>"
            )
    parts.append("</table>")

    parts.append("<h2>Delta Heatmaps</h2><p>Negative numbers are good: training beat the supplied initial model.</p><div class='grid'>")
    for path in heatmaps:
        parts.append("<div class='card'>" + img(path, report_dir) + "</div>")
    parts.append("</div>")

    parts.append("<h2>All Runs</h2><table><tr><th>strategy</th><th>model</th><th>init</th><th>reward</th><th>status</th><th>ctrl</th><th>steps</th><th>initial</th><th>best</th><th>final</th><th>delta best</th><th>ratio std</th><th>clip</th><th>run</th></tr>")
    for r in rows:
        cls = "good" if r["status"] == "complete" and math.isfinite(float(r["delta_best_vs_init"])) and float(r["delta_best_vs_init"]) < 0 else "bad"
        run_dir = Path(r["run_dir"])
        vis = run_dir / "phase7_visuals" / "models.png"
        run_link = f"<a href='{html.escape(rel(vis, report_dir))}'>models</a>" if vis.exists() else html.escape(str(run_dir))
        parts.append(
            f"<tr class='{cls}'><td>{r['strategy']}</td><td>{r['model']}</td><td>{r['init']}</td><td>{r['reward']}</td><td>{r['status']}</td>"
            f"<td>{r['nx_ctrl']}x{r['nz_ctrl']}</td><td>{r['steps_done']}</td><td>{fmt(r['initial_mae'])}</td><td>{fmt(r['best_mae'])}</td>"
            f"<td>{fmt(r['final_oracle_mae'])}</td><td>{fmt(r['delta_best_vs_init'])}</td><td>{fmt(r['ratio_std'],3)}</td><td>{fmt(r['clip_frac'],3)}</td><td>{run_link}</td></tr>"
        )
    parts.append("</table>")
    parts.append("</body></html>")
    with (report_dir / "index.html").open("w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase7 strategy-suite HTML report")
    parser.add_argument("--runs_root", default=str(ROOT / "runs" / "phase7_strategy"))
    parser.add_argument("--report_dir", default=str(ROOT / "reports" / "phase7_strategy_suite"))
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    report_dir = Path(args.report_dir)
    rows = collect_runs(runs_root)
    save_tables(rows, report_dir)
    heatmaps = plot_strategy_heatmaps(rows, report_dir)
    build_html(rows, report_dir, heatmaps, runs_root)
    print(f"[ok] report: {report_dir / 'index.html'}")


if __name__ == "__main__":
    main()
