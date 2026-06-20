"""Build an HTML report for Phase7 multiscale B-spline optimization experiments."""

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
INITS = ["far_uniform", "medium_gradient"]
PIPELINES = ["ms_tt", "ms_tt_ncc", "ms_tt_w2", "ms_ncc", "ms_tt_ncc_w2"]

PIPELINE_TEXT = {
    "ms_tt": "coarse tt_only -> mid tt_only -> full tt_only",
    "ms_tt_ncc": "coarse tt_only -> mid tt_only -> full ncc_zero",
    "ms_tt_w2": "coarse tt_only -> mid tt_only -> full wasserstein_w2",
    "ms_ncc": "coarse ncc_zero -> mid ncc_zero -> full ncc_zero",
    "ms_tt_ncc_w2": "coarse tt_only -> mid ncc_zero -> full wasserstein_w2",
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


def stage_dirs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    dirs = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("stage")]
    return sorted(dirs, key=lambda p: p.name)


def collect(runs_root: Path) -> list[dict]:
    rows: list[dict] = []
    for pipeline in PIPELINES:
        for model in MODELS:
            for init in INITS:
                base = runs_root / pipeline / model / init
                stages = []
                original_initial_mae = math.nan
                pipeline_best_mae = math.inf
                pipeline_best_stage = ""
                final_mae = math.nan
                final_stage = ""
                complete_stages = 0
                for sd in stage_dirs(base):
                    cfg = read_json(sd / "config.json")
                    metrics = read_csv(sd / "metrics.csv")
                    last = metrics[-1] if metrics else {}
                    status = "complete" if (sd / "policy_final.pt").exists() and (sd / "final_velocity.npy").exists() else "missing"
                    if metrics and status == "missing":
                        status = "partial"
                    if status == "complete":
                        complete_stages += 1
                    if math.isnan(original_initial_mae):
                        original_initial_mae = fnum(last.get("initial_model_mae"))
                    best_mae = fnum(last.get("best_mae_global"))
                    if math.isfinite(best_mae) and best_mae < pipeline_best_mae:
                        pipeline_best_mae = best_mae
                        pipeline_best_stage = sd.name
                    final_mae = fnum(last.get("mae_oracle_best"))
                    final_stage = sd.name
                    stages.append(
                        {
                            "stage": sd.name,
                            "status": status,
                            "steps": int(fnum(last.get("step"), 0)),
                            "initial_mae": fnum(last.get("initial_model_mae")),
                            "best_mae": best_mae,
                            "final_mae": final_mae,
                            "delta_vs_stage_init": fnum(last.get("delta_best_vs_init"), best_mae - fnum(last.get("initial_model_mae"))),
                            "nx_ctrl": cfg.get("nx_ctrl", ""),
                            "nz_ctrl": cfg.get("nz_ctrl", ""),
                        }
                    )
                status = "complete" if complete_stages == 3 else ("partial" if complete_stages > 0 else "missing")
                delta_best = pipeline_best_mae - original_initial_mae if math.isfinite(pipeline_best_mae) and math.isfinite(original_initial_mae) else math.nan
                delta_final = final_mae - original_initial_mae if math.isfinite(final_mae) and math.isfinite(original_initial_mae) else math.nan
                rows.append(
                    {
                        "pipeline": pipeline,
                        "model": model,
                        "init": init,
                        "status": status,
                        "complete_stages": complete_stages,
                        "original_initial_mae": original_initial_mae,
                        "pipeline_best_mae": pipeline_best_mae if math.isfinite(pipeline_best_mae) else math.nan,
                        "pipeline_best_stage": pipeline_best_stage,
                        "final_stage": final_stage,
                        "final_mae": final_mae,
                        "delta_best_vs_original": delta_best,
                        "delta_final_vs_original": delta_final,
                        "stages": stages,
                        "run_dir": str(base),
                    }
                )
    return rows


def rel(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve()).replace("\\", "/")


def save(rows: list[dict], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    flat = []
    for r in rows:
        item = {k: v for k, v in r.items() if k != "stages"}
        item["stages_json"] = json.dumps(r["stages"], ensure_ascii=False)
        flat.append(item)
    with (report_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        if flat:
            writer = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            writer.writeheader()
            writer.writerows(flat)


def plot_heatmaps(rows: list[dict], report_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_paths = []
    for model in MODELS:
        for init in INITS:
            sub = [r for r in rows if r["model"] == model and r["init"] == init]
            arr = np.full((len(PIPELINES), 2), np.nan, dtype=np.float32)
            for r in sub:
                i = PIPELINES.index(r["pipeline"])
                arr[i, 0] = float(r["delta_best_vs_original"]) if math.isfinite(float(r["delta_best_vs_original"])) else np.nan
                arr[i, 1] = float(r["delta_final_vs_original"]) if math.isfinite(float(r["delta_final_vs_original"])) else np.nan
            fig, ax = plt.subplots(figsize=(6, 4.8))
            lim = float(np.nanpercentile(np.abs(arr), 95)) if np.isfinite(arr).any() else 1.0
            lim = max(lim, 1.0)
            im = ax.imshow(arr, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            ax.set_xticks([0, 1], ["best", "final"])
            ax.set_yticks(np.arange(len(PIPELINES)), PIPELINES)
            ax.set_title(f"{model} / {init}: multiscale delta vs original init")
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    val = arr[i, j]
                    ax.text(j, i, "-" if np.isnan(val) else f"{val:+.0f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, label="MAE delta (m/s), negative is improvement")
            fig.tight_layout()
            out = report_dir / f"delta_{model}_{init}.png"
            fig.savefig(out, dpi=170)
            plt.close(fig)
            out_paths.append(out)
    return out_paths


def fmt(x, nd=1) -> str:
    try:
        x = float(x)
    except Exception:
        return "-"
    if not math.isfinite(x):
        return "-"
    return f"{x:.{nd}f}"


def img(path: Path, report_dir: Path) -> str:
    if not path.exists():
        return f"<div class='missing'>missing: {html.escape(str(path))}</div>"
    return f"<img src='{html.escape(rel(path, report_dir))}' alt='{html.escape(path.stem)}'>"


def build_html(rows: list[dict], report_dir: Path, heatmaps: list[Path], runs_root: Path) -> None:
    css = """
    body{font-family:Inter,Segoe UI,Arial,sans-serif;margin:24px;color:#17202a;background:#fafafa}
    table{border-collapse:collapse;width:100%;font-size:13px;background:white}
    th,td{border:1px solid #d8dee8;padding:6px 8px;text-align:left}
    th{background:#eef2f7;position:sticky;top:0} tr.good{background:#edf8ef} tr.bad{background:#fff1f0}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}
    .card{background:white;border:1px solid #dde3ee;border-radius:8px;padding:14px}
    img{max-width:100%;border:1px solid #d8dee8;border-radius:6px;background:white}
    code{background:#eef2f7;padding:1px 4px;border-radius:4px}.small{font-size:12px;color:#5d6d7e}
    """
    complete = sum(1 for r in rows if r["status"] == "complete")
    improved = [r for r in rows if r["status"] == "complete" and math.isfinite(float(r["delta_best_vs_original"])) and float(r["delta_best_vs_original"]) < 0]
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Phase7 Multiscale Suite Report</title>",
        f"<style>{css}</style></head><body>",
        "<h1>Phase7 Multiscale Suite Report</h1>",
        "<p>Coarse-to-mid-to-full B-spline optimization for general initial models.</p>",
        f"<p class='small'>Runs root: <code>{html.escape(str(runs_root))}</code></p>",
        f"<p>Completed pipelines: <b>{complete}/{len(rows)}</b>. Improved over original initial model: <b>{len(improved)}</b>.</p>",
        "<h2>Pipelines</h2><table><tr><th>pipeline</th><th>schedule</th></tr>",
    ]
    for p, text in PIPELINE_TEXT.items():
        parts.append(f"<tr><td><code>{p}</code></td><td>{html.escape(text)}</td></tr>")
    parts.append("</table>")

    parts.append("<h2>Best Per Model / Init</h2><table><tr><th>model</th><th>init</th><th>pipeline</th><th>initial</th><th>best</th><th>final</th><th>delta best</th><th>best stage</th></tr>")
    for model in MODELS:
        for init in INITS:
            sub = [r for r in rows if r["model"] == model and r["init"] == init and r["status"] == "complete"]
            if not sub:
                continue
            best = min(sub, key=lambda r: float(r["pipeline_best_mae"]))
            cls = "good" if float(best["delta_best_vs_original"]) < 0 else "bad"
            parts.append(
                f"<tr class='{cls}'><td>{model}</td><td>{init}</td><td>{best['pipeline']}</td><td>{fmt(best['original_initial_mae'])}</td>"
                f"<td>{fmt(best['pipeline_best_mae'])}</td><td>{fmt(best['final_mae'])}</td><td>{fmt(best['delta_best_vs_original'])}</td><td>{best['pipeline_best_stage']}</td></tr>"
            )
    parts.append("</table>")

    parts.append("<h2>Delta Heatmaps</h2><div class='grid'>")
    for path in heatmaps:
        parts.append("<div class='card'>" + img(path, report_dir) + "</div>")
    parts.append("</div>")

    parts.append("<h2>All Pipelines</h2><table><tr><th>pipeline</th><th>model</th><th>init</th><th>status</th><th>stages</th><th>initial</th><th>best</th><th>final</th><th>delta best</th><th>delta final</th><th>run</th></tr>")
    for r in rows:
        cls = "good" if r["status"] == "complete" and math.isfinite(float(r["delta_best_vs_original"])) and float(r["delta_best_vs_original"]) < 0 else "bad"
        final_vis = Path(r["run_dir"]) / str(r["final_stage"]) / "phase7_visuals" / "models.png"
        run_cell = f"<a href='{html.escape(rel(final_vis, report_dir))}'>final models</a>" if final_vis.exists() else html.escape(r["run_dir"])
        parts.append(
            f"<tr class='{cls}'><td>{r['pipeline']}</td><td>{r['model']}</td><td>{r['init']}</td><td>{r['status']}</td>"
            f"<td>{r['complete_stages']}</td><td>{fmt(r['original_initial_mae'])}</td><td>{fmt(r['pipeline_best_mae'])}</td><td>{fmt(r['final_mae'])}</td>"
            f"<td>{fmt(r['delta_best_vs_original'])}</td><td>{fmt(r['delta_final_vs_original'])}</td><td>{run_cell}</td></tr>"
        )
    parts.append("</table></body></html>")
    with (report_dir / "index.html").open("w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase7 multiscale-suite HTML report")
    parser.add_argument("--runs_root", default=str(ROOT / "runs" / "phase7_multiscale"))
    parser.add_argument("--report_dir", default=str(ROOT / "reports" / "phase7_multiscale_suite"))
    args = parser.parse_args()
    runs_root = Path(args.runs_root)
    report_dir = Path(args.report_dir)
    rows = collect(runs_root)
    save(rows, report_dir)
    heatmaps = plot_heatmaps(rows, report_dir)
    build_html(rows, report_dir, heatmaps, runs_root)
    print(f"[ok] report: {report_dir / 'index.html'}")


if __name__ == "__main__":
    main()
