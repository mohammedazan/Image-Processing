#!/usr/bin/env python3
"""
compare_results.py

Comparative analysis for experiments:
- Collect metrics.json (or metrics.csv) from each experiment directory under --results_dir
- Create summary CSV, plots (accuracy vs time, multi-metric barplot)
- Create grid of confusion matrix images
- Produce an interpretation report (Markdown) with basic automated notes

Usage (debug / fast):
    python compare_results.py --results_dir ../experiments --out_dir ../reports --fast --max_exp 6 --verbose

Full run example:
    python compare_results.py --results_dir experiments \
                              --out_dir reports \
                              --verbose

Outputs:
- reports/comparison_plots/*.png
- reports/comparison_table.csv
- reports/comparison_notes.md

Author: generated
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import json
import csv
import sys
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("compare_results")

# -----------------------
# Utilities
# -----------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def find_experiment_dirs(results_dir: Path) -> List[Path]:
    """Return list of immediate subdirectories likely to contain model results."""
    if not results_dir.exists():
        logger.error(f"Results directory {results_dir} does not exist.")
        return []
    dirs = sorted([p for p in results_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    return dirs

def load_metrics_json(p: Path) -> Optional[Dict[str, Any]]:
    """Load metrics.json or metrics.csv (first row) if exists."""
    json_path = p / "metrics.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as e:
            logger.warning(f"Failed to read {json_path}: {e}")
    # fallback to metrics.csv
    csv_path = p / "metrics.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            # try to convert to dict: if columns metric/class/value
            if set(["metric", "value"]).issubset(df.columns):
                out = {}
                for _, row in df.iterrows():
                    m = row["metric"]
                    cls = row.get("class", "")
                    val = row["value"]
                    if cls and not pd.isna(cls):
                        out.setdefault(m, {})[str(cls)] = float(val)
                    else:
                        out[m] = float(val)
                return out
            else:
                # fallback: create simple dict using columns
                return {"csv_table": df.to_dict(orient="records")}
        except Exception as e:
            logger.warning(f"Failed to parse {csv_path}: {e}")
    return None

def find_confusion_matrix_image(p: Path) -> Optional[Path]:
    """Return path to confusion_matrix.png (or any png that looks like confusion matrix)."""
    candidates = list(p.glob("confusion_matrix.*")) + list(p.glob("*confusion*matrix*.png"))
    if candidates:
        return candidates[0]
    # find any png in dir as fallback
    imgs = list(p.glob("*.png"))
    return imgs[0] if imgs else None

# -----------------------
# Aggregation
# -----------------------
def aggregate_results(results_dir: Path, max_experiments: Optional[int] = None, fast: bool = False) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    exp_dirs = find_experiment_dirs(results_dir)
    if max_experiments is not None:
        exp_dirs = exp_dirs[:max_experiments]
    rows = []
    details = []
    for d in exp_dirs:
        name = d.name
        metrics = load_metrics_json(d)
        img = find_confusion_matrix_image(d)
        if metrics is None:
            logger.info(f"No metrics found in {d} -> skipping")
            continue
        # Normalize expected fields
        acc = metrics.get("accuracy", metrics.get("acc", np.nan))
        bal = metrics.get("balanced_accuracy", metrics.get("balanced_acc", np.nan))
        mf1 = metrics.get("macro_f1", metrics.get("macro_f1_score", np.nan))
        pred_time = metrics.get("prediction_time", metrics.get("prediction_seconds", np.nan))
        # If metrics stored nested (precision -> dict), attempt to compute mean precision/recall
        precision_map = metrics.get("precision", {})
        recall_map = metrics.get("recall", {})
        # convert to numeric arrays if dict
        try:
            if isinstance(precision_map, dict):
                mean_prec = np.nanmean([float(v) for v in precision_map.values()]) if precision_map else np.nan
            else:
                mean_prec = float(precision_map)
        except Exception:
            mean_prec = np.nan
        try:
            if isinstance(recall_map, dict):
                mean_rec = np.nanmean([float(v) for v in recall_map.values()]) if recall_map else np.nan
            else:
                mean_rec = float(recall_map)
        except Exception:
            mean_rec = np.nan

        rows.append({
            "experiment": name,
            "accuracy": float(acc) if pd.notna(acc) else np.nan,
            "balanced_accuracy": float(bal) if pd.notna(bal) else np.nan,
            "macro_f1": float(mf1) if pd.notna(mf1) else np.nan,
            "mean_precision": float(mean_prec) if pd.notna(mean_prec) else np.nan,
            "mean_recall": float(mean_rec) if pd.notna(mean_rec) else np.nan,
            "prediction_time": float(pred_time) if pd.notna(pred_time) else np.nan,
            "metrics_raw": metrics,
            "confusion_image": str(img) if img is not None else ""
        })
        details.append({"dir": str(d), "metrics": metrics, "confusion_image": str(img) if img else ""})
    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No experiment metrics aggregated.")
    # sort by accuracy desc (helpful)
    df = df.sort_values(by="accuracy", ascending=False).reset_index(drop=True)
    return df, details

# -----------------------
# Plotting
# -----------------------
def plot_accuracy_vs_time(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        logger.warning("Empty dataframe for accuracy vs time plot.")
        return
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    ax = sns.scatterplot(
        data=df,
        x="prediction_time",
        y="accuracy",
        hue="experiment",
        s=120,
        palette="tab10",
        legend=False
    )
    for i, row in df.iterrows():
        x = row["prediction_time"]
        y = row["accuracy"]
        if np.isfinite(x) and np.isfinite(y):
            ax.text(x, y, " " + str(row["experiment"]), fontsize=9, va="center")
    plt.xlabel("Prediction time (s)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Prediction Time")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Wrote {out_path}")

def plot_multi_metric_bars(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "mean_precision", "mean_recall"]
    plot_df = df.set_index("experiment")[metrics].fillna(0)
    plt.figure(figsize=(10, max(4, len(plot_df)*0.5)))
    plot_df.plot(kind="bar", rot=45)
    plt.ylabel("Score")
    plt.title("Comparison of metrics by experiment")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Wrote {out_path}")

def plot_confusion_grid(details: List[Dict[str, Any]], out_path: Path, per_row: int = 3) -> None:
    imgs = []
    titles = []
    for item in details:
        imgp = item.get("confusion_image", "")
        metrics = item.get("metrics", {})
        if imgp:
            try:
                im = Image.open(imgp).convert("RGBA")
                imgs.append(im)
                titles.append(Path(item["dir"]).name)
            except Exception as e:
                logger.debug(f"Failed loading image {imgp}: {e}")
                imgs.append(None)
                titles.append(Path(item["dir"]).name)
        else:
            imgs.append(None)
            titles.append(Path(item["dir"]).name)

    n = len(imgs)
    if n == 0:
        logger.warning("No confusion matrix images to plot.")
        return
    cols = per_row
    rows = int(np.ceil(n / cols))
    thumb_w = 240
    thumb_h = 180
    fig_w = min(16, cols * 4)
    fig_h = max(3, rows * 3)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    axes = axes.reshape(rows, cols)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            if idx < n and imgs[idx] is not None:
                im = imgs[idx].copy()
                im.thumbnail((thumb_w, thumb_h))
                ax.imshow(im)
                ax.set_title(titles[idx], fontsize=9)
            elif idx < n:
                ax.text(0.5, 0.5, titles[idx], ha='center', va='center', fontsize=10)
            idx += 1
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Wrote {out_path}")

# -----------------------
# Report generation
# -----------------------
def make_interpretation_notes(df: pd.DataFrame, details: List[Dict[str, Any]], out_md: Path) -> None:
    ensure_dir(out_md.parent)
    lines = []
    lines.append("# Comparative analysis report\n")
    lines.append(f"Generated: {pd.Timestamp.now()}\n")
    if df.empty:
        lines.append("No metrics available to analyze.\n")
        out_md.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Wrote {out_md}")
        return

    lines.append("## Summary table\n")
    lines.append("Top experiments by accuracy:\n")
    lines.append("\n")
    lines.append(df[["experiment", "accuracy", "balanced_accuracy", "macro_f1", "prediction_time"]].to_markdown(index=False))
    lines.append("\n---\n")

    # Best/Worst
    best = df.iloc[0]
    worst = df.iloc[-1]
    lines.append("## Highlights\n")
    lines.append(f"- Best experiment: **{best['experiment']}** — accuracy={best['accuracy']:.4f}, macro_f1={best['macro_f1']:.4f}, time={best['prediction_time']}\n")
    lines.append(f"- Worst experiment: **{worst['experiment']}** — accuracy={worst['accuracy']:.4f}, macro_f1={worst['macro_f1']:.4f}\n")

    # Check for degenerate model (predicted single class)
    degenerate = []
    for item in details:
        m = item.get("metrics", {})
        # heuristics: macro_f1 near 0 or accuracy equals recall of single class
        mf1 = m.get("macro_f1", None) or m.get("macro_f1_score", None)
        if mf1 is not None and mf1 < 1e-3:
            degenerate.append(Path(item["dir"]).name)
    if degenerate:
        lines.append("\n### Degenerate models detected\n")
        lines.append("- Models with near-zero macro F1 (likely predicting a single class):\n")
        for ex in degenerate:
            lines.append(f"  - {ex}\n")
        lines.append("\n")

    # Data imbalance indications
    lines.append("## Automated interpretation notes\n")
    lines.append("- If a model predicts a single class for most samples, macro-F1 will be close to 0 while accuracy depends on class distribution.\n")
    lines.append("- Balanced accuracy gives more insight when classes are imbalanced; prefer it when dataset is skewed.\n")
    lines.append("- Check confusion matrices (plots) for which classes are confused and whether a model always predicts the same class.\n")

    # Add per-experiment short notes (precision/recall per-class if present)
    lines.append("\n## Per-experiment quick notes\n")
    for _, row in df.iterrows():
        ex = row["experiment"]
        metrics = row["metrics_raw"]
        lines.append(f"### {ex}\n")
        acc = row.get("accuracy", float("nan"))
        mf1 = row.get("macro_f1", float("nan"))
        lines.append(f"- accuracy={acc:.4f}, macro_f1={mf1:.4f}\n")
        # show top-3 high/low precision classes if present
        prec = metrics.get("precision", {})
        rec = metrics.get("recall", {})
        if isinstance(prec, dict) and prec:
            try:
                # sort by value
                sorted_prec = sorted(prec.items(), key=lambda t: float(t[1]), reverse=True)
                lines.append("- Top precision classes: " + ", ".join([f"{k}({v:.2f})" for k, v in sorted_prec[:3]]) + "\n")
            except Exception:
                pass
        if isinstance(rec, dict) and rec:
            try:
                sorted_rec = sorted(rec.items(), key=lambda t: float(t[1]), reverse=True)
                lines.append("- Top recall classes: " + ", ".join([f"{k}({v:.2f})" for k, v in sorted_rec[:3]]) + "\n")
            except Exception:
                pass
        lines.append("\n")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote {out_md}")

# -----------------------
# Main
# -----------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Comparative analysis of experiment results (D2).")
    p.add_argument("--results_dir", type=str, default="experiments", help="Directory with experiment subfolders")
    p.add_argument("--out_dir", type=str, default="reports", help="Output directory for plots and report")
    p.add_argument("--max_exp", type=int, default=None, help="Limit number of experiments processed (for debug)")
    p.add_argument("--fast", action="store_true", help="Fast/debug mode: limit to small set automatically")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    plots_dir = out_dir / "comparison_plots"
    ensure_dir(plots_dir)

    if args.fast:
        max_experiments = args.max_exp if args.max_exp is not None else 8
    else:
        max_experiments = args.max_exp

    logger.info(f"Reading experiments from {results_dir} (max={max_experiments})")
    df, details = aggregate_results(results_dir, max_experiments=max_experiments, fast=args.fast)
    # Save aggregated table
    table_path = out_dir / "comparison_table.csv"
    df_to_save = df.drop(columns=["metrics_raw", "confusion_image"])
    df_to_save.to_csv(table_path, index=False)
    logger.info(f"Wrote aggregated table -> {table_path}")

    # Plots
    plot_accuracy_vs_time(df, plots_dir / "accuracy_vs_time.png")
    plot_multi_metric_bars(df, plots_dir / "metrics_barplot.png")
    plot_confusion_grid(details, plots_dir / "confusion_grid.png", per_row=3)

    # Report markdown
    md_path = out_dir / "comparison_notes.md"
    make_interpretation_notes(df, details, md_path)

    logger.info("All done. Outputs:")
    logger.info(f"- aggregated table: {table_path}")
    logger.info(f"- plots: {plots_dir}")
    logger.info(f"- report: {md_path}")

if __name__ == "__main__":
    main()
