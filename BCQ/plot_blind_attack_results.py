"""Plot weight, key-recovery, and functional metrics for blind B*-only attacks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


DEFAULT_ROOT = Path("BCQ/attacks/cpu-smoke-test-002-mixed-q6-q8-activation-aware/blind-bstar-only")
METHODS = (
    ("base_only_control", "Base control"),
    ("a2_public_order", "A2"),
    ("a3_structural_key_search", "A3"),
    ("a4_synthetic_prior", "A4"),
    ("a5_random_base_projection", "A5"),
)
COLORS = ("#334155", "#2563EB", "#DC2626", "#D97706", "#059669")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_ROOT / "evaluation/attack_results.json")
    parser.add_argument("--summary", type=Path, default=DEFAULT_ROOT / "summary.json")
    parser.add_argument(
        "--randomization-validation",
        type=Path,
        default=DEFAULT_ROOT / "secret/randomization_validation.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "figures")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "axes.titlecolor": "#0F172A",
            "axes.titleweight": "bold",
            "font.size": 9,
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def finish_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", alpha=0.9)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def annotate_bars(
    axis: plt.Axes,
    bars,
    formatter,
    *,
    logarithmic: bool = False,
) -> None:
    for bar in bars:
        value = bar.get_height()
        if logarithmic:
            y = value * 1.12
        else:
            lower, upper = axis.get_ylim()
            y = value + (upper - lower) * 0.025
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            formatter(value),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#334155",
        )


def save_figure(figure: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"{stem}.{suffix}", dpi=dpi)
    plt.close(figure)


def plot_weight_metrics(results: dict, output_dir: Path, dpi: int) -> None:
    labels = [label for _, label in METHODS]
    records = [results[method]["weight_metrics"] for method, _ in METHODS]
    panels = (
        ("Sign accuracy", "sign_accuracy", lambda value: value * 100, "Percent", "{:.1f}%"),
        ("Mean absolute error", "mae", lambda value: value, "MAE", "{:.4f}"),
        ("Normalized RMSE", "nrmse", lambda value: value, "NRMSE", "{:.3f}"),
        (
            "Absolute-weight ranking",
            "abs_weight_spearman_sampled",
            lambda value: value,
            "Spearman rho",
            "{:.3f}",
        ),
        (
            "Top-10% magnitude overlap",
            "top_10_percent_abs_overlap",
            lambda value: value * 100,
            "Percent",
            "{:.1f}%",
        ),
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.2))
    axes_flat = axes.ravel()
    for axis, (title, key, transform, ylabel, number_format) in zip(axes_flat, panels, strict=False):
        values = [transform(record[key]) for record in records]
        bars = axis.bar(labels, values, color=COLORS, width=0.72)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        if key == "sign_accuracy":
            axis.axhline(50, color="#64748B", linestyle="--", linewidth=1, label="Random: 50%")
            axis.set_ylim(0, 110)
            axis.legend(fontsize=8)
        elif key == "abs_weight_spearman_sampled":
            axis.axhline(0, color="#64748B", linewidth=1)
            axis.set_ylim(-1.08, 1.18)
        elif key == "top_10_percent_abs_overlap":
            axis.axhline(10, color="#64748B", linestyle="--", linewidth=1, label="Random: 10%")
            axis.set_ylim(0, 110)
            axis.legend(fontsize=8)
        else:
            axis.set_ylim(0, max(values) * 1.2)
        annotate_bars(axis, bars, lambda value, fmt=number_format: fmt.format(value))
        finish_axis(axis)
    axes_flat[-1].axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in COLORS]
    axes_flat[-1].legend(handles, labels, loc="center", fontsize=10, title="Evaluated model")
    figure.suptitle("Blind B*-Only Attack: Weight Reconstruction", fontsize=15, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(figure, output_dir, "01_weight_reconstruction", dpi)


def plot_key_recovery(report: dict, validation: dict, output_dir: Path, dpi: int) -> None:
    measured = report["a3_key_recovery"]
    permutation_chance = validation["row_count"] / validation["key_elements"]
    labels = ["Sign S", "Permutation pi", "Joint S/pi"]
    measured_values = [
        measured["sign_recovery_accuracy"] * 100,
        measured["permutation_position_accuracy"] * 100,
        measured["joint_s_pi_position_accuracy"] * 100,
    ]
    chance_values = [50, permutation_chance * 100, permutation_chance * 50]
    positions = range(len(labels))
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    width = 0.36
    measured_bars = axis.bar(
        [position - width / 2 for position in positions],
        measured_values,
        width,
        label="A3 measured",
        color="#DC2626",
    )
    chance_bars = axis.bar(
        [position + width / 2 for position in positions],
        chance_values,
        width,
        label="Random baseline",
        color="#94A3B8",
    )
    axis.set_xticks(list(positions), labels)
    axis.set_ylabel("Recovery accuracy (%)")
    axis.set_ylim(0, 62)
    axis.set_title("A3 Secret-Key Recovery")
    axis.legend()
    annotate_bars(axis, measured_bars, lambda value: f"{value:.2f}%")
    annotate_bars(axis, chance_bars, lambda value: f"{value:.2f}%")
    finish_axis(axis)
    figure.tight_layout()
    save_figure(figure, output_dir, "02_key_recovery", dpi)


def plot_functional_screening(report: dict, output_dir: Path, dpi: int) -> None:
    labels = ["W_true", *[label for _, label in METHODS]]
    control = report["w_true_screen_control"]
    records = [control, *[report["results"][method]["direct_functional"] for method, _ in METHODS]]
    colors = ("#111827", *COLORS)
    panels = (
        ("Next-token accuracy", "eval_accuracy", lambda value: value * 100, "Accuracy (%)", False),
        ("Validation loss", "eval_loss", lambda value: value, "Cross-entropy loss", False),
        ("Perplexity", "perplexity", lambda value: value, "PPL (log scale)", True),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, (title, key, transform, ylabel, logarithmic) in zip(axes, panels, strict=True):
        values = [transform(record[key]) for record in records]
        bars = axis.bar(labels, values, color=colors, width=0.72)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=35)
        if logarithmic:
            axis.set_yscale("log")
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            annotate_bars(axis, bars, lambda value: f"{value:,.0f}", logarithmic=True)
        elif key == "eval_accuracy":
            axis.set_ylim(0, max(values) * 1.18)
            annotate_bars(axis, bars, lambda value: f"{value:.2f}%")
        else:
            axis.set_ylim(0, max(values) * 1.18)
            annotate_bars(axis, bars, lambda value: f"{value:.2f}")
        finish_axis(axis)
    figure.suptitle(
        "Blind B*-Only Attack: WikiText-2 Functional Screening (16 Blocks)",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(figure, output_dir, "03_functional_screening", dpi)


def plot_official_confirmation(summary: dict, output_dir: Path, dpi: int) -> None:
    result = summary["official_128_block_best_result"]
    control = result["w_true_control"]
    labels = ["W_true", "Best blind attack (A2)"]
    records = [control, result]
    panels = (
        ("Next-token accuracy", "eval_accuracy", lambda value: value * 100, "Accuracy (%)", False),
        ("Validation loss", "eval_loss", lambda value: value, "Cross-entropy loss", False),
        ("Perplexity", "perplexity", lambda value: value, "PPL (log scale)", True),
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.8, 4.5))
    for axis, (title, key, transform, ylabel, logarithmic) in zip(axes, panels, strict=True):
        values = [transform(record[key]) for record in records]
        bars = axis.bar(labels, values, color=("#111827", "#2563EB"), width=0.64)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=18)
        if logarithmic:
            axis.set_yscale("log")
            annotate_bars(axis, bars, lambda value: f"{value:,.0f}", logarithmic=True)
        elif key == "eval_accuracy":
            axis.set_ylim(0, max(values) * 1.18)
            annotate_bars(axis, bars, lambda value: f"{value:.2f}%")
        else:
            axis.set_ylim(0, max(values) * 1.18)
            annotate_bars(axis, bars, lambda value: f"{value:.2f}")
        finish_axis(axis)
    figure.suptitle("Official run_clm.py Confirmation (128 Blocks)", fontsize=15, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(figure, output_dir, "04_official_best_attack", dpi)


def write_functional_csv(report: dict, output_dir: Path) -> None:
    rows = [("w_true", report["w_true_screen_control"])]
    rows.extend((method, report["results"][method]["direct_functional"]) for method, _ in METHODS)
    with (output_dir / "functional_screening.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "eval_accuracy", "eval_loss", "perplexity"])
        for name, record in rows:
            writer.writerow([name, record["eval_accuracy"], record["eval_loss"], record["perplexity"]])


def main() -> None:
    args = parse_args()
    report = load_json(args.results.expanduser().resolve())
    summary = load_json(args.summary.expanduser().resolve())
    validation = load_json(args.randomization_validation.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    plot_weight_metrics(report["results"], output_dir, args.dpi)
    plot_key_recovery(report, validation, output_dir, args.dpi)
    plot_functional_screening(report, output_dir, args.dpi)
    plot_official_confirmation(summary, output_dir, args.dpi)
    write_functional_csv(report, output_dir)
    print(f"Saved PNG, PDF, and CSV figures to {output_dir}")


if __name__ == "__main__":
    main()
