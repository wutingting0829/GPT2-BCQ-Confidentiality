"""Evaluate how accurately BCQ represents extracted GPT-2 weights."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open

from extract_gpt2_weights import greedy_residual_bcq, sha256_file


@dataclass
class ErrorStatistics:
    elements: int = 0
    squared_error: float = 0.0
    absolute_error: float = 0.0
    max_absolute_error: float = 0.0
    signal_energy: float = 0.0
    reconstruction_energy: float = 0.0
    dot_product: float = 0.0

    def update(self, reference: torch.Tensor, reconstruction: torch.Tensor) -> None:
        if reference.shape != reconstruction.shape:
            raise ValueError(
                f"Shape mismatch: reference={tuple(reference.shape)}, "
                f"reconstruction={tuple(reconstruction.shape)}"
            )
        if not torch.isfinite(reference).all() or not torch.isfinite(reconstruction).all():
            raise ValueError("Metrics require finite reference and reconstruction tensors")

        error = reconstruction.double() - reference.double()
        reference64 = reference.double()
        reconstruction64 = reconstruction.double()
        self.elements += reference.numel()
        self.squared_error += error.square().sum().item()
        self.absolute_error += error.abs().sum().item()
        self.max_absolute_error = max(self.max_absolute_error, error.abs().max().item())
        self.signal_energy += reference64.square().sum().item()
        self.reconstruction_energy += reconstruction64.square().sum().item()
        self.dot_product += (reference64 * reconstruction64).sum().item()

    def metrics(self) -> dict[str, float | int]:
        if self.elements == 0 or self.signal_energy == 0.0:
            raise ValueError("Cannot compute metrics for an empty or all-zero reference")

        mse = self.squared_error / self.elements
        cosine_denominator = math.sqrt(self.signal_energy * self.reconstruction_energy)
        return {
            "elements": self.elements,
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": self.absolute_error / self.elements,
            "max_abs_error": self.max_absolute_error,
            "relative_frobenius_error": math.sqrt(self.squared_error / self.signal_energy),
            "cosine_similarity": self.dot_product / cosine_denominator,
            "sqnr_db": 10.0 * math.log10(self.signal_energy / self.squared_error),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("BCQ/artifacts/cpu-smoke-test-002-q3"),
        help="Directory containing reference, baseline, and manifest artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Report directory (default: ARTIFACT_DIR/quantization_baseline).",
    )
    parser.add_argument(
        "--q-values",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Values of Q to evaluate (default: 1 2 3 4).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing non-empty report directory.",
    )
    return parser.parse_args()


def markdown_table(summary_by_q: dict[int, dict[str, float | int]]) -> str:
    rows = [
        "| Q | Relative Frobenius error | Cosine similarity | SQNR (dB) | RMSE | MAE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for q, metrics in sorted(summary_by_q.items()):
        rows.append(
            f"| {q} | {metrics['relative_frobenius_error']:.6f} "
            f"| {metrics['cosine_similarity']:.6f} | {metrics['sqnr_db']:.3f} "
            f"| {metrics['rmse']:.6f} | {metrics['mae']:.6f} |"
        )
    return "\n".join(rows)


def markdown_type_table(summary_by_type: dict[str, dict[str, float | int]]) -> str:
    rows = [
        "| Matrix type | Relative Frobenius error | Cosine similarity | SQNR (dB) |",
        "|---|---:|---:|---:|",
    ]
    for matrix_type, metrics in sorted(summary_by_type.items()):
        rows.append(
            f"| {matrix_type} | {metrics['relative_frobenius_error']:.6f} "
            f"| {metrics['cosine_similarity']:.6f} | {metrics['sqnr_db']:.3f} |"
        )
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    q_values = sorted(set(args.q_values))
    if not q_values or q_values[0] < 1:
        raise ValueError(f"Every Q must be at least 1, got {args.q_values}")

    artifact_dir = args.artifact_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else artifact_dir / "quantization_baseline"
    )
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    if output_dir.is_dir() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new --output-dir or pass --overwrite explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_path = artifact_dir / manifest["artifacts"]["reference_w_true"]["file"]
    baseline_path = artifact_dir / manifest["artifacts"]["baseline_w_bcq"]["file"]
    artifact_q = int(manifest["bcq"]["q"])

    aggregate_statistics = {q: ErrorStatistics() for q in q_values}
    type_statistics = {q: defaultdict(ErrorStatistics) for q in q_values}
    matrix_rows: list[dict[str, float | int | str]] = []
    saved_baseline_max_difference = 0.0

    with (
        safe_open(reference_path, framework="pt", device="cpu") as reference_file,
        safe_open(baseline_path, framework="pt", device="cpu") as baseline_file,
    ):
        matrix_names = sorted(reference_file.keys())
        if matrix_names != sorted(baseline_file.keys()):
            raise ValueError("W_true and saved W_BCQ contain different matrix names")

        for matrix_name in matrix_names:
            reference = reference_file.get_tensor(matrix_name)
            matrix_type = matrix_name.rsplit(".", maxsplit=1)[-1]
            layer = int(matrix_name.split(".", maxsplit=1)[0].removeprefix("layer_"))

            for q in q_values:
                _, _, _, reconstruction = greedy_residual_bcq(reference, q)
                statistics = ErrorStatistics()
                statistics.update(reference, reconstruction)
                metrics = statistics.metrics()
                aggregate_statistics[q].update(reference, reconstruction)
                type_statistics[q][matrix_type].update(reference, reconstruction)
                matrix_rows.append(
                    {
                        "q": q,
                        "layer": layer,
                        "matrix": matrix_name,
                        "matrix_type": matrix_type,
                        "rows": reference.shape[0],
                        "columns": reference.shape[1],
                        **metrics,
                    }
                )

                if q == artifact_q:
                    saved_baseline = baseline_file.get_tensor(matrix_name)
                    difference = (saved_baseline - reconstruction).abs().max().item()
                    saved_baseline_max_difference = max(saved_baseline_max_difference, difference)

    summary_by_q = {q: statistics.metrics() for q, statistics in aggregate_statistics.items()}
    summary_by_type = {
        str(q): {
            matrix_type: statistics.metrics()
            for matrix_type, statistics in sorted(statistics_by_type.items())
        }
        for q, statistics_by_type in type_statistics.items()
    }
    if artifact_q in q_values and saved_baseline_max_difference != 0.0:
        raise ValueError(
            f"Recomputed Q={artifact_q} BCQ differs from the saved baseline; "
            f"max difference={saved_baseline_max_difference}"
        )

    csv_path = output_dir / "matrix_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(matrix_rows[0]))
        writer.writeheader()
        writer.writerows(matrix_rows)

    implementation_path = Path(__file__).resolve()
    quantizer_path = implementation_path.with_name("extract_gpt2_weights.py")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "claim_scope": "weight_level_representation_only",
        "artifact_dir": str(artifact_dir),
        "q_values": q_values,
        "matrix_count": len(matrix_names),
        "saved_baseline_q": artifact_q,
        "saved_baseline_max_recomputation_difference": saved_baseline_max_difference,
        "source": {
            "transformers_commit": manifest["source_repository"]["transformers_commit"],
            "checkpoint_files": manifest["source_weight_files"],
            "reference_w_true_sha256": sha256_file(reference_path),
            "baseline_w_bcq_sha256": sha256_file(baseline_path),
            "quantizer_script_sha256": sha256_file(quantizer_path),
            "evaluator_script_sha256": sha256_file(implementation_path),
        },
        "metric_definitions": {
            "relative_frobenius_error": "||W_true - W_BCQ||_F / ||W_true||_F (lower is better)",
            "cosine_similarity": "<W_true, W_BCQ> / (||W_true||_F ||W_BCQ||_F) (higher is better)",
            "sqnr_db": "10 log10(||W_true||_F^2 / ||W_true - W_BCQ||_F^2) (higher is better)",
        },
        "aggregate_by_q": {str(q): metrics for q, metrics in summary_by_q.items()},
        "aggregate_by_q_and_matrix_type": summary_by_type,
    }
    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "# BCQ Weight-Level Quantization Baseline\n\n"
        "This report measures how well BCQ represents the extracted GPT-2 block weights. "
        "It does not establish functional language-model utility.\n\n"
        f"- Matrices: {len(matrix_names)}\n"
        f"- Convention: `W_true[out_features, in_features]`\n"
        f"- Saved Q={artifact_q} baseline recomputation max difference: "
        f"`{saved_baseline_max_difference}`\n\n"
        "## Aggregate Results\n\n"
        f"{markdown_table(summary_by_q)}\n\n"
        f"## Q={artifact_q} Results by Matrix Type\n\n"
        f"{markdown_type_table(summary_by_type[str(artifact_q)])}\n\n"
        "Detailed per-matrix values are in `matrix_metrics.csv`; machine-readable aggregate "
        "values and provenance are in `summary.json`.\n",
        encoding="utf-8",
    )
    shutil.copyfile(implementation_path, output_dir / "evaluate_quantization_baseline.snapshot.py")

    print(markdown_table(summary_by_q))
    print(f"\nSaved baseline report to {output_dir}")


if __name__ == "__main__":
    main()
