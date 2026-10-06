"""Screen a mixed-Q alternating BCQ configuration on WikiText-2."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_bcq_gpt2_checkpoint import copy_conventional_weight
from evaluate_quantization_baseline import ErrorStatistics
from extract_gpt2_weights import alternating_bcq, sha256_file
from sweep_alternating_bcq import (
    MATRIX_NAMES,
    evaluate_model,
    original_target_weights,
    prepare_validation_data,
    restore_target_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("outputs/cpu-smoke-test-002"))
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=Path("BCQ/artifacts/cpu-smoke-test-002-q3/reference_w_true.safetensors"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("BCQ/sweeps/mixed-q6-q8-alternating-50"),
    )
    parser.add_argument("--base-q", type=int, default=6)
    parser.add_argument("--sensitive-q", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def matrix_q(layer_index: int, matrix_name: str, base_q: int, sensitive_q: int) -> int:
    if layer_index == 0 or matrix_name in {"W_O", "W_PROJ"}:
        return sensitive_q
    return base_q


@torch.inference_mode()
def quantize_and_inject_mixed(
    model: torch.nn.Module,
    reference_file: object,
    base_q: int,
    sensitive_q: int,
    max_iterations: int,
) -> tuple[dict[str, float | int], dict[str, dict[str, int]], float]:
    statistics = ErrorStatistics()
    matrix_settings: dict[str, dict[str, int]] = {}
    total_binary_values = 0
    total_weights = 0

    for layer_index, block in enumerate(model.transformer.h):
        prefix = f"layer_{layer_index:02d}"
        reconstructed: dict[str, torch.Tensor] = {}
        for matrix_name in MATRIX_NAMES:
            artifact_name = f"{prefix}.{matrix_name}"
            reference = reference_file.get_tensor(artifact_name)
            q = matrix_q(layer_index, matrix_name, base_q, sensitive_q)
            _, _, _, bcq_weight, iterations = alternating_bcq(
                reference,
                q,
                max_iterations=max_iterations,
            )
            reconstructed[matrix_name] = bcq_weight
            statistics.update(reference, bcq_weight)
            matrix_settings[artifact_name] = {"q": q, "iterations": iterations}
            total_binary_values += q * reference.numel()
            total_weights += reference.numel()

        qkv = torch.cat(
            [reconstructed["W_Q"], reconstructed["W_K"], reconstructed["W_V"]],
            dim=0,
        )
        copy_conventional_weight(block.attn.c_attn.weight, qkv, f"{prefix}.W_QKV")
        copy_conventional_weight(block.attn.c_proj.weight, reconstructed["W_O"], f"{prefix}.W_O")
        copy_conventional_weight(block.mlp.c_fc.weight, reconstructed["W_FC"], f"{prefix}.W_FC")
        copy_conventional_weight(block.mlp.c_proj.weight, reconstructed["W_PROJ"], f"{prefix}.W_PROJ")

    return statistics.metrics(), matrix_settings, total_binary_values / total_weights


def main() -> None:
    args = parse_args()
    if min(args.base_q, args.sensitive_q, args.max_iterations, args.eval_samples) < 1:
        raise ValueError("Q, iterations, and eval samples must be at least 1")
    if args.sensitive_q <= args.base_q:
        raise ValueError("--sensitive-q must be larger than --base-q")

    model_path = args.model_path.expanduser().resolve()
    reference_path = args.reference_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not model_path.is_dir() or not reference_path.is_file():
        raise FileNotFoundError(f"Missing model or reference: {model_path}, {reference_path}")
    if output_dir.is_dir() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new --output-dir or pass --overwrite explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    dataset = prepare_validation_data(tokenizer, args.eval_samples)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    model.eval()
    original = original_target_weights(model)
    control_metrics = evaluate_model(model, dataset)

    with safe_open(reference_path, framework="pt", device="cpu") as reference_file:
        weight_metrics, matrix_settings, effective_q = quantize_and_inject_mixed(
            model,
            reference_file,
            args.base_q,
            args.sensitive_q,
            args.max_iterations,
        )
    functional_metrics = evaluate_model(model, dataset)
    restore_target_weights(model, original)
    for layer_index, block in enumerate(model.transformer.h):
        assert torch.equal(block.attn.c_attn.weight, original[layer_index]["qkv"])
        assert torch.equal(block.attn.c_proj.weight, original[layer_index]["o"])
        assert torch.equal(block.mlp.c_fc.weight, original[layer_index]["fc"])
        assert torch.equal(block.mlp.c_proj.weight, original[layer_index]["proj"])

    implementation_path = Path(__file__).resolve()
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "method": "mixed_q_alternating",
        "configuration": {
            "base_q": args.base_q,
            "sensitive_q": args.sensitive_q,
            "sensitive_rule": "layer == 0 or matrix in {W_O, W_PROJ}",
            "effective_weighted_average_q": effective_q,
            "max_iterations": args.max_iterations,
        },
        "evaluation": {
            "dataset": "Salesforce/wikitext",
            "dataset_config": "wikitext-2-raw-v1",
            "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "split": "validation",
            "block_size": 1024,
            "samples": args.eval_samples,
            "device": "cpu",
        },
        "control": control_metrics,
        "candidate": {
            "weight_metrics": weight_metrics,
            "functional_metrics": functional_metrics,
            "accuracy_retention": functional_metrics["eval_accuracy"] / control_metrics["eval_accuracy"],
            "perplexity_ratio": functional_metrics["perplexity"] / control_metrics["perplexity"],
        },
        "matrix_settings": matrix_settings,
        "provenance": {
            "model_path": str(model_path),
            "reference_path": str(reference_path),
            "reference_sha256": sha256_file(reference_path),
            "script_sha256": sha256_file(implementation_path),
        },
        "restore_verification": "passed",
    }
    report_path = output_dir / "mixed_q_results.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(implementation_path, output_dir / "screen_mixed_q_bcq.snapshot.py")
    print(json.dumps(report["candidate"], indent=2), flush=True)
    print(f"Effective weighted-average Q: {effective_q:.6f}", flush=True)
    print(f"Saved mixed-Q report to {report_path}", flush=True)


if __name__ == "__main__":
    main()
