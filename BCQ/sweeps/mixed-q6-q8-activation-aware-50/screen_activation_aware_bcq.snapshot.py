"""Screen diagonal activation-aware mixed-Q BCQ on WikiText-2."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from extract_gpt2_weights import sha256_file
from screen_mixed_q_bcq import quantize_and_inject_mixed
from sweep_alternating_bcq import (
    evaluate_model,
    original_target_weights,
    prepare_lm_data,
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
        default=Path("BCQ/sweeps/mixed-q6-q8-activation-aware-50"),
    )
    parser.add_argument("--base-q", type=int, default=6)
    parser.add_argument("--sensitive-q", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--calibration-samples", type=int, default=16)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def collect_input_second_moments(
    model: torch.nn.Module,
    dataset: object,
) -> dict[str, torch.Tensor]:
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles = []

    def register(module: torch.nn.Module, names: tuple[str, ...]) -> None:
        def hook(_module, inputs, _output):
            values = inputs[0].detach().double().reshape(-1, inputs[0].shape[-1])
            second_moment_sum = values.square().sum(dim=0).cpu()
            for name in names:
                if name not in sums:
                    sums[name] = second_moment_sum.clone()
                    counts[name] = values.shape[0]
                else:
                    sums[name] += second_moment_sum
                    counts[name] += values.shape[0]

        handles.append(module.register_forward_hook(hook))

    for layer_index, block in enumerate(model.transformer.h):
        prefix = f"layer_{layer_index:02d}"
        register(
            block.attn.c_attn,
            (f"{prefix}.W_Q", f"{prefix}.W_K", f"{prefix}.W_V"),
        )
        register(block.attn.c_proj, (f"{prefix}.W_O",))
        register(block.mlp.c_fc, (f"{prefix}.W_FC",))
        register(block.mlp.c_proj, (f"{prefix}.W_PROJ",))

    try:
        for example in dataset:
            input_ids = torch.tensor(example["input_ids"], dtype=torch.long).unsqueeze(0)
            model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    moments = {
        name: (sums[name] / counts[name]).clamp_min(torch.finfo(torch.float64).eps)
        for name in sums
    }
    if len(moments) != 72:
        raise ValueError(f"Expected activation moments for 72 matrices, got {len(moments)}")
    return moments


def main() -> None:
    args = parse_args()
    values = (
        args.base_q,
        args.sensitive_q,
        args.max_iterations,
        args.calibration_samples,
        args.eval_samples,
    )
    if min(values) < 1 or args.sensitive_q <= args.base_q:
        raise ValueError("Require positive settings and sensitive_q > base_q")

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
    calibration_data = prepare_lm_data(
        tokenizer,
        args.calibration_samples,
        split="train",
    )
    validation_data = prepare_validation_data(tokenizer, args.eval_samples)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    model.eval()
    original = original_target_weights(model)
    control_metrics = evaluate_model(model, validation_data)
    moments = collect_input_second_moments(model, calibration_data)

    moments_path = output_dir / "activation_second_moments.safetensors"
    save_file(
        {name: value.float().contiguous() for name, value in moments.items()},
        moments_path,
        metadata={
            "role": "calibration_statistics",
            "approximation": "diagonal_input_second_moment",
        },
    )

    with safe_open(reference_path, framework="pt", device="cpu") as reference_file:
        weight_metrics, matrix_settings, effective_q = quantize_and_inject_mixed(
            model,
            reference_file,
            args.base_q,
            args.sensitive_q,
            args.max_iterations,
            column_weights_by_matrix=moments,
        )
    functional_metrics = evaluate_model(model, validation_data)
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
        "method": "diagonal_activation_aware_mixed_q_alternating",
        "configuration": {
            "base_q": args.base_q,
            "sensitive_q": args.sensitive_q,
            "sensitive_rule": "layer == 0 or matrix in {W_O, W_PROJ}",
            "effective_weighted_average_q": effective_q,
            "max_iterations": args.max_iterations,
            "coefficient_objective": "sum_j E[x_j^2] * (W_true[i,j] - W_BCQ[i,j])^2",
        },
        "calibration": {
            "dataset": "Salesforce/wikitext",
            "dataset_config": "wikitext-2-raw-v1",
            "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "split": "train",
            "block_size": 1024,
            "samples": args.calibration_samples,
            "moments_file": moments_path.name,
            "moments_sha256": sha256_file(moments_path),
        },
        "evaluation": {
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
    report_path = output_dir / "activation_aware_results.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(implementation_path, output_dir / "screen_activation_aware_bcq.snapshot.py")
    print(json.dumps(report["candidate"], indent=2), flush=True)
    print(f"Saved activation-aware report to {report_path}", flush=True)


if __name__ == "__main__":
    main()
