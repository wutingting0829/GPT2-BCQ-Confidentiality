"""Screen alternating BCQ configurations without saving full weight artifacts."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path

import torch
import transformers
from datasets import __version__ as datasets_version
from datasets import load_dataset
from safetensors import __version__ as safetensors_version
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_bcq_gpt2_checkpoint import copy_conventional_weight
from evaluate_quantization_baseline import ErrorStatistics
from extract_gpt2_weights import alternating_bcq, sha256_file


MATRIX_NAMES = ("W_Q", "W_K", "W_V", "W_O", "W_FC", "W_PROJ")


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
        default=Path("BCQ/sweeps/q4-q6-alternating-50"),
    )
    parser.add_argument("--q-values", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_validation_data(tokenizer: object, sample_count: int):
    raw_dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="validation",
    )
    column_names = list(raw_dataset.features)

    def tokenize(examples):
        return tokenizer(examples["text"])

    def group_texts(examples):
        concatenated = {key: list(chain(*examples[key])) for key in examples}
        total_length = (len(concatenated["input_ids"]) // 1024) * 1024
        grouped = {
            key: [values[index : index + 1024] for index in range(0, total_length, 1024)]
            for key, values in concatenated.items()
        }
        grouped["labels"] = grouped["input_ids"].copy()
        return grouped

    tokenized = raw_dataset.map(tokenize, batched=True, remove_columns=column_names)
    grouped = tokenized.map(group_texts, batched=True)
    if sample_count > len(grouped):
        raise ValueError(f"Requested {sample_count} samples but validation only has {len(grouped)}")
    return grouped.select(range(sample_count))


@torch.inference_mode()
def evaluate_model(model: torch.nn.Module, dataset: object) -> dict[str, float | int]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    token_count = 0

    for example in dataset:
        input_ids = torch.tensor(example["input_ids"], dtype=torch.long).unsqueeze(0)
        outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        loss_sum += outputs.loss.item()
        predictions = outputs.logits.argmax(dim=-1)
        correct += (predictions[:, :-1] == input_ids[:, 1:]).sum().item()
        token_count += input_ids[:, 1:].numel()

    loss = loss_sum / len(dataset)
    return {
        "eval_samples": len(dataset),
        "evaluated_tokens": token_count,
        "eval_accuracy": correct / token_count,
        "eval_loss": loss,
        "perplexity": math.exp(loss),
    }


def original_target_weights(model: torch.nn.Module) -> dict[int, dict[str, torch.Tensor]]:
    return {
        layer_index: {
            "qkv": block.attn.c_attn.weight.detach().clone(),
            "o": block.attn.c_proj.weight.detach().clone(),
            "fc": block.mlp.c_fc.weight.detach().clone(),
            "proj": block.mlp.c_proj.weight.detach().clone(),
        }
        for layer_index, block in enumerate(model.transformer.h)
    }


@torch.inference_mode()
def restore_target_weights(
    model: torch.nn.Module,
    original: dict[int, dict[str, torch.Tensor]],
) -> None:
    for layer_index, block in enumerate(model.transformer.h):
        block.attn.c_attn.weight.copy_(original[layer_index]["qkv"])
        block.attn.c_proj.weight.copy_(original[layer_index]["o"])
        block.mlp.c_fc.weight.copy_(original[layer_index]["fc"])
        block.mlp.c_proj.weight.copy_(original[layer_index]["proj"])


@torch.inference_mode()
def quantize_and_inject(
    model: torch.nn.Module,
    reference_file: object,
    q: int,
    max_iterations: int,
) -> tuple[dict[str, float | int], dict[str, int]]:
    statistics = ErrorStatistics()
    iteration_counts: dict[str, int] = {}

    for layer_index, block in enumerate(model.transformer.h):
        prefix = f"layer_{layer_index:02d}"
        reconstructed: dict[str, torch.Tensor] = {}
        for matrix_name in MATRIX_NAMES:
            artifact_name = f"{prefix}.{matrix_name}"
            reference = reference_file.get_tensor(artifact_name)
            _, _, _, bcq_weight, iterations = alternating_bcq(
                reference,
                q,
                max_iterations=max_iterations,
            )
            reconstructed[matrix_name] = bcq_weight
            statistics.update(reference, bcq_weight)
            iteration_counts[artifact_name] = iterations

        qkv = torch.cat(
            [reconstructed["W_Q"], reconstructed["W_K"], reconstructed["W_V"]],
            dim=0,
        )
        copy_conventional_weight(block.attn.c_attn.weight, qkv, f"{prefix}.W_QKV")
        copy_conventional_weight(block.attn.c_proj.weight, reconstructed["W_O"], f"{prefix}.W_O")
        copy_conventional_weight(block.mlp.c_fc.weight, reconstructed["W_FC"], f"{prefix}.W_FC")
        copy_conventional_weight(block.mlp.c_proj.weight, reconstructed["W_PROJ"], f"{prefix}.W_PROJ")

    return statistics.metrics(), iteration_counts


def main() -> None:
    args = parse_args()
    q_values = sorted(set(args.q_values))
    if not q_values or q_values[0] < 1:
        raise ValueError(f"Every Q must be at least 1, got {args.q_values}")
    if args.max_iterations < 1 or args.eval_samples < 1:
        raise ValueError("--max-iterations and --eval-samples must be at least 1")

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
    print(f"W_true: {control_metrics}", flush=True)

    results: dict[str, dict[str, object]] = {}
    with safe_open(reference_path, framework="pt", device="cpu") as reference_file:
        for q in q_values:
            restore_target_weights(model, original)
            weight_metrics, iteration_counts = quantize_and_inject(
                model,
                reference_file,
                q,
                args.max_iterations,
            )
            functional_metrics = evaluate_model(model, dataset)
            results[str(q)] = {
                "weight_metrics": weight_metrics,
                "functional_metrics": functional_metrics,
                "optimization_iterations": {
                    "minimum": min(iteration_counts.values()),
                    "maximum": max(iteration_counts.values()),
                    "mean": sum(iteration_counts.values()) / len(iteration_counts),
                    "per_matrix": iteration_counts,
                },
                "accuracy_retention": (
                    functional_metrics["eval_accuracy"] / control_metrics["eval_accuracy"]
                ),
                "perplexity_ratio": functional_metrics["perplexity"] / control_metrics["perplexity"],
            }
            print(f"Q={q}: {results[str(q)]}", flush=True)

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
        "method": "alternating",
        "q_values": q_values,
        "max_iterations": args.max_iterations,
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
        "results": results,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets_version,
            "safetensors": safetensors_version,
        },
        "provenance": {
            "model_path": str(model_path),
            "reference_path": str(reference_path),
            "reference_sha256": sha256_file(reference_path),
            "script_sha256": sha256_file(implementation_path),
        },
        "restore_verification": "passed",
    }
    report_path = output_dir / "sweep_results.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(implementation_path, output_dir / "sweep_alternating_bcq.snapshot.py")
    print(f"Saved sweep report to {report_path}")


if __name__ == "__main__":
    main()
