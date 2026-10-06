"""Evaluate public-only W_attack candidates at weight and GPT-2 utility levels."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from build_bcq_gpt2_checkpoint import copy_conventional_weight
from datasets import load_dataset
from extract_gpt2_weights import extract_transformer_matrices, sha256_file
from safetensors import safe_open
from scipy.stats import spearmanr

from transformers import AutoModelForCausalLM, AutoTokenizer


MATRIX_NAMES = ("W_Q", "W_K", "W_V", "W_O", "W_FC", "W_PROJ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--reference-w-true", type=Path, required=True)
    parser.add_argument("--attack-dir", type=Path, required=True)
    parser.add_argument("--secret-key", type=Path, required=True)
    parser.add_argument("--base-model", default="openai-community/gpt2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--full-best-samples", type=int, default=128)
    parser.add_argument("--spearman-samples-per-matrix", type=int, default=5000)
    parser.add_argument("--oracle-fraction", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_validation_data(tokenizer: object, sample_count: int):
    raw_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
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
        raise ValueError(f"Requested {sample_count} samples, but only {len(grouped)} are available")
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


@torch.inference_mode()
def inject_matrices(model: torch.nn.Module, get: Callable[[str], torch.Tensor]) -> None:
    for layer_index, block in enumerate(model.transformer.h):
        prefix = f"layer_{layer_index:02d}"
        qkv = torch.cat([get(f"{prefix}.{name}") for name in ("W_Q", "W_K", "W_V")])
        copy_conventional_weight(block.attn.c_attn.weight, qkv, f"{prefix}.W_QKV")
        copy_conventional_weight(block.attn.c_proj.weight, get(f"{prefix}.W_O"), f"{prefix}.W_O")
        copy_conventional_weight(block.mlp.c_fc.weight, get(f"{prefix}.W_FC"), f"{prefix}.W_FC")
        copy_conventional_weight(block.mlp.c_proj.weight, get(f"{prefix}.W_PROJ"), f"{prefix}.W_PROJ")


def sampled_values(values: torch.Tensor, count: int) -> np.ndarray:
    flat = values.detach().float().reshape(-1)
    stride = max(1, flat.numel() // count)
    return flat[::stride][:count].numpy()


def weight_metrics(
    attack_get: Callable[[str], torch.Tensor],
    reference_file: object,
    names: list[str],
    spearman_samples: int,
) -> dict[str, float | int]:
    absolute_error = 0.0
    squared_error = 0.0
    reference_energy = 0.0
    sign_correct = 0
    element_count = 0
    top_intersection = 0
    top_count = 0
    attack_rank_samples = []
    true_rank_samples = []
    for name in names:
        attack = attack_get(name).float()
        true = reference_file.get_tensor(name).float()
        error = attack - true
        absolute_error += error.abs().sum().item()
        squared_error += error.square().sum().item()
        reference_energy += true.square().sum().item()
        sign_correct += (torch.sign(attack) == torch.sign(true)).sum().item()
        element_count += true.numel()

        k = max(1, math.ceil(0.10 * true.numel()))
        attack_top = torch.topk(attack.abs().reshape(-1), k, sorted=False).indices
        true_top = torch.topk(true.abs().reshape(-1), k, sorted=False).indices
        top_intersection += len(set(attack_top.tolist()).intersection(true_top.tolist()))
        top_count += k
        attack_rank_samples.append(sampled_values(attack.abs(), spearman_samples))
        true_rank_samples.append(sampled_values(true.abs(), spearman_samples))

    spearman = spearmanr(np.concatenate(attack_rank_samples), np.concatenate(true_rank_samples)).statistic
    return {
        "elements": element_count,
        "sign_accuracy": sign_correct / element_count,
        "mae": absolute_error / element_count,
        "nrmse": math.sqrt(squared_error / reference_energy),
        "abs_weight_spearman_sampled": float(spearman),
        "top_10_percent_abs_overlap": top_intersection / top_count,
        "spearman_sample_count": len(names) * spearman_samples,
    }


def key_recovery_metrics(predicted_path: Path, secret_path: Path) -> dict[str, float | int]:
    sign_correct = 0
    permutation_correct = 0
    joint_correct = 0
    count = 0
    with safe_open(predicted_path, framework="pt", device="cpu") as predicted:
        with safe_open(secret_path, framework="pt", device="cpu") as secret:
            if set(predicted.keys()) != set(secret.keys()):
                raise ValueError("Predicted and secret key names differ")
            for key in predicted.keys():
                if not key.endswith(".S"):
                    continue
                prefix = key.removesuffix(".S")
                predicted_s = predicted.get_tensor(key)
                true_s = secret.get_tensor(key)
                predicted_pi = predicted.get_tensor(f"{prefix}.pi")
                true_pi = secret.get_tensor(f"{prefix}.pi")
                sign_match = predicted_s == true_s
                permutation_match = predicted_pi == true_pi
                sign_correct += sign_match.sum().item()
                permutation_correct += permutation_match.sum().item()
                joint_correct += (sign_match & permutation_match).sum().item()
                count += true_s.numel()
    return {
        "key_elements": count,
        "sign_recovery_accuracy": sign_correct / count,
        "permutation_position_accuracy": permutation_correct / count,
        "joint_s_pi_position_accuracy": joint_correct / count,
    }


def oracle_getter(
    attack_get: Callable[[str], torch.Tensor], reference_file: object, fraction: float
) -> Callable[[str], torch.Tensor]:
    def get(name: str) -> torch.Tensor:
        attack = attack_get(name).float()
        true = reference_file.get_tensor(name).float()
        k = max(1, math.ceil(fraction * attack.numel()))
        selected = torch.topk(attack.abs().reshape(-1), k, sorted=False).indices
        completed = true.clone().reshape(-1)
        completed[selected] = attack.reshape(-1)[selected]
        return completed.reshape_as(true)

    return get


def add_ratios(metrics: dict[str, float | int], control: dict[str, float | int]) -> dict[str, float | int]:
    return {
        **metrics,
        "accuracy_retention": metrics["eval_accuracy"] / control["eval_accuracy"],
        "perplexity_ratio": metrics["perplexity"] / control["perplexity"],
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.oracle_fraction < 1:
        raise ValueError("--oracle-fraction must be between zero and one")
    model_path = args.model_path.expanduser().resolve()
    reference_path = args.reference_w_true.expanduser().resolve()
    attack_dir = args.attack_dir.expanduser().resolve()
    secret_path = args.secret_key.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    required = [model_path, reference_path, attack_dir, secret_path]
    if not all(path.exists() for path in required):
        raise FileNotFoundError(f"Missing required input among: {required}")
    if output_dir.is_dir() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((attack_dir / "attack_manifest.json").read_text(encoding="utf-8"))
    attack_paths = {name: attack_dir / metadata["file"] for name, metadata in manifest["methods"].items()}
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    screen_data = prepare_validation_data(tokenizer, args.eval_samples)
    full_data = prepare_validation_data(tokenizer, args.full_best_samples)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    model.eval()
    true_control = evaluate_model(model, screen_data)

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=True)
    base_weights, _ = extract_transformer_matrices(base_model)
    del base_model
    names = sorted(base_weights)
    results: dict[str, object] = {}

    with safe_open(reference_path, framework="pt", device="cpu") as reference_file:

        def base_get(name: str) -> torch.Tensor:
            return base_weights[name]

        base_weight_metrics = weight_metrics(base_get, reference_file, names, args.spearman_samples_per_matrix)
        inject_matrices(model, base_get)
        base_functional = add_ratios(evaluate_model(model, screen_data), true_control)
        results["base_only_control"] = {
            "weight_metrics": base_weight_metrics,
            "direct_functional": base_functional,
        }

        for method, path in attack_paths.items():
            with safe_open(path, framework="pt", device="cpu") as attack_file:
                attack_get = attack_file.get_tensor
                metrics = weight_metrics(attack_get, reference_file, names, args.spearman_samples_per_matrix)
                inject_matrices(model, attack_get)
                direct = add_ratios(evaluate_model(model, screen_data), true_control)
                inject_matrices(model, oracle_getter(attack_get, reference_file, args.oracle_fraction))
                oracle = add_ratios(evaluate_model(model, screen_data), true_control)
                results[method] = {
                    "weight_metrics": metrics,
                    "direct_functional": direct,
                    "oracle_completion_functional": oracle,
                }
                print(f"{method}: direct={direct}, oracle={oracle}", flush=True)

        attack_names = list(attack_paths)
        best_method = min(
            attack_names,
            key=lambda name: results[name]["direct_functional"]["perplexity"],
        )
        with safe_open(attack_paths[best_method], framework="pt", device="cpu") as best_file:
            inject_matrices(model, best_file.get_tensor)
            best_full = add_ratios(
                evaluate_model(model, full_data),
                evaluate_model(AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True), full_data),
            )
            checkpoint_dir = output_dir / "best_attack_checkpoint"
            model.save_pretrained(checkpoint_dir, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_dir)

    predicted_key_path = attack_dir / manifest["predicted_key"]["file"]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "threat_model": manifest["threat_model"],
        "evaluation": {
            "dataset": "Salesforce/wikitext",
            "dataset_config": "wikitext-2-raw-v1",
            "split": "validation",
            "screen_samples": args.eval_samples,
            "full_best_samples": args.full_best_samples,
            "block_size": 1024,
            "oracle_completion": ("use W_attack for its top-|W_attack| fraction and W_true for all other weights"),
            "oracle_fraction": args.oracle_fraction,
        },
        "w_true_screen_control": true_control,
        "results": results,
        "a3_key_recovery": key_recovery_metrics(predicted_key_path, secret_path),
        "best_public_attack": best_method,
        "best_public_attack_full_functional": best_full,
        "provenance": {
            "reference_w_true_sha256": sha256_file(reference_path),
            "attack_manifest_sha256": sha256_file(attack_dir / "attack_manifest.json"),
            "secret_key_sha256": sha256_file(secret_path),
            "transformers_commit": "7c65cdb570646e8b01cc30d069579f2f4b60c398",
        },
    }
    report_path = output_dir / "attack_results.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    script_path = Path(__file__).resolve()
    shutil.copyfile(script_path, output_dir / "evaluate_public_b_attacks.snapshot.py")
    print(f"Best public attack: {best_method}")
    print(f"Full functional result: {best_full}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
