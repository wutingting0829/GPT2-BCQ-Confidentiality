"""Inject conventional [out, in] BCQ weights into a Hugging Face GPT-2 checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from extract_gpt2_weights import extract_transformer_matrices, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("outputs/cpu-smoke-test-002"),
        help="Fine-tuned GPT-2 checkpoint whose block weights will be replaced.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("BCQ/artifacts/cpu-smoke-test-002-q3"),
        help="BCQ artifact directory containing manifest.json and baseline W_BCQ.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cpu-smoke-test-002-bcq-q3"),
        help="Destination for the loadable BCQ GPT-2 checkpoint.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in an existing non-empty output directory.",
    )
    return parser.parse_args()


def copy_conventional_weight(parameter: torch.nn.Parameter, conventional: torch.Tensor, name: str) -> None:
    """Copy [out, in] into GPT-2 Conv1D's [in, out] parameter."""
    hf_weight = conventional.T.contiguous()
    if parameter.shape != hf_weight.shape:
        raise ValueError(
            f"Shape mismatch for {name}: parameter={tuple(parameter.shape)}, "
            f"converted BCQ={tuple(hf_weight.shape)}"
        )
    parameter.copy_(hf_weight.to(device=parameter.device, dtype=parameter.dtype))


@torch.inference_mode()
def inject_bcq_weights(model: torch.nn.Module, baseline_path: Path) -> tuple[int, float]:
    with safe_open(baseline_path, framework="pt", device="cpu") as baseline_file:
        expected_names = {
            f"layer_{layer_index:02d}.{matrix_name}"
            for layer_index in range(model.config.n_layer)
            for matrix_name in ("W_Q", "W_K", "W_V", "W_O", "W_FC", "W_PROJ")
        }
        actual_names = set(baseline_file.keys())
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            unexpected = sorted(actual_names - expected_names)
            raise ValueError(f"Unexpected BCQ matrix set; missing={missing}, unexpected={unexpected}")

        for layer_index, block in enumerate(model.transformer.h):
            prefix = f"layer_{layer_index:02d}"
            qkv = torch.cat(
                [
                    baseline_file.get_tensor(f"{prefix}.W_Q"),
                    baseline_file.get_tensor(f"{prefix}.W_K"),
                    baseline_file.get_tensor(f"{prefix}.W_V"),
                ],
                dim=0,
            )
            copy_conventional_weight(block.attn.c_attn.weight, qkv, f"{prefix}.W_QKV")
            copy_conventional_weight(
                block.attn.c_proj.weight,
                baseline_file.get_tensor(f"{prefix}.W_O"),
                f"{prefix}.W_O",
            )
            copy_conventional_weight(
                block.mlp.c_fc.weight,
                baseline_file.get_tensor(f"{prefix}.W_FC"),
                f"{prefix}.W_FC",
            )
            copy_conventional_weight(
                block.mlp.c_proj.weight,
                baseline_file.get_tensor(f"{prefix}.W_PROJ"),
                f"{prefix}.W_PROJ",
            )

        injected_weights, _ = extract_transformer_matrices(model)
        max_difference = max(
            (injected_weights[name] - baseline_file.get_tensor(name)).abs().max().item()
            for name in sorted(expected_names)
        )
        return len(expected_names), max_difference


def main() -> None:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    artifact_manifest_path = artifact_dir / "manifest.json"

    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    if not artifact_manifest_path.is_file():
        raise FileNotFoundError(f"Artifact manifest does not exist: {artifact_manifest_path}")
    if output_dir.is_dir() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new --output-dir or pass --overwrite explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    baseline_path = artifact_dir / artifact_manifest["artifacts"]["baseline_w_bcq"]["file"]
    if not baseline_path.is_file():
        raise FileNotFoundError(f"BCQ baseline does not exist: {baseline_path}")

    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if model.config.model_type != "gpt2":
        raise ValueError(f"Expected GPT-2, got model_type={model.config.model_type!r}")

    matrix_count, max_injection_difference = inject_bcq_weights(model, baseline_path)
    if max_injection_difference != 0.0:
        raise ValueError(f"BCQ injection verification failed; max difference={max_injection_difference}")

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    implementation_path = Path(__file__).resolve()
    snapshot_path = output_dir / "build_bcq_gpt2_checkpoint.snapshot.py"
    shutil.copyfile(implementation_path, snapshot_path)

    output_weight_files = sorted(output_dir.glob("*.safetensors")) + sorted(
        output_dir.glob("pytorch_model*.bin")
    )
    build_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "source_model_path": str(model_path),
        "bcq_artifact_dir": str(artifact_dir),
        "bcq_q": artifact_manifest["bcq"]["q"],
        "matrix_convention": "W_BCQ[out_features, in_features]",
        "injection": {
            "target": "GPT-2 Conv1D weights[in_features, out_features]",
            "matrix_count": matrix_count,
            "max_verification_difference": max_injection_difference,
            "replaced": ["W_Q", "W_K", "W_V", "W_O", "W_FC", "W_PROJ"],
            "preserved": ["biases", "layer_norm", "token_embedding", "position_embedding", "lm_head"],
        },
        "provenance": {
            "transformers_commit": artifact_manifest["source_repository"]["transformers_commit"],
            "source_checkpoint_files": artifact_manifest["source_weight_files"],
            "baseline_w_bcq_sha256": sha256_file(baseline_path),
            "builder_script_sha256": sha256_file(implementation_path),
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "output_weight_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in output_weight_files
        },
    }
    build_manifest_path = output_dir / "bcq_build_manifest.json"
    build_manifest_path.write_text(json.dumps(build_manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Injected and verified {matrix_count} BCQ matrices; max difference={max_injection_difference}")
    print(f"Saved loadable GPT-2 checkpoint to {output_dir}")
    print(f"Build manifest: {build_manifest_path}")


if __name__ == "__main__":
    main()
