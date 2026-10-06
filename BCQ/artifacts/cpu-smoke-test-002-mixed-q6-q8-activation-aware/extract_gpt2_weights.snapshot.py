"""Extract GPT-2 block weights and build a row-wise BCQ baseline.

The saved artifacts deliberately separate public binary matrices from secret
scales and offsets so an attack cannot accidentally consume secret values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import safetensors
import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM


MATRIX_DESCRIPTIONS = {
    "W_Q": "attention query projection",
    "W_K": "attention key projection",
    "W_V": "attention value projection",
    "W_O": "attention output projection",
    "W_FC": "MLP input projection",
    "W_PROJ": "MLP output projection",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("outputs/cpu-smoke-test-002"),
        help="Local GPT-2 checkpoint produced by run_clm.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("BCQ/artifacts/cpu-smoke-test-002-q3"),
        help="Directory in which to write the separated BCQ artifacts.",
    )
    parser.add_argument(
        "--q",
        type=int,
        default=3,
        help="Number of binary bases per weight (default: 3).",
    )
    parser.add_argument(
        "--method",
        choices=("greedy", "alternating"),
        default="greedy",
        help="BCQ optimizer (default: greedy).",
    )
    parser.add_argument(
        "--alternating-iterations",
        type=int,
        default=20,
        help="Maximum alternating optimization sweeps (default: 20).",
    )
    parser.add_argument(
        "--sensitive-q",
        type=int,
        default=None,
        help="Use this Q for layer 0, W_O, and W_PROJ; use --q elsewhere.",
    )
    parser.add_argument(
        "--column-weights-path",
        type=Path,
        default=None,
        help="Optional safetensors file of per-matrix activation second moments.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in an existing non-empty output directory.",
    )
    return parser.parse_args()


def extract_transformer_matrices(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Return all GPT-2 attention/MLP matrices in [out_features, in_features] layout."""
    matrices: dict[str, torch.Tensor] = {}
    source_parameters: dict[str, str] = {}

    for layer_index, block in enumerate(model.transformer.h):
        prefix = f"layer_{layer_index:02d}"

        # GPT-2 Conv1D stores weights as [in_features, out_features].
        qkv = block.attn.c_attn.weight.detach().cpu().float().T.contiguous()
        query, key, value = torch.chunk(qkv, 3, dim=0)
        layer_matrices = {
            "W_Q": query.contiguous(),
            "W_K": key.contiguous(),
            "W_V": value.contiguous(),
            "W_O": block.attn.c_proj.weight.detach().cpu().float().T.contiguous(),
            "W_FC": block.mlp.c_fc.weight.detach().cpu().float().T.contiguous(),
            "W_PROJ": block.mlp.c_proj.weight.detach().cpu().float().T.contiguous(),
        }

        for matrix_name, weight in layer_matrices.items():
            artifact_name = f"{prefix}.{matrix_name}"
            matrices[artifact_name] = weight
            if matrix_name in {"W_Q", "W_K", "W_V"}:
                source_parameters[artifact_name] = f"transformer.h.{layer_index}.attn.c_attn.weight"
            elif matrix_name == "W_O":
                source_parameters[artifact_name] = f"transformer.h.{layer_index}.attn.c_proj.weight"
            elif matrix_name == "W_FC":
                source_parameters[artifact_name] = f"transformer.h.{layer_index}.mlp.c_fc.weight"
            else:
                source_parameters[artifact_name] = f"transformer.h.{layer_index}.mlp.c_proj.weight"

    return matrices, source_parameters


@torch.inference_mode()
def greedy_residual_bcq(
    weight: torch.Tensor, q: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate one matrix with per-row offsets and greedy binary bases.

    Shapes:
        weight, reconstruction: [rows, columns]
        binary: [q, rows, columns]
        alpha: [rows, q]
        offset: [rows, 1]
    """
    if weight.ndim != 2:
        raise ValueError(f"BCQ expects a 2D matrix, got shape {tuple(weight.shape)}")
    if q < 1:
        raise ValueError(f"q must be at least 1, got {q}")

    offset = weight.mean(dim=1, keepdim=True)
    residual = weight - offset
    binary_bases: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []

    for _ in range(q):
        binary = torch.where(residual >= 0, 1, -1).to(torch.int8)
        alpha = residual.abs().mean(dim=1)
        residual = residual - alpha[:, None] * binary
        binary_bases.append(binary)
        scales.append(alpha)

    binary_tensor = torch.stack(binary_bases, dim=0)
    alpha_tensor = torch.stack(scales, dim=1)
    reconstruction = offset + torch.einsum("rq,qrc->rc", alpha_tensor, binary_tensor.float())
    return binary_tensor, alpha_tensor, offset, reconstruction


def fit_rowwise_coefficients(
    weight: torch.Tensor,
    binary: torch.Tensor,
    column_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Jointly fit per-row z and alpha for fixed binary bases by least squares."""
    weight64 = weight.double()
    binary64 = binary.double()
    design = torch.cat(
        (
            torch.ones((*weight.shape, 1), dtype=torch.float64),
            binary64.permute(1, 2, 0),
        ),
        dim=2,
    )
    if column_weights is None:
        normalized_weights = torch.ones(weight.shape[1], dtype=torch.float64)
    else:
        if column_weights.shape != (weight.shape[1],):
            raise ValueError(
                f"Expected {weight.shape[1]} column weights, got {tuple(column_weights.shape)}"
            )
        normalized_weights = column_weights.double()
        if not torch.isfinite(normalized_weights).all() or (normalized_weights <= 0).any():
            raise ValueError("column_weights must be finite and strictly positive")
        normalized_weights = normalized_weights / normalized_weights.mean()

    weighted_design = design * normalized_weights[None, :, None]
    gram = torch.einsum("rnp,rnq->rpq", design, weighted_design)
    rhs = torch.einsum("rnp,rn->rp", weighted_design, weight64)
    try:
        coefficients = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)
    except torch.linalg.LinAlgError:
        coefficients = torch.einsum("rpq,rq->rp", torch.linalg.pinv(gram), rhs)

    offset = coefficients[:, :1]
    alpha = coefficients[:, 1:]
    reconstruction = offset + torch.einsum("rq,qrc->rc", alpha, binary64)
    return alpha, offset, reconstruction


@torch.inference_mode()
def alternating_bcq(
    weight: torch.Tensor,
    q: int,
    max_iterations: int = 20,
    column_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Alternately optimize binary bases and joint least-squares row coefficients."""
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be at least 1, got {max_iterations}")

    binary, _, _, _ = greedy_residual_bcq(weight, q)
    binary64 = binary.double()
    alpha, offset, reconstruction = fit_rowwise_coefficients(
        weight,
        binary64,
        column_weights=column_weights,
    )
    objective_weights = (
        torch.ones(weight.shape[1], dtype=torch.float64)
        if column_weights is None
        else column_weights.double() / column_weights.double().mean()
    )
    previous_error = (
        (reconstruction - weight.double()).square() * objective_weights[None, :]
    ).sum().item()
    completed_iterations = 0

    for iteration in range(max_iterations):
        reconstruction = offset + torch.einsum("rq,qrc->rc", alpha, binary64)
        changed_bits = 0

        for basis_index in range(q):
            contribution = alpha[:, basis_index, None] * binary64[basis_index]
            residual_without_basis = weight.double() - (reconstruction - contribution)
            new_binary = torch.where(
                residual_without_basis * alpha[:, basis_index, None] >= 0,
                1.0,
                -1.0,
            )
            changed_bits += (new_binary != binary64[basis_index]).sum().item()
            reconstruction = (
                reconstruction - contribution + alpha[:, basis_index, None] * new_binary
            )
            binary64[basis_index] = new_binary

        alpha, offset, reconstruction = fit_rowwise_coefficients(
            weight,
            binary64,
            column_weights=column_weights,
        )
        error = (
            (reconstruction - weight.double()).square() * objective_weights[None, :]
        ).sum().item()
        if error > previous_error + 1e-10 * max(previous_error, 1.0):
            raise RuntimeError(
                f"Alternating BCQ objective increased at iteration {iteration}: "
                f"{previous_error} -> {error}"
            )
        completed_iterations = iteration + 1
        relative_improvement = (previous_error - error) / max(previous_error, 1e-30)
        previous_error = error
        if changed_bits == 0 or relative_improvement < 1e-7:
            break

    # Keep scales non-negative without changing the reconstruction.
    for basis_index in range(q):
        negative = alpha[:, basis_index] < 0
        binary64[basis_index, negative] *= -1
        alpha[negative, basis_index] *= -1

    binary_out = binary64.to(torch.int8)
    alpha_out = alpha.to(weight.dtype)
    offset_out = offset.to(weight.dtype)
    reconstruction_out = offset_out + torch.einsum(
        "rq,qrc->rc", alpha_out, binary_out.float()
    )
    return binary_out, alpha_out, offset_out, reconstruction_out, completed_iterations


def reconstruction_metrics(weight: torch.Tensor, reconstruction: torch.Tensor) -> dict[str, float]:
    error = reconstruction - weight
    squared_error = error.square().sum()
    weight_energy = weight.square().sum()
    return {
        "mse": error.square().mean().item(),
        "rmse": error.square().mean().sqrt().item(),
        "max_abs_error": error.abs().max().item(),
        "relative_frobenius_error": (squared_error / weight_energy).sqrt().item(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(repo_root: Path) -> dict[str, object]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        status = git("status", "--short")
        return {
            "transformers_commit": git("rev-parse", "HEAD"),
            "repository_dirty": bool(status),
            "git_status": status.splitlines(),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "transformers_commit": None,
            "repository_dirty": None,
            "git_status": [],
        }


def main() -> None:
    args = parse_args()
    if args.q < 1:
        raise ValueError(f"--q must be at least 1, got {args.q}")
    if args.alternating_iterations < 1:
        raise ValueError(
            f"--alternating-iterations must be at least 1, got {args.alternating_iterations}"
        )
    if args.sensitive_q is not None and args.sensitive_q <= args.q:
        raise ValueError("--sensitive-q must be larger than --q")
    if args.column_weights_path is not None and args.method != "alternating":
        raise ValueError("--column-weights-path requires --method alternating")

    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    if output_dir.is_dir() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new --output-dir or pass --overwrite explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    column_weights_by_matrix: dict[str, torch.Tensor] | None = None
    column_weights_path = None
    if args.column_weights_path is not None:
        column_weights_path = args.column_weights_path.expanduser().resolve()
        if not column_weights_path.is_file():
            raise FileNotFoundError(f"Column weights do not exist: {column_weights_path}")
        with safe_open(column_weights_path, framework="pt", device="cpu") as column_weights_file:
            column_weights_by_matrix = {
                name: column_weights_file.get_tensor(name) for name in column_weights_file.keys()
            }

    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    model.eval()
    if model.config.model_type != "gpt2":
        raise ValueError(f"Expected a GPT-2 checkpoint, got model_type={model.config.model_type!r}")

    true_weights, source_parameters = extract_transformer_matrices(model)
    if column_weights_by_matrix is not None and set(column_weights_by_matrix) != set(true_weights):
        missing = sorted(set(true_weights) - set(column_weights_by_matrix))
        unexpected = sorted(set(column_weights_by_matrix) - set(true_weights))
        raise ValueError(
            f"Column-weight matrix set does not match W_true; missing={missing}, "
            f"unexpected={unexpected}"
        )
    public_binary: dict[str, torch.Tensor] = {}
    secret_parameters: dict[str, torch.Tensor] = {}
    baseline_weights: dict[str, torch.Tensor] = {}
    matrix_metadata: dict[str, dict[str, object]] = {}

    total_squared_error = 0.0
    total_weight_energy = 0.0
    total_elements = 0
    total_binary_values = 0

    for name, weight in true_weights.items():
        layer_name, short_name = name.split(".", maxsplit=1)
        layer_index = int(layer_name.removeprefix("layer_"))
        selected_q = (
            args.sensitive_q
            if args.sensitive_q is not None
            and (layer_index == 0 or short_name in {"W_O", "W_PROJ"})
            else args.q
        )
        column_weights = (
            column_weights_by_matrix[name] if column_weights_by_matrix is not None else None
        )
        if args.method == "alternating":
            binary, alpha, offset, reconstruction, optimization_iterations = alternating_bcq(
                weight,
                selected_q,
                max_iterations=args.alternating_iterations,
                column_weights=column_weights,
            )
        else:
            binary, alpha, offset, reconstruction = greedy_residual_bcq(weight, selected_q)
            optimization_iterations = 0
        public_binary[name] = binary
        secret_parameters[f"{name}.alpha"] = alpha
        secret_parameters[f"{name}.z"] = offset
        baseline_weights[name] = reconstruction

        error = reconstruction - weight
        total_squared_error += error.square().sum().item()
        total_weight_energy += weight.square().sum().item()
        total_elements += weight.numel()
        total_binary_values += selected_q * weight.numel()
        metrics = reconstruction_metrics(weight, reconstruction)
        matrix_metadata[name] = {
            "description": MATRIX_DESCRIPTIONS[short_name],
            "source_parameter": source_parameters[name],
            "shape": list(weight.shape),
            "layout": "out_features_by_in_features",
            "q": selected_q,
            "optimization_iterations": optimization_iterations,
            "metrics": metrics,
        }
        print(f"{name}: shape={tuple(weight.shape)}, relative_error={metrics['relative_frobenius_error']:.6f}")

    q_label = (
        f"mixed_q{args.q}_q{args.sensitive_q}"
        if args.sensitive_q is not None
        else f"q{args.q}"
    )
    artifact_paths = {
        "reference_w_true": output_dir / "reference_w_true.safetensors",
        "public_b": output_dir / f"public_b_{q_label}.safetensors",
        "secret_alpha_z": output_dir / f"secret_alpha_z_{q_label}.safetensors",
        "baseline_w_bcq": output_dir / f"baseline_w_bcq_{q_label}.safetensors",
    }
    common_metadata = {
        "format": "gpt2-row-wise-bcq-v1",
        "q": q_label,
        "method": args.method,
        "matrix_layout": "out_features_by_in_features",
    }
    save_file(
        true_weights,
        artifact_paths["reference_w_true"],
        metadata={**common_metadata, "role": "private_reference"},
    )
    save_file(public_binary, artifact_paths["public_b"], metadata={**common_metadata, "role": "public"})
    save_file(
        secret_parameters,
        artifact_paths["secret_alpha_z"],
        metadata={**common_metadata, "role": "secret"},
    )
    save_file(
        baseline_weights,
        artifact_paths["baseline_w_bcq"],
        metadata={**common_metadata, "role": "private_baseline"},
    )

    repo_root = Path(__file__).resolve().parents[1]
    script_snapshot_path = output_dir / "extract_gpt2_weights.snapshot.py"
    shutil.copyfile(Path(__file__).resolve(), script_snapshot_path)
    artifact_paths["implementation_snapshot"] = script_snapshot_path
    source_weight_files = sorted(model_path.glob("*.safetensors")) + sorted(
        model_path.glob("pytorch_model*.bin")
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "model_path": str(model_path),
        "source_weight_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in source_weight_files
        },
        "model": {
            "model_type": model.config.model_type,
            "architectures": model.config.architectures,
            "n_layer": model.config.n_layer,
            "n_embd": model.config.n_embd,
            "n_head": model.config.n_head,
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "safetensors": safetensors.__version__,
        },
        "source_repository": git_metadata(repo_root),
        "bcq": {
            "q": args.q,
            "sensitive_q": args.sensitive_q,
            "mixed_q_rule": (
                "layer == 0 or matrix in {W_O, W_PROJ}" if args.sensitive_q is not None else None
            ),
            "effective_weighted_average_q": total_binary_values / total_elements,
            "formula": "W_BCQ[i,:] = z[i] + sum_k alpha[i,k] * B[k,i,:]",
            "method": args.method,
            "binary_values": [-1, 1],
            "alpha_granularity": "per_row_per_binary_basis",
            "offset_granularity": "per_row",
            "initialization": "mean_centered_greedy_residual_binarization",
            "alternating_max_iterations": (
                args.alternating_iterations if args.method == "alternating" else 0
            ),
            "coefficient_update": (
                "joint_per_row_least_squares" if args.method == "alternating" else "greedy_mean_absolute_residual"
            ),
            "binary_update": (
                "coordinate_sign_update" if args.method == "alternating" else "greedy_residual_sign"
            ),
            "column_weighting": (
                "diagonal_input_second_moment" if column_weights_by_matrix is not None else None
            ),
            "column_weights_file": (
                {
                    "path": str(column_weights_path),
                    "sha256": sha256_file(column_weights_path),
                }
                if column_weights_path is not None
                else None
            ),
        },
        "scope": {
            "layers": list(range(model.config.n_layer)),
            "matrices_per_layer": list(MATRIX_DESCRIPTIONS),
            "matrix_count": len(true_weights),
            "excluded": ["biases", "layer_norm", "token_embedding", "position_embedding", "lm_head"],
        },
        "aggregate_metrics": {
            "mse": total_squared_error / total_elements,
            "rmse": (total_squared_error / total_elements) ** 0.5,
            "relative_frobenius_error": (total_squared_error / total_weight_energy) ** 0.5,
        },
        "matrices": matrix_metadata,
        "artifacts": {},
    }
    manifest["artifacts"] = {
        role: {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for role, path in artifact_paths.items()
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"\nSaved {len(true_weights)} matrices from {model.config.n_layer} layers to {output_dir}")
    print(f"Aggregate relative Frobenius error: {manifest['aggregate_metrics']['relative_frobenius_error']:.6f}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
