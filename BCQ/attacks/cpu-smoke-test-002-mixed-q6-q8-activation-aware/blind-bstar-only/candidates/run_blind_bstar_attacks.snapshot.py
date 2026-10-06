"""Generate architecture-agnostic W_attack candidates from public B* only."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from extract_gpt2_weights import fit_rowwise_coefficients, greedy_residual_bcq, sha256_file


FEATURE_COUNT = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-b-star", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--synthetic-rows", type=int, default=4096)
    parser.add_argument("--synthetic-columns", type=int, default=256)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def shape_only_scale(rows: int, columns: int) -> float:
    """Xavier RMS inferred only from the dimensions of the public tensor."""
    return math.sqrt(2.0 / (rows + columns))


def normalize_rows(weight: torch.Tensor, target_rms: float) -> torch.Tensor:
    rms = weight.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
    return weight * (target_rms / rms)


def basis_features(binary: torch.Tensor) -> torch.Tensor:
    """Compute model-independent, sign-equivariant features [Q, rows, 12]."""
    values = binary.float()
    basis_sum = values.sum(dim=2)
    orientation = torch.where(basis_sum != 0, basis_sum.sign(), values[:, :, 0])
    mean = values.mean(dim=2)
    transition = (values[:, :, 1:] != values[:, :, :-1]).float().mean(dim=2)
    chunks = torch.stack([chunk.mean(dim=2) for chunk in values.tensor_split(8, dim=2)], dim=2)
    return torch.cat(
        [
            orientation[:, :, None],
            mean[:, :, None],
            mean.abs()[:, :, None],
            (orientation * transition)[:, :, None],
            chunks,
        ],
        dim=2,
    )


def randomize_bases(
    binary: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, rows, _ = binary.shape
    permutation = torch.stack([torch.randperm(q, generator=generator) for _ in range(rows)])
    signs = torch.randint(0, 2, (rows, q), generator=generator, dtype=torch.int8) * 2 - 1
    row_indices = torch.arange(rows)
    public = torch.empty_like(binary)
    coefficients = torch.empty((rows, q), dtype=torch.float32)
    return public, coefficients, permutation, signs, row_indices


def fit_synthetic_prior(
    q: int,
    rows: int,
    columns: int,
    ridge: float,
    generator: torch.Generator,
) -> torch.Tensor:
    samples = torch.randn((rows, columns), generator=generator)
    # Mix Gaussian and heavy-tailed rows without using any model/checkpoint distribution.
    heavy_rows = torch.rand(rows, generator=generator) < 0.5
    samples[heavy_rows] *= torch.empty((int(heavy_rows.sum()), 1)).exponential_(
        1.0, generator=generator
    )
    samples -= samples.mean(dim=1, keepdim=True)
    samples = normalize_rows(samples, 1.0)
    binary, alpha, _, _ = greedy_residual_bcq(samples, q)

    public, targets, permutation, signs, row_indices = randomize_bases(binary, generator)
    for public_index in range(q):
        source_index = permutation[:, public_index]
        public[public_index] = (
            binary[source_index, row_indices] * signs[:, public_index, None]
        )
        targets[:, public_index] = (
            alpha[row_indices, source_index] * signs[:, public_index]
        )

    design = basis_features(public).permute(1, 0, 2).reshape(-1, FEATURE_COUNT).double()
    target = targets.reshape(-1).double()
    regularizer = ridge * torch.eye(FEATURE_COUNT, dtype=torch.float64)
    return torch.linalg.solve(design.T @ design + regularizer, design.T @ target)


def a2_public_order(binary: torch.Tensor, target_rms: float) -> torch.Tensor:
    q = binary.shape[0]
    coefficients = 0.5 ** torch.arange(q, dtype=torch.float32)
    reconstruction = torch.einsum("q,qrc->rc", coefficients, binary.float())
    return normalize_rows(reconstruction, target_rms)


def infer_structural_key(binary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = binary.float()
    q, rows, _ = values.shape
    transition = (values[:, :, 1:] != values[:, :, :-1]).float().mean(dim=2)
    rank_order = torch.argsort(transition, dim=0)
    predicted_pi = torch.empty_like(rank_order)
    predicted_pi.scatter_(0, rank_order, torch.arange(q)[:, None].expand(q, rows))
    basis_sum = values.sum(dim=2)
    predicted_s = torch.where(basis_sum != 0, basis_sum.sign(), values[:, :, 0])
    return predicted_s.T.to(torch.int8), predicted_pi.T.to(torch.int8)


def a3_structural_key_search(
    binary: torch.Tensor, target_rms: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted_s, predicted_pi = infer_structural_key(binary)
    coefficients = predicted_s.float() * (0.5 ** predicted_pi.float())
    reconstruction = torch.einsum("rq,qrc->rc", coefficients, binary.float())
    return normalize_rows(reconstruction, target_rms), predicted_s, predicted_pi


def a4_synthetic_prior(
    binary: torch.Tensor, model: torch.Tensor, target_rms: float
) -> torch.Tensor:
    features = basis_features(binary).double()
    coefficients = torch.einsum("qrd,d->rq", features, model).float()
    reconstruction = torch.einsum("rq,qrc->rc", coefficients, binary.float())
    return normalize_rows(reconstruction, target_rms)


def a5_random_base_projection(
    binary: torch.Tensor, target_rms: float, generator: torch.Generator
) -> torch.Tensor:
    rows, columns = binary.shape[1:]
    random_base = torch.randn((rows, columns), generator=generator) * target_rms
    _, _, reconstruction = fit_rowwise_coefficients(random_base, binary)
    return reconstruction.float()


def save_attack(
    output_dir: Path,
    name: str,
    tensors: dict[str, torch.Tensor],
    description: str,
) -> dict[str, str]:
    path = output_dir / f"w_attack_{name}.safetensors"
    save_file(
        {key: value.half().contiguous() for key, value in tensors.items()},
        path,
        metadata={"role": "blind_bstar_only_attack", "attack": name},
    )
    return {"file": path.name, "sha256": sha256_file(path), "description": description}


def main() -> None:
    args = parse_args()
    public_path = args.public_b_star.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not public_path.is_file():
        raise FileNotFoundError(public_path)
    if output_dir.is_dir() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator().manual_seed(args.seed)
    synthetic_models = {
        q: fit_synthetic_prior(
            q,
            args.synthetic_rows,
            args.synthetic_columns,
            args.ridge,
            generator,
        )
        for q in (6, 8)
    }
    attacks: dict[str, dict[str, torch.Tensor]] = {
        "a2_public_order": {},
        "a3_structural_key_search": {},
        "a4_synthetic_prior": {},
        "a5_random_base_projection": {},
    }
    predicted_keys: dict[str, torch.Tensor] = {}
    with safe_open(public_path, framework="pt", device="cpu") as public_file:
        names = sorted(public_file.keys())
        for matrix_index, name in enumerate(names, start=1):
            public = public_file.get_tensor(name)
            q, rows, columns = public.shape
            if q not in synthetic_models:
                raise ValueError(f"Unsupported public basis count Q={q} for {name}")
            target_rms = shape_only_scale(rows, columns)
            attacks["a2_public_order"][name] = a2_public_order(public, target_rms)
            a3, predicted_s, predicted_pi = a3_structural_key_search(public, target_rms)
            attacks["a3_structural_key_search"][name] = a3
            predicted_keys[f"{name}.S"] = predicted_s.contiguous()
            predicted_keys[f"{name}.pi"] = predicted_pi.contiguous()
            attacks["a4_synthetic_prior"][name] = a4_synthetic_prior(
                public, synthetic_models[q], target_rms
            )
            attacks["a5_random_base_projection"][name] = a5_random_base_projection(
                public, target_rms, generator
            )
            print(f"[{matrix_index:02d}/{len(names)}] generated blind attacks for {name}")

    descriptions = {
        "a2_public_order": "geometric residual weights assigned in observed B* order; shape-only scale",
        "a3_structural_key_search": "infer residual order from transition rates and signs from basis orientation",
        "a4_synthetic_prior": "ridge coefficient prior trained only on generic synthetic random matrices",
        "a5_random_base_projection": "project a seeded shape-only random matrix onto span(1, B*)",
    }
    artifacts = {
        name: save_attack(output_dir, name, tensors, descriptions[name])
        for name, tensors in attacks.items()
    }
    predicted_key_path = output_dir / "a3_predicted_s_pi.safetensors"
    save_file(predicted_keys, predicted_key_path, metadata={"role": "blind_attacker_key_estimate"})

    script_path = Path(__file__).resolve()
    snapshot_path = output_dir / "run_blind_bstar_attacks.snapshot.py"
    shutil.copyfile(script_path, snapshot_path)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "threat_model": {
            "name": "blind B*-only",
            "attacker_inputs": [
                "public B* values",
                "public tensor shapes",
                "published residual-binarization algorithm",
            ],
            "forbidden_inputs": [
                "GPT-2 identity, config, tokenizer, or pretrained checkpoint",
                "any pretrained/model-family weights",
                "W_true",
                "W_BCQ",
                "alpha",
                "z",
                "S",
                "pi",
            ],
            "scale_assumption": "shape-only Xavier RMS sqrt(2 / (rows + columns))",
            "model_assembly": "performed later by the evaluator, outside the attacker boundary",
        },
        "seed": args.seed,
        "synthetic_training": {
            "rows_per_q": args.synthetic_rows,
            "columns": args.synthetic_columns,
            "q_values": [6, 8],
            "ridge": args.ridge,
        },
        "storage_dtype": "float16",
        "methods": artifacts,
        "predicted_key": {
            "file": predicted_key_path.name,
            "sha256": sha256_file(predicted_key_path),
        },
        "public_b_star_sha256": sha256_file(public_path),
        "implementation_snapshot_sha256": sha256_file(snapshot_path),
    }
    (output_dir / "attack_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved four blind B*-only W_attack candidates to {output_dir}")


if __name__ == "__main__":
    main()
