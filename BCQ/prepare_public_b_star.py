"""Create a keyed public B* artifact without exposing the secret row keys."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from extract_gpt2_weights import sha256_file
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def obfuscate_rows(
    binary: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, rows, _ = binary.shape
    permutation = torch.stack([torch.randperm(q, generator=generator) for _ in range(rows)])
    signs = torch.randint(0, 2, (rows, q), generator=generator, dtype=torch.int8) * 2 - 1
    row_indices = torch.arange(rows)
    public = torch.empty_like(binary)
    for public_index in range(q):
        public[public_index] = binary[permutation[:, public_index], row_indices] * signs[:, public_index, None]
    return public, signs, permutation.to(torch.int8)


def main() -> None:
    args = parse_args()
    public_b_path = args.public_b.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    public_dir = output_dir / "public"
    secret_dir = output_dir / "secret"
    if not public_b_path.is_file():
        raise FileNotFoundError(public_b_path)
    if output_dir.is_dir() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    public_dir.mkdir(parents=True, exist_ok=True)
    secret_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator().manual_seed(args.seed)
    public_b_star: dict[str, torch.Tensor] = {}
    secret_keys: dict[str, torch.Tensor] = {}
    with safe_open(public_b_path, framework="pt", device="cpu") as source:
        for name in source.keys():
            binary = source.get_tensor(name)
            transformed, signs, permutation = obfuscate_rows(binary, generator)
            public_b_star[name] = transformed
            secret_keys[f"{name}.S"] = signs
            secret_keys[f"{name}.pi"] = permutation

    public_path = public_dir / "public_b_star.safetensors"
    secret_path = secret_dir / "secret_s_pi.safetensors"
    metadata = {
        "format": "gpt2-row-wise-keyed-bcq-v1",
        "transformation": "B_star[k,i,:] = S[i,k] * B[pi[i,k],i,:]",
    }
    save_file(public_b_star, public_path, metadata={**metadata, "role": "public"})
    save_file(secret_keys, secret_path, metadata={**metadata, "role": "secret"})

    script_path = Path(__file__).resolve()
    snapshot_path = output_dir / "prepare_public_b_star.snapshot.py"
    shutil.copyfile(script_path, snapshot_path)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "seed": args.seed,
        "matrix_count": len(public_b_star),
        "key_granularity": "per_matrix_per_row_per_basis",
        "transformation": metadata["transformation"],
        "threat_boundary": {
            "public": ["public/public_b_star.safetensors"],
            "secret": [
                "secret/secret_s_pi.safetensors",
                "source BCQ alpha and z artifact",
            ],
        },
        "artifacts": {
            "source_public_b": {
                "path": str(public_b_path),
                "sha256": sha256_file(public_b_path),
            },
            "public_b_star": {
                "path": str(public_path.relative_to(output_dir)),
                "sha256": sha256_file(public_path),
            },
            "secret_s_pi": {
                "path": str(secret_path.relative_to(output_dir)),
                "sha256": sha256_file(secret_path),
            },
            "implementation_snapshot": {
                "path": snapshot_path.name,
                "sha256": sha256_file(snapshot_path),
            },
        },
    }
    (output_dir / "threat_model.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Created public B* with {len(public_b_star)} matrices: {public_path}")
    print(f"Stored S/pi behind the secret boundary: {secret_path}")


if __name__ == "__main__":
    main()
