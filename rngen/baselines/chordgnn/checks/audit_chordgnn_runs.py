"""Check that the selected ChordGNN checkpoint files exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


SEEDS = (42, 43, 44)


def audit_runs(
    *,
    run_dirs: Mapping[int, Path],
    output_path: Path,
    model_name: str = "ChordGNN",
) -> dict[str, Any]:
    """Check best.pt availability without loading weights or training records.

    Seed keys identify the supplied run directories; this does not verify how
    those checkpoints were trained. Evaluation can also run without this audit.
    """
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite audit {output_path}")
    if not run_dirs:
        raise ValueError("At least one run directory is required")
    records = {}
    for seed, run_dir in sorted(run_dirs.items()):
        checkpoint = run_dir / "checkpoints" / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint is missing: {checkpoint}")
        records[str(seed)] = {
            "run_dir": str(run_dir.resolve()),
            "best_checkpoint": str(checkpoint.resolve()),
            "exists": True,
        }
    payload = {
        "audit": f"{model_name} checkpoint availability",
        "status": "ready",
        "checks": ["checkpoint_file_exists"],
        "seeds": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return payload


def parse_args(description: str = __doc__) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    for seed in SEEDS:
        parser.add_argument(f"--seed-{seed}-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit_runs(
        run_dirs={seed: getattr(args, f"seed_{seed}_run") for seed in SEEDS},
        output_path=args.output,
    )
    print(json.dumps({"status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
