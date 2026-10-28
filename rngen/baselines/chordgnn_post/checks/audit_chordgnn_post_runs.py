"""Check that the selected ChordGNN +Post checkpoint files exist."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from rngen.baselines.chordgnn.checks.audit_chordgnn_runs import SEEDS, audit_runs as audit_checkpoints
from rngen.baselines.chordgnn.checks.audit_chordgnn_runs import parse_args


def audit_runs(
    *, run_dirs: Mapping[int, Path], output_path: Path
) -> dict[str, Any]:
    """Check Post best.pt files; loading the paired base is an evaluation step."""
    return audit_checkpoints(
        run_dirs=run_dirs, output_path=output_path, model_name="ChordGNN+Post"
    )


def main() -> None:
    args = parse_args(__doc__)
    payload = audit_runs(
        run_dirs={seed: getattr(args, f"seed_{seed}_run") for seed in SEEDS},
        output_path=args.output,
    )
    print(json.dumps({"status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
