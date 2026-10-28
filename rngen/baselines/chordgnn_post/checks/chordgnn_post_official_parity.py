"""CPU parity audit for ChordGNN +Post against upstream commit bcbdc72."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from rngen.baselines.chordgnn.checks.chordgnn_official_parity import (
    UPSTREAM_SOURCE_HASHES,
    _load_upstream,
    _sha256,
)
from rngen.baselines.shared.chordgnn_provenance import full_provenance
from rngen.baselines.augnet_mi20.data.pdcml_augmentednet_converter import COMPONENTS
from rngen.baselines.chordgnn.data.pdcml_chordgnn_converter import UPSTREAM_COMMIT
from rngen.baselines.chordgnn_post.model import ChordGNNPostProcessor


def run_parity(
    *,
    source_root: Path,
    ontology_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite Post parity audit {output_path}")
    commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Expected upstream {UPSTREAM_COMMIT}, got {commit}")
    upstream = _load_upstream(source_root)
    ontology = json.loads(ontology_path.read_text())
    vocabularies = ontology["vocabularies"]
    head_sizes = {component: len(vocabularies[component]) for component in COMPONENTS}
    if sum(head_sizes.values()) != 379:
        raise ValueError("Official Post parity requires the 379-class ontology")

    torch.manual_seed(1729)
    local = ChordGNNPostProcessor(
        head_sizes,
        hidden_size=256,
        n_layers=1,
        dropout=0.4,
    ).cpu()
    official = (
        upstream["chord"]
        .PostProcessingMLTModel(
            head_sizes,
            256,
            1,
            dropout=0.4,
        )
        .cpu()
    )
    official.lstm.load_state_dict(local.lstm.state_dict())
    official.fc.load_state_dict(local.projection.state_dict())
    for component in COMPONENTS:
        official.clf[component].load_state_dict(
            local.classifiers[component].state_dict()
        )
    local.eval()
    official.eval()
    generator = torch.Generator(device="cpu").manual_seed(314159)
    base_logits = {
        component: torch.randn(
            2,
            7,
            head_sizes[component],
            generator=generator,
        )
        for component in COMPONENTS
    }
    lengths = torch.tensor([7, 5], dtype=torch.long)
    with torch.no_grad():
        observed = local(base_logits, lengths)
        expected = official(base_logits)
    errors: dict[str, float] = {}
    for component in COMPONENTS:
        torch.testing.assert_close(
            observed[component],
            expected[component],
            atol=1e-6,
            rtol=1e-5,
        )
        errors[component] = float(
            (observed[component] - expected[component]).abs().max().item()
        )
    payload = {
        "audit": "ChordGNN+Post official bcbdc72 CPU parity",
        "status": "passed",
        "upstream_commit": commit,
        "source_root": str(source_root.resolve()),
        "source_hashes": UPSTREAM_SOURCE_HASHES,
        "ontology": str(ontology_path.resolve()),
        "ontology_sha256": _sha256(ontology_path),
        "head_order": list(COMPONENTS),
        "head_sizes": head_sizes,
        "input_size": sum(head_sizes.values()),
        "batch_shape": [2, 7],
        "unpacked_lstm": True,
        "atol": 1e-6,
        "rtol": 1e-5,
        "max_abs_error_by_head": errors,
        "provenance": full_provenance(require_git=False),
        "torch": torch.__version__,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Local ChordGNN checkout at the pinned upstream commit.",
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        required=True,
        help="Cache ontology JSON containing the official 379-class vocabularies.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_parity(
        source_root=args.source_root,
        ontology_path=args.ontology,
        output_path=args.output,
    )
    print(json.dumps({"status": payload["status"], "input_size": 379}))


if __name__ == "__main__":
    main()
