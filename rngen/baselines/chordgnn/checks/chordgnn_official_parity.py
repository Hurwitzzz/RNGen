"""Executable parity audit against the pinned public ChordGNN submission.

The upstream checkout is never imported as an installed package and is never
modified.  Its source modules are loaded from an explicitly supplied commit
checkout after their SHA-256 digests have been verified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn

from rngen.baselines.shared.chordgnn_provenance import full_provenance
from rngen.baselines.augnet_mi20.data.pdcml_augmentednet_converter import COMPONENTS
from rngen.baselines.chordgnn.data.pdcml_chordgnn_converter import (
    FEATURE_NAMES,
    UPSTREAM_COMMIT,
    _create_divisions,
    _graph_edges,
    _note_features,
    _tie_consecutive_notes,
)
from rngen.baselines.chordgnn.model import (
    ChordGNN,
    ChordGraphBatch,
    add_reverse_edges,
)


FIXTURE_NAMES = (
    "wir-bach-chorales-8.tsv",
    "bps-09-op014-no1-1.tsv",
    "abc-op18-no6-4.tsv",
)
UPSTREAM_SOURCE_HASHES = {
    "chordgnn/utils/chord_representations.py": "7accd7a2ff1a9aa9ba38c235d7a80587cbc4d211aacd46b2c8de8dbeb467fadc",
    "chordgnn/utils/hgraph.py": "7ac61f3078d9938e7cf296556a751dbd3a2e98658f4261aca7b0a739473858b6",
    "chordgnn/descriptors/utils/note_features.py": "d63094d2899942c0727b702913ba9669a3c1d6e18aa7f9f91f1934311e93d06e",
    "chordgnn/models/core/gnn.py": "df00f113d16f1f0e623fa747756ed8988aadc47ed94bc065c3e67200aa44d094",
    "chordgnn/models/core/hgnn.py": "65c1eb70aa0929e0a8fb6c62bd490efd2e17dc958509dc0d0a6a4eb37db20fe4",
    "chordgnn/models/core/mlp.py": "0afeaa82777ecf7b16f85afe7d572d3bc7a16997583b380933156855e45dfac2",
    "chordgnn/models/chord.py": "27782df351aadee3b1955a6c37926b739321611091e573b648653cec311c57b9",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pinned upstream module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _package(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module
    return module


def _load_upstream(source_root: Path) -> dict[str, Any]:
    for relative, expected in UPSTREAM_SOURCE_HASHES.items():
        path = source_root / relative
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(
                f"Pinned upstream source hash mismatch for {relative}: {observed}"
            )

    chordgnn = _package("chordgnn", source_root / "chordgnn")
    utils = _package("chordgnn.utils", source_root / "chordgnn/utils")
    descriptors = _package("chordgnn.descriptors", source_root / "chordgnn/descriptors")
    _package(
        "chordgnn.descriptors.utils",
        source_root / "chordgnn/descriptors/utils",
    )
    models = _package("chordgnn.models", source_root / "chordgnn/models")
    core = _package("chordgnn.models.core", source_root / "chordgnn/models/core")
    chordgnn.utils = utils
    chordgnn.descriptors = descriptors
    chordgnn.models = models

    _load_module(
        "chordgnn.utils.globals",
        source_root / "chordgnn/utils/globals.py",
    )
    _load_module(
        "chordgnn.utils.general",
        source_root / "chordgnn/utils/general.py",
    )
    _load_module(
        "chordgnn.utils.chord_representations_latest",
        source_root / "chordgnn/utils/chord_representations_latest.py",
    )
    representations = _load_module(
        "chordgnn.utils.chord_representations",
        source_root / "chordgnn/utils/chord_representations.py",
    )
    utils.chord_to_intervalVector = representations.chord_to_intervalVector
    utils.time_divided_tsv_to_note_array = (
        representations.time_divided_tsv_to_note_array
    )
    utils.time_divided_tsv_to_part = representations.time_divided_tsv_to_part

    note_features = _load_module(
        "chordgnn.descriptors.utils.note_features",
        source_root / "chordgnn/descriptors/utils/note_features.py",
    )
    descriptor_utils = sys.modules["chordgnn.descriptors.utils"]
    for name in dir(note_features):
        if not name.startswith("_"):
            setattr(descriptor_utils, name, getattr(note_features, name))
    descriptor_general = _load_module(
        "chordgnn.descriptors.general",
        source_root / "chordgnn/descriptors/general.py",
    )
    hgraph = _load_module(
        "chordgnn.utils.hgraph",
        source_root / "chordgnn/utils/hgraph.py",
    )

    gnn = _load_module(
        "chordgnn.models.core.gnn",
        source_root / "chordgnn/models/core/gnn.py",
    )
    mlp = _load_module(
        "chordgnn.models.core.mlp",
        source_root / "chordgnn/models/core/mlp.py",
    )
    for name in dir(gnn):
        if not name.startswith("_"):
            setattr(core, name, getattr(gnn, name))
    for name in dir(mlp):
        if not name.startswith("_"):
            setattr(core, name, getattr(mlp, name))
    hgnn = _load_module(
        "chordgnn.models.core.hgnn",
        source_root / "chordgnn/models/core/hgnn.py",
    )
    for name in dir(hgnn):
        if not name.startswith("_"):
            setattr(core, name, getattr(hgnn, name))

    torch_sparse = types.ModuleType("torch_sparse")
    torch_sparse.coalesce = lambda *args, **kwargs: args[0]
    sys.modules["torch_sparse"] = torch_sparse
    lightning = types.ModuleType("pytorch_lightning")
    lightning.LightningModule = nn.Module
    sys.modules["pytorch_lightning"] = lightning
    metrics = _package("chordgnn.metrics", source_root / "chordgnn/metrics")
    metrics_eval = types.ModuleType("chordgnn.metrics.eval")
    metrics_eval.MultitaskAccuracy = object
    sys.modules["chordgnn.metrics.eval"] = metrics_eval
    metrics.eval = metrics_eval
    sys.modules["rotograd"] = types.ModuleType("rotograd")
    chord = _load_module(
        "chordgnn.models.chord",
        source_root / "chordgnn/models/chord.py",
    )
    return {
        "representations": representations,
        "note_features": note_features,
        "descriptor_general": descriptor_general,
        "hgraph": hgraph,
        "chord": chord,
    }


def _frame_from_note_array(note_array: np.ndarray) -> pd.DataFrame:
    spelling = []
    for step, alteration in zip(note_array["step"], note_array["alter"]):
        accidental = (
            "#" * int(alteration) if alteration >= 0 else "-" * -int(alteration)
        )
        spelling.append(f"{step}{accidental}")
    return pd.DataFrame(
        {
            "pitch": note_array["pitch"],
            "spelling": spelling,
            "onset": note_array["onset_beat"],
            "release": note_array["onset_beat"] + note_array["duration_beat"],
            "ts_numerator": note_array["ts_beats"],
            "ts_denominator": note_array["ts_beat_type"],
        }
    )


def _copy_model_weights(local: ChordGNN, upstream: nn.Module) -> None:
    encoder = upstream.encoder
    encoder.pitch_embedding.load_state_dict(local.pitch_embedding.state_dict())
    encoder.spelling_embedding.load_state_dict(local.spelling_embedding.state_dict())
    encoder.embedding.load_state_dict(local.feature_embedding.state_dict())
    relation_map = {"silence": "rests", "silence_rev": "rests_rev"}
    for local_layer, upstream_layer in zip(
        local.encoder.layers,
        encoder.encoder.layers,
    ):
        for relation in local_layer.convs:
            upstream_layer.conv[relation_map.get(relation, relation)].load_state_dict(
                local_layer.convs[relation].state_dict()
            )
    encoder.pool.load_state_dict(local.pool.state_dict())
    encoder.proj1.load_state_dict(local.proj1.state_dict())
    encoder.layernorm1.load_state_dict(local.batch_norm1.state_dict())
    encoder.proj2.load_state_dict(local.proj2.state_dict())
    encoder.layernorm2.load_state_dict(local.batch_norm2.state_dict())
    encoder.gru.load_state_dict(local.gru.state_dict())
    encoder.layernormgru.load_state_dict(local.gru_norm.state_dict())
    for component in COMPONENTS:
        official_head = upstream.classifier.classifier[component]
        official_head.layers[0].load_state_dict(
            local.classifiers[component].first.state_dict()
        )
        official_head.normalize.load_state_dict(
            local.classifiers[component].batch_norm.state_dict()
        )
        official_head.layers[1].load_state_dict(
            local.classifiers[component].output.state_dict()
        )


def _parity_piece(path: Path, upstream: Mapping[str, Any]) -> dict[str, Any]:
    representations = upstream["representations"]
    raw_notes, raw_labels = representations.time_divided_tsv_to_note_array(
        str(path), transpose=False
    )
    official_notes, official_labels = representations.tie_consecutive_notes(
        raw_notes, raw_labels
    )
    official_notes = representations.create_divs_from_beats(official_notes)
    official_features = (
        upstream["descriptor_general"]
        .select_features(official_notes, "chord")
        .astype(np.float32)
    )
    _, official_edges = upstream["hgraph"].hetero_graph_from_note_array(
        official_notes.copy()
    )

    local_tied, tied_count = _tie_consecutive_notes(_frame_from_note_array(raw_notes))
    onset_div, duration_div, divisions = _create_divisions(local_tied)
    local_tied = local_tied.assign(onset_div=onset_div, duration_div=duration_div)
    local_features = _note_features(local_tied)
    local_edge_index, local_edge_type = _graph_edges(onset_div, duration_div)
    local_edges = np.vstack((local_edge_index, local_edge_type))

    np.testing.assert_array_equal(
        local_tied["pitch"].to_numpy(), official_notes["pitch"]
    )
    np.testing.assert_array_equal(onset_div, official_notes["onset_div"])
    np.testing.assert_array_equal(duration_div, official_notes["duration_div"])
    np.testing.assert_allclose(
        local_tied["onset_beat"].to_numpy(),
        official_notes["onset_beat"],
        atol=0,
        rtol=0,
    )
    np.testing.assert_allclose(
        local_tied["duration_beat"].to_numpy(),
        official_notes["duration_beat"],
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(local_features, official_features, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(local_edges, official_edges)
    np.testing.assert_array_equal(
        np.unique(onset_div), np.unique(official_notes["onset_div"])
    )
    np.testing.assert_array_equal(
        official_labels[:, -1].reshape(-1), np.unique(official_notes["onset_beat"])
    )

    return {
        "fixture": path.name,
        "fixture_sha256": _sha256(path),
        "raw_nodes": int(len(raw_notes)),
        "tied_nodes": int(len(official_notes)),
        "tied_rows": int(tied_count),
        "onsets": int(len(np.unique(onset_div))),
        "labels": int(len(official_labels)),
        "forward_edges": int(official_edges.shape[1]),
        "relations": {
            str(relation): int((official_edges[2] == relation).sum())
            for relation in range(4)
        },
        "divisions_per_quarter": int(divisions),
        "max_feature_abs_error": float(
            np.max(np.abs(local_features - official_features))
        ),
        "_arrays": {
            "features": local_features,
            "edge_index": local_edge_index,
            "edge_type": local_edge_type,
            "onset_div": onset_div,
        },
    }


def _logit_parity(
    fixture: Mapping[str, Any],
    upstream: Mapping[str, Any],
) -> dict[str, Any]:
    arrays = fixture["_arrays"]
    features = torch.from_numpy(arrays["features"])
    edge_index = torch.from_numpy(arrays["edge_index"])
    edge_type = torch.from_numpy(arrays["edge_type"].astype(np.int64))
    onset_edges = edge_index[:, edge_type == 0]
    edge_index, edge_type = add_reverse_edges(edge_index, edge_type)
    onset_div = arrays["onset_div"]
    _, representatives = np.unique(onset_div, return_index=True)
    representatives_tensor = torch.from_numpy(representatives.astype(np.int64))
    length = len(representatives)
    head_sizes = {
        "local_key": 37,
        "primary_degree": 22,
        "secondary_degree": 23,
        "inversion": 4,
        "quality": 16,
        "chord_root": 35,
        "bass": 35,
        "roman_numeral_76": 76,
        "tonicized_key": 35,
        "pitch_class_set": 94,
        "harmonic_rhythm": 2,
    }
    torch.manual_seed(1729)
    local = ChordGNN(head_sizes, hidden_size=256, n_layers=1, dropout=0.5).cpu()
    official = (
        upstream["chord"]
        .ChordPredictionModel(
            83,
            n_hidden=256,
            tasks=head_sizes,
            n_layers=1,
            dropout=0.5,
            use_nade=False,
            use_jk=False,
        )
        .cpu()
    )
    _copy_model_weights(local, official)
    local.eval()
    official.eval()
    labels = {
        component: torch.zeros((1, length), dtype=torch.long)
        for component in COMPONENTS
    }
    masks = {
        component: torch.ones((1, length), dtype=torch.bool) for component in COMPONENTS
    }
    batch = ChordGraphBatch(
        x=features,
        edge_index=edge_index,
        edge_type=edge_type,
        onset_edges=onset_edges,
        onset_representatives=representatives_tensor,
        lengths=torch.tensor([length]),
        labels=labels,
        masks=masks,
        onset_quarter=(torch.from_numpy(np.unique(onset_div)),),
        relative_paths=(str(fixture["fixture"]),),
    )
    with torch.no_grad():
        local_logits = local(batch)
        official_logits = official(
            (
                features,
                edge_index,
                edge_type,
                onset_edges,
                representatives_tensor,
                None,
            )
        )
    errors = {}
    for component in COMPONENTS:
        observed = local_logits[component][0]
        expected = official_logits[component]
        torch.testing.assert_close(observed, expected, atol=1e-6, rtol=1e-5)
        errors[component] = float((observed - expected).abs().max().item())
    return {
        "fixture": fixture["fixture"],
        "atol": 1e-6,
        "rtol": 1e-5,
        "max_abs_error_by_head": errors,
    }


def run_parity(
    source_root: Path, fixture_dir: Path, output_path: Path
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite parity audit {output_path}")
    if not (source_root / ".git").exists():
        raise ValueError(f"Upstream source is not a git checkout: {source_root}")
    import subprocess

    commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Expected upstream {UPSTREAM_COMMIT}, got {commit}")
    upstream = _load_upstream(source_root)
    pieces = []
    raw_results = []
    for name in FIXTURE_NAMES:
        path = fixture_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result = _parity_piece(path, upstream)
        raw_results.append(result)
        pieces.append({key: value for key, value in result.items() if key != "_arrays"})
    logit = _logit_parity(raw_results[0], upstream)
    payload = {
        "audit": "ChordGNN official bcbdc72 CPU parity",
        "provenance": full_provenance(require_git=False),
        "status": "passed",
        "upstream_commit": commit,
        "source_root": str(source_root.resolve()),
        "source_hashes": UPSTREAM_SOURCE_HASHES,
        "feature_names": list(FEATURE_NAMES),
        "feature_tolerance": {"atol": 1e-7, "rtol": 0},
        "pieces": pieces,
        "logit_parity": logit,
        "torch": torch.__version__,
        "numpy": np.__version__,
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
        "--fixture-dir",
        type=Path,
        required=True,
        help="Directory containing the three upstream TSV fixtures.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_parity(args.source_root, args.fixture_dir, args.output)
    print(json.dumps({"status": payload["status"], "pieces": len(payload["pieces"])}))


if __name__ == "__main__":
    main()
