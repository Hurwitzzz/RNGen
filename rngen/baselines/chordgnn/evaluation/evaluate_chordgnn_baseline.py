"""Evaluate a controlled ChordGNN checkpoint at exact source-note onsets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from rngen.paths import PROCESSED_SCORES_DIR

from rngen.baselines.shared.chordgnn_provenance import full_provenance
from rngen.baselines.chordgnn.evaluation.chordgnn_diagnostics import score_paper_fixed_grid
from rngen.baselines.chordgnn.evaluation.chordgnn_classwise import classwise_exact_onsets
from rngen.baselines.shared.exact_onset_evaluation import (
    PredictionTimeline,
    merge_piece_reports,
    score_exact_onsets,
)
from rngen.baselines.augnet_mi20.data.pdcml_augmentednet_converter import (
    COMPONENTS,
    CORE_COMPONENTS,
    build_vocabularies,
    convert_piece as convert_reference_piece,
    read_split_paths,
)
from rngen.baselines.chordgnn.data.pdcml_chordgnn_converter import (
    convert_piece as convert_graph_piece,
)
from rngen.baselines.chordgnn.model import ChordGNN
from rngen.baselines.chordgnn.trainer import collate_chord_graphs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _prepare_output(output_dir: Path, dataset_root: Path) -> None:
    try:
        output_dir.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("Refusing to write evaluation inside the source root")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _read_evaluation_paths(manifest: Path) -> list[str]:
    paths = read_split_paths(manifest)
    if not paths:
        raise ValueError(f"Evaluation manifest is empty: {manifest}")
    if len(paths) != len(set(paths)):
        raise ValueError(f"Evaluation manifest contains duplicates: {manifest}")
    return paths


@torch.no_grad()
def evaluate(
    *,
    checkpoint: Path,
    run_config_path: Path,
    ontology_path: Path,
    dataset_root: Path,
    split_dir: Path,
    manifest: Path,
    split: str,
    output_dir: Path,
    seed: int,
    device_name: str,
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("ChordGNN evaluation split must be validation or test")
    ontology = json.loads(ontology_path.read_text())
    vocabularies = ontology["vocabularies"]
    canonical_vocabularies, _, _ = build_vocabularies(dataset_root, split_dir)
    if canonical_vocabularies != vocabularies:
        raise ValueError("Evaluation ontology differs from the corpus ontology")
    run_config = json.loads(run_config_path.read_text())
    configured_seed = int(run_config["training"]["seed"])
    if configured_seed != seed:
        raise ValueError(
            f"Requested seed {seed} differs from run seed {configured_seed}"
        )
    evaluation_provenance = full_provenance(require_git=False)
    ontology_hash = _sha256(ontology_path)
    if run_config.get("ontology_sha256") != ontology_hash:
        raise ValueError("Evaluation ontology hash differs from the run config")
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    checkpoint_payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if checkpoint_payload.get("config_sha256") != run_config.get("config_sha256"):
        raise ValueError("Checkpoint config hash differs from the run config")
    if checkpoint_payload.get("ontology_sha256") != ontology_hash:
        raise ValueError(
            "Checkpoint ontology hash differs from the evaluation ontology"
        )
    _prepare_output(output_dir, dataset_root)
    head_sizes = {component: len(vocabularies[component]) for component in COMPONENTS}
    model = ChordGNN(
        head_sizes,
        hidden_size=int(run_config["training"]["hidden_size"]),
        n_layers=int(run_config["training"]["n_layers"]),
        dropout=float(run_config["training"]["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint_payload["model_state"])
    model.eval()
    relative_paths = _read_evaluation_paths(manifest)
    piece_reports: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    paper_grid_reports: list[dict[str, Any]] = []
    native_core_correct = 0
    native_core_support = 0
    native_counts = {
        component: {"correct": 0, "support": 0} for component in COMPONENTS
    }
    for index, relative_path in enumerate(relative_paths, start=1):
        source_path = dataset_root / relative_path
        graph = convert_graph_piece(
            source_path,
            dataset_root=dataset_root,
            split_dir=split_dir,
            relative_path=relative_path,
            vocabularies=vocabularies,
        )
        batch = collate_chord_graphs([graph]).to(device)
        logits = model(batch)
        decoded = {
            component: np.asarray(
                [
                    vocabularies[component][class_index]
                    for class_index in logits[component][0, : len(graph.onset_quarter)]
                    .argmax(dim=-1)
                    .detach()
                    .cpu()
                    .tolist()
                ],
                dtype=str,
            )
            for component in COMPONENTS
        }
        timeline = PredictionTimeline(
            offsets=graph.onset_quarter,
            labels=decoded,
            source_resolution="ChordGNN native graph onsets with zero-order hold",
        )
        reference = convert_reference_piece(source_path, relative_path=relative_path)
        report, records = score_exact_onsets(reference, timeline)
        paper_grid = score_paper_fixed_grid(reference, timeline)
        report["paper_1_32_diagnostic"] = paper_grid
        paper_grid_reports.append(paper_grid)
        piece_reports.append(report)
        prediction_records.extend(records)
        core_mask = np.logical_and.reduce(
            [graph.masks[component] for component in CORE_COMPONENTS]
        )
        core_correct = core_mask.copy()
        for component in CORE_COMPONENTS:
            component_prediction = (
                logits[component][0, : len(graph.onset_quarter)]
                .argmax(dim=-1)
                .detach()
                .cpu()
                .numpy()
            )
            core_correct &= component_prediction == graph.labels[component]
        native_core_correct += int(core_correct.sum())
        native_core_support += int(core_mask.sum())
        for component in COMPONENTS:
            mask = graph.masks[component]
            prediction = (
                logits[component][0, : len(graph.onset_quarter)]
                .argmax(dim=-1)
                .detach()
                .cpu()
                .numpy()
            )
            native_counts[component]["correct"] += int(
                (prediction[mask] == graph.labels[component][mask]).sum()
            )
            native_counts[component]["support"] += int(mask.sum())
        print(
            f"[{split}] {index}/{len(relative_paths)} {relative_path}",
            flush=True,
        )
    aggregate = merge_piece_reports(piece_reports)
    paper_counts = {
        key: sum(item["counts"][key] for item in paper_grid_reports)
        for key in paper_grid_reports[0]["counts"]
    }
    paper_support = paper_counts["support"]
    aggregate["paper_1_32_diagnostic"] = {
        "definition": paper_grid_reports[0]["definition"],
        "counts": paper_counts,
        "metrics": {
            "CSR": paper_counts["rna_correct"] / paper_support,
            "K_acc": paper_counts["key_correct"] / paper_support,
            "D_acc": paper_counts["degree_correct"] / paper_support,
            "Q_acc": paper_counts["quality_correct"] / paper_support,
            "I_acc": paper_counts["inversion_correct"] / paper_support,
        },
    }
    aggregate.update(
        {
            "audit": "controlled ChordGNN exact-onset evaluation",
            "provenance": evaluation_provenance,
            "split": split,
            "seed": seed,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_identity": {
                "epoch": int(checkpoint_payload["epoch"]),
                "config_sha256": checkpoint_payload["config_sha256"],
                "ontology_sha256": checkpoint_payload["ontology_sha256"],
            },
            "run_config": str(run_config_path.resolve()),
            "run_config_sha256": _sha256(run_config_path),
            "ontology": str(ontology_path.resolve()),
            "ontology_sha256": _sha256(ontology_path),
            "manifest": str(manifest.resolve()),
            "manifest_sha256": _sha256(manifest),
            "class_index_decoder_failures": 0,
            "semantic_component_decode_failures": aggregate["counts"][
                "prediction_decode_failures"
            ],
            "metric_definition": (
                "RNA is the conjunction of local key, both degrees, quality, "
                "inversion, and chord root at every annotated source-note onset"
            ),
            "native_graph_metrics": {
                "RNA": {
                    "correct": native_core_correct,
                    "support": native_core_support,
                    "accuracy": native_core_correct / native_core_support,
                },
                **{
                    component: {
                        **counts,
                        "accuracy": (
                            counts["correct"] / counts["support"]
                            if counts["support"]
                            else 0.0
                        ),
                    }
                    for component, counts in native_counts.items()
                },
            },
        }
    )
    _write_json(output_dir / "onset_metrics.json", aggregate)
    _write_json(output_dir / "per_piece_metrics.json", piece_reports)
    _write_json(
        output_dir / "per_class_metrics.json",
        classwise_exact_onsets(prediction_records),
    )
    failures = [record for record in prediction_records if not record["correct"]["rna"]]
    _write_json(output_dir / "failure_cases.json", failures[:200])
    with (output_dir / "onset_predictions.jsonl").open("w") as handle:
        for record in prediction_records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROCESSED_SCORES_DIR,
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data_splits/p_dcml"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(
        checkpoint=args.checkpoint,
        run_config_path=args.run_config,
        ontology_path=args.ontology,
        dataset_root=args.dataset_root,
        split_dir=args.split_dir,
        manifest=args.manifest,
        split=args.split,
        output_dir=args.output_dir,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps(result["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
