"""Test-only exact-onset evaluation for ChordGNN +Post checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rngen.paths import PROCESSED_SCORES_DIR

from rngen.baselines.chordgnn.evaluation.chordgnn_classwise import classwise_exact_onsets
from rngen.baselines.chordgnn.evaluation.chordgnn_diagnostics import score_paper_fixed_grid
from rngen.baselines.shared.chordgnn_provenance import full_provenance
from rngen.baselines.chordgnn.evaluation.evaluate_chordgnn_baseline import (
    _prepare_output,
    _read_evaluation_paths,
    _sha256,
    _write_json,
)
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
)
from rngen.baselines.chordgnn.data.pdcml_chordgnn_converter import (
    convert_piece as convert_graph_piece,
)
from rngen.baselines.chordgnn_post.model import ChordGNNPostProcessor
from rngen.baselines.chordgnn_post.trainer import load_frozen_base
from rngen.baselines.chordgnn.trainer import collate_chord_graphs


def _validate_checkpoint(
    *,
    checkpoint: Path,
    run_config_path: Path,
    ontology_path: Path,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    ontology = json.loads(ontology_path.read_text())
    ontology_hash = _sha256(ontology_path)
    run_config = json.loads(run_config_path.read_text())
    training = run_config["training"]
    if int(training["seed"]) != seed:
        raise ValueError("Requested Post seed differs from the Post run config")
    if run_config.get("ontology_sha256") != ontology_hash:
        raise ValueError("Post test ontology differs from the run")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = {
        "config_sha256": run_config["config_sha256"],
        "ontology_sha256": ontology_hash,
        "base_checkpoint_sha256": run_config["base_model"]["checkpoint_sha256"],
        "base_config_sha256": run_config["base_model"]["config_sha256"],
        "base_state_sha256": run_config["base_model"]["state_sha256"],
        "base_seed": int(training["base_seed"]),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Post test checkpoint mismatch for {key}")
    return ontology, run_config, payload, ontology_hash


@torch.no_grad()
def evaluate(
    *,
    checkpoint: Path,
    run_config_path: Path,
    ontology_path: Path,
    dataset_root: Path,
    split_dir: Path,
    manifest: Path,
    output_dir: Path,
    seed: int,
    device_name: str,
    subset: str,
) -> dict[str, Any]:
    ontology, run_config, checkpoint_payload, ontology_hash = _validate_checkpoint(
        checkpoint=checkpoint,
        run_config_path=run_config_path,
        ontology_path=ontology_path,
        seed=seed,
    )
    evaluation_provenance = full_provenance(require_git=False)
    vocabularies = ontology["vocabularies"]
    canonical_vocabularies, _, _ = build_vocabularies(dataset_root, split_dir)
    if canonical_vocabularies != vocabularies:
        raise ValueError("Post test ontology differs from the corpus ontology")
    relative_paths = _read_evaluation_paths(manifest)
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    training = run_config["training"]
    base_model, base_identity = load_frozen_base(
        checkpoint_path=Path(training["base_checkpoint"]),
        run_config_path=Path(training["base_run_config"]),
        ontology=ontology,
        ontology_hash=ontology_hash,
        expected_seed=int(training["base_seed"]),
        device=device,
        require_completed_formal_run=False,
    )
    for key in (
        "checkpoint_sha256",
        "config_sha256",
        "ontology_sha256",
        "state_sha256",
        "seed",
    ):
        if base_identity[key] != run_config["base_model"][key]:
            raise ValueError(f"Post test base identity mismatch for {key}")

    head_sizes = {component: len(vocabularies[component]) for component in COMPONENTS}
    post_model = ChordGNNPostProcessor(
        head_sizes,
        hidden_size=int(training["hidden_size"]),
        n_layers=int(training["n_layers"]),
        dropout=float(training["dropout"]),
    ).to(device)
    post_model.load_state_dict(checkpoint_payload["post_state"])
    post_model.eval()
    _prepare_output(output_dir, dataset_root)

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
        base_logits = base_model(batch)
        logits = post_model(base_logits, batch.lengths)
        length = len(graph.onset_quarter)
        decoded = {
            component: np.asarray(
                [
                    vocabularies[component][class_index]
                    for class_index in logits[component][0, :length]
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
            source_resolution=(
                "ChordGNN+Post native graph onsets with zero-order hold"
            ),
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
            prediction = logits[component][0, :length].argmax(dim=-1).cpu().numpy()
            core_correct &= prediction == graph.labels[component]
        native_core_correct += int(core_correct.sum())
        native_core_support += int(core_mask.sum())
        for component in COMPONENTS:
            mask = graph.masks[component]
            prediction = logits[component][0, :length].argmax(dim=-1).cpu().numpy()
            native_counts[component]["correct"] += int(
                (prediction[mask] == graph.labels[component][mask]).sum()
            )
            native_counts[component]["support"] += int(mask.sum())
        print(
            f"[test +Post {subset}] "
            f"{index}/{len(relative_paths)} {relative_path}",
            flush=True,
        )

    aggregate = merge_piece_reports(piece_reports)
    paper_counts = {
        key: sum(report["counts"][key] for report in paper_grid_reports)
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
            "audit": "controlled ChordGNN+Post exact-onset test evaluation",
            "provenance": evaluation_provenance,
            "split": "test",
            "test_subset": subset,
            "base_seed": int(training["base_seed"]),
            "seed": seed,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_identity": {
                "epoch": int(checkpoint_payload["epoch"]),
                "config_sha256": checkpoint_payload["config_sha256"],
                "ontology_sha256": checkpoint_payload["ontology_sha256"],
                "base_checkpoint_sha256": checkpoint_payload["base_checkpoint_sha256"],
            },
            "run_config": str(run_config_path.resolve()),
            "run_config_sha256": _sha256(run_config_path),
            "ontology": str(ontology_path.resolve()),
            "ontology_sha256": ontology_hash,
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
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--subset", required=True, help="Name recorded in the test report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate = evaluate(
        checkpoint=args.checkpoint,
        run_config_path=args.run_config,
        ontology_path=args.ontology,
        dataset_root=args.dataset_root,
        split_dir=args.split_dir,
        manifest=args.manifest,
        output_dir=args.output_dir,
        seed=args.seed,
        device_name=args.device,
        subset=args.subset,
    )
    print(json.dumps(aggregate["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
