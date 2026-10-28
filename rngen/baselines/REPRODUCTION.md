# Reproduce the baselines

This guide covers AugmentedNet (AugN), Micchi2020 (Mi20), ChordGNN, and
ChordGNN+Post on P.DCML. Run commands from the code checkout's root. Train
each model with seeds **42, 43, and 44**, evaluate the best validation-selected
checkpoint, and aggregate the three test reports. Baseline checkpoints and
caches are generated locally; they are not part of the RNGen weight release.

## 1. Prepare score data and shared paths

Download the processed scores using the root
[download instructions](../../README.md#weights-and-data). Complete steps 1
and 2 of the [data-preparation guide](../data_prep/README.md): they create
`outputs/data_preparation/augmented_scores/` with 4,260 training variants
and unchanged validation/test copies. The label-aware transposition also
updates absolute chord tones. Baselines read these note-level CSVs directly;
do not run the RNGen phrase encoder or JSON builders for baseline training.

Create the matching Linux GPU environment using the
[environment instructions](README.md#environments). In a new Bash shell,
activate `rngen-augnet-mi20` or `rngen-chordgnn` before running the shared
setup below, so its `python` is available and the preprocessing shell's
`PYTHONPATH` settings are not carried over. Set these paths, changing the
augmentation location if needed:

```bash
set -e
export SCORE_ROOT="$PWD/datasets/releases/rngen/processed_scores"
export SPLIT_ROOT="$PWD/data_splits/p_dcml"
export AUGMENTED_ROOT="$PWD/outputs/data_preparation/augmented_scores"
export AUGMENTATION_SETTINGS="$PWD/outputs/data_preparation/augmentation_settings.json"
export BASELINE_ROOT="$PWD/outputs/baselines"
```

Use a fresh output location for each complete reproduction. Keep these shell
variables when switching baseline environments; repeat the setup in a new
terminal if necessary. Original manifests select 355/44/47 train/valid/test
scores. The augmented training manifest selects 355 × 12 files. Validation
and test remain untransposed. Cache builders consume the explicit manifests
and do not create new splits.

Create the augmentation record required by the AugN/Mi20 audits, and a
Beethoven test manifest for the optional eight-score comparison:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

settings = {
    "aug_by_key": True,
    "aug_by_key_n_keys": 12,
    "aug_within_range": False,
    "aug_rhythms": False,
}
path = Path(os.environ["AUGMENTATION_SETTINGS"])
path.parent.mkdir(parents=True, exist_ok=True)
if path.exists():
    assert json.loads(path.read_text()) == settings
else:
    path.write_text(json.dumps(settings, indent=2) + "\n")
test_paths = (Path(os.environ["SPLIT_ROOT"]) / "test_paths.txt").read_text().splitlines()
beethoven = [p for p in test_paths if p.startswith("BeethovenPianoSonatas/")]
assert len(test_paths) == 47 and len(beethoven) == 8
out = Path(os.environ["BASELINE_ROOT"]) / "manifests"
out.mkdir(parents=True, exist_ok=True)
(out / "beethoven_test_paths.txt").write_text("\n".join(beethoven) + "\n")
PY
```

This record describes the upstream training transposition. It is different
from the RNGen sequence settings, where `aug_by_key` is false to prevent
augmenting scores a second time.

Keep the baseline environments separate from the Python 3.11 RNGen and
data-preparation environments. Neither baseline family needs the preprocessing
submodules at runtime. Confirm GPU availability before full training.

## 2. AugN and Mi20: audits and shared cache

```bash
conda activate rngen-augnet-mi20
export AUGNET_WORK="$BASELINE_ROOT/augnet_mi20"

python -m rngen.baselines.augnet_mi20.data.pdcml_augmentednet_converter audit \
  --dataset-root "$SCORE_ROOT" --split-dir "$SPLIT_ROOT" \
  --output-json "$AUGNET_WORK/audits/ontology.json"

python -m rngen.baselines.augnet_mi20.data.pdcml_augmentednet_converter audit-augmented-training \
  --dataset-root "$SCORE_ROOT" --split-dir "$SPLIT_ROOT" \
  --augmented-root "$AUGMENTED_ROOT" \
  --augmented-split-dir "$SPLIT_ROOT/augmented_splits" \
  --augmentation-settings "$AUGMENTATION_SETTINGS" \
  --output-json "$AUGNET_WORK/audits/augmented_training.json"

python -m rngen.baselines.augnet_mi20.data.materialize_augmentednet_full_cache \
  --base-root "$SCORE_ROOT" --base-split-dir "$SPLIT_ROOT" \
  --augmented-root "$AUGMENTED_ROOT" \
  --augmented-split-dir "$SPLIT_ROOT/augmented_splits" \
  --augmentation-settings "$AUGMENTATION_SETTINGS" \
  --ontology-audit "$AUGNET_WORK/audits/ontology.json" \
  --reference-conversion-audit "$AUGNET_WORK/audits/augmented_training.json" \
  --output-dir "$AUGNET_WORK/cache"
```

Run these once; both architectures and all seeds share this cache. The first
audit records the fixed ontology and source conversion diagnostics. The
augmented-training audit checks transposition and supplies reference totals
for materialization. The cache contains per-score NPZs, `train_npz_paths.txt`,
`validation_npz_paths.txt`, and `materialization_report.json`. It stores the
training and validation inputs, including aligned Mi20 downsampled labels
and masks. No test NPZs are required.

An optional independent cache check is:

```bash
python -m rngen.baselines.augnet_mi20.checks.audit_augmentednet_full_cache \
  --cache-root "$AUGNET_WORK/cache" \
  --output-json "$AUGNET_WORK/audits/cache.json"
```

### Train, evaluate, and summarize

```bash
for ARCH in AugmentedNet Micchi2020; do
  for SEED in 42 43 44; do
    RUN="$AUGNET_WORK/runs/$ARCH/seed_$SEED"
    python -m rngen.baselines.train augnet_mi20 \
      architecture="$ARCH" seed="$SEED" \
      train_manifest="$AUGNET_WORK/cache/train_npz_paths.txt" \
      validation_manifest="$AUGNET_WORK/cache/validation_npz_paths.txt" \
      cache_report="$AUGNET_WORK/cache/materialization_report.json" \
      ontology_audit="$AUGNET_WORK/audits/ontology.json" \
      output_dir="$RUN/training"

    for SUBSET in full_test beethoven_test; do
      MANIFEST="$SPLIT_ROOT/test_paths.txt"
      if [ "$SUBSET" = beethoven_test ]; then
        MANIFEST="$BASELINE_ROOT/manifests/beethoven_test_paths.txt"
      fi
      python -m rngen.baselines.augnet_mi20.evaluation.evaluate_augmentednet_baseline \
        --architecture "$ARCH" \
        --checkpoint "$RUN/training/checkpoints/best.weights.h5" \
        --ontology-audit "$AUGNET_WORK/audits/ontology.json" \
        --dataset-root "$SCORE_ROOT" --manifest "$MANIFEST" \
        --output-dir "$RUN/evaluation/$SUBSET"
    done
  done

  python -m rngen.baselines.augnet_mi20.evaluation.audit_augmentednet_three_seed_results \
    --architecture "$ARCH" --run-root "$AUGNET_WORK/runs/$ARCH" \
    --output "$AUGNET_WORK/aggregates/${ARCH}_three_seeds.json"
done
```

The supplied recipe runs 100 epochs, batch size 16, with learning rate 0.001.
The best weights minimize the sum of the six monitored validation-head losses.
Both architectures use the 11-head implementation without synthetic-score
augmentation. Test evaluation converts the original CSVs on demand and writes
`onset_metrics.json`, per-piece/classwise reports, prediction records, and
failure examples.

The three-seed audit expects the exact directory layout above, including
**both** `full_test` and `beethoven_test` for every seed. It verifies report
arithmetic, identical within-subset denominators/manifests, and agreement of
the Beethoven records with their entries in the full report. It checks the
checkpoint files exist; it does not require private training history.

## 3. ChordGNN: graph cache, training, and test

Activate the PyTorch baseline environment and repeat the shared shell path
setup if this is a new terminal:

```bash
conda activate rngen-chordgnn
export CHORD_WORK="$BASELINE_ROOT/chordgnn"

python -m rngen.baselines.chordgnn.data.materialize_chordgnn_cache \
  --base-root "$SCORE_ROOT" --split-dir "$SPLIT_ROOT" \
  --augmented-root "$AUGMENTED_ROOT" \
  --augmented-split-dir "$SPLIT_ROOT/augmented_splits" \
  --output-dir "$CHORD_WORK/cache"

for SEED in 42 43 44; do
  RUN="$CHORD_WORK/runs/seed_$SEED"
  python -m rngen.baselines.train chordgnn seed="$SEED" \
    train_manifest="$CHORD_WORK/cache/train_npz_paths.txt" \
    validation_manifest="$CHORD_WORK/cache/validation_npz_paths.txt" \
    cache_report="$CHORD_WORK/cache/materialization_report.json" \
    ontology="$CHORD_WORK/cache/ontology.json" \
    output_dir="$RUN/training"

  for SUBSET in pdcml_full pdcml_bps; do
    MANIFEST="$SPLIT_ROOT/test_paths.txt"
    if [ "$SUBSET" = pdcml_bps ]; then
      MANIFEST="$BASELINE_ROOT/manifests/beethoven_test_paths.txt"
    fi
    python -m rngen.baselines.chordgnn.evaluation.evaluate_chordgnn_baseline \
      --checkpoint "$RUN/training/checkpoints/best.pt" \
      --run-config "$RUN/training/run_config.json" \
      --ontology "$CHORD_WORK/cache/ontology.json" \
      --dataset-root "$SCORE_ROOT" --split-dir "$SPLIT_ROOT" \
      --manifest "$MANIFEST" --split test --seed "$SEED" \
      --output-dir "$RUN/test/$SUBSET"
  done
done

for SUBSET in pdcml_full pdcml_bps; do
  python -m rngen.baselines.chordgnn.evaluation.aggregate_chordgnn_three_seeds \
    --seed-42-evaluation "$CHORD_WORK/runs/seed_42/test/$SUBSET" \
    --seed-43-evaluation "$CHORD_WORK/runs/seed_43/test/$SUBSET" \
    --seed-44-evaluation "$CHORD_WORK/runs/seed_44/test/$SUBSET" \
    --output "$CHORD_WORK/aggregates/${SUBSET}_three_seed.json"
done
```

The graph cache contains 4,260 train graphs and 44 validation graphs, the
ontology, manifests, and a materialization report. It is independent of the
AugN/Mi20 audits and cache; +Post reuses this same graph cache.

The formal configuration uses FP32, AdamW, batch 100, up to 100 epochs, and
early stopping with patience 20. Checkpoint selection minimizes the unweighted
sum of six core masked validation cross-entropies. Keep the saved
`run_config.json`, `ontology.json`, `run_summary.json`, and checkpoints with
the training run. Evaluation checks the checkpoint/config/ontology pair and
seed; test graphs are built from original CSVs on demand.

## 4. ChordGNN+Post: paired-base training and test

Stay in `rngen-chordgnn`. Run this after all three base ChordGNN runs complete:

```bash
export POST_WORK="$BASELINE_ROOT/chordgnn_post"

for SEED in 42 43 44; do
  BASE="$CHORD_WORK/runs/seed_$SEED/training"
  RUN="$POST_WORK/runs/seed_$SEED"
  python -m rngen.baselines.train chordgnn_post seed="$SEED" base_seed="$SEED" \
    train_manifest="$CHORD_WORK/cache/train_npz_paths.txt" \
    validation_manifest="$CHORD_WORK/cache/validation_npz_paths.txt" \
    cache_report="$CHORD_WORK/cache/materialization_report.json" \
    ontology="$CHORD_WORK/cache/ontology.json" \
    base_checkpoint="$BASE/checkpoints/best.pt" \
    base_run_config="$BASE/run_config.json" \
    output_dir="$RUN/training"

  for SUBSET in pdcml_full pdcml_bps; do
    MANIFEST="$SPLIT_ROOT/test_paths.txt"
    if [ "$SUBSET" = pdcml_bps ]; then
      MANIFEST="$BASELINE_ROOT/manifests/beethoven_test_paths.txt"
    fi
    python -m rngen.baselines.chordgnn_post.evaluation.evaluate_chordgnn_post_test \
      --checkpoint "$RUN/training/checkpoints/best.pt" \
      --run-config "$RUN/training/run_config.json" \
      --ontology "$CHORD_WORK/cache/ontology.json" \
      --dataset-root "$SCORE_ROOT" --split-dir "$SPLIT_ROOT" \
      --manifest "$MANIFEST" --seed "$SEED" --subset "$SUBSET" \
      --output-dir "$RUN/test/$SUBSET"
  done
done

for SUBSET in pdcml_full pdcml_bps; do
  python -m rngen.baselines.chordgnn_post.evaluation.aggregate_chordgnn_post_three_seeds \
    --subset "$SUBSET" \
    --seed-42-evaluation "$POST_WORK/runs/seed_42/test/$SUBSET" \
    --seed-43-evaluation "$POST_WORK/runs/seed_43/test/$SUBSET" \
    --seed-44-evaluation "$POST_WORK/runs/seed_44/test/$SUBSET" \
    --output "$POST_WORK/aggregates/${SUBSET}_three_seed.json"
done
```

+Post freezes its paired base and trains the Post module on 379 concatenated
head probabilities. Pair base 42 with Post 42, and likewise for 43 and 44.
The formal recipe uses batch 100, up to 31 epochs, and early-stopping patience
10. The best checkpoint uses the same six-core-head validation-loss criterion.
Training checks that the supplied base is the best checkpoint of a completed
formal run. Post evaluation requires the paired base weights and configuration
at the paths saved in its run configuration, so keep those local run directories
in place.

For an optional exact-onset **validation** report, use
`rngen.baselines.chordgnn_post.evaluation.evaluate_chordgnn_post` with
`--manifest "$SPLIT_ROOT/valid_paths.txt"` and the same checkpoint/config/ontology
arguments. This entrypoint records validation results. The `_test` entrypoint
above records test results and takes an explicit subset name. The base
ChordGNN evaluator supports validation with `--split validation` and the
validation manifest. Never select a checkpoint using test results.

## Metrics and reference results

All headline metrics use annotated reference note onsets from the original
processed CSVs. Native frame/graph predictions are aligned to those onsets;
fixed-grid CSR and native graph/frame diagnostics are separate from the
headline exact-onset results. Reports store fractions in `[0, 1]`.

The full P.DCML test split has **47 scores and 31,920 annotated onsets**; the
Beethoven subset has **8 scores and 9,268 onsets**. Every seed must use the
same manifest and denominator. Aggregates report the arithmetic mean of the
three per-seed accuracies and the sample standard deviation (`ddof=1`).

The following three-seed mean percentages match Table 2 of the
[RNGen paper](https://ismir2026program.ismir.net/poster_405.html) and the saved
evaluation reports:

| Model | RNA | K | Q | D | I |
| --- | ---: | ---: | ---: | ---: | ---: |
| AugN (11 heads, no synthetic scores) | 39.03 | 78.14 | 77.11 | 63.48 | 71.70 |
| Mi20 | 39.42 | 79.36 | 77.72 | 64.92 | 70.41 |
| ChordGNN | 45.04 | 81.94 | 77.93 | 67.00 | 74.86 |
| ChordGNN+Post | 48.38 | 79.57 | 80.38 | 69.22 | 76.25 |

Compare `full_test` for AugN/Mi20 with `pdcml_full` for ChordGNN/+Post.
The Beethoven runs are an additional subset check, not the main table.
Different GPU/software stacks can change retraining results; these reference
numbers do not establish a fixed numerical tolerance. First verify counts,
manifests, masks, pairing, and decoding diagnostics when investigating a gap.

## Git records and saved-run checks

Public cache construction, training, and evaluation support a modified checkout
or a source archive without Git history. Git metadata is recorded when
available, but a clean worktree, a private commit, or the cache's Git identity
is not required. The formal configuration still enforces the experiment's
hyperparameters, seeds, cache counts, ontology and paired-base checks. Cache
reports and source protection remain required.

Training configuration fingerprints do not include incidental Git state.
Resuming with unchanged training settings retains the saved configuration
identifier, including identifiers in older runs. Changes to training settings,
ontology, or the paired base still cause resume validation to fail.
The checkpoint availability audits are optional; their output is not an unlock
token for training or evaluation. Generated reports and caches belong outside
source/data directories and are not committed with the code.

## Upstream papers

Please cite the RNGen paper for this controlled comparison and the applicable
upstream work:

- Néstor Nápoles López, Mark Gotham, and Ichiro Fujinaga (2021).
  [AugmentedNet: A Roman Numeral Analysis Network with Synthetic Training Examples and Additional Tonal Tasks](https://archives.ismir.net/ismir2021/paper/000050.pdf).
  ISMIR, pp. 404–411. The shared implementation also supplies the Mi20 model.
- Gianluca Micchi, Mark Gotham, and Mathieu Giraud (2020).
  [Not All Roads Lead to Rome: Pitch Representation and Model Architecture for Automatic Harmonic Analysis](https://doi.org/10.5334/tismir.45).
  TISMIR 3(1), pp. 42–54.
- Emmanouil Karystinaios and Gerhard Widmer (2023).
  [Roman Numeral Analysis with Graph Neural Networks: Onset-wise Predictions from Note-wise Features](https://archives.ismir.net/ismir2023/paper/000070.pdf).
  ISMIR, pp. 597–604. This is the source for ChordGNN and its learned +Post stage.

Fixed source revisions and complete upstream code license notices are listed
in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
