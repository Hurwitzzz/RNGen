# Baseline implementations

Each baseline directory contains its model, trainer, data conversion, cache
preparation, evaluation, and supporting tools.

| Directory | Contents | Runtime |
| --- | --- | --- |
| [augnet_mi20/](augnet_mi20/) | Shared AugmentedNet/Micchi2020 implementation | TensorFlow |
| [chordgnn/](chordgnn/) | Base ChordGNN implementation | PyTorch |
| [chordgnn_post/](chordgnn_post/) | Learned +Post module with a frozen ChordGNN base | PyTorch |
| [shared/](shared/) | Exact-onset scoring, denominator checks, and environment records | Per-tool dependencies |

Every model directory provides `model.py` and `trainer.py`. The command-line
entrypoints retain descriptive names such as `train_chordgnn_baseline.py` and
`evaluate_chordgnn_post_test.py`. Importing the baseline packages does not
initialize a framework; each implementation loads its own dependencies.

Files within each model family use the following layout:

| Location | Purpose |
| --- | --- |
| Root | `model.py`, `trainer.py`, and the training entrypoint |
| `data/` | Score conversion, cache materialization, and cache-reading helpers |
| `evaluation/` | Evaluation entrypoints, classwise scoring, diagnostics, and multi-seed result aggregation |
| `checks/` | Cache audits, checkpoint availability, and official-implementation parity checks |

ChordGNN+Post reuses the base ChordGNN data pipeline and has no `data/`
directory. The small cross-model `shared/` package remains flat.

For example:

```text
chordgnn/
├── environment.yml
├── model.py
├── trainer.py
├── train_chordgnn_baseline.py
├── data/
│   ├── pdcml_chordgnn_converter.py
│   └── materialize_chordgnn_cache.py
├── evaluation/
│   ├── evaluate_chordgnn_baseline.py
│   ├── aggregate_chordgnn_three_seeds.py
│   ├── chordgnn_classwise.py
│   └── chordgnn_diagnostics.py
└── checks/
    ├── audit_chordgnn_runs.py
    └── chordgnn_official_parity.py
```

## Environments

The environment recipes target Linux x86_64 with NVIDIA GPUs. Use Conda to
create a separate environment for each framework family. The host must have
a compatible NVIDIA driver; CUDA runtime libraries are included in the recipes.

| Baselines | Environment file | Environment name | Python | Framework and CUDA |
| --- | --- | --- | --- | --- |
| AugmentedNet and Micchi2020 | [augnet_mi20/environment.yml](augnet_mi20/environment.yml) | `rngen-augnet-mi20` | 3.8.20 | TensorFlow 2.4.1, CUDA 11.0.3, cuDNN 8.0.5.39 |
| ChordGNN and ChordGNN+Post | [chordgnn/environment.yml](chordgnn/environment.yml) | `rngen-chordgnn` | 3.10.19 | PyTorch 2.4.1, CUDA 12.1, torch-scatter 2.1.2 |

Run from the repository root. For AugN or Mi20:

```bash
conda env create -f rngen/baselines/augnet_mi20/environment.yml
conda activate rngen-augnet-mi20
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
python -m rngen.baselines.train augnet_mi20 --dry-run
```

For ChordGNN or +Post:

```bash
conda env create -f rngen/baselines/chordgnn/environment.yml
conda activate rngen-chordgnn
python -c "import torch, torch_scatter; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch_scatter.__version__)"
python -m rngen.baselines.train chordgnn --dry-run
python -m rngen.baselines.train chordgnn_post --dry-run
```

Confirm that TensorFlow lists a GPU or PyTorch reports CUDA availability before
GPU training. The ChordGNN recipe selects the `torch-scatter` wheel for
PyTorch 2.4 and CUDA 12.1.

Baseline modules run directly from the checkout. Keep the RNGen core installer
(`install.py`) and `rngen_requirements.txt` in a separate Python 3.11 environment;
they select different framework versions. The baseline environment files include
PyYAML and the dependencies needed by their training and evaluation tools.

## Training

For the full CSV → cache → three-seed training → test → aggregation sequence,
follow [Reproduce the baselines](REPRODUCTION.md). It includes all required
audit inputs, paired +Post base runs, exact test denominators, verified paper
results, and upstream paper citations.

Activate the appropriate baseline environment, then run `python -m
rngen.baselines.train` with the baseline name. The launcher requires PyYAML,
reads `configs/<baseline>/train.yaml`, and passes its parameters to the
corresponding training module using the same Python interpreter. It does not
import a model or framework before starting the training process.

```bash
python -m rngen.baselines.train augnet_mi20 --dry-run
python -m rngen.baselines.train chordgnn --dry-run
python -m rngen.baselines.train chordgnn_post --dry-run
```

Use `--dry-run` to preview the command, `key=value` arguments such as `seed=43`
or `architecture=Micchi2020` to override configuration values, and
`--config /path/to/custom.yaml` to select a local configuration. Replace path
placeholders before training. See [public configurations](../../configs/README.md#baseline-training)
for cache inputs, environment requirements, overrides, and seeds.

Individual training and evaluation entrypoints provide their own CLI help:

```bash
python -m rngen.baselines.augnet_mi20.train_augmentednet_baseline --help
python -m rngen.baselines.augnet_mi20.evaluation.evaluate_augmentednet_baseline --help
python -m rngen.baselines.chordgnn.train_chordgnn_baseline --help
python -m rngen.baselines.chordgnn_post.train_chordgnn_post --help
python -m rngen.baselines.chordgnn_post.evaluation.evaluate_chordgnn_post --help
python -m rngen.baselines.chordgnn_post.evaluation.evaluate_chordgnn_post_test --help
```

AugN and Mi20 use the same implementation and select the architecture with
`--architecture AugmentedNet` or `--architecture Micchi2020`.

## Data preparation and evaluation

`augnet_mi20/data/augmentednet_cache.py` provides NPZ materialization helpers for
`augnet_mi20/data/materialize_augmentednet_full_cache.py`. The cache contains the
training and validation splits. Test evaluation reads source CSVs from
`--dataset-root` using the supplied test manifest and converts them on demand;
a test NPZ cache is not required.

Data-root defaults follow the Hugging Face download layout in the root README:
`--dataset-root` and `--base-root` select
`datasets/releases/rngen/processed_scores/` through `rngen/paths.py`. Supply
either flag explicitly when your processed scores are stored elsewhere.

Training augmentation is prepared locally. Cache builders and the augmented-data
audit require `--augmented-root`; the AugN/Mi20 tools also require the
`--augmentation-settings` file describing that augmentation. The AugN/Mi20 cache
builder takes explicit `--ontology-audit` and `--reference-conversion-audit`
JSONs from the converter's `audit` and `audit-augmented-training` commands.
AugN/Mi20 training and prediction evaluation require `--ontology-audit`;
`--ground-truth-audit-only` does not require an ontology file.

The training configurations supply the local cache's `materialization_report.json`
as `cache_report`. Training reads the original and augmented CSV roots from that
report and protects them against output writes, together with the default
published score directory. Supply additional protected roots with repeatable or
space-separated `--source-root` arguments, or a `source_root` list in the training
YAML. Direct AugN/Mi20 training from custom NPZs without a cache report should
specify its CSV roots with `--source-root`.

Post reuses base ChordGNN helpers. ChordGNN and exact-onset scoring reuse CSV,
ontology, and RN decoder functions from the AugN/Mi20 implementation. Shared
scoring and provenance helpers are in `shared/`.

Post has separate validation and test entrypoints:
`chordgnn_post/evaluation/evaluate_chordgnn_post.py` and
`chordgnn_post/evaluation/evaluate_chordgnn_post_test.py`. Test evaluation takes an explicit
`--subset` report name, such as `pdcml_full` or `pdcml_bps`. Evaluators check
configuration, ontology, and paired-base compatibility. Aggregation requires
matching test manifests and exact-onset denominators.

Git state is recorded when available; public cache preparation, training, and
evaluation also accept modified checkouts and source archives. Formal recipes
retain their data and hyperparameter checks. See the
[saved-run requirements](REPRODUCTION.md#git-records-and-saved-run-checks).

## Optional checks

### Checkpoint availability and saved results

`chordgnn/checks/audit_chordgnn_runs.py` and
`chordgnn_post/checks/audit_chordgnn_post_runs.py` check that supplied training
directories contain `checkpoints/best.pt`. They do not depend on Git history,
training logs, GPU models, or matching training settings across seeds. They
are optional checks, not prerequisites for evaluation. The Post check covers
Post weights; evaluation also loads the paired base weights and configuration.

`augnet_mi20/evaluation/audit_augmentednet_three_seed_results.py` checks checkpoint-file
availability and aggregates saved evaluation reports. It does not prove that
training runs differ only by seed.

### Upstream implementation checks

Upstream sources, copyright notices, and MIT license texts for AugmentedNet
and ChordGNN are provided in the
[third-party code notices](../../THIRD_PARTY_NOTICES.md).

These tools compare the local model implementation with a separately downloaded
upstream checkout. Use AugmentedNet commit
`709dc59470412453fd9046ef4fb53e5663141144` or ChordGNN commit
`bcbdc72d2850e986c5f577727f800dee0f95f0c4`, respectively. Run each check in its
matching baseline environment and supply your local source and input paths.

```bash
python -m rngen.baselines.augnet_mi20.checks.augmentednet_graph_parity \
  --official-root /path/to/AugmentedNet \
  --output-json /path/to/outputs/augnet_parity.json

python -m rngen.baselines.chordgnn.checks.chordgnn_official_parity \
  --source-root /path/to/ChordGNN \
  --fixture-dir /path/to/fixtures \
  --output /path/to/outputs/chordgnn_parity.json

python -m rngen.baselines.chordgnn_post.checks.chordgnn_post_official_parity \
  --source-root /path/to/ChordGNN \
  --ontology /path/to/chordgnn_cache/ontology.json \
  --output /path/to/outputs/chordgnn_post_parity.json
```

The AugN/Mi20 check runs on CPU by default and records
`augnet_mi20/environment.yml` and its SHA-256 in the report. Use
`--environment /path/to/environment.yml` to record a different environment
definition, or `--allow-gpu` to keep GPUs visible.

The ChordGNN fixture directory must contain `wir-bach-chorales-8.tsv`,
`bps-09-op014-no1-1.tsv`, and `abc-op18-no6-4.tsv`. The Post check uses the
official 379-class ontology in the cache's `vocabularies` mapping. Upstream
checkouts and fixtures are supplied separately by the user.

Generated reports, run logs, caches, and checkpoints should be stored outside
the source directories and excluded from version control.
