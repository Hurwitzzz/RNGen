# Public experiment configurations

Training and evaluation configurations for RNGen and its baselines.
Replace `/path/to/...` placeholders with your local dataset, checkpoint, and
output paths. Run commands from the repository root in the appropriate
installed environment.

## RNGen

The released dataset uses `key/`, `mixed/`, and `rna/`, each containing
`train.json`, `valid.json`, and `test.json`. Set `jsonl_dir` to the matching
stage directory. The training and evaluation entrypoints also accept the
historical split filenames when the new filenames are absent. Keep spelled
and unspelled input variants in separate directories.

| Configuration | Purpose | Initialization |
| --- | --- | --- |
| `rngen/sft_common.yaml` | Shared model and SFT parameters; inherit this file | Not a standalone run |
| `rngen/key.yaml` | Key SFT | Supplied fine-tuned RNBERT encoder + selected LLM |
| `rngen/curriculum_stage2.yaml` | Six-task RNA curriculum | Matching Key-stage RNGen checkpoint |
| `rngen/rna.yaml` | Full RNA SFT | Matching previous-stage RNGen checkpoint |
| `rngen/evaluation.yaml` | Saved-checkpoint evaluation | Selected trained RNGen checkpoint |

Stage files inherit `sft_common.yaml` through `_base_`.
Replace the dataset, output, and checkpoint placeholders in the selected file,
or supply training overrides as `key=value`. Give each model/stage its own
output directory. Select `llm_model_name` from:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct` (default)
- `Qwen/Qwen2.5-14B-Instruct`

A continued RNGen checkpoint must
match the selected model and encoder variant; changing the model-name string
cannot convert a checkpoint to a different architecture.

| Setting | Key | Stage 2 | RNA |
| --- | --- | --- | --- |
| Epochs | 2 | 1 | 2 |
| Learning rate | 5e-5 | 5e-5 | 5e-5 |
| Scheduler / warm-up ratio | cosine / 0.1 | cosine / 0.1 | cosine / 0.1 |
| Effective training batch | 4 | 4 | 4 |
| Evaluation/save interval | 1800 steps | 0.1 of total training steps | 1800 steps |
| Best-checkpoint metric | `eval_loss` | `eval_stage2_selection_loss` | `eval_loss` |
| Validation batch per device | 1 | 1 | 1 |


All public RNGen stages default to one training process, batch 4 per device,
and gradient accumulation 1. Preserve an effective batch of 4 when changing
hardware:

`effective batch = per-device batch × gradient accumulation × data-parallel processes`

Example training invocation (after editing paths):

```bash
python -m rngen.sft_ft --config configs/rngen/key.yaml \
  llm_model_name=Qwen/Qwen2.5-7B-Instruct
python -m rngen.sft_ft --config configs/rngen/curriculum_stage2.yaml
python -m rngen.sft_ft --config configs/rngen/rna.yaml
```

On one GPU with a smaller microbatch, add
`per_device_train_batch_size=1 gradient_accumulation_steps=4`.
For two GPUs, for example:

```bash
accelerate launch --multi_gpu --num_processes 2 --num_machines 1 \
  --mixed_precision bf16 --dynamo_backend no --module rngen.sft_ft \
  --config configs/rngen/curriculum_stage2.yaml \
  per_device_train_batch_size=2 gradient_accumulation_steps=1
```

The recipes use BF16, LoRA (rank 16, alpha 32), and W&B reporting.
Disable W&B with both `report_to=none wandb_project=null` so the custom example
callback does not initialize a W&B project. The default seed is 42.

The root [README](../README.md#train-a-new-model) provides a complete 7B/14B
Key → Mixed → RNA command sequence using the downloaded dataset and weights,
including selection of each stage's best checkpoint.

### Ablations and checkpoint settings

- The main Key configuration uses the domain-fine-tuned, spelled RNBERT encoder.
  For the unspelled pretrained-MusicBERT control, override
  `rnbert_checkpoint_path=/path/to/pretrained_musicbert.pt`,
  `use_spelled_musicbert=false`, and `load_from_fted_rnbert=false` in `key.yaml`.
  Continue from matching control checkpoints in Stage 2 and RNA, retaining
  `use_spelled_musicbert=false` and `load_from_fted_rnbert=false` in each stage.
  To reproduce this control's Stage 2 settings, use effective training batch 8
  and `per_device_eval_batch_size=4`. For its RNA stage, set
  `early_stopping_patience=3`.
- `save_total_limit` controls checkpoint retention and defaults to 2.
- `load_from_fted_rnbert` labels the run name only. `rnbert_checkpoint_path`
  selects the initial encoder; `finetuned_rngen_checkpoint` loads a complete
  prior RNGen checkpoint when supplied.

### Evaluation

Evaluation loads model architecture from the checkpoint. It has a separate,
small configuration containing only fields consumed by `rngen.sft_eval`.
The default is RNA with deterministic beam search (4 beams, 2048 new tokens).

```bash
python -m rngen.sft_eval --config configs/rngen/evaluation.yaml
# Key evaluation: choose a matching Key checkpoint and test annotations.
python -m rngen.sft_eval --config configs/rngen/evaluation.yaml \
  --task key_sft --model-path /path/to/key_checkpoint \
  --json-path /path/to/rngen/key/test.json
```

The evaluation CLI uses `--flag value`, unlike the SFT CLI's `key=value`
overrides. For an unspelled MusicBERT control, set `use_spelled_musicbert: false`
in the evaluation YAML and select the corresponding unspelled JSON. For a
subset such as Beethoven, supply that subset's annotation JSON explicitly.

## Baseline training

The [complete baseline reproduction guide](../rngen/baselines/REPRODUCTION.md)
provides ordered cache, training, evaluation, and three-seed aggregation
commands, with the expected P.DCML test results.

Run baseline training with `python -m rngen.baselines.train <baseline>`.
The launcher reads its YAML configuration and passes the parameters to the
training module.
Baseline implementations are grouped under
[`rngen/baselines/`](../rngen/baselines/README.md).
Create and activate the matching Conda environment using the
[baseline environment instructions](../rngen/baselines/README.md#environments).
AugN/Mi20 share one environment, and ChordGNN/+Post share another. The launcher
uses the active environment's Python interpreter; PyYAML is included in both
recipes.

| Configuration | Training launcher | Main settings |
| --- | --- | --- |
| `augnet_mi20/train.yaml` | `python -m rngen.baselines.train augnet_mi20` | 100 epochs, batch 16, LR 0.001 |
| `chordgnn/train.yaml` | `python -m rngen.baselines.train chordgnn` | 100 epochs, batch 100, LR 0.0015, patience 20 |
| `chordgnn_post/train.yaml` | `python -m rngen.baselines.train chordgnn_post` | 31 epochs, batch 100, LR 0.001, patience 10 |

```bash
# Preview commands without loading models or starting training.
python -m rngen.baselines.train augnet_mi20 --dry-run
python -m rngen.baselines.train chordgnn --dry-run
python -m rngen.baselines.train chordgnn_post --dry-run

# Train after filling paths; the two TF baselines share one configuration.
python -m rngen.baselines.train augnet_mi20 architecture=Micchi2020 \
  output_dir=/path/to/outputs/mi20/seed_42
python -m rngen.baselines.train chordgnn seed=43 \
  output_dir=/path/to/outputs/chordgnn/seed_43
python -m rngen.baselines.train chordgnn_post seed=43 base_seed=43 \
  base_checkpoint=/path/to/chordgnn/seed_43/training/checkpoints/best.pt \
  base_run_config=/path/to/chordgnn/seed_43/training/run_config.json \
  output_dir=/path/to/outputs/chordgnn_post/seed_43
```

Use `--config /path/to/custom.yaml` for a local copy. Repeat with seeds 42, 43,
and 44 and distinct output directories for the three-seed runs. Post requires
the corresponding base checkpoint and its run configuration. Device selection
uses the underlying framework and `CUDA_VISIBLE_DEVICES`.

Supply prepared caches with manifests pointing to your local NPZ files. Paths
inside manifests and relative CLI paths are interpreted from the repository
root. The AugN/Mi20 launcher expands its manifests
into `--train-npz` / `--validation-npz`; its ontology JSON must contain the
`vocabularies` mapping expected by the converter/trainer. ChordGNN additionally
requires its materialization report and ontology. Prepare caches and obtain
required weights before training. Run evaluation separately using the
corresponding evaluation CLI.

All three baseline configurations accept `cache_report` and optional
`source_root` values. The cache report supplies the source CSV directories that
training protects from output writes. Set `source_root` to a YAML list to protect
additional directories, or leave it `null` to use the report and default published
score directory. These settings do not change the training hyperparameters.
