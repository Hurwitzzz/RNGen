# Training runtime and metrics

This directory contains the RNGen SFT runtime and scoring helpers. Training and evaluation entrypoints call these modules; they are not
standalone launch scripts.

The [RNA metric reference](../../doc/reference/rna_component_metrics.md)
defines the implemented scores, averaging rules, and report fields.

## RNGen runtime

| File | Purpose and callers |
| --- | --- |
| `sft_trainer.py` | `RNGenSFTTrainer`, used by `sft_ft.py` and `sft_eval.py`; handles loss evaluation, generation, metadata collection, and checkpoint loading/saving. |
| `task_registry.py` | Routes `key_sft`, `rna_sft`, and `rna_curriculum_stage2_sft` to their metric builders. |
| `callbacks.py` | `WandBCallback`, registered by `sft_ft.py`, logs sampled predictions and references when supplied by the caller. Task metric modules save prediction snapshots. |

Baseline models, trainers, and their command-line entrypoints are maintained
in [../baselines/](../baselines/README.md), grouped by model family.

## Task metrics and shared helpers

| File | Purpose |
| --- | --- |
| `metrics_key_generation.py` | Key SFT metric builder, key scoring, and prediction snapshots. |
| `metrics_rna.py` | Full RNA SFT metric builder, note/onset scoring, prediction snapshots, and class-wise reports. |
| `metrics_rna_curriculum_stage2.py` | Groups validation losses by the six intermediate tasks and computes `stage2_selection_loss`; writes `stage2_eval_loss_summary.json`. |
| `metrics_rna_components.py` | Parses and scores RNA components, including functions, chord tones, and tone-factor pairs; reused by RNA scoring and analysis reports. |
| `metrics_common.py` | Prediction/reference decoding, match records, and prediction snapshot persistence. |
| `metrics_event_ranges.py` | Event-token ranges and note/onset counting. |
| `metrics_key_shared.py` | Key-segment parsing and region/note scoring shared with analysis code. |
| `metrics_rna_shared.py` | Shared RNA note/onset matching and aggregation. |

Historical dummy-task metrics are retained in
[`../legacy/metrics_dummy_test.py`](../legacy/metrics_dummy_test.py) and are not
registered as an active SFT task.

## Stage 2 loss semantics

Stage 2 expands each source example into six task-specific question-answer
examples, giving each task equal representation. The examples are mixed and
shuffled during training.

Training uses the causal language model's supervised next-token loss over
unmasked answer tokens. Prompt, inserted music embeddings, and padding are
masked with `-100`.

The best checkpoint is selected by the lowest mean validation loss across
examples. The Stage 2 config uses `eval_stage2_selection_loss`: the metric
builder averages the Trainer-provided losses within each task, then averages
the six task means. With equal validation example counts across all six tasks,
this equals the Trainer's overall `eval_loss`, up to floating-point rounding.
