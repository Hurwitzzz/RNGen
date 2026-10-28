# RNGen

Official implementation of:

**[RNGen: A Large Language Model Approach to Roman Numeral Analysis Toward Explainable Harmonic Analysis](https://ismir2026program.ismir.net/poster_405.html)**

Hewei Gao, Tim Beyer, Xiaoxuan Wang, and Martin Rohrmeier

Accepted at **[ISMIR 2026](https://ismir2026.ismir.net/) (main conference)**,
the 27th International Society for Music Information Retrieval Conference.

**Conference dates:** November 8–12, 2026 · Abu Dhabi, UAE.

[Paper](https://ismir2026program.ismir.net/poster_405.html) ·
[Checkpoints](https://huggingface.co/Hurwitz/RNGen-checkpoints) ·
[Dataset](https://huggingface.co/datasets/Hurwitz/RNGen-data) ·
[Citation](#citation)

RNGen combines a domain-fine-tuned RNBERT encoder with a language model for Roman numeral analysis. The training pipeline has three stages: **Key → Mixed → RNA**.

## Installation

Use Python 3.11. From a checkout of this repository, the installer creates an isolated `.venv`, initializes the required `musicbert_hf` submodule, installs the core dependencies, and checks imports, the event vocabulary, and prompt templates.

```bash
git clone https://github.com/Hurwitzzz/RNGen.git
cd RNGen
python3.11 install.py
source .venv/bin/activate
```

On Linux, the default PyTorch 2.6.0 build uses CUDA 12.4. For a different supported build, select it explicitly:

```bash
python3.11 install.py --torch-build cu118
python3.11 install.py --torch-build cu126
# CPU environment for local development and checks:
python3.11 install.py --torch-build cpu
```

Choose one build. On macOS, the installer uses the native PyTorch build. Full 7B/14B training and evaluation require a suitable GPU platform. CUDA build selection and driver requirements follow the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v260).

The installer accepts `--venv /path/to/environment` to use a different environment directory. On Windows, activate it with `.venv\Scripts\Activate.ps1` in PowerShell.

Core package versions are maintained in `pyproject.toml`. `rngen_requirements.txt` installs the project and its local MusicBERT dependency without maintaining a second version list. The installer uses the project's MusicBERT fork at the submodule commit recorded in this repository. This environment covers training and evaluation with the released JSON files. CSV preprocessing, baseline training, and optional plotting have separate dependencies; only `musicbert_hf` is needed as a submodule for the core pipeline.

For baseline training and evaluation, use the separate
[baseline Conda environments](rngen/baselines/README.md#environments).

For an existing, dedicated Python 3.11 environment, initialize MusicBERT and install from the repository root:

```bash
git submodule update --init musicbert_hf
python -m pip install -r rngen_requirements.txt
```

## Weights and data

Weights and datasets are hosted on Hugging Face separately from the Git repository:

| Repository | Contents | Pinned revision |
| --- | --- | --- |
| [Hurwitz/RNGen-checkpoints](https://huggingface.co/Hurwitz/RNGen-checkpoints) | Base encoder weights and all six RNGen checkpoints | `e8c2ba060ea6c0f1f5ebe30753ff94941167d271` |
| [Hurwitz/RNGen-data](https://huggingface.co/datasets/Hurwitz/RNGen-data) | Key, Mixed, and RNA annotations; processed scores and metadata; preprocessing settings | `6cdfc9d59b50b2c491874057d77aea3f8522a089` |

Both repositories are currently private. Downloading requires a Hugging Face
account with access. After activating the RNGen environment, authenticate in
the terminal:

```bash
huggingface-cli login
```

Run these commands from the repository root. The fixed revisions select the
files associated with this release:

```bash
huggingface-cli download Hurwitz/RNGen-checkpoints \
  --revision e8c2ba060ea6c0f1f5ebe30753ff94941167d271 \
  --local-dir ckpts/releases
huggingface-cli download Hurwitz/RNGen-data --repo-type dataset \
  --revision 6cdfc9d59b50b2c491874057d77aea3f8522a089 \
  --local-dir datasets/releases/rngen
```

These commands preserve the repository file layout. Download both the base encoder weights and the RNGen checkpoints without flattening their directories: the RNGen metadata uses relative paths to the accompanying RNBERT checkpoint. The [Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/v0.30.2/guides/download) describes downloads to local directories.

```text
ckpts/releases/
├── pretrained/
│   └── checkpoint_last_musicbert_base.pt
├── rnbert/
│   ├── unconditioned_checkpoint_best.pt
│   └── 1742830064_synced_metrics.csv
└── rngen/
    ├── RNGen-7B-Key/
    ├── RNGen-7B-Mixed/
    ├── RNGen-7B-RNA/
    ├── RNGen-14B-Key/
    ├── RNGen-14B-Mixed/
    └── RNGen-14B-RNA/

datasets/releases/rngen/
├── key/                    # train.json, valid.json, test.json
├── mixed/                  # train.json, valid.json, test.json
├── rna/                    # train.json, valid.json, test.json
├── processed_scores/       # Unaugmented CSVs, JSON metadata, and README
└── preprocessing/          # Sequence and Octuple encoding configurations
```

Each RNGen checkpoint contains the encoder, projector, and Qwen LoRA adapters. The original Qwen base model is downloaded separately through Transformers when first loaded; it is not included in the RNGen checkpoint files.

To prepare new annotations from the processed scores, follow the
[CSV-to-JSON guide](rngen/data_prep/README.md). It covers training-only
transposition, Octuple encoding and phrase segmentation, and the Key, RNA,
and Mixed builders. Reproducing the released checkpoints uses the supplied
JSONs directly.

## Evaluate released checkpoints

Run from the repository root with the RNGen environment active and the files
downloaded into the layout above. Evaluation requires the selected RNGen
checkpoint, the accompanying `rnbert/unconditioned_checkpoint_best.pt`, and
the matching test JSON. It does not require training a model or regenerating
annotations. The Qwen base model is fetched on first use.

| Stage | Released checkpoints | Task | Test annotations |
| --- | --- | --- | --- |
| Key | `RNGen-7B-Key`, `RNGen-14B-Key` | `key_sft` | `key/test.json` (621 examples) |
| Mixed | `RNGen-7B-Mixed`, `RNGen-14B-Mixed` | `rna_curriculum_stage2_sft` | `mixed/test.json` (3,726 examples) |
| RNA | `RNGen-7B-RNA`, `RNGen-14B-RNA` | `rna_sft` | `rna/test.json` (621 examples) |

The following loop evaluates all six checkpoints in sequence. To evaluate one
model size, use only `7B` or `14B` in the loop.

```bash
MODEL_ROOT="$PWD/ckpts/releases"
DATA_ROOT="$PWD/datasets/releases/rngen"
EVAL_ROOT="$PWD/outputs/evaluation"

for SIZE in 7B 14B; do
  python -m rngen.sft_eval --config configs/rngen/evaluation.yaml \
    --model-path "$MODEL_ROOT/rngen/RNGen-${SIZE}-Key" \
    --task key_sft --json-path "$DATA_ROOT/key/test.json" \
    --max-new-tokens 256 \
    --output-dir "$EVAL_ROOT/RNGen-${SIZE}-Key"

  python -m rngen.sft_eval --config configs/rngen/evaluation.yaml \
    --model-path "$MODEL_ROOT/rngen/RNGen-${SIZE}-Mixed" \
    --task rna_curriculum_stage2_sft --json-path "$DATA_ROOT/mixed/test.json" \
    --output-dir "$EVAL_ROOT/RNGen-${SIZE}-Mixed"

  python -m rngen.sft_eval --config configs/rngen/evaluation.yaml \
    --model-path "$MODEL_ROOT/rngen/RNGen-${SIZE}-RNA" \
    --task rna_sft --json-path "$DATA_ROOT/rna/test.json" \
    --output-dir "$EVAL_ROOT/RNGen-${SIZE}-RNA"
done
```

These commands use evaluation batch size 1. Key and RNA use deterministic
decoding with four beams and generation limits of 256 and 2,048 tokens,
respectively. Mixed computes losses without generating text.
Metrics are printed with the `eval_` prefix, even though these commands use the
test split. Key and RNA evaluation also write prediction snapshots containing
metrics to the chosen output directory. Mixed evaluation writes
`stage2_eval_loss_summary.json` with six task losses and
`stage2_selection_loss`; it does not report full-RNA accuracy. To measure the
RNA performance of a Mixed checkpoint, evaluate it with `--task rna_sft` and
`--json-path "$DATA_ROOT/rna/test.json"`, using a separate output directory.

See the [RNA metric reference](doc/reference/rna_component_metrics.md) for
definitions, averaging rules, supported denominators, and saved report fields.

## Train a new model

Training starts from the supplied domain-fine-tuned RNBERT checkpoint and the
original Qwen model. It requires all three splits from `key/`, `mixed/`, and
`rna/`. The six released RNGen checkpoints, pretrained-MusicBERT control
weights, and processed score CSVs are not required for this training route.
Use the released JSONs to retain the training inputs associated with the
published experiments; see [data preparation](rngen/data_prep/README.md) before
regenerating annotations.

Run this chain once with `SIZE=7B`, then repeat from the setup block with
`SIZE=14B` for the other model size. Each chain uses its own output directory.
Keep the same shell open between stages so the selected checkpoint variables
remain available.

```bash
SIZE=7B
LLM="Qwen/Qwen2.5-${SIZE}-Instruct"
MODEL_ROOT="$PWD/ckpts/releases"
DATA_ROOT="$PWD/datasets/releases/rngen"
RUN_ROOT="$PWD/outputs/reproduction/${SIZE}_$(date +%Y%m%d_%H%M%S)"

best_checkpoint() {
  python - "$1" <<'PY'
import json
from pathlib import Path
import sys

states = list(Path(sys.argv[1]).glob("*/checkpoint-*/trainer_state.json"))
if not states:
    raise SystemExit("No saved Trainer state found; check that training completed.")
latest = max(states, key=lambda path: path.stat().st_mtime_ns)
best = json.loads(latest.read_text()).get("best_model_checkpoint")
if not best or not Path(best).is_dir():
    raise SystemExit("The selected best checkpoint is missing; inspect trainer_state.json.")
print(Path(best).resolve())
PY
}
```

The helper reads the best checkpoint selected on validation data from the most
recent saved Trainer state. It avoids assuming a particular checkpoint step
number. Run the next stage only after the preceding training command finishes
successfully.

### 1. Key

```bash
python -m rngen.sft_ft --config configs/rngen/key.yaml \
  llm_model_name="$LLM" \
  rnbert_checkpoint_path="$MODEL_ROOT/rnbert/unconditioned_checkpoint_best.pt" \
  jsonl_dir="$DATA_ROOT/key" output_dir="$RUN_ROOT/key" \
  report_to=none wandb_project=null

KEY_CHECKPOINT="$(best_checkpoint "$RUN_ROOT/key")"
```

This initializes a new RNGen model from domain-fine-tuned RNBERT and Qwen. The
encoder, projector, and LoRA parameters are trained for two epochs. The best
checkpoint is selected by minimum validation `eval_loss`.

### 2. Mixed

```bash
python -m rngen.sft_ft --config configs/rngen/curriculum_stage2.yaml \
  llm_model_name="$LLM" finetuned_rngen_checkpoint="$KEY_CHECKPOINT" \
  jsonl_dir="$DATA_ROOT/mixed" output_dir="$RUN_ROOT/mixed" \
  report_to=none wandb_project=null

MIXED_CHECKPOINT="$(best_checkpoint "$RUN_ROOT/mixed")"
```

Mixed continues from the best Key checkpoint of the same model size for one
epoch. It mixes the six intermediate tasks and selects the best checkpoint by
minimum validation `eval_stage2_selection_loss`.

### 3. RNA

```bash
python -m rngen.sft_ft --config configs/rngen/rna.yaml \
  llm_model_name="$LLM" finetuned_rngen_checkpoint="$MIXED_CHECKPOINT" \
  jsonl_dir="$DATA_ROOT/rna" output_dir="$RUN_ROOT/rna" \
  report_to=none wandb_project=null

RNA_CHECKPOINT="$(best_checkpoint "$RUN_ROOT/rna")"
```

RNA continues from the best Mixed checkpoint for two epochs and selects the
best checkpoint by minimum validation `eval_loss`.

The training entrypoint evaluates the selected model on the matching test split
after each stage and saves `test_results.json` under that stage's generated
run directory. Saved checkpoints and `trainer_state.json` are under its
`checkpoint-*` directories. For a separate evaluation of the newly trained RNA
model:

```bash
python -m rngen.sft_eval --config configs/rngen/evaluation.yaml \
  --model-path "$RNA_CHECKPOINT" \
  --task rna_sft --json-path "$DATA_ROOT/rna/test.json" \
  --output-dir "$RUN_ROOT/rna_evaluation"
```

The examples disable W&B with both `report_to=none` and `wandb_project=null`.
To use W&B, log in with `wandb login` and omit those two overrides. All stages
use effective batch size 4 by default. If a batch of four does not fit on one
GPU, append `per_device_train_batch_size=1 gradient_accumulation_steps=4` to
each training command. See [public configurations](configs/README.md) for
distributed examples and the remaining hyperparameters.

`finetuned_rngen_checkpoint` restores the encoder, projector, and LoRA weights
from the selected previous stage; it starts a new optimizer and scheduler for
the next stage. `rnbert_checkpoint_path` supplies the base encoder when creating
or restoring model structure. `load_from_fted_rnbert` labels the run name only.
Keep the base RNBERT file available when loading saved checkpoints, and keep
the same model size throughout the chain. See the
[configuration notes](rngen/config/README.md) for details.

Baseline training uses `python -m rngen.baselines.train <baseline>`. See
[baseline implementations](rngen/baselines/README.md#training) for launch
instructions and separate runtime requirements.

## Citation

If you use RNGen, its checkpoints, or its data in a research publication,
please cite our paper:

```bibtex
@inproceedings{gao2026rngen,
  author = {Gao, Hewei and Beyer, Tim and Wang, Xiaoxuan and Rohrmeier, Martin},
  title = {{RNGen}: A Large Language Model Approach to Roman Numeral Analysis Toward Explainable Harmonic Analysis},
  booktitle = {Proceedings of the 27th International Society for Music Information Retrieval Conference},
  year = {2026},
  url = {https://ismir2026program.ismir.net/poster_405.html}
}
```

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

## License

Original RNGen code is released under the [MIT License](LICENSE).
Third-party code and submodules retain their respective licenses. See
[third-party code notices](THIRD_PARTY_NOTICES.md) for source credits and
complete license texts. Model checkpoints and datasets are subject to their
own terms; the source-code
license does not replace those terms. The citation request above is separate
from the MIT License conditions.
