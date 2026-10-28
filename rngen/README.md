This directory contains the RNGen package code.

Start here:
- `sft_ft.py`: supervised fine-tuning entrypoint driven by YAML config plus CLI
  overrides
- `sft_eval.py`: manual checkpoint evaluation entrypoint driven by CLI flags or
  a post-hoc YAML config
- `cli/arguments.py`: shared entrypoint argument definitions, task validation, and
  generation-kwargs helpers
- `cli/entrypoint_helpers.py`: small helpers shared between the training and manual
  evaluation entrypoints

RNGen uses RNBERT hidden states without its masked-language-model (MLM) head.
RNGen checkpoints omit MLM weights. Older checkpoint weights named `cls.*` are
ignored during loading; the remaining encoder weights are still validated.

Supporting subdirectories:
- `data_prep/`: offline Key, ordinary RNA SFT, and RNA curriculum Stage 2 annotation generators
  See [data preparation](data_prep/README.md) for inputs and commands.
- `training/`: trainer runtime, metric builders, and shared metric helpers
  See `training/README.md` for the module map.
- `config/`: data-preparation settings and prompt templates
  See [config/README.md](config/README.md) for details; public training and
  evaluation recipes live in the root [configs/](../configs/README.md) directory.
- `analysis/`: post-hoc scoring and result analysis tools
- `baselines/`: AugmentedNet/Mi20, ChordGNN, and ChordGNN+Post implementations,
  with model and trainer at each baseline's root and supporting tools in
  `data/`, `evaluation/`, and `checks/`
  See [baseline implementations](baselines/README.md).

Legacy experiments:
- `legacy/` contains the standalone spelled-pitch warm-up and historical
  dummy-task metrics. These are outside the standard RNGen training pipeline.
  See [legacy/README.md](legacy/README.md) for the module map and entrypoint.
