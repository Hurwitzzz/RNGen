This directory contains RNGen data-preparation settings and prompt templates.

| File | Purpose | Entrypoint |
| --- | --- | --- |
| `rna_sft_csv_seq_segment.yaml` | Phrase segmentation, feature selection, and context for Key and RNA data | `python -m write_seqs --data-settings ...` |
| `templates.json` | Prompts and output formats for Key, RNA, and the six Stage 2 tasks | Annotation generators |
| `.gitignore` | Keep configuration formats tracked while ignoring other local files | Git |

Public training and evaluation recipes are maintained in
[`configs/`](../../configs/README.md). RNGen has one shared SFT configuration
and one configuration per stage (Key, curriculum Stage 2, and RNA), plus a
standalone evaluation configuration. Model names
are selected with CLI overrides.

Replace the explicit dataset, output, and checkpoint placeholders before use.
All three training stages default
to effective batch 4 on one GPU; adjust per-device batch and gradient
accumulation when changing the number of training processes. The public
configuration README documents model choices, checkpoint dependencies,
and ablation settings.

Run from the repository root after editing paths:

```bash
python -m rngen.sft_ft --config configs/rngen/key.yaml
python -m rngen.sft_ft --config configs/rngen/curriculum_stage2.yaml
python -m rngen.sft_ft --config configs/rngen/rna.yaml
python -m rngen.sft_eval --config configs/rngen/evaluation.yaml
```

Baseline configurations and training launchers are documented in the same
configuration README.

`load_from_fted_rnbert` is a run-name label only. It does not select, load, or
validate weights. `rnbert_checkpoint_path` selects encoder weights when creating
a new RNGen model. A supplied `finetuned_rngen_checkpoint` instead loads the
saved RNGen model.

For `rna_sft_csv_seq_segment.yaml`, key transposition is already performed
upstream. `aug_by_key: false` prevents a second augmentation pass. This config
segments by phrase, uses a 500-note target for splitting long phrases at bar
boundaries, and adds one context bar on each side where available. Context
notes and a single oversized bar can make a segment exceed 500 notes.

The [complete CSV-to-JSON guide](../data_prep/README.md) covers preprocessing
installation, score manifests, label-aware transposition, encoding, and the
Key/RNA/Mixed builders. Its encoding command reads the complete
`preprocessing/sequence_settings.json` and `octuple_encoding_settings.json`
snapshots distributed with the dataset. This YAML is a concise configuration
for the same main feature and segmentation settings; both settings formats
are accepted by the sequence writer. The annotation builders read the
resulting `sequences/data/` directory, not the note-level processed CSVs.

`templates.json` is read when generating annotations, including by
`rngen/data_prep/build_key_annos.py` and
`rngen/data_prep/build_rna_annos.py`, as well as
`rngen/data_prep/build_rna_curriculum_stage2_annos.py`. Training reads the
template embedded in each annotation JSON. Editing `templates.json` therefore
requires regenerating annotations to change their prompts.

Key and ordinary RNA annotation generation take CLI paths with defaults from
`rngen/paths.py` (`RNGEN_DATA_ROOT`, `RNGEN_OUTPUT_ROOT`, and
`RNGEN_ANNOS_ROOT`). See [Annotation preparation](../data_prep/README.md)
for CSV inputs and path overrides.
