# RNGen annotation preparation

| Module | Input | Output |
| --- | --- | --- |
| `build_key_annos` | Segmented sequence CSVs | Key SFT JSONs |
| `build_rna_annos` | Segmented sequence CSVs with harmony features | Ordinary RNA SFT JSONs |
| `build_rna_curriculum_stage2_annos` | Ordinary RNA SFT JSONs | Six-task Stage 2 JSONs |

For reproduction with the released checkpoints, use the published JSON files
directly, as described in the root [training and evaluation guide](../../README.md).
The steps below prepare **new annotations** from the published processed scores:

1. Select the published score splits and transpose training scores into 12 keys.
2. Encode and segment the scores with `write_seqs` and `reprs`.
3. Build Key and RNA JSONs from the resulting sequence CSVs.
4. Derive the six-task Mixed JSONs from the RNA JSONs.

No model checkpoint or GPU is required for data preparation. Regeneration is
not a byte-for-byte reconstruction of the published JSONs: current accidental
token handling, random bar offsets, local source paths, and file traversal or
worker order can change event strings, metadata, and numeric sample IDs.

## Released dataset layout

```text
rngen/
├── key/                         # train.json, valid.json, test.json
├── mixed/                       # train.json, valid.json, test.json
├── rna/                         # train.json, valid.json, test.json
├── processed_scores/            # Note-level CSVs and their JSON metadata
│   └── README.md
└── preprocessing/
    ├── sequence_settings.json
    └── octuple_encoding_settings.json
```

`processed_scores/` contains unaugmented score CSVs, not the segmented sequence
CSVs consumed by these converters. Each CSV has a same-stem metadata JSON, and
each corpus directory contains `attrs.json`. Keep both kinds of metadata with
the CSVs: they provide the key, transposition columns, and corpus properties
used by preprocessing. Its README describes the metadata and training augmentation.
The files in `preprocessing/` record the sequence and encoding settings.
The released JSON contents and CSV score filenames are preserved; source paths
inside annotation metadata remain traceability records, not paths required for
training or evaluation.

## 1. Prepare the environment and paths

Download [RNGen-data](https://huggingface.co/datasets/Hurwitz/RNGen-data) to
`datasets/releases/rngen/` using the root README's
[download instructions](../../README.md#weights-and-data).
Run the following from the code checkout's root. These three submodules supply
music dataframes, Octuple encoding/segmentation, and sequence writing:

```bash
git submodule update --init -- music_df reprs write_seqs

python3.11 -m venv .venv-data-prep
source .venv-data-prep/bin/activate
python -m pip install \
  numpy==2.2.3 pandas==2.2.3 PyYAML==6.0.2 omegaconf==2.3.0 tqdm==4.67.1 \
  -e ./music_df -e ./reprs -e ./write_seqs

export PYTHONPATH="$PWD/music_df:$PWD/reprs:$PWD/write_seqs${PYTHONPATH:+:$PYTHONPATH}"
python -c "from write_seqs.write_seqs import CorpusItem; from reprs.oct import OctupleEncodingSettings"

export RELEASE_ROOT="$PWD/datasets/releases/rngen"
export SCORE_ROOT="$RELEASE_ROOT/processed_scores"
export SPLIT_ROOT="$PWD/data_splits/p_dcml"
export PREP_ROOT="$PWD/outputs/data_preparation"
```

Use the submodule revisions recorded in this checkout.
Keep the editable (`-e`) installation flags and the `PYTHONPATH` setup so
source-tree helper modules, including `write_seqs.utils`, remain importable
when running from the repository root.
The package metadata installs additional preprocessing dependencies, including
`dacite`, `mido`, `mspell`, and `music21`; `music_df` also declares a Git-sourced
`metricker` dependency, so Git and network access are needed during installation.
Neither `musicbert_hf` nor `musicbert_fork` is required for this CSV-to-JSON route.
The RNGen training environment is separate; reactivate it before training.

`PREP_ROOT` is a working directory for generated files, separate from the
published data. The example shell variables are passed explicitly to commands;
they do not change `rngen/paths.py`. Keep the same shell open for later steps.
Choose a new `PREP_ROOT` when regenerating data rather than overwriting an
existing run.

## 2. Materialize training augmentation

The original manifests select **355 train, 44 validation, and 47 test scores**.
Read them from `data_splits/p_dcml/`, not `data_splits/rnbert/`. Do not resplit
by scanning all 447 CSVs in the release. The augmented manifests identify
4,260 training variants and the unchanged validation/test scores.

The preparation example below uses the existing `transpose_to_key` API to
write a separate working corpus. It also transposes each absolute pitch
spelling in RNGen's hyphen-separated `chord_tones` column: that added column
is not handled by the generic upstream transposer's metadata alone.
Relative `gt_rnbert` labels and `chord_tone_factors` remain unchanged.

```bash
python - <<'PY'
import json
import os
from pathlib import Path
import shutil

import pandas as pd
from music_df.transpose import SpellingAlongLineOfFifthsTransposer, transpose_to_key
from write_seqs.write_seqs import CorpusItem

source = Path(os.environ["SCORE_ROOT"]).resolve()
split_root = Path(os.environ["SPLIT_ROOT"]).resolve()
destination = Path(os.environ["PREP_ROOT"]).resolve() / "augmented_scores"
assert source != destination and source not in destination.parents
assert destination not in source.parents
assert not destination.exists(), f"Choose a new working directory: {destination}"

splits = {
    split: [Path(line.strip()) for line in
            (split_root / f"{split}_paths.txt").read_text().splitlines() if line.strip()]
    for split in ("train", "valid", "test")
}
assert [len(splits[s]) for s in splits] == [355, 44, 47]
for paths in splits.values():
    for relative in paths:
        assert not relative.is_absolute() and ".." not in relative.parts
        original = source / relative
        assert original.is_file() and original.with_suffix(".json").is_file()
        assert (original.parent / "attrs.json").is_file()

transpose_spelling = SpellingAlongLineOfFifthsTransposer()
destination.mkdir(parents=True)
for split, paths in splits.items():
    for relative in paths:
        original = source / relative
        folder = destination / relative.parent
        folder.mkdir(parents=True, exist_ok=True)
        attrs_path = folder / "attrs.json"
        if not attrs_path.exists():
            shutil.copy2(original.parent / "attrs.json", attrs_path)
        if split != "train":
            shutil.copy2(original, folder / original.name)
            shutil.copy2(original.with_suffix(".json"), folder / original.with_suffix(".json").name)
            continue

        frame = CorpusItem(str(original)).read_df()
        original_signature = int(frame.attrs["global_key_sig"])
        for variant, target_signature in enumerate(range(-6, 6)):
            interval = target_signature - original_signature
            # A fresh copy prevents one transposition from affecting the next.
            augmented = transpose_to_key(frame.copy(deep=True), target_signature, inplace=True)
            def transpose_tones(value):
                if pd.isna(value) or value in ("", "na", "nan"):
                    return value
                return "-".join(transpose_spelling(tone, interval) for tone in value.split("-"))
            augmented["chord_tones"] = frame["chord_tones"].map(transpose_tones)
            if frame.attrs.get("global_key"):
                augmented.attrs["global_key"] = transpose_spelling(frame.attrs["global_key"], interval)
            output = folder / f"{original.stem}_aug_{variant}.csv"
            augmented.to_csv(output)
            with output.with_suffix(".json").open("w") as f:
                json.dump(augmented.attrs, f, indent=2, ensure_ascii=False)

for split in splits:
    manifest = split_root / "augmented_splits" / f"{split}_paths.txt"
    paths = [destination / line.strip() for line in manifest.read_text().splitlines() if line.strip()]
    assert all(path.is_file() and path.with_suffix(".json").is_file() for path in paths)
    print(split, len(paths))  # 4260, 44, 47
PY
```

This uses the 12 enharmonically distinct target key signatures from six flats
through five sharps. `_aug_0` through `_aug_11` identify those variants; they
are not semitone offsets. No rhythmic augmentation or pitch-range augmentation
is applied. Validation and test CSVs and score metadata are copied unchanged.
The new score metadata records the transposition and accompanies each CSV;
corpus `attrs.json` files are also retained. All outputs stay under `PREP_ROOT`.

## 3. Encode and segment the working corpus

Use the two settings files distributed with the data. Both JSON files are
accepted by the existing YAML/config readers:

```bash
python -m write_seqs \
  --src-data-dir "$PREP_ROOT/augmented_scores" \
  --input-paths-dir "$SPLIT_ROOT/augmented_splits" \
  --data-settings "$RELEASE_ROOT/preprocessing/sequence_settings.json" \
  --repr-settings "$RELEASE_ROOT/preprocessing/octuple_encoding_settings.json" \
  --output-dir "$PREP_ROOT/sequences" \
  --seed 42 --num-workers 1
```

`--input-paths-dir` fixes score membership to the published manifests instead
of asking `write_seqs` to create a new split. Paths inside those manifests
are relative to `--src-data-dir`. Use an absolute output path, as above;
relative outputs are resolved against `WRITE_SEQS_BASE_DIR` if it is set.
One worker keeps the example small and avoids parallel chunk completion order;
it does not promise the historical file or sample order.

The sequence settings retain `key`, `spelling`, `mn`, `gt_rnbert`,
`chord_tones`, and `chord_tone_factors`. They use phrase segmentation,
`max_num_notes: 500`, `num_context_bars: 1`, no short-phrase merging, and
long-phrase splitting. Splitting occurs at bar boundaries: context adds input
notes, and a single oversized bar is not split to enforce a strict 500-note
total. Context is included on both sides where available; only target bars
are labeled.

Keep `aug_by_key: false` here because step 2 has already transposed training
scores. Enabling it again would augment them a second time. The maintained
`rna_sft_csv_seq_segment.yaml` in [`rngen/config/`](../config/README.md)
provides the same main segmentation/feature settings for customization.

Outputs include settings snapshots, vocabulary bookkeeping, and sequence CSVs:

```text
<PREP_ROOT>/sequences/
├── SequenceDataSettings_settings.json
├── OctupleEncodingSettings_settings.json
└── data/
    ├── train/*.csv
    ├── valid/*.csv
    └── test/*.csv
```

Each CSV row is one encoded segment; each chunk file can contain multiple
scores. The JSON builders consume **`sequences/data/`**, not the note-level
`augmented_scores/` folder. Warnings about absent feature vocabulary JSONs
do not prevent these annotation builders from using the feature strings.

## 4. Build Key JSONs

The input directory contains `train/`, `valid/`, and `test/`, each with sequence
CSVs. Required columns are `events`, `key`, `target_bar_masks`,
`target_bar_numbers_w_offset`, and `csv_path`; `spelling` is also required when
spelling tokens are enabled. The `key` sequence has one leading `<s>` and one
label per note. Event groups contain eight tokens per note before conversion.

This conversion only requires Python, `tqdm`, and this repository; it does not
import a submodule. Preparing these sequence CSVs from labeled note-level CSVs
uses `write_seqs` (dataset orchestration), `reprs` (Octuple encoding and phrase
segmentation), and `music_df` (music dataframe processing). MusicBERT model
implementations are not needed for that preprocessing step. The upstream CSVs
must already provide the required labels and features. Key transposition is
performed upstream in the paper pipeline.

### Accidental event tokens

Spelling-aware inputs use seven accidental tokens: `<8-h>`, `<8-#>`, `<8-##>`,
`<8-###>`, `<8-b>`, `<8-bb>`, and `<8-bbb>`. Triple accidentals retain their own
tokens during both Key and RNA annotation generation. Token IDs are defined in
`rngen/data/events_incl_spelling_stoi.json`; the triple-sharp and triple-flat IDs
are 1241 and 1242. Fresh encoders size their embedding tables using the maximum
ID plus one (1243 rows).

Regenerate Key and RNA JSONs from the sequence CSVs to use these tokens, then
build Stage 2 annotations from the generated RNA JSONs. Write them to a separate
output directory to retain annotations used with released checkpoints.
Existing JSONs are read as stored; changing the converter does not update them.
Existing checkpoints retain their saved embedding sizes. A checkpoint with
1242 rows cannot accept token ID 1242; use matching annotations or explicitly
adapt and train the encoder before using the expanded input vocabulary.

### Paths and invocation

Shared defaults live in `rngen/paths.py`:

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `RNGEN_DATA_ROOT` | `<repo>/data` | Dataset base directory |
| `RNGEN_OUTPUT_ROOT` | `<repo>/outputs` | General experiment-output base; not used by this converter |
| `RNGEN_ANNOS_ROOT` | `<data root>/annotations` | Generated annotation base directory |

The default input is
`<data root>/rngen_rna_sft_DcmlCorpora_key_augmented/phrase_max500/data`;
the default output is `<annotation root>/key`.
Templates default to `rngen/config/templates.json` in this checkout.
Environment variables are read at process startup. CLI arguments override the
derived defaults. Other existing launchers have not yet been migrated to this
path module.

Run from the repository root:

```bash
python -m rngen.data_prep.build_key_annos \
  --data-dir "$PREP_ROOT/sequences/data" \
  --output-dir "$PREP_ROOT/annotations/key" \
  --templates-path "$PWD/rngen/config/templates.json"
```

The CLI keeps the original entrypoint's `target_bars_only=True` and
`include_spelling=True`. Use `--no-target-bars-only` or `--no-include-spelling`
to disable them, `--templates-path` to choose another template file, and
`--splits test` to process a single split. Generated filenames are
`train.json`, `valid.json`, and `test.json`. Filename suffixes are optional
and can be supplied with `--suffix`. Use separate output directories for
different input variants.

### Compatibility

Conversion preserves key-region boundaries, response text, note counting,
metadata, and JSON layout. Accidental tokens distinguish single, double, and
triple sharps/flats. Malformed inputs now
raise an error with the source file and row instead of continuing with misaligned
events. In particular, failed spelling conversion cannot fall back to an
eight-token sequence while being interpreted as nine-token groups.

The Key builder selects `KEY_SFT` and produces textual key regions. It does
not generate Roman-numeral harmony labels.

## 5. Build RNA JSONs

`build_rna_annos.py` uses the Key event helpers and
`rngen.utils.rna.parse_gt_rnbert` to produce ordinary RNA SFT annotations.

The input is the same sequence CSV directory used by Key preparation. RNA
conversion reads `events`, `gt_rnbert`, `key`, `spelling`, `chord_tones`,
`chord_tone_factors`, `target_bar_numbers_w_offset`, and `csv_path`.
The default output is `<annotation root>/rna`, configured by `RNA_ANNOS_DIR`
in `rngen/paths.py`.

```bash
python -m rngen.data_prep.build_rna_annos \
  --data-dir "$PREP_ROOT/sequences/data" \
  --output-dir "$PREP_ROOT/annotations/rna" \
  --templates-path "$PWD/rngen/config/templates.json"
```

This writes `train.json`, `valid.json`, and `test.json`.
Use `--no-include-spelling` to generate eight-token events in a separate output
directory, or `--splits test` to process one split. `--suffix` optionally appends
a filename suffix after the split name. `--templates-path` selects the file containing `RNA_SFT`.
The default labels include chord tones and factors and retain the historical
quarterbeat convention `position / 16 + 1`. The output also includes note counts,
source metadata, and `rna_info` for later Stage 2 conversion and scoring.

CSV traversal and numeric sample-ID assignment retain the historical behavior.
Filesystem traversal order can differ between machines, so identical source-row
content can receive different numeric IDs. No sorting or reference-based
reordering is applied. For exact historical IDs, use the published JSON files.

Use the annotation files associated with a released checkpoint when reproducing
its reported evaluation results. Regenerating annotations with seven accidental
tokens changes the encoder inputs of examples containing triple accidentals.

## 6. Build Mixed JSONs

`build_rna_curriculum_stage2_annos.py` converts existing RNA SFT JSONs into
six-task Stage 2 JSONs. Source and output directories remain required CLI
arguments. Its default template path comes from `rngen/paths.py`; task
selection, sample ordering, limits, and serialization are unchanged.

```bash
python -m rngen.data_prep.build_rna_curriculum_stage2_annos \
  --source-dir "$PREP_ROOT/annotations/rna" \
  --output-dir "$PREP_ROOT/annotations/mixed" \
  --template-path "$PWD/rngen/config/templates.json"
```

It reads `train.json`, `valid.json`, and `test.json` by default and writes the
corresponding Mixed JSONs plus `build_summary.json`. The optional `--suffix`
applies to both source and output filenames. Use `--help` for task,
split, suffix, and formatting options.

Mixed produces one example for each of six tasks per source RNA segment:
`harmony_span_prediction`, `chord_tone_inventory_prediction`,
`tone_factor_pair_prediction`, `quality_prediction`,
`inversion_figure_prediction`, and `ps_function_prediction`. Music sequences
are shared through `seq_data`. Do not use `--max-samples-per-split` or select
fewer tasks when preparing the complete training set.

## 7. Inspect the outputs and train

Check that all nine annotation files are present and that Mixed has six
examples per RNA example:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PREP_ROOT"]) / "annotations"
for split in ("train", "valid", "test"):
    counts = {}
    for stage in ("key", "rna", "mixed"):
        with (root / stage / f"{split}.json").open() as f:
            annotations = json.load(f)
        counts[stage] = sum(key.isdecimal() for key in annotations)
        assert counts[stage] > 0
        assert "template" in annotations
    assert counts["key"] == counts["rna"]
    assert counts["mixed"] == 6 * counts["rna"]
    print(split, counts)
PY
```

The released Key/RNA files have 53,928/510/621 train/valid/test examples;
Mixed has 323,568/3,060/3,726. These are reference counts for the **released**
files, not a promise that every preprocessing run reproduces their exact
serialization or sample IDs.

Activate the RNGen training environment, then follow the root
[training guide](../../README.md#train-a-new-model), setting each stage's
`jsonl_dir` to the corresponding directory under
`<PREP_ROOT>/annotations/`. Use `configs/rngen/key.yaml`,
`configs/rngen/curriculum_stage2.yaml`, and `configs/rngen/rna.yaml` for
Key, Mixed, and RNA respectively. The parameter is named `jsonl_dir`, but
the files are JSON dictionaries, not JSON Lines. Each stage requires
`train.json`, `valid.json`, and `test.json`; the training entrypoint evaluates
the test split after training.
