# RNA evaluation metrics

This reference describes the metrics computed for full Roman numeral analysis
by the released RNGen evaluation code. The
[evaluation guide](../../README.md#evaluate-released-checkpoints) explains how
to run `rngen.sft_eval` with `--task rna_sft`. Six-task Mixed evaluation reports
losses separately; it does not compute these generation metrics.

The implementation is shared by the training metric builder and saved-prediction
scoring tools:

- [metrics_rna_shared.py](../../rngen/training/metrics_rna_shared.py): reference
  spans, note/onset assignments, and metric aggregation.
- [metrics_rna_components.py](../../rngen/training/metrics_rna_components.py):
  component labels, pitch-class conversion, set metrics, and classwise reports.
- [metrics_rna.py](../../rngen/training/metrics_rna.py): evaluation integration
  and output files.

Accuracy, precision, recall, and F1 values are returned on a **0–1 scale**.
Paper tables display them as percentages. Counts and loss are not percentages.

## Evaluation units and denominators

Reference and predicted outputs are JSON objects mapping local keys to lists
of harmony strings. Harmonies are sorted by their bar and position. Each
harmony applies from its start, inclusive, to the next harmony's start,
exclusive; the final harmony extends through `metadata.last_target_bar`.
Quarterbeat positions are converted to event positions using
`position = round((quarterbeats - 1) * 16)`.

The scorer uses the input event sequence to assign harmonies to individual notes
and distinct `(bar, position)` onsets. Simultaneous notes share one onset.
Scoring is restricted to notes/onsets covered by the reference harmony spans;
context and unannotated portions outside those spans do not contribute.
An empty reference object contributes no scored notes or onsets.

Scores are aggregated across samples by pooling these reference assignments.
They are not an unweighted average over pieces, segments, or harmony changes,
and are not weighted by elapsed musical duration. A longer span contributes
more weight only when it covers more scored notes or onsets. Assignments are
counted within each sample; overlapping samples are not deduplicated across
the dataset.

`onset_level_total` and `note_level_total` report the resulting denominators.
The input `note_count` is checked for consistency, but does not replace the
reference-derived denominator. Component accuracies exclude reference labels
that cannot be canonicalized for that component. A missing or unsupported
prediction remains incorrect for a supported reference. Zero-denominator
accuracies are returned as zero.

References must be valid JSON objects. Scoring metadata supplies `events`,
`last_target_bar`, and `note_count`; `sample_id` identifies samples in consistency
warnings. The shared scorer logs and skips non-object references, missing or
invalid event sequences, and missing `last_target_bar`. Invalid reference JSON
raises an error.

## Full RNA accuracy

At each scored reference onset, a prediction is correct when both its local-key
string and full Roman-numeral string match the reference exactly. This includes
the Roman numeral's case, accidentals, quality markers, secondary function,
and inversion.
Equivalent spellings or alternative surface forms are not canonicalized for
this exact-match score.

| Metric | Definition |
| --- | --- |
| `onset_level_accuracy` | Correct reference onsets divided by `onset_level_total`; the paper's RNA accuracy. |
| `onset_level_correct`, `onset_level_total` | Correct and total scored onsets. |
| `note_level_accuracy` | The same exact key/RN match, weighted by individual notes. |
| `note_level_correct`, `note_level_total` | Correct and total scored notes. |

The standard RNA entrypoint does not require matching chord-tone or factor
strings for exact RNA accuracy. Those attributes are scored separately below.

## Component accuracies

Component labels are extracted from the parsed Roman numeral and local key.
Each accuracy counts exact matches of its canonical labels over supported
reference onsets.

| Metric | Canonical label |
| --- | --- |
| `K_acc` | Local-key string, preserving case and enharmonic spelling: `G`, `g`, and `Bb` are distinct labels. |
| `P_acc` | Primary alteration plus primary numeral, with the numeral uppercased. `I` and `i` match; `bII` and `II` do not. |
| `S_acc` | Secondary alteration plus secondary numeral, with the numeral uppercased. No secondary function is the empty string, which is a valid label. |
| `S_case_acc` | Secondary alteration and numeral with case preserved; `iii` and `III` differ. |
| `D_acc` | The ordered pair `(P, S)` using the case-insensitive P/S definitions. |
| `Q_acc` | The quality label returned by the RN parser, such as `M`, `m`, `Mm7`, or `o7`. |
| `I_acc` | The inversion label returned by the RN parser, such as `6`, `64`, or `65`; root position is the empty string. |

Whitespace around labels is stripped. P/S accuracy does not separately require
matching chord quality; Q scores that attribute. D does not use the
case-sensitive `S_case` variant.

## Chord-tone and tone-factor-pair F1

Chord tones and factors are read from hyphen-separated strings, for example
`G-B-D` and `R-T-F`. Chord-tone scoring constructs a set of tones. Tone-factor
scoring pairs the entries by their original list positions, then constructs
a set of `(tone, factor)` pairs. Set comparison removes duplicate entries and
does not require matching inventory order.

If the tone and factor lists have different lengths, the shorter list is padded
with `<missing-tone>` or `<missing-factor>`. Unpaired entries therefore remain
in the set comparison and can contribute false positives or false negatives.

| Metric | Items compared |
| --- | --- |
| `chord_tone_f1_spelling` | Spelled-tone strings; the paper's **CT-F1**. |
| `tone_factor_pair_f1_spelling` | Spelled-tone/factor pairs; the paper's **TF-F1**. |
| `chord_tone_f1_pc` | Tones converted to pitch classes modulo 12. |
| `tone_factor_pair_f1_pc` | Pitch-class/factor pairs; factor labels still match exactly. |

Spelling-sensitive scores distinguish enharmonic spellings such as `A#` and
`Bb`. Pitch-class scores merge them and also recognize repeated sharps/flats
and `x`/`X` double sharps. Unparsable pitch spellings receive nonmatching
placeholders in the pitch-class comparison.

For each scored reference onset, let A be the predicted set and B the reference
set. The set F1 is `2 * |A intersection B| / (|A| + |B|)`. Two empty sets score
1; an empty set against a nonempty set scores 0. An onset with no predicted
harmony receives 0, including when the reference inventory is empty.

Each reported CT/TF metric is the sum of these per-onset F1 values divided by
`onset_level_total`. It is not computed by pooling tone-level true positives
across the dataset. The set helpers also return precision, recall, exact match,
and TP/FP/FN for individual comparisons; the standard RNA summary reports the
four F1 metrics listed above.

## Secondary-function presence

`secondary_presence_precision`, `secondary_presence_recall`, and
`secondary_presence_f1` treat a nonempty canonical secondary function as the
positive class. Counts are pooled over reference onsets with a supported
secondary-function label. A missing or unparseable predicted function is treated
as absent. A prediction with no secondary function is a valid negative case.

Precision is `TP / (TP + FP)`; recall is `TP / (TP + FN)`; F1 is their harmonic
mean. A zero denominator returns zero.

## Absolute-pitch-class diagnostics

These diagnostics compare the pitches implied by a function after resolving
its tonal reference frame. They do not rewrite predicted Roman numerals into
the reference key and do not replace exact K/P/S/D accuracy.

- `P_abs_pc_acc`: compare primary-root pitch classes. Without a secondary
  function, resolve P in the local key. With a secondary function, first resolve
  the tonicized tonic, then interpret P in that key; the secondary numeral's
  case determines its major/minor mode.
- `S_abs_pc_acc`: compare the tonicized tonic pitch classes implied by the local
  key and S. Absence of a secondary function is the empty-string label.
- `D_abs_pc_acc`: compare the pair `(primary absolute pitch class,
  secondary absolute pitch class or empty string)`.

Local keys use uppercase tonic letters for major and lowercase for minor,
unless an explicit `major`/`minor` suffix is present. Relative degrees use major
offsets `0, 2, 4, 5, 7, 9, 11` or minor offsets `0, 2, 3, 5, 7, 8, 10`, plus
the alteration prefix. The supported `N` degree maps to a flattened second.
Results are reduced modulo 12.

Reference functions that cannot be mapped are excluded separately for each
diagnostic. Unsupported predictions count as incorrect rather than reducing
the denominator. The effective reference counts are returned as
`P_abs_pc_supported_onsets`, `S_abs_pc_supported_onsets`, and
`D_abs_pc_supported_onsets`.

## Classwise reports

The scorer produces reports for **K, P, S, Q, D, and I** using the same
canonical labels and scored reference onsets as their component accuracies.
There is no separate S_case classwise report.

Each class row contains `label`, `support_count`, `support_ratio`,
`predicted_count`, `tp`, `fp`, `fn`, `precision`, `recall`, and `f1`. D rows also
expose `primary_function` and `secondary_function`. Reference support counts
onsets, not samples or training-label frequencies. Predictions outside scored
reference onsets do not contribute to these classwise counts.

For each family, `total_onsets` is the total reference support. `macro_f1` is the
unweighted mean over classes with positive reference support, and `weighted_f1`
uses those support counts as weights. Predicted-only labels appear as rows with
zero support and are excluded from the macro average. Undefined class precision
or recall is zero.

## JSON format diagnostic and saved results

For samples reaching prediction scoring, `malformed_prediction_count` counts
predictions that cannot be recovered as JSON objects. The corresponding
`malformed_prediction_rate` divides this count by the number of supplied
predictions. JSON repair is attempted before counting a decode failure.
A repaired object is scored normally; a valid empty object is not a format
failure. Invalid individual harmony entries may be skipped with warnings
without increasing this sample-level count. A missing predicted harmony still
receives no matching credit at a scored reference onset.

The evaluation CLI prints metrics with an `eval_` prefix, including when its
input is the test split. The Trainer reports `eval_loss` separately from these
generation metrics.

With an output directory, RNA evaluation saves:

- `best_predictions.json` and timestamped prediction snapshots containing
  `predictions`, `references`, `metrics`, and `loss`, plus text-match records.
- `best_rna_classwise_metrics.json` and timestamped classwise JSON reports.

The `best_*` files track the lowest finite evaluation loss observed by that
metric builder. Snapshot `matches` compare decoded strings directly; they are
not the onset-level RNA score. No historical result files are required to run
these metrics.
