# Third-party code notices

The root [LICENSE](LICENSE) covers original RNGen contributions. Third-party
code retains its applicable upstream copyright and license terms. This file
records source-code attribution and reproduces the verified notices below.
Model checkpoints and datasets have separate licensing terms.

## Baseline implementations

| RNGen component | Upstream source and revision | Copyright notice | License text |
| --- | --- | --- | --- |
| `rngen/baselines/augnet_mi20/` (AugmentedNet and the Micchi2020 implementation provided by AugmentedNet) | [napulen/AugmentedNet, v1.0.0, 709dc59470412453fd9046ef4fb53e5663141144](https://github.com/napulen/AugmentedNet/tree/709dc59470412453fd9046ef4fb53e5663141144) | Copyright (c) 2021 Néstor Nápoles López | [MIT — AugmentedNet](#augmentednet) |
| `rngen/baselines/chordgnn/` and `rngen/baselines/chordgnn_post/` | [manoskary/ChordGNN, bcbdc72d2850e986c5f577727f800dee0f95f0c4](https://github.com/manoskary/ChordGNN/tree/bcbdc72d2850e986c5f577727f800dee0f95f0c4) | Copyright (c) 2023 Emmanouil Karystinaios | [MIT — ChordGNN](#chordgnn) |

RNGen adapts these implementations to its data conversion, training, evaluation,
and checkpoint interfaces. The upstream copyright and permission notices
remain applicable to the adapted code.

## Submodules

These submodules are maintained in forks of Malcolm Sailor's repositories.
The revisions below identify the source trees referenced by RNGen. Authorship
is taken from the upstream package metadata and source credits.

| Submodule | Upstream author and repository | Referenced fork revision | Upstream license information |
| --- | --- | --- | --- |
| `musicbert_hf` | Malcolm Sailor; [malcolmsailor/musicbert_hf](https://github.com/malcolmsailor/musicbert_hf) | [f71524010dac152a124b47f9c2d4b9735a9a55ed](https://github.com/Hurwitzzz/musicbert_hf/tree/f71524010dac152a124b47f9c2d4b9735a9a55ed) | Full MIT notice in `LICENSE`; reproduced [below](#musicbert-hf). |
| `music_df` | Malcolm Sailor; [malcolmsailor/music_df](https://github.com/malcolmsailor/music_df) | [b89656d66688dd5bde354b3f4a2dcea6689405a0](https://github.com/Hurwitzzz/music_df/tree/b89656d66688dd5bde354b3f4a2dcea6689405a0) | MIT, declared in `pyproject.toml`. See the [optional MS3 conversion](#optional-ms3-conversion) note. |
| `reprs` | Malcolm Sailor; [malcolmsailor/reprs](https://github.com/malcolmsailor/reprs) | [d29aaa21c6f06ee9d99dab39d31a3340ef2b2a02](https://github.com/Hurwitzzz/reprs/tree/d29aaa21c6f06ee9d99dab39d31a3340ef2b2a02) | MIT, declared in `setup.py`. |
| `write_seqs` | Malcolm Sailor; [malcolmsailor/write_seqs](https://github.com/malcolmsailor/write_seqs) | [733acc27da6be3c9f82d38f372690166fc262a60](https://github.com/Hurwitzzz/write_seqs/tree/733acc27da6be3c9f82d38f372690166fc262a60) | MIT, declared in `setup.py`; the hash utility has the [reader attribution](#reader). |
| `musicbert_fork` | Malcolm Sailor's RNBERT adaptations; [malcolmsailor/musicbert_fork](https://github.com/malcolmsailor/musicbert_fork) | [f87ac934e1cceea15d3120adcfdcfe0aea4fde2f](https://github.com/Hurwitzzz/musicbert_fork/tree/f87ac934e1cceea15d3120adcfdcfe0aea4fde2f) | Identified MusicBERT and Fairseq components carry MIT notices, reproduced below. The fork has no repository-wide license declaration. |

## Sources credited within submodules

- **Microsoft MusicBERT:** `musicbert_fork/musicbert/_musicbert.py` retains
  Microsoft's copyright and MIT header. `reprs/reprs/oct.py` also credits
  MusicBERT preprocessing code. The original implementation is in
  [microsoft/muzic, musicbert/](https://github.com/microsoft/muzic/tree/main/musicbert).
  The [Microsoft MIT notice](#microsoft-musicbert) is reproduced below.
- **Fairseq:** `musicbert_fork/musicbert/token_classification.py` credits
  [Fairseq pull request 1709](https://github.com/facebookresearch/fairseq/pull/1709);
  `musicbert_fork/musicbert/freezable_roberta.py` credits
  [pull request 1710](https://github.com/facebookresearch/fairseq/pull/1710).
  The [Fairseq MIT notice](#fairseq) is reproduced below for the identified
  Fairseq-based code.
- **reader:** `write_seqs/write_seqs/utils/get_hash.py` credits
  [lemon24/reader's hash utility at 1efcd38c78f70dcc4e0d279e0fa2a0276749111e](https://github.com/lemon24/reader/blob/1efcd38c78f70dcc4e0d279e0fa2a0276749111e/src/reader/_hash_utils.py).
  Its [BSD-3-Clause notice](#reader) is reproduced below.

## Optional MS3 conversion

`music_df/music_df/conversions/ms3.py` credits
[MS3](https://github.com/johentsch/ms3), a GPL-3.0 project, for its
`tpc2name` conversion. This module is outside RNGen's training, evaluation,
and released-CSV-to-JSON workflows. Submodule source is obtained separately
and is excluded from RNGen's wheel and source distributions.

Preserve the credited upstream terms when using or redistributing this
optional conversion code. The MIT package declarations above do not
relicense MS3 material. Separately installed dependencies retain their own
licenses and notices.

## Full upstream copyright and license notices

### AugmentedNet

Source: [upstream license](https://github.com/napulen/AugmentedNet/blob/709dc59470412453fd9046ef4fb53e5663141144/LICENSE).

```text
MIT License

Copyright (c) 2021 Néstor Nápoles López

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### ChordGNN

Source: [upstream license](https://github.com/manoskary/ChordGNN/blob/bcbdc72d2850e986c5f577727f800dee0f95f0c4/LICENSE).

```text
MIT License

Copyright (c) 2023 Emmanouil Karystinaios

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### MusicBERT HF

Source: [upstream license](https://github.com/Hurwitzzz/musicbert_hf/blob/f71524010dac152a124b47f9c2d4b9735a9a55ed/LICENSE).

```text
Copyright 2025 Malcolm Sailor

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

### Microsoft MusicBERT

Source: [upstream license](https://github.com/microsoft/muzic/blob/main/LICENSE).

```text
    MIT License

    Copyright (c) Microsoft Corporation.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE
```

### Fairseq

Source: [upstream license](https://github.com/facebookresearch/fairseq/blob/main/LICENSE).

```text
MIT License

Copyright (c) Facebook, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### reader

Source: [upstream license](https://github.com/lemon24/reader/blob/1efcd38c78f70dcc4e0d279e0fa2a0276749111e/LICENSE).

```text
Copyright (c) 2018 by lemon24.

Some rights reserved.

Redistribution and use in source and binary forms of the software as well
as documentation, with or without modification, are permitted provided
that the following conditions are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the following
  disclaimer in the documentation and/or other materials provided
  with the distribution.

* The names of the contributors may not be used to endorse or
  promote products derived from this software without specific
  prior written permission.

THIS SOFTWARE AND DOCUMENTATION IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE AND DOCUMENTATION, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
```

