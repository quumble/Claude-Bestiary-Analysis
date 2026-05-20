# The Bestiary Reanalysis

A unified re-coding and cross-study analysis of the three-paper line:

- **The Artificial Bestiary** — three studies, four models, ~5,000 trials. How
  language models respond to nonce-word stimuli framed as real/imaginary/type-of
  × animal/object/idea. [Zenodo](https://github.com/quumble/the-artificial-bestiary)
- **Bestiary Chess** — three sub-studies, ~4,800 trials. How nonce stimuli's
  surface form and author modulate the response, including the
  GPT-on-GPT-words vs GPT-on-Claude-words asymmetry.
  [Zenodo](https://github.com/quumble/Bestiary-Chess)
- **The Question Question** — one balanced 4,140-trial crossover. How the
  speech act of the prompt (directive vs question vs statement vs naturalist
  vs first-person) interacts with model family. The naturalist-frame effect.
  [Zenodo](https://github.com/quumble/The-Question-Question)

Each paper was written against its own coded subset using its own heuristic.
Each paper's heuristic was calibrated against its own hand-coded validation
sample. The three studies' findings are consistent in shape but were never
re-coded under one codebook against one hand-coded calibration corpus. This
project does that.

The product is a single analytical pass over all ~14,045 trials with one
final-pass heuristic, calibrated against the master-600 hand-coded sample
plus a fresh hand-coded calibration of ~400 boundary trials, and a written
synthesis that asks whether the three papers' headlines hold up when the
coding is unified.

## Roadmap

### Step 1 — Unification, recoding, hand-code sample (THIS FOLDER)

Status: **complete**.

- `step1_unified_data/data/unified_corpus.jsonl` — every trial from every
  source run, harmonized to a single schema (documented in
  `step1_unified_data/SCHEMA.md`). 14,045 rows. Each row carries its
  source identifiers, its master-600 hand code if applicable, and (after
  step 1's heuristic pass) a v4 label plus the discrete features the
  heuristic computed.
- `step1_unified_data/scripts/unify_corpus.py` — the loader; one entry
  per source run, projects each into the canonical schema.
- `step1_unified_data/scripts/final_pass_heuristic_v4.py` — the unified
  v4 heuristic. Synthesis of BC v3 and QQ v3.1 with prompt-echo handling
  and a structural-fallback guard that fixes the dominant DEFLECT→DESCRIBE
  errors both prior coders made.
- `step1_unified_data/scripts/heuristic_diagnostics.py` — runs all three
  coders (BC v3, QQ v3.1, v4) against the master-600 hand codes and
  reports raw agreement, Cohen's κ, and per-class F1.
- `step1_unified_data/handcode_sample/handcode_sample.jsonl` — 396 trials
  stratified across 8 sample blocks chosen to be the places where v4
  is most likely to be wrong or where the empirical findings most depend
  on the coding. Excludes master-600 to avoid double work.
- `step1_unified_data/tools/handcoder.html` — self-contained HTML
  hand-coder. Loads the sample inline. Codes save to localStorage.
  Export JSONL when done.

Final-pass heuristic vs master-600 hand codes (n=599, OTHER_REVIEW excluded):

| heuristic | raw agreement | Cohen's κ |
|---|---:|---:|
| BC v3 | 69.8% | 0.537 |
| QQ v3.1 | 74.5% | 0.516 |
| **v4** | **80.3%** | **0.675** |

v4 is the best of three by a meaningful margin. The remaining ~20% disagreement
with master-600 is what Step 2's hand-coding will help arbitrate.

### Step 2 — Hand-coding pass + final heuristic calibration

Status: **awaiting Bo's hand-coding**.

You open `step1_unified_data/tools/handcoder.html` in a browser and code
through the 396 trials. Each trial shows the prompt, the response, and v4's
prediction (with the features it used to get there). You can agree by pressing
the matching key, or override.

Codes save to localStorage as you go. When you're done, click **Export JSONL**
and that file is the hand-coded calibration set.

Once that file is in hand, Step 2's analysis does three things:

1. Compute v4's κ against this fresh calibration set (independent of master-600).
   Use it as a cross-check on the master-600-only validation in Step 1.
2. Identify systematic v4 errors (cells where Bo overrides at high rates) and
   produce a v4.1 with regex fixes for the dominant error patterns.
3. Apply v4.1 to the full 14,045-trial corpus, producing the final coded
   corpus that Step 3 analyzes.

Step 2 also does a within-coder consistency check: a subset (say 30 trials)
gets shown to Bo twice in shuffled order, with the v4 prediction hidden the
second time. This is the same trick the QQ paper used for its blinded recode.

### Step 3 — Cross-study reanalysis and the unifying paper

Status: **after Step 2**.

This is the synthesis paper. It uses the v4.1-coded corpus to reproduce
or revise the headline findings of each of the three papers under one
codebook, and asks the cross-study question each paper said it could not
answer:

- **AB** said the cross-architecture differences (Sonnet floor, GPT
  confabulation) might be hallucination rate, or might be something else.
- **BC** said stimulus author drives 13× variation in confabulation rate.
- **QQ** said speech act drives a factor-of-three swing — and that the
  three studies between them found three orthogonal levers, each capable
  of moving the headline by a factor of three, and that any single
  pooled hallucination rate averages over all three.

The Step 3 paper is the one that says what the three together amount to,
under a coding pass that doesn't depend on different heuristics for different
slices of the corpus. If the headlines hold up, that's worth saying. If
they don't all hold up, that's worth saying too. The point is to find out.

### What's not in scope

- Re-running any model. The four (Sonnet, Haiku, GPT-5.4-mini,
  GPT-5.4-nano) plus Opus 4.7 (QQ) outputs are what we have. No new
  generations.
- A non-Claude analyst on the synthesis. QQ §10 said this would be the
  ideal next step. It would be a different project; this one stays inside
  the family.
- Settling the SUBSTITUTE-audit problem (whether bare-assertion
  substitutions like "evaruq → walrus" or "tavuni → West African basket"
  are real retrievals or confident fabrications). v4 catches some but
  not all; the audit is named as a Step 3 caveat, not addressed.

## How to use this folder

```bash
# Unify the corpus from the three unpacked repos
python3 step1_unified_data/scripts/unify_corpus.py \
  --repo-root . \
  --out-dir step1_unified_data/data

# Code it with the final-pass heuristic
python3 step1_unified_data/scripts/final_pass_heuristic_v4.py \
  --in-jsonl step1_unified_data/data/unified_corpus.jsonl \
  --out-csv step1_unified_data/data/unified_corpus_v4_coded.csv \
  --out-jsonl step1_unified_data/data/unified_corpus_v4_coded.jsonl

# Compare v4 against prior heuristics on the master-600 hand codes
python3 step1_unified_data/scripts/heuristic_diagnostics.py \
  --corpus-jsonl step1_unified_data/data/unified_corpus.jsonl \
  --bc-v3 Bestiary-Chess-main/Bestiary_Chess_3/heuristic_v3.py \
  --qq-v3-1 The-Question-Question-main/Naturalist_Crossover_Test/qq_heuristic_v3_1.py \
  --v4 step1_unified_data/scripts/final_pass_heuristic_v4.py \
  --out-csv step1_unified_data/data/heuristic_compare_master600.csv

# Build the stratified hand-code sample (already done; rerun to reshuffle)
python3 step1_unified_data/scripts/build_handcode_sample.py \
  --coded-jsonl step1_unified_data/data/unified_corpus_v4_coded.jsonl \
  --n-per-block 50 \
  --exclude-master-600 \
  --out-jsonl step1_unified_data/handcode_sample/handcode_sample.jsonl \
  --out-csv step1_unified_data/handcode_sample/handcode_sample.csv

# Build the hand-coder HTML with the sample embedded
python3 step1_unified_data/scripts/build_handcode_tool.py \
  --template step1_unified_data/tools/handcoder_template.html \
  --sample-jsonl step1_unified_data/handcode_sample/handcode_sample.jsonl \
  --out step1_unified_data/tools/handcoder.html
```

Then: open `step1_unified_data/tools/handcoder.html` in a browser and code.

## Folder layout

```
final_reanalysis/
├── README.md                         (this file)
├── step1_unified_data/
│   ├── SCHEMA.md                     unified row schema docs
│   ├── data/
│   │   ├── unified_corpus.jsonl      14,045 rows, full text
│   │   ├── unified_corpus.csv        same, response truncated
│   │   ├── unified_corpus_v4_coded.jsonl
│   │   ├── unified_corpus_v4_coded.csv
│   │   ├── unified_manifest.csv      per-source load counts
│   │   └── heuristic_compare_master600.csv
│   ├── scripts/
│   │   ├── unify_corpus.py
│   │   ├── final_pass_heuristic_v4.py
│   │   ├── heuristic_diagnostics.py
│   │   ├── build_handcode_sample.py
│   │   └── build_handcode_tool.py
│   ├── handcode_sample/
│   │   ├── handcode_sample.jsonl     396 trials, 8 strata
│   │   └── handcode_sample.csv
│   └── tools/
│       ├── handcoder_template.html   template for build_handcode_tool.py
│       └── handcoder.html            self-contained, open in browser
└── (step2/ and step3/ to follow)
```
