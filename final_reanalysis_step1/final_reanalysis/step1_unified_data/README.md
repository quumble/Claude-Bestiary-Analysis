# Step 1 — Unified data + final-pass heuristic + hand-code sample

This folder is the substrate for Step 2's hand-coding and Step 3's
cross-study analysis. Everything here is generated from the three source
repos by the scripts in `scripts/`.

## What's here

| path | what | who needs it |
|---|---|---|
| `SCHEMA.md` | row-by-row schema for the unified corpus | anyone reading the JSONL |
| `data/unified_corpus.jsonl` | 14,045 trials × 30+ fields | every downstream analysis |
| `data/unified_corpus.csv` | same, but response truncated to 1200 chars | spreadsheet inspection |
| `data/unified_corpus_v4_coded.jsonl` | unified corpus + v4 code + features | Step 3 |
| `data/unified_corpus_v4_coded.csv` | same, easier for pandas | analysis scripts |
| `data/unified_manifest.csv` | per-source load counts (provenance) | reproducibility |
| `data/heuristic_compare_master600.csv` | v4 vs BC v3 vs QQ v3.1 vs hand codes (599 trials) | the calibration evidence |
| `handcode_sample/handcode_sample.jsonl` | 396 stratified trials to hand-code | Step 2 |
| `handcode_sample/handcode_sample.csv` | same, spreadsheet-friendly | optional fast scan |
| `tools/handcoder.html` | self-contained hand-coder, sample baked in | Step 2 work session |
| `tools/handcoder_template.html` | template the build script fills in | regenerating the tool |

## Heuristic calibration result

Three coders evaluated against the master-600 hand-coded sample
(599 trials; OTHER_REVIEW row excluded):

| coder | raw agreement | Cohen's κ | DEFLECT F1 | DESCRIBE F1 | HYBRID F1 | SUBSTITUTE F1 |
|---|---:|---:|---:|---:|---:|---:|
| BC v3 (BC paper) | 69.8% | 0.537 | 79.1 | 52.9 | 71.8 | 22.2 |
| QQ v3.1 (QQ paper) | 74.5% | 0.516 | 87.3 | 53.1 | 57.5 | 0.0 |
| **v4 (this folder)** | **80.3%** | **0.675** | **92.0** | **62.4** | **73.4** | **27.9** |

Reading: v4 is uniformly the best of three. The biggest single win is
DEFLECT precision (a structural-fallback guard prevents bullet-rich
deflections from being labelled DESCRIBE). SUBSTITUTE remains the
weakest class because the dominant SUBSTITUTE failure mode is
bare-assertion ("a tavuni is a traditional West African basket") which
no regex can distinguish from true description without world knowledge.

## Hand-coding sample design

The 396-trial sample is stratified into 8 blocks of 50 each (some smaller
where the block has fewer available rows). Master-600 trials are excluded
so hand-coding effort isn't duplicated.

| block | what it tests | n |
|---|---|---:|
| `AB_imaginary_hybrid` | v4's prompt-echo handling under `imaginary_*` | 50 |
| `AB_real_describe` | hard-confab candidates under `real_*` on AB stimuli | 50 |
| `BC_substitute_candidates` | v4's SUBSTITUTE precision | 50 |
| `BC_describe_candidates` | bare-substitute false negatives (BC's audit problem) | 50 |
| `QQ_F4_commits` | the naturalist-loophole effect, central QQ finding | 50 |
| `QQ_F1_deflect` | F1 directive baseline (for QQ's F4-vs-F1 contrast) | 50 |
| `Sonnet_real_any` | "Sonnet floor" — paper's strongest cross-architecture finding | 50 |
| `Haiku_trolnique_real` | Haiku's word-specific confabulation pattern | 46 |

## Workflow

Open `tools/handcoder.html` in a browser. You see the prompt, the
response, and v4's prediction with the features it computed. Code with
**D**escribe, **H**ybrid, **S**ubstitute, de**F**lect (because D is taken),
**R**efuse, or **O**ther.

Keyboard: D/H/S/F/R/O to code; J/← previous; K/→/Space next; U jumps to
next uncoded.

Codes save to localStorage. When done, **Export JSONL** writes a file with
all 396 trials plus your `hand_code` and optional `hand_note` fields.

## Reproducing this folder

See the top-level `README.md` for the command sequence. Each script is
idempotent: running it twice produces the same output.
