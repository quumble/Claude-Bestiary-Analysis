# Unified corpus schema

`unified_corpus.jsonl` is one JSON object per trial. The schema is a
superset of QQ's row shape, with AB-style and BC-style fields preserved
where they exist.

## Identity

| field | type | description |
|---|---|---|
| `global_trial_id` | str | stable per-trial hash. Format: `{source_run}::{source_row:05d}::{sha1[:12]}`. Deterministic across re-runs of the unifier. |
| `study` | str | `"Bestiary"`, `"Bestiary Chess"`, or `"Question Question"` |
| `source_run` | str | one of nine labels: `bestiary_sonnet_pilot`, `bestiary_sonnet_retest`, `bestiary_gpt_nanomini_retest`, `bestiary_haiku_retest`, `chess1_sonnet_manifest`, `chess2_gpt_on_claude_words`, `chess3_gpt_on_gpt_words`, `qq_v1_prerun`, `qq_crossover_completion` |
| `source_file` | str | relative path back to the source JSONL/CSV |
| `source_row` | int | **1-indexed** row number within the source file, matching the convention used by the master-600 hand-coded sample, so master codes can be joined back via `(source_run, source_row)`. |
| `trial_id_original` | str | whatever per-trial ID the source carried, if any (`"00633"`, `"r00641"`, etc.). Empty when absent. |

## Stimulus

| field | description |
|---|---|
| `word` | the nonce word, e.g. `borthorpunius`, `mavika`, `trolnique` |
| `word_author` | `"claude"` / `"gpt"` / `""`. Populated explicitly only for runs where author is unambiguous from construction. For AB pilot the field is empty (the author was Chesterton; mark as `""` rather than `"claude"`). |
| `word_meta` | extra annotation the source carried (mostly QQ-only) |
| `word_set` | for AB retest: `"original"` or `"analyst"`. For BC: `"gpt_authored"`, `"claude_authored"`, or BC1's `"opaque_ish"` etc. |

## Framing

| field | description |
|---|---|
| `status` | reality status: `"real"`, `"imaginary"`, `"neutral"`, or `""`. AB and BC populate this; QQ has all `"real"` (the manipulation is the frame, not the reality claim). |
| `reality` | same as `status` for AB/BC. Distinct field for QQ where the source schema kept them separate. |
| `category` | `"animal"`, `"object"`, `"idea"`, or `""`. |
| `ontology` | synonym BC used for `category`. Kept for trace. |
| `condition` | AB/BC: `"real_animal"`, `"imaginary_idea"`, etc. QQ: `"real_animal"` (constant) but the discriminator is `frame_id`. |
| `frame_id` | QQ only: `F1`–`F5`. Empty for AB/BC. |
| `frame_name` | QQ only: `directive`, `question`, `statement`, `3rd_person` (the naturalist frame), `1st_person`. |
| `speech_act`, `person` | QQ-specific metadata |
| `prompt` | exact prompt text issued to the model |

## Trial bookkeeping

| field | description |
|---|---|
| `rep_n` | replicate number within (word × condition × model) cell |
| `trial_n` | alt name some sources used; same value as `rep_n` |

## Model

| field | description |
|---|---|
| `model` | exact model string from the source (`"gpt-5.4-mini"`, `"claude-haiku-4-5-20251001"`, etc.) |
| `model_family` | `"GPT"` or `"Claude"` |
| `model_tier` | `"nano"`, `"mini"`, `"main"` (GPT) / `"haiku"`, `"sonnet"`, `"opus"` (Claude) |

## Response

| field | description |
|---|---|
| `response` | full response text. Some sources used `response_text` (chess1) or `output_text` (GPT nano/mini CSV); the unifier normalizes the field name. |

## Prior labels (trace only — never used as ground truth)

| field | description |
|---|---|
| `prev_pass1_code` | first-pass code from the master-600 hand-coding pipeline. Empty unless this trial was sampled into the master-600. |
| `prev_adjudicated_code` | final adjudicated code from the master-600. The reliable "ground truth" reference where present. |
| `prev_in_master_600` | `"1"` if this trial appears in `harmonized_handcoding_final_600_unique_trials.csv`, else empty. |

## v4 heuristic outputs (added by `final_pass_heuristic_v4.py`)

| field | description |
|---|---|
| `v4_code` | one of `DESCRIBE`, `HYBRID`, `SUBSTITUTE`, `DEFLECT`, `REFUSE` |
| `v4_features` | dict of binary/integer signals: `n_words`, `n_chars`, `has_fiction_flag`, `has_non_recog`, `has_offer`, `has_honesty`, `has_substitute`, `has_speculation`, `has_desc_signal`, `has_naturalist_demo`, `bullets`, `headers`, `blockquote`, `bracketed_placeholder` |

## Sizes

| source run | rows |
|---|---:|
| bestiary_sonnet_pilot | 1,800 |
| bestiary_sonnet_retest | 800 |
| bestiary_gpt_nanomini_retest | 1,600 |
| bestiary_haiku_retest | 800 |
| chess1_sonnet_manifest | 315 |
| chess2_gpt_on_claude_words | 2,400 |
| chess3_gpt_on_gpt_words | 2,100 |
| qq_v1_prerun | 2,160 |
| qq_crossover_completion | 2,070 |
| **total** | **14,045** |

The master-600 hand-coded subset is a stratified sample over these 14,045
rows; it is NOT a separate corpus.
