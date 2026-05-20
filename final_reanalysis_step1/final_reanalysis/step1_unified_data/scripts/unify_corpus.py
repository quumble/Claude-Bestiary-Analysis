"""
unify_corpus.py — combine all nine source runs into one canonical JSONL.

The three-paper line — Artificial Bestiary, Bestiary Chess, Question Question —
spans nine distinct data-generation runs, written under five different schemas.
This script normalizes them into one row shape so that a single heuristic and
a single hand-coding tool can be applied across the whole corpus.

Inputs (relative to --repo-root, which should contain the three unzipped repos):

    the-artificial-bestiary-main/artificial_bestiary_1/Results/results.jsonl
        bestiary_sonnet_pilot                                       1800 trials
    the-artificial-bestiary-main/Retest_nonanimal_nonsense/Results/retest.jsonl
        bestiary_sonnet_retest                                       800 trials
    the-artificial-bestiary-main/GPT_Retest_1600_nanomini/gpt_full_800_nano_mini_results.csv
        bestiary_gpt_nanomini_retest                                1600 trials
    the-artificial-bestiary-main/Haiku v3 Retest/Results/haiku_results.jsonl
        bestiary_haiku_retest                                        800 trials
    Bestiary-Chess-main/Bestiary_Chess_1/results.jsonl
        chess1_sonnet_manifest                                       315 trials
    Bestiary-Chess-main/Bestiary_Chess_2/Results/gpt54_results.jsonl
        chess2_gpt_on_claude_words                                  2400 trials
    Bestiary-Chess-main/Bestiary_Chess_3/Results/gpt54_on_gpt_words_results.jsonl
        chess3_gpt_on_gpt_words                                     2100 trials
    The-Question-Question-main/Question_question_prerun/results/qq_v1_results.jsonl
        qq_v1_prerun                                                2160 trials
    The-Question-Question-main/Naturalist_Crossover_Test/results/qq_crossover_missing_no_flurbenheim.jsonl
        qq_crossover_completion                                     2070 trials

Total expected: 14,045 rows.

The unified schema (one row per trial) is documented in SCHEMA.md and is
designed to be a superset of QQ's row shape — that schema already carried
the most metadata, so other sources are projected into it with empty fields
where they don't supply a value.

Output: data/unified_corpus.jsonl   (one JSON object per line)
        data/unified_corpus.csv     (same data, but the response field is
                                     truncated to 1200 chars for spreadsheet
                                     sanity — full text is in the JSONL)
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import re
from pathlib import Path

# ---------- Canonical schema ----------

CANONICAL_FIELDS = [
    "global_trial_id",        # stable hash, deterministic across re-runs
    "study",                  # "Bestiary" | "Bestiary Chess" | "Question Question"
    "source_run",             # one of the 9 source_run labels
    "source_file",            # relative path
    "source_row",             # 0-indexed row number within source
    "trial_id_original",      # whatever ID the source carried (may be empty)
    # Stimulus
    "word",
    "word_author",            # "claude" | "gpt" | "" (empty for AB pilot — Chesterton-generated)
    "word_meta",              # any extra annotation the source carried
    "word_set",               # "original" | "analyst" | "" (AB-style) or BC's word_set names
    # Framing
    "status",                 # "real" | "imaginary" | "neutral" — AB-style reality status
    "reality",                # same idea; QQ uses this differently — kept distinct
    "category",               # "animal" | "object" | "idea" | ""
    "ontology",               # synonym BC used; kept for trace
    "condition",              # AB/BC: "real_animal" etc. QQ: "real_animal" too, but frame_id is the discriminator
    "frame_id",               # QQ: F1..F5. Empty for AB/BC.
    "frame_name",             # QQ: "directive" | "question" | "statement" | "3rd_person" | "1st_person"
    "speech_act",             # QQ
    "person",                 # QQ
    "prompt",                 # exact prompt text
    # Trial bookkeeping
    "rep_n",                  # replicate number within cell
    "trial_n",                # alt name some sources used
    # Model
    "model",                  # exact model string
    "model_family",           # "GPT" | "Claude"
    "model_tier",             # "nano" | "mini" | "main" | "haiku" | "sonnet" | "opus"
    # Response
    "response",               # full response text
    # Pre-existing labels (carried for trace, NOT used as ground truth)
    "prev_pass1_code",        # from master-600 if available
    "prev_adjudicated_code",  # from master-600 if available
    "prev_in_master_600",     # bool
]


# ---------- Helpers ----------

def make_global_trial_id(source_run, source_row, word, model, condition_or_frame, rep):
    """Stable hash. Deterministic across re-runs.
    Note: source_row is 1-indexed to match the convention used in the
    master-600 hand-coded sample, so master-600 codes can be joined back."""
    raw = f"{source_run}::{source_row}::{word}::{model}::{condition_or_frame}::{rep}"
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{source_run}::{source_row:05d}::{h}"


def family_and_tier(model_str):
    """Map a model string to (family, tier)."""
    if not model_str:
        return ("", "")
    m = model_str.lower()
    if "gpt" in m:
        family = "GPT"
        if "nano" in m:
            tier = "nano"
        elif "mini" in m:
            tier = "mini"
        else:
            tier = "main"
        return (family, tier)
    if "claude" in m:
        family = "Claude"
        if "haiku" in m:
            tier = "haiku"
        elif "opus" in m:
            tier = "opus"
        elif "sonnet" in m:
            tier = "sonnet"
        else:
            tier = ""
        return (family, tier)
    return ("", "")


def split_condition(cond):
    """Pull reality + category out of an AB/BC-style condition string."""
    if not cond:
        return ("", "", "")
    parts = cond.split("_", 1)
    if len(parts) == 2:
        reality, category = parts
        status = reality  # AB style: status == reality
        return (status, reality, category)
    if cond == "neutral":
        return ("neutral", "neutral", "")
    return ("", "", "")


def _empty_row():
    return {k: "" for k in CANONICAL_FIELDS}


# ---------- Per-source loaders ----------

def load_bestiary_sonnet_pilot(path, source_run):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            cond = rec.get("condition", "")
            status, reality, category = split_condition(cond)
            model = rec.get("model", "claude-sonnet-4-6")
            family, tier = family_and_tier(model)
            row = _empty_row()
            row.update({
                "study": "Bestiary",
                "source_run": source_run,
                "source_row": i + 1,
                "trial_id_original": rec.get("trial_id", ""),
                "word": rec.get("word", ""),
                "word_author": "",       # Chesterton-generated, not author-tagged
                "word_set": "original",  # AB pilot used 9 Chesterton words
                "status": status,
                "reality": rec.get("reality", reality),
                "category": rec.get("category", category),
                "ontology": rec.get("category", category),
                "condition": cond,
                "prompt": rec.get("prompt", ""),
                "rep_n": rec.get("trial_n", ""),
                "trial_n": rec.get("trial_n", ""),
                "model": model,
                "model_family": family,
                "model_tier": tier,
                "response": rec.get("response", ""),
            })
            row["global_trial_id"] = make_global_trial_id(
                source_run, i + 1, row["word"], row["model"], cond, row["rep_n"])
            yield row


def load_bestiary_sonnet_retest(path, source_run):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            cond = rec.get("condition", "")
            status, reality, category = split_condition(cond)
            model = rec.get("model", "claude-sonnet-4-6")
            family, tier = family_and_tier(model)
            row = _empty_row()
            ws_raw = rec.get("word_set", "")
            # In the retest, word_set is "original" or "analyst"
            row.update({
                "study": "Bestiary",
                "source_run": source_run,
                "source_row": i + 1,
                "trial_id_original": rec.get("trial_id", ""),
                "word": rec.get("word", ""),
                "word_author": "",  # original = Chesterton, analyst = Claude analyst
                "word_set": ws_raw,
                "status": status,
                "reality": rec.get("reality", reality),
                "category": rec.get("category", category),
                "ontology": rec.get("category", category),
                "condition": cond,
                "prompt": rec.get("prompt", ""),
                "rep_n": rec.get("trial_n", ""),
                "trial_n": rec.get("trial_n", ""),
                "model": model,
                "model_family": family,
                "model_tier": tier,
                "response": rec.get("response", ""),
            })
            row["global_trial_id"] = make_global_trial_id(
                source_run, i + 1, row["word"], row["model"], cond, row["rep_n"])
            yield row


def load_bestiary_gpt_nanomini_csv(path, source_run):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, rec in enumerate(reader):
            cond = rec.get("condition", "")
            status, reality, category = split_condition(cond)
            model = rec.get("model", "")
            family, tier = family_and_tier(model)
            row = _empty_row()
            row.update({
                "study": "Bestiary",
                "source_run": source_run,
                "source_row": i + 1,
                "trial_id_original": rec.get("trial_id", ""),
                "word": rec.get("word", ""),
                "word_author": "",
                "word_set": "",
                "status": status,
                "reality": rec.get("reality", reality),
                "category": rec.get("category", category),
                "ontology": rec.get("category", category),
                "condition": cond,
                "prompt": rec.get("prompt", ""),
                "rep_n": rec.get("trial_n", ""),
                "trial_n": rec.get("trial_n", ""),
                "model": model,
                "model_family": family,
                "model_tier": tier,
                "response": rec.get("output_text", ""),  # CSV uses 'output_text'
            })
            row["global_trial_id"] = make_global_trial_id(
                source_run, i + 1, row["word"], row["model"], cond, row["rep_n"])
            yield row


def load_bestiary_haiku_retest(path, source_run):
    # Same shape as bestiary_sonnet_retest
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            cond = rec.get("condition", "")
            status, reality, category = split_condition(cond)
            model = rec.get("model", "claude-haiku-4-5")
            family, tier = family_and_tier(model)
            row = _empty_row()
            row.update({
                "study": "Bestiary",
                "source_run": source_run,
                "source_row": i + 1,
                "trial_id_original": rec.get("trial_id", ""),
                "word": rec.get("word", ""),
                "word_author": "",
                "word_set": rec.get("word_set", ""),
                "status": status,
                "reality": rec.get("reality", reality),
                "category": rec.get("category", category),
                "ontology": rec.get("category", category),
                "condition": cond,
                "prompt": rec.get("prompt", ""),
                "rep_n": rec.get("trial_n", ""),
                "trial_n": rec.get("trial_n", ""),
                "model": model,
                "model_family": family,
                "model_tier": tier,
                "response": rec.get("response", ""),
            })
            row["global_trial_id"] = make_global_trial_id(
                source_run, i + 1, row["word"], row["model"], cond, row["rep_n"])
            yield row


def load_chess1(path, source_run):
    # Uses 'response_text' and 'replicate' and 'ontology'/'status'
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            cond = rec.get("condition", "")
            # chess1 puts reality status and ontology in separate fields
            status = rec.get("status", "")
            ontology = rec.get("ontology", "")
            if not cond and status and ontology:
                cond = f"{status}_{ontology}"
            model = rec.get("model", "")
            family, tier = family_and_tier(model)
            row = _empty_row()
            row.update({
                "study": "Bestiary Chess",
                "source_run": source_run,
                "source_row": i + 1,
                "trial_id_original": rec.get("trial_id", ""),
                "word": rec.get("word", ""),
                "word_author": "",  # mixed; chess1 word_set tells us indirectly
                "word_set": rec.get("word_set", ""),
                "status": status,
                "reality": status,
                "category": ontology,
                "ontology": ontology,
                "condition": cond,
                "prompt": rec.get("prompt", ""),
                "rep_n": rec.get("replicate", ""),
                "trial_n": rec.get("replicate", ""),
                "model": model,
                "model_family": family,
                "model_tier": tier,
                "response": rec.get("response_text", ""),
            })
            row["global_trial_id"] = make_global_trial_id(
                source_run, i + 1, row["word"], row["model"], cond, row["rep_n"])
            yield row


def load_chess2(path, source_run):
    # GPT-family on Claude-authored words
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            cond = rec.get("condition", "")
            status, reality, category = split_condition(cond)
            model = rec.get("model", "")
            family, tier = family_and_tier(model)
            row = _empty_row()
            row.update({
                "study": "Bestiary Chess",
                "source_run": source_run,
                "source_row": i + 1,
                "trial_id_original": rec.get("trial_id", ""),
                "word": rec.get("word", ""),
                "word_author": "claude",  # by construction: chess2 = GPT on Claude-authored
                "word_set": "claude_authored",
                "status": status,
                "reality": rec.get("reality", reality),
                "category": rec.get("category", category),
                "ontology": rec.get("category", category),
                "condition": cond,
                "prompt": rec.get("prompt", ""),
                "rep_n": rec.get("trial_n", ""),
                "trial_n": rec.get("trial_n", ""),
                "model": model,
                "model_family": family,
                "model_tier": tier,
                "response": rec.get("response", ""),
            })
            row["global_trial_id"] = make_global_trial_id(
                source_run, i + 1, row["word"], row["model"], cond, row["rep_n"])
            yield row


def load_chess3(path, source_run):
    # GPT-family on GPT-authored words
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            cond = rec.get("condition", "")
            status, reality, category = split_condition(cond)
            model = rec.get("model", "")
            family, tier = family_and_tier(model)
            row = _empty_row()
            row.update({
                "study": "Bestiary Chess",
                "source_run": source_run,
                "source_row": i + 1,
                "trial_id_original": rec.get("trial_id", ""),
                "word": rec.get("word", ""),
                "word_author": "gpt",
                "word_set": "gpt_authored",
                "status": status,
                "reality": rec.get("reality", reality),
                "category": rec.get("category", category),
                "ontology": rec.get("category", category),
                "condition": cond,
                "prompt": rec.get("prompt", ""),
                "rep_n": rec.get("trial_n", ""),
                "trial_n": rec.get("trial_n", ""),
                "model": model,
                "model_family": family,
                "model_tier": tier,
                "response": rec.get("response", ""),
            })
            row["global_trial_id"] = make_global_trial_id(
                source_run, i + 1, row["word"], row["model"], cond, row["rep_n"])
            yield row


def load_qq(path, source_run):
    # QQ schema is the richest — closest to canonical
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            model = rec.get("model", "")
            family, tier = family_and_tier(model)
            cond = rec.get("condition", "")
            row = _empty_row()
            row.update({
                "study": "Question Question",
                "source_run": source_run,
                "source_row": i + 1,
                "trial_id_original": rec.get("trial_id", ""),
                "word": rec.get("word", ""),
                "word_author": rec.get("word_author", ""),
                "word_meta": rec.get("word_meta", ""),
                "status": rec.get("status", ""),
                "reality": rec.get("status", ""),     # QQ holds reality in 'status' (all "real")
                "category": rec.get("category", ""),
                "ontology": rec.get("category", ""),
                "condition": cond,
                "frame_id": rec.get("frame_id", ""),
                "frame_name": rec.get("frame_name", ""),
                "speech_act": rec.get("speech_act", ""),
                "person": rec.get("person", ""),
                "prompt": rec.get("prompt", ""),
                "rep_n": rec.get("rep_n", ""),
                "trial_n": rec.get("rep_n", ""),
                "model": model,
                "model_family": rec.get("model_family", family),
                "model_tier": rec.get("model_tier", tier),
                "response": rec.get("response", ""),
            })
            # Build a frame-aware key so the global ID disambiguates F1..F5 trials
            cond_key = f"{cond}::{rec.get('frame_id','')}"
            row["global_trial_id"] = make_global_trial_id(
                source_run, i + 1, row["word"], row["model"], cond_key, row["rep_n"])
            yield row


SOURCE_LOADERS = [
    ("bestiary_sonnet_pilot",
     "the-artificial-bestiary-main/artificial_bestiary_1/Results/results.jsonl",
     load_bestiary_sonnet_pilot, 1800),
    ("bestiary_sonnet_retest",
     "the-artificial-bestiary-main/Retest_nonanimal_nonsense/Results/retest.jsonl",
     load_bestiary_sonnet_retest, 800),
    ("bestiary_gpt_nanomini_retest",
     "the-artificial-bestiary-main/GPT_Retest_1600_nanomini/gpt_full_800_nano_mini_results.csv",
     load_bestiary_gpt_nanomini_csv, 1600),
    ("bestiary_haiku_retest",
     "the-artificial-bestiary-main/Haiku v3 Retest/Results/haiku_results.jsonl",
     load_bestiary_haiku_retest, 800),
    ("chess1_sonnet_manifest",
     "Bestiary-Chess-main/Bestiary_Chess_1/results.jsonl",
     load_chess1, 315),
    ("chess2_gpt_on_claude_words",
     "Bestiary-Chess-main/Bestiary_Chess_2/Results/gpt54_results.jsonl",
     load_chess2, 2400),
    ("chess3_gpt_on_gpt_words",
     "Bestiary-Chess-main/Bestiary_Chess_3/Results/gpt54_on_gpt_words_results.jsonl",
     load_chess3, 2100),
    ("qq_v1_prerun",
     "The-Question-Question-main/Question_question_prerun/results/qq_v1_results.jsonl",
     load_qq, 2160),
    ("qq_crossover_completion",
     "The-Question-Question-main/Naturalist_Crossover_Test/results/qq_crossover_missing_no_flurbenheim.jsonl",
     load_qq, 2070),
]


def load_master600_lookup(repo_root):
    """Return {(source_run, source_row): {pass1_code, adjudicated_code}}."""
    p = repo_root / "The-Question-Question-main" / "Master_set_handcoding" / \
        "harmonized_handcoding_final_600_unique_trials.csv"
    lookup = {}
    if not p.exists():
        return lookup
    with open(p, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            try:
                src_row = int(rec.get("source_row", -1))
            except (ValueError, TypeError):
                continue
            key = (rec.get("source_run", ""), src_row)
            lookup[key] = {
                "pass1_code": rec.get("pass1_code", ""),
                "adjudicated_code": rec.get("adjudicated_code", ""),
            }
    return lookup


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True,
                    help="Directory containing the three unzipped repos")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    master600 = load_master600_lookup(repo_root)

    out_jsonl = out_dir / "unified_corpus.jsonl"
    out_csv = out_dir / "unified_corpus.csv"
    manifest_path = out_dir / "unified_manifest.csv"

    total = 0
    per_source = {}
    manifest_rows = []

    with open(out_jsonl, "w", encoding="utf-8") as fj, \
         open(out_csv, "w", encoding="utf-8", newline="") as fc:
        writer = csv.DictWriter(fc, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()

        for source_run, relpath, loader, expected in SOURCE_LOADERS:
            path = repo_root / relpath
            count = 0
            if not path.exists():
                print(f"  [MISSING] {source_run}: {path}", file=sys.stderr)
                manifest_rows.append({
                    "source_run": source_run, "path": relpath,
                    "expected": expected, "loaded": 0, "status": "MISSING",
                })
                continue
            for row in loader(path, source_run):
                key = (source_run, row["source_row"])
                if key in master600:
                    row["prev_pass1_code"] = master600[key]["pass1_code"]
                    row["prev_adjudicated_code"] = master600[key]["adjudicated_code"]
                    row["prev_in_master_600"] = "1"
                row["source_file"] = relpath
                # CSV row: shorten response for spreadsheet sanity
                csv_row = dict(row)
                resp = csv_row.get("response", "") or ""
                if len(resp) > 1200:
                    csv_row["response"] = resp[:1200] + "...[TRUNCATED]"
                writer.writerow(csv_row)
                fj.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
            per_source[source_run] = count
            total += count
            status = "OK" if count == expected else f"MISMATCH (expected {expected})"
            manifest_rows.append({
                "source_run": source_run, "path": relpath,
                "expected": expected, "loaded": count, "status": status,
            })
            print(f"  {source_run}: {count} rows ({status})", file=sys.stderr)

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source_run", "path", "expected", "loaded", "status"])
        w.writeheader()
        for r in manifest_rows:
            w.writerow(r)

    print(f"\nTotal rows unified: {total}", file=sys.stderr)
    print(f"Wrote: {out_jsonl}", file=sys.stderr)
    print(f"Wrote: {out_csv}", file=sys.stderr)
    print(f"Wrote: {manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
