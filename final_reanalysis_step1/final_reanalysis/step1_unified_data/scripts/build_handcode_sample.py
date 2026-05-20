"""
build_handcode_sample.py — pick a stratified sample for hand-coding.

The aim: pick 300–500 trials that, between them, do the most work in
calibrating v4. The sample is stratified along three axes:

  1. v4_code disagreement candidates — trials where v4's call is on a
     boundary (the score features are mixed). These are the ones where
     a hand label most updates v4.

  2. Study-balanced — every study, every source run, every model family
     gets representation.

  3. Master-600 overlap — trials already in master-600 are excluded
     by default, since hand-coding them again is redundant (we already
     have one careful pass over them).

The output is a JSONL plus the HTML hand-coder loads from it directly.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


# Categories of trial to oversample, roughly equally:
SAMPLE_BLOCKS = {
    # AB-style trials where v4 is HYBRID under imaginary_* — these test the
    # prompt-echo distinction
    "AB_imaginary_hybrid":
        lambda r: r["study"] == "Bestiary"
                  and (r.get("condition") or "").startswith("imaginary_")
                  and r["v4_code"] == "HYBRID",
    # AB-style trials where v4 is DESCRIBE under real_* (hard confab candidates)
    "AB_real_describe":
        lambda r: r["study"] == "Bestiary"
                  and (r.get("condition") or "").startswith("real_")
                  and r["v4_code"] in ("DESCRIBE", "HYBRID"),
    # BC trials where v4 SUBSTITUTE — heuristic catches some but undercounts
    "BC_substitute_candidates":
        lambda r: r["study"] == "Bestiary Chess"
                  and r["v4_code"] == "SUBSTITUTE",
    # BC trials where v4 DESCRIBE (potential bare-substitute false-negatives)
    "BC_describe_candidates":
        lambda r: r["study"] == "Bestiary Chess"
                  and r["v4_code"] == "DESCRIBE",
    # QQ F4 commits — the naturalist-loophole effect, central to QQ
    "QQ_F4_commits":
        lambda r: r["study"] == "Question Question"
                  and r.get("frame_id") == "F4"
                  and r["v4_code"] in ("DESCRIBE", "HYBRID"),
    # QQ F1 vs F4 comparison: F1 DEFLECTs (the baseline)
    "QQ_F1_deflect":
        lambda r: r["study"] == "Question Question"
                  and r.get("frame_id") == "F1"
                  and r["v4_code"] == "DEFLECT",
    # Cross-architecture floor: Sonnet under real_* (paper says zero confab)
    "Sonnet_real_any":
        lambda r: r.get("model") == "claude-sonnet-4-6"
                  and (r.get("condition") or "").startswith("real_"),
    # Haiku trolnique etc. — known confabulation magnets
    "Haiku_trolnique_real":
        lambda r: r.get("model_tier") == "haiku"
                  and r.get("word") in ("trolnique", "kovashent", "plindorf",
                                         "purtaneolotomous")
                  and (r.get("condition") or "").startswith("real_"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coded-jsonl", required=True,
                    help="v4-coded unified corpus JSONL")
    ap.add_argument("--n-per-block", type=int, default=40,
                    help="Target sample size per block (will be capped by"
                         " available rows)")
    ap.add_argument("--exclude-master-600", action="store_true",
                    help="Skip trials already in master-600")
    ap.add_argument("--seed", type=int, default=20260520)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Load all rows
    rows = []
    with open(args.coded_jsonl, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    # Bucket each row into the blocks it qualifies for
    blocks = defaultdict(list)
    for r in rows:
        if args.exclude_master_600 and r.get("prev_in_master_600") == "1":
            continue
        for block_name, pred in SAMPLE_BLOCKS.items():
            try:
                if pred(r):
                    blocks[block_name].append(r)
            except Exception:
                continue

    # Sample evenly from each block
    selected = {}
    block_meta = {}
    for block_name, available in blocks.items():
        n = min(args.n_per_block, len(available))
        chosen = rng.sample(available, n) if n > 0 else []
        for r in chosen:
            # Don't double-count if a row falls in multiple blocks; keep its
            # first block but allow it to count.
            if r["global_trial_id"] not in selected:
                selected[r["global_trial_id"]] = (block_name, r)
        block_meta[block_name] = {"available": len(available), "sampled": n}

    # Write outputs
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for i, (gid, (block_name, r)) in enumerate(sorted(selected.items())):
        # Construct the hand-coder row.
        out = {
            "queue_order": i,
            "sample_block": block_name,
            "global_trial_id": gid,
            "study": r.get("study", ""),
            "source_run": r.get("source_run", ""),
            "source_row": r.get("source_row", ""),
            "word": r.get("word", ""),
            "word_author": r.get("word_author", ""),
            "model": r.get("model", ""),
            "model_family": r.get("model_family", ""),
            "model_tier": r.get("model_tier", ""),
            "condition": r.get("condition", ""),
            "frame_id": r.get("frame_id", ""),
            "frame_name": r.get("frame_name", ""),
            "prompt": r.get("prompt", ""),
            "response": r.get("response", ""),
            "v4_code": r.get("v4_code", ""),
            "v4_features": r.get("v4_features", {}),
            # The hand_code field is what the user fills in.
            "hand_code": "",
            "hand_note": "",
        }
        out_rows.append(out)

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Also a CSV for spreadsheet inspection (response truncated)
    import csv
    cols = ["queue_order", "sample_block", "global_trial_id", "study",
            "source_run", "word", "word_author", "model", "model_family",
            "model_tier", "condition", "frame_id", "prompt",
            "response", "v4_code", "hand_code", "hand_note"]
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            row = {k: r.get(k, "") for k in cols}
            resp = row.get("response", "") or ""
            if len(resp) > 1200:
                row["response"] = resp[:1200] + "...[TRUNCATED]"
            w.writerow(row)

    print(f"\nSample blocks (block: available / sampled):")
    for b, m in block_meta.items():
        print(f"  {b}: {m['available']} / {m['sampled']}")
    print(f"\nTotal unique rows sampled: {len(out_rows)}")
    print(f"Wrote: {args.out_jsonl}")
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
