"""
heuristic_diagnostics.py — compare v4 against prior heuristics on the master-600.

For each heuristic available (BC v3, QQ v3.1, and our new v4), this script:
  1. Runs the heuristic over the 600 master-coded trials
  2. Compares its output to master-600 adjudicated_code
  3. Reports raw agreement, Cohen's kappa, and per-class precision/recall/F1
  4. Outputs a confusion matrix and a per-trial CSV with all three labels

The baseline question: what κ should v4 be aiming for? BC v3 was published as
κ=0.741 on its own 200-trial validation set, but that set was the BC-only
slice. QQ v3.1 was κ=0.731 on its own 210-trial calibration set. Neither was
tested against the master-600. The master-600 is stratified to oversample
hard cases — the floor for "comparable to prior work" is whatever BC v3 and
QQ v3.1 score on that sample.
"""

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cohens_kappa(hand, heur):
    """Compute Cohen's kappa between two label sequences of equal length."""
    assert len(hand) == len(heur)
    n = len(hand)
    codes = sorted(set(hand) | set(heur))
    agree = sum(1 for h, k in zip(hand, heur) if h == k)
    po = agree / n
    hd = Counter(hand)
    kd = Counter(heur)
    pe = sum((hd[c] / n) * (kd[c] / n) for c in codes)
    if pe >= 1:
        return 0.0, po
    return (po - pe) / (1 - pe), po


def per_class(hand, heur):
    codes = sorted(set(hand) | set(heur))
    out = {}
    for c in codes:
        hn = sum(1 for h in hand if h == c)
        kn = sum(1 for k in heur if k == c)
        tp = sum(1 for h, k in zip(hand, heur) if h == c and k == c)
        prec = tp / kn if kn else 0
        rec = tp / hn if hn else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        out[c] = {
            "hand_n": hn, "heur_n": kn, "tp": tp,
            "precision": prec, "recall": rec, "f1": f1,
        }
    return out


def confusion(hand, heur):
    codes = sorted(set(hand) | set(heur))
    mat = {h: {k: 0 for k in codes} for h in codes}
    for h, k in zip(hand, heur):
        mat[h][k] += 1
    return mat, codes


def print_confusion(mat, codes, title):
    print(f"\n{title}")
    print(f"  {'hand \\ heur':<15} " + " ".join(f"{c:>11}" for c in codes))
    for h in codes:
        row = " ".join(f"{mat[h][k]:>11}" for k in codes)
        print(f"  {h:<15} {row}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-jsonl", required=True,
                    help="Unified corpus JSONL (output of unify_corpus.py)")
    ap.add_argument("--bc-v3", required=True,
                    help="Path to BC heuristic_v3.py")
    ap.add_argument("--qq-v3-1", required=True,
                    help="Path to QQ qq_heuristic_v3_1.py")
    ap.add_argument("--v4", required=True,
                    help="Path to v4 heuristic")
    ap.add_argument("--out-csv", required=True,
                    help="Per-trial CSV with all three labels")
    args = ap.parse_args()

    bc_v3 = load_module(args.bc_v3, "bc_v3")
    qq_v3_1 = load_module(args.qq_v3_1, "qq_v3_1")
    v4 = load_module(args.v4, "v4")

    rows = []
    with open(args.corpus_jsonl, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("prev_adjudicated_code") and \
                    rec.get("prev_adjudicated_code") not in ("", "OTHER_REVIEW"):
                rows.append(rec)

    print(f"Master-600 rows (excluding OTHER_REVIEW): {len(rows)}", file=sys.stderr)

    hand_codes = []
    bc_codes = []
    qq_codes = []
    v4_codes = []

    out_records = []
    for rec in rows:
        text = rec.get("response", "")
        condition = rec.get("condition", "")
        # BC v3 takes (text, condition)
        try:
            bc_label = bc_v3.heuristic_v3(text, condition)
        except Exception as e:
            bc_label = "ERROR"
        # QQ v3.1 takes only text; condition was implicit
        try:
            qq_label, _ = qq_v3_1.classify(text)
        except Exception as e:
            qq_label = "ERROR"
        # v4 takes (text, condition, frame_id)
        v4_label, _ = v4.classify(text, condition=condition,
                                   frame_id=rec.get("frame_id", ""))

        hand = rec["prev_adjudicated_code"]
        hand_codes.append(hand)
        bc_codes.append(bc_label)
        qq_codes.append(qq_label)
        v4_codes.append(v4_label)

        out_records.append({
            "global_trial_id": rec.get("global_trial_id", ""),
            "study": rec.get("study", ""),
            "source_run": rec.get("source_run", ""),
            "model_family": rec.get("model_family", ""),
            "model": rec.get("model", ""),
            "word": rec.get("word", ""),
            "condition": condition,
            "frame_id": rec.get("frame_id", ""),
            "hand_code": hand,
            "bc_v3_code": bc_label,
            "qq_v3_1_code": qq_label,
            "v4_code": v4_label,
        })

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_records[0].keys()))
        w.writeheader()
        w.writerows(out_records)

    print("\n=== AGREEMENT (vs hand codes) ===")
    for name, codes in [("BC v3", bc_codes), ("QQ v3.1", qq_codes), ("v4", v4_codes)]:
        kappa, po = cohens_kappa(hand_codes, codes)
        print(f"  {name:<8}: raw={po*100:5.1f}%  kappa={kappa:.3f}")

    print("\n=== PER-CLASS F1 ===")
    print(f"  {'code':<12} {'BC v3':>15} {'QQ v3.1':>15} {'v4':>15}")
    all_codes = sorted(set(hand_codes) | set(bc_codes) | set(qq_codes) | set(v4_codes))
    bc_pc = per_class(hand_codes, bc_codes)
    qq_pc = per_class(hand_codes, qq_codes)
    v4_pc = per_class(hand_codes, v4_codes)
    for c in all_codes:
        def fmt(d):
            r = d.get(c, {"f1": 0, "precision": 0, "recall": 0})
            return f"F1={r['f1']*100:4.1f} P/R={r['precision']*100:.0f}/{r['recall']*100:.0f}"
        print(f"  {c:<12} {fmt(bc_pc):>15} {fmt(qq_pc):>15} {fmt(v4_pc):>15}")

    print_confusion(*confusion(hand_codes, bc_codes), "=== BC v3 vs hand ===")
    print_confusion(*confusion(hand_codes, qq_codes), "=== QQ v3.1 vs hand ===")
    print_confusion(*confusion(hand_codes, v4_codes), "=== v4 vs hand ===")


if __name__ == "__main__":
    main()
