"""
build_handcode_tool.py — inject the JSONL sample into the HTML template
so the hand-coder is a single self-contained file (no fetch / CORS issues).
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True,
                    help="HTML template with __SAMPLE_DATA_PLACEHOLDER__")
    ap.add_argument("--sample-jsonl", required=True,
                    help="Sample JSONL to embed")
    ap.add_argument("--out", required=True,
                    help="Output HTML path")
    args = ap.parse_args()

    rows = []
    with open(args.sample_jsonl, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    with open(args.template, encoding="utf-8") as f:
        html = f.read()

    embedded = json.dumps(rows, ensure_ascii=False)
    # Defensive: replace common script-breaking patterns
    embedded = embedded.replace("</script>", "<\\/script>")
    html_out = html.replace("__SAMPLE_DATA_PLACEHOLDER__", embedded)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html_out)

    print(f"Embedded {len(rows)} rows into {args.out}")
    print(f"File size: {Path(args.out).stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
