"""Is config B really better than config A? Paired bootstrap over the questions both runs share.

    uv run python eval/paired_diff.py eval/results/A.json eval/results/B.json

Prints mean(B - A) with a 95% confidence interval for every metric both runs have.
If the interval contains 0, the difference may be just sampling noise.
"""

from __future__ import annotations

import json
import random
import sys

METRICS = [
    "recall@1", "recall@5", "recall@10", "hit@5", "context_recall", "context_hit",
    "answer_f1", "evidence_f1", "judge_correct",
]  # fmt: skip


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = (json.load(open(p)) for p in sys.argv[1:])
    ra = {r["question_id"]: r for r in a["per_question"]}
    rb = {r["question_id"]: r for r in b["per_question"]}
    shared = sorted(ra.keys() & rb.keys())
    print(f"A = {a['name']}, B = {b['name']}, {len(shared)} shared questions")
    if len(shared) < max(len(ra), len(rb)):
        print("Warning: the runs do not cover the same questions; comparing only the shared ones.")

    rng = random.Random(0)
    for m in METRICS:
        diffs = [
            float(rb[q][m]) - float(ra[q][m]) for q in shared if ra[q].get(m) is not None and rb[q].get(m) is not None
        ]
        if not diffs:
            continue
        mean = sum(diffs) / len(diffs)
        boots = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(5000))
        lo, hi = boots[124], boots[4874]
        verdict = "B better" if lo > 0 else "A better" if hi < 0 else "no clear difference"
        print(f"  {m:<14} B - A = {mean:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   n={len(diffs)}   {verdict}")


if __name__ == "__main__":
    main()
