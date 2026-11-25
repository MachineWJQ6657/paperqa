"""Print all saved evaluation results as one Markdown table (the ablation table).

    uv run python eval/compare.py
    uv run python eval/compare.py eval/results/baseline_dense.json eval/results/hybrid.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def fmt(x, digits=3):
    return "-" if x is None else f"{x:.{digits}f}"


def row(res: dict) -> list[str]:
    s, c = res["summary"], res["config"]
    r, e, a = s["retrieval"], s["efficiency"], s.get("answer", {})
    sample = f"{c['split']}/n={c['n']}/seed={c['seed']}"
    return [
        res["name"],
        sample,
        str(s["n_questions"]),
        fmt(r["recall@5"]),
        fmt(r["hit@5"]),
        fmt(a.get("answer_f1")),
        fmt(a.get("evidence_f1")),
        fmt(a.get("judge_accuracy")),
        fmt(e.get("latency_s_mean"), 2),
        fmt(e.get("writer_input_tokens_mean"), 0) + " / " + fmt(e.get("writer_output_tokens_mean"), 0),
    ]


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or sorted(RESULTS_DIR.glob("*.json"))
    results = sorted((json.loads(p.read_text()) for p in paths), key=lambda r: r["created_at"])
    header = ["config", "sample", "n", "Recall@5", "Hit@5", "Answer F1", "Evidence F1", "Judge acc", "Latency (s)", "Tokens in / out"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for res in results:
        print("| " + " | ".join(row(res)) + " |")
    if len({(r["config"]["split"], r["config"]["n"], r["config"]["seed"]) for r in results}) > 1:
        print("\nWarning: these runs use different question samples; numbers are not directly comparable.")


if __name__ == "__main__":
    main()
