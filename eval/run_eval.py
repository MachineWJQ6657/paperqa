"""Evaluate the pipeline on QASPER and save the results to eval/results/<name>.json.

    uv run python eval/run_eval.py --name baseline_dense                    # retrieval + Writer + judge
    uv run python eval/run_eval.py --name baseline_dense --no-judge         # official F1 only, no judge calls
    uv run python eval/run_eval.py --name dense_retrieval --retrieval-only  # Recall@k only, no LLM, free
    uv run python eval/run_eval.py --name grader --grader                   # Grader + re-query loop before the Writer
    uv run python eval/run_eval.py --name baseline_api --backend api        # LLM calls billed to the API key

LLM calls go through `claude -p` (Claude subscription) unless --backend api is given.
The run refuses to start with uncommitted code changes, so every result records the
exact commit it came from; --allow-dirty skips that check for debugging.

The question sample depends only on --split/--n/--seed, so runs with the same values
are evaluated on exactly the same questions and can be compared row by row.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from tqdm import tqdm

from paperqa.agents import Grader, Writer, gather_passages
from paperqa.config import CHUNKING_VERSION, JUDGE_MODEL, LLM_BACKEND, LLM_MODEL, RESULTS_DIR
from paperqa.llm import BACKENDS, ClaudeCodeError, make_llm
from paperqa.metrics import (
    LLMJudge,
    evidence_hit,
    evidence_recall,
    max_answer_f1,
    max_evidence_f1,
    mean,
)
from paperqa.qasper import Question, load_split, normalize_text, paper_chunks, paper_questions
from paperqa.retrieve import RETRIEVERS, Hit, make_retriever

RECALL_KS = (1, 3, 5, 10)
# Config errors: fail the whole run instead of recording 100 identical per-question errors.
FATAL_ERRORS = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
    anthropic.BadRequestError,
)


def sample_questions(split: str, n: int, seed: int) -> tuple[list[Question], dict[str, str], dict]:
    """Deterministic sample of answerable questions + {chunk_id: normalized text} for the split."""
    papers = load_split(split)
    questions: list[Question] = []
    chunk_text: dict[str, str] = {}
    for pid, paper in papers.items():
        chunks = paper_chunks(pid, paper)
        chunk_text.update({c.chunk_id: normalize_text(c.text) for c in chunks})
        questions.extend(paper_questions(pid, paper, chunks))

    answerable = sorted((q for q in questions if q.answerable), key=lambda q: q.question_id)
    picked = answerable if n <= 0 else random.Random(seed).sample(answerable, min(n, len(answerable)))
    pool = {
        "split_questions": len(questions),
        "split_unanswerable_skipped": len(questions) - len(answerable),
        "split_answerable": len(answerable),
    }
    return picked, chunk_text, pool


def retrieval_record(q: Question, hits: list[Hit], chunk_text: dict[str, str], latency: float) -> dict:
    texts = [chunk_text[h.chunk_id] for h in hits]
    gold_sets = q.gold_evidence_sets
    gold_union = set().union(*gold_sets) if gold_sets else set()
    rec = {
        "question_id": q.question_id,
        "paper_id": q.paper_id,
        "question": q.question,
        "references": [{"answer": r.answer, "type": r.type, "n_evidence": len(r.evidence)} for r in q.references],
        "has_gold_evidence": bool(gold_sets),
        "retrieved": [
            {"chunk_id": h.chunk_id, "score": round(h.score, 4), "gold": t in gold_union} for h, t in zip(hits, texts)
        ],
        "latency_s": {"retrieval": latency},
        "tokens": {},
        "error": None,
    }
    for k in RECALL_KS:
        rec[f"recall@{k}"] = evidence_recall(texts[:k], gold_sets)
        rec[f"hit@{k}"] = evidence_hit(texts[:k], gold_sets)
    return rec


def grade_and_requery(
    rec: dict, q: Question, hits: list[Hit], grader: Grader, search: Callable[..., list[Hit]], k: int, max_requeries: int
) -> list[Hit]:
    """Run the Grader loop and record what it did. On a Grader failure, keep the first retrieval."""
    info: dict = {"rounds": [], "requeried": False, "error": None}
    rec["grader"] = info
    try:
        g = gather_passages(q.question, hits, lambda query, n: search(query, q.paper_id, n), grader, k, max_requeries)
    except FATAL_ERRORS:
        raise
    except (anthropic.APIError, ClaudeCodeError) as e:
        info["error"] = f"grader: {type(e).__name__}: {e}"
        return hits
    for i, grade in enumerate(g.grades):
        info["rounds"].append(
            {"sufficient": grade.sufficient, "query": grade.query, "added": g.added[i] if i < len(g.added) else []}
        )
    info["requeried"] = any(g.added)
    info["error"] = next((grade.error for grade in g.grades if grade.error), None)
    rec["latency_s"]["grader"] = sum(grade.llm.latency_s for grade in g.grades)
    rec["latency_s"]["requery"] = g.search_latency_s
    rec["tokens"].update(
        grader_in=sum(grade.llm.input_tokens for grade in g.grades),
        grader_out=sum(grade.llm.output_tokens for grade in g.grades),
    )
    return g.hits


def answer_and_score(
    rec: dict,
    q: Question,
    hits: list[Hit],
    chunk_text: dict[str, str],
    writer: Writer,
    judge: LLMJudge | None,
    requery: Callable[[dict, Question, list[Hit]], list[Hit]] | None = None,
) -> dict:
    if requery is not None:
        hits = requery(rec, q, hits)
    # What the Writer actually sees. Without a Grader this is the top-k, so context_recall == recall@k.
    context = [chunk_text[h.chunk_id] for h in hits]
    rec["context"] = [h.chunk_id for h in hits]
    rec["context_recall"] = evidence_recall(context, q.gold_evidence_sets)
    rec["context_hit"] = evidence_hit(context, q.gold_evidence_sets)
    try:
        out = writer.answer(q.question, hits)
        rec["latency_s"]["writer"] = out.llm.latency_s
        rec["tokens"].update(writer_in=out.llm.input_tokens, writer_out=out.llm.output_tokens)
        rec["writer_model"] = out.llm.model
        rec["answer"], rec["cited"], rec["error"] = out.answer, out.cited_chunk_ids, out.error
    except FATAL_ERRORS:
        raise
    except (anthropic.APIError, ClaudeCodeError) as e:  # rate limit / overload / network, after retries
        rec["answer"], rec["cited"], rec["error"] = "", [], f"writer: {type(e).__name__}: {e}"

    # Official QASPER metrics. A failed prediction scores 0, as in the official evaluator.
    rec["answer_f1"], rec["answer_type"] = max_answer_f1(rec["answer"], q.references)
    rec["evidence_f1"] = max_evidence_f1([chunk_text[c] for c in rec["cited"]], q.references)

    if judge is None:
        return rec
    if rec["error"]:
        rec["judge_correct"], rec["judge_reason"] = False, "writer failed"
        return rec
    try:
        j = judge.judge(q.question, q.references, rec["answer"])
        rec["judge_correct"], rec["judge_reason"] = j.correct, j.reason
        rec["tokens"].update(judge_in=j.llm.input_tokens, judge_out=j.llm.output_tokens)
    except FATAL_ERRORS:
        raise
    except (anthropic.APIError, ClaudeCodeError) as e:
        rec["judge_correct"], rec["judge_reason"] = None, f"judge: {type(e).__name__}: {e}"
    return rec


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round(p / 100 * (len(values) - 1)))]


def summarize(records: list[dict], retrieval_only: bool, judged: bool) -> dict:
    with_gold = [r for r in records if r["has_gold_evidence"]]
    summary: dict = {
        "n_questions": len(records),
        "retrieval": {"n_with_gold_evidence": len(with_gold)}
        | {f"recall@{k}": mean([r[f"recall@{k}"] for r in records]) for k in RECALL_KS}
        | {f"hit@{k}": mean([r[f"hit@{k}"] for r in records]) for k in RECALL_KS},
    }
    if not retrieval_only:
        summary["retrieval"] |= {
            "context_recall": mean([r["context_recall"] for r in records]),
            "context_hit": mean([r["context_hit"] for r in records]),
            "context_passages_mean": mean([float(len(r["context"])) for r in records]),
        }
    graded = [r for r in records if "grader" in r]
    if graded:
        summary["grader"] = {
            "requery_rate": mean([float(r["grader"]["requeried"]) for r in graded]),
            "calls_mean": mean([float(len(r["grader"]["rounds"])) for r in graded]),
            "n_errors": sum(r["grader"]["error"] is not None for r in graded),
        }
    lat_retrieval = [r["latency_s"]["retrieval"] for r in records]
    efficiency: dict = {"retrieval_latency_s_mean": mean(lat_retrieval)}

    if not retrieval_only:
        by_type: dict[str, list[float]] = {}
        for r in records:
            by_type.setdefault(r["answer_type"], []).append(r["answer_f1"])
        summary["answer"] = {
            "answer_f1": mean([r["answer_f1"] for r in records]),
            "answer_f1_by_type": {t: {"f1": mean(v), "n": len(v)} for t, v in sorted(by_type.items())},
            "evidence_f1": mean([r["evidence_f1"] for r in records]),
            "judge_accuracy": mean([float(r["judge_correct"]) for r in records if r.get("judge_correct") is not None])
            if judged
            else None,
            "n_judged": sum(r.get("judge_correct") is not None for r in records) if judged else 0,
            "n_unanswerable_predictions": sum(r["answer"].strip().lower().rstrip(".") == "unanswerable" for r in records),
            "n_errors": sum(r["error"] is not None for r in records),
        }
        # End-to-end latency of the pipeline (retrieval + Grader loop + Writer); the judge is eval overhead.
        total = [
            sum(r["latency_s"].get(part, 0.0) for part in ("retrieval", "grader", "requery", "writer")) for r in records
        ]
        efficiency |= {
            "latency_s_mean": mean(total),
            "latency_s_p50": percentile(total, 50),
            "latency_s_p95": percentile(total, 95),
            "writer_input_tokens_mean": mean([r["tokens"].get("writer_in") for r in records]),
            "writer_output_tokens_mean": mean([r["tokens"].get("writer_out") for r in records]),
        }
        if graded:
            efficiency |= {
                "grader_input_tokens_mean": mean([float(r["tokens"].get("grader_in", 0)) for r in records]),
                "grader_output_tokens_mean": mean([float(r["tokens"].get("grader_out", 0)) for r in records]),
            }
        if judged:
            efficiency |= {
                "judge_input_tokens_mean": mean([r["tokens"].get("judge_in") for r in records]),
                "judge_output_tokens_mean": mean([r["tokens"].get("judge_out") for r in records]),
            }
    summary["efficiency"] = efficiency
    return summary


def git_state() -> tuple[str | None, list[str]]:
    """(short commit, changed files). Result files do not count as changes: runs create them."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL)
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", ".", ":(exclude,top)eval/results"], text=True
        )
        return sha.strip(), [line[3:] for line in status.splitlines()]
    except Exception:  # not a git checkout, or git is not installed
        return None, []


def fmt(x: float | None, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def print_summary(name: str, s: dict) -> None:
    r, e = s["retrieval"], s["efficiency"]
    print(f"\n=== {name}: {s['n_questions']} questions ===")
    print(f"Retrieval ({r['n_with_gold_evidence']} with gold evidence)")
    print("  " + "  ".join(f"R@{k}={fmt(r[f'recall@{k}'])}" for k in RECALL_KS))
    print("  " + "  ".join(f"Hit@{k}={fmt(r[f'hit@{k}'])}" for k in RECALL_KS))
    if "answer" in s:
        a = s["answer"]
        print(f"Answer  F1={fmt(a['answer_f1'])}  Evidence F1={fmt(a['evidence_f1'])}  Judge acc={fmt(a['judge_accuracy'])}")
        print("  by type: " + ", ".join(f"{t}={fmt(v['f1'])} (n={v['n']})" for t, v in a["answer_f1_by_type"].items()))
        print(f"  'Unanswerable' predictions: {a['n_unanswerable_predictions']}  errors: {a['n_errors']}")
        print(
            f"Writer context: recall={fmt(r['context_recall'])}  hit={fmt(r['context_hit'])}  "
            f"passages={fmt(r['context_passages_mean'], 1)}"
        )
        if "grader" in s:
            g = s["grader"]
            print(
                f"Grader: re-queried {fmt(g['requery_rate'])} of questions  calls/question={fmt(g['calls_mean'], 2)}  "
                f"errors: {g['n_errors']} | tokens in={fmt(e['grader_input_tokens_mean'], 0)} "
                f"out={fmt(e['grader_output_tokens_mean'], 0)}"
            )
        print(
            f"Latency mean={fmt(e['latency_s_mean'], 2)}s p50={fmt(e['latency_s_p50'], 2)}s "
            f"p95={fmt(e['latency_s_p95'], 2)}s | Writer tokens in={fmt(e['writer_input_tokens_mean'], 0)} "
            f"out={fmt(e['writer_output_tokens_mean'], 0)}"
        )
    else:
        print(f"Retrieval latency mean={fmt(e['retrieval_latency_s_mean'] * 1000, 1)}ms")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="config name, used as the results file name")
    ap.add_argument("--split", default="dev", choices=["dev", "train"])
    ap.add_argument("--n", type=int, default=100, help="number of answerable questions (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=5, help="passages given to the Writer")
    ap.add_argument("--retriever", default="dense", choices=sorted(RETRIEVERS))
    ap.add_argument(
        "--backend",
        default=LLM_BACKEND,
        choices=sorted(BACKENDS),
        help="api = Anthropic API key; claude-code = `claude -p`, billed to the Claude Code login",
    )
    ap.add_argument("--model", default=LLM_MODEL)
    ap.add_argument("--effort", default="medium", help="Writer effort: low|medium|high|xhigh|max|none")
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--judge-effort", default="low")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM-as-judge")
    ap.add_argument("--grader", action="store_true", help="grade the passages and re-query before the Writer")
    ap.add_argument("--grader-model", default=None, help="default: same as --model")
    ap.add_argument("--grader-effort", default="medium")
    ap.add_argument("--max-requeries", type=int, default=1, help="re-query rounds the Grader may trigger")
    ap.add_argument("--retrieval-only", action="store_true", help="only compute Recall@k; no LLM calls")
    ap.add_argument("--workers", type=int, default=4, help="parallel LLM requests")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--allow-dirty", action="store_true", help="run even with uncommitted code changes")
    args = ap.parse_args()

    # Read the commit before the run: files edited while it runs must not change what is recorded.
    sha, changed = git_state()
    if changed and not args.allow_dirty:
        sys.exit(
            "Uncommitted changes in: " + ", ".join(changed[:5]) + (" ..." if len(changed) > 5 else "") + "\n"
            "Commit them first so the result records the code it was produced with, or pass --allow-dirty."
        )
    commit = sha + "-dirty" if sha and changed else sha
    effort = None if args.effort == "none" else args.effort
    judge_effort = None if args.judge_effort == "none" else args.judge_effort
    judged = not (args.no_judge or args.retrieval_only)
    use_grader = args.grader and not args.retrieval_only
    grader_model = args.grader_model or args.model
    grader_effort = None if args.grader_effort == "none" else args.grader_effort

    questions, chunk_text, pool = sample_questions(args.split, args.n, args.seed)
    print(f"{len(questions)} questions sampled ({pool['split_unanswerable_skipped']} unanswerable skipped in split)")

    retriever = make_retriever(args.retriever, args.split)
    n_retrieve = max(args.k, *RECALL_KS)
    retrieved: list[tuple[Question, list[Hit], dict]] = []
    for q in tqdm(questions, desc="retrieve"):
        t0 = time.perf_counter()
        hits = retriever.search(q.question, paper_id=q.paper_id, k=n_retrieve)
        rec = retrieval_record(q, hits, chunk_text, time.perf_counter() - t0)
        retrieved.append((q, hits[: args.k], rec))

    if args.retrieval_only:
        records = [rec for _, _, rec in retrieved]
    else:
        writer = Writer(make_llm(args.backend, args.model, effort))
        judge = LLMJudge(make_llm(args.backend, args.judge_model, judge_effort)) if judged else None
        if args.backend == "claude-code":
            # Fail fast if the CLI is missing, not logged in, or the model name is wrong.
            try:
                writer.llm.complete("Reply with the single word OK.", "ping")
            except ClaudeCodeError as e:
                sys.exit(f"claude -p failed: {e}")

        requery = None
        if use_grader:
            grader = Grader(make_llm(args.backend, grader_model, grader_effort))
            search_lock = threading.Lock()  # the embedder and Chroma are shared by the worker threads

            def search(query: str, paper_id: str, n: int) -> list[Hit]:
                with search_lock:
                    return retriever.search(query, paper_id=paper_id, k=n)

            def requery(rec: dict, q: Question, hits: list[Hit]) -> list[Hit]:
                return grade_and_requery(rec, q, hits, grader, search, args.k, args.max_requeries)

        def run(item: tuple[Question, list[Hit], dict]) -> dict:
            q, hits, rec = item
            return answer_and_score(rec, q, hits, chunk_text, writer, judge, requery)

        try:
            first = run(retrieved[0])  # fail fast on a bad key or model name
            with ThreadPoolExecutor(max_workers=args.workers) as pool_exec:
                rest = list(tqdm(pool_exec.map(run, retrieved[1:]), total=len(retrieved) - 1, desc="answer"))
        except FATAL_ERRORS as e:
            sys.exit(f"LLM request failed: {type(e).__name__}: {e}")
        records = [first, *rest]

    summary = summarize(records, args.retrieval_only, judged)
    result = {
        "name": args.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "config": {
            "split": args.split,
            "n": args.n,
            "seed": args.seed,
            "k": args.k,
            **retriever.describe(),
            "chunking": CHUNKING_VERSION,
            "retrieval_only": args.retrieval_only,
            "backend": None if args.retrieval_only else args.backend,
            "writer_model": None if args.retrieval_only else args.model,
            "writer_effort": None if args.retrieval_only else effort,
            "judge_model": args.judge_model if judged else None,
            "judge_effort": judge_effort if judged else None,
            "grader": use_grader,
            "grader_model": grader_model if use_grader else None,
            "grader_effort": grader_effort if use_grader else None,
            "max_requeries": args.max_requeries if use_grader else None,
            "workers": args.workers,
        }
        | pool,
        "summary": summary,
        "per_question": records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print_summary(args.name, summary)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
