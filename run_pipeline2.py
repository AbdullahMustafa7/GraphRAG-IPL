"""
Batch runner — Pipeline 2 (Basic RAG)
Runs all 50 evaluation questions, saves full results, and prints a
side-by-side comparison against Pipeline 1 (LLM-Only).

Usage:
    python run_pipeline2.py                    # full 50-question run
    python run_pipeline2.py --build-only       # build FAISS index, then exit
    python run_pipeline2.py --limit 5          # quick smoke-test
    python run_pipeline2.py --delay 6          # slower if rate-limited
    python run_pipeline2.py --type multi       # only multi-hop questions
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import EVALUATION_DIR, RESULTS_DIR
from pipeline2_basic_rag.pipeline import run_batch, build_index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_questions(
    limit: int | None = None,
    qtype: str | None = None,
) -> list[dict]:
    path = EVALUATION_DIR / "questions.json"
    if not path.exists():
        print("[ERR] evaluation/questions.json not found.")
        print("      Run:  python evaluation/generate_questions.py")
        sys.exit(1)

    questions = json.loads(path.read_text(encoding="utf-8"))

    if qtype:
        questions = [q for q in questions if q.get("type", "").startswith(qtype)]
        print(f"Filtered to {len(questions)} '{qtype}-hop' questions.")

    if limit:
        questions = questions[:limit]
        print(f"Limited to first {len(questions)} questions.")

    return questions


def load_existing_results(results_path: Path) -> list[dict]:
    """
    Load any previously saved results from disk.
    Returns an empty list if the file doesn't exist or has no results array.
    Prints a resume summary: how many completed cleanly vs errored.
    """
    if not results_path.exists():
        return []

    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print(f"[WARN] Could not parse {results_path} — starting fresh.")
        return []

    records = data.get("results", [])
    if not records:
        return []

    ok_ids  = sorted(r["id"] for r in records if not r.get("error"))
    err_ids = sorted(r["id"] for r in records if r.get("error"))

    print(f"[RESUME] Found {len(records)} existing records in {results_path.name}")
    print(f"         Completed cleanly : {len(ok_ids)}  "
          f"(IDs {ok_ids[:8]}{'...' if len(ok_ids) > 8 else ''})")
    if err_ids:
        print(f"         Errored (will retry): {len(err_ids)}  "
              f"(IDs {err_ids[:8]}{'...' if len(err_ids) > 8 else ''})")

    return records


def compare_pipelines(p2_summary: dict) -> None:
    """Load Pipeline 1 results and print a comparison table."""
    p1_path = RESULTS_DIR / "pipeline1_results.json"
    if not p1_path.exists():
        print("[INFO] No pipeline1_results.json found — skipping comparison.")
        print("       Run run_pipeline1.py first to enable comparison.\n")
        return

    p1_data    = json.loads(p1_path.read_text(encoding="utf-8"))
    p1_summary = p1_data.get("summary", {})
    if not p1_summary:
        return

    # Pipeline 1 uses "avg_tokens_per_query"; Pipeline 2 uses "avg_total_tokens"
    p1_avg_tok  = p1_summary.get("avg_tokens_per_query", 0)
    p1_cost     = p1_summary.get("total_cost_usd", 0)
    p1_latency  = p1_summary.get("avg_latency_ms", 0)

    p2_avg_tok  = p2_summary.get("avg_total_tokens", 0)
    p2_cost     = p2_summary.get("total_cost_usd", 0)
    p2_latency  = p2_summary.get("avg_latency_ms", 0)

    def delta(new, old, fmt=".1f") -> str:
        if old == 0:
            return "  n/a"
        diff = ((new - old) / old) * 100
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:{fmt}}%"

    print("=" * 62)
    print("  Pipeline comparison: RAG (P2) vs LLM-Only (P1)")
    print("=" * 62)
    print(f"  {'Metric':<26}  {'P1 (LLM-Only)':>12}  {'P2 (RAG)':>12}  {'Delta':>9}")
    print("-" * 62)
    print(f"  {'Avg tokens / query':<26}  {p1_avg_tok:>12,.1f}  {p2_avg_tok:>12,.1f}  {delta(p2_avg_tok, p1_avg_tok):>9}")
    print(f"  {'Avg latency (ms)':<26}  {p1_latency:>12,.1f}  {p2_latency:>12,.1f}  {delta(p2_latency, p1_latency):>9}")
    print(f"  {'Total cost (USD)':<26}  ${p1_cost:>11.6f}  ${p2_cost:>11.6f}  {delta(p2_cost, p1_cost):>9}")
    print("=" * 62)
    print()
    print("  Note: P2 tokens are higher because the RAG prompt includes")
    print("  retrieved context (~5 x 512-token chunks).")
    print("  P2 uses Groq (llama-3.3-70b-versatile); P1 used Gemini Flash.")
    print("  Cost/latency figures are not directly comparable across providers.")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Pipeline 2 (Basic RAG) batch evaluation"
    )
    p.add_argument(
        "--delay", type=float, default=3.0,
        help="Seconds between Groq API requests (default: 3).",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Run only the first N questions (useful for testing).",
    )
    p.add_argument(
        "--type", dest="qtype", choices=["single", "multi"], default=None,
        help="Filter to only single-hop or multi-hop questions.",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Custom output file path (default: results/pipeline2_results.json).",
    )
    p.add_argument(
        "--build-only", action="store_true",
        help="Build (or verify) the FAISS index, then exit without running queries.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Build-only mode ──────────────────────────────────────────────────────
    if args.build_only:
        print("Building FAISS index (--build-only mode) ...")
        build_index()
        print("[DONE] Index ready.")
        return

    # ── Normal batch run ─────────────────────────────────────────────────────
    questions = load_questions(limit=args.limit, qtype=args.qtype)

    results_path = (
        Path(args.output) if args.output
        else RESULTS_DIR / "pipeline2_results.json"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load any results already on disk so we can resume without re-running them
    existing = load_existing_results(results_path)

    print(f"\nLoaded {len(questions)} questions.")
    print(f"Results path : {results_path}")
    print(f"Request delay: {args.delay}s  |  backoff on rate-limit: 30s / 60s / 120s\n")

    summary = run_batch(
        questions=questions,
        delay_between_requests=args.delay,
        results_path=results_path,
        existing_results=existing,
    )

    # ── Compare vs Pipeline 1 ────────────────────────────────────────────────
    compare_pipelines(summary)

    print(f"[DONE] Full results saved to: {results_path}")
    print()
    print("  Next steps:")
    print("  - Inspect results/pipeline2_results.json")
    print("  - Build Pipeline 3 (GraphRAG) for the final comparison")
    print("  - Run evaluation/evaluate.py to score all pipelines\n")


if __name__ == "__main__":
    main()
