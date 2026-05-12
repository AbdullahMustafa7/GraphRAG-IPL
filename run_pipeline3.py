"""
Batch runner -- Pipeline 3 (GraphRAG)
Runs all 50 evaluation questions through the TigerGraph GraphRAG pipeline
and prints a 3-way comparison: P1 (LLM-Only) vs P2 (Basic RAG) vs P3 (GraphRAG).

Usage:
    python run_pipeline3.py                    # full 50-question run
    python run_pipeline3.py --limit 5          # quick smoke-test (5 questions)
    python run_pipeline3.py --delay 7          # slower if rate-limited
    python run_pipeline3.py --type single      # only single-hop questions
    python run_pipeline3.py --type multi       # only multi-hop questions
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import EVALUATION_DIR, RESULTS_DIR
from pipeline3_graphrag.pipeline import run_batch


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
        print("      Run: python evaluation/generate_questions.py")
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
    Load any previously saved Pipeline 3 results from disk for resume logic.
    Only skips records that completed without error.
    """
    if not results_path.exists():
        return []

    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print(f"[WARN] Could not parse {results_path} -- starting fresh.")
        return []

    records = data.get("results", [])
    if not records:
        return []

    ok_ids  = sorted(r["id"] for r in records if not r.get("error"))
    err_ids = sorted(r["id"] for r in records if r.get("error"))
    fb_ids  = sorted(r["id"] for r in records if r.get("fallback") and not r.get("error"))

    print(f"[RESUME] Found {len(records)} existing records in {results_path.name}")
    print(f"         Completed OK    : {len(ok_ids)}  "
          f"(IDs {ok_ids[:8]}{'...' if len(ok_ids) > 8 else ''})")
    if fb_ids:
        print(f"         Fallbacks       : {len(fb_ids)}  "
              f"(IDs {fb_ids[:8]}{'...' if len(fb_ids) > 8 else ''})")
    if err_ids:
        print(f"         Errored (retry) : {len(err_ids)}  "
              f"(IDs {err_ids[:8]}{'...' if len(err_ids) > 8 else ''})")

    return records


def _load_summary(path: Path) -> dict:
    """Safely load the summary block from a results JSON file."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("summary", {})
    except Exception:
        return {}


def compare_pipelines(p3_summary: dict) -> None:
    """Print a 3-way comparison table: P1 vs P2 vs P3."""
    p1 = _load_summary(RESULTS_DIR / "pipeline1_results.json")
    p2 = _load_summary(RESULTS_DIR / "pipeline2_results.json")

    # Normalise token field names (P1 uses avg_tokens_per_query)
    p1_tok  = p1.get("avg_tokens_per_query",  p1.get("avg_total_tokens", 0))
    p2_tok  = p2.get("avg_total_tokens",  p2.get("avg_tokens_per_query", 0))
    p3_tok  = p3_summary.get("avg_total_tokens", 0)

    p1_cost = p1.get("total_cost_usd", 0)
    p2_cost = p2.get("total_cost_usd", 0)
    p3_cost = p3_summary.get("total_cost_usd", 0)

    p1_lat  = p1.get("avg_latency_ms", 0)
    p2_lat  = p2.get("avg_latency_ms", 0)
    p3_lat  = p3_summary.get("avg_latency_ms", 0)

    p3_ents = p3_summary.get("avg_entities_found", 0)
    p3_fbs  = p3_summary.get("fallbacks", 0)

    def pct_delta(new: float, base: float) -> str:
        if base == 0:
            return "   n/a"
        d = (new - base) / base * 100
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1f}%"

    avail = {
        "P1": bool(p1),
        "P2": bool(p2),
        "P3": True,
    }

    print()
    print("=" * 70)
    print("  Pipeline Comparison: P1 (LLM-Only) vs P2 (RAG) vs P3 (GraphRAG)")
    print("=" * 70)
    print(f"  {'Metric':<28}  {'P1':>9}  {'P2':>9}  {'P3':>9}  {'P3 vs P1':>9}  {'P3 vs P2':>9}")
    print("-" * 70)

    # Tokens
    print(
        f"  {'Avg tokens / query':<28}  "
        f"{p1_tok:>9,.1f}  " if avail["P1"] else f"  {'Avg tokens / query':<28}  {'n/a':>9}  "
        , end=""
    )
    print(
        f"{p2_tok:>9,.1f}  " if avail["P2"] else f"{'n/a':>9}  ", end=""
    )
    print(
        f"{p3_tok:>9,.1f}  "
        f"{pct_delta(p3_tok, p1_tok):>9}  "
        f"{pct_delta(p3_tok, p2_tok):>9}"
    )

    # Latency
    p1_lat_s = f"{p1_lat:>9,.1f}" if avail["P1"] else f"{'n/a':>9}"
    p2_lat_s = f"{p2_lat:>9,.1f}" if avail["P2"] else f"{'n/a':>9}"
    print(
        f"  {'Avg latency (ms)':<28}  {p1_lat_s}  {p2_lat_s}  "
        f"{p3_lat:>9,.1f}  "
        f"{pct_delta(p3_lat, p1_lat):>9}  "
        f"{pct_delta(p3_lat, p2_lat):>9}"
    )

    # Cost
    p1_cost_s = f"${p1_cost:>8.6f}" if avail["P1"] else f"{'n/a':>9}"
    p2_cost_s = f"${p2_cost:>8.6f}" if avail["P2"] else f"{'n/a':>9}"
    print(
        f"  {'Total cost (USD)':<28}  {p1_cost_s}  {p2_cost_s}  "
        f"${p3_cost:>8.6f}  "
        f"{pct_delta(p3_cost, p1_cost):>9}  "
        f"{pct_delta(p3_cost, p2_cost):>9}"
    )

    print("-" * 70)
    print(f"  P3 avg graph entities retrieved : {p3_ents:.1f}")
    print(f"  P3 fallbacks (no graph match)   : {p3_fbs} / {p3_summary.get('total_questions', 50)}")
    print("=" * 70)
    print()

    if p3_tok and p2_tok:
        reduction = (1 - p3_tok / p2_tok) * 100
        if reduction > 0:
            print(f"  Graph context reduced prompt tokens by {reduction:.1f}% vs P2 (RAG).")
        else:
            print(f"  P3 tokens are {abs(reduction):.1f}% higher than P2 (RAG).")
        print(f"  P3 graph traversal adds 2 hops of relational context at 0 extra LLM tokens.")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Pipeline 3 (GraphRAG) batch evaluation"
    )
    p.add_argument(
        "--delay", type=float, default=5.0,
        help="Seconds between Groq API requests (default: 5).",
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
        help="Custom output file path (default: results/pipeline3_results.json).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    questions = load_questions(limit=args.limit, qtype=args.qtype)

    results_path = (
        Path(args.output) if args.output
        else RESULTS_DIR / "pipeline3_results.json"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    existing = load_existing_results(results_path)

    print(f"\nLoaded {len(questions)} questions.")
    print(f"Results path : {results_path}")
    print(f"Request delay: {args.delay}s")
    print()

    summary = run_batch(
        questions=questions,
        delay_between_requests=args.delay,
        results_path=results_path,
        existing_results=existing,
    )

    # Print summary
    print()
    print("=" * 50)
    print("  Pipeline 3 (GraphRAG) -- Run Summary")
    print("=" * 50)
    print(f"  Questions completed  : {summary['completed']} / {summary['total_questions']}")
    print(f"  Fallbacks            : {summary['fallbacks']}")
    print(f"  Avg tokens / query   : {summary['avg_total_tokens']:,.1f}")
    print(f"    prompt tokens      : {summary['avg_prompt_tokens']:,.1f}")
    print(f"    completion tokens  : {summary['avg_completion_tokens']:,.1f}")
    print(f"  Avg entities found   : {summary['avg_entities_found']:.1f}")
    print(f"  Avg latency (ms)     : {summary['avg_latency_ms']:,.1f}")
    print(f"  Total cost (USD)     : ${summary['total_cost_usd']:.6f}")
    print(f"  Model                : {summary['model']}")
    print("=" * 50)
    print()

    compare_pipelines(summary)

    print(f"[DONE] Full results saved to: {results_path}")
    print()
    print("  Next steps:")
    print("  - Inspect results/pipeline3_results.json")
    print("  - Run communities.py to add community summaries (improves context quality)")
    print("  - Run evaluation/evaluate.py to score all 3 pipelines\n")


if __name__ == "__main__":
    main()
