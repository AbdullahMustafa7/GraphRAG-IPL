"""
Batch runner — Pipeline 1 (LLM-Only)
Runs all 50 evaluation questions through Gemini and saves results + summary.

Usage:
    python run_pipeline1.py
    python run_pipeline1.py --delay 5       # custom seconds between requests
    python run_pipeline1.py --limit 10      # run only first N questions
    python run_pipeline1.py --type multi    # only run multi-hop questions
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import EVALUATION_DIR, RESULTS_DIR
from pipeline1_llm_only.pipeline import run_batch


def load_questions(limit: int | None = None, qtype: str | None = None) -> list[dict]:
    questions_path = EVALUATION_DIR / "questions.json"
    if not questions_path.exists():
        print("✗ evaluation/questions.json not found.")
        print("  Run this first:  python evaluation/generate_questions.py")
        sys.exit(1)

    questions = json.loads(questions_path.read_text(encoding="utf-8"))

    if qtype:
        questions = [q for q in questions if q.get("type", "").startswith(qtype)]
        print(f"Filtered to {len(questions)} '{qtype}' questions.")

    if limit:
        questions = questions[:limit]
        print(f"Limited to first {len(questions)} questions.")

    return questions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Pipeline 1 (LLM-Only) batch evaluation")
    p.add_argument(
        "--delay",
        type=float,
        default=4.0,
        help="Seconds to wait between API requests (default: 4). "
             "Gemini free tier allows 15 RPM.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N questions (useful for testing).",
    )
    p.add_argument(
        "--type",
        dest="qtype",
        choices=["single", "multi"],
        default=None,
        help="Filter to only single-hop or multi-hop questions.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output file path (default: results/pipeline1_results.json).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    questions = load_questions(limit=args.limit, qtype=args.qtype)

    results_path = (
        Path(args.output) if args.output else RESULTS_DIR / "pipeline1_results.json"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nLoaded {len(questions)} questions.")
    print(f"Results will be saved to: {results_path}")
    print(f"Request delay: {args.delay}s  (adjust with --delay if you hit rate limits)\n")

    run_batch(
        questions=questions,
        delay_between_requests=args.delay,
        results_path=results_path,
    )

    print(f"\n✓ Full results saved to: {results_path}")
    print("  Next steps:")
    print("  - Inspect results/pipeline1_results.json")
    print("  - Run pipeline2 and pipeline3 for comparison")
    print("  - Run evaluation/evaluate.py to score answers\n")


if __name__ == "__main__":
    main()
