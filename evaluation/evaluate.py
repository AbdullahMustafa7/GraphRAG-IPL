"""
evaluation/evaluate.py — LLM-as-Judge evaluation
Scores Pipeline 1 and Pipeline 2 answers against ground-truth using Groq.

Judge model : llama-3.3-70b-versatile
Verdict     : PASS / FAIL per question
Output      : results/evaluation_report.json + summary table to stdout

Usage
-----
  python evaluation/evaluate.py
  python evaluation/evaluate.py --delay 2     # seconds between judge calls (default: 2)
  python evaluation/evaluate.py --limit 10    # evaluate only first N questions (testing)
  python evaluation/evaluate.py --include-p3  # also evaluate Pipeline 3 if results exist
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline3_graphrag"))

try:
    import _tg_dns_fix  # noqa: F401  — fix broken local DNS for TigerGraph calls
except ImportError:
    pass

from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=ROOT / ".env")

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
JUDGE_MODEL   = "llama-3.3-70b-versatile"

RESULTS_DIR    = ROOT / "results"
EVAL_DIR       = ROOT / "evaluation"
QUESTIONS_FILE = EVAL_DIR / "questions.json"
REPORT_FILE    = RESULTS_DIR / "evaluation_report.json"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_questions(limit: int | None = None) -> dict[int, dict]:
    """Returns {id: {question, ground_truth, type}}."""
    raw = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    if limit:
        raw = raw[:limit]
    return {q["id"]: q for q in raw}


def load_results(path: Path) -> dict[int, dict]:
    """
    Returns {id: {answer, total_tokens, cost_usd, error}}.
    Handles P1/P2 schema (metrics sub-dict) and P3 schema (top-level fields).
    """
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("results", raw) if isinstance(raw, dict) else raw
    out: dict[int, dict] = {}
    for r in items:
        qid = r.get("id")
        if qid is None:
            continue
        m = r.get("metrics", {})
        out[qid] = {
            "answer":       (r.get("answer") or "").strip(),
            "total_tokens": m.get("total_tokens") or r.get("total_tokens", 0),
            "cost_usd":     m.get("cost_usd")     or r.get("cost_usd", 0.0),
            "error":        r.get("error"),
        }
    return out


# ── LLM Judge ─────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are a strict factual evaluator. "
    "Given a question, the correct answer, and a pipeline's answer, "
    "decide if the pipeline answer is correct. "
    "Reply with ONLY the word PASS or FAIL — no explanation, no punctuation."
)


def judge(client: Groq, question: str, ground_truth: str, answer: str) -> str:
    """Returns 'PASS', 'FAIL', or 'ERROR'."""
    if not answer:
        return "FAIL"
    user_msg = (
        f"Question: {question}\n"
        f"Correct answer: {ground_truth}\n"
        f"Pipeline answer: {answer}\n"
        "Does the pipeline answer correctly address the question? "
        "Reply with only PASS or FAIL and nothing else."
    )
    for attempt, backoff in enumerate([30, 60, None]):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=5,
                temperature=0.0,
            )
            verdict = (resp.choices[0].message.content or "").strip().upper()
            return "PASS" if "PASS" in verdict else "FAIL"
        except Exception as exc:
            err = str(exc)
            is_rate = any(k in err.lower() for k in ("429", "rate_limit", "rate limit"))
            if is_rate and backoff:
                print(f"    [judge] Rate limited — sleeping {backoff}s ...")
                time.sleep(backoff)
                continue
            print(f"    [judge] Error: {err[:100]}")
            return "ERROR"
    return "ERROR"


# ── Per-pipeline eval ─────────────────────────────────────────────────────────

def evaluate_pipeline(
    label: str,
    pipeline_results: dict[int, dict],
    questions: dict[int, dict],
    client: Groq,
    delay: float,
) -> dict:
    answered_ids = sorted(
        qid for qid, r in pipeline_results.items()
        if r["answer"] and not r["error"] and qid in questions
    )
    n = len(answered_ids)
    total = len(questions)

    print(f"\n{'='*60}")
    print(f"  {label}  ({n}/{total} questions answered)")
    print(f"{'='*60}")

    per_q: list[dict] = []
    passes = fails = errors = 0

    for i, qid in enumerate(answered_ids, 1):
        q = questions[qid]
        r = pipeline_results[qid]
        verdict = judge(client, q["question"], q["ground_truth"], r["answer"])

        if   verdict == "PASS":  passes += 1
        elif verdict == "FAIL":  fails  += 1
        else:                    errors += 1

        icon = "P" if verdict == "PASS" else ("E" if verdict == "ERROR" else "F")
        print(f"  [{i:02d}/{n}] Q{qid:02d} {icon}  {q['question'][:70]}")

        per_q.append({
            "id":           qid,
            "type":         q.get("type", ""),
            "question":     q["question"],
            "ground_truth": q["ground_truth"],
            "answer":       r["answer"],
            "verdict":      verdict,
            "total_tokens": r["total_tokens"],
            "cost_usd":     r["cost_usd"],
        })

        if i < n:
            time.sleep(delay)

    # Add unanswered questions as SKIP
    for qid in sorted(questions):
        if qid not in answered_ids:
            q = questions[qid]
            r = pipeline_results.get(qid, {})
            per_q.append({
                "id":           qid,
                "type":         q.get("type", ""),
                "question":     q["question"],
                "ground_truth": q["ground_truth"],
                "answer":       r.get("answer", ""),
                "verdict":      "SKIP",
                "total_tokens": r.get("total_tokens", 0),
                "cost_usd":     r.get("cost_usd", 0.0),
                "error":        r.get("error", "no result"),
            })
    per_q.sort(key=lambda x: x["id"])

    pass_pct   = round(passes / n * 100, 1) if n else 0.0
    avg_tokens = round(sum(r["total_tokens"] for r in pipeline_results.values()
                           if not r["error"]) / n, 1) if n else 0.0
    avg_cost   = round(sum(r["cost_usd"] for r in pipeline_results.values()
                           if not r["error"]) / n, 8) if n else 0.0

    # Type breakdown
    single = [p for p in per_q if "single" in p.get("type","") and p["verdict"] != "SKIP"]
    multi  = [p for p in per_q if "multi"  in p.get("type","") and p["verdict"] != "SKIP"]
    s_pass_pct = round(sum(1 for p in single if p["verdict"]=="PASS") / len(single)*100,1) if single else 0.0
    m_pass_pct = round(sum(1 for p in multi  if p["verdict"]=="PASS") / len(multi) *100,1) if multi  else 0.0

    print(f"\n  Pass: {passes}  Fail: {fails}  Error: {errors}  "
          f"Pass%: {pass_pct}%  (single: {s_pass_pct}%  multi: {m_pass_pct}%)")

    return {
        "pipeline":        label,
        "total_questions": total,
        "answered":        n,
        "passes":          passes,
        "fails":           fails,
        "errors":          errors,
        "pass_pct":        pass_pct,
        "single_hop_pass_pct": s_pass_pct,
        "multi_hop_pass_pct":  m_pass_pct,
        "avg_tokens":      avg_tokens,
        "avg_cost_usd":    avg_cost,
        "per_question":    per_q,
    }


# ── Summary table ─────────────────────────────────────────────────────────────

def print_table(results: list[dict]) -> None:
    W = 72
    print(f"\n{'='*W}")
    print(f"  {'Pipeline':<28} {'Judge Pass%':>11} {'Avg Tokens':>11} {'Avg Cost USD':>14}")
    print(f"  {'-'*28} {'-'*11} {'-'*11} {'-'*14}")
    for r in results:
        print(
            f"  {r['pipeline']:<28} "
            f"{r['pass_pct']:>10.1f}% "
            f"{r['avg_tokens']:>11.1f} "
            f"  ${r['avg_cost_usd']:>12.6f}"
        )
    print(f"{'='*W}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--delay",       type=float, default=2.0,
                   help="Seconds between judge calls (default: 2)")
    p.add_argument("--limit",       type=int,   default=None,
                   help="Only evaluate first N questions")
    p.add_argument("--include-p3",  action="store_true",
                   help="Also evaluate Pipeline 3 results (if file exists)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set in .env")
        sys.exit(1)

    client = Groq(api_key=GROQ_API_KEY)

    print("[load] Loading questions ...")
    questions = load_questions(limit=args.limit)
    print(f"       {len(questions)} questions")

    p1 = load_results(RESULTS_DIR / "pipeline1_results.json")
    p2 = load_results(RESULTS_DIR / "pipeline2_results.json")
    p3 = load_results(RESULTS_DIR / "pipeline3_results.json")

    print(f"[load] P1: {len(p1)} results | P2: {len(p2)} results | P3: {len(p3)} results")

    pipelines: list[tuple[str, dict]] = [
        ("Pipeline 1 (LLM-Only)",  p1),
        ("Pipeline 2 (Basic RAG)", p2),
    ]
    if args.include_p3 and p3:
        pipelines.append(("Pipeline 3 (GraphRAG)", p3))
    elif args.include_p3:
        print("[skip] Pipeline 3 has no results")

    all_results: list[dict] = []
    for label, pr in pipelines:
        result = evaluate_pipeline(label, pr, questions, client, args.delay)
        all_results.append(result)

    print_table(all_results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "judge_model":   JUDGE_MODEL,
        "delay_s":       args.delay,
        "pipelines":     all_results,
    }
    REPORT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[save] Report saved to {REPORT_FILE}")


if __name__ == "__main__":
    main()
