"""
Pipeline 1 — LLM-Only Baseline
Sends questions directly to Groq (llama-3.3-70b-versatile) with no retrieval context.
Switched from Gemini → Groq because Gemini free quota was exhausted.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

from groq import Groq

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_INPUT_COST_PER_MILLION,
    GROQ_OUTPUT_COST_PER_MILLION,
    MAX_OUTPUT_TOKENS,
    TEMPERATURE,
    RESULTS_DIR,
    MAX_RETRIES,
    RETRY_DELAY,
)

# Use Groq model for Pipeline 1
P1_MODEL = GROQ_MODEL  # llama-3.3-70b-versatile

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

_CLIENT: Groq | None = None


def get_model() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")
        _CLIENT = Groq(api_key=GROQ_API_KEY)
    return _CLIENT


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    input_cost  = (prompt_tokens     / 1_000_000) * GROQ_INPUT_COST_PER_MILLION
    output_cost = (completion_tokens / 1_000_000) * GROQ_OUTPUT_COST_PER_MILLION
    return round(input_cost + output_cost, 8)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert on cricket, specifically the Indian Premier League (IPL). "
    "Answer the following question as accurately and concisely as possible using your "
    "training knowledge. If you are uncertain, say so briefly but still provide your "
    "best answer. Do not make up statistics."
)


def run_query(question: str) -> dict:
    """
    Send a question to Groq and return a result dict with:
      answer, prompt_tokens, completion_tokens, total_tokens, latency_ms, cost_usd
    """
    client = get_model()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.monotonic()
            response = client.chat.completions.create(
                model=P1_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": question},
                ],
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
            )
            latency_ms = round((time.monotonic() - t0) * 1000, 2)

            usage = response.usage
            prompt_tokens     = usage.prompt_tokens     if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens      = usage.total_tokens      if usage else (prompt_tokens + completion_tokens)

            answer = (response.choices[0].message.content or "").strip()

            return {
                "answer":             answer,
                "prompt_tokens":      prompt_tokens,
                "completion_tokens":  completion_tokens,
                "total_tokens":       total_tokens,
                "latency_ms":         latency_ms,
                "cost_usd":           calculate_cost(prompt_tokens, completion_tokens),
                "error":              None,
            }

        except Exception as exc:
            error_str = str(exc)
            is_rate_limit = any(
                kw in error_str.lower()
                for kw in ("429", "quota", "rate", "resource exhausted", "rate_limit")
            )

            if is_rate_limit and attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"    Rate limit hit (attempt {attempt}/{MAX_RETRIES}). "
                      f"Waiting {wait}s ...")
                time.sleep(wait)
                continue

            return {
                "answer":             "",
                "prompt_tokens":      0,
                "completion_tokens":  0,
                "total_tokens":       0,
                "latency_ms":         0.0,
                "cost_usd":           0.0,
                "error":              error_str,
            }

    return {
        "answer":             "",
        "prompt_tokens":      0,
        "completion_tokens":  0,
        "total_tokens":       0,
        "latency_ms":         0.0,
        "cost_usd":           0.0,
        "error":              "Max retries exceeded",
    }


# ---------------------------------------------------------------------------
# Batch runner  (also callable from run_pipeline1.py)
# ---------------------------------------------------------------------------

def run_batch(
    questions: list[dict],
    delay_between_requests: float = 2.0,
    results_path: Path | None = None,
) -> dict:
    """
    Run all questions through Pipeline 1 and return a summary dict.
    Intermediate results are flushed to disk after every question.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = RESULTS_DIR / "pipeline1_results.json"

    results       = []
    total_cost    = 0.0
    total_tokens  = 0
    total_latency = 0.0
    errors        = 0

    print(f"\n{'='*60}")
    print(f"  Pipeline 1 -- LLM-Only Baseline")
    print(f"  Model  : {P1_MODEL}")
    print(f"  Queries: {len(questions)}")
    print(f"{'='*60}\n")

    for i, q in enumerate(questions, start=1):
        qid      = q["id"]
        question = q["question"]
        qtype    = q.get("type", "unknown")

        print(f"[{i:02d}/{len(questions):02d}] Q{qid} ({qtype})")
        print(f"       {question[:90]}{'...' if len(question) > 90 else ''}")

        result = run_query(question)

        record = {
            "id":           qid,
            "question":     question,
            "type":         qtype,
            "ground_truth": q.get("ground_truth", ""),
            "pipeline":     "llm_only",
            "answer":       result["answer"],
            "metrics": {
                "prompt_tokens":     result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens":      result["total_tokens"],
                "latency_ms":        result["latency_ms"],
                "cost_usd":          result["cost_usd"],
            },
            "error":     result["error"],
            "timestamp": datetime.utcnow().isoformat(),
        }
        results.append(record)

        if result["error"]:
            errors += 1
            print(f"       ERROR: {result['error'][:120]}")
        else:
            total_cost    += result["cost_usd"]
            total_tokens  += result["total_tokens"]
            total_latency += result["latency_ms"]
            print(
                f"       tokens={result['total_tokens']:,}  "
                f"latency={result['latency_ms']:.0f}ms  "
                f"cost=${result['cost_usd']:.6f}"
            )

        _flush(results_path, results)

        if i < len(questions):
            time.sleep(delay_between_requests)

    successful = len(questions) - errors
    avg_tokens  = total_tokens  / successful if successful else 0
    avg_latency = total_latency / successful if successful else 0

    summary = {
        "pipeline":             "llm_only",
        "model":                P1_MODEL,
        "total_questions":      len(questions),
        "successful":           successful,
        "errors":               errors,
        "avg_tokens_per_query": round(avg_tokens, 1),
        "avg_latency_ms":       round(avg_latency, 1),
        "total_cost_usd":       round(total_cost, 6),
        "total_tokens":         total_tokens,
        "breakdown_by_type":    _type_breakdown(results),
        "run_timestamp":        datetime.utcnow().isoformat(),
    }

    final_payload = {"summary": summary, "results": results}
    _flush(results_path, final_payload)
    _print_summary(summary)
    return summary


def _flush(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _type_breakdown(results: list[dict]) -> dict:
    breakdown: dict = {}
    for r in results:
        t = r.get("type", "unknown")
        if t not in breakdown:
            breakdown[t] = {"count": 0, "errors": 0, "total_tokens": 0, "total_cost": 0.0}
        breakdown[t]["count"] += 1
        if r.get("error"):
            breakdown[t]["errors"] += 1
        else:
            breakdown[t]["total_tokens"] += r["metrics"]["total_tokens"]
            breakdown[t]["total_cost"]   += r["metrics"]["cost_usd"]
    return breakdown


def _print_summary(s: dict) -> None:
    sep = "=" * 54
    div = "-" * 51
    lines = [
        "",
        sep,
        "  Pipeline 1 -- Summary",
        f"  Total questions  : {s['total_questions']}",
        f"  Successful       : {s['successful']}",
        f"  Errors           : {s['errors']}",
        f"  {div}",
        f"  Avg tokens/query : {s['avg_tokens_per_query']:,.1f}",
        f"  Avg latency      : {s['avg_latency_ms']:,.1f} ms",
        f"  Total tokens     : {s['total_tokens']:,}",
        f"  Total cost       : ${s['total_cost_usd']:.6f} USD",
        sep,
        "",
    ]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI entry point  (single question)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py \"<question>\"")
        sys.exit(1)

    question_text = " ".join(sys.argv[1:])
    print(f"\nQuestion: {question_text}\n")
    res = run_query(question_text)

    if res["error"]:
        print(f"Error: {res['error']}")
        sys.exit(1)

    print(f"Answer:\n{res['answer']}")
    print(f"\nMetrics:")
    print(f"  Prompt tokens     : {res['prompt_tokens']:,}")
    print(f"  Completion tokens : {res['completion_tokens']:,}")
    print(f"  Total tokens      : {res['total_tokens']:,}")
    print(f"  Latency           : {res['latency_ms']:.1f} ms")
    print(f"  Cost              : ${res['cost_usd']:.6f} USD")
