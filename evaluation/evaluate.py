"""
evaluation/evaluate.py — LLM-as-Judge + BERTScore evaluation
Compares Pipeline 1, 2, and 3 answers against ground-truth answers.

Metrics
-------
  Judge Pass%   : Groq llama-3.3-70b-versatile rates each answer PASS/FAIL
  BERTScore F1  : Semantic similarity vs ground truth (distilbert-base-uncased)
  Avg Tokens    : Mean tokens per query (from results files)
  Avg Cost USD  : Mean cost per query (from results files)

Usage
-----
  python evaluation/evaluate.py
  python evaluation/evaluate.py --skip-p3       # skip Pipeline 3 (not enough results)
  python evaluation/evaluate.py --judge-delay 3 # seconds between judge calls

Output
------
  results/evaluation_report.json   -- full per-question breakdown
  Prints summary table to stdout
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline3_graphrag"))

# DNS fix for TigerGraph (harmless if TG not needed here)
try:
    import _tg_dns_fix  # noqa: F401
except ImportError:
    pass

from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
JUDGE_MODEL  = "llama-3.3-70b-versatile"

RESULTS_DIR    = ROOT / "results"
EVAL_DIR       = ROOT / "evaluation"
QUESTIONS_FILE = EVAL_DIR / "questions.json"
REPORT_FILE    = RESULTS_DIR / "evaluation_report.json"


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_questions() -> dict[int, dict]:
    """Returns {question_id: {question, ground_truth, type}}."""
    raw = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    return {q["id"]: q for q in raw}


def load_pipeline_results(path: Path) -> dict[int, dict]:
    """
    Returns {question_id: {answer, total_tokens, cost_usd, error}}.
    Handles both P1/P2 schema (metrics sub-dict) and P3 schema (top-level).
    """
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    results_list = raw.get("results", raw) if isinstance(raw, dict) else raw
    out: dict[int, dict] = {}
    for r in results_list:
        qid = r.get("id")
        if qid is None:
            continue
        # Normalise token / cost fields across schema variants
        metrics  = r.get("metrics", {})
        tokens   = metrics.get("total_tokens") or r.get("total_tokens", 0)
        cost     = metrics.get("cost_usd")     or r.get("cost_usd", 0.0)
        latency  = metrics.get("latency_ms")   or r.get("latency_ms", 0.0)
        out[qid] = {
            "answer":       (r.get("answer") or "").strip(),
            "total_tokens": tokens,
            "cost_usd":     cost,
            "latency_ms":   latency,
            "error":        r.get("error"),
        }
    return out


# ── LLM-as-Judge ─────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are a strict factual evaluator. "
    "Given a question, the correct answer, and a pipeline's answer, "
    "decide whether the pipeline answer is correct. "
    "Reply with ONLY the word PASS or FAIL — no explanation, no punctuation."
)


def judge_answer(
    client: Groq,
    question: str,
    ground_truth: str,
    pipeline_answer: str,
) -> str:
    """Returns 'PASS', 'FAIL', or 'ERROR'."""
    if not pipeline_answer:
        return "FAIL"
    user_msg = (
        f"Question: {question}\n"
        f"Correct answer: {ground_truth}\n"
        f"Pipeline answer: {pipeline_answer}\n"
        "Does the pipeline answer correctly address the question? "
        "Reply with only PASS or FAIL and nothing else."
    )
    backoff = [30, 60]
    for attempt in range(len(backoff) + 1):
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
            # Normalise — model may add punctuation despite instructions
            if "PASS" in verdict:
                return "PASS"
            if "FAIL" in verdict:
                return "FAIL"
            return "FAIL"  # unexpected response → treat as fail
        except Exception as exc:
            err = str(exc)
            is_rate = any(k in err.lower() for k in ("429", "rate_limit", "rate limit"))
            if is_rate and attempt < len(backoff):
                wait = backoff[attempt]
                print(f"    [judge] Rate limited, sleeping {wait}s ...")
                time.sleep(wait)
                continue
            print(f"    [judge] Error: {err[:100]}")
            return "ERROR"
    return "ERROR"


# ── BERTScore ─────────────────────────────────────────────────────────────────

def compute_bert_scores(
    predictions: list[str],
    references: list[str],
    model_type: str = "distilbert-base-uncased",
) -> list[float]:
    """
    Returns F1 scores per (prediction, reference) pair.
    Falls back to 0.0 per item on import error.
    """
    try:
        from bert_score import score as bert_score_fn
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [bert] Computing BERTScore on {len(predictions)} pairs "
              f"(model={model_type}, device={device}) ...")
        P, R, F1 = bert_score_fn(
            predictions, references,
            model_type=model_type,
            device=device,
            verbose=False,
        )
        return [round(float(f), 4) for f in F1]
    except Exception as exc:
        print(f"  [bert] BERTScore failed: {exc}. Using 0.0 for all.")
        return [0.0] * len(predictions)


# ── Per-pipeline evaluation ───────────────────────────────────────────────────

def evaluate_pipeline(
    name: str,
    pipeline_results: dict[int, dict],
    questions: dict[int, dict],
    groq_client: Groq,
    judge_delay: float,
) -> dict:
    """
    Runs Judge + BERTScore for every answered question in this pipeline.
    Returns a rich dict with per-question details and aggregate stats.
    """
    answered_ids = sorted(
        qid for qid, r in pipeline_results.items()
        if r["answer"] and not r["error"]
    )
    total_qs = len(questions)
    answered = len(answered_ids)

    print(f"\n[{name}] {answered}/{total_qs} questions answered — running evaluation ...")

    judge_results: dict[int, str]   = {}
    bert_preds:    list[str]        = []
    bert_refs:     list[str]        = []
    bert_ids:      list[int]        = []

    # ── Phase 1: LLM Judge (sequential with delay) ──────────────────────────
    for i, qid in enumerate(answered_ids, 1):
        q   = questions[qid]
        r   = pipeline_results[qid]
        verdict = judge_answer(
            groq_client,
            q["question"],
            q["ground_truth"],
            r["answer"],
        )
        judge_results[qid] = verdict
        status_char = "P" if verdict == "PASS" else ("E" if verdict == "ERROR" else "F")
        print(f"  Q{qid:02d} [{i:02d}/{answered}] {status_char}  {q['question'][:60]}")

        # Collect for BERTScore
        bert_preds.append(r["answer"])
        bert_refs.append(q["ground_truth"])
        bert_ids.append(qid)

        if i < answered:
            time.sleep(judge_delay)

    # ── Phase 2: BERTScore (batch) ───────────────────────────────────────────
    bert_f1_scores = compute_bert_scores(bert_preds, bert_refs)
    bert_map: dict[int, float] = dict(zip(bert_ids, bert_f1_scores))

    # ── Aggregate stats ───────────────────────────────────────────────────────
    passes  = sum(1 for v in judge_results.values() if v == "PASS")
    fails   = sum(1 for v in judge_results.values() if v == "FAIL")
    errs    = sum(1 for v in judge_results.values() if v == "ERROR")
    pass_pct = round(passes / answered * 100, 1) if answered else 0.0

    avg_bert = round(sum(bert_f1_scores) / len(bert_f1_scores), 4) if bert_f1_scores else 0.0

    all_tokens = [pipeline_results[qid]["total_tokens"] for qid in answered_ids]
    all_costs  = [pipeline_results[qid]["cost_usd"]     for qid in answered_ids]
    avg_tokens = round(sum(all_tokens) / answered, 1) if answered else 0.0
    avg_cost   = round(sum(all_costs)  / answered, 8) if answered else 0.0

    # Per-question detail records
    per_question = []
    for qid in answered_ids:
        q = questions[qid]
        r = pipeline_results[qid]
        per_question.append({
            "id":            qid,
            "type":          q.get("type", ""),
            "question":      q["question"],
            "ground_truth":  q["ground_truth"],
            "answer":        r["answer"],
            "judge_verdict": judge_results.get(qid, "SKIP"),
            "bert_f1":       bert_map.get(qid, 0.0),
            "total_tokens":  r["total_tokens"],
            "cost_usd":      r["cost_usd"],
            "latency_ms":    r["latency_ms"],
        })

    # Include unanswered questions
    for qid in sorted(questions.keys()):
        if qid not in answered_ids:
            q = questions[qid]
            r = pipeline_results.get(qid, {})
            per_question.append({
                "id":            qid,
                "type":          q.get("type", ""),
                "question":      q["question"],
                "ground_truth":  q["ground_truth"],
                "answer":        r.get("answer", ""),
                "judge_verdict": "SKIP",
                "bert_f1":       0.0,
                "total_tokens":  r.get("total_tokens", 0),
                "cost_usd":      r.get("cost_usd", 0.0),
                "latency_ms":    r.get("latency_ms", 0.0),
                "error":         r.get("error", "no answer"),
            })

    per_question.sort(key=lambda x: x["id"])

    return {
        "pipeline":      name,
        "total_questions": total_qs,
        "answered":      answered,
        "judge_pass":    passes,
        "judge_fail":    fails,
        "judge_error":   errs,
        "judge_pass_pct": pass_pct,
        "avg_bert_f1":   avg_bert,
        "avg_tokens":    avg_tokens,
        "avg_cost_usd":  avg_cost,
        "per_question":  per_question,
    }


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(results: list[dict]) -> None:
    header = f"{'Pipeline':<20} {'Judge Pass%':>11} {'BERTScore F1':>13} {'Avg Tokens':>11} {'Avg Cost USD':>13}"
    divider = "-" * len(header)
    print(f"\n{divider}")
    print("  Evaluation Summary")
    print(divider)
    print(header)
    print(divider)
    for r in results:
        print(
            f"  {r['pipeline']:<18} "
            f"{r['judge_pass_pct']:>10.1f}% "
            f"{r['avg_bert_f1']:>13.4f} "
            f"{r['avg_tokens']:>11.1f} "
            f"${r['avg_cost_usd']:>12.6f}"
        )
    print(divider)

    # Type breakdown (single-hop vs multi-hop)
    for r in results:
        single = [q for q in r["per_question"] if "single" in q.get("type","")]
        multi  = [q for q in r["per_question"] if "multi"  in q.get("type","")]
        s_pass = sum(1 for q in single if q["judge_verdict"] == "PASS")
        m_pass = sum(1 for q in multi  if q["judge_verdict"] == "PASS")
        s_answered = sum(1 for q in single if q["judge_verdict"] != "SKIP")
        m_answered = sum(1 for q in multi  if q["judge_verdict"] != "SKIP")
        s_pct = round(s_pass / s_answered * 100, 1) if s_answered else 0.0
        m_pct = round(m_pass / m_answered * 100, 1) if m_answered else 0.0
        print(f"  {r['pipeline']:<18}  single-hop: {s_pct:.1f}%  multi-hop: {m_pct:.1f}%")
    print(divider + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pipeline results")
    p.add_argument("--skip-p3",      action="store_true",
                   help="Skip Pipeline 3 evaluation (useful if results are sparse)")
    p.add_argument("--judge-delay",  type=float, default=3.0,
                   help="Seconds between LLM judge calls (default: 3)")
    p.add_argument("--bert-model",   default="distilbert-base-uncased",
                   help="HuggingFace model for BERTScore")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set in .env")
        sys.exit(1)

    groq_client = Groq(api_key=GROQ_API_KEY)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("[load] Loading questions ...")
    questions = load_questions()
    print(f"       {len(questions)} questions loaded")

    p1 = load_pipeline_results(RESULTS_DIR / "pipeline1_results.json")
    p2 = load_pipeline_results(RESULTS_DIR / "pipeline2_results.json")
    p3 = load_pipeline_results(RESULTS_DIR / "pipeline3_results.json")

    print(f"[load] P1: {len(p1)} results  |  P2: {len(p2)} results  |  P3: {len(p3)} results")

    pipelines_to_eval: list[tuple[str, dict[int, dict]]] = [
        ("Pipeline 1 (LLM-Only)",  p1),
        ("Pipeline 2 (Basic RAG)", p2),
    ]
    if p3 and not args.skip_p3:
        pipelines_to_eval.append(("Pipeline 3 (GraphRAG)", p3))
    elif args.skip_p3:
        print("[skip] Pipeline 3 skipped (--skip-p3)")
    else:
        print("[skip] Pipeline 3 has no results — skipped")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    all_results: list[dict] = []
    for name, pr in pipelines_to_eval:
        result = evaluate_pipeline(
            name=name,
            pipeline_results=pr,
            questions=questions,
            groq_client=groq_client,
            judge_delay=args.judge_delay,
        )
        all_results.append(result)

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary_table(all_results)

    # ── Save report ───────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "judge_model":   JUDGE_MODEL,
        "bert_model":    args.bert_model,
        "judge_delay_s": args.judge_delay,
        "pipelines":     all_results,
    }
    REPORT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[save] Full report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
