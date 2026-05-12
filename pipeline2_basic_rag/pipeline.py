"""
Pipeline 2 — Basic RAG
Four stages in one file:
  1. CHUNKING   – corpus.txt → overlapping token chunks → chunks.json
  2. EMBEDDING  – chunks → all-MiniLM-L6-v2 → FAISS index (built once)
  3. RETRIEVAL  – question embedding → top-K FAISS lookup
  4. GENERATION – retrieved context + question → Groq (llama-3.3-70b-versatile)
"""

import sys
import io
import json
import time
import re
import numpy as np
from pathlib import Path
from datetime import datetime

import tiktoken
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from groq import Groq

# Force UTF-8 output so Unicode article titles don't crash on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
elif hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GROQ_API_KEY, GROQ_MODEL,
    GROQ_INPUT_COST_PER_MILLION, GROQ_OUTPUT_COST_PER_MILLION,
    MAX_OUTPUT_TOKENS, TEMPERATURE,
    CORPUS_FILE, RESULTS_DIR,
    MAX_RETRIES, RETRY_DELAY,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PIPELINE_DIR     = Path(__file__).parent
CHUNKS_FILE      = PIPELINE_DIR / "chunks.json"
CHUNK_META_FILE  = PIPELINE_DIR / "chunk_metadata.json"
FAISS_INDEX_FILE = PIPELINE_DIR / "faiss_index.bin"

CHUNK_SIZE           = 512   # tokens per chunk
OVERLAP              = 50    # token overlap between consecutive chunks
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM        = 384   # dimension for all-MiniLM-L6-v2
BATCH_SIZE           = 64    # sentence-transformer batch size
TOP_K                = 5     # chunks to retrieve per query

# ---------------------------------------------------------------------------
# Shared encoder (tiktoken)
# ---------------------------------------------------------------------------
_ENC = None

def get_encoder() -> tiktoken.Encoding:
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC

def count_tokens(text: str) -> int:
    return len(get_encoder().encode(text))

# ---------------------------------------------------------------------------
# Groq client (lazy init)
# ---------------------------------------------------------------------------
_GROQ_CLIENT: Groq | None = None

def get_groq() -> Groq:
    global _GROQ_CLIENT
    if _GROQ_CLIENT is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")
        _GROQ_CLIENT = Groq(api_key=GROQ_API_KEY)
    return _GROQ_CLIENT

# ---------------------------------------------------------------------------
# Embedding model (lazy init, stays in memory across queries)
# ---------------------------------------------------------------------------
_EMBED_MODEL: SentenceTransformer | None = None

def get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        print(f"[EMBED] Loading '{EMBEDDING_MODEL_NAME}' (first call only) ...")
        _EMBED_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBED_MODEL

# ===========================================================================
# STAGE 1 — CHUNKING
# ===========================================================================

def _parse_corpus(corpus_path: Path) -> list[tuple[str, str]]:
    """
    Parse corpus.txt into (title, text) pairs.
    corpus.txt format (written by collect_dataset.py):
        \\n\\n{=*60}\\nARTICLE: <title>\\n{=*60}\\n<text>
    """
    print(f"[CHUNK] Reading corpus from {corpus_path} ...")
    raw = corpus_path.read_text(encoding="utf-8")

    # Split on the header pattern; first element is pre-article preamble
    parts = re.split(r"\n={60}\nARTICLE: ", raw)
    articles: list[tuple[str, str]] = []

    for part in parts[1:]:          # skip empty preamble
        try:
            nl1 = part.index("\n")
            title = part[:nl1].strip()
            rest  = part[nl1 + 1:]  # starts with the closing ===...=== line
            nl2   = rest.index("\n")
            text  = rest[nl2 + 1:].strip()
            if title and text:
                articles.append((title, text))
        except ValueError:
            continue                # malformed section, skip

    print(f"[CHUNK] Parsed {len(articles)} articles from corpus.")
    return articles


def _chunk_article(title: str, text: str, global_start_id: int) -> list[dict]:
    """
    Tokenize `text` and slice into overlapping windows of CHUNK_SIZE tokens.
    Returns a list of chunk dicts ready for chunks.json.
    """
    enc    = get_encoder()
    tokens = enc.encode(text)
    if not tokens:
        return []

    chunks: list[dict] = []
    pos = 0
    while pos < len(tokens):
        end      = min(pos + CHUNK_SIZE, len(tokens))
        window   = tokens[pos:end]
        chunk_text = enc.decode(window)
        chunks.append({
            "chunk_id":    global_start_id + len(chunks),
            "source":      title,
            "text":        chunk_text,
            "token_count": len(window),
        })
        if end == len(tokens):
            break
        pos += CHUNK_SIZE - OVERLAP   # advance with overlap

    return chunks


def build_chunks(corpus_path: Path = CORPUS_FILE) -> list[dict]:
    """Parse corpus → chunk all articles → save chunks.json. Returns chunk list."""
    articles   = _parse_corpus(corpus_path)
    all_chunks: list[dict] = []

    for title, text in tqdm(articles, desc="Chunking", unit="article"):
        article_chunks = _chunk_article(title, text, global_start_id=len(all_chunks))
        all_chunks.extend(article_chunks)

    print(f"[CHUNK] Total chunks created: {len(all_chunks):,}")
    CHUNKS_FILE.write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[CHUNK] Saved -> {CHUNKS_FILE}")
    return all_chunks

# ===========================================================================
# STAGE 2 — EMBEDDING + FAISS
# ===========================================================================

def _embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Embed texts in batches; returns float32 array of shape (N, EMBEDDING_DIM)."""
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit vectors → cosine sim == inner product
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def build_index() -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Build (or load) the FAISS index and chunk metadata.

    * If faiss_index.bin + chunk_metadata.json already exist on disk,
      load them immediately (skip re-embedding).
    * Otherwise: chunk corpus → embed → build IndexFlatIP → save both.
    Returns (faiss_index, metadata_list).
    """
    if FAISS_INDEX_FILE.exists() and CHUNK_META_FILE.exists():
        return _load_index()

    print("\n" + "=" * 58)
    print("  Building RAG index  (runs once, then cached to disk)")
    print("=" * 58)

    # --- chunking ---
    if CHUNKS_FILE.exists():
        print(f"[CHUNK] Loading existing chunks from {CHUNKS_FILE} ...")
        chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
        print(f"[CHUNK] Loaded {len(chunks):,} chunks.")
    else:
        chunks = build_chunks()

    # --- embedding ---
    model = get_embed_model()
    texts = [c["text"] for c in chunks]
    print(f"\n[EMBED] Embedding {len(texts):,} chunks ...")
    embeddings = _embed_texts(texts, model)
    print(f"[EMBED] Embeddings shape: {embeddings.shape}  dtype: {embeddings.dtype}")

    # --- FAISS IndexFlatIP (exact cosine similarity via inner product) ---
    print(f"\n[FAISS] Building IndexFlatIP  dim={EMBEDDING_DIM} ...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    print(f"[FAISS] Index contains {index.ntotal:,} vectors.")

    # --- persist ---
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    print(f"[FAISS] Saved -> {FAISS_INDEX_FILE}")

    # chunk_metadata.json: lightweight parallel array to the FAISS index.
    # Position i in this list corresponds to vector i in the index.
    metadata = [
        {"chunk_id": c["chunk_id"], "source": c["source"], "text": c["text"]}
        for c in chunks
    ]
    CHUNK_META_FILE.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[META]  Saved -> {CHUNK_META_FILE}")

    print("\n[OK] Index build complete.\n")
    return index, metadata


def _load_index() -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Load FAISS index + chunk metadata from disk."""
    print("[LOAD] Found cached index — loading from disk ...")
    index    = faiss.read_index(str(FAISS_INDEX_FILE))
    metadata = json.loads(CHUNK_META_FILE.read_text(encoding="utf-8"))
    print(f"[LOAD] FAISS index: {index.ntotal:,} vectors | "
          f"Metadata: {len(metadata):,} chunks")
    return index, metadata

# ===========================================================================
# STAGE 3 — RETRIEVAL
# ===========================================================================

def _retrieve(
    question: str,
    index: faiss.IndexFlatIP,
    metadata: list[dict],
    model: SentenceTransformer,
    k: int = TOP_K,
) -> list[dict]:
    """
    Embed `question`, search FAISS for top-k, return chunk dicts
    augmented with similarity `score`.
    """
    q_vec = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    scores, indices = index.search(q_vec, k)

    retrieved: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(metadata):
            chunk = dict(metadata[idx])
            chunk["score"] = float(score)
            retrieved.append(chunk)
    return retrieved


def _build_prompt(question: str, chunks: list[dict]) -> str:
    """Format the RAG prompt with numbered context blocks."""
    context_blocks = "\n\n".join(
        f"[Source {i}: {c['source']}]\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )
    return (
        "Answer the following question using only the context provided. "
        "Be concise and accurate. "
        "If the context does not contain enough information to answer, say so clearly.\n\n"
        f"Context:\n{context_blocks}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

# ===========================================================================
# STAGE 4 — GENERATION  (Groq — llama-3.3-70b-versatile)
# ===========================================================================

SYSTEM_PROMPT = (
    "You are an expert on cricket, specifically the Indian Premier League (IPL). "
    "Answer the following question using only the context provided. "
    "Be concise and accurate. "
    "If the context does not contain enough information to answer, say so clearly."
)

def _call_groq(prompt: str) -> tuple[str, int, int, float]:
    """
    Call Groq with exponential backoff on rate limits.
    Waits: 30s → 60s → 120s, then gives up on that question.
    Returns (answer, prompt_tokens, completion_tokens, latency_ms).

    Groq returns exact token counts via response.usage.
    """
    client  = get_groq()
    backoff = [30, 60, 120]
    for attempt in range(1, len(backoff) + 2):
        try:
            t0       = time.monotonic()
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
            )
            latency_ms = round((time.monotonic() - t0) * 1000, 2)

            usage             = response.usage
            prompt_tokens     = usage.prompt_tokens     if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            answer = (response.choices[0].message.content or "").strip()
            return answer, prompt_tokens, completion_tokens, latency_ms

        except Exception as exc:
            err = str(exc)
            rate_limited = any(
                kw in err.lower()
                for kw in ("429", "rate_limit", "rate limit", "too many requests")
            )
            retry_idx = attempt - 1
            if rate_limited and retry_idx < len(backoff):
                wait = backoff[retry_idx]
                print(f"    [WAIT] Rate limit — attempt {attempt}/{len(backoff) + 1}, "
                      f"sleeping {wait}s ...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded after backoff schedule")


def _cost(prompt_tokens: int, completion_tokens: int) -> float:
    return round(
        (prompt_tokens     / 1_000_000) * GROQ_INPUT_COST_PER_MILLION
        + (completion_tokens / 1_000_000) * GROQ_OUTPUT_COST_PER_MILLION,
        8,
    )

# ===========================================================================
# Public API: run_query + run_batch
# ===========================================================================

def run_query(
    question: str,
    index: faiss.IndexFlatIP,
    metadata: list[dict],
    embed_model: SentenceTransformer,
) -> dict:
    """
    Full RAG pipeline for a single question.
    Returns:
        answer          (str)
        retrieved_sources (list[str])
        retrieved_scores  (list[float])
        metrics         (dict)
        error           (str | None)
    """
    t_start = time.monotonic()

    # Retrieve
    retrieved = _retrieve(question, index, metadata, embed_model)
    prompt    = _build_prompt(question, retrieved)

    # Generate
    try:
        answer, p_tok, c_tok, latency_ms = _call_groq(prompt)
        error = None
    except Exception as exc:
        answer   = ""
        p_tok    = c_tok = 0
        latency_ms = round((time.monotonic() - t_start) * 1000, 2)
        error    = str(exc)

    return {
        "answer":            answer,
        "retrieved_sources": [c["source"] for c in retrieved],
        "retrieved_scores":  [round(c["score"], 4) for c in retrieved],
        "metrics": {
            "retrieval_chunks":   len(retrieved),
            "prompt_tokens":      p_tok,
            "completion_tokens":  c_tok,
            "total_tokens":       p_tok + c_tok,
            "latency_ms":         latency_ms,
            "cost_usd":           _cost(p_tok, c_tok),
        },
        "error": error,
    }


def run_batch(
    questions: list[dict],
    delay_between_requests: float = 4.0,
    results_path: Path | None = None,
    existing_results: list[dict] | None = None,
) -> dict:
    """
    Run all questions through the RAG pipeline.
    - existing_results: records already on disk; successful ones are SKIPPED,
      errored ones are RETRIED.
    - Flushes to disk after every query (crash-safe).
    Returns the summary dict.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results_path is None:
        results_path = RESULTS_DIR / "pipeline2_results.json"

    # ── Seed state from previous successful results ───────────────────────────
    tot_cost = tot_tokens = tot_prompt = tot_completion = tot_latency = 0.0
    errors   = 0

    # Keep only the records that completed without error; errored ones are retried
    done_ids: set[int] = set()
    results: list[dict] = []
    for r in (existing_results or []):
        if not r.get("error"):
            results.append(r)
            done_ids.add(r["id"])
            m = r.get("metrics", {})
            tot_cost       += m.get("cost_usd",           0)
            tot_tokens     += m.get("total_tokens",        0)
            tot_prompt     += m.get("prompt_tokens",       0)
            tot_completion += m.get("completion_tokens",   0)
            tot_latency    += m.get("latency_ms",          0)
        # errored records are simply dropped so they get re-run below

    pending = [q for q in questions if q["id"] not in done_ids]

    # ── Resume banner ─────────────────────────────────────────────────────────
    if done_ids:
        id_preview = str(sorted(done_ids)[:8])[1:-1]
        if len(done_ids) > 8:
            id_preview += ", ..."
        print(f"\n[RESUME] {len(done_ids)} questions already completed "
              f"(IDs: {id_preview}) — skipping.")
        if pending:
            print(f"[RESUME] Resuming from Q{pending[0]['id']}  "
                  f"({len(pending)} remaining out of {len(questions)} total)")
        else:
            print("[RESUME] All questions already completed.")

    # Ensure index is ready before the timed loop
    index, metadata = build_index()
    embed_model     = get_embed_model()

    print(f"\n{'='*60}")
    print(f"  Pipeline 2 — Basic RAG")
    print(f"  LLM       : {GROQ_MODEL}  (Groq)")
    print(f"  Embedding : {EMBEDDING_MODEL_NAME}")
    print(f"  Retrieval : top-{TOP_K} chunks  |  chunk size: {CHUNK_SIZE} tok  |  overlap: {OVERLAP} tok")
    print(f"  Total     : {len(questions)} questions  ({len(pending)} to run, {len(done_ids)} skipped)")
    print(f"{'='*60}\n")

    need_delay = False   # don't sleep before the very first real request
    for i, q in enumerate(questions, start=1):
        qid      = q["id"]
        question = q["question"]
        qtype    = q.get("type", "unknown")

        # ── Skip already-completed questions ──────────────────────────────────
        if qid in done_ids:
            print(f"[{i:02d}/{len(questions):02d}] Q{qid} ({qtype}) [SKIP]")
            continue

        print(f"[{i:02d}/{len(questions):02d}] Q{qid} ({qtype})")
        print(f"       {question[:90]}{'...' if len(question) > 90 else ''}")

        if need_delay:
            time.sleep(delay_between_requests)

        result = run_query(question, index, metadata, embed_model)
        need_delay = True   # sleep before every subsequent request

        record = {
            "id":               qid,
            "question":         question,
            "type":             qtype,
            "ground_truth":     q.get("ground_truth", ""),
            "pipeline":         "basic_rag",
            "answer":           result["answer"],
            "retrieved_sources": result["retrieved_sources"],
            "retrieved_scores":  result["retrieved_scores"],
            "metrics":          result["metrics"],
            "error":            result["error"],
            "timestamp":        datetime.utcnow().isoformat(),
        }
        results.append(record)

        m = result["metrics"]
        if result["error"]:
            errors += 1
            print(f"       [ERR] {result['error'][:120]}")
        else:
            tot_cost       += m["cost_usd"]
            tot_tokens     += m["total_tokens"]
            tot_prompt     += m["prompt_tokens"]
            tot_completion += m["completion_tokens"]
            tot_latency    += m["latency_ms"]
            src_preview     = str(result["retrieved_sources"][:2])[1:-1]
            print(
                f"       [OK]  tokens={m['total_tokens']:,}  "
                f"latency={m['latency_ms']:.0f}ms  "
                f"cost=${m['cost_usd']:.6f}"
            )
            print(f"             top sources: {src_preview}")

        # Flush after every query — safe against crashes mid-run
        _flush(results_path, {"summary": {}, "results": results})

    ok = len(questions) - errors
    summary = {
        "pipeline":              "basic_rag",
        "model":                 GROQ_MODEL,
        "embedding_model":       EMBEDDING_MODEL_NAME,
        "top_k":                 TOP_K,
        "chunk_size":            CHUNK_SIZE,
        "overlap":               OVERLAP,
        "total_questions":       len(questions),
        "successful":            ok,
        "errors":                errors,
        "avg_prompt_tokens":     round(tot_prompt     / ok, 1) if ok else 0,
        "avg_completion_tokens": round(tot_completion / ok, 1) if ok else 0,
        "avg_total_tokens":      round(tot_tokens     / ok, 1) if ok else 0,
        "avg_latency_ms":        round(tot_latency    / ok, 1) if ok else 0,
        "total_cost_usd":        round(tot_cost, 6),
        "total_tokens":          int(tot_tokens),
        "run_timestamp":         datetime.utcnow().isoformat(),
    }

    _flush(results_path, {"summary": summary, "results": results})
    _print_summary(summary)
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flush(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _print_summary(s: dict) -> None:
    print()
    print("=" * 58)
    print("  Pipeline 2 (Basic RAG) — Summary")
    print(f"  Total questions    : {s['total_questions']}")
    print(f"  Successful         : {s['successful']}")
    print(f"  Errors             : {s['errors']}")
    print("-" * 58)
    print(f"  Avg prompt tokens  : {s['avg_prompt_tokens']:>10,.1f}")
    print(f"  Avg completion tok : {s['avg_completion_tokens']:>10,.1f}")
    print(f"  Avg total tokens   : {s['avg_total_tokens']:>10,.1f}")
    print(f"  Avg latency        : {s['avg_latency_ms']:>10,.1f} ms")
    print(f"  Total tokens       : {s['total_tokens']:>10,}")
    print(f"  Total cost         : ${s['total_cost_usd']:>12.6f} USD")
    print("=" * 58)
    print()


# ===========================================================================
# CLI  (single question or --build-only)
# ===========================================================================

if __name__ == "__main__":
    if "--build" in sys.argv:
        build_index()
        sys.exit(0)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Build index  : python pipeline.py --build")
        print("  Single query : python pipeline.py \"<question>\"")
        sys.exit(1)

    q_text = " ".join(a for a in sys.argv[1:] if not a.startswith("--"))
    print(f"\nQuestion: {q_text}\n")

    idx, meta = build_index()
    emb       = get_embed_model()
    res       = run_query(q_text, idx, meta, emb)

    if res["error"]:
        print(f"Error: {res['error']}")
        sys.exit(1)

    print(f"\nAnswer:\n{res['answer']}")
    print(f"\nRetrieved sources ({len(res['retrieved_sources'])}):")
    for src, score in zip(res["retrieved_sources"], res["retrieved_scores"]):
        print(f"  [{score:.4f}]  {src}")
    print("\nMetrics:")
    for k, v in res["metrics"].items():
        print(f"  {k:<25}: {v}")
