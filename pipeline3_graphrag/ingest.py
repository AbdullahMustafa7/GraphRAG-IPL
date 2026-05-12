"""
Pipeline 3 -- Step 2: Ingest articles into TigerGraph knowledge graph.

For each .txt article in data/raw/:
  1. Extract entities via Groq (PERSON, TEAM, VENUE, AWARD, SEASON, EVENT)
  2. Extract relationships via Groq
  3. Upsert Document + Entity vertices and CONTAINS + RELATED_TO edges

Progress is saved to results/ingest_progress.json after every batch.
Failed articles are logged to results/ingest_failures.json.

Usage:
  python pipeline3_graphrag/ingest.py
"""

import os
import sys
import re
import json
import time
import glob
import requests
import _tg_dns_fix  # Override broken local DNS → uses Google-resolved IPs for TigerGraph
from pathlib import Path
from datetime import datetime, timezone

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from groq import Groq
import pyTigerGraph as tg
from dotenv import load_dotenv

# ── Config ─────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

TG_HOST      = "https://tg-1727b3d7-e9cd-4032-9168-238043254e0c.tg-2635877100.i.tgcloud.io"
TG_SECRET    = "dvhejqo4r39v302aqi2vfim23ihqp8vn"
TG_GRAPHNAME = "MyDatabase"
TG_USERNAME  = "tigergraph"

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
# Use a fast small model for ingestion -- it has ~8x higher daily token limits
# than llama-3.3-70b (which is reserved for the query pipeline).
GROQ_MODEL     = "llama-3.1-8b-instant"

DATA_DIR       = Path(__file__).parent.parent / "data" / "raw"
RESULTS_DIR    = Path(__file__).parent.parent / "results"
PROGRESS_FILE  = RESULTS_DIR / "ingest_progress.json"
FAILURES_FILE  = RESULTS_DIR / "ingest_failures.json"

BATCH_SIZE     = 10
BATCH_DELAY    = 3       # seconds between batches
MAX_CONTENT    = 1500    # chars sent to Groq for extraction — trim to ~375 tokens
                         # (saves ~3x tokens; 1500 chars still covers key facts per article)
GROQ_DELAY     = 3.0     # seconds between the two Groq calls per article

ENTITY_PROMPT = (
    "Extract named entities from this IPL cricket text.\n"
    "Return ONLY a JSON array, no other text:\n"
    '[{"name": "...", "type": "PERSON/TEAM/VENUE/AWARD/SEASON/EVENT", '
    '"description": "one sentence"}]\n'
    "If no entities found, return []"
)

RELATIONSHIP_PROMPT = (
    "Given these entities from an IPL cricket article, "
    "list relationships between them.\n"
    "Return ONLY a JSON array, no other text:\n"
    '[{"from": "entity_name", "to": "entity_name", '
    '"relationship": "brief description"}]\n'
    "Only include relationships explicitly stated in the text.\n"
    "If none, return []"
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def sanitize_entity_name(name: str) -> str:
    """
    Normalize entity name to a safe TigerGraph vertex ID:
      - lowercase
      - spaces -> underscores
      - strip non-alphanumeric (keep underscores)
      - truncate to 100 chars
    """
    name = name.strip().lower()
    name = name.replace(" ", "_")
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name[:100]


def sanitize_doc_id(filename: str) -> str:
    """Turn a raw filename (no extension) into a safe doc_id."""
    name = Path(filename).stem.strip().lower()
    name = name.replace(" ", "_")
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name[:200]


# ── Auth + connection ────────────────────────────────────────────────────────

def get_token(host: str, secret: str) -> str:
    url = f"{host}/gsql/v1/tokens"
    resp = requests.post(url, json={"secret": secret}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    token = data.get("token") or data.get("results", {}).get("token")
    if not token:
        raise ValueError(f"Token not found in response: {data}")
    return token


def get_connection(token: str) -> tg.TigerGraphConnection:
    return tg.TigerGraphConnection(
        host=TG_HOST,
        username=TG_USERNAME,
        restppPort="443",
        gsPort="443",
        graphname=TG_GRAPHNAME,
        apiToken=token,
        tgCloud=True,
    )


# ── Progress / failure persistence ──────────────────────────────────────────

def load_progress() -> dict:
    """
    Returns dict with keys:
      processed      : set of article filenames already done
      total_entities : int
      total_relationships : int
      total_failures : int
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if PROGRESS_FILE.exists():
        try:
            raw = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
            return {
                "processed":           set(raw.get("processed", [])),
                "total_entities":      int(raw.get("total_entities", 0)),
                "total_relationships": int(raw.get("total_relationships", 0)),
                "total_failures":      int(raw.get("total_failures", 0)),
            }
        except Exception as exc:
            print(f"[progress] Could not load {PROGRESS_FILE}: {exc} -- starting fresh")
    return {
        "processed":           set(),
        "total_entities":      0,
        "total_relationships": 0,
        "total_failures":      0,
    }


def save_progress(progress: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    serializable = {
        "processed":           sorted(progress["processed"]),
        "total_entities":      progress["total_entities"],
        "total_relationships": progress["total_relationships"],
        "total_failures":      progress["total_failures"],
        "last_updated":        datetime.now(timezone.utc).isoformat(),
    }
    PROGRESS_FILE.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_failure(failure: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    if FAILURES_FILE.exists():
        try:
            failures = json.loads(FAILURES_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    failures.append(failure)
    FAILURES_FILE.write_text(
        json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── Groq extraction ──────────────────────────────────────────────────────────

def call_groq_json(client: Groq, system_prompt: str, user_content: str,
                   article_name: str, call_label: str) -> tuple[list, str | None]:
    """
    Call Groq, parse JSON array response.
    Returns (parsed_list, error_string_or_None).
    """
    backoff = [30, 60, 120]
    last_err = ""
    for attempt in range(len(backoff) + 1):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ],
                max_tokens=2048,
                temperature=0.0,
            )
            raw = (resp.choices[0].message.content or "").strip()

            # Strip markdown code fences if present
            raw_clean = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw_clean = re.sub(r"\s*```$", "", raw_clean).strip()

            parsed = json.loads(raw_clean)
            if not isinstance(parsed, list):
                return [], f"{call_label}: response was not a JSON array: {raw[:200]}"
            return parsed, None

        except json.JSONDecodeError as exc:
            # Non-retryable — bad JSON
            return [], f"{call_label}: JSON parse error ({exc}): {raw[:300]}"

        except Exception as exc:
            err = str(exc)
            is_rate = any(k in err.lower() for k in ("429", "rate_limit", "rate limit", "too many requests"))
            is_conn = any(k in err.lower() for k in ("connection error", "connectionerror", "connection reset",
                                                       "timed out", "timeout", "remotedisconnected"))
            should_retry = (is_rate or is_conn) and attempt < len(backoff)
            if is_rate and should_retry:
                print(f"  [groq] Rate limited on {call_label}, sleeping {backoff[attempt]}s ...")
            elif is_conn and should_retry:
                print(f"  [groq] Connection error on {call_label}, sleeping {backoff[attempt]}s ...")
            if should_retry:
                time.sleep(backoff[attempt])
                last_err = err
                continue
            return [], f"{call_label}: API error: {err}"

    return [], f"{call_label}: exhausted retries. Last error: {last_err}"


def extract_entities(client: Groq, text: str, article_name: str) -> tuple[list[dict], str | None]:
    user_msg = f"Article: {text[:MAX_CONTENT]}"
    return call_groq_json(client, ENTITY_PROMPT, user_msg, article_name, "entity_extraction")


def extract_relationships(client: Groq, text: str, entities: list[dict],
                          article_name: str) -> tuple[list[dict], str | None]:
    entity_names = [e.get("name", "") for e in entities if e.get("name")]
    user_msg = (
        f"Entities: {json.dumps(entity_names)}\n\n"
        f"Article: {text[:MAX_CONTENT]}"
    )
    return call_groq_json(client, RELATIONSHIP_PROMPT, user_msg, article_name, "relationship_extraction")


# ── TigerGraph insertion ─────────────────────────────────────────────────────

def insert_to_tigergraph(
    conn: tg.TigerGraphConnection,
    doc_id: str,
    content: str,
    source: str,
    entities: list[dict],
    relationships: list[dict],
) -> tuple[int, int]:
    """
    Upserts all vertices and edges for one article.
    Returns (entities_inserted, relationships_inserted).
    """
    # -- Document vertex
    conn.upsertVertex("Document", doc_id, {
        "content": content[:8000],   # TG attribute limit safety
        "source":  source,
    })

    # -- Entity vertices + CONTAINS edges
    valid_entities: list[tuple[str, str, str, str]] = []   # (eid, name, etype, desc)
    for ent in entities:
        raw_name = str(ent.get("name", "")).strip()
        if not raw_name:
            continue
        eid   = sanitize_entity_name(raw_name)
        if not eid:
            continue
        etype = str(ent.get("type", "UNKNOWN")).strip().upper()[:50]
        desc  = str(ent.get("description", "")).strip()[:500]
        valid_entities.append((eid, raw_name, etype, desc))

    if valid_entities:
        conn.upsertVertices("Entity", [
            (eid, {"name": name, "entity_type": etype, "description": desc})
            for eid, name, etype, desc in valid_entities
        ])
        conn.upsertEdges("Document", "CONTAINS", "Entity", [
            (doc_id, eid, {}) for eid, *_ in valid_entities
        ])

    # -- RELATED_TO edges
    entity_id_lookup = {}
    for eid, name, etype, desc in valid_entities:
        entity_id_lookup[name.lower()] = eid

    rel_count = 0
    seen_rels: set[tuple[str, str]] = set()
    for rel in relationships:
        from_name = str(rel.get("from", "")).strip()
        to_name   = str(rel.get("to",   "")).strip()
        rel_text  = str(rel.get("relationship", "")).strip()[:300]

        from_id = entity_id_lookup.get(from_name.lower()) or sanitize_entity_name(from_name)
        to_id   = entity_id_lookup.get(to_name.lower())   or sanitize_entity_name(to_name)

        if not from_id or not to_id or from_id == to_id:
            continue

        # Deduplicate within this article (undirected: normalise order)
        pair = (min(from_id, to_id), max(from_id, to_id))
        if pair in seen_rels:
            continue
        seen_rels.add(pair)

        try:
            conn.upsertEdge("Entity", from_id, "RELATED_TO", "Entity", to_id,
                            {"relationship": rel_text})
            rel_count += 1
        except Exception:
            # If entity doesn't exist yet (edge from unknown name), skip
            pass

    return len(valid_entities), rel_count


# ── Per-article processing ───────────────────────────────────────────────────

def process_article(
    client: Groq,
    conn: tg.TigerGraphConnection,
    filepath: Path,
    progress: dict,
    run_start: float,
    articles_done_this_run: int,
    total_articles: int,
) -> tuple[int, int, bool]:
    """
    Process one article file.
    Returns (entities_added, relationships_added, success).
    """
    article_name = filepath.name

    # Read content
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        error_msg = f"Could not read file: {exc}"
        print(f"  [FAIL] {article_name} -- {error_msg}")
        save_failure({"article": article_name, "stage": "read", "error": error_msg})
        return 0, 0, False

    # Entity extraction
    entities, err = extract_entities(client, content, article_name)
    if err:
        print(f"  [FAIL] {article_name} -- {err}")
        save_failure({
            "article": article_name, "stage": "entity_extraction",
            "error": err, "content_preview": content[:200]
        })
        return 0, 0, False

    time.sleep(GROQ_DELAY)

    # Relationship extraction (only if we have entities)
    relationships: list[dict] = []
    if entities:
        relationships, err = extract_relationships(client, content, entities, article_name)
        if err:
            # Non-fatal: log failure but still insert entities
            print(f"  [WARN] {article_name} -- relationship extraction failed: {err}")
            save_failure({
                "article": article_name, "stage": "relationship_extraction",
                "error": err, "entities_count": len(entities)
            })
            relationships = []

    # Insert into TigerGraph
    doc_id = sanitize_doc_id(article_name)
    source = str(filepath.name)
    try:
        n_ents, n_rels = insert_to_tigergraph(
            conn, doc_id, content, source, entities, relationships
        )
    except Exception as exc:
        error_msg = f"TigerGraph insert error: {exc}"
        print(f"  [FAIL] {article_name} -- {error_msg}")
        save_failure({
            "article": article_name, "stage": "tg_insert",
            "error": error_msg, "entities_count": len(entities)
        })
        return 0, 0, False

    print(f"  [OK] {article_name} -- {n_ents} entities, {n_rels} relationships")
    return n_ents, n_rels, True


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Pipeline 3 -- TigerGraph Ingestion ===\n")

    if not GROQ_API_KEY:
        print("[ERROR] GROQ_API_KEY not set in .env")
        sys.exit(1)

    # Gather all articles
    all_files = sorted(DATA_DIR.glob("*.txt"))
    total_articles = len(all_files)
    print(f"[init] Found {total_articles} articles in {DATA_DIR}")

    # Load progress
    progress = load_progress()
    already_done = len(progress["processed"])
    if already_done:
        print(f"[init] Resuming: {already_done} articles already processed, "
              f"{total_articles - already_done} remaining")
    else:
        print("[init] Starting fresh ingestion")

    # Filter to remaining articles
    remaining = [f for f in all_files if f.name not in progress["processed"]]
    print(f"[init] Articles to process this run: {len(remaining)}\n")

    if not remaining:
        print("[init] Nothing to do -- all articles already ingested.")
        print(f"[init] Totals: {progress['total_entities']} entities, "
              f"{progress['total_relationships']} relationships, "
              f"{progress['total_failures']} failures")
        return

    # Auth + connect
    print("[auth] Fetching TigerGraph token ...")
    token = get_token(TG_HOST, TG_SECRET)
    print(f"[auth] Token obtained")

    print("[conn] Connecting to TigerGraph ...")
    conn = get_connection(token)
    print(f"[conn] Connected to {TG_GRAPHNAME}\n")

    # Groq client
    client = Groq(api_key=GROQ_API_KEY)

    # Batch loop
    run_start          = time.monotonic()
    articles_this_run  = 0
    batch_num          = 0

    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start : batch_start + BATCH_SIZE]
        batch_num += 1
        total_done = already_done + articles_this_run

        print(f"-- Batch {batch_num} "
              f"(articles {total_done + 1}–{total_done + len(batch)} of {total_articles}) --")

        for filepath in batch:
            n_ents, n_rels, ok = process_article(
                client, conn, filepath, progress,
                run_start, articles_this_run, total_articles
            )

            if ok:
                progress["processed"].add(filepath.name)
                progress["total_entities"]      += n_ents
                progress["total_relationships"] += n_rels
            else:
                progress["total_failures"] += 1
                progress["processed"].add(filepath.name)   # mark done (failed) to not retry endlessly

            articles_this_run += 1

        # Save progress after every batch
        save_progress(progress)

        # Running totals every 50 articles
        total_processed = already_done + articles_this_run
        if total_processed % 50 == 0 or batch_start + BATCH_SIZE >= len(remaining):
            elapsed     = time.monotonic() - run_start
            rate        = articles_this_run / elapsed if elapsed > 0 else 0
            remaining_n = total_articles - total_processed
            eta_sec     = (remaining_n / rate) if rate > 0 else 0
            eta_min     = round(eta_sec / 60, 1)
            print(
                f"\n  Progress: {total_processed}/{total_articles} | "
                f"Entities: {progress['total_entities']} | "
                f"Relationships: {progress['total_relationships']} | "
                f"Failures: {progress['total_failures']} | "
                f"ETA: ~{eta_min} minutes\n"
            )

        # Delay between batches (skip after last batch)
        if batch_start + BATCH_SIZE < len(remaining):
            time.sleep(BATCH_DELAY)

    # Final summary
    print("\n=== Ingestion Complete ===")
    total_processed = already_done + articles_this_run
    elapsed_total   = time.monotonic() - run_start
    print(f"  Articles processed : {total_processed}/{total_articles}")
    print(f"  Total entities     : {progress['total_entities']}")
    print(f"  Total relationships: {progress['total_relationships']}")
    print(f"  Total failures     : {progress['total_failures']}")
    print(f"  Run time           : {elapsed_total / 60:.1f} minutes")
    if progress["total_failures"]:
        print(f"  Failure details    : {FAILURES_FILE}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupted] Progress saved. Re-run to resume.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
