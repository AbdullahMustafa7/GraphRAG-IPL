"""
Pipeline 3 -- GraphRAG Query Pipeline

Flow per question:
  Step 1  Entity extraction from question   (~50 tokens, Groq)
  Step 2  2-hop graph traversal             (0 tokens, TigerGraph REST)
  Step 3  Community context lookup          (0 tokens, TigerGraph REST)
  Step 4  Build focused prompt              (target <800 tokens)
  Step 5  Groq LLM answer generation
  Step 6  Fallback to direct LLM if graph is empty/entity not found,
          or if TigerGraph is unreachable (tg_unavailable=True)

Token management:
  - Token is fetched once with fetch_token() and stored in tg_token.txt
  - load_token() reads that file at startup — no network call needed
  - Run `python pipeline3_graphrag/refresh_token.py` to rotate the token
"""

import os
import re
import sys
import json
import time
import requests
from pathlib import Path
from urllib.parse import quote

from groq import Groq
import pyTigerGraph as tg
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# ── Config ───────────────────────────────────────────────────────────────────
TG_HOST      = "https://tg-1727b3d7-e9cd-4032-9168-238043254e0c.tg-2635877100.i.tgcloud.io"
TG_SECRET    = "dvhejqo4r39v302aqi2vfim23ihqp8vn"
TG_GRAPHNAME = "MyDatabase"
TG_USERNAME  = "tigergraph"
TG_TOKEN_FILE = Path(__file__).parent / "tg_token.txt"

TG_TIMEOUT = 10  # seconds for all TigerGraph REST calls

GROQ_API_KEY          = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL            = "llama-3.3-70b-versatile"
GROQ_EXTRACT_MODEL    = "llama-3.1-8b-instant"

GROQ_INPUT_COST_PER_MILLION  = 0.59
GROQ_OUTPUT_COST_PER_MILLION = 0.79

MAX_SEED_ENTITIES  = 5
MAX_HOP1_PER_SEED  = 10
MAX_HOP2_PER_HOP1  = 5
MAX_CONTEXT_ENTS   = 30
MAX_CONTEXT_RELS   = 20
MAX_SUMMARY_CHARS  = 600

# Prompts
ENTITY_EXTRACT_PROMPT = (
    "Extract entity names from this question.\n"
    "Return ONLY a JSON array of strings: [\"entity1\", \"entity2\"]\n"
    "Sanitize: lowercase, spaces to underscores, remove special chars\n"
    "If no entities, return []"
)

SYSTEM_PROMPT = (
    "You are an expert on cricket, specifically the Indian Premier League (IPL). "
    "Answer the following question using only the knowledge provided. "
    "Be concise and accurate. "
    "If the provided knowledge does not contain enough information, say so clearly."
)

FALLBACK_SYSTEM_PROMPT = (
    "You are an expert on cricket, specifically the Indian Premier League (IPL). "
    "Answer the following question based on your knowledge of IPL cricket. "
    "Be concise and accurate."
)


# ── Exceptions ────────────────────────────────────────────────────────────────

class TigerGraphUnavailable(Exception):
    """Raised when TigerGraph times out or is unreachable."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def sanitize_entity_name(name: str) -> str:
    """Normalise to safe TigerGraph vertex ID. Must match ingest.py."""
    name = name.strip().lower()
    name = name.replace(" ", "_")
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name[:100]


def _cost(prompt_tokens: int, completion_tokens: int) -> float:
    return round(
        (prompt_tokens  / 1_000_000) * GROQ_INPUT_COST_PER_MILLION
        + (completion_tokens / 1_000_000) * GROQ_OUTPUT_COST_PER_MILLION,
        8,
    )


# ── Token management ──────────────────────────────────────────────────────────

def load_token() -> str:
    """
    Read the TigerGraph bearer token from tg_token.txt.
    Returns empty string if the file is missing or empty — TG calls will
    fail gracefully (401 → TigerGraphUnavailable fallback kicks in).
    """
    if not TG_TOKEN_FILE.exists():
        return ""
    return TG_TOKEN_FILE.read_text(encoding="utf-8").strip()


def fetch_token() -> str:
    """
    Fetch a fresh bearer token from TigerGraph and persist it to tg_token.txt.
    Call this once to bootstrap, or via refresh_token.py to rotate.
    """
    resp = requests.post(
        f"{TG_HOST}/gsql/v1/tokens",
        json={"secret": TG_SECRET},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    token = data.get("token") or data.get("results", {}).get("token")
    if not token:
        raise ValueError(f"Token not found in response: {data}")
    TG_TOKEN_FILE.write_text(token, encoding="utf-8")
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


# ── TigerGraph REST helpers ───────────────────────────────────────────────────

def _tg_get(token: str, path: str, params: dict | None = None) -> list:
    """
    GET {TG_HOST}/restpp/graph/{TG_GRAPHNAME}/{path}
    Returns the 'results' list, [] on 404/400, or raises TigerGraphUnavailable
    on timeout or connection failure.
    """
    url = f"{TG_HOST}/restpp/graph/{TG_GRAPHNAME}/{path}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(url, headers=headers, params=params,
                            timeout=TG_TIMEOUT)
        if resp.status_code in (404, 400):
            return []
        resp.raise_for_status()
        return resp.json().get("results", [])
    except (requests.Timeout, requests.ConnectionError) as exc:
        raise TigerGraphUnavailable(str(exc)) from exc
    except Exception:
        return []


def tg_get_vertex(token: str, vertex_type: str, vertex_id: str) -> dict | None:
    """Fetch a single vertex by ID. Returns attribute dict merged with id, or None."""
    results = _tg_get(token, f"vertices/{vertex_type}/{quote(vertex_id, safe='')}")
    if results:
        v = results[0]
        return {"id": v.get("v_id", vertex_id), **v.get("attributes", {})}
    return None


def tg_get_edges(token: str, src_type: str, src_id: str,
                 edge_type: str, tgt_type: str | None = None) -> list[dict]:
    """
    Fetch all edges of edge_type from src vertex.
    Returns list of dicts with keys: from_id, to_id, attributes.
    Propagates TigerGraphUnavailable if the host is unreachable.
    """
    path = f"edges/{src_type}/{quote(src_id, safe='')}/{edge_type}"
    if tgt_type:
        path += f"/{tgt_type}"
    results = _tg_get(token, path)
    edges = []
    for e in results:
        edges.append({
            "from_id":    e.get("from_id", src_id),
            "to_id":      e.get("to_id", ""),
            "attributes": e.get("attributes", {}),
        })
    return edges


# ── Pipeline class ────────────────────────────────────────────────────────────

class Pipeline3:

    def __init__(self, token: str, groq_client: Groq):
        self.token = token
        self.groq  = groq_client

    # -- Step 1: Extract entities from question --------------------------------
    def extract_question_entities(self, question: str) -> list[str]:
        """
        Ask Groq to pull entity names from the question.
        Returns list of sanitized entity IDs ready for TigerGraph lookup.
        (~50 tokens)
        """
        backoff = [20, 40]
        for attempt in range(len(backoff) + 1):
            try:
                resp = self.groq.chat.completions.create(
                    model=GROQ_EXTRACT_MODEL,
                    messages=[
                        {"role": "system", "content": ENTITY_EXTRACT_PROMPT},
                        {"role": "user",   "content": question},
                    ],
                    max_tokens=128,
                    temperature=0.0,
                )
                raw = (resp.choices[0].message.content or "").strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
                raw = re.sub(r"\s*```$", "", raw).strip()
                parsed = json.loads(raw)
                if not isinstance(parsed, list):
                    return []
                ids = []
                for item in parsed:
                    s = sanitize_entity_name(str(item))
                    if s:
                        ids.append(s)
                return ids[:MAX_SEED_ENTITIES]

            except json.JSONDecodeError:
                return []
            except Exception as exc:
                err = str(exc)
                is_rate = any(k in err.lower() for k in ("429", "rate_limit", "too many requests"))
                if is_rate and attempt < len(backoff):
                    time.sleep(backoff[attempt])
                    continue
                return []
        return []

    # -- Step 2: 2-hop graph traversal ----------------------------------------
    def graph_lookup(self, entity_ids: list[str]) -> dict:
        """
        For each seed entity ID:
          Hop 1: direct RELATED_TO neighbors
          Hop 2: their neighbors (capped)

        Returns dict with:
          entities      -- list of {id, name, entity_type, description}
          relationships -- list of {from, to, relationship}
          seeds_found   -- how many seed entities existed in the graph

        Raises TigerGraphUnavailable if the host is unreachable.
        """
        seen_eids: set[str]    = set()
        entities:  list[dict]  = []
        rels:      list[dict]  = []
        seen_pairs: set[tuple] = set()

        def fetch_and_add_entity(eid: str) -> bool:
            if eid in seen_eids:
                return True
            v = tg_get_vertex(self.token, "Entity", eid)
            if v:
                seen_eids.add(eid)
                entities.append({
                    "id":          eid,
                    "name":        v.get("name", eid),
                    "entity_type": v.get("entity_type", ""),
                    "description": v.get("description", ""),
                })
                return True
            return False

        def fetch_and_add_edges(src_id: str, limit: int) -> list[str]:
            edges = tg_get_edges(self.token, "Entity", src_id, "RELATED_TO", "Entity")
            neighbors = []
            for e in edges[:limit]:
                fid = e["from_id"]
                tid = e["to_id"]
                nbr = tid if fid == src_id else fid
                pair = (min(src_id, nbr), max(src_id, nbr))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    rels.append({
                        "from":         fid,
                        "to":           tid,
                        "relationship": e["attributes"].get("relationship", "related"),
                    })
                neighbors.append(nbr)
            return neighbors

        found_seeds = [eid for eid in entity_ids if fetch_and_add_entity(eid)]

        hop1_all: list[str] = []
        for seed in found_seeds:
            nbrs = fetch_and_add_edges(seed, MAX_HOP1_PER_SEED)
            for nbr in nbrs:
                fetch_and_add_entity(nbr)
            hop1_all.extend(nbrs)

        for hop1_id in hop1_all:
            if len(entities) >= MAX_CONTEXT_ENTS:
                break
            nbrs = fetch_and_add_edges(hop1_id, MAX_HOP2_PER_HOP1)
            for nbr in nbrs:
                if len(entities) >= MAX_CONTEXT_ENTS:
                    break
                fetch_and_add_entity(nbr)

        return {
            "entities":      entities[:MAX_CONTEXT_ENTS],
            "relationships": rels[:MAX_CONTEXT_RELS],
            "seeds_found":   len(found_seeds),
        }

    # -- Step 3: Community context ---------------------------------------------
    def get_community_context(self, entity_ids: list[str]) -> str:
        """
        Follow BELONGS_TO edges from entities -> Community vertices.
        Returns deduplicated summaries joined by ' | ', truncated.
        Raises TigerGraphUnavailable if the host is unreachable.
        """
        seen_cids: set[str] = set()
        summaries: list[str] = []

        for eid in entity_ids[:MAX_SEED_ENTITIES]:
            edges = tg_get_edges(self.token, "Entity", eid, "BELONGS_TO", "Community")
            for e in edges[:2]:
                cid = e["to_id"]
                if cid in seen_cids:
                    continue
                seen_cids.add(cid)
                v = tg_get_vertex(self.token, "Community", cid)
                if v and v.get("summary"):
                    summaries.append(v["summary"].strip())

        if not summaries:
            return ""
        combined = " | ".join(summaries)
        return combined[:MAX_SUMMARY_CHARS]

    # -- Step 4: Build focused prompt ------------------------------------------
    def build_prompt(self, question: str, graph_ctx: dict,
                     community_summary: str) -> str:
        blocks: list[str] = []

        entities = graph_ctx.get("entities", [])
        if entities:
            ent_lines = ["Entities:"]
            for ent in entities:
                name  = ent.get("name", ent.get("id", ""))
                etype = ent.get("entity_type", "")
                desc  = ent.get("description", "")
                line  = f"  {name}"
                if etype:
                    line += f" [{etype}]"
                if desc:
                    line += f" -- {desc}"
                ent_lines.append(line)
            blocks.append("\n".join(ent_lines))

        rels = graph_ctx.get("relationships", [])
        if rels:
            rel_lines = ["Relationships:"]
            for r in rels:
                rel_lines.append(f"  {r['from']} -> {r['to']}: {r['relationship']}")
            blocks.append("\n".join(rel_lines))

        if community_summary:
            blocks.append(f"Context:\n  {community_summary}")

        knowledge = "\n\n".join(blocks)

        return (
            "Answer this question about IPL cricket using the structured "
            "knowledge below.\n\n"
            f"{knowledge}\n\n"
            f"Question: {question}\n"
            "Answer concisely."
        )

    # -- Steps 5 + 6: LLM call with fallback -----------------------------------
    def call_llm(self, prompt: str,
                 fallback: bool = False) -> tuple[str, int, int, float]:
        """
        Groq LLM call.
        Returns (answer, prompt_tokens, completion_tokens, latency_ms).
        """
        system = FALLBACK_SYSTEM_PROMPT if fallback else SYSTEM_PROMPT
        backoff = [30, 60, 120]
        for attempt in range(len(backoff) + 1):
            try:
                t0 = time.monotonic()
                resp = self.groq.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=512,
                    temperature=0.1,
                )
                latency_ms = round((time.monotonic() - t0) * 1000, 2)
                usage = resp.usage
                pt = usage.prompt_tokens     if usage else 0
                ct = usage.completion_tokens if usage else 0
                answer = (resp.choices[0].message.content or "").strip()
                return answer, pt, ct, latency_ms

            except Exception as exc:
                err = str(exc)
                is_rate = any(k in err.lower() for k in ("429", "rate_limit", "too many requests"))
                if is_rate and attempt < len(backoff):
                    time.sleep(backoff[attempt])
                    continue
                raise

    # -- Main entry point per question -----------------------------------------
    def run_query(self, question_id: int, question: str) -> dict:
        """
        Full GraphRAG pipeline for one question.
        Returns result dict with tg_unavailable=True when TigerGraph is
        unreachable (answer still comes from direct Groq fallback).
        """
        try:
            tg_unavailable = False

            try:
                # Step 1: entity extraction (Groq, always available)
                entity_ids = self.extract_question_entities(question)

                # Steps 2 + 3: graph traversal (TigerGraph — may be down)
                if entity_ids:
                    graph_ctx = self.graph_lookup(entity_ids)
                else:
                    graph_ctx = {"entities": [], "relationships": [], "seeds_found": 0}

                seed_entity_ids = [e["id"] for e in graph_ctx["entities"]]
                community_summary = (
                    self.get_community_context(seed_entity_ids)
                    if seed_entity_ids else ""
                )

            except TigerGraphUnavailable as exc:
                print(f"    [TG UNAVAILABLE] {exc} — using direct LLM fallback")
                tg_unavailable = True
                entity_ids    = []
                graph_ctx     = {"entities": [], "relationships": [], "seeds_found": 0}
                community_summary = ""

            n_entities   = len(graph_ctx["entities"])
            # Fallback when: TG down, or no entities found in graph
            use_fallback = tg_unavailable or (n_entities == 0)

            if use_fallback:
                prompt = question
            else:
                prompt = self.build_prompt(question, graph_ctx, community_summary)

            answer, pt, ct, latency_ms = self.call_llm(prompt, fallback=use_fallback)

            return {
                "id":                    question_id,
                "question":              question,
                "answer":                answer,
                "prompt_tokens":         pt,
                "completion_tokens":     ct,
                "total_tokens":          pt + ct,
                "latency_ms":            latency_ms,
                "cost_usd":              _cost(pt, ct),
                "graph_entities_found":  n_entities,
                "graph_hops":            2,
                "fallback":              use_fallback,
                "tg_unavailable":        tg_unavailable,
                "model":                 GROQ_MODEL,
                "pipeline":              "graphrag",
                "error":                 None,
                "_extracted_ids":        entity_ids,
                "_seeds_found":          graph_ctx.get("seeds_found", 0),
                "_graph_entities":       [e["name"] for e in graph_ctx["entities"][:10]],
            }

        except Exception as exc:
            return {
                "id":                    question_id,
                "question":              question,
                "answer":                "",
                "prompt_tokens":         0,
                "completion_tokens":     0,
                "total_tokens":          0,
                "latency_ms":            0.0,
                "cost_usd":              0.0,
                "graph_entities_found":  0,
                "graph_hops":            2,
                "fallback":              True,
                "tg_unavailable":        False,
                "model":                 GROQ_MODEL,
                "pipeline":              "graphrag",
                "error":                 str(exc),
            }


# ── Batch runner (called by run_pipeline3.py) ─────────────────────────────────

def run_batch(
    questions: list[dict],
    delay_between_requests: float,
    results_path: Path,
    existing_results: list[dict],
) -> dict:
    """
    Run all questions through Pipeline 3, skipping already-completed ones.
    Saves results incrementally to results_path after every question.
    Returns a summary dict.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in .env")

    print("[p3] Loading TigerGraph token from tg_token.txt ...")
    token = load_token()
    if not token:
        print("[p3] WARNING: tg_token.txt is empty — TG calls will fall back to direct LLM")
    else:
        print("[p3] Token loaded")

    groq_client = Groq(api_key=GROQ_API_KEY)
    pipeline    = Pipeline3(token=token, groq_client=groq_client)

    done_ids    = {r["id"] for r in existing_results if not r.get("error")}
    all_results = {r["id"]: r for r in existing_results}
    total_q     = len(questions)

    ok_existing = [r for r in existing_results if not r.get("error")]
    accum_pt    = sum(r.get("prompt_tokens",     0) for r in ok_existing)
    accum_ct    = sum(r.get("completion_tokens", 0) for r in ok_existing)
    accum_lat   = sum(r.get("latency_ms",        0) for r in ok_existing)
    accum_cost  = sum(r.get("cost_usd",          0) for r in ok_existing)
    accum_ents  = sum(r.get("graph_entities_found", 0) for r in ok_existing)
    n_ok        = len(ok_existing)
    n_fallback  = sum(1 for r in ok_existing if r.get("fallback"))
    n_tg_down   = sum(1 for r in ok_existing if r.get("tg_unavailable"))

    for q in questions:
        qid = q["id"]
        if qid in done_ids:
            print(f"  [skip] Q{qid} already done")
            continue

        print(f"\n  [Q{qid}/{total_q}] {q['question']}")

        result = pipeline.run_query(qid, q["question"])

        if result.get("error"):
            print(f"    [ERR] {result['error']}")
        else:
            extracted  = result.get("_extracted_ids", [])
            seeds_hit  = result.get("_seeds_found", 0)
            graph_ents = result.get("_graph_entities", [])
            print(f"    Extracted IDs  : {extracted if extracted else '(none)'}")
            if result.get("tg_unavailable"):
                print(f"    ** TG UNAVAILABLE ** — direct LLM fallback used")
            elif not result["fallback"]:
                print(f"    Seeds in graph : {seeds_hit}/{len(extracted)}  "
                      f"-> {result['graph_entities_found']} total graph entities")
                if graph_ents:
                    shown = graph_ents[:8]
                    more  = result["graph_entities_found"] - len(shown)
                    extra = f" + {more} more" if more > 0 else ""
                    print(f"    Entities (sample): {', '.join(shown)}{extra}")
            else:
                print(f"    ** FALLBACK ** (0 entities found in graph)")
            fb_tag = "  [FALLBACK]" if result["fallback"] else ""
            print(
                f"    Tokens : {result['prompt_tokens']} prompt + "
                f"{result['completion_tokens']} completion = "
                f"{result['total_tokens']} total{fb_tag}"
            )
            print(f"    Cost   : ${result['cost_usd']:.6f}  |  "
                  f"Latency: {result['latency_ms']:.0f}ms")
            answer_lines = result["answer"].replace("\n", " ").strip()
            print(f"    Answer : {answer_lines[:300]}"
                  f"{'...' if len(result['answer']) > 300 else ''}")

            n_ok        += 1
            accum_pt    += result["prompt_tokens"]
            accum_ct    += result["completion_tokens"]
            accum_lat   += result["latency_ms"]
            accum_cost  += result["cost_usd"]
            accum_ents  += result["graph_entities_found"]
            if result["fallback"]:
                n_fallback += 1
            if result.get("tg_unavailable"):
                n_tg_down += 1

        all_results[qid] = result
        _save_results(list(all_results.values()), results_path)

        remaining = sum(1 for qq in questions if qq["id"] not in all_results
                        or all_results[qq["id"]].get("error"))
        if remaining > 0:
            time.sleep(delay_between_requests)

    summary = {
        "total_questions":       total_q,
        "completed":             n_ok,
        "fallbacks":             n_fallback,
        "tg_unavailable_count":  n_tg_down,
        "avg_total_tokens":      round((accum_pt + accum_ct) / max(n_ok, 1), 1),
        "avg_prompt_tokens":     round(accum_pt / max(n_ok, 1), 1),
        "avg_completion_tokens": round(accum_ct / max(n_ok, 1), 1),
        "avg_latency_ms":        round(accum_lat  / max(n_ok, 1), 1),
        "avg_entities_found":    round(accum_ents / max(n_ok, 1), 1),
        "total_cost_usd":        round(accum_cost, 6),
        "model":                 GROQ_MODEL,
    }
    _save_results(list(all_results.values()), results_path, summary=summary)
    return summary


def _save_results(results: list[dict], path: Path,
                  summary: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"results": sorted(results, key=lambda r: r["id"])}
    if summary:
        payload["summary"] = summary
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
