"""
Pipeline 3 -- Step 3: Community Detection + Summarisation

Run AFTER ingestion is 100% complete.

Algorithm:
  1. Fetch all Entity vertices from TigerGraph via REST
  2. Fetch all RELATED_TO edges
  3. Build an undirected graph with networkx
  4. Filter to entities with 3+ RELATED_TO connections (connected enough to form a community)
  5. Detect communities with Louvain (networkx >= 3.x) or greedy modularity fallback
  6. Skip singleton / tiny groups (< 3 members)
  7. For each community, call Groq to generate a 2-sentence summary
  8. Upsert Community vertices + BELONGS_TO edges into TigerGraph

Usage:
  python pipeline3_graphrag/communities.py

Options:
  --min-size N        Minimum community size (default: 3)
  --max-communities N Max communities to process (default: 200, 0 = all)
  --dry-run           Detect communities but skip Groq + TigerGraph write
"""

import os
import re
import sys
import json
import time
import argparse
import requests
import _tg_dns_fix  # Override broken local DNS → uses Google-resolved IPs for TigerGraph
from pathlib import Path

from groq import Groq
import pyTigerGraph as tg
from dotenv import load_dotenv

try:
    import networkx as nx
    from networkx.algorithms import community as nx_community
except ImportError:
    print("[ERROR] networkx is required: pip install networkx")
    sys.exit(1)

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# ── Config ───────────────────────────────────────────────────────────────────
TG_HOST      = "https://tg-1727b3d7-e9cd-4032-9168-238043254e0c.tg-2635877100.i.tgcloud.io"
TG_SECRET    = "dvhejqo4r39v302aqi2vfim23ihqp8vn"
TG_GRAPHNAME = "MyDatabase"
TG_USERNAME  = "tigergraph"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"    # quality matters for summaries

MIN_DEGREE   = 3    # entity must have >= this many RELATED_TO edges to be included
MIN_SIZE     = 3    # minimum entities per community
MAX_COMM     = 200  # max communities to summarise (0 = all)
GROQ_DELAY   = 2.0  # seconds between Groq calls

RESULTS_DIR  = Path(__file__).parent.parent / "results"
PROGRESS_FILE = RESULTS_DIR / "communities_progress.json"

COMMUNITY_PROMPT = (
    "These IPL entities are related: {entities}.\n"
    "Write a 2 sentence summary of what connects them."
)


# ── Auth + connection ─────────────────────────────────────────────────────────

def get_token() -> str:
    resp = requests.post(
        f"{TG_HOST}/gsql/v1/tokens", json={"secret": TG_SECRET}, timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    token = data.get("token") or data.get("results", {}).get("token")
    if not token:
        raise ValueError(f"Token not found: {data}")
    return token


def get_connection(token: str) -> tg.TigerGraphConnection:
    return tg.TigerGraphConnection(
        host=TG_HOST, username=TG_USERNAME,
        restppPort="443", gsPort="443",
        graphname=TG_GRAPHNAME, apiToken=token, tgCloud=True,
    )


# ── TigerGraph data fetching ──────────────────────────────────────────────────

def fetch_all_entities(token: str) -> dict[str, dict]:
    """
    Fetch all Entity vertices via GSQL SELECT (handles large result sets).
    Returns {entity_id: {name, entity_type, description}}.
    """
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "text/plain"}
    gsql = f"USE GRAPH {TG_GRAPHNAME}\nSELECT e.entity_id, e.name, e.entity_type, e.description FROM Entity:e"
    resp = requests.post(
        f"{TG_HOST}/gsql/v1/statements",
        data=gsql.encode(), headers=headers, timeout=120
    )
    entities: dict[str, dict] = {}
    try:
        raw = resp.text
        # GSQL returns lines of "Using graph..." then JSON
        # Extract JSON block
        json_start = raw.find("[")
        json_end   = raw.rfind("]") + 1
        if json_start >= 0 and json_end > json_start:
            rows = json.loads(raw[json_start:json_end])
            for row in rows:
                eid = row.get("e.entity_id") or row.get("entity_id", "")
                if eid:
                    entities[eid] = {
                        "name":        row.get("e.name", eid),
                        "entity_type": row.get("e.entity_type", ""),
                        "description": row.get("e.description", ""),
                    }
    except Exception as exc:
        print(f"  [WARN] GSQL entity fetch parse error: {exc}")

    # Fallback: paginated REST
    if not entities:
        print("  [info] GSQL parse failed, falling back to REST pagination ...")
        headers_rest = {"Authorization": f"Bearer {token}"}
        limit   = 1000
        offset  = 0
        max_pages = 200          # safety cap (200K entities max)
        pages_done = 0
        while pages_done < max_pages:
            # Retry up to 4 times on transient connection/DNS errors
            r = None
            for attempt in range(4):
                try:
                    r = requests.get(
                        f"{TG_HOST}/restpp/graph/{TG_GRAPHNAME}/vertices/Entity",
                        headers=headers_rest,
                        params={"limit": limit, "offset": offset},
                        timeout=60,
                    )
                    break
                except Exception as conn_err:
                    wait = 10 * (attempt + 1)
                    print(f"  [WARN] page {pages_done} connection error (attempt {attempt+1}/4): {conn_err}")
                    print(f"  [WARN] Retrying in {wait}s ...")
                    time.sleep(wait)
            if r is None:
                print("  [ERROR] All retries failed — stopping pagination")
                break
            if r.status_code != 200:
                break
            results = r.json().get("results", [])
            if not results:
                break
            new_count = 0
            for v in results:
                eid   = v.get("v_id", "")
                attrs = v.get("attributes", {})
                if eid and eid not in entities:
                    entities[eid] = {
                        "name":        attrs.get("name", eid),
                        "entity_type": attrs.get("entity_type", ""),
                        "description": attrs.get("description", ""),
                    }
                    new_count += 1
            pages_done += 1
            # Stop if no new entities arrived (TigerGraph may wrap offset)
            if new_count == 0:
                print(f"  [info] No new entities at offset {offset} — pagination complete")
                break
            if len(results) < limit:
                break
            offset += limit

    return entities


def _fetch_edges_for_entity(args: tuple) -> list[tuple[str, str, str]]:
    """Worker: fetch RELATED_TO edges for one entity. Used by ThreadPoolExecutor."""
    eid, token = args
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(
            f"{TG_HOST}/restpp/graph/{TG_GRAPHNAME}/edges/Entity/{eid}/RELATED_TO",
            headers=headers, timeout=30,
        )
        if r.status_code != 200:
            return []
        result = []
        for e in r.json().get("results", []):
            fid = e.get("from_id", eid)
            tid = e.get("to_id", "")
            rel = e.get("attributes", {}).get("relationship", "")
            if fid and tid and fid != tid:
                result.append((fid, tid, rel))
        return result
    except Exception:
        return []


def fetch_all_edges(token: str, entity_ids: list[str] | None = None) -> list[tuple[str, str, str]]:
    """
    Fetch all RELATED_TO edges using concurrent REST calls (20 workers).
    Falls back gracefully if any individual call fails.
    Returns list of (from_id, to_id, relationship) — deduplicated.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if entity_ids is None:
        entity_ids = list(fetch_all_entities(token).keys())

    total = len(entity_ids)
    print(f"  [info] Fetching edges for {total} entities (20 concurrent workers) ...")

    seen_pairs: set[tuple[str, str]] = set()
    edges: list[tuple[str, str, str]] = []
    done = 0

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(_fetch_edges_for_entity, (eid, token)): eid
                   for eid in entity_ids}
        for future in as_completed(futures):
            done += 1
            if done % 100 == 0:
                print(f"  [info] {done}/{total} entities fetched ({len(edges)} edges so far)")
            for fid, tid, rel in future.result():
                pair = (min(fid, tid), max(fid, tid))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    edges.append((fid, tid, rel))

    print(f"  [info] {len(edges)} unique RELATED_TO edges fetched")
    return edges


# ── Graph building ────────────────────────────────────────────────────────────

def build_entity_graph(
    entities: dict[str, dict],
    edges: list[tuple[str, str, str]],
    min_degree: int,
) -> nx.Graph:
    """
    Build undirected networkx graph.
    Only include entities with degree >= min_degree.
    """
    G = nx.Graph()

    # Add all entity nodes
    for eid, attrs in entities.items():
        G.add_node(eid, **attrs)

    # Add all edges
    for src, tgt, rel in edges:
        if G.has_node(src) and G.has_node(tgt):
            if G.has_edge(src, tgt):
                # Merge relationship text
                existing = G[src][tgt].get("relationship", "")
                if rel and rel not in existing:
                    G[src][tgt]["relationship"] = existing + "; " + rel
            else:
                G.add_edge(src, tgt, relationship=rel)

    # Filter to nodes with enough connections
    low_degree = [n for n, d in G.degree() if d < min_degree]
    G.remove_nodes_from(low_degree)

    # Remove isolated nodes
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    return G


# ── Community detection ───────────────────────────────────────────────────────

def detect_communities(G: nx.Graph) -> list[frozenset]:
    """
    Detect communities with Louvain (networkx >= 3.x) or
    greedy modularity fallback. Returns list of node-sets.
    """
    if len(G) == 0:
        return []

    # Try Louvain first (networkx >= 3.0)
    try:
        comms = nx_community.louvain_communities(G, seed=42)
        print(f"  [community] Louvain detected {len(comms)} communities")
        return comms
    except AttributeError:
        pass

    # Fallback: greedy modularity
    try:
        comms = list(nx_community.greedy_modularity_communities(G))
        print(f"  [community] Greedy modularity detected {len(comms)} communities")
        return comms
    except Exception as exc:
        print(f"  [WARN] Community detection error: {exc}")

    # Last resort: connected components
    comms = [c for c in nx.connected_components(G)]
    print(f"  [community] Connected components: {len(comms)} groups")
    return comms


# ── Groq summarisation ────────────────────────────────────────────────────────

def summarise_community(
    client: Groq,
    entity_names: list[str],
    community_id: str,
) -> str | None:
    """
    Ask Groq for a 2-sentence summary of what connects these entities.
    Returns summary string or None on failure.
    """
    # Cap entity list length in the prompt
    names_str = ", ".join(entity_names[:30])
    if len(entity_names) > 30:
        names_str += f" (and {len(entity_names) - 30} more)"

    prompt = COMMUNITY_PROMPT.format(entities=names_str)

    backoff = [20, 60]
    for attempt in range(len(backoff) + 1):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            err = str(exc)
            is_rate = any(k in err.lower() for k in ("429", "rate_limit", "too many requests"))
            is_conn = any(k in err.lower() for k in ("connection error", "timed out", "timeout"))
            if (is_rate or is_conn) and attempt < len(backoff):
                wait = backoff[attempt]
                print(f"    [groq] {'Rate limit' if is_rate else 'Connection error'}, "
                      f"sleeping {wait}s ...")
                time.sleep(wait)
                continue
            print(f"    [groq] Failed for community {community_id}: {err[:100]}")
            return None
    return None


# ── TigerGraph write ──────────────────────────────────────────────────────────

def write_community_to_tg(
    conn: tg.TigerGraphConnection,
    community_id: str,
    summary: str,
    level: int,
    entity_ids: list[str],
) -> None:
    """Upsert Community vertex + BELONGS_TO edges for all member entities."""
    conn.upsertVertex("Community", community_id, {
        "summary": summary[:2000],
        "level":   level,
    })
    if entity_ids:
        conn.upsertEdges("Entity", "BELONGS_TO", "Community", [
            (eid, community_id, {}) for eid in entity_ids
        ])


# ── Progress persistence ──────────────────────────────────────────────────────

def load_progress() -> set[str]:
    """Return set of already-processed community IDs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if PROGRESS_FILE.exists():
        try:
            data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
            return set(data.get("done", []))
        except Exception:
            pass
    return set()


def save_progress(done: set[str]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(
        json.dumps({"done": sorted(done)}, indent=2), encoding="utf-8"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main(min_size: int = MIN_SIZE, max_communities: int = MAX_COMM,
         dry_run: bool = False) -> None:
    print("=== Pipeline 3 -- Community Detection ===\n")

    if not GROQ_API_KEY and not dry_run:
        print("[ERROR] GROQ_API_KEY not set")
        sys.exit(1)

    # Auth + connect
    print("[auth] Fetching TigerGraph token ...")
    token = get_token()
    print("[auth] Token obtained")

    conn = get_connection(token) if not dry_run else None

    groq_client = Groq(api_key=GROQ_API_KEY) if not dry_run else None

    # 1. Fetch data
    print("\n[data] Fetching Entity vertices ...")
    entities = fetch_all_entities(token)
    print(f"[data] {len(entities)} entities fetched")

    if not entities:
        print("[ERROR] No entities found. Run ingest.py first.")
        sys.exit(1)

    print("[data] Fetching RELATED_TO edges ...")
    edges = fetch_all_edges(token, entity_ids=list(entities.keys()))
    print(f"[data] {len(edges)} edges fetched")

    if not edges:
        print("[WARN] No RELATED_TO edges found. Graph may be empty or ingestion incomplete.")
        print("       Re-run this script after ingestion finishes.")
        sys.exit(0)

    # 2. Build graph
    print(f"\n[graph] Building entity graph (min_degree={MIN_DEGREE}) ...")
    G = build_entity_graph(entities, edges, min_degree=MIN_DEGREE)
    print(f"[graph] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"(after filtering degree < {MIN_DEGREE})")

    if G.number_of_nodes() == 0:
        print(f"[WARN] No entities pass the degree-{MIN_DEGREE} filter. "
              f"Try --min-size 1 or wait for more ingestion.")
        sys.exit(0)

    # 3. Detect communities
    print("\n[community] Running community detection ...")
    all_communities = detect_communities(G)

    # Filter by min size
    communities = [c for c in all_communities if len(c) >= min_size]
    communities.sort(key=len, reverse=True)   # largest first
    print(f"[community] {len(communities)} communities with >= {min_size} members")
    print(f"[community] Size distribution: "
          + ", ".join(f"{len(c)}" for c in communities[:15])
          + ("..." if len(communities) > 15 else ""))

    if not communities:
        print("[WARN] No communities large enough. Try --min-size 2.")
        sys.exit(0)

    if max_communities and len(communities) > max_communities:
        print(f"[community] Capping at {max_communities} communities (--max-communities)")
        communities = communities[:max_communities]

    if dry_run:
        print("\n[dry-run] Skipping Groq + TigerGraph writes. Done.")
        for i, comm in enumerate(communities[:20]):
            names = [entities[n]["name"] for n in comm if n in entities][:8]
            print(f"  Community {i+1:3d} ({len(comm):3d} members): {', '.join(names)}")
        return

    # 4. Summarise + write
    done = load_progress()
    print(f"\n[summarise] Processing {len(communities)} communities "
          f"({len(done)} already done) ...")

    total_written = 0
    for idx, comm_set in enumerate(communities):
        comm_list  = sorted(comm_set)
        comm_id    = f"community_{idx:04d}"

        if comm_id in done:
            print(f"  [skip] {comm_id} ({len(comm_list)} members)")
            continue

        member_names = [entities[n]["name"] for n in comm_list if n in entities]
        print(f"  [{idx+1}/{len(communities)}] {comm_id} "
              f"({len(comm_list)} members): {', '.join(member_names[:5])}"
              + (f" + {len(member_names)-5} more" if len(member_names) > 5 else ""))

        summary = summarise_community(groq_client, member_names, comm_id)
        if not summary:
            print(f"    [WARN] No summary generated, using member list as fallback")
            summary = (
                f"This community includes {len(member_names)} IPL-related entities: "
                f"{', '.join(member_names[:10])}."
                + (" They are connected through IPL cricket events and relationships."
                   if len(member_names) > 10 else "")
            )

        print(f"    Summary: {summary[:120]}{'...' if len(summary) > 120 else ''}")

        write_community_to_tg(conn, comm_id, summary, level=0, entity_ids=comm_list)
        done.add(comm_id)
        total_written += 1
        save_progress(done)

        time.sleep(GROQ_DELAY)

    print(f"\n=== Community detection complete ===")
    print(f"  Communities written : {total_written}")
    print(f"  Total processed     : {len(done)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect communities and summarise with Groq")
    parser.add_argument("--min-size", type=int, default=MIN_SIZE,
                        help=f"Minimum community size (default: {MIN_SIZE})")
    parser.add_argument("--max-communities", type=int, default=MAX_COMM,
                        help=f"Max communities to process (default: {MAX_COMM}, 0=all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect communities but skip Groq + TigerGraph write")
    args = parser.parse_args()

    try:
        main(
            min_size=args.min_size,
            max_communities=args.max_communities,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        print("\n[interrupted] Progress saved. Re-run to resume.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)
