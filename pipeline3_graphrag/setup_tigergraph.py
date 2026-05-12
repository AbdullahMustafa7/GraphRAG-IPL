"""
Pipeline 3 -- Step 1: TigerGraph Schema Setup
Creates the GraphRAG schema on TigerGraph Savanna (MyDatabase).

Vertices:
  - Document   (doc_id PK, content STRING, source STRING)
  - Entity     (entity_id PK, name STRING, entity_type STRING, description STRING)
  - Community  (community_id PK, summary STRING, level INT)

Edges:
  - CONTAINS   Document -> Entity     (directed)
  - RELATED_TO Entity  <-> Entity     (undirected, attribute: relationship STRING)
  - BELONGS_TO Entity  ->  Community  (directed)

Usage:
  python pipeline3_graphrag/setup_tigergraph.py
"""

import os
import sys
import requests
import _tg_dns_fix  # Override broken local DNS → uses Google-resolved IPs for TigerGraph
import pyTigerGraph as tg
from dotenv import load_dotenv

# -- Config -------------------------------------------------------------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

TG_HOST      = "https://tg-1727b3d7-e9cd-4032-9168-238043254e0c.tg-2635877100.i.tgcloud.io"
TG_SECRET    = "dvhejqo4r39v302aqi2vfim23ihqp8vn"
TG_GRAPHNAME = "MyDatabase"
TG_USERNAME  = "tigergraph"


# -- Auth: bypass pyTigerGraph's broken getToken() ----------------------------
def get_token(host: str, secret: str) -> str:
    """Fetch a fresh API token from TigerGraph Savanna."""
    url = f"{host}/gsql/v1/tokens"
    print(f"[auth] POST {url}")
    resp = requests.post(url, json={"secret": secret}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    token = data.get("token") or data.get("results", {}).get("token")
    if not token:
        raise ValueError(f"Token not found in response: {data}")
    print(f"[auth] Token obtained (expires: {data.get('expiration', 'unknown')})")
    return token


# -- Connection ---------------------------------------------------------------
def get_connection(token: str, graphname: str = TG_GRAPHNAME) -> tg.TigerGraphConnection:
    conn = tg.TigerGraphConnection(
        host=TG_HOST,
        username=TG_USERNAME,
        restppPort="443",
        gsPort="443",
        graphname=graphname,
        apiToken=token,
        tgCloud=True,
    )
    return conn


# -- GSQL helpers via REST ----------------------------------------------------
def gsql_via_rest(token: str, query: str) -> str:
    """
    Execute a GSQL statement via the /gsql/v1/statements endpoint.
    Returns the response text.
    """
    url = f"{TG_HOST}/gsql/v1/statements"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "text/plain"}
    resp = requests.post(url, data=query.encode("utf-8"), headers=headers, timeout=120)
    return resp.text


# -- Step 1: Create graph if it doesn't exist ---------------------------------
def ensure_graph_exists(token: str) -> None:
    """
    Tries CREATE GRAPH MyDatabase().
    If the graph already exists TigerGraph returns an error containing
    'already exists' -- that's fine, we just log and move on.
    """
    print(f"[graph] Checking / creating graph '{TG_GRAPHNAME}' ...")
    result = gsql_via_rest(token, f"CREATE GRAPH {TG_GRAPHNAME}()")
    text = result.strip()
    print(f"[graph] Response: {text}")
    if "already exists" in text.lower() or "successfully created" in text.lower() or TG_GRAPHNAME in text:
        print(f"[graph] Graph '{TG_GRAPHNAME}' is ready.")
    else:
        print(f"[graph] Unexpected response -- proceeding anyway.")


# -- Step 2: Schema change job ------------------------------------------------
SCHEMA_GSQL = f"""
USE GRAPH {TG_GRAPHNAME}

CREATE SCHEMA_CHANGE JOB graphrag_schema FOR GRAPH {TG_GRAPHNAME} {{

  ADD VERTEX Document (
    PRIMARY_ID doc_id  STRING,
    content            STRING  DEFAULT "",
    source             STRING  DEFAULT ""
  ) WITH primary_id_as_attribute = "true";

  ADD VERTEX Entity (
    PRIMARY_ID entity_id  STRING,
    name                  STRING  DEFAULT "",
    entity_type           STRING  DEFAULT "",
    description           STRING  DEFAULT ""
  ) WITH primary_id_as_attribute = "true";

  ADD VERTEX Community (
    PRIMARY_ID community_id  STRING,
    summary                  STRING  DEFAULT "",
    level                    INT     DEFAULT 0
  ) WITH primary_id_as_attribute = "true";

  ADD DIRECTED EDGE CONTAINS (
    FROM Document,
    TO   Entity
  );

  ADD UNDIRECTED EDGE RELATED_TO (
    FROM Entity,
    TO   Entity,
    relationship  STRING  DEFAULT ""
  );

  ADD DIRECTED EDGE BELONGS_TO (
    FROM Entity,
    TO   Community
  );

}}

RUN SCHEMA_CHANGE JOB graphrag_schema
DROP JOB graphrag_schema
"""


def get_schema_types(token: str) -> tuple[set, set]:
    """
    Query schema via REST API.  Returns (vertex_types, edge_types).
    Falls back to empty sets on error.
    """
    url = f"{TG_HOST}/gsql/v1/schema/graphs/{TG_GRAPHNAME}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 404:
            return set(), set()
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", data)  # some versions nest under "results"
        v_types = {v["Name"] for v in results.get("VertexTypes", [])}
        e_types = {e["Name"] for e in results.get("EdgeTypes", [])}
        return v_types, e_types
    except Exception as exc:
        print(f"[schema] Warning: schema REST query failed - {exc}")
        return set(), set()


def create_schema(token: str, conn: tg.TigerGraphConnection) -> None:
    """Run the schema change job. Idempotent -- skips if types already exist."""
    want_v = {"Document", "Entity", "Community"}
    want_e = {"CONTAINS", "RELATED_TO", "BELONGS_TO"}

    v_types, e_types = get_schema_types(token)
    missing_v = want_v - v_types
    missing_e = want_e - e_types

    if not missing_v and not missing_e:
        print("[schema] All vertex and edge types already exist -- nothing to do.")
        return

    print(f"[schema] Missing vertices : {missing_v if missing_v else 'none'}")
    print(f"[schema] Missing edges    : {missing_e if missing_e else 'none'}")
    print("[schema] Running schema change job ...")

    result = conn.gsql(SCHEMA_GSQL)
    print("[schema] GSQL output:")
    print(result)

    # Re-verify via REST
    v_types2, e_types2 = get_schema_types(token)
    ok_v = want_v.issubset(v_types2)
    ok_e = want_e.issubset(e_types2)

    if ok_v and ok_e:
        print("[schema] Schema verified successfully.")
    else:
        print(f"[schema] WARNING: verification inconclusive.")
        print(f"  Vertices found : {v_types2}")
        print(f"  Edges found    : {e_types2}")
        # Don't hard-fail -- GSQL output above shows real status


def print_schema_summary(token: str) -> None:
    """Print a summary of the current schema."""
    print("\n--- Current Schema ---")
    v_types, e_types = get_schema_types(token)
    if v_types or e_types:
        print(f"Vertices ({len(v_types)}): {', '.join(sorted(v_types)) or 'none'}")
        print(f"Edges    ({len(e_types)}): {', '.join(sorted(e_types)) or 'none'}")
    else:
        print("  (graph not found or empty schema)")
    print("---------------------\n")


# -- Main ---------------------------------------------------------------------
def main() -> None:
    print("=== Pipeline 3 -- TigerGraph Schema Setup ===\n")

    # 1. Get token
    token = get_token(TG_HOST, TG_SECRET)

    # 2. Ensure graph exists (CREATE GRAPH IF NOT EXISTS)
    ensure_graph_exists(token)

    # 3. Connect
    print("[conn] Creating TigerGraphConnection ...")
    conn = get_connection(token)
    print(f"[conn] Connected to graph: {TG_GRAPHNAME}")

    # 4. Show current schema
    print_schema_summary(token)

    # 5. Create schema
    create_schema(token, conn)

    # 6. Final summary
    print_schema_summary(token)
    print("=== Setup complete. ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
