"""Quick diagnostic: ingest progress + TigerGraph vertex counts + entity lookups."""
import json, requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# -- Ingest progress ----------------------------------------------------------
p = Path("results/ingest_progress.json")
if p.exists():
    d = json.loads(p.read_text())
    print(f"Ingest progress : {len(d['processed'])} / 665 articles done")
    print(f"  Total entities     : {d['total_entities']}")
    print(f"  Total relationships: {d['total_relationships']}")
    print(f"  Total failures     : {d['total_failures']}")
else:
    print("No ingest_progress.json yet")

# -- TigerGraph token ---------------------------------------------------------
TG_HOST   = "https://tg-1727b3d7-e9cd-4032-9168-238043254e0c.tg-2635877100.i.tgcloud.io"
TG_SECRET = "dvhejqo4r39v302aqi2vfim23ihqp8vn"

token = requests.post(
    f"{TG_HOST}/gsql/v1/tokens", json={"secret": TG_SECRET}, timeout=30
).json().get("token")
headers = {"Authorization": f"Bearer {token}"}

print()

# -- Vertex counts ------------------------------------------------------------
for vtype in ["Document", "Entity", "Community"]:
    r = requests.get(
        f"{TG_HOST}/graph/MyDatabase/vertices/{vtype}",
        headers=headers, params={"count_only": True}, timeout=30
    )
    print(f"TG vertex count [{vtype:10s}]: {r.text.strip()}")

print()

# -- Entity spot checks -------------------------------------------------------
for eid in ["chennai_super_kings", "rajasthan_royals", "ms_dhoni",
            "mumbai_indians", "virat_kohli", "2008_indian_premier_league"]:
    r = requests.get(
        f"{TG_HOST}/graph/MyDatabase/vertices/Entity/{eid}",
        headers=headers, timeout=15
    )
    results = r.json().get("results", []) if r.status_code == 200 else []
    if results:
        attrs = results[0].get("attributes", {})
        print(f"  FOUND  [{eid}]  type={attrs.get('entity_type','')}  "
              f"name={attrs.get('name','')}")
    else:
        print(f"  MISSING [{eid}]  (HTTP {r.status_code})")
