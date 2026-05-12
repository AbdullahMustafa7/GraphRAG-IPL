"""Verify TigerGraph data via GSQL (count) and direct REST lookup."""
import json, requests
from dotenv import load_dotenv
load_dotenv()

TG_HOST   = "https://tg-1727b3d7-e9cd-4032-9168-238043254e0c.tg-2635877100.i.tgcloud.io"
TG_SECRET = "dvhejqo4r39v302aqi2vfim23ihqp8vn"
TG_GRAPH  = "MyDatabase"

token = requests.post(
    f"{TG_HOST}/gsql/v1/tokens", json={"secret": TG_SECRET}, timeout=30
).json().get("token")
headers = {"Authorization": f"Bearer {token}"}
print(f"Token: {token[:20]}...\n")

# 1. Vertex counts via GSQL SELECT
print("=== Vertex counts via GSQL ===")
for vtype in ["Document", "Entity", "Community"]:
    gsql = f"USE GRAPH {TG_GRAPH}\nSELECT COUNT(*) FROM {vtype}"
    r = requests.post(
        f"{TG_HOST}/gsql/v1/statements",
        data=gsql.encode(), headers={**headers, "Content-Type": "text/plain"}, timeout=30
    )
    print(f"  {vtype}: {r.text.strip()[:200]}")

# 2. Try restpp endpoint for vertex count
print("\n=== REST vertex count (restpp) ===")
for vtype in ["Document", "Entity"]:
    r = requests.get(
        f"{TG_HOST}/restpp/graph/{TG_GRAPH}/vertices/{vtype}",
        headers=headers, params={"limit": 1}, timeout=30
    )
    print(f"  restpp /vertices/{vtype} -> HTTP {r.status_code}: {r.text[:200]}")

# 3. Try graph endpoint (no restpp prefix)
print("\n=== REST vertex count (graph) ===")
for vtype in ["Document", "Entity"]:
    r = requests.get(
        f"{TG_HOST}/graph/{TG_GRAPH}/vertices/{vtype}",
        headers=headers, params={"limit": 3}, timeout=30
    )
    print(f"  /graph/vertices/{vtype} -> HTTP {r.status_code}: {r.text[:300]}")

# 4. Manual REST upsert + query test
print("\n=== Manual upsert test ===")
upsert_body = {
    "vertices": {
        "Entity": {
            "test_entity_123": {
                "name": {"value": "Test Entity"},
                "entity_type": {"value": "TEST"},
                "description": {"value": "inserted via direct REST"}
            }
        }
    }
}
r = requests.post(
    f"{TG_HOST}/graph/{TG_GRAPH}",
    json=upsert_body, headers=headers, timeout=30
)
print(f"  Upsert -> HTTP {r.status_code}: {r.text[:200]}")

# 5. Query it back
r2 = requests.get(
    f"{TG_HOST}/graph/{TG_GRAPH}/vertices/Entity/test_entity_123",
    headers=headers, timeout=15
)
print(f"  Query back -> HTTP {r2.status_code}: {r2.text[:300]}")
