import json
from pathlib import Path
p = Path("results/ingest_progress.json")
if p.exists():
    d = json.loads(p.read_text())
    print(f"Processed : {len(d.get('processed', []))} / 665")
    print(f"Entities  : {d.get('total_entities', 0)}")
    print(f"Relations : {d.get('total_relationships', 0)}")
    print(f"Failures  : {d.get('total_failures', 0)}")
    done = len(d.get('processed', []))
    remaining = 665 - done
    print(f"Remaining : {remaining}")
else:
    print("No progress file yet")
