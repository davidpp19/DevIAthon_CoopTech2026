"""Script para monitorear el estado del pipeline."""
import httpx
import json
import time

time.sleep(25)

r = httpx.get("http://localhost:8080/api/v1/pipeline/status", timeout=10)
d = r.json()

print(f"Status: {d['status']}")
print(f"Phase: {d['current_phase']}")
print(f"Duration: {d.get('pipeline_duration_seconds', 0)}s")
print(f"Phase durations: {json.dumps(d.get('phase_durations', {}), indent=2)}")
print("\nAgentes:")
for k, v in d.get("agents", {}).items():
    print(f"  {k}: {v['status']}")
