import json, glob
files = sorted(glob.glob("/root/rivermind-data/ta-pp/workspace_M9_smoke/diagnose_3a/diagnose_3a_*.json"))
assert files, "no json"
r = json.load(open(files[-1], encoding="utf-8"))
print("file", files[-1])
print("episodes", r["episodes"])
print("C0", r["fused"]["C0"])
print("OS", r["OS_C0_minus_C2mean"])
print("sanity", r.get("sanity"))
assert r["episodes"] == 20
assert "C0" in r["fused"]
print("SMOKE_OK")
